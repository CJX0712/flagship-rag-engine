/*
 * APEX-ENGINE · 端到端 推理引擎 + RAG 检索
 * ------------------------------------------------------------------
 * 本文件把前面三块地基（apex-forge 算法库 / apex-transformer 参考实现）
 * 变成一套能真的跑起来的系统。三大件：
 *
 *   1) 分词器（可往返）
 *   2) KV-Cache 推理引擎 —— 工业界推理标配：每步只算新 token，复用历史 K/V。
 *      正确性由"与全量重算逐位等价"这条硬不变量保证（不是靠"看起来对"）。
 *      采样支持 greedy / temperature / top-k / top-p（复用 apex-forge）。
 *   3) BM25 检索器 —— 信息检索领域的黄金标准 lexical 排序函数。
 *      在已知 ground-truth 的合成语料上实测 Recall@k / MRR。
 *   4) RAG 管线：query → BM25 top-k → 拼 prompt → LM 生成。
 *
 * 站在世界顶级成果之上：
 *   - KV-Cache：现代 LLM 推理服务（vLLM / TGI / TensorRT-LLM）通用做法
 *   - BM25：Robertson & Zaragoza 2009, "The Probabilistic Relevance Framework"
 *   - Top-k / Top-p 采样：Fan 2018 / Holtzman 2020（复用 apex-forge.Sample）
 *   - Transformer / RoPE / AdamW：见 apex-transformer README
 *
 * 诚实声明：本引擎验证的是"引擎本身正确"（KV-Cache 等价、检索指标、管线连通），
 *          搭载的是玩具级小模型，**不宣称**生成质量可用。
 *
 * 作者：晨星 (GitHub: CJX0712) · 许可：MIT
 */

(function (global) {
  'use strict';

  let APEX, TR;
  if (typeof module !== 'undefined' && module.exports) {
    APEX = require('./apex-forge.js');
    TR = require('./apex-transformer.js');
  } else {
    APEX = global.APEX; TR = global.APEX_T;
  }
  const T = TR.T;
  const zeros2 = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));

  // ===================== 1. 分词器（词级，可往返） =====================
  function makeTokenizer(texts) {
    const vocabSet = new Set();
    texts.forEach((t) => t.split(/\s+/).filter(Boolean).forEach((w) => vocabSet.add(w)));
    const vocab = Array.from(vocabSet).sort();
    const stoi = {}, itos = {};
    vocab.forEach((w, i) => { stoi[w] = i; itos[i] = w; });
    return {
      vocab, stoi, itos, size: vocab.length,
      encode(s) { return s.split(/\s+/).filter(Boolean).map((w) => stoi[w]).filter((i) => i !== undefined); },
      decode(ids) { return ids.map((i) => itos[i] !== undefined ? itos[i] : '?').join(' '); },
    };
  }

  // ===================== 2. KV-Cache 推理引擎 =====================
  function newCache(layers) { return Array.from({ length: layers }, () => ({ K: [], V: [] })); }

  // 单步：只算新 token，复用历史 K/V
  function forwardCached(m, tokId, pos, cache, heads, ropeOpts) {
    const rOpts = ropeOpts || { mode: 'none', base: 10000 };
    let x = TR.embedding([tokId], m.E);                    // [1,d]
    for (let li = 0; li < m.blocks.length; li++) {
      const b = m.blocks[li];
      const xn = TR.rmsNorm(x, b.ln1);
      let Q = TR.ropeWith(TR.addBias(TR.matmul(xn, b.Wq), b.bq), [pos], rOpts);
      let K = TR.ropeWith(TR.addBias(TR.matmul(xn, b.Wk), b.bk), [pos], rOpts);
      const V = TR.addBias(TR.matmul(xn, b.Wv), b.bv);
      cache[li].K.push(K.data[0].slice());
      cache[li].V.push(V.data[0].slice());
      const Kall = new T(cache[li].K.map((r) => r.slice()));
      const Vall = new T(cache[li].V.map((r) => r.slice()));
      const L = Kall.shape[0];
      const mask = [new Array(L).fill(1)];                  // 新 token 可见所有历史位置
      const attn = TR.multiHeadAttention(Q, Kall, Vall, heads, mask);
      x = TR.add(x, TR.matmul(attn, b.Wo));
      const h = TR.gelu(TR.addBias(TR.matmul(TR.rmsNorm(x, b.ln2), b.W1), b.b1));
      x = TR.add(x, TR.addBias(TR.matmul(h, b.W2), b.b2));
    }
    return TR.addBias(TR.matmul(x, m.Wlm), m.blm).data[0];  // logits [vocab]
  }

  // 参考实现：每步全量重算（慢，但用来验证 KV-Cache 等价性）
  function forwardFull(m, ids, pos, ropeOpts) {
    return m.forward(ids, pos, 0, ropeOpts).logits;
  }

  // 采样：greedy / temperature / top-k / top-p（后三者复用 apex-forge）
  function sampleFromLogits(logits, opts, rng) {
    opts = opts || {};
    if (opts.greedy || !opts.temperature) {
      let best = 0;
      for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
      return best;
    }
    let l = APEX.Sample.applyTemperature(logits, opts.temperature);
    if (opts.topK) l = APEX.Sample.topK(l, opts.topK);
    if (opts.topP) l = APEX.Sample.topP(l, opts.topP);
    const p = APEX.Act.softmax(l);
    const r = rng();
    let c = 0;
    for (let j = 0; j < p.length; j++) { c += p[j]; if (r < c) return j; }
    return p.length - 1;
  }

  // 生成：prefill(prompt) → 逐 token 解码（走 KV-Cache）
  function generate(m, promptIds, opts) {
    opts = opts || {};
    const heads = opts.heads, maxTokens = opts.maxTokens || 8;
    const ropeOpts = opts.ropeOpts || { mode: 'none', base: 10000 };
    const rng = opts.rng || Math.random;
    const cache = newCache(m.blocks.length);
    let logits = null;
    for (let i = 0; i < promptIds.length; i++) logits = forwardCached(m, promptIds[i], i, cache, heads, ropeOpts);
    const out = [];
    for (let s = 0; s < maxTokens; s++) {
      const next = sampleFromLogits(logits, opts, rng);
      out.push(next);
      logits = forwardCached(m, next, promptIds.length + s, cache, heads, ropeOpts);
    }
    return { tokens: out, text: opts.tokenizer ? opts.tokenizer.decode(out) : out.join(',') };
  }

  // 参考生成：每步全量重算
  function generateFull(m, promptIds, opts) {
    opts = opts || {};
    const maxTokens = opts.maxTokens || 8;
    const ropeOpts = opts.ropeOpts || { mode: 'none', base: 10000 };
    const rng = opts.rng || Math.random;
    const ids = promptIds.slice(), out = [];
    for (let s = 0; s < maxTokens; s++) {
      const pos = ids.map((_, j) => j);
      const logits = forwardFull(m, ids, pos, ropeOpts);
      const next = sampleFromLogits(logits, opts, rng);
      out.push(next); ids.push(next);
    }
    return { tokens: out };
  }

  // ===================== 3. BM25 检索器 =====================
  // score(q,d) = Σ_t IDF(t) · f(t,d)(k1+1) / (f(t,d) + k1(1-b+b·|d|/avgdl))
  // IDF(t) = ln(1 + (N - df(t) + 0.5)/(df(t) + 0.5))
  function buildBM25(docs, tokenizer, opts) {
    opts = opts || {};
    const k1 = opts.k1 !== undefined ? opts.k1 : 1.5;
    const b = opts.b !== undefined ? opts.b : 0.75;
    const N = docs.length;
    const docTokens = docs.map((d) => tokenizer.encode(d));
    const docLen = docTokens.map((t) => t.length);
    const avgdl = docLen.reduce((a, c) => a + c, 0) / N;
    const df = new Map();
    docTokens.forEach((ts) => { new Set(ts).forEach((t) => df.set(t, (df.get(t) || 0) + 1)); });
    function idf(t) { const d = df.get(t) || 0; return Math.log(1 + (N - d + 0.5) / (d + 0.5)); }
    function score(qTokens, di) {
      const ts = docTokens[di], f = new Map();
      ts.forEach((t) => f.set(t, (f.get(t) || 0) + 1));
      let s = 0;
      new Set(qTokens).forEach((t) => {
        const freq = f.get(t) || 0; if (!freq) return;
        s += idf(t) * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * docLen[di] / avgdl));
      });
      return s;
    }
    function topK(query, k) {
      const q = tokenizer.encode(query);
      const scored = docs.map((d, i) => [i, score(q, i)]);
      scored.sort((x, y) => y[1] - x[1]);
      return scored.slice(0, k || 3);
    }
    return { idf, score, topK, df, docTokens, docLen, avgdl, k1, b, N };
  }

  // 检索评估：Recall@k 与 MRR（ground-truth 已知）
  function retrievalEval(bm, queries, gold, k) {
    let hit = 0, mrr = 0;
    queries.forEach((q, qi) => {
      const ranks = bm.topK(q, k).map((x) => x[0]);
      const r = ranks.indexOf(gold[qi]);
      if (r >= 0) { hit++; mrr += 1 / (r + 1); }
    });
    return { recallAtK: hit / queries.length, mrr: mrr / queries.length, k };
  }

  // ===================== 4. 训练一个玩具级小 LM（给 RAG 管线搭载） =====================
  function trainTinyLM(texts, tokenizer, opts) {
    opts = opts || {};
    const cfg = opts.cfg || { d: 32, heads: 2, layers: 2, ffn: 64, maxlen: 64 };
    const vocab = tokenizer.size;
    const seqs = texts.map((t) => tokenizer.encode(t)).filter((s) => s.length > 1);
    const rng = APEX.makeRng(opts.seed || 20261021);
    const m = TR.buildTransformer(vocab, cfg, rng);
    const opt = new TR.AdamWOpt(m.params, 2e-3, 0.9, 0.999, 1e-8, 0);
    const steps = opts.steps || 2000;
    const ropeOpts = { mode: 'none', base: 10000 };
    for (let s = 0; s < steps; s++) {
      const seq = seqs[s % seqs.length];
      opt.lr = 2e-3 * (1 - (s / steps) * 0.9);
      const i = 1 + Math.floor(rng() * (seq.length - 1));
      const ids = seq.slice(0, i), target = seq[i], pos = ids.map((_, j) => j);
      m.params.forEach((p) => { p.grad = p.grad.map((r) => r.map(() => 0)); });
      const { loss } = m.forward(ids, pos, target, ropeOpts);
      TR.backward(loss); TR.clipGrads(m.params, 1.0); opt.step();
    }
    return { model: m, cfg, vocab };
  }

  // ===================== 5. RAG 管线 =====================
  function rag(query, docs, bm, lm, tokenizer, opts) {
    opts = opts || {};
    const k = opts.k || 2;
    const top = bm.topK(query, k);
    const ctx = top.map((x) => docs[x[0]]).join(' ');
    const prompt = query + ' ' + ctx;
    const promptIds = tokenizer.encode(prompt);
    const gen = generate(lm, promptIds, {
      heads: opts.heads, maxTokens: opts.maxTokens || 6,
      greedy: opts.greedy !== false, tokenizer,
      ropeOpts: { mode: 'none', base: 10000 },
    });
    return { retrieved: top.map((x) => ({ doc: x[0], score: x[1], text: docs[x[0]] })), prompt, generated: gen };
  }

  // ===================== 6. 自检（可被独立实现交叉验证的不变量） =====================
  function maxAbsDiff(a, b) {
    let m = 0;
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
    return m;
  }

  function runSelfTest(opts) {
    opts = opts || {};
    const results = [];

    // ① 分词器往返一致
    {
      const tk = makeTokenizer(['注意力 机制 计算 相关性', '优化器 Adam 参数']);
      const s = '注意力 机制 计算 相关性';
      const ok = tk.decode(tk.encode(s)) === s;
      results.push({ name: '分词器 往返一致', pass: ok, detail: `"${s}" -> ${ok ? '一致' : '不一致'}` });
    }

    // ② KV-Cache 与全量重算逐位等价（硬不变量）
    {
      const cfg = { d: 32, heads: 2, layers: 2, ffn: 64, maxlen: 32 };
      const m = TR.buildTransformer(24, cfg, APEX.makeRng(11));
      const prompt = [3, 7, 1, 9, 4];
      const ropeOpts = { mode: 'none', base: 10000 };
      let worst = 0;
      // prefill 等价
      const cache = newCache(m.blocks.length);
      let lc = null;
      for (let i = 0; i < prompt.length; i++) lc = forwardCached(m, prompt[i], i, cache, cfg.heads, ropeOpts);
      const lf = forwardFull(m, prompt, prompt.map((_, j) => j), ropeOpts);
      worst = Math.max(worst, maxAbsDiff(lc, lf));
      // 再走两步解码，继续比对
      const ids = prompt.slice();
      for (let s = 0; s < 2; s++) {
        let b = 0; for (let j = 1; j < lc.length; j++) if (lc[j] > lc[b]) b = j;
        ids.push(b);
        const nc = forwardCached(m, b, ids.length - 1, cache, cfg.heads, ropeOpts);
        const nf = forwardFull(m, ids, ids.map((_, j) => j), ropeOpts);
        worst = Math.max(worst, maxAbsDiff(nc, nf));
        lc = nc;
      }
      results.push({ name: 'KV-Cache ≡ 全量重算', pass: worst < 1e-9, detail: `最大偏差=${worst.toExponential(2)}` });
    }

    // ③ greedy 解码：KV-Cache 路径 == 全量重算路径（token 级一致）
    {
      const cfg = { d: 32, heads: 2, layers: 2, ffn: 64, maxlen: 32 };
      const m = TR.buildTransformer(24, cfg, APEX.makeRng(13));
      const prompt = [2, 5, 8];
      const g1 = generate(m, prompt, { heads: cfg.heads, maxTokens: 5, greedy: true });
      const g2 = generateFull(m, prompt, { maxTokens: 5, greedy: true });
      const same = g1.tokens.join(',') === g2.tokens.join(',');
      results.push({ name: 'greedy 解码一致', pass: same, detail: `cache=[${g1.tokens}] full=[${g2.tokens}]` });
    }

    // ④ top-p 采样保留质量 ≥ p（复用 apex-forge 不变量）
    {
      const logits = [1.2, 0.3, -0.5, 2.1, 0.7, -1.0];
      const p = 0.8;
      const masked = APEX.Sample.topP(logits, p);
      const probs = APEX.Act.softmax(logits);
      let kept = 0;
      for (let i = 0; i < masked.length; i++) if (masked[i] > -Infinity) kept += probs[i];
      results.push({ name: 'top-p 保留质量≥p', pass: kept >= p, detail: `保留=${kept.toFixed(3)} ≥ p=${p}` });
    }

    // ⑤ 采样可复现（同种子两次结果一致）
    {
      const cfg = { d: 32, heads: 2, layers: 2, ffn: 64, maxlen: 32 };
      const m = TR.buildTransformer(24, cfg, APEX.makeRng(17));
      const o = { heads: cfg.heads, maxTokens: 5, temperature: 0.9, topP: 0.9 };
      const a = generate(m, [4, 1], Object.assign({}, o, { rng: APEX.makeRng(5) }));
      const b = generate(m, [4, 1], Object.assign({}, o, { rng: APEX.makeRng(5) }));
      results.push({ name: '采样可复现(同种子)', pass: a.tokens.join(',') === b.tokens.join(','), detail: `[${a.tokens}] vs [${b.tokens}]` });
    }

    // ⑥ BM25 IDF 单调：越稀有的词 IDF 越高
    {
      const docs = ['猫 和 狗', '猫 又 出现', '狗 也 出现', '鸟 出现'];
      const tk = makeTokenizer(docs);
      const bm = buildBM25(docs, tk);
      const idfRare = bm.idf(tk.encode('鸟')[0]);      // df=1
      const idfCommon = bm.idf(tk.encode('出现')[0]);  // df=3
      results.push({ name: 'BM25 IDF 单调', pass: idfRare > idfCommon, detail: `IDF(鸟)=${idfRare.toFixed(3)} > IDF(出现)=${idfCommon.toFixed(3)}` });
    }

    // ⑦ BM25 词频饱和：重复同一个词的增益递减
    {
      const docs = ['alpha', 'alpha alpha', 'alpha alpha alpha'];
      const tk = makeTokenizer(docs);
      const bm = buildBM25(docs, tk);
      const q = tk.encode('alpha');
      const s1 = bm.score(q, 0), s2 = bm.score(q, 1), s3 = bm.score(q, 2);
      const d1 = s2 - s1, d2 = s3 - s2;
      results.push({ name: 'BM25 词频饱和', pass: d1 > 0 && d2 > 0 && d2 < d1, detail: `+f: ${d1.toFixed(3)} → ${d2.toFixed(3)} (递减)` });
    }

    // ⑧ 检索指标：已知 ground-truth 语料上的 Recall@k / MRR
    {
      const docs = [
        '注意力 机制 通过 缩放点积 计算 相关性',
        '优化器 Adam 使用 动量 二阶矩 更新 参数',
        '位置 编码 RoPE 通过 旋转 注入 相对 位置',
        '归一化 RMSNorm 稳定 训练 残差 连接',
      ];
      const queries = ['缩放点积 相关性', 'Adam 动量 参数', 'RoPE 旋转 位置', 'RMSNorm 归一化'];
      const gold = [0, 1, 2, 3];
      const tk = makeTokenizer(docs.concat(queries));
      const bm = buildBM25(docs, tk);
      const ev = retrievalEval(bm, queries, gold, 1);
      results.push({ name: 'BM25 检索 Recall@1/MRR', pass: ev.recallAtK === 1 && ev.mrr === 1, detail: `Recall@1=${ev.recallAtK.toFixed(2)} MRR=${ev.mrr.toFixed(2)}` });
    }

    // ⑨ 端到端 RAG 管线：检索命中 + 生成连通
    {
      const docs = [
        '注意力 机制 通过 缩放点积 计算 相关性',
        '优化器 Adam 使用 动量 二阶矩 更新 参数',
        '位置 编码 RoPE 通过 旋转 注入 相对 位置',
        '归一化 RMSNorm 稳定 训练 残差 连接',
      ];
      const queries = ['缩放点积 相关性', 'Adam 动量 参数', 'RoPE 旋转 位置', 'RMSNorm 归一化'];
      const gold = [0, 1, 2, 3];
      const tk = makeTokenizer(docs.concat(queries));
      const bm = buildBM25(docs, tk);
      const lm = trainTinyLM(docs, tk, { steps: 2000, seed: 20261021 });
      const out = rag('Adam 动量 参数', docs, bm, lm.model, tk, { k: 2, heads: lm.cfg.heads, maxTokens: 5 });
      const hitTop1 = out.retrieved[0].doc === 1;
      const genOk = out.generated.tokens.length === 5 && out.generated.tokens.every((t) => t >= 0 && t < tk.size);
      results.push({ name: '端到端 RAG 管线', pass: hitTop1 && genOk, detail: `命中doc=${out.retrieved[0].doc}(期望1) 生成${out.generated.tokens.length}token` });
    }

    return results;
  }

  const ENGINE = {
    makeTokenizer, newCache, forwardCached, forwardFull,
    sampleFromLogits, generate, generateFull,
    buildBM25, retrievalEval, trainTinyLM, rag, runSelfTest,
  };
  global.APEX_E = ENGINE;
  if (typeof module !== 'undefined' && module.exports) module.exports = ENGINE;

  if (typeof require !== 'undefined' && typeof module !== 'undefined' && require.main === module) {
    const r = runSelfTest();
    let pass = 0;
    console.log('\n=== APEX-ENGINE 自检 ===');
    r.forEach((x) => { console.log(`${x.pass ? '✅' : '❌'} ${x.name.padEnd(24)} | ${x.detail}`); if (x.pass) pass++; });
    console.log(`\n结果：${pass}/${r.length} 全绿`);
    process.exit(pass === r.length ? 0 : 1);
  }
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
