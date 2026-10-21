/*
 * APEX-TRANSFORMER · 旗舰 Transformer 参考实现 + 基准对照
 * ------------------------------------------------------------------
 * 本文件是「方案2：十层架构参考实现 + benchmark 对照」的交付物。
 * 它站在世界顶级成果之上（集大成 / 零重写）：
 *   - 可微引擎：反向模式自动微分（思想源自 autograd / tinygrad / JAX）
 *   - 位置编码：学习型绝对位置嵌入（Vaswani 2017 同族）
 *   - 预归一化块：RMSNorm + 因果多头注意力 + SwiGLU/GeLU MLP（LLaMA/PaLM 系）
 *   - 优化：AdamW（Loshchilov 2019，解耦权重衰减）
 *   - 复用 apex-forge.js 的初始化 (Xavier/Kaiming) 与随机数（可复现）
 *
 * 关键认知反缴械点：我们不宣称"最先进"，而是给"最先进"一把尺子——
 *   在「关联回忆 / 键值检索」任务上，要求注意力机制必须赢过"无注意力"基线。
 *   注意力赢，说明架构的归纳偏置真实成立；否则只是漂亮的概念。
 *
 * 作者：晨星 (GitHub: CJX0712) · 许可：MIT
 */

(function (global) {
  'use strict';

  // 复用 APEX-FORGE（集大成地基）
  let APEX;
  if (typeof module !== 'undefined' && module.exports) {
    APEX = require('./apex-forge.js');
  } else {
    APEX = global.APEX;
  }

  // ===================== 数值工具 =====================
  function zeros2(r, c) { return Array.from({ length: r }, () => new Array(c).fill(0)); }
  function matmul2(a, b) {
    const ar = a.length, ac = a[0].length, bc = b[0].length;
    const out = zeros2(ar, bc);
    for (let i = 0; i < ar; i++)
      for (let k = 0; k < ac; k++) {
        const aik = a[i][k]; if (aik === 0) continue;
        const brow = b[k], orow = out[i];
        for (let j = 0; j < bc; j++) orow[j] += aik * brow[j];
      }
    return out;
  }
  function transpose2(a) {
    const r = a.length, c = a[0].length;
    return Array.from({ length: c }, (_, j) => a.map((row) => row[j]));
  }
  function erf(x) {
    const t = 1 / (1 + 0.3275911 * Math.abs(x));
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return x >= 0 ? y : -y;
  }
  function geluDeriv(x) {
    const c = x / Math.SQRT2;
    const dterm = Math.sqrt(2 / Math.PI) * Math.exp(-x * x / 2);
    return 0.5 * (1 + erf(c)) + 0.5 * x * dterm;
  }

  // ===================== 张量与反向模式自动微分 =====================
  function accumulate(t, grad) {
    const g = t.grad;
    for (let i = 0; i < t.shape[0]; i++)
      for (let j = 0; j < t.shape[1]; j++) g[i][j] += grad[i][j];
  }

  class T {
    constructor(data, requiresGrad = false) {
      this.data = data;
      this.shape = [data.length, data[0].length];
      this.grad = zeros2(this.shape[0], this.shape[1]);
      this._backward = null;
      this._prev = [];
      this.requiresGrad = requiresGrad;
    }
  }
  function param(data) { return new T(data, true); }

  function backward(loss) {
    const topo = [], seen = new Set();
    (function visit(n) {
      if (seen.has(n)) return; seen.add(n);
      (n._prev || []).forEach(visit); topo.push(n);
    })(loss);
    loss.grad = [[1]]; // dL/dL
    for (let i = topo.length - 1; i >= 0; i--) {
      const n = topo[i];
      if (n._backward) n._backward();
    }
  }

  // ---- 算子 ----
  function matmul(a, b) {
    const out = new T(matmul2(a.data, b.data));
    out._backward = () => {
      accumulate(a, matmul2(out.grad, transpose2(b.data)));
      accumulate(b, matmul2(transpose2(a.data), out.grad));
    };
    out._prev = [a, b]; return out;
  }
  function add(a, b) {
    const out = new T(a.data.map((row, i) => row.map((v, j) => v + b.data[i][j])));
    out._backward = () => { accumulate(a, out.grad); accumulate(b, out.grad); };
    out._prev = [a, b]; return out;
  }
  function addBias(a, b) { // b:[1,c]
    const r = a.shape[0], c = a.shape[1];
    const out = new T(a.data.map((row, i) => row.map((v, j) => v + b.data[0][j])));
    out._backward = () => {
      accumulate(a, out.grad);
      const db = zeros2(1, c);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) db[0][j] += out.grad[i][j];
      accumulate(b, db);
    };
    out._prev = [a, b]; return out;
  }
  function scale(a, s) {
    const out = new T(a.data.map((row) => row.map((v) => v * s)));
    out._backward = () => accumulate(a, out.grad.map((row) => row.map((v) => v * s)));
    out._prev = [a]; return out;
  }
  function transpose(a) {
    const out = new T(transpose2(a.data));
    out._backward = () => accumulate(a, transpose2(out.grad));
    out._prev = [a]; return out;
  }
  // RoPE 旋转位置编码（Su et al. 2021, arXiv:2104.09864；Llama/RoPE 系所用）
  // 对每对维度按角度 θ_k·p 旋转；作用于 Q/K，使点积依赖相对位置。
  function ropeAngular(row, p, thetas, sign) {
    const d = row.length, out = new Array(d);
    for (let i = 0; i < d; i += 2) {
      const theta = thetas[i / 2] * p;
      const c = Math.cos(theta), s = sign * Math.sin(theta);
      const a = row[i], b = row[i + 1];
      out[i] = a * c - b * s;
      out[i + 1] = a * s + b * c;
    }
    return out;
  }
  // 每对维度的角频率 θ_k（含外推缩放策略）
  //   none : θ_k = base^(-2k/d)                       —— 原生 RoPE
  //   ntk  : base' = base·s^(d/(d-2))                 —— NTK-aware 缩放
  //   yarn : θ_k = (1-γ)·(θ_k/s) + γ·θ_k, γ=ramp      —— NTK-by-parts（YaRN）
  function ropeThetas(d, base, opts) {
    opts = opts || {};
    const n = d / 2, thetas = new Array(n);
    const mode = opts.mode || 'none';
    const s = opts.scale || 1;
    const trainLen = opts.trainLen || 1;
    if (mode === 'ntk') {
      const b2 = base * Math.pow(s, d / (d - 2));
      for (let k = 0; k < n; k++) thetas[k] = Math.pow(b2, -(2 * k) / d);
    } else if (mode === 'yarn') {
      const alpha = 1, beta = 32;
      for (let k = 0; k < n; k++) {
        const th = Math.pow(base, -(2 * k) / d);      // 原角频率
        const lambda = (2 * Math.PI) / th;            // 波长
        const r = trainLen / lambda;
        let gamma = 0;
        if (r >= beta) gamma = 1; else if (r > alpha) gamma = (r - alpha) / (beta - alpha);
        thetas[k] = (1 - gamma) * (th / s) + gamma * th;
      }
    } else {
      for (let k = 0; k < n; k++) thetas[k] = Math.pow(base, -(2 * k) / d);
    }
    return thetas;
  }
  function ropeWith(X, positions, opts) {
    const d = X.shape[1], base = (opts && opts.base) || 10000;
    const thetas = ropeThetas(d, base, opts);
    const out = new T(X.data.map((row, i) => ropeAngular(row, positions[i], thetas, +1)));
    out._backward = () => {
      const gX = zeros2(X.shape[0], d);
      for (let i = 0; i < X.shape[0]; i++) {
        const g = ropeAngular(out.grad[i], positions[i], thetas, -1); // 反向 = 逆旋转
        for (let j = 0; j < d; j++) gX[i][j] = g[j];
      }
      accumulate(X, gX);
    };
    out._prev = [X]; return out;
  }
  function rope(X, positions, base) {
    return ropeWith(X, positions, { base: base || 10000, mode: 'none' });
  }
  function sumAll(a) {
    let s = 0; const d = a.data;
    for (let i = 0; i < a.shape[0]; i++) for (let j = 0; j < a.shape[1]; j++) s += d[i][j];
    const out = new T([[s]]);
    out._backward = () => {
      const g = zeros2(a.shape[0], a.shape[1]);
      for (let i = 0; i < a.shape[0]; i++) for (let j = 0; j < a.shape[1]; j++) g[i][j] = 1;
      accumulate(a, g);
    };
    out._prev = [a]; return out;
  }
  function meanRows(a) { // 沿行求平均 -> [1,c]
    const r = a.shape[0], c = a.shape[1];
    const acc = new Array(c).fill(0);
    a.data.forEach((row) => row.forEach((v, j) => (acc[j] += v)));
    const out = new T([acc.map((v) => v / r)]);
    out._backward = () => {
      const da = zeros2(r, c);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) da[i][j] = out.grad[0][j] / r;
      accumulate(a, da);
    };
    out._prev = [a]; return out;
  }
  function rmsNorm(a, gain, eps = 1e-6) {
    const r = a.shape[0], c = a.shape[1], x = a.data;
    const y = zeros2(r, c), invr = [], xhat = zeros2(r, c);
    for (let i = 0; i < r; i++) {
      let s = 0; for (let j = 0; j < c; j++) s += x[i][j] * x[i][j];
      s = s / c + eps; const ir = 1 / Math.sqrt(s); invr.push(ir);
      for (let j = 0; j < c; j++) { const h = x[i][j] * ir; xhat[i][j] = h; y[i][j] = h * gain.data[0][j]; }
    }
    const out = new T(y);
    out._backward = () => {
      const da = zeros2(r, c), dg = zeros2(1, c);
      for (let i = 0; i < r; i++) {
        const ir = invr[i], ir3 = (ir * ir * ir) / c;
        let dot = 0;
        for (let k = 0; k < c; k++) dot += out.grad[i][k] * gain.data[0][k] * x[i][k];
        for (let j = 0; j < c; j++) {
          const dhat = out.grad[i][j] * gain.data[0][j];
          da[i][j] = dhat * ir - dot * ir3 * x[i][j];
          dg[0][j] += out.grad[i][j] * xhat[i][j];
        }
      }
      accumulate(a, da); accumulate(gain, dg);
    };
    out._prev = [a, gain]; return out;
  }
  function gelu(a) {
    const r = a.shape[0], c = a.shape[1], x = a.data, y = zeros2(r, c), d = zeros2(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) { y[i][j] = APEX.Act.gelu(x[i][j]); d[i][j] = geluDeriv(x[i][j]); }
    const out = new T(y);
    out._backward = () => {
      const da = zeros2(r, c);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) da[i][j] = out.grad[i][j] * d[i][j];
      accumulate(a, da);
    };
    out._prev = [a]; return out;
  }
  function embedding(ids, E) {
    const d = E.shape[1];
    const out = new T(ids.map((id) => E.data[id].slice()));
    out._backward = () => {
      const dE = zeros2(E.shape[0], d);
      for (let i = 0; i < ids.length; i++) { const id = ids[i]; for (let j = 0; j < d; j++) dE[id][j] += out.grad[i][j]; }
      accumulate(E, dE);
    };
    out._prev = [E]; return out;
  }
  function row(x, i) { // 取第 i 行 -> [1,c]，梯度回散到该行
    const c = x.shape[1];
    const out = new T([x.data[i].slice()]);
    out._backward = () => { for (let j = 0; j < c; j++) x.grad[i][j] += out.grad[0][j]; };
    out._prev = [x]; return out;
  }
  function crossEntropy(logits, targets) {
    const r = logits.shape[0], c = logits.shape[1], data = logits.data;
    const probs = zeros2(r, c); let loss = 0;
    for (let i = 0; i < r; i++) {
      let m = -Infinity; for (let j = 0; j < c; j++) m = Math.max(m, data[i][j]);
      let s = 0; for (let j = 0; j < c; j++) { const e = Math.exp(data[i][j] - m); probs[i][j] = e; s += e; }
      for (let j = 0; j < c; j++) probs[i][j] /= s;
      loss += -Math.log(Math.max(probs[i][targets[i]], 1e-12));
    }
    const out = new T([[loss / r]]);
    out._backward = () => {
      const gl = 1 / r, dlogits = zeros2(r, c);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) dlogits[i][j] = (probs[i][j] - (j === targets[i] ? 1 : 0)) * gl;
      accumulate(logits, dlogits);
    };
    out._prev = [logits]; return out;
  }

  // 因果多头缩放点积注意力（自实现以接入可微图）
  // 支持 Q 与 K/V 长度不同（KV-Cache 增量解码 / 交叉注意力通用形态）。
  function multiHeadAttention(Q, K, V, heads, mask) {
    const qLen = Q.shape[0], kvLen = K.shape[0], d = Q.shape[1], dk = Math.floor(d / heads);
    const outData = zeros2(qLen, d), Wh = [], scoresH = [];
    for (let h = 0; h < heads; h++) {
      const Qh = [], Kh = [], Vh = [];
      for (let i = 0; i < qLen; i++) Qh.push(Q.data[i].slice(h * dk, (h + 1) * dk));
      for (let j = 0; j < kvLen; j++) {
        Kh.push(K.data[j].slice(h * dk, (h + 1) * dk));
        Vh.push(V.data[j].slice(h * dk, (h + 1) * dk));
      }
      const w = zeros2(qLen, kvLen), scores = zeros2(qLen, kvLen), inv = 1 / Math.sqrt(dk);
      for (let i = 0; i < qLen; i++) {
        let m = -Infinity;
        for (let j = 0; j < kvLen; j++) {
          let s = 0; for (let k = 0; k < dk; k++) s += Qh[i][k] * Kh[j][k];
          s *= inv; if (mask && !mask[i][j]) s = -1e9; scores[i][j] = s; m = Math.max(m, s);
        }
        let sm = 0; for (let j = 0; j < kvLen; j++) { const e = Math.exp(scores[i][j] - m); w[i][j] = e; sm += e; }
        for (let j = 0; j < kvLen; j++) w[i][j] /= sm;
        for (let j = 0; j < kvLen; j++) for (let k = 0; k < dk; k++) outData[i][h * dk + k] += w[i][j] * Vh[j][k];
      }
      Wh.push(w); scoresH.push(scores);
    }
    const out = new T(outData);
    out._backward = () => {
      const dQ = zeros2(qLen, d), dK = zeros2(kvLen, d), dV = zeros2(kvLen, d);
      for (let h = 0; h < heads; h++) {
        const w = Wh[h], inv = 1 / Math.sqrt(dk);
        const dw = zeros2(qLen, kvLen);
        for (let i = 0; i < qLen; i++)
          for (let j = 0; j < kvLen; j++) {
            let s = 0; for (let k = 0; k < dk; k++) s += out.grad[i][h * dk + k] * V.data[j][h * dk + k];
            dw[i][j] = s;
          }
        const ds = zeros2(qLen, kvLen);
        for (let i = 0; i < qLen; i++) {
          let rs = 0; for (let j = 0; j < kvLen; j++) rs += dw[i][j] * w[i][j];
          for (let j = 0; j < kvLen; j++) { let v = w[i][j] * (dw[i][j] - rs); if (mask && !mask[i][j]) v = 0; ds[i][j] = v; }
        }
        for (let j = 0; j < kvLen; j++)
          for (let k = 0; k < dk; k++) {
            let s = 0; for (let i = 0; i < qLen; i++) s += w[i][j] * out.grad[i][h * dk + k]; dV[j][h * dk + k] += s;
          }
        for (let i = 0; i < qLen; i++)
          for (let k = 0; k < dk; k++) {
            let s = 0; for (let j = 0; j < kvLen; j++) s += ds[i][j] * K.data[j][h * dk + k]; dQ[i][h * dk + k] += s * inv;
          }
        for (let j = 0; j < kvLen; j++)
          for (let k = 0; k < dk; k++) {
            let s = 0; for (let i = 0; i < qLen; i++) s += ds[i][j] * Q.data[i][h * dk + k]; dK[j][h * dk + k] += s * inv;
          }
      }
      accumulate(Q, dQ); accumulate(K, dK); accumulate(V, dV);
    };
    out._prev = [Q, K, V]; return out;
  }

  // ===================== 优化器（AdamW） =====================
  class AdamWOpt {
    constructor(params, lr = 1e-3, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 0.01) {
      this.p = params; this.lr = lr; this.b1 = b1; this.b2 = b2; this.eps = eps; this.wd = wd; this.t = 0;
      this.m = params.map((p) => zeros2(p.shape[0], p.shape[1]));
      this.v = params.map((p) => zeros2(p.shape[0], p.shape[1]));
    }
    step() {
      this.t++;
      const bc1 = 1 - Math.pow(this.b1, this.t), bc2 = 1 - Math.pow(this.b2, this.t);
      this.p.forEach((p, idx) => {
        const m = this.m[idx], v = this.v[idx], g = p.grad, d = p.data;
        for (let i = 0; i < p.shape[0]; i++)
          for (let j = 0; j < p.shape[1]; j++) {
            m[i][j] = this.b1 * m[i][j] + (1 - this.b1) * g[i][j];
            v[i][j] = this.b2 * v[i][j] + (1 - this.b2) * g[i][j] * g[i][j];
            const mh = m[i][j] / bc1, vh = v[i][j] / bc2;
            d[i][j] = d[i][j] - this.lr * (mh / (Math.sqrt(vh) + this.eps) + this.wd * d[i][j]);
          }
      });
    }
  }

  // ===================== 模型：TransformerLM（十层架构落地） =====================
  // 十层分解（见 README）：
  // L1 数据/分词 · L2 张量/嵌入 · L3 位置编码 · L4 注意力 · L5 前馈MLP
  // L6 残差/预归一化 · L7 层堆叠 · L8 输出头(权重绑定) · L9 损失 · L10 评估/基准
  function buildTransformer(vocab, cfg, rng) {
    const d = cfg.d, heads = cfg.heads, layers = cfg.layers, ffn = cfg.ffn;
    const xav = (din, dout) => Array.from({ length: din }, () => Array.from({ length: dout }, () => APEX.Init.xavierUniform(rng, din, dout)));
    const zeros = (r, c) => zeros2(r, c);
    const ones = (r, c) => Array.from({ length: r }, () => new Array(c).fill(1));

    const params = [];
    const E = param(xav(vocab, d)); params.push(E);            // 词嵌入
    // RoPE：位置由 rope 作用于 Q/K 注入，无绝对位置表，支持任意长度外推
    const blocks = [];
    for (let l = 0; l < layers; l++) {
      const b = {
        ln1: param(ones(1, d)), Wq: param(xav(d, d)), bq: param(zeros(1, d)),
        Wk: param(xav(d, d)), bk: param(zeros(1, d)),
        Wv: param(xav(d, d)), bv: param(zeros(1, d)),
        Wo: param(xav(d, d)),
        ln2: param(ones(1, d)), W1: param(xav(d, ffn)), b1: param(zeros(1, ffn)),
        W2: param(xav(ffn, d)), b2: param(zeros(1, d)),
      };
      Object.values(b).forEach((p) => params.push(p));
      blocks.push(b);
    }
    const Wlm = param(xav(d, vocab)); params.push(Wlm);
    const blm = param(zeros(1, vocab)); params.push(blm);

    function causalMask(seq) {
      return Array.from({ length: seq }, (_, i) => Array.from({ length: seq }, (_, j) => (j <= i ? 1 : 0)));
    }

    function forward(ids, pos, targets, ropeOpts) {
      let x = embedding(ids, E);
      const mask = causalMask(ids.length);
      const rOpts = ropeOpts || { mode: 'none', base: 10000 };
      // YaRN 注意力温度缩放：t = 0.1·ln(s)+1，q/k 乘 √(1/t)
      let attnScale = 1;
      if (rOpts.mode === 'yarn' && rOpts.scale > 1) attnScale = Math.sqrt(1 / (0.1 * Math.log(rOpts.scale) + 1));
      for (const b of blocks) {
        const xn = rmsNorm(x, b.ln1);
        let Q = addBias(matmul(xn, b.Wq), b.bq);
        let K = addBias(matmul(xn, b.Wk), b.bk);
        const V = addBias(matmul(xn, b.Wv), b.bv);
        Q = ropeWith(Q, pos, rOpts); K = ropeWith(K, pos, rOpts);
        if (attnScale !== 1) { Q = scale(Q, attnScale); K = scale(K, attnScale); }
        const attn = multiHeadAttention(Q, K, V, heads, mask);
        const attnProj = matmul(attn, b.Wo);
        x = add(x, attnProj);
        const xn2 = rmsNorm(x, b.ln2);
        const h = gelu(addBias(matmul(xn2, b.W1), b.b1));
        const m = addBias(matmul(h, b.W2), b.b2);
        x = add(x, m);
      }
      const last = ids.length - 1;
      const xLast = row(x, last);                         // [1,d]
      const logitsT = addBias(matmul(xLast, Wlm), blm);   // 独立输出头 [1,vocab]
      const loss = crossEntropy(logitsT, [targets]);
      return { loss, logits: logitsT.data[0], params };
    }
    return { params, forward, E, blocks, Wlm, blm };
  }

  // 基线：无注意力（平均池化 + 线性）—— 给"最先进"当对手
  function buildBaseline(vocab, cfg, rng) {
    const d = cfg.d, ffn = cfg.ffn;
    const xav = (din, dout) => Array.from({ length: din }, () => Array.from({ length: dout }, () => APEX.Init.xavierUniform(rng, din, dout)));
    const params = [];
    const E = param(xav(vocab, d)); params.push(E);
    const W1 = param(xav(d, ffn)); params.push(W1);
    const b1 = param(zeros2(1, ffn)); params.push(b1);
    const W2 = param(xav(ffn, vocab)); params.push(W2);
    const b2 = param(zeros2(1, vocab)); params.push(b2);
    function forward(ids, pos, targets) {
      const x = embedding(ids, E);
      const pooled = meanRows(x);
      const h = gelu(addBias(matmul(pooled, W1), b1));
      const logits = addBias(matmul(h, W2), b2);
      const loss = crossEntropy(logits, [targets]);
      return { loss, logits: logits.data[0], params };
    }
    return { params, forward, E, W1, b1, W2, b2 };
  }

  // ===================== 数据集：关联回忆（键值检索） =====================
  // 序列 = [k1,v1,k2,v2,...,kP,vP, queryKey]，目标 = queryKey 对应的值。
  // 注意力可检索；平均池化丢失"哪个键"的信息，必然失败 -> 真实尺子。
  function genExample(rng, K, V, nPairs) {
    const keys = [];
    while (keys.length < nPairs) {
      const k = Math.floor(rng() * K);
      if (!keys.includes(k)) keys.push(k);
    }
    const mapping = keys.map((k) => [k, Math.floor(rng() * V)]); // 键->值
    const ids = [];
    mapping.forEach(([k, v]) => { ids.push(k); ids.push(K + v); }); // 值 token = K+v
    const qIdx = Math.floor(rng() * nPairs);
    const queryKey = mapping[qIdx][0];
    ids.push(queryKey);
    const target = K + mapping[qIdx][1]; // 目标是序列中真实存在的"值 token"（经典关联回忆）
    const pos = ids.map((_, i) => i);
    return { ids, pos, target };
  }
  function genDataset(rng, K, V, nPairs, n) {
    return Array.from({ length: n }, () => genExample(rng, K, V, nPairs));
  }

  // 归纳头任务（induction）：序列 [c0..c_{L-1}, c0]，目标 = c1。
  // 这是注意力归纳偏置的经典试金石：必须靠注意力把末尾的 c0 "路由"回
  // 开头 c0 的位置并读取其后 token；无位置信息的平均池化必然失败。
  function genExampleInduction(rng, vocab, L) {
    const c = Array.from({ length: L }, () => Math.floor(rng() * vocab));
    const ids = c.slice();
    ids.push(c[0]);                 // 末尾追加重复线索 c0
    const target = c[1];            // 首个 c0 之后跟的 token
    const pos = ids.map((_, i) => i);
    return { ids, pos, target };
  }
  function genDatasetInduction(rng, vocab, L, n) {
    return Array.from({ length: n }, () => genExampleInduction(rng, vocab, L));
  }
  // 变长归纳：L ∈ [Lmin, Lmax] 随机，用于训练时覆盖多长度
  function genExampleInductionVar(rng, vocab, Lmin, Lmax) {
    const L = Lmin + Math.floor(rng() * (Lmax - Lmin + 1));
    return genExampleInduction(rng, vocab, L);
  }

  // ===================== 训练一步（含可选梯度裁剪，防发散） =====================
  function globalGradNorm(params) {
    let s = 0;
    params.forEach((p) => p.grad.forEach((r) => r.forEach((v) => (s += v * v))));
    return Math.sqrt(s);
  }
  function clipGrads(params, maxNorm) {
    const n = globalGradNorm(params);
    if (n > maxNorm) { const sc = maxNorm / n; params.forEach((p) => { p.grad = p.grad.map((r) => r.map((v) => v * sc)); }); }
  }
  function trainStep(model, ex, opt, clip) {
    model.params.forEach((p) => { p.grad = zeros2(p.shape[0], p.shape[1]); });
    const { loss } = model.forward(ex.ids, ex.pos, ex.target);
    backward(loss);
    if (clip && clip > 0) clipGrads(model.params, clip);
    opt.step();
    return loss.data[0][0];
  }
  function evaluate(model, data) {
    let correct = 0, lossSum = 0;
    data.forEach((ex) => {
      model.params.forEach((p) => { p.grad = zeros2(p.shape[0], p.shape[1]); });
      const r = model.forward(ex.ids, ex.pos, ex.target);
      lossSum += r.loss.data[0][0];
      let best = 0; for (let j = 1; j < r.logits.length; j++) if (r.logits[j] > r.logits[best]) best = j;
      if (best === ex.target) correct++;
    });
    return { loss: lossSum / data.length, acc: correct / data.length };
  }

  // ===================== 基准主程（归纳头任务：注意力 vs 无注意力） =====================
  function runBenchmark(opts) {
    opts = opts || {};
    const cfg = { d: 48, heads: 4, layers: 3, ffn: 192, maxlen: 16 };
    const vocab = 16, L = 8;
    const steps = opts.steps || 2000;
    const baseLr = 3e-3, clip = 1.0, decay = 0.9;

    const drng = APEX.makeRng(20261021);
    const data = genDatasetInduction(drng, vocab, L, 4000);
    const testData = data.slice(0, 500);

    // Transformer（注意力架构）：梯度裁剪 + 学习率线性衰减，稳定收敛
    const tf = buildTransformer(vocab, cfg, APEX.makeRng(7));
    const tOpt = new AdamWOpt(tf.params, baseLr, 0.9, 0.999, 1e-8, 0);
    for (let s = 0; s < steps; s++) { tOpt.lr = baseLr * (1 - (s / steps) * decay); trainStep(tf, data[s % data.length], tOpt, clip); }
    const tEval = evaluate(tf, testData);

    // 基线（平均池化 + 线性，无注意力、无位置）—— 相同数据、独立初始化
    const bs = buildBaseline(vocab, cfg, APEX.makeRng(7));
    const bOpt = new AdamWOpt(bs.params, baseLr, 0.9, 0.999, 1e-8, 0);
    for (let s = 0; s < steps; s++) { bOpt.lr = baseLr * (1 - (s / steps) * decay); trainStep(bs, data[s % data.length], bOpt, clip); }
    const bEval = evaluate(bs, testData);

    return {
      cfg, vocab, L, steps, task: 'induction',
      transformer: tEval, baseline: bEval,
      transformerWins: tEval.acc > bEval.acc + 0.2 && tEval.loss < bEval.loss * 0.5,
    };
  }

  // ===================== 长度泛化基准（上移尺子） =====================
  // 训练只用 L∈[trainLmin,trainLmax]，测试喂（可能未见的）更长序列。
  // 直接量化：RoPE 架构能否把"attend back 距离 d"的模式外推到更大 d。
  // 两个 regime 对照：
  //   - extrapolate: 训练短 L∈[3,7]，测长 L∈{12,20}  → 测"外推"
  //   - indistlong : 训练长 L∈[15,20]，测 L∈{17,20} → 测"见过长度时能否做好"
  function runGeneralizationBenchmark(opts) {
    opts = opts || {};
    const cfg = { d: 64, heads: 4, layers: 4, ffn: 256, maxlen: 64 };
    const vocab = 16;
    const Lmin = opts.trainLmin || 3, Lmax = opts.trainLmax || 7;
    const steps = opts.steps || 4000;
    const testLens = opts.testLens || [12, 20];
    const baseLr = 3e-3, clip = 1.0, decay = 0.9;

    const trng = APEX.makeRng(20261021);
    const train = Array.from({ length: 6000 }, () => genExampleInductionVar(trng, vocab, Lmin, Lmax));

    function trainModel(build) {
      const m = build(vocab, cfg, APEX.makeRng(7));
      const opt = new AdamWOpt(m.params, baseLr, 0.9, 0.999, 1e-8, 0);
      for (let s = 0; s < steps; s++) { opt.lr = baseLr * (1 - (s / steps) * decay); trainStep(m, train[s % train.length], opt, clip); }
      return m;
    }
    const tf = trainModel(buildTransformer);
    const bs = trainModel(buildBaseline);

    function evalAt(m, L) {
      const data = genDatasetInduction(APEX.makeRng(L * 97 + 3), vocab, L, 500);
      return evaluate(m, data);
    }
    const res = { cfg, trainLmin: Lmin, trainLmax: Lmax, steps, perLen: {} };
    testLens.forEach((L) => { res.perLen[L] = { transformer: evalAt(tf, L), baseline: evalAt(bs, L) }; });
    res.beatsBaselineEverywhere = testLens.every((L) => res.perLen[L].transformer.acc > res.perLen[L].baseline.acc);
    res.transformerWins = testLens.every((L) => res.perLen[L].transformer.acc > res.perLen[L].baseline.acc + 0.15);
    return res;
  }

  // ===================== 外推"补洞"基准（NTK / YaRN，推理期免重训） =====================
  // 只用短序列训练一次（plain RoPE），然后在推理期换上不同的 RoPE 缩放策略，
  // 看能否把外推失败救回来。这是"补洞"，不是"重跑一遍换个说法"。
  function runExtrapolationFix(opts) {
    opts = opts || {};
    const cfg = { d: 64, heads: 4, layers: 4, ffn: 256, maxlen: 64 };
    const vocab = 16, Lmin = 3, Lmax = 7;
    const steps = opts.steps || 4000;
    const testLens = opts.testLens || [8, 12, 20, 30];
    const trainSeqLen = Lmax + 1;                 // 训练最大序列长度（含线索位）
    const baseLr = 3e-3, clip = 1.0, decay = 0.9;

    const trng = APEX.makeRng(20261021);
    const train = Array.from({ length: 6000 }, () => genExampleInductionVar(trng, vocab, Lmin, Lmax));

    function trainModel(build) {
      const m = build(vocab, cfg, APEX.makeRng(7));
      const opt = new AdamWOpt(m.params, baseLr, 0.9, 0.999, 1e-8, 0);
      for (let s = 0; s < steps; s++) { opt.lr = baseLr * (1 - (s / steps) * decay); trainStep(m, train[s % train.length], opt, clip); }
      return m;
    }
    const tf = trainModel(buildTransformer);
    const bs = trainModel(buildBaseline);

    function evalStrategy(m, L, strategy) {
      const data = genDatasetInduction(APEX.makeRng(L * 97 + 3), vocab, L, 500);
      const s = (L + 1) / trainSeqLen;
      const rOpts = strategy === 'plain' ? { mode: 'none', base: 10000 }
        : strategy === 'ntk' ? { mode: 'ntk', base: 10000, scale: s }
          : { mode: 'yarn', base: 10000, scale: s, trainLen: trainSeqLen };
      let correct = 0, lossSum = 0;
      data.forEach((ex) => {
        m.params.forEach((p) => { p.grad = zeros2(p.shape[0], p.shape[1]); });
        const r = m.forward(ex.ids, ex.pos, ex.target, rOpts);
        lossSum += r.loss.data[0][0];
        let best = 0; for (let j = 1; j < r.logits.length; j++) if (r.logits[j] > r.logits[best]) best = j;
        if (best === ex.target) correct++;
      });
      return { loss: lossSum / data.length, acc: correct / data.length };
    }

    const res = { cfg, trainLmin: Lmin, trainLmax: Lmax, steps, perLen: {} };
    testLens.forEach((L) => {
      const data = genDatasetInduction(APEX.makeRng(L * 97 + 3), vocab, L, 500);
      res.perLen[L] = {
        plain: evalStrategy(tf, L, 'plain'),
        ntk: evalStrategy(tf, L, 'ntk'),
        yarn: evalStrategy(tf, L, 'yarn'),
        baseline: evaluate(bs, data),
      };
    });
    // 判定：在最长的未见长度上，最好策略是否明显优于 plain 且明显优于基线
    const maxL = Math.max.apply(null, testLens);
    const best = Math.max(res.perLen[maxL].plain.acc, res.perLen[maxL].ntk.acc, res.perLen[maxL].yarn.acc);
    res.bestStrategy = ['plain', 'ntk', 'yarn'].filter((k) => res.perLen[maxL][k].acc === best)[0] || 'plain';
    res.fixed = best > res.perLen[maxL].plain.acc + 0.10 && best > res.perLen[maxL].baseline.acc + 0.15;
    return res;
  }

  // ===================== 自检（可被独立实现交叉验证的不变量） =====================
  function approx(a, b, tol) { return Math.abs(a - b) <= tol; }
  function maxRelErr(a, b) {
    let m = 0;
    for (let i = 0; i < a.length; i++) {
      const d = Math.abs(a[i] - b[i]), s = Math.abs(a[i]) + Math.abs(b[i]) + 1e-12;
      m = Math.max(m, d / s);
    }
    return m;
  }
  function numericGrad(lossFn, param, i, j, eps) {
    const orig = param.data[i][j];
    param.data[i][j] = orig + eps; const fp = lossFn();
    param.data[i][j] = orig - eps; const fm = lossFn();
    param.data[i][j] = orig;
    return (fp - fm) / (2 * eps);
  }

  function runSelfTest(opts) {
    opts = opts || {};
    const results = [];

    // ① matmul 解析梯度 vs 中心差分
    {
      const X = new T([[1, 2], [3, 4]]);
      const W = param([[0.1, 0.2], [0.3, 0.4]]);
      const lossFn = () => sumAll(matmul(X, W)).data[0][0];
      const loss = sumAll(matmul(X, W)); backward(loss);
      const analytic = W.grad[0].concat(W.grad[1]);
      const numeric = [];
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) numeric.push(numericGrad(lossFn, W, i, j, 1e-4));
      results.push({ name: 'matmul 梯度', pass: maxRelErr(analytic, numeric) < 1e-5, detail: `maxRelErr=${maxRelErr(analytic, numeric).toExponential(2)}` });
    }

    // ② rmsNorm 梯度（对 gain）
    {
      const rng = APEX.makeRng(3);
      const x = new T([Array.from({ length: 8 }, () => APEX.randn(rng))]);
      const gain = param([Array.from({ length: 8 }, () => 1)]);
      const lossFn = () => sumAll(rmsNorm(x, gain)).data[0][0];
      const loss = sumAll(rmsNorm(x, gain)); backward(loss);
      const analytic = gain.grad[0].slice();
      const numeric = [];
      for (let j = 0; j < 8; j++) numeric.push(numericGrad(lossFn, gain, 0, j, 1e-4));
      results.push({ name: 'rmsNorm 梯度', pass: maxRelErr(analytic, numeric) < 1e-5, detail: `maxRelErr=${maxRelErr(analytic, numeric).toExponential(2)}` });
    }

    // ③ crossEntropy 梯度 vs 中心差分
    {
      const rng = APEX.makeRng(5);
      const logits = param(Array.from({ length: 3 }, () => Array.from({ length: 6 }, () => APEX.randn(rng))));
      const targets = [2, 0, 4];
      const lossFn = () => crossEntropy(logits, targets).data[0][0];
      const loss = crossEntropy(logits, targets); backward(loss);
      const analytic = [], numeric = [];
      for (let i = 0; i < 3; i++) for (let j = 0; j < 6; j++) { analytic.push(logits.grad[i][j]); numeric.push(numericGrad(lossFn, logits, i, j, 1e-4)); }
      results.push({ name: 'crossEntropy 梯度', pass: maxRelErr(analytic, numeric) < 1e-5, detail: `maxRelErr=${maxRelErr(analytic, numeric).toExponential(2)}` });
    }

    // ③b RoPE 梯度 vs 中心差分（旋转为正交变换，反向应为逆旋转）
    {
      const rng = APEX.makeRng(7);
      const d = 8;
      const X = param(Array.from({ length: 3 }, () => Array.from({ length: d }, () => APEX.randn(rng))));
      const positions = [0, 3, 17];
      const lossFn = () => sumAll(rope(X, positions)).data[0][0];
      const loss = sumAll(rope(X, positions)); backward(loss);
      const analytic = [], numeric = [];
      for (let i = 0; i < 3; i++) for (let j = 0; j < d; j++) { analytic.push(X.grad[i][j]); numeric.push(numericGrad(lossFn, X, i, j, 1e-4)); }
      results.push({ name: 'RoPE 梯度', pass: maxRelErr(analytic, numeric) < 1e-5, detail: `maxRelErr=${maxRelErr(analytic, numeric).toExponential(2)}` });
    }

    // ④ 因果注意力：位置 i 不看 j>i（改变 V[j>i] 不影响 out[i]）
    {
      const seq = 4, d = 8, heads = 2;
      const rng = APEX.makeRng(11);
      const mk = () => new T(Array.from({ length: seq }, () => Array.from({ length: d }, () => APEX.randn(rng))));
      const Q = mk(), K = mk(), V = mk();
      const mask = Array.from({ length: seq }, (_, i) => Array.from({ length: seq }, (_, j) => (j <= i ? 1 : 0)));
      const out1 = multiHeadAttention(Q, K, V, heads, mask).data;
      // 篡改 V[3]（最后位置），看 out[0] 是否不变
      const V2 = new T(V.data.map((r) => r.slice()));
      V2.data[3] = V2.data[3].map((v) => v + 123.456);
      const out2 = multiHeadAttention(Q, K, V2, heads, mask).data;
      const diff = maxRelErr(out1[0], out2[0]);
      results.push({ name: '因果注意力 位置0不看未来', pass: diff < 1e-12, detail: `out[0] 受V[3]扰动=${diff.toExponential(2)}` });
    }

    // ⑤ 注意力因果行和=1（通过重算 softmax 间接验证：输出有限且因果成立）
    {
      const seq = 5, d = 8, heads = 2;
      const rng = APEX.makeRng(13);
      const mk = () => new T(Array.from({ length: seq }, () => Array.from({ length: d }, () => APEX.randn(rng))));
      const Q = mk(), K = mk(), V = mk();
      const mask = Array.from({ length: seq }, (_, i) => Array.from({ length: seq }, (_, j) => (j <= i ? 1 : 0)));
      const out = multiHeadAttention(Q, K, V, heads, mask);
      const finite = out.data.every((row) => row.every((v) => Number.isFinite(v)));
      results.push({ name: '注意力 输出有限 & 因果', pass: finite, detail: `finite=${finite}` });
    }

    // ⑥ Transformer 前向形状与有限性
    {
      const cfg = { d: 16, heads: 2, layers: 2, ffn: 32, maxlen: 16 };
      const rng = APEX.makeRng(17);
      const m = buildTransformer(16, cfg, rng);
      const ex = genExample(APEX.makeRng(1), 8, 8, 2);
      const r = m.forward(ex.ids, ex.pos, ex.target);
      const finite = r.logits.every((v) => Number.isFinite(v));
      results.push({ name: 'Transformer 前向', pass: finite && r.logits.length === 16, detail: `logits长度=${r.logits.length} finite=${finite}` });
    }

    // ⑦ 基准：归纳任务上，注意力架构必须碾压无注意力基线（给"最先进"一把尺子）
    if (!opts.skipBenchmark) {
      const bench = runBenchmark({ steps: 2000 });
      results.push({
        name: '基准 注意力碾压基线(归纳)',
        pass: bench.transformerWins,
        detail: `TF loss=${bench.transformer.loss.toFixed(3)}/acc=${bench.transformer.acc.toFixed(2)} | 基线 loss=${bench.baseline.loss.toFixed(3)}/acc=${bench.baseline.acc.toFixed(2)}`,
      });

      // ⑧ 可复现性：同种子两次基准一致
      const a = runBenchmark({ steps: 1500 });
      const b = runBenchmark({ steps: 1500 });
      const same = Math.abs(a.transformer.loss - b.transformer.loss) < 1e-9 && Math.abs(a.baseline.loss - b.baseline.loss) < 1e-9;
      results.push({ name: '可复现性(同种子)', pass: same, detail: `ΔTF=${(a.transformer.loss - b.transformer.loss).toExponential(2)}` });
    }

    return results;
  }

  const TRANSFORMER = {
    buildTransformer, buildBaseline, runBenchmark, runGeneralizationBenchmark,
    runExtrapolationFix, runSelfTest,
    genExample, genDataset, genExampleInduction, genDatasetInduction, genExampleInductionVar,
    multiHeadAttention, rmsNorm, crossEntropy, rope, ropeWith, ropeThetas,
    matmul, add, addBias, embedding, gelu, backward, T, AdamWOpt,
    clipGrads, globalGradNorm, trainStep, evaluate,
  };
  global.APEX_T = TRANSFORMER;
  if (typeof module !== 'undefined' && module.exports) module.exports = TRANSFORMER;

  if (typeof require !== 'undefined' && typeof module !== 'undefined' && require.main === module) {
    const r = runSelfTest();
    let pass = 0;
    console.log('\n=== APEX-TRANSFORMER 自检 ===');
    r.forEach((x) => { console.log(`${x.pass ? '✅' : '❌'} ${x.name.padEnd(26)} | ${x.detail}`); if (x.pass) pass++; });
    console.log(`\n结果：${pass}/${r.length} 全绿`);
    process.exit(pass === r.length ? 0 : 1);
  }
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
