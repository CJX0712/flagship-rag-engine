# APEX-ENGINE · 端到端推理引擎 + RAG 检索

> 把前三块地基（[apex-forge](https://github.com/CJX0712/flagship-algo-forge) 算法库 / [apex-transformer](https://github.com/CJX0712/flagship-transformer) 参考实现）变成一套**真能跑**的系统。

## 旗舰哲学（贯穿全链路）

- **零重写 · 集大成**：每一行代码站在世界顶级成果之上，凡有更强现成方案立即复用、组合、超越，从不从零造轮子。
- **闭环交付**：可运行、可复现、可验证，开源至 GitHub（署名 **晨星**）。
- **尺子先行**：用可被独立实现交叉验证的**数学不变量**给自己打分，不靠"看起来对"。

## 四大件（站在谁肩上）

| 组件 | 世界顶级出处 | 说明 |
|------|--------------|------|
| 分词器 | 词级 tokenization（标准做法） | 可往返 `encode/decode` |
| **KV-Cache 推理引擎** | vLLM / TGI / TensorRT-LLM 工业标配 | 每步只算新 token，复用历史 K/V；正确性由「与全量重算**逐位等价**」硬保证 |
| 采样 | Top-k (Fan 2018) / Top-p (Holtzman 2020)，复用 apex-forge | greedy / temperature / top-k / top-p |
| **BM25 检索器** | Robertson & Zaragoza 2009,《The Probabilistic Relevance Framework》 | 信息检索黄金标准 lexical 排序；实测 Recall@k / MRR |
| RAG 管线 | 标准 RAG（query → BM25 top-k → 拼 prompt → LM 生成） | 端到端连通 |

## 自检结果（9/9 全绿，可被独立实现交叉验证）

| # | 不变量 | 结果 | 说明 |
|---|--------|------|------|
| ① | 分词器往返一致 | ✅ | `decode(encode(s)) === s` |
| ② | **KV-Cache ≡ 全量重算** | ✅ | 最大偏差 `0.00e+0`（逐位等价） |
| ③ | greedy 解码一致 | ✅ | KV-Cache 路径 == 全量重算路径（token 级） |
| ④ | top-p 保留质量 ≥ p | ✅ | 保留 `0.853 ≥ 0.8` |
| ⑤ | 采样可复现（同种子） | ✅ | 两次 `rng` 同 → token 序列同 |
| ⑥ | BM25 IDF 单调 | ✅ | 稀有词 IDF > 常见词 IDF |
| ⑦ | BM25 词频饱和 | ✅ | 重复同一词增益递减 |
| ⑧ | BM25 检索 Recall@1/MRR | ✅ | 已知 ground-truth 语料上 `1.00 / 1.00` |
| ⑨ | 端到端 RAG 管线 | ✅ | 检索命中 + 生成连通（5 个合法 token） |

运行：`node apex-engine.js`

## 诚实记录：这一轮照出的洞（已修）

**KV-Cache 与全量重算不等价（最早偏差 4.19）**。

根因：底层 `multiHeadAttention` 用 `seq = Q.shape[0]` 同时驱动 Q、K、V 三个切片循环。KV-Cache 增量解码时 Q 只有 1 行、K/V 有 L 行，于是 K/V 只取到了**第一个缓存 key**——表现就是 `len=1` 恰好正确、`len≥2` 直接崩。

修复：把 `seq` 拆成 `qLen = Q.shape[0]` 与 `kvLen = K.shape[0]`，注意力与反向传播都按各自长度走（这也是 KV-Cache 解码与交叉注意力的通用形态）。修复后两条路径**逐位等价（偏差 0.00）**。

**这条不能外包给 AI**：KV-Cache 是工业界推理标配，如果"看起来能生成"就交差，没人会发现它只看了第一个历史 token——生成结果在短序列上居然"碰巧"对的。没有"逐位等价"这道硬尺子，这个洞会被永远埋着。

## 用法

```js
const E = require('./apex-engine.js');

// 1) 分词器
const tk = E.makeTokenizer(['注意力 机制 计算 相关性']);

// 2) KV-Cache 生成（默认 greedy）
const m = TR.buildTransformer(vocab, cfg, rng);
const { tokens, text } = E.generate(m, promptIds, { heads: cfg.heads, maxTokens: 8, tokenizer: tk });

// 3) BM25 检索
const bm = E.buildBM25(docs, tk);
const top = bm.topK('查询', 3);

// 4) 端到端 RAG
const rag = E.rag(query, docs, bm, m, tk, { k: 2, heads: cfg.heads });
```

浏览器：打开 `index.html` → 点「运行全部自检」看 9 张卡片；点「载入 RAG 演示」训练玩具小模型后可实时检索+生成。

## 诚实声明

本引擎验证的是**「引擎本身正确」**（KV-Cache 等价、检索指标、管线连通）。搭载的是**玩具级小模型**，**不宣称**生成质量可用——RAG 演示里 LM 生成的文本是随机性的，重点在"检索命中 + 管线跑通"，不在"语言流畅"。

## 作者与许可

作者：**晨星**（GitHub: [CJX0712](https://github.com/CJX0712)） · 许可：MIT
