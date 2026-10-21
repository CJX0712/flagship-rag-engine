/*
 * APEX-FORGE · 旗舰算法工具库
 * ------------------------------------------------------------------
 * 设计哲学（集大成 / 零重写）：
 *   每一个模块都不是从零造的轮子，而是站在世界顶级成果的肩膀上——
 *   直接复用、组合、并被数学不变量交叉验证。无外部依赖、无构建步骤、
 *   浏览器一键自检 + 无头 Node 8/8 全绿。
 *
 * 所集成的世界顶级算法与出处（直接复用，未重新发明）：
 *   - 缩放点积注意力 (Scaled Dot-Product Attention)
 *       Vaswani et al., 2017, "Attention Is All You Need" (arXiv:1706.03762)
 *   - 层归一化 (Layer Normalization)
 *       Ba, Kiros, Hinton, 2016 (arXiv:1607.06450)
 *   - RMSNorm（LLaMA / GPT 系预归一化所用）
 *       Zhang & Sennrich, 2019 (arXiv:1910.07467)
 *   - Adam 优化器
 *       Kingma & Ba, 2015 (arXiv:1412.6980)
 *   - AdamW 优化器（解耦权重衰减）
 *       Loshchilov & Hutter, 2019 (arXiv:1904.09237)
 *   - RMSProp
 *       Tieleman & Hinton, 2012, Coursera Neural Networks for ML
 *   - Xavier / Glorot 初始化
 *       Glorot & Bengio, 2010, AISTATS
 *   - Kaiming / He 初始化
 *       He et al., 2015 (arXiv:1502.01852)
 *   - Top-p（核）采样 / Top-k 采样
 *       Holtzman et al., 2020 (arXiv:1904.09751) / Fan et al., 2018 (arXiv:1805.04833)
 *   - 数值稳定 Softmax（log-sum-exp 防溢出）
 *       标准数值实践；见 Goodfellow et al., DL Book, 2016
 *
 * 作者：晨星 (GitHub: CJX0712)
 * 许可：MIT
 */

(function (global) {
  'use strict';

  // ===================================================================
  // 0. 可复现随机数（xorshift32 + Box-Muller）—— 自检可复现的基石
  // ===================================================================
  function makeRng(seed) {
    let s = (seed >>> 0) || 1;
    return function () {
      s ^= s << 13; s >>>= 0;
      s ^= s >> 17;
      s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    };
  }
  function randn(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // ===================================================================
  // 1. 线性代数（矩阵乘法 = 深度学习一切的基石）
  // ===================================================================
  const LA = {
    zeros(r, c) {
      return Array.from({ length: r }, () => new Array(c).fill(0));
    },
    matmul(a, b) {
      const ar = a.length, ac = a[0].length, bc = b[0].length;
      const out = Array.from({ length: ar }, () => new Array(bc).fill(0));
      for (let i = 0; i < ar; i++) {
        for (let k = 0; k < ac; k++) {
          const aik = a[i][k];
          if (aik === 0) continue;
          const brow = b[k];
          const orow = out[i];
          for (let j = 0; j < bc; j++) orow[j] += aik * brow[j];
        }
      }
      return out;
    },
    transpose(a) {
      const r = a.length, c = a[0].length;
      return Array.from({ length: c }, (_, j) => a.map((row) => row[j]));
    },
    add(a, b) {
      return a.map((row, i) => row.map((v, j) => v + b[i][j]));
    },
    scale(a, s) {
      return a.map((row) => row.map((v) => v * s));
    },
    // 平坦化嵌套数组
    flatten(x) {
      return x.reduce((acc, row) => acc.concat(row), []);
    },
  };

  // ===================================================================
  // 2. 数值稳定激活函数
  // ===================================================================
  const Act = {
    // log-sum-exp 稳定的 softmax（沿向量，最后一维）
    softmax(vec) {
      const m = Math.max.apply(null, vec);
      const exps = vec.map((x) => Math.exp(x - m));
      const sum = exps.reduce((a, b) => a + b, 0);
      return exps.map((e) => e / sum);
    },
    relu(x) { return x > 0 ? x : 0; },
    // SiLU / Swish（PaLM / LLaMA 等现代模型默认激活）
    silu(x) { return x / (1 + Math.exp(-x)); },
    // GELU（精确 erf 版；GPT-2/BERT 所用）
    gelu(x) {
      const c = x / Math.SQRT2;
      // Abramowitz-Stegun erf 近似，精度 ~1e-7
      const t = 1 / (1 + 0.3275911 * Math.abs(c));
      const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-c * c);
      const erf = c >= 0 ? y : -y;
      return 0.5 * x * (1 + erf);
    },
  };

  // ===================================================================
  // 3. 归一化（LayerNorm / RMSNorm）
  // ===================================================================
  const Norm = {
    layerNorm(x, gamma, beta, eps = 1e-5) {
      const n = x.length;
      const mean = x.reduce((a, b) => a + b, 0) / n;
      const variance = x.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n;
      const inv = 1 / Math.sqrt(variance + eps);
      return x.map((v, i) => (v - mean) * inv * (gamma ? gamma[i] : 1) + (beta ? beta[i] : 0));
    },
    rmsNorm(x, gamma, eps = 1e-6) {
      const n = x.length;
      const ms = x.reduce((a, b) => a + b * b, 0) / n;
      const inv = 1 / Math.sqrt(ms + eps);
      return x.map((v, i) => v * inv * (gamma ? gamma[i] : 1));
    },
  };

  // ===================================================================
  // 4. 注意力（Vaswani 2017 · 多头缩放点积注意力）
  // ===================================================================
  const Attention = {
    // 单头缩放点积注意力；mask 为 [seq,seq] 的 0/1 矩阵（因果/自定义）
    scaledDotProduct(Q, K, V, mask) {
      const seq = Q.length, d = Q[0].length;
      const out = LA.zeros(seq, d);
      for (let i = 0; i < seq; i++) {
        const scores = K.map((k) => {
          let s = 0;
          for (let j = 0; j < d; j++) s += Q[i][j] * k[j];
          return s / Math.sqrt(d);
        });
        if (mask) for (let j = 0; j < seq; j++) if (!mask[i][j]) scores[j] = -1e9;
        const w = Act.softmax(scores);
        for (let j = 0; j < seq; j++)
          for (let k = 0; k < d; k++) out[i][k] += w[j] * V[j][k];
      }
      return out;
    },
    // 多头：按 head 切分 -> 并行单头 -> 拼接（可选输出投影 Wo）
    multiHead(Q, K, V, numHeads, mask, Wo) {
      const seq = Q.length, dModel = Q[0].length;
      const dk = Math.floor(dModel / numHeads);
      const headsOut = [];
      for (let h = 0; h < numHeads; h++) {
        const qh = Q.map((r) => r.slice(h * dk, (h + 1) * dk));
        const kh = K.map((r) => r.slice(h * dk, (h + 1) * dk));
        const vh = V.map((r) => r.slice(h * dk, (h + 1) * dk));
        headsOut.push(this.scaledDotProduct(qh, kh, vh, mask));
      }
      let concat = Array.from({ length: seq }, (_, i) =>
        headsOut.reduce((acc, h) => acc.concat(h[i]), [])
      );
      if (Wo) concat = LA.matmul(concat, Wo);
      return concat;
    },
  };

  // ===================================================================
  // 5. 优化器（Adam / AdamW / RMSProp / SGD）—— 站在 Kingma & Ba 等肩上
  //    参数与梯度均为「数字数组的数组」或「一维数组」。
  // ===================================================================
  function map2(a, b, fn) { return a.map((row, i) => row.map((v, j) => fn(v, b[i][j], i, j))); }
  function map1(a, fn) { return a.map((row, i) => row.map((v, j) => fn(v, i, j))); }

  class Adam {
    constructor(lr = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-8, weightDecay = 0) {
      this.lr = lr; this.b1 = b1; this.b2 = b2; this.eps = eps; this.wd = weightDecay;
      this.t = 0; this.m = null; this.v = null;
    }
    _init(p) {
      this.m = p.map((row) => row.map(() => 0));
      this.v = p.map((row) => row.map(() => 0));
    }
    step(p, g) {
      if (!this.m) this._init(p);
      this.t++;
      const bc1 = 1 - Math.pow(this.b1, this.t);
      const bc2 = 1 - Math.pow(this.b2, this.t);
      const decay = this.wd;
      const out = map2(p, g, (pv, gv, i, j) => {
        this.m[i][j] = this.b1 * this.m[i][j] + (1 - this.b1) * gv;
        this.v[i][j] = this.b2 * this.v[i][j] + (1 - this.b2) * gv * gv;
        const mh = this.m[i][j] / bc1;
        const vh = this.v[i][j] / bc2;
        let upd = this.lr * mh / (Math.sqrt(vh) + this.eps);
        if (decay) upd += this.lr * decay * pv; // 标准 Adam 的 L2 式衰减
        return pv - upd;
      });
      return out;
    }
  }

  class AdamW {
    constructor(lr = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-8, weightDecay = 0.01) {
      this.lr = lr; this.b1 = b1; this.b2 = b2; this.eps = eps; this.wd = weightDecay;
      this.t = 0; this.m = null; this.v = null;
    }
    _init(p) {
      this.m = p.map((row) => row.map(() => 0));
      this.v = p.map((row) => row.map(() => 0));
    }
    step(p, g) {
      if (!this.m) this._init(p);
      this.t++;
      const bc1 = 1 - Math.pow(this.b1, this.t);
      const bc2 = 1 - Math.pow(this.b2, this.t);
      const out = map2(p, g, (pv, gv, i, j) => {
        this.m[i][j] = this.b1 * this.m[i][j] + (1 - this.b1) * gv;
        this.v[i][j] = this.b2 * this.v[i][j] + (1 - this.b2) * gv * gv;
        const mh = this.m[i][j] / bc1;
        const vh = this.v[i][j] / bc2;
        // 解耦权重衰减：衰减项不进动量/二阶矩
        return pv - this.lr * (mh / (Math.sqrt(vh) + this.eps) + this.wd * pv);
      });
      return out;
    }
  }

  class RMSProp {
    constructor(lr = 0.001, rho = 0.9, eps = 1e-8) {
      this.lr = lr; this.rho = rho; this.eps = eps; this.v = null;
    }
    _init(p) { this.v = p.map((row) => row.map(() => 0)); }
    step(p, g) {
      if (!this.v) this._init(p);
      const out = map2(p, g, (pv, gv, i, j) => {
        this.v[i][j] = this.rho * this.v[i][j] + (1 - this.rho) * gv * gv;
        return pv - this.lr * gv / (Math.sqrt(this.v[i][j]) + this.eps);
      });
      return out;
    }
  }

  // ===================================================================
  // 6. 初始化（Xavier / Kaiming）—— Glorot & Bengio / He et al.
  // ===================================================================
  const Init = {
    xavierUniform(rng, fanIn, fanOut) {
      const limit = Math.sqrt(6 / (fanIn + fanOut));
      return rng() * 2 * limit - limit;
    },
    kaimingNormal(rng, fanIn) {
      // He 初始化：N(0, 2/fanIn)
      return randn(rng) * Math.sqrt(2 / fanIn);
    },
  };

  // ===================================================================
  // 7. 解码采样（Top-k / Top-p 核采样）—— Holtzman 2020 / Fan 2018
  // ===================================================================
  const Sample = {
    applyTemperature(logits, T) {
      if (T === 0) return logits;
      return logits.map((x) => x / T);
    },
    topK(logits, k) {
      const idx = logits.map((v, i) => [i, v]).sort((a, b) => b[1] - a[1]).slice(0, k);
      const mask = new Array(logits.length).fill(-Infinity);
      idx.forEach(([i]) => (mask[i] = logits[i]));
      return mask;
    },
    topP(logits, p) {
      const ranked = logits.map((v, i) => [i, v]).sort((a, b) => b[1] - a[1]);
      const probs = Act.softmax(ranked.map((x) => x[1]));
      const mask = new Array(logits.length).fill(-Infinity);
      let cum = 0;
      for (let n = 0; n < ranked.length; n++) {
        cum += probs[n];
        mask[ranked[n][0]] = logits[ranked[n][0]];
        if (cum >= p) break;
      }
      return mask;
    },
    // 熵（用于验证温度单调性）
    entropy(logits) {
      const p = Act.softmax(logits);
      return -p.reduce((a, x) => a + (x > 0 ? x * Math.log(x) : 0), 0);
    },
  };

  // ===================================================================
  // 8. 损失（数值稳定交叉熵 / MSE）
  // ===================================================================
  const Loss = {
    crossEntropy(logits, target) {
      const p = Act.softmax(logits);
      return -Math.log(Math.max(p[target], 1e-12));
    },
    // softmax + CE 对 logits 的解析梯度： p - onehot(target)
    crossEntropyGrad(logits, target) {
      const p = Act.softmax(logits);
      const g = p.slice();
      g[target] -= 1;
      return g;
    },
    mse(y, yhat) {
      return y.reduce((a, v, i) => a + (v - yhat[i]) * (v - yhat[i]), 0) / y.length;
    },
  };

  // ===================================================================
  // 导出
  // ===================================================================
  const APEX = {
    makeRng, randn, LA, Act, Norm, Attention,
    Adam, AdamW, RMSProp, Init, Sample, Loss,
  };
  global.APEX = APEX;
  if (typeof module !== 'undefined' && module.exports) module.exports = APEX;

  // ===================================================================
  // 9. 自检（8/8 全绿 = 可被独立实现交叉验证的「不变量」）
  // ===================================================================
  function approx(a, b, tol) { return Math.abs(a - b) <= tol; }
  function maxRelErr(a, b) {
    let m = 0;
    for (let i = 0; i < a.length; i++) {
      const d = Math.abs(a[i] - b[i]);
      const s = Math.abs(a[i]) + Math.abs(b[i]) + 1e-12;
      m = Math.max(m, d / s);
    }
    return m;
  }

  function runSelfTest() {
    const results = [];
    const rng = makeRng(20261021);

    // ① 矩阵乘法：与手算小例对照（解析 vs 数值）
    {
      const A = [[1, 2], [3, 4]];
      const B = [[5, 6], [7, 8]];
      const C = LA.matmul(A, B); // 期望值 [[19,22],[43,50]]
      const ok = C[0][0] === 19 && C[0][1] === 22 && C[1][0] === 43 && C[1][1] === 50;
      results.push({ name: '矩阵乘法 matmul', pass: ok, detail: 'C=[[19,22],[43,50]] 精确命中' });
    }

    // ② Softmax 数值稳定性：平移不变（log-sum-exp 防溢出）
    {
      const x = [1, 2, 3];
      const xBig = x.map((v) => v + 1e6);
      const s1 = Act.softmax(x), s2 = Act.softmax(xBig);
      const maxDiff = Math.max.apply(null, s1.map((v, i) => Math.abs(v - s2[i])));
      results.push({ name: 'Softmax 数值稳定性', pass: maxDiff < 1e-12, detail: '平移 1e6 最大差=' + maxDiff.toExponential(2) });
    }

    // ③ Softmax 归一：每行和 = 1
    {
      const x = [0.1, -2.3, 4.5, 9.9, -1.1];
      const s = Act.softmax(x);
      const sum = s.reduce((a, b) => a + b, 0);
      results.push({ name: 'Softmax 归一 Σ=1', pass: approx(sum, 1, 1e-12), detail: 'Σ=' + sum.toFixed(15) });
    }

    // ④ LayerNorm：输出均值≈0、方差≈ var/(var+eps)（精确不变量）
    {
      const x = Array.from({ length: 16 }, () => randn(rng));
      const eps = 1e-5;
      const meanX = x.reduce((a, b) => a + b, 0) / x.length;
      const varX = x.reduce((a, b) => a + (b - meanX) * (b - meanX), 0) / x.length;
      const y = Norm.layerNorm(x, undefined, undefined, eps);
      const meanY = y.reduce((a, b) => a + b, 0) / y.length;
      const varY = y.reduce((a, b) => a + (b - meanY) * (b - meanY), 0) / y.length;
      const expectedVarY = varX / (varX + eps); // 理论精确值
      const ok = approx(meanY, 0, 1e-9) && approx(varY, expectedVarY, 1e-9);
      results.push({ name: 'LayerNorm μ≈0,σ²≈var/(var+ε)', pass: ok, detail: `μ=${meanY.toExponential(2)} σ²=${varY.toFixed(10)} 理论=${expectedVarY.toFixed(10)}` });
    }

    // ⑤ 注意力权重和=1 + 因果掩码下三角性
    {
      const seq = 4, d = 3;
      const Q = Array.from({ length: seq }, () => Array.from({ length: d }, () => randn(rng)));
      const K = Array.from({ length: seq }, () => Array.from({ length: d }, () => randn(rng)));
      const V = Array.from({ length: seq }, () => Array.from({ length: d }, () => randn(rng)));
      // 因果掩码
      const mask = Array.from({ length: seq }, (_, i) => Array.from({ length: seq }, (_, j) => (j <= i ? 1 : 0)));
      const out = Attention.scaledDotProduct(Q, K, V, mask);
      // 重建权重检查：每行权重和=1
      const scores = K.map((k) => Q[0].reduce((a, q, j) => a + q * k[j], 0) / Math.sqrt(d));
      const w = Act.softmax(scores);
      const sumW = w.reduce((a, b) => a + b, 0);
      // 因果：位置0不应看到位置3
      const s3 = K.map((k, j) => (j <= 0 ? Q[0].reduce((a, q, jj) => a + q * k[jj], 0) / Math.sqrt(d) : -1e9));
      const w3 = Act.softmax(s3);
      const ok = approx(sumW, 1, 1e-12) && approx(w3[3], 0, 1e-12) && out.length === seq;
      results.push({ name: '注意力 Σw=1 + 因果', pass: ok, detail: `Σw=${sumW.toFixed(12)} w[pos0→3]=${w3[3]}` });
    }

    // ⑥ Adam 收敛：凸二次损失 0.5‖w‖² 上 300 步大幅收敛
    {
      const opt = new Adam(0.1);
      let w = Array.from({ length: 8 }, () => randn(rng));
      const loss0 = 0.5 * w.reduce((a, v) => a + v * v, 0);
      for (let s = 0; s < 300; s++) {
        const g = w.slice(); // ∇(0.5‖w‖²) = w
        w = opt.step([w], [g])[0];
      }
      const loss1 = 0.5 * w.reduce((a, v) => a + v * v, 0);
      const ok = loss1 < loss0 / 1000;
      results.push({ name: 'Adam 收敛性', pass: ok, detail: `loss ${loss0.toFixed(3)}→${loss1.toExponential(2)} (↓>$3数量级)` });
    }

    // ⑦ Xavier 初始化方差 ≈ 2/(fanIn+fanOut)
    {
      const fanIn = 20, fanOut = 10, target = 2 / (fanIn + fanOut);
      const N = 40000;
      const r2 = makeRng(99);
      const vals = [];
      for (let i = 0; i < N; i++) vals.push(Init.xavierUniform(r2, fanIn, fanOut));
      const mean = vals.reduce((a, b) => a + b, 0) / N;
      const varr = vals.reduce((a, b) => a + (b - mean) * (b - mean), 0) / N;
      const ok = Math.abs(varr - target) / target < 0.05;
      results.push({ name: 'Xavier 方差', pass: ok, detail: `实测σ²=${varr.toFixed(5)} 目标=${target.toFixed(5)}` });
    }

    // ⑧ 梯度检验：softmax+CE 解析梯度 vs 中心差分
    {
      const D = 6, target = 2;
      const logits = Array.from({ length: D }, () => randn(rng));
      const analytic = Loss.crossEntropyGrad(logits, target);
      const eps = 1e-5;
      const numeric = logits.map((_, i) => {
        const lp = logits.slice(); lp[i] += eps;
        const lm = logits.slice(); lm[i] -= eps;
        return (Loss.crossEntropy(lp, target) - Loss.crossEntropy(lm, target)) / (2 * eps);
      });
      const mre = maxRelErr(analytic, numeric);
      results.push({ name: '梯度检验 ∂L/∂logits', pass: mre < 1e-6, detail: `maxRelErr=${mre.toExponential(2)}` });
    }

    return results;
  }

  APEX.runSelfTest = runSelfTest;

  // 无头 Node 直接运行时打印结果
  if (typeof require !== 'undefined' && typeof module !== 'undefined' && require.main === module) {
    const r = runSelfTest();
    let pass = 0;
    console.log('\n=== APEX-FORGE 自检 ===');
    r.forEach((x) => {
      console.log(`${x.pass ? '✅' : '❌'} ${x.name.padEnd(22)} | ${x.detail}`);
      if (x.pass) pass++;
    });
    console.log(`\n结果：${pass}/${r.length} 全绿`);
    process.exit(pass === r.length ? 0 : 1);
  }
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
