(() => {
  "use strict";

  const $ = id => document.getElementById(id);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const sigmoid = z => 1 / (1 + Math.exp(-clamp(z, -30, 30)));
  const fmt = (v, d = 2) => Number(v).toFixed(d);
  const pct = v => `${fmt(v * 100, 1)}%`;

  class RNG {
    constructor(seed = 20260609) { this.s = seed >>> 0; }
    next() { this.s = (1664525 * this.s + 1013904223) >>> 0; return this.s / 4294967296; }
    normal() {
      const u = Math.max(this.next(), 1e-9), v = this.next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
  }

  const MODEL_NAMES = {
    knn: "k-Nearest Neighbors", tree: "Decision Tree", logistic: "Logistic Regression",
    svm: "Linear SVM", rbfsvm: "SVM with RBF Kernel", mlp: "Multilayer Neural Network",
    naivebayes: "Gaussian Naive Bayes",
    lda: "Linear Discriminant Analysis", perceptron: "Perceptron"
  };
  const DATASET_NAMES = {
    students: "Students", linear: "Linear separation", xor: "XOR", circles: "Circles",
    moons: "Two moons", spiral: "Spiral", stripes: "Horizontal stripes", outlier: "Outlier dataset",
    noisy: "Noisy classification", imbalanced: "Imbalanced dataset"
  };
  const EXPLAINS = {
    knn: "Classifies a point by looking at the labels of its nearest training examples. It learns no explicit formula.",
    tree: "Recursively splits one feature at a time, choosing thresholds that reduce Gini impurity.",
    logistic: "Learns a linear probability boundary. The sigmoid converts a weighted feature score into a Class 1 probability.",
    svm: "Finds a linear boundary with the widest possible margin between classes. C controls the penalty for margin violations.",
    rbfsvm: "Uses radial basis kernels to compare points by local similarity, creating curved boundaries while preserving the SVM margin idea.",
    mlp: "Combines several hidden nonlinear units, then learns all connection weights with backpropagation.",
    naivebayes: "Models each feature with a Gaussian distribution inside each class and combines likelihoods using Bayes' rule.",
    lda: "Finds a linear direction that separates class means while accounting for shared within-class variance.",
    perceptron: "Updates a linear boundary whenever it misclassifies a training point. It is an early neural learning rule."
  };
  const TITLES = {
    knn: "Vote with nearby examples", tree: "Ask a sequence of questions",
    logistic: "Turn a score into probability", svm: "Maximize the separating margin",
    rbfsvm: "Build a nonlinear similarity margin", mlp: "Compose hidden nonlinear features",
    naivebayes: "Compare class likelihoods", lda: "Separate projected class means",
    perceptron: "Learn from each mistake"
  };
  const defaults = {
    knn: { k: 7, weighted: true, metric: "euclidean" },
    tree: { maxDepth: 4, minSplit: 2 },
    logistic: { learningRate: .12, epochs: 350, lambda: .05 },
    svm: { c: 1, learningRate: .02, epochs: 450 },
    rbfsvm: { c: 1.5, gamma: 1.2, learningRate: .08, epochs: 120 },
    mlp: { hidden: 8, learningRate: .08, epochs: 500, lambda: .001 },
    naivebayes: { smoothing: .15, priors: "data" },
    lda: { shrinkage: .1 },
    perceptron: { learningRate: .08, epochs: 120 }
  };

  const state = {
    dataset: "moons", modelType: "knn", params: JSON.parse(JSON.stringify(defaults)),
    points: [], originalPoints: [], model: null, trained: false, metrics: null,
    tool: "add", addClass: 1, brushRadius: 24, selected: new Set(), confusionFilter: null,
    dragging: false, dragStart: null, dragCurrent: null, hoverIndex: -1, nextId: 1,
    bounds: { xmin: -3, xmax: 3, ymin: -3, ymax: 3 }, regionImage: null
  };

  function generateDataset(type) {
    const rng = new RNG(9137 + Object.keys(DATASET_NAMES).indexOf(type) * 811), pts = [];
    const add = (x, y, label) => pts.push({ id: state.nextId++, x, y, label });
    if (type === "students") {
      for (let i = 0; i < 90; i++) {
        const x = clamp(1 + rng.next() * 8 + rng.normal() * .25, 0, 10);
        const y = clamp(3.5 + rng.next() * 5 + rng.normal() * .25, 2, 10);
        add(x, y, .75 * x + .55 * y + rng.normal() * 1.15 > 7.7 ? 1 : 0);
      }
    } else if (type === "linear") {
      for (let i = 0; i < 90; i++) { const x = rng.next() * 6 - 3, y = rng.next() * 6 - 3; add(x, y, y > .65 * x + rng.normal() * .55 ? 1 : 0); }
    } else if (type === "xor") {
      for (let i = 0; i < 100; i++) { const x = rng.next() * 6 - 3, y = rng.next() * 6 - 3; add(x, y, (x > 0) !== (y > 0) ? 1 : 0); }
    } else if (type === "circles") {
      for (let i = 0; i < 110; i++) {
        const outer = i >= 50, a = rng.next() * Math.PI * 2, r = (outer ? 2.25 : 1) + rng.normal() * .2;
        add(Math.cos(a) * r, Math.sin(a) * r, outer ? 0 : 1);
      }
    } else if (type === "moons") {
      for (let i = 0; i < 100; i++) {
        const c = i < 50 ? 0 : 1, a = rng.next() * Math.PI;
        add((c ? 1 - Math.cos(a) : Math.cos(a)) * 2 + rng.normal() * .15,
          (c ? -.55 - Math.sin(a) : Math.sin(a)) * 1.7 + rng.normal() * .15, c);
      }
    } else if (type === "spiral") {
      for (let c = 0; c < 2; c++) for (let i = 0; i < 55; i++) {
        const r = .25 + i / 22, t = i / 8 + c * Math.PI;
        add(r * Math.cos(t) + rng.normal() * .12, r * Math.sin(t) + rng.normal() * .12, c);
      }
    } else if (type === "stripes") {
      for (let i = 0; i < 110; i++) { const x = rng.next() * 6 - 3, y = rng.next() * 6 - 3; add(x, y, Math.floor((y + 3) / 1.05) % 2); }
    } else if (type === "outlier") {
      for (let i = 0; i < 82; i++) { const c = i < 41 ? 0 : 1; add((c ? 1.3 : -1.3) + rng.normal() * .65, (c ? 1 : -1) + rng.normal() * .7, c); }
      add(-2.3, -2.1, 1); add(2.35, 2.2, 0); add(-2.1, 1.8, 1); add(2.1, -1.8, 0);
    } else if (type === "noisy") {
      for (let i = 0; i < 115; i++) {
        const x = rng.next() * 6 - 3, y = rng.next() * 6 - 3;
        let label = y > .45 * x + Math.sin(x * 1.6) ? 1 : 0;
        if (rng.next() < .18) label = 1 - label;
        add(x, y, label);
      }
    } else {
      for (let i = 0; i < 105; i++) {
        const minority = i < 18;
        add((minority ? 1.45 : -.55) + rng.normal() * (minority ? .58 : 1.05),
          (minority ? 1.15 : -.35) + rng.normal() * (minority ? .58 : 1), minority ? 1 : 0);
      }
    }
    return pts;
  }

  function featureStats(points) {
    const mx = points.reduce((s, p) => s + p.x, 0) / points.length;
    const my = points.reduce((s, p) => s + p.y, 0) / points.length;
    const sx = Math.sqrt(points.reduce((s, p) => s + (p.x - mx) ** 2, 0) / points.length) || 1;
    const sy = Math.sqrt(points.reduce((s, p) => s + (p.y - my) ** 2, 0) / points.length) || 1;
    return { mx, my, sx, sy, transform: (x, y) => [(x - mx) / sx, (y - my) / sy] };
  }

  function gini(samples) {
    if (!samples.length) return 0;
    const p = samples.reduce((s, q) => s + q.label, 0) / samples.length;
    return 1 - p * p - (1 - p) * (1 - p);
  }
  function buildTree(samples, maxDepth, minSplit, depth = 0) {
    const prob = samples.reduce((s, p) => s + p.label, 0) / samples.length;
    const node = { leaf: true, prob, pred: prob >= .5 ? 1 : 0, depth };
    if (depth >= maxDepth || samples.length < minSplit || gini(samples) < 1e-8) return node;
    let best = null;
    for (let f = 0; f < 2; f++) {
      const values = [...new Set(samples.map(p => f ? p.y : p.x))].sort((a, b) => a - b);
      const stride = Math.max(1, Math.ceil(values.length / 35));
      for (let i = 1; i < values.length; i += stride) {
        const t = (values[i - 1] + values[i]) / 2;
        const left = samples.filter(p => (f ? p.y : p.x) <= t), right = samples.filter(p => (f ? p.y : p.x) > t);
        if (!left.length || !right.length) continue;
        const loss = (left.length * gini(left) + right.length * gini(right)) / samples.length;
        if (!best || loss < best.loss) best = { f, t, left, right, loss };
      }
    }
    if (!best) return node;
    return { leaf: false, feature: best.f, threshold: best.t, depth, left: buildTree(best.left, maxDepth, minSplit, depth + 1), right: buildTree(best.right, maxDepth, minSplit, depth + 1) };
  }
  function treeProb(node, x, y) {
    while (!node.leaf) node = ((node.feature ? y : x) <= node.threshold) ? node.left : node.right;
    return node.prob;
  }
  function treeDepth(node) { return node.leaf ? node.depth : Math.max(treeDepth(node.left), treeDepth(node.right)); }
  function treeLeaves(node) { return node.leaf ? 1 : treeLeaves(node.left) + treeLeaves(node.right); }

  function trainModel() {
    const points = state.points, classes = new Set(points.map(p => p.label));
    if (points.length < 4 || classes.size < 2) {
      state.model = null; state.metrics = null; state.trained = false;
      showWarning(classes.size < 2 ? "Add points from both classes to train the model." : "Add at least four points to train the model.");
      updateAll(); return;
    }
    hideWarning();
    const type = state.modelType, p = state.params[type], stats = featureStats(points);
    if (type === "knn") {
      const distance = (a, b) => p.metric === "manhattan" ? Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) : Math.hypot(a[0] - b[0], a[1] - b[1]);
      state.model = {
        type, stats, complexity: `Stores ${points.length} examples`,
        neighbors(x, y) {
          const q = stats.transform(x, y);
          return points.map((pt, i) => ({ i, point: pt, d: distance(q, stats.transform(pt.x, pt.y)) })).sort((a, b) => a.d - b.d).slice(0, Math.min(p.k, points.length));
        },
        predictProba(x, y) {
          const ns = this.neighbors(x, y);
          let yes = 0, total = 0;
          ns.forEach(n => { const w = p.weighted ? 1 / (n.d + .08) : 1; yes += w * n.point.label; total += w; });
          return yes / total;
        }
      };
    } else if (type === "tree") {
      const root = buildTree(points, p.maxDepth, p.minSplit);
      state.model = { type, root, complexity: `${treeLeaves(root)} leaves · depth ${treeDepth(root)}`, predictProba: (x, y) => treeProb(root, x, y) };
    } else if (type === "logistic") {
      let w = [0, 0], b = 0, losses = [];
      for (let e = 0; e < p.epochs; e++) {
        let gw0 = 0, gw1 = 0, gb = 0, loss = 0;
        points.forEach(pt => {
          const z = stats.transform(pt.x, pt.y), pr = sigmoid(w[0] * z[0] + w[1] * z[1] + b), err = pr - pt.label;
          gw0 += err * z[0]; gw1 += err * z[1]; gb += err;
          loss -= pt.label * Math.log(clamp(pr, 1e-7, 1)) + (1 - pt.label) * Math.log(clamp(1 - pr, 1e-7, 1));
        });
        w[0] -= p.learningRate * (gw0 / points.length + p.lambda * w[0]);
        w[1] -= p.learningRate * (gw1 / points.length + p.lambda * w[1]);
        b -= p.learningRate * gb / points.length;
        if (e % Math.max(1, Math.floor(p.epochs / 60)) === 0) losses.push(loss / points.length);
      }
      state.model = { type, w, b, stats, losses, complexity: "3 learned coefficients", predictProba(x, y) { const z = stats.transform(x, y); return sigmoid(w[0] * z[0] + w[1] * z[1] + b); } };
    } else if (type === "svm") {
      let w = [0, 0], b = 0, losses = [];
      for (let e = 0; e < p.epochs; e++) {
        let hinge = 0;
        points.forEach(pt => {
          const z = stats.transform(pt.x, pt.y), y = pt.label ? 1 : -1, margin = y * (w[0] * z[0] + w[1] * z[1] + b);
          w[0] -= p.learningRate * w[0] / points.length; w[1] -= p.learningRate * w[1] / points.length;
          if (margin < 1) { w[0] += p.learningRate * p.c * y * z[0]; w[1] += p.learningRate * p.c * y * z[1]; b += p.learningRate * p.c * y; hinge += 1 - margin; }
        });
        if (e % Math.max(1, Math.floor(p.epochs / 60)) === 0) losses.push(.5 * (w[0] ** 2 + w[1] ** 2) + p.c * hinge / points.length);
      }
      state.model = { type, w, b, stats, losses, complexity: "Linear margin boundary", score(x, y) { const z = stats.transform(x, y); return w[0] * z[0] + w[1] * z[1] + b; }, predictProba(x, y) { return sigmoid(this.score(x, y) * 1.7); } };
    } else if (type === "rbfsvm") {
      const z = points.map(pt => stats.transform(pt.x, pt.y));
      const labels = points.map(pt => pt.label ? 1 : -1);
      const alpha = Array(points.length).fill(0);
      const kernel = (a, b) => Math.exp(-p.gamma * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2));
      const gram = z.map(a => z.map(b => kernel(a, b)));
      let b = 0, losses = [];
      for (let e = 0; e < p.epochs; e++) {
        let violations = 0;
        for (let i = 0; i < points.length; i++) {
          let score = b;
          for (let j = 0; j < points.length; j++) if (alpha[j]) score += alpha[j] * labels[j] * gram[j][i];
          const margin = labels[i] * score;
          if (margin < 1) {
            alpha[i] = Math.min(p.c, alpha[i] + p.learningRate);
            b += p.learningRate * labels[i] * .08;
            violations++;
          } else {
            alpha[i] *= Math.max(0, 1 - p.learningRate * .002);
          }
        }
        if (e % Math.max(1, Math.floor(p.epochs / 60)) === 0) losses.push(violations / points.length);
      }
      const support = alpha.map((a, i) => ({ a, i })).filter(q => q.a > .001);
      state.model = {
        type, alpha, labels, z, b, stats, gamma: p.gamma, support, losses,
        complexity: `${support.length} support vectors`,
        score(x, y) {
          const q = stats.transform(x, y);
          return b + support.reduce((sum, s) => sum + s.a * labels[s.i] * kernel(z[s.i], q), 0);
        },
        predictProba(x, y) { return sigmoid(this.score(x, y) * 1.8); }
      };
    } else if (type === "mlp") {
      const rng = new RNG(77831), h = p.hidden;
      const w1 = Array.from({ length: h }, () => [(rng.next() - .5) * 1.2, (rng.next() - .5) * 1.2]);
      const b1 = Array(h).fill(0), w2 = Array.from({ length: h }, () => (rng.next() - .5) * 1.2);
      let b2 = 0, losses = [];
      for (let e = 0; e < p.epochs; e++) {
        let loss = 0;
        for (const pt of points) {
          const input = stats.transform(pt.x, pt.y);
          const hidden = w1.map((w, j) => Math.tanh(w[0] * input[0] + w[1] * input[1] + b1[j]));
          const out = sigmoid(hidden.reduce((s, value, j) => s + value * w2[j], b2));
          const deltaOut = out - pt.label;
          const oldW2 = w2.slice();
          for (let j = 0; j < h; j++) {
            w2[j] -= p.learningRate * (deltaOut * hidden[j] + p.lambda * w2[j]);
            const deltaHidden = deltaOut * oldW2[j] * (1 - hidden[j] ** 2);
            w1[j][0] -= p.learningRate * (deltaHidden * input[0] + p.lambda * w1[j][0]);
            w1[j][1] -= p.learningRate * (deltaHidden * input[1] + p.lambda * w1[j][1]);
            b1[j] -= p.learningRate * deltaHidden;
          }
          b2 -= p.learningRate * deltaOut;
          loss -= pt.label * Math.log(clamp(out, 1e-7, 1)) + (1 - pt.label) * Math.log(clamp(1 - out, 1e-7, 1));
        }
        if (e % Math.max(1, Math.floor(p.epochs / 70)) === 0) losses.push(loss / points.length);
      }
      state.model = {
        type, w1, b1, w2, b2, stats, losses,
        complexity: `2 → ${h} → 1 network`,
        activations(x, y) {
          const input = stats.transform(x, y);
          return w1.map((w, j) => Math.tanh(w[0] * input[0] + w[1] * input[1] + b1[j]));
        },
        predictProba(x, y) {
          const hidden = this.activations(x, y);
          return sigmoid(hidden.reduce((s, value, j) => s + value * w2[j], b2));
        }
      };
    } else if (type === "naivebayes") {
      const byClass = [points.filter(q => !q.label), points.filter(q => q.label)];
      const prior1 = p.priors === "equal" ? .5 : byClass[1].length / points.length;
      const params = byClass.map(group => {
        const mx = group.reduce((s, q) => s + q.x, 0) / group.length, my = group.reduce((s, q) => s + q.y, 0) / group.length;
        return { mx, my, vx: group.reduce((s, q) => s + (q.x - mx) ** 2, 0) / group.length + p.smoothing, vy: group.reduce((s, q) => s + (q.y - my) ** 2, 0) / group.length + p.smoothing };
      });
      const logLike = (x, y, c) => {
        const q = params[c];
        return -.5 * (Math.log(2 * Math.PI * q.vx) + (x - q.mx) ** 2 / q.vx + Math.log(2 * Math.PI * q.vy) + (y - q.my) ** 2 / q.vy);
      };
      state.model = { type, params, prior1, complexity: "2 Gaussian features per class", predictProba(x, y) { const a = logLike(x, y, 0) + Math.log(1 - prior1), b = logLike(x, y, 1) + Math.log(prior1); return sigmoid(b - a); } };
    } else if (type === "lda") {
      const c0 = points.filter(q => !q.label), c1 = points.filter(q => q.label);
      const mean = g => [g.reduce((s, q) => s + q.x, 0) / g.length, g.reduce((s, q) => s + q.y, 0) / g.length];
      const m0 = mean(c0), m1 = mean(c1);
      let a = 0, b = 0, d = 0;
      [c0, c1].forEach((g, c) => g.forEach(q => { const m = c ? m1 : m0; a += (q.x - m[0]) ** 2; b += (q.x - m[0]) * (q.y - m[1]); d += (q.y - m[1]) ** 2; }));
      a = a / points.length + p.shrinkage; b /= points.length; d = d / points.length + p.shrinkage;
      const det = a * d - b * b || 1e-6, inv = [d / det, -b / det, a / det];
      const w = [inv[0] * (m1[0] - m0[0]) + inv[1] * (m1[1] - m0[1]), inv[1] * (m1[0] - m0[0]) + inv[2] * (m1[1] - m0[1])];
      const prior = c1.length / points.length, bias = -.5 * (w[0] * (m1[0] + m0[0]) + w[1] * (m1[1] + m0[1])) + Math.log(prior / (1 - prior));
      state.model = { type, w, b: bias, means: [m0, m1], complexity: "Shared covariance boundary", predictProba(x, y) { return sigmoid(w[0] * x + w[1] * y + bias); } };
    } else {
      let w = [0, 0], b = 0, mistakes = [];
      for (let e = 0; e < p.epochs; e++) {
        let errors = 0;
        points.forEach(pt => {
          const z = stats.transform(pt.x, pt.y), y = pt.label ? 1 : -1, pred = w[0] * z[0] + w[1] * z[1] + b >= 0 ? 1 : -1;
          if (pred !== y) { w[0] += p.learningRate * y * z[0]; w[1] += p.learningRate * y * z[1]; b += p.learningRate * y; errors++; }
        });
        mistakes.push(errors);
        if (!errors) break;
      }
      state.model = { type, w, b, stats, losses: mistakes, complexity: `${mistakes.length} learning passes`, score(x, y) { const z = stats.transform(x, y); return w[0] * z[0] + w[1] * z[1] + b; }, predictProba(x, y) { return sigmoid(this.score(x, y) * 2); } };
    }
    state.trained = true; computeMetrics(); buildRegionImage(); updateAll();
  }

  function predict(x, y) {
    if (!state.model) return { label: null, proba: .5 };
    const proba = clamp(state.model.predictProba(x, y), 0, 1);
    return { label: proba >= .5 ? 1 : 0, proba };
  }
  function computeMetrics() {
    let tp = 0, tn = 0, fp = 0, fn = 0;
    const predictions = state.points.map(p => {
      const pr = predict(p.x, p.y);
      if (p.label && pr.label) tp++; else if (!p.label && !pr.label) tn++; else if (!p.label) fp++; else fn++;
      return pr;
    });
    const precision = tp / (tp + fp || 1), recall = tp / (tp + fn || 1), n = tp + tn + fp + fn;
    state.metrics = { tp, tn, fp, fn, predictions, accuracy: (tp + tn) / n, precision, recall, f1: 2 * precision * recall / (precision + recall || 1) };
  }
  function calculateBounds() {
    if (!state.points.length) return { xmin: -3, xmax: 3, ymin: -3, ymax: 3 };
    const xs = state.points.map(p => p.x), ys = state.points.map(p => p.y);
    let xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
    const xp = Math.max(.65, (xmax - xmin) * .16), yp = Math.max(.65, (ymax - ymin) * .16);
    return { xmin: xmin - xp, xmax: xmax + xp, ymin: ymin - yp, ymax: ymax + yp };
  }

  const canvas = $("plotCanvas"), ctx = canvas.getContext("2d"), chart = $("miniChart"), chartCtx = chart.getContext("2d");
  const PAD = { l: 53, r: 18, t: 18, b: 42 };
  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1, r = canvas.getBoundingClientRect(), w = Math.max(320, Math.floor(r.width)), h = Math.max(300, Math.floor(r.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); buildRegionImage(); }
    drawPlot(); drawChart();
  }
  function toCanvas(x, y) {
    const w = canvas.clientWidth, h = canvas.clientHeight, b = state.bounds;
    return { x: PAD.l + (x - b.xmin) / (b.xmax - b.xmin) * (w - PAD.l - PAD.r), y: PAD.t + (b.ymax - y) / (b.ymax - b.ymin) * (h - PAD.t - PAD.b) };
  }
  function toData(x, y) {
    const w = canvas.clientWidth, h = canvas.clientHeight, b = state.bounds;
    return { x: b.xmin + (x - PAD.l) / (w - PAD.l - PAD.r) * (b.xmax - b.xmin), y: b.ymax - (y - PAD.t) / (h - PAD.t - PAD.b) * (b.ymax - b.ymin) };
  }
  function buildRegionImage() {
    state.bounds = calculateBounds();
    if (!state.trained || !state.model || !canvas.clientWidth) { state.regionImage = null; return; }
    const w = Math.max(100, Math.floor((canvas.clientWidth - PAD.l - PAD.r) * .32)), h = Math.max(100, Math.floor((canvas.clientHeight - PAD.t - PAD.b) * .32));
    const off = document.createElement("canvas"); off.width = w; off.height = h;
    const oc = off.getContext("2d"), img = oc.createImageData(w, h);
    for (let py = 0; py < h; py++) for (let px = 0; px < w; px++) {
      const x = state.bounds.xmin + px / (w - 1) * (state.bounds.xmax - state.bounds.xmin);
      const y = state.bounds.ymax - py / (h - 1) * (state.bounds.ymax - state.bounds.ymin);
      const pr = predict(x, y).proba, base = pr >= .5 ? [49, 87, 213] : [231, 76, 91], i = (py * w + px) * 4;
      img.data[i] = base[0]; img.data[i + 1] = base[1]; img.data[i + 2] = base[2]; img.data[i + 3] = 45 + Math.abs(pr - .5) * 90;
    }
    oc.putImageData(img, 0, 0); state.regionImage = off;
  }
  function niceStep(v) { const p = 10 ** Math.floor(Math.log10(v)), n = v / p; return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * p; }
  function drawPlot() {
    if (!canvas.clientWidth) return;
    const w = canvas.clientWidth, h = canvas.clientHeight, b = state.bounds;
    ctx.clearRect(0, 0, w, h); ctx.fillStyle = "#fbfcfe"; ctx.fillRect(0, 0, w, h);
    if (state.regionImage) ctx.drawImage(state.regionImage, PAD.l, PAD.t, w - PAD.l - PAD.r, h - PAD.t - PAD.b);
    ctx.strokeStyle = "rgba(106,119,145,.13)"; ctx.fillStyle = "#8993a6"; ctx.font = "10px DM Sans";
    const xs = niceStep((b.xmax - b.xmin) / 7), ys = niceStep((b.ymax - b.ymin) / 6);
    for (let x = Math.ceil(b.xmin / xs) * xs; x <= b.xmax; x += xs) { const q = toCanvas(x, 0); ctx.beginPath(); ctx.moveTo(q.x, PAD.t); ctx.lineTo(q.x, h - PAD.b); ctx.stroke(); ctx.textAlign = "center"; ctx.fillText(Math.abs(x) < 1e-8 ? "0" : fmt(x, 1), q.x, h - 20); }
    for (let y = Math.ceil(b.ymin / ys) * ys; y <= b.ymax; y += ys) { const q = toCanvas(0, y); ctx.beginPath(); ctx.moveTo(PAD.l, q.y); ctx.lineTo(w - PAD.r, q.y); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(Math.abs(y) < 1e-8 ? "0" : fmt(y, 1), PAD.l - 8, q.y + 3); }
    drawModelOverlay();
    state.points.forEach((p, i) => drawPoint(p, i));
    if (state.dragging && state.tool === "select" && state.dragStart && state.dragCurrent) {
      const x = Math.min(state.dragStart.x, state.dragCurrent.x), y = Math.min(state.dragStart.y, state.dragCurrent.y);
      const rw = Math.abs(state.dragStart.x - state.dragCurrent.x), rh = Math.abs(state.dragStart.y - state.dragCurrent.y);
      ctx.fillStyle = "rgba(49,87,213,.1)"; ctx.fillRect(x, y, rw, rh); ctx.strokeStyle = "#3157d5"; ctx.setLineDash([5, 4]); ctx.strokeRect(x, y, rw, rh); ctx.setLineDash([]);
    }
  }
  function drawModelOverlay() {
    if (!state.trained || !state.model) return;
    if (state.modelType === "knn" && state.hoverIndex >= 0) {
      const p = state.points[state.hoverIndex], q = toCanvas(p.x, p.y), ns = state.model.neighbors(p.x, p.y);
      ctx.strokeStyle = "rgba(23,32,51,.3)"; ctx.lineWidth = 1;
      ns.forEach(n => { const a = toCanvas(n.point.x, n.point.y); ctx.beginPath(); ctx.moveTo(q.x, q.y); ctx.lineTo(a.x, a.y); ctx.stroke(); });
    }
    if (state.modelType === "svm") drawLinearContour(0, "#172033", 2.2), drawLinearContour(1, "rgba(23,32,51,.45)", 1), drawLinearContour(-1, "rgba(23,32,51,.45)", 1);
    else if (["logistic", "lda", "perceptron"].includes(state.modelType)) drawProbabilityContour(.5, "#172033", 2);
  }
  function drawProbabilityContour(level, color, width) {
    const b = state.bounds, steps = 120; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
    let prev = null;
    for (let ix = 0; ix <= steps; ix++) {
      const x = b.xmin + ix / steps * (b.xmax - b.xmin);
      let bestY = b.ymin, bestD = Infinity;
      for (let iy = 0; iy <= 80; iy++) { const y = b.ymin + iy / 80 * (b.ymax - b.ymin), d = Math.abs(state.model.predictProba(x, y) - level); if (d < bestD) { bestD = d; bestY = y; } }
      const q = toCanvas(x, bestY); if (!prev) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y); prev = q;
    }
    ctx.stroke();
  }
  function drawLinearContour(level, color, width) {
    const m = state.model, b = state.bounds;
    if (!m.score) return;
    const findY = x => {
      let lo = b.ymin, hi = b.ymax;
      for (let i = 0; i < 35; i++) { const mid = (lo + hi) / 2; if (m.score(x, mid) < level) lo = mid; else hi = mid; }
      return (lo + hi) / 2;
    };
    const a = toCanvas(b.xmin, findY(b.xmin)), z = toCanvas(b.xmax, findY(b.xmax));
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(z.x, z.y); ctx.stroke();
  }
  function drawPoint(p, i) {
    let alpha = 1;
    if (state.confusionFilter && state.metrics) {
      const pr = state.metrics.predictions[i].label, key = p.label ? (pr ? "tp" : "fn") : (pr ? "fp" : "tn");
      if (key !== state.confusionFilter) alpha = .12;
    }
    const q = toCanvas(p.x, p.y); ctx.globalAlpha = alpha; ctx.beginPath(); ctx.arc(q.x, q.y, 6, 0, Math.PI * 2); ctx.fillStyle = p.label ? "#3157d5" : "#e74c5b"; ctx.fill();
    ctx.strokeStyle = state.selected.has(p.id) ? "#111827" : "#fff"; ctx.lineWidth = state.selected.has(p.id) ? 3 : 1.8; ctx.stroke();
    if (state.modelType === "rbfsvm" && state.model?.alpha?.[i] > .001) {
      ctx.beginPath(); ctx.arc(q.x, q.y, 10, 0, Math.PI * 2); ctx.strokeStyle = "#172033"; ctx.lineWidth = 1.5; ctx.stroke();
    }
    if (state.metrics && state.metrics.predictions[i].label !== p.label) { ctx.strokeStyle = "#182033"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(q.x - 4, q.y - 4); ctx.lineTo(q.x + 4, q.y + 4); ctx.moveTo(q.x + 4, q.y - 4); ctx.lineTo(q.x - 4, q.y + 4); ctx.stroke(); }
    if (i === state.hoverIndex) { ctx.beginPath(); ctx.arc(q.x, q.y, 11, 0, Math.PI * 2); ctx.strokeStyle = "#172033"; ctx.lineWidth = 1.4; ctx.stroke(); }
    ctx.globalAlpha = 1;
  }

  function eventPos(e) { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
  function inside(p) { return p.x >= PAD.l && p.x <= canvas.clientWidth - PAD.r && p.y >= PAD.t && p.y <= canvas.clientHeight - PAD.b; }
  function nearest(pos, max = 13) { let best = -1, dist = max; state.points.forEach((p, i) => { const q = toCanvas(p.x, p.y), d = Math.hypot(q.x - pos.x, q.y - pos.y); if (d < dist) { best = i; dist = d; } }); return best; }
  function invalidate() { state.model = null; state.metrics = null; state.trained = false; state.regionImage = null; state.confusionFilter = null; updateAll(); }
  function addAt(pos, label) { if (!inside(pos)) return; const q = toData(pos.x, pos.y); state.points.push({ id: state.nextId++, x: q.x, y: q.y, label }); invalidate(); }
  function brushAt(pos, label) {
    if (!inside(pos)) return; const rng = new RNG(Math.floor(pos.x * 31 + pos.y * 17 + state.points.length * 101));
    for (let i = 0; i < 3; i++) { const a = rng.next() * Math.PI * 2, r = Math.sqrt(rng.next()) * state.brushRadius, p = { x: pos.x + Math.cos(a) * r, y: pos.y + Math.sin(a) * r }; if (inside(p)) { const q = toData(p.x, p.y); state.points.push({ id: state.nextId++, x: q.x, y: q.y, label }); } }
    invalidate();
  }
  function eraseAt(pos) { state.points = state.points.filter(p => { const q = toCanvas(p.x, p.y); return Math.hypot(q.x - pos.x, q.y - pos.y) > state.brushRadius; }); invalidate(); }
  canvas.addEventListener("pointerdown", e => {
    const p = eventPos(e); if (!inside(p)) return; state.dragging = true; canvas.setPointerCapture(e.pointerId);
    if (state.tool === "select") { state.dragStart = p; state.dragCurrent = p; } else if (state.tool === "brush0") brushAt(p, 0); else if (state.tool === "brush1") brushAt(p, 1); else if (state.tool === "erase") eraseAt(p);
  });
  canvas.addEventListener("pointermove", e => {
    const p = eventPos(e);
    if (state.dragging) {
      if (state.tool === "select") { state.dragCurrent = p; drawPlot(); } else if (state.tool === "brush0") brushAt(p, 0); else if (state.tool === "brush1") brushAt(p, 1); else if (state.tool === "erase") eraseAt(p);
      return;
    }
    const i = nearest(p, 14); state.hoverIndex = i; if (i >= 0) showHover(i, p); else $("hoverCard").classList.add("hidden"); drawPlot();
  });
  canvas.addEventListener("pointerup", e => {
    const p = eventPos(e);
    if (state.tool === "add") { const i = nearest(p, 11); if (i >= 0) { state.points[i].label = 1 - state.points[i].label; invalidate(); } else addAt(p, state.addClass); }
    else if (state.tool === "select" && state.dragStart) {
      const x0 = Math.min(state.dragStart.x, p.x), x1 = Math.max(state.dragStart.x, p.x), y0 = Math.min(state.dragStart.y, p.y), y1 = Math.max(state.dragStart.y, p.y);
      state.selected.clear(); state.points.forEach(q => { const c = toCanvas(q.x, q.y); if (c.x >= x0 && c.x <= x1 && c.y >= y0 && c.y <= y1) state.selected.add(q.id); });
      state.dragStart = state.dragCurrent = null; updateTable();
    }
    state.dragging = false; drawPlot();
  });
  canvas.addEventListener("pointerleave", () => { if (!state.dragging) { state.hoverIndex = -1; $("hoverCard").classList.add("hidden"); drawPlot(); } });
  function showHover(i, pos) {
    const p = state.points[i], box = $("hoverCard"), pr = state.trained ? predict(p.x, p.y) : null;
    let html = `<b>Point ${i + 1}</b><br>Actual class: ${p.label}`;
    if (pr) {
      html += `<br>Prediction: Class ${pr.label}<br>Class 1 probability: ${pct(pr.proba)}`;
      if (state.modelType === "knn") { const ns = state.model.neighbors(p.x, p.y), blue = ns.filter(n => n.point.label).length; html += `<br><b>Neighbors:</b> ${blue} blue · ${ns.length - blue} red`; }
      updateReadout(p);
    }
    box.innerHTML = html; box.classList.remove("hidden"); box.style.left = `${clamp(pos.x + 14, 4, canvas.clientWidth - 210)}px`; box.style.top = `${clamp(pos.y - 18, 4, canvas.clientHeight - 105)}px`;
  }

  function range(key, label, min, max, step, value) { return `<label class="range-row">${label}<b data-value="${key}">${value}</b><input data-param="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${value}"></label>`; }
  function select(key, label, value, options) { return `<label class="field-label spaced">${label}<select data-param="${key}">${options.map(([v, t]) => `<option value="${v}" ${value === v ? "selected" : ""}>${t}</option>`).join("")}</select></label>`; }
  function toggle(key, label, checked) { return `<label class="toggle-row"><span>${label}</span><input data-param="${key}" type="checkbox" ${checked ? "checked" : ""}></label>`; }
  function parameterSchema() {
    const p = state.params[state.modelType];
    if (state.modelType === "knn") return range("k", "Number of neighbors (k)", 1, 25, 2, p.k) + select("metric", "Distance metric", p.metric, [["euclidean", "Euclidean"], ["manhattan", "Manhattan"]]) + toggle("weighted", "Weight closer neighbors", p.weighted);
    if (state.modelType === "tree") return range("maxDepth", "Max depth", 1, 10, 1, p.maxDepth) + range("minSplit", "Min samples split", 2, 12, 1, p.minSplit);
    if (state.modelType === "logistic") return range("learningRate", "Learning rate", .01, .5, .01, p.learningRate) + range("epochs", "Training epochs", 50, 800, 25, p.epochs) + range("lambda", "L2 regularization", 0, 2, .05, p.lambda);
    if (state.modelType === "svm") return range("c", "Penalty C", .1, 10, .1, p.c) + range("learningRate", "Learning rate", .005, .1, .005, p.learningRate) + range("epochs", "Training epochs", 50, 800, 25, p.epochs);
    if (state.modelType === "rbfsvm") return range("c", "Penalty C", .1, 10, .1, p.c) + range("gamma", "RBF gamma", .05, 5, .05, p.gamma) + range("learningRate", "Dual step size", .01, .25, .01, p.learningRate) + range("epochs", "Training epochs", 20, 300, 10, p.epochs);
    if (state.modelType === "mlp") return range("hidden", "Hidden neurons", 2, 24, 1, p.hidden) + range("learningRate", "Learning rate", .005, .3, .005, p.learningRate) + range("epochs", "Training epochs", 50, 1200, 25, p.epochs) + range("lambda", "L2 regularization", 0, .1, .001, p.lambda);
    if (state.modelType === "naivebayes") return range("smoothing", "Variance smoothing", .01, 2, .01, p.smoothing) + select("priors", "Class priors", p.priors, [["data", "Learn from data"], ["equal", "Equal priors"]]);
    if (state.modelType === "lda") return range("shrinkage", "Covariance shrinkage", .01, 2, .01, p.shrinkage);
    return range("learningRate", "Learning rate", .01, .5, .01, p.learningRate) + range("epochs", "Maximum epochs", 10, 400, 10, p.epochs);
  }
  function renderParameters() {
    $("parameterFields").innerHTML = parameterSchema();
    $("parameterFields").querySelectorAll("[data-param]").forEach(el => el.addEventListener("input", () => {
      const key = el.dataset.param;
      state.params[state.modelType][key] = el.type === "checkbox" ? el.checked : el.tagName === "SELECT" ? el.value : Number(el.value);
      const out = document.querySelector(`[data-value="${key}"]`); if (out) out.textContent = el.value;
      updateParamSummary(); clearTimeout(renderParameters.timer); renderParameters.timer = setTimeout(trainModel, 160);
    }));
  }
  function updateParamSummary() {
    const p = state.params[state.modelType];
    const text = {
      knn: `k = ${p.k} · ${p.weighted ? "weighted" : "uniform"}`, tree: `depth ${p.maxDepth}`,
      logistic: `λ ${p.lambda} · ${p.epochs} epochs`, svm: `C ${p.c} · ${p.epochs} epochs`,
      rbfsvm: `C ${p.c} · γ ${p.gamma}`, mlp: `${p.hidden} hidden · ${p.epochs} epochs`,
      naivebayes: `smoothing ${p.smoothing}`, lda: `shrinkage ${p.shrinkage}`,
      perceptron: `${p.epochs} epochs`
    };
    $("paramSummary").textContent = text[state.modelType];
  }
  function updateMetrics() {
    const m = state.metrics;
    $("metricGrid").innerHTML = [["Accuracy", m ? pct(m.accuracy) : "—"], ["Precision", m ? pct(m.precision) : "—"], ["Recall", m ? pct(m.recall) : "—"], ["F1 score", m ? pct(m.f1) : "—"], ["Points", state.points.length], ["Complexity", state.model?.complexity || "—"]].map(([a, b]) => `<div class="metric-tile"><small>${a}</small><strong>${b}</strong></div>`).join("");
    let extra = "Train the model to reveal classifier-specific diagnostics.";
    if (state.model) {
      if (state.modelType === "knn") extra = `<b>Lazy learner:</b> stores all ${state.points.length} points.<br>No fitting iterations are required.`;
      else if (state.modelType === "tree") extra = `<b>Tree structure:</b> ${state.model.complexity}`;
      else if (["logistic", "svm", "perceptron"].includes(state.modelType)) extra = `<b>Feature weights:</b> X ${fmt(state.model.w[0], 2)}, Y ${fmt(state.model.w[1], 2)}<br><b>Bias:</b> ${fmt(state.model.b, 2)}`;
      else if (state.modelType === "rbfsvm") extra = `<b>Support vectors:</b> ${state.model.support.length}<br><b>Kernel gamma:</b> ${state.model.gamma}`;
      else if (state.modelType === "mlp") extra = `<b>Architecture:</b> ${state.model.complexity}<br><b>Trainable parameters:</b> ${state.params.mlp.hidden * 4 + 1}`;
      else if (state.modelType === "naivebayes") extra = `<b>Class 1 prior:</b> ${pct(state.model.prior1)}<br>Independent Gaussian features.`;
      else extra = `<b>Discriminant weights:</b> X ${fmt(state.model.w[0], 2)}, Y ${fmt(state.model.w[1], 2)}`;
    }
    $("metricExtra").innerHTML = `<div class="metric-extra">${extra}</div>`;
    $("accuracyLabel").textContent = m ? `${pct(m.accuracy)} accuracy` : "Not trained";
    ["tp", "tn", "fp", "fn"].forEach(k => $(`${k}Value`).textContent = m ? m[k] : 0);
  }
  function updateVisual() {
    $("insightTitle").textContent = TITLES[state.modelType]; $("insightText").textContent = EXPLAINS[state.modelType];
    const host = $("modelVisual");
    const cards = {
      knn: [["1", "Measure distance", "Scale features"], ["2", "Select k closest", `${state.params.knn.k} neighbors`], ["3", "Count weighted votes", "Predict class"]],
      tree: [["1", "Test candidate splits", "Gini impurity"], ["2", "Partition recursively", "Rectangles"], ["3", "Vote inside a leaf", "Class probability"]],
      logistic: [["1", "Weight X and Y", "Linear score"], ["2", "Apply sigmoid", "0 to 1"], ["3", "Threshold probability", "Class label"]],
      svm: [["1", "Find separating line", "Hyperplane"], ["2", "Widen the margin", "Support vectors"], ["3", "Penalize violations", `C = ${state.params.svm.c}`]],
      rbfsvm: [["1", "Measure RBF similarity", `γ = ${state.params.rbfsvm.gamma}`], ["2", "Weight support vectors", `C = ${state.params.rbfsvm.c}`], ["3", "Combine local influences", "Curved margin"]],
      mlp: [["1", "Feed X and Y forward", "Input layer"], ["2", "Activate hidden neurons", `${state.params.mlp.hidden} tanh units`], ["3", "Backpropagate error", "Sigmoid output"]],
      naivebayes: [["1", "Fit class Gaussians", "Mean + variance"], ["2", "Multiply likelihoods", "Naive independence"], ["3", "Apply Bayes rule", "Posterior class"]],
      lda: [["1", "Estimate class means", "Centers"], ["2", "Pool covariance", "Shared shape"], ["3", "Project for separation", "Linear boundary"]],
      perceptron: [["1", "Start with a line", "Weights"], ["2", "Find a mistake", "Compare label"], ["3", "Move the boundary", "Repeat"]]
    };
    host.innerHTML = `<div class="flow-steps">${cards[state.modelType].map(c => `<div class="flow-step"><span class="num">${c[0]}</span><span>${c[1]}</span><small>${c[2]}</small></div>`).join("")}</div>`;
  }
  function updateReadout(point) {
    const pr = predict(point.x, point.y);
    let text = `Class ${pr.label}, with ${pct(pr.label ? pr.proba : 1 - pr.proba)} confidence.`;
    if (state.modelType === "knn") {
      const ns = state.model.neighbors(point.x, point.y), blue = ns.filter(n => n.point.label).length;
      text = `${blue} of ${ns.length} nearest examples are blue. Prediction: Class ${pr.label}.`;
    } else if (state.modelType === "tree") text = `The point lands in a leaf with Class 1 proportion ${pct(pr.proba)}.`;
    else if (state.modelType === "svm") text = `Signed margin score ${fmt(state.model.score(point.x, point.y), 2)}. Prediction: Class ${pr.label}.`;
    else if (state.modelType === "rbfsvm") text = `Kernel margin score ${fmt(state.model.score(point.x, point.y), 2)}, combined from ${state.model.support.length} support vectors.`;
    else if (state.modelType === "mlp") {
      const active = state.model.activations(point.x, point.y).filter(v => Math.abs(v) > .5).length;
      text = `${active} of ${state.params.mlp.hidden} hidden neurons are strongly active. Class 1 probability: ${pct(pr.proba)}.`;
    }
    $("selectedReadout").innerHTML = `<span class="readout-icon">${pr.label}</span><div><small>Classifier decision</small><p>${text}</p></div>`;
  }
  function drawChart() {
    const r = chart.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; if (!r.width) return;
    chart.width = r.width * dpr; chart.height = r.height * dpr; chartCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = r.width, h = r.height; chartCtx.clearRect(0, 0, w, h); chartCtx.strokeStyle = "#e4e8f0";
    for (let i = 1; i < 4; i++) { chartCtx.beginPath(); chartCtx.moveTo(28, i * h / 4); chartCtx.lineTo(w - 8, i * h / 4); chartCtx.stroke(); }
    const values = state.model?.losses || [];
    if (!values.length) { chartCtx.fillStyle = "#99a2b4"; chartCtx.font = "11px DM Sans"; chartCtx.textAlign = "center"; chartCtx.fillText(state.modelType === "knn" ? "k-NN has no iterative training curve" : "Train to reveal the learning curve", w / 2, h / 2); return; }
    const min = Math.min(...values), max = Math.max(...values), span = max - min || 1; chartCtx.beginPath();
    values.forEach((v, i) => { const x = 28 + i / Math.max(1, values.length - 1) * (w - 42), y = 12 + (max - v) / span * (h - 34); i ? chartCtx.lineTo(x, y) : chartCtx.moveTo(x, y); });
    chartCtx.strokeStyle = "#3157d5"; chartCtx.lineWidth = 2.5; chartCtx.stroke();
  }
  function updateChartText() {
    const iterative = ["logistic", "svm", "rbfsvm", "mlp", "perceptron"].includes(state.modelType);
    $("chartTitle").textContent = iterative ? (state.modelType === "perceptron" ? "Mistakes by epoch" : "Optimization curve") : "Model diagnostics";
    $("chartCaption").textContent = iterative ? "The curve shows how the optimization objective or number of mistakes changes during training." : `${MODEL_NAMES[state.modelType]} is fitted directly and does not require an iterative loss curve in this demonstration.`;
  }
  function updateTable() {
    $("tableCount").textContent = `(${state.points.length} points)`;
    $("dataTableBody").innerHTML = state.points.map((p, i) => `<tr data-id="${p.id}" class="${state.selected.has(p.id) ? "selected-row" : ""}"><td>${i + 1}</td><td>${fmt(p.x, 3)}</td><td>${fmt(p.y, 3)}</td><td><select class="class-select" data-class-id="${p.id}"><option value="0" ${!p.label ? "selected" : ""}>Class 0</option><option value="1" ${p.label ? "selected" : ""}>Class 1</option></select></td><td>${state.metrics ? `<span class="pred-pill c${state.metrics.predictions[i].label}">Class ${state.metrics.predictions[i].label}</span>` : "—"}</td><td><button class="row-delete" data-delete-id="${p.id}">Delete</button></td></tr>`).join("");
    document.querySelectorAll("[data-class-id]").forEach(el => el.addEventListener("change", () => { const p = state.points.find(q => q.id === Number(el.dataset.classId)); p.label = Number(el.value); trainModel(); }));
    document.querySelectorAll("[data-delete-id]").forEach(el => el.addEventListener("click", () => { state.points = state.points.filter(p => p.id !== Number(el.dataset.deleteId)); trainModel(); }));
  }
  function updateStatus() {
    if (state.trained) { $("plotStatus").className = "plot-status"; $("plotStatus").innerHTML = "<span></span> Model trained"; $("canvasMessage").classList.add("hidden"); }
    else { $("plotStatus").className = "plot-status untrained"; $("plotStatus").innerHTML = "<span></span> Needs update"; $("canvasMessage").textContent = state.points.length ? "Data or settings changed. Train the model to update its boundary." : "Add points or load a dataset to begin."; $("canvasMessage").classList.remove("hidden"); }
  }
  function updateAll() {
    $("datasetLabel").textContent = DATASET_NAMES[state.dataset]; $("modelLabel").textContent = MODEL_NAMES[state.modelType]; $("heroModel").textContent = MODEL_NAMES[state.modelType];
    $("plotTitle").textContent = `${MODEL_NAMES[state.modelType]} on ${DATASET_NAMES[state.dataset]}`; $("modelQuickExplain").textContent = EXPLAINS[state.modelType];
    updateParamSummary(); updateStatus(); updateMetrics(); updateVisual(); updateChartText(); updateTable(); drawPlot(); drawChart();
  }
  function showWarning(t) { $("trainingWarning").textContent = t; $("trainingWarning").classList.remove("hidden"); }
  function hideWarning() { $("trainingWarning").classList.add("hidden"); }
  function loadDataset() { state.points = generateDataset(state.dataset); state.originalPoints = state.points.map(p => ({ ...p })); state.selected.clear(); trainModel(); }
  function deleteSelected() { state.points = state.points.filter(p => !state.selected.has(p.id)); state.selected.clear(); trainModel(); }
  function clearData() { state.points = []; state.selected.clear(); invalidate(); }

  document.querySelectorAll(".menu-trigger").forEach(button => button.addEventListener("click", e => {
    e.stopPropagation(); const menu = $(button.dataset.menu), open = menu.classList.contains("open");
    document.querySelectorAll(".popover").forEach(p => p.classList.remove("open")); document.querySelectorAll(".menu-trigger").forEach(b => b.classList.remove("open"));
    if (!open) { menu.classList.add("open"); button.classList.add("open"); }
  }));
  document.querySelectorAll(".popover").forEach(p => p.addEventListener("click", e => e.stopPropagation()));
  document.addEventListener("click", () => { document.querySelectorAll(".popover").forEach(p => p.classList.remove("open")); document.querySelectorAll(".menu-trigger").forEach(b => b.classList.remove("open")); });
  document.querySelectorAll(".side-dropdown").forEach(panel => panel.addEventListener("toggle", () => { if (panel.open) { document.querySelectorAll(".side-dropdown").forEach(p => { if (p !== panel) p.open = false; }); requestAnimationFrame(resizeCanvas); } }));
  $("datasetSelect").addEventListener("change", e => { state.dataset = e.target.value; $("datasetLabel").textContent = DATASET_NAMES[state.dataset]; });
  $("loadDataset").addEventListener("click", loadDataset);
  $("modelSelect").addEventListener("change", e => { state.modelType = e.target.value; state.model = null; state.metrics = null; state.trained = false; renderParameters(); updateAll(); trainModel(); });
  $("trainButton").addEventListener("click", trainModel);
  $("resetDataset").addEventListener("click", () => { state.points = state.originalPoints.map(p => ({ ...p, id: state.nextId++ })); trainModel(); });
  $("brushRadius").addEventListener("input", e => { state.brushRadius = Number(e.target.value); $("brushRadiusValue").textContent = `${e.target.value} px`; });
  $("toolButtons").querySelectorAll("[data-tool]").forEach(button => button.addEventListener("click", () => {
    state.tool = button.dataset.tool; document.querySelectorAll(".tool").forEach(b => b.classList.toggle("active", b === button));
    $("toolLabel").textContent = { add: "Add point", brush1: "Brush blue", brush0: "Brush red", erase: "Eraser", select: "Select region" }[state.tool];
  }));
  $("deleteSelected").addEventListener("click", deleteSelected); $("tableDeleteSelected").addEventListener("click", deleteSelected);
  $("clearData").addEventListener("click", clearData); $("tableClear").addEventListener("click", clearData);
  document.querySelectorAll(".matrix-cell").forEach(cell => cell.addEventListener("click", () => {
    if (!state.metrics) return; state.confusionFilter = cell.dataset.cell; document.querySelectorAll(".matrix-cell").forEach(c => c.classList.toggle("active", c === cell));
    $("matrixExplanation").textContent = { tp: "Showing true positives.", tn: "Showing true negatives.", fp: "Showing false positives.", fn: "Showing false negatives." }[state.confusionFilter]; drawPlot();
  }));
  $("showAll").addEventListener("click", () => { state.confusionFilter = null; document.querySelectorAll(".matrix-cell").forEach(c => c.classList.remove("active")); $("matrixExplanation").textContent = "Click a cell to isolate those points on the plot."; drawPlot(); });
  $("predictButton").addEventListener("click", () => {
    if (!state.trained) return;
    const x = Number($("predictX").value), y = Number($("predictY").value), pr = predict(x, y), conf = pr.label ? pr.proba : 1 - pr.proba;
    let explanation = `${MODEL_NAMES[state.modelType]} assigns ${pct(conf)} confidence to Class ${pr.label}.`;
    if (state.modelType === "knn") { const ns = state.model.neighbors(x, y), blue = ns.filter(n => n.point.label).length; explanation = `${blue} of ${ns.length} nearest examples are Class 1.`; }
    $("predictionOutput").innerHTML = `<div class="prediction-class ${pr.label ? "" : "red"}">${pr.label}</div><div><small>Predicted Class ${pr.label} · ${pct(conf)}</small><p>${explanation}</p></div>`;
  });

  const qa = [
    ["Why can k-NN create very irregular decision boundaries?", "It follows local examples directly, so noise and small neighborhoods can create many small regions."],
    ["How does increasing k change k-NN?", "It smooths the boundary and reduces variance, but a very large k can miss local structure and increase bias."],
    ["Why does feature scaling matter for k-NN and SVM?", "Features with larger numeric ranges otherwise dominate distances and margins."],
    ["What shapes can a decision tree represent?", "Axis-aligned rectangular regions, combined recursively into complex nonlinear boundaries."],
    ["Why is logistic regression called regression if it classifies?", "It regresses the log-odds or probability of a class, then thresholds that probability."],
    ["What is the role of the margin in an SVM?", "A wider margin aims to make the classifier more robust to small changes in new points."],
    ["How do C and gamma affect an RBF SVM?", "Larger C penalizes mistakes more strongly. Larger gamma makes each support vector's influence more local, producing a more detailed boundary."],
    ["Why can a multilayer network learn XOR while logistic regression cannot?", "Hidden nonlinear units transform the inputs, allowing the network to combine multiple linear pieces into a nonlinear boundary."],
    ["What naive assumption does Naive Bayes make?", "It assumes features are conditionally independent given the class."],
    ["When will linear models struggle?", "They struggle when classes require curved, disconnected, XOR-like, or strongly local boundaries."]
  ];
  $("questionList").innerHTML = qa.map(q => `<li>${q[0]}</li>`).join(""); $("answerList").innerHTML = qa.map(q => `<li>${q[1]}</li>`).join("");
  const style = document.createElement("style"); style.textContent = `.toggle-row{display:flex;align-items:center;justify-content:space-between;margin-top:12px;color:#566078;font-size:12px;font-weight:700}.toggle-row input{width:17px;height:17px;accent-color:#3157d5}.field-label.spaced{margin-top:14px}`; document.head.appendChild(style);
  window.addEventListener("resize", () => { clearTimeout(resizeCanvas.timer); resizeCanvas.timer = setTimeout(resizeCanvas, 80); });
  document.addEventListener("keydown", e => { if ((e.key === "Delete" || e.key === "Backspace") && state.selected.size && !/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) deleteSelected(); });
  renderParameters(); requestAnimationFrame(() => { resizeCanvas(); loadDataset(); });
})();
