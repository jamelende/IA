(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const sigmoid = (x) => 1 / (1 + Math.exp(-clamp(x, -30, 30)));
  const fmt = (v, digits = 2) => Number(v).toFixed(digits);

  class RNG {
    constructor(seed = 20260609) { this.state = seed >>> 0; }
    next() {
      this.state = (1664525 * this.state + 1013904223) >>> 0;
      return this.state / 4294967296;
    }
    normal() {
      const u = Math.max(this.next(), 1e-9), v = this.next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    pick(n) { return Math.floor(this.next() * n); }
  }

  const MODEL_NAMES = {
    tree: "Single Decision Tree",
    forest: "Random Forest",
    adaboost: "AdaBoost",
    gradient: "Gradient Boosting",
    xgboost: "XGBoost-style Boosting"
  };
  const DATASET_NAMES = {
    students: "Students", linear: "Linear separation", xor: "XOR", circles: "Circles",
    moons: "Two moons", spiral: "Spiral", stripes: "Horizontal stripes", outlier: "Outlier dataset",
    noisy: "Noisy classification", imbalanced: "Imbalanced dataset"
  };
  const EXPLAINS = {
    tree: "A single tree splits the plane into simple rectangular regions using the feature and threshold that best reduce Gini impurity.",
    forest: "Trains many trees on different bootstrap samples and predicts by majority vote. Diversity helps the group generalize.",
    adaboost: "Sequentially trains weak learners and gives more weight to previously misclassified points.",
    gradient: "Builds trees one after another to correct previous errors by following the loss gradient.",
    xgboost: "A regularized boosting model inspired by XGBoost, simplified for teaching."
  };

  const defaultParams = {
    tree: { maxDepth: 4, minSplit: 2 },
    forest: { trees: 30, maxDepth: 4, featureSampling: "one", bootstrap: true, showTrees: true, treeOpacity: .28 },
    adaboost: { estimators: 25, learningRate: .6, maxDepth: 1, showWeights: true, showContribution: true },
    gradient: { estimators: 30, learningRate: .12, maxDepth: 2, subsample: .85 },
    xgboost: { estimators: 35, learningRate: .12, maxDepth: 3, lambda: 2, minChildWeight: 2, subsample: .85, colsample: 1 }
  };

  const state = {
    dataset: "moons",
    modelType: "forest",
    params: JSON.parse(JSON.stringify(defaultParams)),
    points: [],
    originalPoints: [],
    model: null,
    trained: false,
    tool: "add",
    addClass: 1,
    brushRadius: 24,
    selected: new Set(),
    confusionFilter: null,
    dragging: false,
    dragStart: null,
    dragCurrent: null,
    bounds: { xmin: -3, xmax: 3, ymin: -3, ymax: 3 },
    metrics: null,
    regionImage: null,
    hoverIndex: -1,
    nextId: 1
  };

  function generateDataset(type) {
    const rng = new RNG(20260609 + Object.keys(DATASET_NAMES).indexOf(type) * 733);
    const pts = [];
    const add = (x, y, label) => pts.push({ id: state.nextId++, x, y, label });
    if (type === "students") {
      for (let i = 0; i < 90; i++) {
        const x = clamp(1 + rng.next() * 8 + rng.normal() * .25, 0, 10);
        const y = clamp(3.5 + rng.next() * 5 + rng.normal() * .25, 2, 10);
        const score = .75 * x + .55 * y + rng.normal() * 1.15;
        add(x, y, score > 7.7 ? 1 : 0);
      }
    } else if (type === "linear") {
      for (let i = 0; i < 90; i++) {
        const x = rng.next() * 6 - 3, y = rng.next() * 6 - 3;
        add(x, y, y > .65 * x + rng.normal() * .55 ? 1 : 0);
      }
    } else if (type === "xor") {
      for (let i = 0; i < 100; i++) {
        const x = rng.next() * 6 - 3, y = rng.next() * 6 - 3;
        add(x, y, (x > 0) !== (y > 0) ? 1 : 0);
      }
    } else if (type === "circles") {
      for (let i = 0; i < 110; i++) {
        const outer = i >= 50;
        const angle = rng.next() * Math.PI * 2;
        const radius = (outer ? 2.25 : 1.0) + rng.normal() * .2;
        add(Math.cos(angle) * radius, Math.sin(angle) * radius, outer ? 0 : 1);
      }
    } else if (type === "moons") {
      for (let i = 0; i < 100; i++) {
        const c = i < 50 ? 0 : 1, angle = rng.next() * Math.PI;
        const x = c === 0 ? Math.cos(angle) : 1 - Math.cos(angle);
        const y = c === 0 ? Math.sin(angle) : -.55 - Math.sin(angle);
        add(x * 2 + rng.normal() * .15, y * 1.7 + rng.normal() * .15, c);
      }
    } else if (type === "spiral") {
      for (let c = 0; c < 2; c++) {
        for (let i = 0; i < 55; i++) {
          const r = .25 + i / 22, t = i / 8 + c * Math.PI;
          add(r * Math.cos(t) + rng.normal() * .12, r * Math.sin(t) + rng.normal() * .12, c);
        }
      }
    } else if (type === "stripes") {
      for (let i = 0; i < 110; i++) {
        const x = rng.next() * 6 - 3, y = rng.next() * 6 - 3;
        add(x, y, Math.floor((y + 3) / 1.05) % 2);
      }
    } else if (type === "outlier") {
      for (let i = 0; i < 82; i++) {
        const c = i < 41 ? 0 : 1;
        add((c ? 1.3 : -1.3) + rng.normal() * .65, (c ? 1 : -1) + rng.normal() * .7, c);
      }
      add(-2.3, -2.1, 1); add(2.35, 2.2, 0); add(-2.1, 1.8, 1); add(2.1, -1.8, 0);
    } else if (type === "noisy") {
      for (let i = 0; i < 115; i++) {
        const x = rng.next() * 6 - 3, y = rng.next() * 6 - 3;
        let label = y > .45 * x + Math.sin(x * 1.6) ? 1 : 0;
        if (rng.next() < .18) label = 1 - label;
        add(x, y, label);
      }
    } else if (type === "imbalanced") {
      for (let i = 0; i < 105; i++) {
        const minority = i < 18;
        add((minority ? 1.45 : -.55) + rng.normal() * (minority ? .58 : 1.05),
          (minority ? 1.15 : -.35) + rng.normal() * (minority ? .58 : 1.0), minority ? 1 : 0);
      }
    }
    return pts;
  }

  function gini(labels, weights = null) {
    if (!labels.length) return 0;
    let w0 = 0, w1 = 0;
    labels.forEach((v, i) => { const w = weights ? weights[i] : 1; v ? w1 += w : w0 += w; });
    const total = w0 + w1 || 1, p0 = w0 / total, p1 = w1 / total;
    return 1 - p0 * p0 - p1 * p1;
  }

  function buildClassTree(samples, options, depth = 0) {
    const { maxDepth = 3, minSplit = 2, rng = new RNG(), featureMode = "both", weights = null } = options;
    const labels = samples.map(s => s.label);
    const localWeights = weights || samples.map(s => s.weight ?? 1);
    const totalWeight = localWeights.reduce((a, b) => a + b, 0) || 1;
    const p1 = samples.reduce((sum, s, i) => sum + (s.label ? localWeights[i] : 0), 0) / totalWeight;
    const node = { leaf: true, pred: p1 >= .5 ? 1 : 0, prob: p1, depth };
    if (depth >= maxDepth || samples.length < minSplit || gini(labels, localWeights) < 1e-9) return node;

    const features = featureMode === "one" ? [rng.pick(2)] : [0, 1];
    let best = null;
    for (const feature of features) {
      const values = [...new Set(samples.map(s => feature ? s.y : s.x))].sort((a, b) => a - b);
      const candidates = [];
      const stride = Math.max(1, Math.ceil(values.length / 28));
      for (let i = stride; i < values.length; i += stride) candidates.push((values[i - 1] + values[i]) / 2);
      for (const threshold of candidates) {
        const left = [], right = [], lw = [], rw = [];
        samples.forEach((s, i) => {
          if ((feature ? s.y : s.x) <= threshold) { left.push(s); lw.push(localWeights[i]); }
          else { right.push(s); rw.push(localWeights[i]); }
        });
        if (!left.length || !right.length) continue;
        const lsum = lw.reduce((a, b) => a + b, 0), rsum = rw.reduce((a, b) => a + b, 0);
        const impurity = (lsum * gini(left.map(s => s.label), lw) + rsum * gini(right.map(s => s.label), rw)) / totalWeight;
        if (!best || impurity < best.impurity) best = { feature, threshold, impurity, left, right, lw, rw };
      }
    }
    if (!best) return node;
    return {
      leaf: false, feature: best.feature, threshold: best.threshold, depth,
      left: buildClassTree(best.left, { ...options, weights: best.lw }, depth + 1),
      right: buildClassTree(best.right, { ...options, weights: best.rw }, depth + 1)
    };
  }

  function treeProb(node, x, y) {
    while (!node.leaf) node = ((node.feature ? y : x) <= node.threshold) ? node.left : node.right;
    return node.prob;
  }
  function treeDepth(node) { return node.leaf ? node.depth : Math.max(treeDepth(node.left), treeDepth(node.right)); }
  function treeFeatureCounts(node, counts = [0, 0]) {
    if (!node.leaf) { counts[node.feature]++; treeFeatureCounts(node.left, counts); treeFeatureCounts(node.right, counts); }
    return counts;
  }

  function buildRegressionTree(samples, targets, options, depth = 0) {
    const { maxDepth = 2, minLeaf = 2, lambda = 0, rng = new RNG(), colsample = 1 } = options;
    const mean = targets.reduce((a, b) => a + b, 0) / (targets.length || 1);
    const value = mean / (1 + lambda / Math.max(1, targets.length));
    const node = { leaf: true, value, depth };
    if (depth >= maxDepth || samples.length < minLeaf * 2) return node;
    const available = colsample < .75 ? [rng.pick(2)] : [0, 1];
    let best = null;
    for (const feature of available) {
      const order = samples.map((s, i) => ({ v: feature ? s.y : s.x, i })).sort((a, b) => a.v - b.v);
      const stride = Math.max(1, Math.ceil(order.length / 28));
      for (let cut = minLeaf; cut <= order.length - minLeaf; cut += stride) {
        const threshold = (order[cut - 1].v + order[cut].v) / 2;
        const li = [], ri = [];
        samples.forEach((s, i) => ((feature ? s.y : s.x) <= threshold ? li : ri).push(i));
        if (li.length < minLeaf || ri.length < minLeaf) continue;
        const lm = li.reduce((sum, i) => sum + targets[i], 0) / li.length;
        const rm = ri.reduce((sum, i) => sum + targets[i], 0) / ri.length;
        const loss = li.reduce((sum, i) => sum + (targets[i] - lm) ** 2, 0) +
          ri.reduce((sum, i) => sum + (targets[i] - rm) ** 2, 0) + lambda * (lm * lm + rm * rm);
        if (!best || loss < best.loss) best = { feature, threshold, li, ri, loss };
      }
    }
    if (!best) return node;
    return {
      leaf: false, feature: best.feature, threshold: best.threshold, depth,
      left: buildRegressionTree(best.li.map(i => samples[i]), best.li.map(i => targets[i]), options, depth + 1),
      right: buildRegressionTree(best.ri.map(i => samples[i]), best.ri.map(i => targets[i]), options, depth + 1)
    };
  }
  function regressionValue(node, x, y) {
    while (!node.leaf) node = ((node.feature ? y : x) <= node.threshold) ? node.left : node.right;
    return node.value;
  }

  function trainModel() {
    const samples = state.points;
    const classes = new Set(samples.map(p => p.label));
    if (samples.length < 4 || classes.size < 2) {
      state.model = null; state.trained = false; state.metrics = null;
      showWarning(classes.size < 2 ? "Add points from both classes to train the model." : "Add at least four points to train the model.");
      updateAll();
      return;
    }
    hideWarning();
    const rng = new RNG(44117);
    const p = state.params[state.modelType];
    if (state.modelType === "tree") {
      const root = buildClassTree(samples, { maxDepth: p.maxDepth, minSplit: p.minSplit, rng });
      state.model = {
        type: "tree", root,
        predictProba: (x, y) => treeProb(root, x, y),
        complexity: `Depth ${treeDepth(root)}`
      };
    } else if (state.modelType === "forest") {
      const trees = [];
      for (let t = 0; t < p.trees; t++) {
        const batch = p.bootstrap
          ? Array.from({ length: samples.length }, () => samples[rng.pick(samples.length)])
          : [...samples].sort(() => rng.next() - .5);
        trees.push(buildClassTree(batch, { maxDepth: p.maxDepth, minSplit: 2, rng, featureMode: p.featureSampling }));
      }
      state.model = {
        type: "forest", trees,
        predictProba: (x, y) => trees.reduce((s, tr) => s + (treeProb(tr, x, y) >= .5 ? 1 : 0), 0) / trees.length,
        complexity: `${trees.length} trees`,
        avgDepth: trees.reduce((s, tr) => s + treeDepth(tr), 0) / trees.length
      };
    } else if (state.modelType === "adaboost") {
      const n = samples.length, weights = Array(n).fill(1 / n), learners = [], losses = [];
      const weightHistory = [weights.slice()];
      for (let t = 0; t < p.estimators; t++) {
        const weightedSamples = samples.map((s, i) => ({ ...s, weight: weights[i] }));
        const tree = buildClassTree(weightedSamples, { maxDepth: p.maxDepth, minSplit: 2, rng, weights });
        let error = 0;
        const predictions = samples.map(s => treeProb(tree, s.x, s.y) >= .5 ? 1 : 0);
        predictions.forEach((pred, i) => { if (pred !== samples[i].label) error += weights[i]; });
        error = clamp(error, 1e-5, .499);
        const alpha = p.learningRate * .5 * Math.log((1 - error) / error);
        learners.push({ tree, alpha, error });
        let sum = 0;
        predictions.forEach((pred, i) => {
          const yi = samples[i].label ? 1 : -1, hi = pred ? 1 : -1;
          weights[i] *= Math.exp(-alpha * yi * hi); sum += weights[i];
        });
        weights.forEach((_, i) => weights[i] /= sum);
        losses.push(error); weightHistory.push(weights.slice());
        if (error < 1e-5) break;
      }
      state.model = {
        type: "adaboost", learners, weights: weights.slice(), weightHistory, losses,
        predictScore: (x, y) => learners.reduce((s, l) => s + l.alpha * (treeProb(l.tree, x, y) >= .5 ? 1 : -1), 0),
        predictProba(x, y) { return sigmoid(2 * this.predictScore(x, y)); },
        complexity: `${learners.length} weak learners`
      };
    } else {
      const isXGB = state.modelType === "xgboost";
      const labels = samples.map(s => s.label);
      const prior = clamp(labels.reduce((a, b) => a + b, 0) / labels.length, .02, .98);
      const initial = Math.log(prior / (1 - prior));
      const scores = Array(samples.length).fill(initial), trees = [], losses = [];
      for (let t = 0; t < p.estimators; t++) {
        const probs = scores.map(sigmoid);
        const residuals = labels.map((y, i) => y - probs[i]);
        const indices = [];
        for (let i = 0; i < samples.length; i++) if (rng.next() <= p.subsample) indices.push(i);
        if (indices.length < 4) indices.push(0, 1, 2, 3);
        const subSamples = indices.map(i => samples[i]), subTargets = indices.map(i => residuals[i]);
        const tree = buildRegressionTree(subSamples, subTargets, {
          maxDepth: p.maxDepth, minLeaf: isXGB ? p.minChildWeight : 2,
          lambda: isXGB ? p.lambda : 0, rng, colsample: isXGB ? p.colsample : 1
        });
        trees.push(tree);
        samples.forEach((s, i) => { scores[i] += p.learningRate * regressionValue(tree, s.x, s.y); });
        const loss = labels.reduce((sum, y, i) => {
          const pr = clamp(sigmoid(scores[i]), 1e-6, 1 - 1e-6);
          return sum - y * Math.log(pr) - (1 - y) * Math.log(1 - pr);
        }, 0) / labels.length;
        losses.push(loss);
      }
      state.model = {
        type: state.modelType, trees, losses, initial, learningRate: p.learningRate,
        predictScore(x, y) { return this.initial + this.trees.reduce((s, tr) => s + this.learningRate * regressionValue(tr, x, y), 0); },
        predictProba(x, y) { return sigmoid(this.predictScore(x, y)); },
        complexity: `${trees.length} sequential trees`
      };
    }
    state.trained = true;
    computeMetrics();
    buildRegionImage();
    updateAll();
  }

  function predict(x, y) {
    if (!state.model) return { label: null, proba: .5 };
    const proba = clamp(state.model.predictProba(x, y), 0, 1);
    return { label: proba >= .5 ? 1 : 0, proba };
  }

  function computeMetrics() {
    let tp = 0, tn = 0, fp = 0, fn = 0;
    const predictions = state.points.map(p => {
      const pred = predict(p.x, p.y);
      if (p.label === 1 && pred.label === 1) tp++;
      else if (p.label === 0 && pred.label === 0) tn++;
      else if (p.label === 0 && pred.label === 1) fp++;
      else fn++;
      return pred;
    });
    const total = tp + tn + fp + fn;
    const precision = tp / (tp + fp || 1), recall = tp / (tp + fn || 1);
    state.metrics = {
      tp, tn, fp, fn, predictions,
      accuracy: (tp + tn) / (total || 1), precision, recall,
      f1: 2 * precision * recall / (precision + recall || 1)
    };
  }

  function calculateBounds() {
    if (!state.points.length) return { xmin: -3, xmax: 3, ymin: -3, ymax: 3 };
    const xs = state.points.map(p => p.x), ys = state.points.map(p => p.y);
    let xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
    const xpad = Math.max(.65, (xmax - xmin) * .16), ypad = Math.max(.65, (ymax - ymin) * .16);
    if (xmax - xmin < .1) { xmin -= 1; xmax += 1; }
    if (ymax - ymin < .1) { ymin -= 1; ymax += 1; }
    return { xmin: xmin - xpad, xmax: xmax + xpad, ymin: ymin - ypad, ymax: ymax + ypad };
  }

  const canvas = $("plotCanvas"), ctx = canvas.getContext("2d");
  const chart = $("miniChart"), chartCtx = chart.getContext("2d");
  const PAD = { l: 53, r: 18, t: 18, b: 42 };

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1, rect = canvas.getBoundingClientRect();
    const w = Math.max(320, Math.floor(rect.width)), h = Math.max(320, Math.floor(rect.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr; canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      buildRegionImage();
    }
    drawPlot();
    drawMiniChart();
  }

  function dataToCanvas(x, y) {
    const w = canvas.clientWidth, h = canvas.clientHeight, b = state.bounds;
    return {
      x: PAD.l + (x - b.xmin) / (b.xmax - b.xmin) * (w - PAD.l - PAD.r),
      y: PAD.t + (b.ymax - y) / (b.ymax - b.ymin) * (h - PAD.t - PAD.b)
    };
  }
  function canvasToData(px, py) {
    const w = canvas.clientWidth, h = canvas.clientHeight, b = state.bounds;
    return {
      x: b.xmin + (px - PAD.l) / (w - PAD.l - PAD.r) * (b.xmax - b.xmin),
      y: b.ymax - (py - PAD.t) / (h - PAD.t - PAD.b) * (b.ymax - b.ymin)
    };
  }

  function buildRegionImage() {
    state.bounds = calculateBounds();
    if (!state.trained || !state.model || !canvas.clientWidth) { state.regionImage = null; drawPlot(); return; }
    const w = Math.max(1, Math.floor(canvas.clientWidth - PAD.l - PAD.r));
    const h = Math.max(1, Math.floor(canvas.clientHeight - PAD.t - PAD.b));
    const off = document.createElement("canvas"), scale = .32;
    off.width = Math.max(100, Math.floor(w * scale)); off.height = Math.max(100, Math.floor(h * scale));
    const octx = off.getContext("2d"), img = octx.createImageData(off.width, off.height);
    for (let py = 0; py < off.height; py++) {
      for (let px = 0; px < off.width; px++) {
        const x = state.bounds.xmin + px / (off.width - 1) * (state.bounds.xmax - state.bounds.xmin);
        const y = state.bounds.ymax - py / (off.height - 1) * (state.bounds.ymax - state.bounds.ymin);
        const prob = predict(x, y).proba;
        const confidence = Math.abs(prob - .5) * 2;
        const blue = [49, 87, 213], red = [231, 76, 91], base = prob >= .5 ? blue : red;
        const alpha = 46 + confidence * 42, i = (py * off.width + px) * 4;
        img.data[i] = base[0]; img.data[i + 1] = base[1]; img.data[i + 2] = base[2]; img.data[i + 3] = alpha;
      }
    }
    octx.putImageData(img, 0, 0);
    state.regionImage = off;
    drawPlot();
  }

  function drawGrid() {
    const w = canvas.clientWidth, h = canvas.clientHeight, b = state.bounds;
    ctx.fillStyle = "#fbfcfe"; ctx.fillRect(0, 0, w, h);
    if (state.regionImage) {
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(state.regionImage, PAD.l, PAD.t, w - PAD.l - PAD.r, h - PAD.t - PAD.b);
    }
    ctx.strokeStyle = "rgba(106,119,145,.13)"; ctx.lineWidth = 1; ctx.font = "10px DM Sans";
    ctx.fillStyle = "#8993a6";
    const xStep = niceStep((b.xmax - b.xmin) / 7), yStep = niceStep((b.ymax - b.ymin) / 6);
    for (let x = Math.ceil(b.xmin / xStep) * xStep; x <= b.xmax; x += xStep) {
      const p = dataToCanvas(x, 0);
      ctx.beginPath(); ctx.moveTo(p.x, PAD.t); ctx.lineTo(p.x, h - PAD.b); ctx.stroke();
      ctx.textAlign = "center"; ctx.fillText(roundTick(x), p.x, h - 20);
    }
    for (let y = Math.ceil(b.ymin / yStep) * yStep; y <= b.ymax; y += yStep) {
      const p = dataToCanvas(0, y);
      ctx.beginPath(); ctx.moveTo(PAD.l, p.y); ctx.lineTo(w - PAD.r, p.y); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(roundTick(y), PAD.l - 9, p.y + 3);
    }
    const zero = dataToCanvas(0, 0);
    ctx.strokeStyle = "rgba(55,67,91,.32)"; ctx.lineWidth = 1.2;
    if (zero.x >= PAD.l && zero.x <= w - PAD.r) { ctx.beginPath(); ctx.moveTo(zero.x, PAD.t); ctx.lineTo(zero.x, h - PAD.b); ctx.stroke(); }
    if (zero.y >= PAD.t && zero.y <= h - PAD.b) { ctx.beginPath(); ctx.moveTo(PAD.l, zero.y); ctx.lineTo(w - PAD.r, zero.y); ctx.stroke(); }
    ctx.fillStyle = "#667085"; ctx.font = "700 11px DM Sans"; ctx.textAlign = "right"; ctx.fillText("X", w - PAD.r, h - 7);
    ctx.save(); ctx.translate(13, PAD.t); ctx.rotate(-Math.PI / 2); ctx.textAlign = "right"; ctx.fillText("Y", 0, 0); ctx.restore();
  }
  function niceStep(raw) {
    const power = 10 ** Math.floor(Math.log10(raw)), n = raw / power;
    return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * power;
  }
  function roundTick(v) { return Math.abs(v) < 1e-9 ? "0" : Math.abs(v) >= 10 ? fmt(v, 0) : fmt(v, 1); }

  function drawPlot() {
    if (!canvas.clientWidth) return;
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    drawGrid();
    state.points.forEach((p, i) => {
      if (state.confusionFilter && state.trained) {
        const pred = state.metrics.predictions[i].label;
        const key = p.label ? (pred ? "tp" : "fn") : (pred ? "fp" : "tn");
        if (key !== state.confusionFilter) {
          drawPoint(p, i, .12);
          return;
        }
      }
      drawPoint(p, i, 1);
    });
    if (state.dragging && state.tool === "select" && state.dragStart && state.dragCurrent) {
      const x = Math.min(state.dragStart.x, state.dragCurrent.x), y = Math.min(state.dragStart.y, state.dragCurrent.y);
      const w = Math.abs(state.dragStart.x - state.dragCurrent.x), h = Math.abs(state.dragStart.y - state.dragCurrent.y);
      ctx.fillStyle = "rgba(49,87,213,.10)"; ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = "#3157d5"; ctx.setLineDash([5, 4]); ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
    }
  }

  function drawPoint(p, i, opacity) {
    const pos = dataToCanvas(p.x, p.y);
    let r = 6;
    if (state.modelType === "adaboost" && state.trained && state.params.adaboost.showWeights && state.model.weights) {
      const avg = 1 / state.points.length;
      r = clamp(4.5 + 7 * Math.sqrt((state.model.weights[i] || avg) / avg), 5, 15);
    }
    ctx.globalAlpha = opacity;
    ctx.beginPath(); ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
    ctx.fillStyle = p.label ? "#3157d5" : "#e74c5b"; ctx.fill();
    ctx.lineWidth = state.selected.has(p.id) ? 3 : 1.8;
    ctx.strokeStyle = state.selected.has(p.id) ? "#111827" : "#ffffff"; ctx.stroke();
    if (state.trained && state.metrics && state.metrics.predictions[i].label !== p.label) {
      ctx.strokeStyle = "#182033"; ctx.lineWidth = 2.3;
      ctx.beginPath(); ctx.moveTo(pos.x - r * .65, pos.y - r * .65); ctx.lineTo(pos.x + r * .65, pos.y + r * .65);
      ctx.moveTo(pos.x + r * .65, pos.y - r * .65); ctx.lineTo(pos.x - r * .65, pos.y + r * .65); ctx.stroke();
    }
    if (i === state.hoverIndex) {
      ctx.beginPath(); ctx.arc(pos.x, pos.y, r + 5, 0, Math.PI * 2); ctx.strokeStyle = "#172033"; ctx.lineWidth = 1.5; ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function eventPosition(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }
  function insidePlot(pos) {
    return pos.x >= PAD.l && pos.x <= canvas.clientWidth - PAD.r && pos.y >= PAD.t && pos.y <= canvas.clientHeight - PAD.b;
  }
  function nearestPoint(pos, maxDist = 13) {
    let best = -1, dBest = maxDist;
    state.points.forEach((p, i) => {
      const q = dataToCanvas(p.x, p.y), d = Math.hypot(q.x - pos.x, q.y - pos.y);
      if (d < dBest) { dBest = d; best = i; }
    });
    return best;
  }
  function addPointAt(pos, label) {
    if (!insidePlot(pos)) return;
    const d = canvasToData(pos.x, pos.y);
    state.points.push({ id: state.nextId++, x: d.x, y: d.y, label });
    invalidateAndRefresh();
  }
  function brushAt(pos, label) {
    if (!insidePlot(pos)) return;
    const rng = new RNG(Math.floor(pos.x * 31 + pos.y * 17 + state.points.length * 101));
    for (let i = 0; i < 3; i++) {
      const angle = rng.next() * Math.PI * 2, radius = Math.sqrt(rng.next()) * state.brushRadius;
      const p = { x: pos.x + Math.cos(angle) * radius, y: pos.y + Math.sin(angle) * radius };
      if (insidePlot(p)) {
        const d = canvasToData(p.x, p.y);
        state.points.push({ id: state.nextId++, x: d.x, y: d.y, label });
      }
    }
    invalidateAndRefresh(false);
  }
  function eraseAt(pos) {
    const before = state.points.length;
    state.points = state.points.filter(p => {
      const q = dataToCanvas(p.x, p.y);
      return Math.hypot(q.x - pos.x, q.y - pos.y) > state.brushRadius;
    });
    if (state.points.length !== before) invalidateAndRefresh(false);
  }

  canvas.addEventListener("pointerdown", (e) => {
    const pos = eventPosition(e);
    if (!insidePlot(pos)) return;
    canvas.setPointerCapture(e.pointerId);
    state.dragging = true;
    if (state.tool === "select") { state.dragStart = pos; state.dragCurrent = pos; drawPlot(); }
    else if (state.tool === "brush0") brushAt(pos, 0);
    else if (state.tool === "brush1") brushAt(pos, 1);
    else if (state.tool === "erase") eraseAt(pos);
  });
  canvas.addEventListener("pointermove", (e) => {
    const pos = eventPosition(e);
    if (state.dragging) {
      if (state.tool === "select") { state.dragCurrent = pos; drawPlot(); }
      else if (state.tool === "brush0") brushAt(pos, 0);
      else if (state.tool === "brush1") brushAt(pos, 1);
      else if (state.tool === "erase") eraseAt(pos);
      return;
    }
    const idx = nearestPoint(pos, 14);
    if (idx !== state.hoverIndex) { state.hoverIndex = idx; drawPlot(); }
    if (idx >= 0) showPointHover(idx, pos); else hidePointHover();
  });
  canvas.addEventListener("pointerup", (e) => {
    const pos = eventPosition(e);
    if (state.tool === "add" && state.dragging) {
      const idx = nearestPoint(pos, 11);
      if (idx >= 0) state.points[idx].label = 1 - state.points[idx].label;
      else addPointAt(pos, state.addClass);
      invalidateAndRefresh();
    } else if (state.tool === "select" && state.dragStart) {
      const minx = Math.min(state.dragStart.x, pos.x), maxx = Math.max(state.dragStart.x, pos.x);
      const miny = Math.min(state.dragStart.y, pos.y), maxy = Math.max(state.dragStart.y, pos.y);
      state.selected.clear();
      state.points.forEach(p => {
        const q = dataToCanvas(p.x, p.y);
        if (q.x >= minx && q.x <= maxx && q.y >= miny && q.y <= maxy) state.selected.add(p.id);
      });
      state.dragStart = null; state.dragCurrent = null;
      updateTable(); drawPlot();
    }
    state.dragging = false;
  });
  canvas.addEventListener("pointerleave", () => { if (!state.dragging) { state.hoverIndex = -1; hidePointHover(); drawPlot(); } });
  canvas.addEventListener("contextmenu", (e) => { e.preventDefault(); state.addClass = 1 - state.addClass; });

  function showPointHover(i, pos) {
    const point = state.points[i], box = $("hoverCard");
    let html = `<b>Point ${i + 1}</b><br>Actual class: ${point.label}<br>`;
    if (state.trained) {
      const pr = predict(point.x, point.y);
      html += `Prediction: Class ${pr.label}<br>Confidence: ${fmt((pr.label ? pr.proba : 1 - pr.proba) * 100, 0)}%`;
      if (state.modelType === "forest") {
        const votes1 = state.model.trees.reduce((s, t) => s + (treeProb(t, point.x, point.y) >= .5), 0);
        html += `<br><b>Votes:</b> ${state.model.trees.length - votes1} red · ${votes1} blue`;
      } else if (state.modelType === "adaboost") {
        html += `<br>Sample weight: ${fmt(state.model.weights[i] || 0, 4)}`;
      }
      updateSelectedReadout(point);
    }
    box.innerHTML = html; box.classList.remove("hidden");
    box.style.left = `${clamp(pos.x + 14, 4, canvas.clientWidth - 210)}px`;
    box.style.top = `${clamp(pos.y - 18, 4, canvas.clientHeight - 105)}px`;
  }
  function hidePointHover() { $("hoverCard").classList.add("hidden"); }

  function invalidateAndRefresh(rebounds = true) {
    state.trained = false; state.model = null; state.metrics = null; state.regionImage = null; state.confusionFilter = null;
    if (rebounds) state.bounds = calculateBounds();
    updateAll(); drawPlot();
  }

  function parameterSchema() {
    const type = state.modelType, p = state.params[type];
    const range = (key, label, min, max, step, value, suffix = "") =>
      `<label class="range-row">${label}<b data-value="${key}">${value}${suffix}</b><input data-param="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${value}"></label>`;
    const toggle = (key, label, checked) =>
      `<label class="toggle-row"><span>${label}</span><input data-param="${key}" type="checkbox" ${checked ? "checked" : ""}></label>`;
    if (type === "tree") return range("maxDepth", "Max depth", 1, 8, 1, p.maxDepth) + range("minSplit", "Min samples split", 2, 10, 1, p.minSplit);
    if (type === "forest") return range("trees", "Number of trees", 1, 100, 1, p.trees) +
      range("maxDepth", "Max depth", 1, 8, 1, p.maxDepth) +
      `<label class="field-label spaced">Feature sampling<select data-param="featureSampling"><option value="both" ${p.featureSampling === "both" ? "selected" : ""}>Both features</option><option value="one" ${p.featureSampling === "one" ? "selected" : ""}>Random one feature per split</option></select></label>` +
      toggle("bootstrap", "Bootstrap samples", p.bootstrap) + toggle("showTrees", "Show individual trees", p.showTrees) +
      range("treeOpacity", "Tree opacity", .05, .7, .05, p.treeOpacity);
    if (type === "adaboost") return range("estimators", "Weak learners", 1, 100, 1, p.estimators) +
      range("learningRate", "Learning rate", .01, 2, .01, p.learningRate) + range("maxDepth", "Weak learner max depth", 1, 3, 1, p.maxDepth) +
      toggle("showWeights", "Show sample weights", p.showWeights) + toggle("showContribution", "Show learner contribution", p.showContribution);
    if (type === "gradient") return range("estimators", "Number of estimators", 1, 100, 1, p.estimators) +
      range("learningRate", "Learning rate", .01, 1, .01, p.learningRate) + range("maxDepth", "Max depth", 1, 4, 1, p.maxDepth) +
      range("subsample", "Subsample ratio", .5, 1, .05, p.subsample);
    return range("estimators", "Number of estimators", 1, 100, 1, p.estimators) +
      range("learningRate", "Learning rate / eta", .01, 1, .01, p.learningRate) + range("maxDepth", "Max depth", 1, 5, 1, p.maxDepth) +
      range("lambda", "L2 regularization lambda", 0, 10, .25, p.lambda) + range("minChildWeight", "Min child weight", 1, 10, 1, p.minChildWeight) +
      range("subsample", "Subsample ratio", .5, 1, .05, p.subsample) + range("colsample", "Column sample ratio", .5, 1, .05, p.colsample);
  }

  function renderParameters() {
    $("parameterFields").innerHTML = parameterSchema();
    $("parameterFields").querySelectorAll("[data-param]").forEach(el => {
      el.addEventListener("input", () => {
        const key = el.dataset.param;
        state.params[state.modelType][key] = el.type === "checkbox" ? el.checked :
          el.tagName === "SELECT" ? el.value : Number(el.value);
        const valueLabel = document.querySelector(`[data-value="${key}"]`);
        if (valueLabel) valueLabel.textContent = el.value;
        updateParamSummary();
        state.trained = false; state.model = null; state.metrics = null; state.regionImage = null;
        updateStatus();
        window.clearTimeout(renderParameters.timer);
        renderParameters.timer = window.setTimeout(trainModel, 180);
      });
      el.addEventListener("change", () => { if (el.tagName === "SELECT" || el.type === "checkbox") trainModel(); });
    });
  }

  function updateParamSummary() {
    const p = state.params[state.modelType];
    if (state.modelType === "tree") $("paramSummary").textContent = `depth ${p.maxDepth}`;
    else $("paramSummary").textContent = `${p.trees || p.estimators} trees · depth ${p.maxDepth}`;
  }

  function updateMetricsUI() {
    const m = state.metrics;
    $("metricGrid").innerHTML = [
      ["Accuracy", m ? pct(m.accuracy) : "—"], ["Precision", m ? pct(m.precision) : "—"],
      ["Recall", m ? pct(m.recall) : "—"], ["F1 score", m ? pct(m.f1) : "—"],
      ["Points", state.points.length], ["Complexity", state.model ? state.model.complexity : "—"]
    ].map(([k, v]) => `<div class="metric-tile"><small>${k}</small><strong>${v}</strong></div>`).join("");
    let extra = "Train the model to reveal ensemble-specific diagnostics.";
    if (state.modelType === "forest" && state.model) extra = `<b>${state.model.trees.length} trees</b><br>Average tree depth: ${fmt(state.model.avgDepth, 1)}`;
    if (state.modelType === "adaboost" && state.model) {
      const err = state.model.learners.at(-1)?.error ?? 0;
      const alphas = state.model.learners.slice(0, 5).map(l => fmt(l.alpha, 2)).join(", ");
      extra = `<b>Weighted training error:</b> ${pct(err)}<br><b>First learner weights:</b> ${alphas}`;
    }
    if ((state.modelType === "gradient" || state.modelType === "xgboost") && state.model) {
      extra = `<b>Final log loss:</b> ${fmt(state.model.losses.at(-1), 3)}<br>Loss is plotted in the training story card.`;
    }
    $("metricExtra").innerHTML = `<div class="metric-extra">${extra}</div>`;
    $("accuracyLabel").textContent = m ? `${pct(m.accuracy)} accuracy` : "Not trained";
    ["tp", "tn", "fp", "fn"].forEach(k => $(`${k}Value`).textContent = m ? m[k] : 0);
  }
  function pct(v) { return `${fmt(v * 100, 1)}%`; }

  function updateEnsemblePanel() {
    $("insightText").textContent = EXPLAINS[state.modelType];
    const visual = $("ensembleVisual");
    if (state.modelType === "tree") {
      $("insightTitle").textContent = "One tree, one opinion";
      visual.innerHTML = `<div class="flow-steps">
        <div class="flow-step"><span class="num">1</span><span>Try candidate splits</span><small>Gini</small></div>
        <div class="flow-step"><span class="num">2</span><span>Keep the purest split</span><small>Recursive</small></div>
        <div class="flow-step"><span class="num">3</span><span>Predict from a leaf</span><small>Class vote</small></div>
      </div>`;
    } else if (state.modelType === "forest") {
      $("insightTitle").textContent = "A forest is a committee";
      visual.innerHTML = `<div class="tree-previews" id="treePreviews"></div>
        <div class="flow-step"><span class="num">Σ</span><span>Final majority vote</span><small>${state.model ? state.model.trees.length : state.params.forest.trees} voices</small></div>`;
      renderTreePreviews();
    } else if (state.modelType === "adaboost") {
      $("insightTitle").textContent = "Learn from hard examples";
      const dots = state.points.slice(0, 30).map((p, i) => {
        const w = state.model?.weights?.[i] || 1 / Math.max(1, state.points.length);
        const size = clamp(4 + Math.sqrt(w * state.points.length) * 7, 5, 20);
        const left = 5 + ((p.x - state.bounds.xmin) / (state.bounds.xmax - state.bounds.xmin)) * 88;
        const top = 5 + ((state.bounds.ymax - p.y) / (state.bounds.ymax - state.bounds.ymin)) * 82;
        return `<i class="weight-dot" style="width:${size}px;height:${size}px;left:${left}%;top:${top}%;background:${p.label ? "#3157d5" : "#e74c5b"}"></i>`;
      }).join("");
      visual.innerHTML = `<div class="weight-dots">${dots}</div><div class="flow-steps">
        <div class="flow-step"><span class="num">1</span><span>Train a weak learner</span><small>Shallow tree</small></div>
        <div class="flow-step"><span class="num">2</span><span>Increase missed-point weights</span><small>Refocus</small></div>
        <div class="flow-step"><span class="num">3</span><span>Combine weighted learners</span><small>α vote</small></div>
      </div>`;
    } else if (state.modelType === "gradient") {
      $("insightTitle").textContent = "Correct mistakes in sequence";
      visual.innerHTML = `<div class="flow-steps">
        <div class="flow-step"><span class="num">0</span><span>Initial prediction</span><small>Class prior</small></div>
        <div class="flow-step"><span class="num">1</span><span>Measure residual errors</span><small>y − p</small></div>
        <div class="flow-step"><span class="num">2</span><span>Fit the next correction</span><small>Small tree</small></div>
        <div class="flow-step"><span class="num">Σ</span><span>Add every correction</span><small>Final model</small></div>
      </div>`;
    } else {
      $("insightTitle").textContent = "Boost, then regularize";
      const counts = state.model ? state.model.trees.reduce((acc, t) => {
        const c = treeFeatureCounts(t); acc[0] += c[0]; acc[1] += c[1]; return acc;
      }, [0, 0]) : [1, 1];
      const total = counts[0] + counts[1] || 1;
      visual.innerHTML = `<p class="hint">This educational model borrows shrinkage, row/column sampling, minimum child size, and L2 leaf regularization from XGBoost.</p>
        <div class="feature-bar"><label><span>X feature importance</span><b>${pct(counts[0] / total)}</b></label><div class="feature-track"><div class="feature-fill" style="width:${counts[0] / total * 100}%"></div></div></div>
        <div class="feature-bar"><label><span>Y feature importance</span><b>${pct(counts[1] / total)}</b></label><div class="feature-track"><div class="feature-fill" style="width:${counts[1] / total * 100}%"></div></div></div>
        <p class="hint">Lower eta and higher lambda usually make the boundary change more gradually.</p>`;
    }
  }

  function renderTreePreviews() {
    const host = $("treePreviews");
    if (!host) return;
    const trees = state.model?.trees?.slice(0, 6) || [];
    host.innerHTML = trees.length ? trees.map((_, i) => `<div class="tree-mini"><canvas data-tree="${i}"></canvas><span>Tree ${i + 1}</span></div>`).join("")
      : Array.from({ length: 6 }, (_, i) => `<div class="tree-mini"><span>Tree ${i + 1}</span></div>`).join("");
    host.querySelectorAll("canvas").forEach(c => {
      const tr = trees[Number(c.dataset.tree)], size = 78, dpr = window.devicePixelRatio || 1;
      c.width = size * dpr; c.height = size * dpr;
      const cx = c.getContext("2d"); cx.scale(dpr, dpr);
      const grid = 20, cw = size / grid;
      for (let y = 0; y < grid; y++) for (let x = 0; x < grid; x++) {
        const dx = state.bounds.xmin + x / grid * (state.bounds.xmax - state.bounds.xmin);
        const dy = state.bounds.ymax - y / grid * (state.bounds.ymax - state.bounds.ymin);
        cx.fillStyle = treeProb(tr, dx, dy) >= .5 ? "rgba(49,87,213,.28)" : "rgba(231,76,91,.28)";
        cx.fillRect(x * cw, y * cw, cw + 1, cw + 1);
      }
    });
  }

  function updateSelectedReadout(point) {
    if (!state.model) return;
    const pred = predict(point.x, point.y);
    let text = `Predicted Class ${pred.label} with ${fmt((pred.label ? pred.proba : 1 - pred.proba) * 100, 0)}% confidence.`;
    if (state.modelType === "forest") {
      const v1 = state.model.trees.reduce((s, t) => s + (treeProb(t, point.x, point.y) >= .5), 0);
      text = `${v1} trees vote blue; ${state.model.trees.length - v1} vote red. Final: Class ${pred.label}.`;
    } else if (state.modelType === "adaboost") {
      text = `Weighted learner score: ${fmt(state.model.predictScore(point.x, point.y), 2)}. Final: Class ${pred.label}.`;
    } else if (state.modelType === "gradient" || state.modelType === "xgboost") {
      text = `Sequential score: ${fmt(state.model.predictScore(point.x, point.y), 2)}. Probability of Class 1: ${pct(pred.proba)}.`;
    }
    $("selectedReadout").innerHTML = `<span class="readout-icon">${pred.label}</span><div><small>Ensemble decision</small><p>${text}</p></div>`;
  }

  function drawMiniChart() {
    const rect = chart.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (!rect.width) return;
    chart.width = rect.width * dpr; chart.height = rect.height * dpr;
    chartCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width, h = rect.height;
    chartCtx.clearRect(0, 0, w, h);
    chartCtx.strokeStyle = "#e4e8f0"; chartCtx.lineWidth = 1;
    for (let i = 1; i < 4; i++) { chartCtx.beginPath(); chartCtx.moveTo(28, i * h / 4); chartCtx.lineTo(w - 8, i * h / 4); chartCtx.stroke(); }
    let values = [];
    if (state.model?.losses?.length) values = state.model.losses;
    else if (state.modelType === "forest" && state.model) {
      values = state.model.trees.map((_, n) => {
        let correct = 0;
        state.points.forEach(p => {
          const v = state.model.trees.slice(0, n + 1).reduce((s, t) => s + (treeProb(t, p.x, p.y) >= .5), 0);
          if ((v / (n + 1) >= .5 ? 1 : 0) === p.label) correct++;
        });
        return correct / state.points.length;
      });
    } else if (state.modelType === "tree" && state.model) {
      values = [0.5, state.metrics?.accuracy || .5];
    }
    if (!values.length) {
      chartCtx.fillStyle = "#99a2b4"; chartCtx.font = "11px DM Sans"; chartCtx.textAlign = "center";
      chartCtx.fillText("Train the model to reveal its learning curve", w / 2, h / 2);
      return;
    }
    const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
    chartCtx.beginPath();
    values.forEach((v, i) => {
      const x = 28 + i / Math.max(1, values.length - 1) * (w - 42);
      const y = 12 + (max - v) / span * (h - 34);
      i ? chartCtx.lineTo(x, y) : chartCtx.moveTo(x, y);
    });
    chartCtx.strokeStyle = "#3157d5"; chartCtx.lineWidth = 2.5; chartCtx.stroke();
    chartCtx.lineTo(w - 14, h - 18); chartCtx.lineTo(28, h - 18); chartCtx.closePath();
    chartCtx.fillStyle = "rgba(49,87,213,.08)"; chartCtx.fill();
  }

  function updateChartText() {
    const boosted = state.modelType === "gradient" || state.modelType === "xgboost" || state.modelType === "adaboost";
    $("chartTitle").textContent = boosted ? "Training loss curve" : state.modelType === "forest" ? "Forest growth" : "Learning snapshot";
    $("chartCaption").textContent = boosted
      ? "Each new learner aims to reduce the remaining error. A flattening curve means extra learners are adding less."
      : state.modelType === "forest"
        ? "Training accuracy as more tree votes join the forest."
        : "A single tree reaches its result in one recursive fitting pass.";
  }

  function updateTable() {
    $("tableCount").textContent = `(${state.points.length} points)`;
    const body = $("dataTableBody");
    body.innerHTML = state.points.map((p, i) => {
      const pr = state.trained && state.metrics ? state.metrics.predictions[i].label : null;
      return `<tr data-id="${p.id}" class="${state.selected.has(p.id) ? "selected-row" : ""}">
        <td>${i + 1}</td><td>${fmt(p.x, 3)}</td><td>${fmt(p.y, 3)}</td>
        <td><select class="class-select" data-class-id="${p.id}"><option value="0" ${p.label === 0 ? "selected" : ""}>Class 0</option><option value="1" ${p.label === 1 ? "selected" : ""}>Class 1</option></select></td>
        <td>${pr === null ? "—" : `<span class="pred-pill c${pr}">Class ${pr}</span>`}</td>
        <td><button class="row-delete" data-delete-id="${p.id}">Delete</button></td>
      </tr>`;
    }).join("");
    body.querySelectorAll("[data-class-id]").forEach(el => el.addEventListener("change", () => {
      const p = state.points.find(q => q.id === Number(el.dataset.classId));
      if (p) { p.label = Number(el.value); invalidateAndRefresh(); trainModel(); }
    }));
    body.querySelectorAll("[data-delete-id]").forEach(el => el.addEventListener("click", () => {
      state.points = state.points.filter(p => p.id !== Number(el.dataset.deleteId));
      invalidateAndRefresh(); trainModel();
    }));
    body.querySelectorAll("tr[data-id]").forEach(row => row.addEventListener("click", (e) => {
      if (e.target.matches("select,button,option")) return;
      const id = Number(row.dataset.id);
      state.selected.has(id) ? state.selected.delete(id) : state.selected.add(id);
      updateTable(); drawPlot();
    }));
  }

  function updateStatus() {
    const status = $("plotStatus"), msg = $("canvasMessage");
    if (state.trained) {
      status.className = "plot-status"; status.innerHTML = "<span></span> Model trained"; msg.classList.add("hidden");
    } else {
      status.className = "plot-status untrained"; status.innerHTML = "<span></span> Needs update";
      msg.textContent = state.points.length ? "The data or settings changed. Select “Train / update” to redraw the model." : "Add points or load a dataset to begin.";
      msg.classList.remove("hidden");
    }
  }
  function showWarning(text) { $("trainingWarning").textContent = text; $("trainingWarning").classList.remove("hidden"); }
  function hideWarning() { $("trainingWarning").classList.add("hidden"); }

  function updateAll() {
    $("datasetLabel").textContent = DATASET_NAMES[state.dataset];
    $("modelLabel").textContent = MODEL_NAMES[state.modelType];
    $("heroModel").textContent = MODEL_NAMES[state.modelType];
    $("plotTitle").textContent = `${MODEL_NAMES[state.modelType]} on ${DATASET_NAMES[state.dataset]}`;
    $("modelQuickExplain").textContent = EXPLAINS[state.modelType];
    updateParamSummary(); updateStatus(); updateMetricsUI(); updateEnsemblePanel(); updateChartText(); updateTable(); drawPlot(); drawMiniChart();
  }

  function loadCurrentDataset() {
    state.points = generateDataset(state.dataset);
    state.originalPoints = state.points.map(p => ({ ...p }));
    state.selected.clear(); state.confusionFilter = null;
    state.bounds = calculateBounds();
    trainModel();
  }

  document.querySelectorAll(".menu-trigger").forEach(button => button.addEventListener("click", (e) => {
    e.stopPropagation();
    const id = button.dataset.menu, menu = $(id), wasOpen = menu.classList.contains("open");
    document.querySelectorAll(".popover").forEach(p => p.classList.remove("open"));
    document.querySelectorAll(".menu-trigger").forEach(b => b.classList.remove("open"));
    if (!wasOpen) { menu.classList.add("open"); button.classList.add("open"); }
  }));
  document.querySelectorAll(".popover").forEach(p => p.addEventListener("click", e => e.stopPropagation()));
  document.addEventListener("click", () => {
    document.querySelectorAll(".popover").forEach(p => p.classList.remove("open"));
    document.querySelectorAll(".menu-trigger").forEach(b => b.classList.remove("open"));
  });
  document.querySelectorAll(".side-dropdown").forEach(panel => panel.addEventListener("toggle", () => {
    if (!panel.open) return;
    document.querySelectorAll(".side-dropdown").forEach(other => {
      if (other !== panel) other.open = false;
    });
    requestAnimationFrame(() => {
      resizeCanvas();
      drawMiniChart();
    });
  }));
  $("datasetSelect").addEventListener("change", e => { state.dataset = e.target.value; $("datasetLabel").textContent = DATASET_NAMES[state.dataset]; });
  $("loadDataset").addEventListener("click", loadCurrentDataset);
  $("modelSelect").addEventListener("change", e => {
    state.modelType = e.target.value;
    state.model = null;
    state.metrics = null;
    state.trained = false;
    state.regionImage = null;
    renderParameters();
    updateAll();
    trainModel();
  });
  $("trainButton").addEventListener("click", trainModel);
  $("resetDataset").addEventListener("click", () => {
    state.points = state.originalPoints.map(p => ({ ...p, id: state.nextId++ }));
    state.selected.clear(); trainModel();
  });
  $("brushRadius").addEventListener("input", e => {
    state.brushRadius = Number(e.target.value); $("brushRadiusValue").textContent = `${state.brushRadius} px`;
  });
  $("toolButtons").querySelectorAll("[data-tool]").forEach(button => button.addEventListener("click", () => {
    state.tool = button.dataset.tool;
    $("toolButtons").querySelectorAll(".tool").forEach(b => b.classList.toggle("active", b === button));
    const labels = { add: "Add point", brush1: "Brush blue", brush0: "Brush red", erase: "Eraser", select: "Select region" };
    $("toolLabel").textContent = labels[state.tool];
  }));
  function deleteSelected() {
    state.points = state.points.filter(p => !state.selected.has(p.id));
    state.selected.clear(); invalidateAndRefresh(); if (state.points.length >= 4) trainModel();
  }
  function clearData() { state.points = []; state.selected.clear(); invalidateAndRefresh(); }
  $("deleteSelected").addEventListener("click", deleteSelected);
  $("tableDeleteSelected").addEventListener("click", deleteSelected);
  $("clearData").addEventListener("click", clearData);
  $("tableClear").addEventListener("click", clearData);
  document.querySelectorAll(".matrix-cell").forEach(cell => cell.addEventListener("click", () => {
    if (!state.trained) return;
    state.confusionFilter = cell.dataset.cell;
    document.querySelectorAll(".matrix-cell").forEach(c => c.classList.toggle("active", c === cell));
    const text = { tp: "Showing true positives: actual 1, predicted 1.", fp: "Showing false positives: actual 0, predicted 1.", fn: "Showing false negatives: actual 1, predicted 0.", tn: "Showing true negatives: actual 0, predicted 0." };
    $("matrixExplanation").textContent = text[state.confusionFilter]; drawPlot();
  }));
  $("showAll").addEventListener("click", () => {
    state.confusionFilter = null; document.querySelectorAll(".matrix-cell").forEach(c => c.classList.remove("active"));
    $("matrixExplanation").textContent = "Click a cell to isolate those points on the plot."; drawPlot();
  });
  $("predictButton").addEventListener("click", () => {
    const x = Number($("predictX").value), y = Number($("predictY").value), out = $("predictionOutput");
    if (!state.trained) { out.innerHTML = `<div class="prediction-class neutral">?</div><div><small>Model not trained</small><p>Train the model before requesting a prediction.</p></div>`; return; }
    const pr = predict(x, y), confidence = (pr.label ? pr.proba : 1 - pr.proba);
    let explanation = `The model assigns ${pct(confidence)} confidence to Class ${pr.label}.`;
    if (state.modelType === "forest") {
      const v1 = state.model.trees.reduce((s, t) => s + (treeProb(t, x, y) >= .5), 0);
      explanation = `${v1} out of ${state.model.trees.length} trees voted for Class 1, so the forest predicts Class ${pr.label}.`;
    } else if (state.modelType === "adaboost") {
      explanation = `The weighted weak-learner score is ${fmt(state.model.predictScore(x, y), 2)}, producing Class ${pr.label}.`;
    } else if (state.modelType === "gradient" || state.modelType === "xgboost") {
      explanation = `Sequential corrections produce a Class 1 probability of ${pct(pr.proba)}.`;
    }
    out.innerHTML = `<div class="prediction-class ${pr.label ? "" : "red"}">${pr.label}</div><div><small>Predicted Class ${pr.label} · ${pct(confidence)}</small><p>${explanation}</p></div>`;
  });

  const questions = [
    ["Why does a single decision tree overfit more easily than a random forest?", "A single tree can memorize small irregularities. A forest averages many diverse trees, reducing variance and making isolated quirks less influential."],
    ["What happens when the number of trees in a random forest increases?", "Predictions usually become more stable and then level off. Computation grows, but overfitting generally does not increase in the same way as tree depth."],
    ["Why does AdaBoost focus more on misclassified points?", "Increasing their weights forces the next weak learner to spend more attention on examples the current ensemble finds difficult."],
    ["What is the effect of the learning rate in boosting?", "It scales each learner's contribution. Smaller values learn more gradually and often need more estimators; larger values move faster but can overfit or become unstable."],
    ["How does max depth affect bias and variance?", "Shallow trees have higher bias and lower variance. Deep trees capture more detail, reducing bias but increasing variance and overfitting risk."],
    ["Why can two models have similar accuracy but very different decision regions?", "Accuracy only checks the observed points. Models can agree on those points while behaving very differently in the unobserved space between them."],
    ["What is the difference between bagging and boosting?", "Bagging trains learners independently on varied samples and averages them. Boosting trains learners sequentially so each one corrects earlier errors."],
    ["Why does regularization help in XGBoost-style models?", "It discourages extreme or overly complex corrections, which can smooth the boundary and improve generalization to new data."]
  ];
  $("questionList").innerHTML = questions.map(q => `<li>${q[0]}</li>`).join("");
  $("answerList").innerHTML = questions.map(q => `<li>${q[1]}</li>`).join("");

  window.addEventListener("resize", () => { window.clearTimeout(resizeCanvas.timer); resizeCanvas.timer = window.setTimeout(resizeCanvas, 80); });
  document.addEventListener("keydown", e => {
    if ((e.key === "Delete" || e.key === "Backspace") && state.selected.size && !/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) deleteSelected();
  });

  const extraStyle = document.createElement("style");
  extraStyle.textContent = `.toggle-row{display:flex;align-items:center;justify-content:space-between;margin-top:12px;color:#566078;font-size:12px;font-weight:700}.toggle-row input{width:17px;height:17px;accent-color:#3157d5}.field-label.spaced{margin-top:14px}`;
  document.head.appendChild(extraStyle);

  renderParameters();
  requestAnimationFrame(() => {
    resizeCanvas();
    loadCurrentDataset();
  });
})();
