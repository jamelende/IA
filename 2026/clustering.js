(() => {
  "use strict";

  const $ = id => document.getElementById(id);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const fmt = (v, d = 2) => Number(v).toFixed(d);
  const COLORS = ["#5268d9", "#ef6672", "#28a879", "#e69a32", "#9b64d5", "#2e9eb5", "#d65aa0", "#708238"];
  const SOFT = ["rgba(82,104,217,.16)", "rgba(239,102,114,.16)", "rgba(40,168,121,.16)", "rgba(230,154,50,.16)", "rgba(155,100,213,.16)", "rgba(46,158,181,.16)"];
  const MODEL_NAMES = { kmeans: "k-Means", hierarchical: "Hierarchical Agglomerative", dbscan: "DBSCAN", gmm: "Gaussian Mixture Model" };
  const DATASET_NAMES = { blobs: "Gaussian blobs", moons: "Two moons", circles: "Concentric circles", anisotropic: "Anisotropic groups", varied: "Varied density", spiral: "Spiral arms", noise: "Clusters with noise", uniform: "Uniform random points" };
  const EXPLAINS = {
    kmeans: "Alternates between assigning every point to its nearest centroid and moving each centroid to the mean of its assigned points.",
    hierarchical: "Starts with one cluster per point, then repeatedly merges the two closest clusters to build a hierarchy from the bottom up.",
    dbscan: "Grows clusters from dense core points. Sparse points that cannot be reached from a dense region are labeled as noise.",
    gmm: "Treats the data as a mixture of Gaussian distributions and alternates soft assignment with updates to each Gaussian."
  };
  const TITLES = { kmeans: "Assign, move, repeat", hierarchical: "Merge the closest groups", dbscan: "Grow dense neighborhoods", gmm: "Estimate soft Gaussian groups" };

  class RNG {
    constructor(seed = 20260609) { this.s = seed >>> 0; }
    next() { this.s = (1664525 * this.s + 1013904223) >>> 0; return this.s / 4294967296; }
    normal() { const u = Math.max(this.next(), 1e-9), v = this.next(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
    pick(n) { return Math.floor(this.next() * n); }
  }

  const defaults = {
    kmeans: { k: 3, maxIterations: 15, initialization: "plusplus" },
    hierarchical: { clusters: 3, linkage: "average" },
    dbscan: { epsilon: .55, minPoints: 4 },
    gmm: { components: 3, iterations: 18, regularization: .08 }
  };
  const state = {
    dataset: "blobs", modelType: "kmeans", params: JSON.parse(JSON.stringify(defaults)),
    points: [], original: [], nextId: 1, history: [], step: 0, playing: false, timer: null,
    speed: 700, tool: "add", brushRadius: 24, selected: new Set(), dragging: false,
    dragStart: null, dragCurrent: null, hover: -1, bounds: { xmin: -4, xmax: 4, ymin: -4, ymax: 4 }
  };

  function generateDataset(type) {
    const rng = new RNG(12031 + Object.keys(DATASET_NAMES).indexOf(type) * 977), pts = [];
    const add = (x, y) => pts.push({ id: state.nextId++, x, y });
    if (type === "blobs") {
      const centers = [[-2, -1.2], [1.7, -1], [.2, 2]];
      centers.forEach((c, ci) => { for (let i = 0; i < 28; i++) add(c[0] + rng.normal() * (.55 + ci * .08), c[1] + rng.normal() * .55); });
    } else if (type === "moons") {
      for (let i = 0; i < 90; i++) { const c = i < 45 ? 0 : 1, a = rng.next() * Math.PI; add((c ? 1 - Math.cos(a) : Math.cos(a)) * 2 + rng.normal() * .12, (c ? -.55 - Math.sin(a) : Math.sin(a)) * 1.6 + rng.normal() * .12); }
    } else if (type === "circles") {
      for (let i = 0; i < 90; i++) { const outer = i >= 42, a = rng.next() * Math.PI * 2, r = (outer ? 2.3 : .9) + rng.normal() * .12; add(Math.cos(a) * r, Math.sin(a) * r); }
    } else if (type === "anisotropic") {
      [[-1.7, -1.2, .9, .22, .7], [1.5, 1.3, .75, .2, -.65], [1.6, -1.8, .55, .28, .3]].forEach(c => {
        for (let i = 0; i < 27; i++) { const a = rng.normal() * c[2], b = rng.normal() * c[3]; add(c[0] + a * Math.cos(c[4]) - b * Math.sin(c[4]), c[1] + a * Math.sin(c[4]) + b * Math.cos(c[4])); }
      });
    } else if (type === "varied") {
      [[-2, -1, .25, 20], [1.5, -1.1, .8, 38], [.2, 2, .45, 27]].forEach(c => { for (let i = 0; i < c[3]; i++) add(c[0] + rng.normal() * c[2], c[1] + rng.normal() * c[2]); });
    } else if (type === "spiral") {
      for (let arm = 0; arm < 3; arm++) for (let i = 0; i < 28; i++) { const r = .2 + i / 10, t = i / 6 + arm * Math.PI * 2 / 3; add(r * Math.cos(t) + rng.normal() * .09, r * Math.sin(t) + rng.normal() * .09); }
    } else if (type === "noise") {
      [[-1.7, -1], [1.5, 1.2], [1.8, -1.5]].forEach(c => { for (let i = 0; i < 22; i++) add(c[0] + rng.normal() * .35, c[1] + rng.normal() * .35); });
      for (let i = 0; i < 18; i++) add(rng.next() * 7 - 3.5, rng.next() * 6 - 3);
    } else {
      for (let i = 0; i < 85; i++) add(rng.next() * 7 - 3.5, rng.next() * 6 - 3);
    }
    return pts;
  }

  const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  function mean(indices) {
    if (!indices.length) return { x: 0, y: 0 };
    return { x: indices.reduce((s, i) => s + state.points[i].x, 0) / indices.length, y: indices.reduce((s, i) => s + state.points[i].y, 0) / indices.length };
  }
  function clusterRadii(assignments, centers) {
    return centers.map((c, k) => {
      const ds = state.points.map((p, i) => assignments[i] === k ? distance(p, c) : null).filter(v => v !== null).sort((a, b) => a - b);
      return ds.length ? ds[Math.floor(ds.length * .82)] + .15 : .2;
    });
  }

  function buildKMeansHistory() {
    const p = state.params.kmeans, rng = new RNG(6819), k = Math.min(p.k, state.points.length);
    let centers = [];
    if (p.initialization === "random") {
      const used = new Set();
      while (centers.length < k) { const i = rng.pick(state.points.length); if (!used.has(i)) { used.add(i); centers.push({ ...state.points[i] }); } }
    } else {
      centers.push({ ...state.points[rng.pick(state.points.length)] });
      while (centers.length < k) {
        const weights = state.points.map(pt => Math.min(...centers.map(c => distance(pt, c) ** 2)));
        let r = rng.next() * weights.reduce((a, b) => a + b, 0), chosen = 0;
        for (let i = 0; i < weights.length; i++) { r -= weights[i]; if (r <= 0) { chosen = i; break; } }
        centers.push({ ...state.points[chosen] });
      }
    }
    const unassigned = Array(state.points.length).fill(-2);
    const history = [{ phase: "initialize", assignments: unassigned, centers: centers.map(c => ({ x: c.x, y: c.y })), objective: null, message: `Placed ${k} initial centroids using ${p.initialization === "plusplus" ? "k-means++" : "random selection"}.` }];
    let previous = null;
    for (let iter = 1; iter <= p.maxIterations; iter++) {
      const assignments = state.points.map(pt => {
        let best = 0, bestD = Infinity;
        centers.forEach((c, ci) => { const d = distance(pt, c); if (d < bestD) { bestD = d; best = ci; } });
        return best;
      });
      const objective = state.points.reduce((s, pt, i) => s + distance(pt, centers[assignments[i]]) ** 2, 0);
      history.push({ phase: "assign", iteration: iter, assignments: assignments.slice(), centers: centers.map(c => ({ ...c })), radii: clusterRadii(assignments, centers), objective, message: `Assignment step ${iter}: every point joins its nearest centroid.` });
      const next = centers.map((c, ci) => { const ids = assignments.map((a, i) => a === ci ? i : -1).filter(i => i >= 0); return ids.length ? mean(ids) : c; });
      const movement = next.reduce((s, c, i) => s + distance(c, centers[i]), 0);
      centers = next;
      history.push({ phase: "update", iteration: iter, assignments: assignments.slice(), centers: centers.map(c => ({ ...c })), radii: clusterRadii(assignments, centers), objective, message: `Update step ${iter}: centroids move to the mean of their assigned points.` });
      if (previous && assignments.every((a, i) => a === previous[i]) || movement < 1e-4) {
        history.push({ phase: "complete", iteration: iter, assignments: assignments.slice(), centers: centers.map(c => ({ ...c })), radii: clusterRadii(assignments, centers), objective, message: `Converged after ${iter} iterations. Assignments no longer change.` });
        break;
      }
      previous = assignments;
    }
    return history;
  }

  function linkageDistance(a, b, mode) {
    const pairs = [];
    a.forEach(i => b.forEach(j => pairs.push(distance(state.points[i], state.points[j]))));
    if (mode === "single") return Math.min(...pairs);
    if (mode === "complete") return Math.max(...pairs);
    return pairs.reduce((x, y) => x + y, 0) / pairs.length;
  }
  function hierarchicalState(clusters, phase, message, merge = null, level = 0) {
    const assignments = Array(state.points.length).fill(-2);
    clusters.forEach((c, ci) => c.forEach(i => assignments[i] = ci));
    const centers = clusters.map(mean), radii = clusterRadii(assignments, centers);
    return { phase, assignments, centers, radii, clusters: clusters.map(c => c.slice()), merge, level, objective: level, message };
  }
  function buildHierarchicalHistory() {
    const target = Math.min(state.params.hierarchical.clusters, state.points.length), mode = state.params.hierarchical.linkage;
    let clusters = state.points.map((_, i) => [i]);
    const history = [hierarchicalState(clusters, "initialize", `Begin with ${clusters.length} singleton clusters.`)];
    while (clusters.length > target) {
      let best = null;
      for (let i = 0; i < clusters.length; i++) for (let j = i + 1; j < clusters.length; j++) {
        const d = linkageDistance(clusters[i], clusters[j], mode);
        if (!best || d < best.d) best = { i, j, d };
      }
      const ca = mean(clusters[best.i]), cb = mean(clusters[best.j]), merged = clusters[best.i].concat(clusters[best.j]);
      clusters = clusters.filter((_, i) => i !== best.i && i !== best.j); clusters.push(merged);
      history.push(hierarchicalState(clusters, "merge", `Merge the closest pair using ${mode} linkage. ${clusters.length} clusters remain.`, { a: ca, b: cb }, best.d));
    }
    history.push(hierarchicalState(clusters, "complete", `Stopped at the requested ${target} clusters.`, null, history.at(-1)?.level || 0));
    return history;
  }

  function buildDBSCANHistory() {
    const { epsilon, minPoints } = state.params.dbscan, n = state.points.length;
    const neighbors = state.points.map((p, i) => state.points.map((q, j) => distance(p, q) <= epsilon ? j : -1).filter(j => j >= 0));
    const core = neighbors.map(ns => ns.length >= minPoints), labels = Array(n).fill(-2), visited = Array(n).fill(false);
    const history = [{ phase: "initialize", assignments: labels.slice(), core, active: null, objective: 0, message: `No points visited. A core point needs at least ${minPoints} points inside ε = ${epsilon}.` }];
    let cluster = 0;
    for (let i = 0; i < n; i++) {
      if (visited[i]) continue;
      visited[i] = true;
      if (!core[i]) {
        labels[i] = -1;
        history.push({ phase: "noise", assignments: labels.slice(), core, active: i, epsilon, objective: cluster, message: `Point ${i + 1} is not dense enough, so it is temporarily marked as noise.` });
        continue;
      }
      labels[i] = cluster;
      const queue = neighbors[i].slice(), queued = new Set(queue);
      history.push({ phase: "seed", assignments: labels.slice(), core, active: i, epsilon, objective: cluster + 1, message: `Core point ${i + 1} starts cluster ${cluster + 1}.` });
      while (queue.length) {
        const j = queue.shift();
        if (!visited[j]) {
          visited[j] = true;
          if (core[j]) neighbors[j].forEach(q => { if (!queued.has(q)) { queued.add(q); queue.push(q); } });
        }
        if (labels[j] < 0) labels[j] = cluster;
      }
      history.push({ phase: "expand", assignments: labels.slice(), core, active: i, epsilon, objective: cluster + 1, message: `Expanded all density-reachable points into cluster ${cluster + 1}.` });
      cluster++;
    }
    history.push({ phase: "complete", assignments: labels.slice(), core, active: null, objective: cluster, message: `DBSCAN found ${cluster} clusters and ${labels.filter(v => v === -1).length} noise points.` });
    return history;
  }

  function gaussian(pt, mean, variance) {
    const vx = Math.max(variance.x, 1e-5), vy = Math.max(variance.y, 1e-5);
    return Math.exp(-.5 * ((pt.x - mean.x) ** 2 / vx + (pt.y - mean.y) ** 2 / vy)) / (2 * Math.PI * Math.sqrt(vx * vy));
  }
  function buildGMMHistory() {
    const p = state.params.gmm, k = Math.min(p.components, state.points.length), rng = new RNG(7781);
    let means = [], used = new Set();
    while (means.length < k) { const i = rng.pick(state.points.length); if (!used.has(i)) { used.add(i); means.push({ x: state.points[i].x, y: state.points[i].y }); } }
    let variances = Array.from({ length: k }, () => ({ x: 1, y: 1 })), weights = Array(k).fill(1 / k);
    let responsibilities = state.points.map(() => Array(k).fill(1 / k));
    const history = [{ phase: "initialize", assignments: Array(state.points.length).fill(-2), means: means.map(m => ({ ...m })), variances: variances.map(v => ({ ...v })), weights: weights.slice(), objective: null, message: `Placed ${k} initial Gaussian components.` }];
    for (let iter = 1; iter <= p.iterations; iter++) {
      let logLikelihood = 0;
      responsibilities = state.points.map(pt => {
        const raw = means.map((m, j) => weights[j] * gaussian(pt, m, variances[j]));
        const total = raw.reduce((a, b) => a + b, 0) || 1e-12; logLikelihood += Math.log(total);
        return raw.map(v => v / total);
      });
      const assignments = responsibilities.map(r => r.indexOf(Math.max(...r)));
      history.push({ phase: "expectation", iteration: iter, assignments, responsibilities: responsibilities.map(r => r.slice()), means: means.map(m => ({ ...m })), variances: variances.map(v => ({ ...v })), weights: weights.slice(), objective: logLikelihood, message: `Expectation step ${iter}: compute each point's probability for every Gaussian.` });
      for (let j = 0; j < k; j++) {
        const mass = responsibilities.reduce((s, r) => s + r[j], 0) || 1e-9;
        means[j] = { x: state.points.reduce((s, pt, i) => s + responsibilities[i][j] * pt.x, 0) / mass, y: state.points.reduce((s, pt, i) => s + responsibilities[i][j] * pt.y, 0) / mass };
        variances[j] = {
          x: state.points.reduce((s, pt, i) => s + responsibilities[i][j] * (pt.x - means[j].x) ** 2, 0) / mass + p.regularization,
          y: state.points.reduce((s, pt, i) => s + responsibilities[i][j] * (pt.y - means[j].y) ** 2, 0) / mass + p.regularization
        };
        weights[j] = mass / state.points.length;
      }
      history.push({ phase: "maximization", iteration: iter, assignments, responsibilities: responsibilities.map(r => r.slice()), means: means.map(m => ({ ...m })), variances: variances.map(v => ({ ...v })), weights: weights.slice(), objective: logLikelihood, message: `Maximization step ${iter}: move and reshape each Gaussian using soft assignments.` });
    }
    const assignments = responsibilities.map(r => r.indexOf(Math.max(...r)));
    history.push({ phase: "complete", assignments, responsibilities, means, variances, weights, objective: history.at(-1).objective, message: `Completed ${p.iterations} EM iterations.` });
    return history;
  }

  function initialize() {
    stopPlaying();
    if (state.points.length < 2) { showWarning("Add at least two points to initialize clustering."); return; }
    hideWarning();
    if (state.modelType === "kmeans") state.history = buildKMeansHistory();
    else if (state.modelType === "hierarchical") state.history = buildHierarchicalHistory();
    else if (state.modelType === "dbscan") state.history = buildDBSCANHistory();
    else state.history = buildGMMHistory();
    state.step = 0; updateAll();
  }
  function current() { return state.history[state.step] || null; }
  function advance() { if (!state.history.length) initialize(); else if (state.step < state.history.length - 1) { state.step++; updateAll(); } else stopPlaying(); }
  function back() { stopPlaying(); if (state.step > 0) { state.step--; updateAll(); } }
  function togglePlay() {
    if (!state.history.length) initialize();
    if (state.playing) return stopPlaying();
    state.playing = true; $("playButton").textContent = "❚❚ Pause";
    const tick = () => { if (!state.playing) return; if (state.step >= state.history.length - 1) { stopPlaying(); return; } state.step++; updateAll(); state.timer = setTimeout(tick, state.speed); };
    tick();
  }
  function stopPlaying() { state.playing = false; clearTimeout(state.timer); if ($("playButton")) $("playButton").textContent = "▶ Play"; }

  function calculateBounds() {
    if (!state.points.length) return { xmin: -4, xmax: 4, ymin: -4, ymax: 4 };
    const xs = state.points.map(p => p.x), ys = state.points.map(p => p.y);
    const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
    const xp = Math.max(.7, (xmax - xmin) * .15), yp = Math.max(.7, (ymax - ymin) * .15);
    return { xmin: xmin - xp, xmax: xmax + xp, ymin: ymin - yp, ymax: ymax + yp };
  }
  const canvas = $("plotCanvas"), ctx = canvas.getContext("2d"), chart = $("miniChart"), chartCtx = chart.getContext("2d");
  const PAD = { l: 53, r: 18, t: 18, b: 42 };
  function toCanvas(x, y) {
    const b = state.bounds, w = canvas.clientWidth, h = canvas.clientHeight;
    return { x: PAD.l + (x - b.xmin) / (b.xmax - b.xmin) * (w - PAD.l - PAD.r), y: PAD.t + (b.ymax - y) / (b.ymax - b.ymin) * (h - PAD.t - PAD.b) };
  }
  function toData(x, y) {
    const b = state.bounds, w = canvas.clientWidth, h = canvas.clientHeight;
    return { x: b.xmin + (x - PAD.l) / (w - PAD.l - PAD.r) * (b.xmax - b.xmin), y: b.ymax - (y - PAD.t) / (h - PAD.t - PAD.b) * (b.ymax - b.ymin) };
  }
  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1, r = canvas.getBoundingClientRect(), w = Math.max(320, Math.floor(r.width)), h = Math.max(300, Math.floor(r.height));
    canvas.width = w * dpr; canvas.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); drawPlot(); drawChart();
  }
  function niceStep(v) { const p = 10 ** Math.floor(Math.log10(v)), n = v / p; return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * p; }
  function drawGrid() {
    const w = canvas.clientWidth, h = canvas.clientHeight, b = state.bounds;
    ctx.fillStyle = "#fbfcfe"; ctx.fillRect(0, 0, w, h); ctx.strokeStyle = "rgba(106,119,145,.13)"; ctx.fillStyle = "#8993a6"; ctx.font = "10px DM Sans";
    const xs = niceStep((b.xmax - b.xmin) / 7), ys = niceStep((b.ymax - b.ymin) / 6);
    for (let x = Math.ceil(b.xmin / xs) * xs; x <= b.xmax; x += xs) { const q = toCanvas(x, 0); ctx.beginPath(); ctx.moveTo(q.x, PAD.t); ctx.lineTo(q.x, h - PAD.b); ctx.stroke(); ctx.textAlign = "center"; ctx.fillText(fmt(x, 1), q.x, h - 20); }
    for (let y = Math.ceil(b.ymin / ys) * ys; y <= b.ymax; y += ys) { const q = toCanvas(0, y); ctx.beginPath(); ctx.moveTo(PAD.l, q.y); ctx.lineTo(w - PAD.r, q.y); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(fmt(y, 1), PAD.l - 8, q.y + 3); }
  }
  function drawPlot() {
    if (!canvas.clientWidth) return; ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight); drawGrid();
    const frame = current();
    if (frame) drawStructures(frame);
    state.points.forEach((p, i) => drawPoint(p, i, frame));
    if (state.dragging && state.tool === "select" && state.dragStart && state.dragCurrent) {
      const x = Math.min(state.dragStart.x, state.dragCurrent.x), y = Math.min(state.dragStart.y, state.dragCurrent.y), w = Math.abs(state.dragStart.x - state.dragCurrent.x), h = Math.abs(state.dragStart.y - state.dragCurrent.y);
      ctx.fillStyle = "rgba(49,87,213,.1)"; ctx.fillRect(x, y, w, h); ctx.strokeStyle = "#3157d5"; ctx.setLineDash([5, 4]); ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
    }
  }
  function dataRadiusToPixels(r) { return r / (state.bounds.xmax - state.bounds.xmin) * (canvas.clientWidth - PAD.l - PAD.r); }
  function drawStructures(frame) {
    if (frame.radii && frame.centers) frame.centers.forEach((c, i) => {
      const q = toCanvas(c.x, c.y), radius = dataRadiusToPixels(frame.radii[i] || .1);
      ctx.beginPath(); ctx.arc(q.x, q.y, radius, 0, Math.PI * 2); ctx.fillStyle = SOFT[i % SOFT.length]; ctx.fill(); ctx.strokeStyle = COLORS[i % COLORS.length]; ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([]);
    });
    if (frame.merge) {
      const a = toCanvas(frame.merge.a.x, frame.merge.a.y), b = toCanvas(frame.merge.b.x, frame.merge.b.y);
      ctx.strokeStyle = "#172033"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    if (state.modelType === "gmm" && frame.means) frame.means.forEach((m, i) => {
      const q = toCanvas(m.x, m.y), v = frame.variances[i], rx = dataRadiusToPixels(Math.sqrt(v.x) * 1.7), ry = dataRadiusToPixels(Math.sqrt(v.y) * 1.7);
      ctx.save(); ctx.translate(q.x, q.y); ctx.beginPath(); ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2); ctx.fillStyle = SOFT[i % SOFT.length]; ctx.fill(); ctx.strokeStyle = COLORS[i % COLORS.length]; ctx.lineWidth = 2; ctx.stroke(); ctx.restore();
    });
    if (state.modelType === "dbscan" && frame.active !== null) {
      const p = state.points[frame.active], q = toCanvas(p.x, p.y);
      ctx.beginPath(); ctx.arc(q.x, q.y, dataRadiusToPixels(frame.epsilon), 0, Math.PI * 2); ctx.fillStyle = "rgba(82,104,217,.08)"; ctx.fill(); ctx.strokeStyle = "#5268d9"; ctx.setLineDash([6, 4]); ctx.stroke(); ctx.setLineDash([]);
    }
    const centers = frame.centers || frame.means;
    if (centers) centers.forEach((c, i) => {
      const q = toCanvas(c.x, c.y); ctx.save(); ctx.translate(q.x, q.y); ctx.rotate(Math.PI / 4); ctx.fillStyle = COLORS[i % COLORS.length]; ctx.fillRect(-7, -7, 14, 14); ctx.strokeStyle = "white"; ctx.lineWidth = 2; ctx.strokeRect(-7, -7, 14, 14); ctx.restore();
    });
  }
  function drawPoint(p, i, frame) {
    const assignment = frame?.assignments?.[i] ?? -2, q = toCanvas(p.x, p.y);
    ctx.beginPath(); ctx.arc(q.x, q.y, frame?.core?.[i] ? 7 : 6, 0, Math.PI * 2);
    ctx.fillStyle = assignment >= 0 ? COLORS[assignment % COLORS.length] : assignment === -1 ? "#9aa3b3" : "#c9ced8"; ctx.fill();
    ctx.strokeStyle = state.selected.has(p.id) ? "#111827" : frame?.core?.[i] ? "#172033" : "#fff"; ctx.lineWidth = state.selected.has(p.id) ? 3 : frame?.core?.[i] ? 2.3 : 1.7; ctx.stroke();
    if (assignment === -1) { ctx.strokeStyle = "#596273"; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.moveTo(q.x - 4, q.y - 4); ctx.lineTo(q.x + 4, q.y + 4); ctx.moveTo(q.x + 4, q.y - 4); ctx.lineTo(q.x - 4, q.y + 4); ctx.stroke(); }
    if (i === state.hover) { ctx.beginPath(); ctx.arc(q.x, q.y, 11, 0, Math.PI * 2); ctx.strokeStyle = "#172033"; ctx.lineWidth = 1.4; ctx.stroke(); }
  }

  function eventPos(e) { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
  function inside(p) { return p.x >= PAD.l && p.x <= canvas.clientWidth - PAD.r && p.y >= PAD.t && p.y <= canvas.clientHeight - PAD.b; }
  function nearest(pos, max = 13) { let best = -1, d = max; state.points.forEach((p, i) => { const q = toCanvas(p.x, p.y), v = Math.hypot(q.x - pos.x, q.y - pos.y); if (v < d) { d = v; best = i; } }); return best; }
  function invalidate() { stopPlaying(); state.history = []; state.step = 0; state.bounds = calculateBounds(); updateAll(); }
  function addAt(pos) { const q = toData(pos.x, pos.y); state.points.push({ id: state.nextId++, x: q.x, y: q.y }); invalidate(); }
  function brushAt(pos) {
    const rng = new RNG(Math.floor(pos.x * 31 + pos.y * 17 + state.points.length * 97));
    for (let i = 0; i < 3; i++) { const a = rng.next() * Math.PI * 2, r = Math.sqrt(rng.next()) * state.brushRadius, s = { x: pos.x + Math.cos(a) * r, y: pos.y + Math.sin(a) * r }; if (inside(s)) { const q = toData(s.x, s.y); state.points.push({ id: state.nextId++, x: q.x, y: q.y }); } }
    invalidate();
  }
  function eraseAt(pos) { state.points = state.points.filter(p => { const q = toCanvas(p.x, p.y); return Math.hypot(q.x - pos.x, q.y - pos.y) > state.brushRadius; }); invalidate(); }
  canvas.addEventListener("pointerdown", e => {
    const p = eventPos(e); if (!inside(p)) return; state.dragging = true; canvas.setPointerCapture(e.pointerId);
    if (state.tool === "select") state.dragStart = state.dragCurrent = p; else if (state.tool === "brush") brushAt(p); else if (state.tool === "erase") eraseAt(p);
  });
  canvas.addEventListener("pointermove", e => {
    const p = eventPos(e);
    if (state.dragging) { if (state.tool === "select") { state.dragCurrent = p; drawPlot(); } else if (state.tool === "brush") brushAt(p); else if (state.tool === "erase") eraseAt(p); return; }
    state.hover = nearest(p, 14); if (state.hover >= 0) showHover(state.hover, p); else $("hoverCard").classList.add("hidden"); drawPlot();
  });
  canvas.addEventListener("pointerup", e => {
    const p = eventPos(e);
    if (state.tool === "add" && inside(p)) addAt(p);
    else if (state.tool === "select" && state.dragStart) {
      const x0 = Math.min(state.dragStart.x, p.x), x1 = Math.max(state.dragStart.x, p.x), y0 = Math.min(state.dragStart.y, p.y), y1 = Math.max(state.dragStart.y, p.y);
      state.selected.clear(); state.points.forEach(q => { const c = toCanvas(q.x, q.y); if (c.x >= x0 && c.x <= x1 && c.y >= y0 && c.y <= y1) state.selected.add(q.id); }); state.dragStart = state.dragCurrent = null; updateTable();
    }
    state.dragging = false; drawPlot();
  });
  canvas.addEventListener("pointerleave", () => { if (!state.dragging) { state.hover = -1; $("hoverCard").classList.add("hidden"); drawPlot(); } });
  function showHover(i, pos) {
    const p = state.points[i], frame = current(), a = frame?.assignments?.[i] ?? -2;
    $("hoverCard").innerHTML = `<b>Point ${i + 1}</b><br>X: ${fmt(p.x, 2)} · Y: ${fmt(p.y, 2)}<br>${a >= 0 ? `Cluster ${a + 1}` : a === -1 ? "Noise" : "Unassigned"}${frame?.core?.[i] ? "<br><b>Core point</b>" : ""}`;
    $("hoverCard").classList.remove("hidden"); $("hoverCard").style.left = `${clamp(pos.x + 14, 4, canvas.clientWidth - 210)}px`; $("hoverCard").style.top = `${clamp(pos.y - 18, 4, canvas.clientHeight - 105)}px`;
  }

  function range(key, label, min, max, step, value) { return `<label class="range-row">${label}<b data-value="${key}">${value}</b><input data-param="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${value}"></label>`; }
  function select(key, label, value, options) { return `<label class="field-label spaced">${label}<select data-param="${key}">${options.map(([v, t]) => `<option value="${v}" ${value === v ? "selected" : ""}>${t}</option>`).join("")}</select></label>`; }
  function parameterSchema() {
    const p = state.params[state.modelType];
    if (state.modelType === "kmeans") return range("k", "Number of clusters (k)", 2, 8, 1, p.k) + range("maxIterations", "Maximum iterations", 2, 30, 1, p.maxIterations) + select("initialization", "Initialization", p.initialization, [["plusplus", "k-means++"], ["random", "Random points"]]);
    if (state.modelType === "hierarchical") return range("clusters", "Final clusters", 2, 8, 1, p.clusters) + select("linkage", "Linkage method", p.linkage, [["single", "Single linkage"], ["average", "Average linkage"], ["complete", "Complete linkage"]]);
    if (state.modelType === "dbscan") return range("epsilon", "Neighborhood epsilon", .15, 2, .05, p.epsilon) + range("minPoints", "Minimum points", 2, 12, 1, p.minPoints);
    return range("components", "Gaussian components", 2, 8, 1, p.components) + range("iterations", "EM iterations", 2, 35, 1, p.iterations) + range("regularization", "Covariance regularization", .01, .8, .01, p.regularization);
  }
  function renderParameters() {
    $("parameterFields").innerHTML = parameterSchema();
    $("parameterFields").querySelectorAll("[data-param]").forEach(el => el.addEventListener("input", () => {
      const key = el.dataset.param; state.params[state.modelType][key] = el.tagName === "SELECT" ? el.value : Number(el.value);
      const out = document.querySelector(`[data-value="${key}"]`); if (out) out.textContent = el.value; updateParamSummary();
      clearTimeout(renderParameters.timer); renderParameters.timer = setTimeout(initialize, 180);
    }));
  }
  function updateParamSummary() {
    const p = state.params[state.modelType], text = {
      kmeans: `k = ${p.k}`, hierarchical: `${p.clusters} clusters · ${p.linkage}`,
      dbscan: `ε ${p.epsilon} · min ${p.minPoints}`, gmm: `${p.components} Gaussians`
    };
    $("paramSummary").textContent = text[state.modelType];
  }
  function updateMetrics() {
    const frame = current(), assigned = frame?.assignments?.filter(v => v >= 0) || [], clusters = new Set(assigned).size, noise = frame?.assignments?.filter(v => v === -1).length || 0;
    const objectiveLabel = state.modelType === "kmeans" ? "Inertia" : state.modelType === "gmm" ? "Log likelihood" : state.modelType === "hierarchical" ? "Merge distance" : "Clusters";
    $("metricGrid").innerHTML = [["Clusters", frame ? clusters : "—"], ["Noise points", frame ? noise : "—"], ["Current step", state.history.length ? `${state.step}/${state.history.length - 1}` : "—"], [objectiveLabel, frame?.objective !== null && frame?.objective !== undefined ? fmt(frame.objective, 2) : "—"], ["Points", state.points.length], ["Phase", frame?.phase || "Not started"]].map(([a, b]) => `<div class="metric-tile"><small>${a}</small><strong>${b}</strong></div>`).join("");
    $("metricExtra").innerHTML = `<div class="metric-extra">${frame ? frame.message : "Initialize the algorithm to create its step-by-step history."}</div>`;
    $("accuracyLabel").textContent = frame ? `${clusters} clusters · step ${state.step}` : "Not initialized";
  }
  function updateVisual() {
    $("insightTitle").textContent = TITLES[state.modelType]; $("insightText").textContent = EXPLAINS[state.modelType];
    const cards = {
      kmeans: [["1", "Place centroids", "Initialize"], ["2", "Assign nearest center", "Color points"], ["3", "Move to cluster mean", "Repeat"]],
      hierarchical: [["1", "One point per cluster", "Singletons"], ["2", "Measure cluster distance", "Linkage"], ["3", "Merge closest pair", "Hierarchy"]],
      dbscan: [["1", "Draw ε neighborhood", "Radius"], ["2", "Identify core points", "Density"], ["3", "Expand reachable region", "Noise remains"]],
      gmm: [["E", "Estimate soft membership", "Probabilities"], ["M", "Update Gaussian shapes", "Means + variance"], ["↻", "Repeat EM cycle", "Likelihood"]]
    };
    $("modelVisual").innerHTML = `<div class="flow-steps">${cards[state.modelType].map(c => `<div class="flow-step"><span class="num">${c[0]}</span><span>${c[1]}</span><small>${c[2]}</small></div>`).join("")}</div>`;
  }
  function updateSteps() {
    const total = Math.max(0, state.history.length - 1);
    $("stepCounter").textContent = `${state.step} / ${total}`;
    $("stepProgress").style.width = `${total ? state.step / total * 100 : 0}%`;
    $("backButton").disabled = state.step <= 0; $("stepButton").disabled = !state.history.length || state.step >= total;
    const frame = current(); $("stepExplanation").textContent = frame?.message || "Initialize the algorithm to begin.";
    const icon = $("selectedReadout").querySelector(".readout-icon");
    if (icon) icon.textContent = frame ? state.step : "◎";
  }
  function drawChart() {
    const r = chart.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; if (!r.width) return;
    chart.width = r.width * dpr; chart.height = r.height * dpr; chartCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = r.width, h = r.height; chartCtx.clearRect(0, 0, w, h); chartCtx.strokeStyle = "#e4e8f0";
    for (let i = 1; i < 4; i++) { chartCtx.beginPath(); chartCtx.moveTo(28, i * h / 4); chartCtx.lineTo(w - 8, i * h / 4); chartCtx.stroke(); }
    const values = state.history.slice(0, state.step + 1).map(s => s.objective).filter(v => Number.isFinite(v));
    if (values.length < 2) { chartCtx.fillStyle = "#99a2b4"; chartCtx.font = "11px DM Sans"; chartCtx.textAlign = "center"; chartCtx.fillText("Advance steps to reveal the objective history", w / 2, h / 2); return; }
    const min = Math.min(...values), max = Math.max(...values), span = max - min || 1; chartCtx.beginPath();
    values.forEach((v, i) => { const x = 28 + i / (values.length - 1) * (w - 42), y = 12 + (max - v) / span * (h - 34); i ? chartCtx.lineTo(x, y) : chartCtx.moveTo(x, y); });
    chartCtx.strokeStyle = "#5268d9"; chartCtx.lineWidth = 2.5; chartCtx.stroke();
  }
  function updateTable() {
    const frame = current(); $("tableCount").textContent = `(${state.points.length} points)`;
    $("dataTableBody").innerHTML = state.points.map((p, i) => {
      const a = frame?.assignments?.[i] ?? -2, label = a >= 0 ? `Cluster ${a + 1}` : a === -1 ? "Noise" : "Unassigned";
      const chip = a >= 0 ? `<span class="cluster-chip" style="background:${COLORS[a % COLORS.length]}">${label}</span>` : `<span class="cluster-chip ${a === -1 ? "noise" : "unassigned"}">${label}</span>`;
      return `<tr data-id="${p.id}" class="${state.selected.has(p.id) ? "selected-row" : ""}"><td>${i + 1}</td><td>${fmt(p.x, 3)}</td><td>${fmt(p.y, 3)}</td><td>${chip}</td><td><span class="iteration-badge">${frame?.core?.[i] ? "Core" : frame?.phase || "Ready"}</span></td><td><button class="row-delete" data-delete-id="${p.id}">Delete</button></td></tr>`;
    }).join("");
    document.querySelectorAll("[data-delete-id]").forEach(el => el.addEventListener("click", () => { state.points = state.points.filter(p => p.id !== Number(el.dataset.deleteId)); invalidate(); }));
  }
  function updateStatus() {
    const frame = current();
    if (frame) { $("plotStatus").className = "plot-status"; $("plotStatus").innerHTML = `<span></span> ${frame.phase}`; $("canvasMessage").classList.add("hidden"); }
    else { $("plotStatus").className = "plot-status untrained"; $("plotStatus").innerHTML = "<span></span> Ready to initialize"; $("canvasMessage").textContent = "Select Initialize, then use Step or Play to construct the clusters."; $("canvasMessage").classList.remove("hidden"); }
  }
  function updateAll() {
    $("datasetLabel").textContent = DATASET_NAMES[state.dataset]; $("modelLabel").textContent = MODEL_NAMES[state.modelType]; $("heroModel").textContent = MODEL_NAMES[state.modelType];
    $("plotTitle").textContent = `${MODEL_NAMES[state.modelType]} on ${DATASET_NAMES[state.dataset]}`; $("modelQuickExplain").textContent = EXPLAINS[state.modelType];
    updateParamSummary(); updateStatus(); updateMetrics(); updateVisual(); updateSteps(); updateTable(); drawPlot(); drawChart();
  }
  function showWarning(t) { $("trainingWarning").textContent = t; $("trainingWarning").classList.remove("hidden"); }
  function hideWarning() { $("trainingWarning").classList.add("hidden"); }
  function loadDataset() { state.points = generateDataset(state.dataset); state.original = state.points.map(p => ({ ...p })); state.selected.clear(); state.bounds = calculateBounds(); initialize(); }
  function deleteSelected() { state.points = state.points.filter(p => !state.selected.has(p.id)); state.selected.clear(); invalidate(); }
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
  $("modelSelect").addEventListener("change", e => { state.modelType = e.target.value; renderParameters(); initialize(); });
  $("trainButton").addEventListener("click", initialize);
  $("resetDataset").addEventListener("click", () => { state.points = state.original.map(p => ({ ...p, id: state.nextId++ })); state.bounds = calculateBounds(); initialize(); });
  $("stepButton").addEventListener("click", advance); $("backButton").addEventListener("click", back); $("playButton").addEventListener("click", togglePlay);
  $("speedSlider").addEventListener("input", e => { state.speed = Number(e.target.value); $("speedValue").textContent = `${state.speed} ms`; });
  $("brushRadius").addEventListener("input", e => { state.brushRadius = Number(e.target.value); $("brushRadiusValue").textContent = `${state.brushRadius} px`; });
  $("toolButtons").querySelectorAll("[data-tool]").forEach(button => button.addEventListener("click", () => {
    state.tool = button.dataset.tool; document.querySelectorAll(".tool").forEach(b => b.classList.toggle("active", b === button));
    $("toolLabel").textContent = { add: "Add point", brush: "Point brush", erase: "Eraser", select: "Select region" }[state.tool];
  }));
  $("deleteSelected").addEventListener("click", deleteSelected); $("tableDeleteSelected").addEventListener("click", deleteSelected);
  $("clearData").addEventListener("click", clearData); $("tableClear").addEventListener("click", clearData);
  $("predictButton").addEventListener("click", () => {
    const frame = current(); if (!frame) return;
    const x = Number($("predictX").value), y = Number($("predictY").value), point = { x, y };
    let cluster = -1, explanation = "";
    if (state.modelType === "dbscan") {
      const close = state.points.map((p, i) => ({ i, d: distance(point, p) })).filter(q => q.d <= state.params.dbscan.epsilon && frame.assignments[q.i] >= 0);
      if (close.length) cluster = frame.assignments[close.sort((a, b) => a.d - b.d)[0].i];
      explanation = cluster >= 0 ? `The point is density-reachable from Cluster ${cluster + 1}.` : "The point is outside the discovered dense regions.";
    } else if (state.modelType === "gmm" && frame.means) {
      const probs = frame.means.map((m, i) => frame.weights[i] * gaussian(point, m, frame.variances[i])); cluster = probs.indexOf(Math.max(...probs));
      explanation = `Gaussian ${cluster + 1} gives the highest weighted likelihood.`;
    } else {
      const centers = frame.centers || []; if (centers.length) cluster = centers.map(c => distance(point, c)).indexOf(Math.min(...centers.map(c => distance(point, c))));
      explanation = cluster >= 0 ? `The nearest current cluster center is Cluster ${cluster + 1}.` : "No cluster center is available yet.";
    }
    $("predictionOutput").innerHTML = `<div class="prediction-class ${cluster < 0 ? "neutral" : ""}" style="${cluster >= 0 ? `background:${COLORS[cluster % COLORS.length]}` : ""}">${cluster >= 0 ? cluster + 1 : "?"}</div><div><small>${cluster >= 0 ? `Cluster ${cluster + 1}` : "Unassigned"}</small><p>${explanation}</p></div>`;
  });

  const qa = [
    ["Why can k-means fail on two moons or circles?", "It assigns points to the nearest mean, which naturally favors compact, roughly spherical clusters."],
    ["How does initialization affect k-means?", "Poor initial centroids can lead to a worse local solution. k-means++ spreads initial centers apart."],
    ["What does linkage mean in hierarchical clustering?", "It defines how distance between two groups is measured: nearest pair, farthest pair, or average pair distance."],
    ["Why does DBSCAN identify noise?", "Points outside density-reachable neighborhoods do not belong to any sufficiently dense cluster."],
    ["How do epsilon and min points change DBSCAN?", "Larger epsilon or smaller minimum counts connect more points, often merging clusters and reducing noise."],
    ["How is a Gaussian mixture different from k-means?", "It gives soft probabilities and learns each component's spread, while k-means uses hard assignments and center distance."],
    ["Why can clustering results be subjective?", "There may be several meaningful groupings, and every algorithm encodes different assumptions about shape and density."],
    ["What does the objective curve tell us?", "It shows whether iterative updates are improving inertia, likelihood, merge level, or discovered density structure."]
  ];
  $("questionList").innerHTML = qa.map(q => `<li>${q[0]}</li>`).join(""); $("answerList").innerHTML = qa.map(q => `<li>${q[1]}</li>`).join("");
  const style = document.createElement("style"); style.textContent = `.field-label.spaced{margin-top:14px}.step-buttons button:disabled{opacity:.4;cursor:not-allowed}`; document.head.appendChild(style);
  window.addEventListener("resize", () => { clearTimeout(resizeCanvas.timer); resizeCanvas.timer = setTimeout(resizeCanvas, 80); });
  document.addEventListener("keydown", e => {
    if (e.key === "ArrowRight") advance(); else if (e.key === "ArrowLeft") back();
    else if ((e.key === "Delete" || e.key === "Backspace") && state.selected.size && !/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) deleteSelected();
  });
  renderParameters(); requestAnimationFrame(() => { resizeCanvas(); loadDataset(); });
})();
