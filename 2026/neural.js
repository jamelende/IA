(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const sigmoid = x => 1 / (1 + Math.exp(-clamp(x, -30, 30)));
  const COLORS = { blue: "#3157d5", red: "#e74c5b" };

  class RNG {
    constructor(seed = 20260609) { this.s = seed >>> 0; }
    next() { this.s = (1664525 * this.s + 1013904223) >>> 0; return this.s / 4294967296; }
    normal() { const u = Math.max(this.next(), 1e-9), v = this.next(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  }

  const state = {
    dataset: "moons", points: [], original: [], nextId: 1, tool: "blue", brush: 18,
    trainSplit: .8, layers: [4, 3], features: ["x", "y"], network: null,
    epoch: 0, playing: false, timer: null, history: [], selectedNode: { type: "output", layer: -1, node: 0 },
    bounds: { xmin: -3, xmax: 3, ymin: -3, ymax: 3 }, dragging: false
  };
  const DATASET_LABELS = { xor: "XOR", moons: "Dos lunas", circles: "Círculos", spiral: "Espiral", linear: "Lineal", custom: "Personalizado" };
  const TOOL_LABELS = { blue: "Pintar azul", red: "Pintar rojo", erase: "Borrar" };

  const dataCanvas = $("dataCanvas"), dataCtx = dataCanvas.getContext("2d");
  const outputCanvas = $("outputCanvas"), outputCtx = outputCanvas.getContext("2d");
  const lossCanvas = $("lossChart"), lossCtx = lossCanvas.getContext("2d");
  const edgeSvg = $("networkEdges");

  function generate(type, preview = false) {
    const rng = new RNG(8119 + ["xor", "moons", "circles", "spiral", "linear", "custom"].indexOf(type) * 971), pts = [];
    const n = preview ? 36 : 100;
    const add = (x, y, label) => pts.push({ id: preview ? 0 : state.nextId++, x, y, label, test: false });
    if (type === "xor") {
      for (let i = 0; i < n; i++) { const x = rng.next() * 5 - 2.5, y = rng.next() * 5 - 2.5; add(x, y, (x > 0) !== (y > 0) ? 1 : 0); }
    } else if (type === "moons") {
      for (let i = 0; i < n; i++) { const c = i < n / 2 ? 0 : 1, a = rng.next() * Math.PI; add((c ? 1 - Math.cos(a) : Math.cos(a)) * 1.8 + rng.normal() * .13, (c ? -.55 - Math.sin(a) : Math.sin(a)) * 1.5 + rng.normal() * .13, c); }
    } else if (type === "circles") {
      for (let i = 0; i < n; i++) { const outer = i >= n * .46, a = rng.next() * Math.PI * 2, r = (outer ? 2.1 : .82) + rng.normal() * .12; add(Math.cos(a) * r, Math.sin(a) * r, outer ? 0 : 1); }
    } else if (type === "spiral") {
      for (let c = 0; c < 2; c++) for (let i = 0; i < n / 2; i++) { const r = .2 + i / (n / 13), t = i / (n / 15) + c * Math.PI; add(r * Math.cos(t) + rng.normal() * .08, r * Math.sin(t) + rng.normal() * .08, c); }
    } else if (type === "linear") {
      for (let i = 0; i < n; i++) { const x = rng.next() * 5 - 2.5, y = rng.next() * 5 - 2.5; add(x, y, y > .6 * x + rng.normal() * .45 ? 1 : 0); }
    } else if (!preview) return [];
    return pts;
  }
  function setDataset(type) {
    stop(); state.dataset = type;
    state.points = type === "custom" ? [] : generate(type);
    splitData(); state.original = state.points.map(p => ({ ...p })); state.bounds = calculateBounds(); resetNetwork(); updateAll();
    $("datasetLabel").textContent = DATASET_LABELS[type];
  }
  function splitData() {
    const rng = new RNG(4401);
    state.points.forEach(p => p.test = rng.next() > state.trainSplit);
  }
  function calculateBounds() {
    if (!state.points.length) return { xmin: -3, xmax: 3, ymin: -3, ymax: 3 };
    const xs = state.points.map(p => p.x), ys = state.points.map(p => p.y), xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
    const xp = Math.max(.6, (xmax - xmin) * .15), yp = Math.max(.6, (ymax - ymin) * .15);
    return { xmin: xmin - xp, xmax: xmax + xp, ymin: ymin - yp, ymax: ymax + yp };
  }

  function featureValue(name, x, y) {
    if (name === "x") return x; if (name === "y") return y; if (name === "x2") return x * x; if (name === "y2") return y * y;
    if (name === "xy") return x * y; if (name === "sinx") return Math.sin(x); return Math.sin(y);
  }
  function inputVector(x, y) { return state.features.map(f => featureValue(f, x, y)); }
  function activation(x) {
    const type = $("activation").value;
    return type === "relu" ? Math.max(0, x) : type === "sigmoid" ? sigmoid(x) : Math.tanh(x);
  }
  function activationDerivative(z, a) {
    const type = $("activation").value;
    return type === "relu" ? (z > 0 ? 1 : 0) : type === "sigmoid" ? a * (1 - a) : 1 - a * a;
  }
  function makeNetwork() {
    const rng = new RNG(19771), sizes = [state.features.length, ...state.layers, 1], layers = [];
    for (let l = 1; l < sizes.length; l++) {
      const scale = Math.sqrt(2 / Math.max(1, sizes[l - 1] + sizes[l]));
      layers.push({
        weights: Array.from({ length: sizes[l] }, () => Array.from({ length: sizes[l - 1] }, () => rng.normal() * scale)),
        biases: Array(sizes[l]).fill(0)
      });
    }
    return { sizes, layers };
  }
  function resetNetwork() {
    stop(); state.network = makeNetwork(); state.epoch = 0; state.history = []; state.selectedNode = { type: "output", layer: state.layers.length, node: 0 };
    renderNetwork(); evaluate(); updateSummaries();
  }
  function forward(input) {
    const activations = [input], zs = [];
    let current = input;
    state.network.layers.forEach((layer, li) => {
      const z = layer.weights.map((row, j) => row.reduce((s, w, i) => s + w * current[i], layer.biases[j]));
      const a = li === state.network.layers.length - 1 ? z.map(sigmoid) : z.map(activation);
      zs.push(z); activations.push(a); current = a;
    });
    return { activations, zs, output: current[0] };
  }
  function trainEpoch() {
    if (!state.points.some(p => !p.test) || state.features.length === 0) return;
    const train = state.points.filter(p => !p.test), lr = Number($("learningRate").value), lambda = Number($("regularization").value);
    const requestedBatch = Number($("batchSize").value), batchSize = requestedBatch > train.length ? train.length : requestedBatch;
    const rng = new RNG(7001 + state.epoch), shuffled = [...train].sort(() => rng.next() - .5);
    for (let start = 0; start < shuffled.length; start += batchSize) {
      const batch = shuffled.slice(start, start + batchSize);
      const gradW = state.network.layers.map(l => l.weights.map(row => row.map(() => 0)));
      const gradB = state.network.layers.map(l => l.biases.map(() => 0));
      batch.forEach(point => {
        const result = forward(inputVector(point.x, point.y)), deltas = Array(state.network.layers.length);
        deltas[deltas.length - 1] = [result.output - point.label];
        for (let l = deltas.length - 2; l >= 0; l--) {
          deltas[l] = result.activations[l + 1].map((a, j) => {
            const upstream = state.network.layers[l + 1].weights.reduce((s, row, k) => s + row[j] * deltas[l + 1][k], 0);
            return upstream * activationDerivative(result.zs[l][j], a);
          });
        }
        deltas.forEach((delta, l) => delta.forEach((d, j) => {
          gradB[l][j] += d;
          result.activations[l].forEach((a, i) => gradW[l][j][i] += d * a);
        }));
      });
      state.network.layers.forEach((layer, l) => layer.weights.forEach((row, j) => row.forEach((w, i) => {
        layer.weights[j][i] -= lr * (gradW[l][j][i] / batch.length + lambda * w);
      })));
      state.network.layers.forEach((layer, l) => layer.biases.forEach((b, j) => layer.biases[j] -= lr * gradB[l][j] / batch.length));
    }
    state.epoch++; evaluate();
    if (state.epoch % 2 === 0 || state.epoch < 10) { drawOutput(); drawNeuronMaps(); drawEdges(); drawLoss(); }
  }
  function evaluate() {
    if (!state.network || !state.points.length) { $("lossValue").textContent = "—"; $("accuracyValue").textContent = "—"; return; }
    let loss = 0, correct = 0, n = 0;
    state.points.forEach(p => {
      const out = clamp(forward(inputVector(p.x, p.y)).output, 1e-7, 1 - 1e-7);
      if (p.test) { loss -= p.label * Math.log(out) + (1 - p.label) * Math.log(1 - out); correct += (out >= .5 ? 1 : 0) === p.label; n++; }
    });
    if (!n) state.points.forEach(p => { const out = clamp(forward(inputVector(p.x, p.y)).output, 1e-7, 1 - 1e-7); loss -= p.label * Math.log(out) + (1 - p.label) * Math.log(1 - out); correct += (out >= .5 ? 1 : 0) === p.label; n++; });
    const metrics = { loss: loss / Math.max(1, n), accuracy: correct / Math.max(1, n) };
    state.history.push({ epoch: state.epoch, ...metrics });
    $("epochValue").textContent = state.epoch; $("lossValue").textContent = metrics.loss.toFixed(3); $("accuracyValue").textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
  }

  function step() { stop(); trainEpoch(); updateStatus(); }
  function play() {
    if (state.playing) return stop();
    state.playing = true; $("playButton").textContent = "❚❚"; $("trainingStatus").textContent = "Entrenando"; $("trainingStatus").classList.add("running");
    const tick = () => { if (!state.playing) return; for (let i = 0; i < 3; i++) trainEpoch(); state.timer = setTimeout(tick, 18); };
    tick();
  }
  function stop() {
    state.playing = false; clearTimeout(state.timer);
    if ($("playButton")) $("playButton").textContent = "▶";
    if ($("trainingStatus")) { $("trainingStatus").textContent = state.epoch ? "Pausada" : "Lista"; $("trainingStatus").classList.remove("running"); }
  }
  function updateStatus() { $("trainingStatus").textContent = state.epoch ? "Actualizada" : "Lista"; }

  function resizeCanvas(canvas, ctx) {
    const dpr = window.devicePixelRatio || 1, r = canvas.getBoundingClientRect(), w = Math.max(50, Math.floor(r.width)), h = Math.max(50, Math.floor(r.height));
    canvas.width = w * dpr; canvas.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return { w, h };
  }
  function toCanvas(canvas, x, y) {
    const b = state.bounds, pad = 10;
    return { x: pad + (x - b.xmin) / (b.xmax - b.xmin) * (canvas.clientWidth - pad * 2), y: pad + (b.ymax - y) / (b.ymax - b.ymin) * (canvas.clientHeight - pad * 2) };
  }
  function toData(canvas, x, y) {
    const b = state.bounds, pad = 10;
    return { x: b.xmin + (x - pad) / (canvas.clientWidth - pad * 2) * (b.xmax - b.xmin), y: b.ymax - (y - pad) / (canvas.clientHeight - pad * 2) * (b.ymax - b.ymin) };
  }
  function colorFor(v, alpha = 1) {
    const t = clamp(v, 0, 1), neutral = [248, 248, 251], target = t >= .5 ? [49, 87, 213] : [231, 76, 91], strength = Math.abs(t - .5) * 1.65;
    const rgb = neutral.map((n, i) => Math.round(n + (target[i] - n) * strength));
    return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
  }
  function drawData() {
    const { w, h } = resizeCanvas(dataCanvas, dataCtx); dataCtx.fillStyle = "#fbfcfe"; dataCtx.fillRect(0, 0, w, h);
    dataCtx.strokeStyle = "rgba(100,112,136,.12)"; for (let i = 1; i < 6; i++) { dataCtx.beginPath(); dataCtx.moveTo(i * w / 6, 0); dataCtx.lineTo(i * w / 6, h); dataCtx.stroke(); dataCtx.beginPath(); dataCtx.moveTo(0, i * h / 6); dataCtx.lineTo(w, i * h / 6); dataCtx.stroke(); }
    state.points.forEach(p => {
      const q = toCanvas(dataCanvas, p.x, p.y); dataCtx.beginPath(); dataCtx.arc(q.x, q.y, 5, 0, Math.PI * 2); dataCtx.fillStyle = p.label ? COLORS.blue : COLORS.red; dataCtx.fill(); dataCtx.strokeStyle = p.test ? "#172033" : "#fff"; dataCtx.lineWidth = p.test ? 2.4 : 1.5; dataCtx.stroke();
    });
    $("pointCount").textContent = `${state.points.length} puntos`;
  }
  function mapValue(x, y) {
    if (!state.network || !state.features.length) return .5;
    const result = forward(inputVector(x, y)), sel = state.selectedNode;
    if (sel.type === "input") {
      const raw = inputVector(x, y)[sel.node] || 0; return sigmoid(raw);
    }
    if (sel.type === "hidden") return (result.activations[sel.layer + 1][sel.node] + 1) / 2;
    return result.output;
  }
  function drawOutput() {
    const { w, h } = resizeCanvas(outputCanvas, outputCtx), scale = .35, off = document.createElement("canvas");
    off.width = Math.max(80, Math.floor(w * scale)); off.height = Math.max(80, Math.floor(h * scale));
    const oc = off.getContext("2d"), img = oc.createImageData(off.width, off.height), b = state.bounds;
    for (let py = 0; py < off.height; py++) for (let px = 0; px < off.width; px++) {
      const x = b.xmin + px / (off.width - 1) * (b.xmax - b.xmin), y = b.ymax - py / (off.height - 1) * (b.ymax - b.ymin), v = mapValue(x, y);
      const c = colorFor(v).match(/\d+/g).map(Number), i = (py * off.width + px) * 4; img.data[i] = c[0]; img.data[i + 1] = c[1]; img.data[i + 2] = c[2]; img.data[i + 3] = 255;
    }
    oc.putImageData(img, 0, 0); outputCtx.drawImage(off, 0, 0, w, h);
    state.points.forEach(p => { const q = toCanvas(outputCanvas, p.x, p.y); outputCtx.beginPath(); outputCtx.arc(q.x, q.y, 4.5, 0, Math.PI * 2); outputCtx.fillStyle = p.label ? COLORS.blue : COLORS.red; outputCtx.fill(); outputCtx.strokeStyle = p.test ? "#172033" : "#fff"; outputCtx.lineWidth = p.test ? 2 : 1.2; outputCtx.stroke(); });
  }
  function drawLoss() {
    const { w, h } = resizeCanvas(lossCanvas, lossCtx); lossCtx.clearRect(0, 0, w, h); lossCtx.strokeStyle = "#e4e8f0";
    for (let i = 1; i < 3; i++) { lossCtx.beginPath(); lossCtx.moveTo(5, i * h / 3); lossCtx.lineTo(w - 3, i * h / 3); lossCtx.stroke(); }
    if (state.history.length < 2) return;
    const values = state.history.slice(-250), max = Math.max(...values.map(v => v.loss), .1);
    lossCtx.beginPath(); values.forEach((v, i) => { const x = 5 + i / (values.length - 1) * (w - 10), y = 4 + v.loss / max * (h - 9); i ? lossCtx.lineTo(x, y) : lossCtx.moveTo(x, y); });
    lossCtx.strokeStyle = COLORS.blue; lossCtx.lineWidth = 2; lossCtx.stroke();
  }

  function renderNetwork() {
    const host = $("networkColumns"), names = state.features.map(f => ({ x: "X₁", y: "X₂", x2: "X₁²", y2: "X₂²", xy: "X₁X₂", sinx: "sin X₁", siny: "sin X₂" })[f]);
    let html = `<div class="layer input-layer"><b class="layer-title">Entradas</b>${names.map((n, i) => neuronHTML("input", -1, i, n)).join("")}</div>`;
    state.layers.forEach((count, l) => {
      html += `<div class="layer"><b class="layer-title">Oculta ${l + 1}</b>${Array.from({ length: count }, (_, i) => neuronHTML("hidden", l, i, `H${i + 1}`)).join("")}<div class="layer-controls"><button data-layer="${l}" data-delta="-1">−</button><button data-layer="${l}" data-delta="1">+</button><button data-remove-layer="${l}">×</button></div></div>`;
    });
    html += `<div class="layer output-layer"><b class="layer-title">Salida</b>${neuronHTML("output", state.layers.length, 0, "ŷ")}</div>`;
    host.innerHTML = html;
    host.querySelectorAll(".neuron").forEach(n => n.addEventListener("click", () => { state.selectedNode = { type: n.dataset.type, layer: Number(n.dataset.layer), node: Number(n.dataset.node) }; updateNeuronSelection(); drawOutput(); }));
    host.querySelectorAll("[data-delta]").forEach(b => b.addEventListener("click", () => { const l = Number(b.dataset.layer); state.layers[l] = clamp(state.layers[l] + Number(b.dataset.delta), 1, 8); resetNetwork(); }));
    host.querySelectorAll("[data-remove-layer]").forEach(b => b.addEventListener("click", () => { state.layers.splice(Number(b.dataset.removeLayer), 1); resetNetwork(); }));
    requestAnimationFrame(() => { drawNeuronMaps(); drawEdges(); updateNeuronSelection(); });
  }
  function neuronHTML(type, layer, node, label) { return `<button class="neuron" data-type="${type}" data-layer="${layer}" data-node="${node}"><canvas></canvas><span>${label}</span></button>`; }
  function updateNeuronSelection() {
    document.querySelectorAll(".neuron").forEach(n => n.classList.toggle("active", n.dataset.type === state.selectedNode.type && Number(n.dataset.layer) === state.selectedNode.layer && Number(n.dataset.node) === state.selectedNode.node));
  }
  function drawNeuronMaps() {
    document.querySelectorAll(".neuron").forEach(n => {
      const c = n.querySelector("canvas"), r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1, w = Math.max(30, Math.floor(r.width)), h = Math.max(30, Math.floor(r.height));
      c.width = w * dpr; c.height = h * dpr; const cx = c.getContext("2d"); cx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const type = n.dataset.type, layer = Number(n.dataset.layer), node = Number(n.dataset.node), grid = 20;
      for (let py = 0; py < grid; py++) for (let px = 0; px < grid; px++) {
        const x = state.bounds.xmin + px / (grid - 1) * (state.bounds.xmax - state.bounds.xmin), y = state.bounds.ymax - py / (grid - 1) * (state.bounds.ymax - state.bounds.ymin);
        const result = state.network ? forward(inputVector(x, y)) : null;
        let v = .5;
        if (type === "input") v = sigmoid(inputVector(x, y)[node] || 0);
        else if (type === "hidden" && result) v = (result.activations[layer + 1][node] + 1) / 2;
        else if (result) v = result.output;
        cx.fillStyle = colorFor(v); cx.fillRect(px * w / grid, py * h / grid, w / grid + 1, h / grid + 1);
      }
    });
  }
  function drawEdges() {
    edgeSvg.innerHTML = "";
    const layers = [...document.querySelectorAll(".layer")];
    layers.slice(0, -1).forEach((layerEl, li) => {
      const from = [...layerEl.querySelectorAll(".neuron")], to = [...layers[li + 1].querySelectorAll(".neuron")], weights = state.network?.layers[li]?.weights || [];
      from.forEach((a, i) => to.forEach((b, j) => {
        const ar = a.getBoundingClientRect(), br = b.getBoundingClientRect(), sr = edgeSvg.getBoundingClientRect(), weight = weights[j]?.[i] || 0;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", ar.right - sr.left); line.setAttribute("y1", ar.top + ar.height / 2 - sr.top); line.setAttribute("x2", br.left - sr.left); line.setAttribute("y2", br.top + br.height / 2 - sr.top);
        line.setAttribute("stroke", weight >= 0 ? COLORS.blue : COLORS.red); line.setAttribute("stroke-width", clamp(Math.abs(weight) * 2.4, .3, 5)); line.setAttribute("stroke-opacity", ".58"); edgeSvg.appendChild(line);
      }));
    });
  }
  function drawDatasetPreviews() {
    document.querySelectorAll("[data-dataset]").forEach(button => {
      const c = button.querySelector("canvas"), r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; c.width = r.width * dpr; c.height = r.height * dpr;
      const cx = c.getContext("2d"); cx.setTransform(dpr, 0, 0, dpr, 0, 0); cx.fillStyle = "#f7f8fb"; cx.fillRect(0, 0, r.width, r.height);
      const pts = button.dataset.dataset === "custom" ? [] : generate(button.dataset.dataset, true);
      pts.forEach(p => { cx.beginPath(); cx.arc(4 + (p.x + 3) / 6 * (r.width - 8), 4 + (3 - p.y) / 6 * (r.height - 8), 1.8, 0, Math.PI * 2); cx.fillStyle = p.label ? COLORS.blue : COLORS.red; cx.fill(); });
      if (!pts.length) { cx.strokeStyle = "#9aa5b8"; cx.setLineDash([3, 3]); cx.strokeRect(7, 7, r.width - 14, r.height - 14); }
    });
  }

  function eventPos(e) { const r = dataCanvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
  function editAt(pos) {
    if (state.tool === "erase") {
      state.points = state.points.filter(p => { const q = toCanvas(dataCanvas, p.x, p.y); return Math.hypot(q.x - pos.x, q.y - pos.y) > state.brush; });
    } else {
      const rng = new RNG(Math.floor(pos.x * 31 + pos.y * 17 + state.points.length * 101)), count = state.dragging ? 2 : 1;
      for (let i = 0; i < count; i++) {
        const a = rng.next() * Math.PI * 2, radius = Math.sqrt(rng.next()) * state.brush, q = toData(dataCanvas, pos.x + Math.cos(a) * radius, pos.y + Math.sin(a) * radius);
        state.points.push({ id: state.nextId++, x: q.x, y: q.y, label: state.tool === "blue" ? 1 : 0, test: false });
      }
    }
    state.dataset = "custom"; document.querySelectorAll("[data-dataset]").forEach(b => b.classList.toggle("active", b.dataset.dataset === "custom"));
    state.bounds = calculateBounds(); resetNetwork(); drawData(); drawOutput();
  }
  dataCanvas.addEventListener("pointerdown", e => { state.dragging = true; dataCanvas.setPointerCapture(e.pointerId); editAt(eventPos(e)); });
  dataCanvas.addEventListener("pointermove", e => { if (state.dragging) editAt(eventPos(e)); });
  dataCanvas.addEventListener("pointerup", () => state.dragging = false);

  function updateSummaries() {
    $("networkSummary").textContent = `${state.layers.length} ${state.layers.length === 1 ? "capa oculta" : "capas ocultas"}`;
    $("trainingSummary").textContent = `${$("activation").options?.[$("activation").selectedIndex]?.text || $("activation").value} · η ${$("learningRate").value}`;
    $("datasetLabel").textContent = DATASET_LABELS[state.dataset];
    $("toolLabel").textContent = TOOL_LABELS[state.tool];
  }
  function updateAll() { drawData(); drawOutput(); drawLoss(); renderNetwork(); drawDatasetPreviews(); $("splitValue").textContent = `${Math.round(state.trainSplit * 100)}%`; updateSummaries(); }
  document.querySelectorAll("[data-dataset]").forEach(b => b.addEventListener("click", () => { document.querySelectorAll("[data-dataset]").forEach(x => x.classList.toggle("active", x === b)); setDataset(b.dataset.dataset); }));
  $("toolButtons").querySelectorAll("[data-tool]").forEach(b => b.addEventListener("click", () => { state.tool = b.dataset.tool; document.querySelectorAll(".tool").forEach(x => x.classList.toggle("active", x === b)); $("toolLabel").textContent = TOOL_LABELS[state.tool]; }));
  $("brushSize").addEventListener("input", e => { state.brush = Number(e.target.value); $("brushValue").textContent = `${state.brush} px`; });
  $("trainSplit").addEventListener("input", e => { state.trainSplit = Number(e.target.value) / 100; $("splitValue").textContent = `${e.target.value}%`; splitData(); resetNetwork(); drawData(); });
  $("clearButton").addEventListener("click", () => { state.points = []; state.dataset = "custom"; state.bounds = calculateBounds(); resetNetwork(); updateAll(); });
  $("playButton").addEventListener("click", play); $("stepButton").addEventListener("click", step); $("resetButton").addEventListener("click", resetNetwork);
  ["learningRate", "regularization", "batchSize"].forEach(id => $(id).addEventListener("change", () => { stop(); updateSummaries(); }));
  $("activation").addEventListener("change", () => { resetNetwork(); updateSummaries(); });
  document.querySelectorAll("[data-feature]").forEach(input => input.addEventListener("change", () => {
    state.features = [...document.querySelectorAll("[data-feature]:checked")].map(x => x.dataset.feature);
    if (!state.features.length) { input.checked = true; state.features = [input.dataset.feature]; }
    resetNetwork();
  }));
  $("addLayer").addEventListener("click", () => { if (state.layers.length < 4) { state.layers.push(3); resetNetwork(); updateSummaries(); } });
  $("predictButton").addEventListener("click", () => {
    const x = Number($("predictX").value), y = Number($("predictY").value), out = forward(inputVector(x, y)).output;
    $("predictionText").innerHTML = `Clase <b>${out >= .5 ? "+1" : "−1"}</b> · probabilidad de +1: <b>${(out * 100).toFixed(1)}%</b>.`;
  });
  document.querySelectorAll(".top-menu-trigger").forEach(trigger => trigger.addEventListener("click", e => {
    e.stopPropagation();
    const menu = $(trigger.dataset.menu), wasOpen = menu.classList.contains("open");
    document.querySelectorAll(".top-popover").forEach(p => p.classList.remove("open"));
    document.querySelectorAll(".top-menu-trigger").forEach(b => b.classList.remove("open"));
    if (!wasOpen) {
      menu.classList.add("open"); trigger.classList.add("open");
      requestAnimationFrame(() => { if (menu.id === "dataMenu") drawDatasetPreviews(); });
    }
  }));
  document.querySelectorAll(".top-popover").forEach(menu => menu.addEventListener("click", e => e.stopPropagation()));
  document.addEventListener("click", () => {
    document.querySelectorAll(".top-popover").forEach(p => p.classList.remove("open"));
    document.querySelectorAll(".top-menu-trigger").forEach(b => b.classList.remove("open"));
  });
  document.querySelectorAll(".output-dropdown").forEach(panel => panel.addEventListener("toggle", () => {
    if (panel.open) requestAnimationFrame(() => { drawLoss(); drawOutput(); });
  }));
  window.addEventListener("resize", () => { clearTimeout(updateAll.resizeTimer); updateAll.resizeTimer = setTimeout(updateAll, 100); });

  requestAnimationFrame(() => { setDataset("moons"); drawDatasetPreviews(); });
})();
