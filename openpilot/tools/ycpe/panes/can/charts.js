// cabana's charts, minimal until the shared plot pane lands: decoded signals over the shared x-range, merged by drag,
// as lines, steps or dots, with value tips. Buckets come from the CAN worker, so drawing is O(pixels).
import {h} from "/dom.js";
import {canOf, subs} from "./common.js";

const DRAG = "application/x-ycpe-paths";
const PAD = {l: 46, r: 8, t: 6, b: 16};

export function create(ctx, state) {
  const can = canOf(ctx), s = subs();
  let charts = state.charts || [], cols = state.cols || 1, style = state.style || "line", cursor = null, views = [], size = new Map();
  const grid = h("div", {class: "can-charts-grid"});
  const select = (title, options, value, fn) => h("select", {class: "can-select", title, onchange: (e) => fn(e.target.value)},
    options.map(([v, label]) => h("option", {value: v, selected: v === value}, label)));
  const el = h("div", {class: "can-pane can-charts", ondragover: (e) => e.preventDefault(), ondrop: (e) => drop(e, null)},
    h("div", {class: "can-toolbar"}, h("span", {class: "can-muted"}, "Drop signals here; drag a chart's legend onto another to merge"),
      h("span", {class: "can-spacer"}),
      select("Style", ["line", "step", "scatter"].map((x) => [x, x]), style, (v) => { style = v; ctx.perf.invalidate(pane); }),
      select("Columns", [1, 2, 3].map((n) => [String(n), `${n} col`]), String(cols), (v) => { cols = +v; build(); }),
      h("button", {class: "can-btn", onclick: () => { charts = []; build(); }}, "Clear")), grid);
  const resizer = new ResizeObserver((entries) => {
    for (const e of entries) size.set(e.target, [e.contentRect.width, e.contentRect.height]);
    ctx.perf.invalidate(pane);
  });

  function add(paths, into) {
    if (into) into.paths = [...new Set([...into.paths, ...paths])];
    else charts.push({paths});
    build();
  }

  function drop(e, chart) {
    e.preventDefault();
    e.stopPropagation();
    const from = e.dataTransfer.getData("text/x-can-chart");
    if (from !== "") {
      const src = charts[+from];
      if (src && src !== chart) {
        charts = charts.filter((c) => c !== src);
        add(src.paths, chart);
      }
      return;
    }
    const paths = JSON.parse(e.dataTransfer.getData(DRAG) || "[]");
    if (paths.length) add(paths, chart);
  }

  function build() {
    resizer.disconnect();
    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    views = charts.map((chart, i) => {
      const canvas = h("canvas", {class: "can-chart-canvas"});
      const keys = chart.paths.map((p) => h("i"));
      const legend = h("div", {class: "can-chart-legend", draggable: true, ondragstart: (e) => e.dataTransfer.setData("text/x-can-chart", String(i))},
        chart.paths.map((p, j) => h("span", {class: "can-chart-key", title: p}, keys[j], p.split("/").slice(-2).join("."),
          h("button", {class: "can-icon", title: "Remove", onclick: () => {
            chart.paths = chart.paths.filter((x) => x !== p);
            if (!chart.paths.length) charts.splice(charts.indexOf(chart), 1);
            build();
          }}, "✕"))));
      canvas.addEventListener("pointermove", (e) => { cursor = {chart, x: e.offsetX}; ctx.perf.invalidate(pane); });
      canvas.addEventListener("pointerleave", () => { cursor = null; ctx.perf.invalidate(pane); });
      canvas.addEventListener("click", (e) => ctx.clock.seek(xToT(e.offsetX, size.get(canvas)?.[0] || 1)));
      canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const [t0, t1] = ctx.clock.range, t = xToT(e.offsetX, size.get(canvas)?.[0] || 1), z = Math.exp(e.deltaY * 0.002);
        ctx.clock.setRange(t - (t - t0) * z, t + (t1 - t) * z);
      }, {passive: false});
      resizer.observe(canvas);
      return {chart, canvas, keys, el: h("div", {class: "can-chart", ondragover: (e) => e.preventDefault(), ondrop: (e) => drop(e, chart)}, legend, canvas)};
    });
    grid.replaceChildren(...views.map((v) => v.el));
    ctx.perf.invalidate(pane);
  }

  const xToT = (x, W) => { const [t0, t1] = ctx.clock.range; return t0 + (x - PAD.l) / (W - PAD.l - PAD.r) * (t1 - t0); };

  // one bucket per pixel from the worker, refetched when the range, size or data change
  async function fetch(v, w, sig) {
    v.pending = sig;
    const [t0, t1] = ctx.clock.range, lines = [], groups = new Map();
    v.chart.paths.forEach((p, j) => {
      const q = can.parse(p);
      if (q) (groups.get(q.k) || groups.set(q.k, []).get(q.k)).push([j, q.sig.name]);
    });
    await Promise.all([...groups].map(async ([k, list]) => {
      const r = await can.decimate(k, list.map((x) => x[1]), t0, t1, w);
      list.forEach(([j], i) => { lines[j] = r[i]; });
    }));
    v.data = {sig, lines, t0, t1};
    v.pending = null;
    ctx.perf.invalidate(pane);
  }

  const pane = {
    name: "can-charts", el,
    render() {
      const [fg, line, muted, mono] = ["--fg", "--line", "--muted", "--mono"].map((x) => ctx.theme.color(x)), [t0, t1] = ctx.clock.range, dpr = devicePixelRatio;
      for (const v of views) {
        const [W, H] = size.get(v.canvas) || [0, 0];
        if (!W || !H) continue;
        if (v.canvas.width !== W * dpr || v.canvas.height !== H * dpr) { v.canvas.width = W * dpr; v.canvas.height = H * dpr; }
        const w = Math.floor(W - PAD.l - PAD.r), ph = H - PAD.t - PAD.b, sig = `${t0},${t1},${w},${can.dataVersion}`;
        if (v.data?.sig !== sig && v.pending !== sig) fetch(v, w, sig);
        const g = v.canvas.getContext("2d");
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, W, H);
        const lines = v.data?.lines || [];
        let min = Infinity, max = -Infinity;
        for (const d of lines) for (let x = 0; x < (d?.min.length || 0); x++) { if (d.min[x] < min) min = d.min[x]; if (d.max[x] > max) max = d.max[x]; }
        if (min === Infinity) { min = 0; max = 1; } else if (min === max) { min -= 1; max += 1; }
        const y = (val) => PAD.t + ph - (val - min) / (max - min) * ph;
        // axes
        g.font = `10px ${mono}`;
        g.fillStyle = muted;
        g.strokeStyle = line;
        g.lineWidth = 1;
        g.textAlign = "right";
        g.textBaseline = "middle";
        g.beginPath();
        for (let k = 0; k <= 3; k++) {
          const val = min + (max - min) * k / 3, yy = Math.round(y(val)) + 0.5;
          g.moveTo(PAD.l, yy);
          g.lineTo(PAD.l + w, yy);
          g.fillText(+val.toPrecision(4), PAD.l - 4, yy);
        }
        g.stroke();
        g.textAlign = "center";
        g.textBaseline = "top";
        for (let k = 0; k <= 4; k++) g.fillText((t0 + (t1 - t0) * k / 4).toFixed(1), PAD.l + w * k / 4, PAD.t + ph + 3);
        // series, shifted when the range moved since the buckets were made
        const colors = v.chart.paths.map((p) => colorOf(can, p));
        const shift = v.data ? (v.data.t0 - t0) / (t1 - t0) * w : 0, scale = v.data ? (v.data.t1 - v.data.t0) / (t1 - t0) : 1;
        lines.forEach((d, j) => {
          if (!d) return;
          g.strokeStyle = g.fillStyle = colors[j];
          g.lineWidth = 1.25;
          g.beginPath();
          let prev = NaN, started = false;
          for (let x = 0; x < d.first.length; x++) {
            if (d.first[x] !== d.first[x]) continue;
            const X = PAD.l + shift + x * scale + 0.5;
            if (style === "scatter") { g.fillRect(X - 1, y(d.max[x]) - 1, 2, Math.max(2, y(d.min[x]) - y(d.max[x]) + 2)); continue; }
            if (!started) { g.moveTo(X, y(d.first[x])); started = true; } else if (style === "step") g.lineTo(X, y(prev));
            g.lineTo(X, y(d.first[x]));
            g.lineTo(X, y(d.min[x]));
            g.lineTo(X, y(d.max[x]));
            g.lineTo(X, y(d.last[x]));
            prev = d.last[x];
          }
          if (style !== "scatter") g.stroke();
        });
        v.keys.forEach((k, j) => { if (k.style.background !== colors[j]) k.style.background = colors[j]; });
        // playhead, and the values under the cursor
        const px = Math.round(PAD.l + (ctx.clock.t - t0) / (t1 - t0) * w) + 0.5;
        g.strokeStyle = fg;
        g.beginPath();
        g.moveTo(px, PAD.t);
        g.lineTo(px, PAD.t + ph);
        g.stroke();
        if (cursor?.chart === v.chart && cursor.x >= PAD.l && cursor.x < PAD.l + w) {
          const bx = Math.round((cursor.x - PAD.l - shift) / scale), right = cursor.x > W / 2, tx = cursor.x + (right ? -6 : 6);
          g.strokeStyle = muted;
          g.setLineDash([3, 3]);
          g.beginPath();
          g.moveTo(cursor.x + 0.5, PAD.t);
          g.lineTo(cursor.x + 0.5, PAD.t + ph);
          g.stroke();
          g.setLineDash([]);
          g.textAlign = right ? "right" : "left";
          lines.forEach((d, j) => {
            let val = NaN;
            for (let x = Math.min(bx, (d?.last.length || 0) - 1); x >= 0 && val !== val; x--) val = d.last[x];
            g.fillStyle = colors[j];
            g.fillText(`${v.chart.paths[j].split("/").pop()} ${val === val ? +val.toPrecision(6) : "–"}`, tx, PAD.t + 2 + j * 12);
          });
          g.fillStyle = muted;
          g.fillText(`${xToT(cursor.x, W).toFixed(3)} s`, tx, PAD.t + 2 + lines.length * 12);
        }
      }
    },
    resize() { ctx.perf.invalidate(pane); },
    dispose() { s.dispose(); resizer.disconnect(); },
    save: () => ({charts, cols, style}),
  };
  s.add(can.on("chart", (path) => { if (!charts.some((c) => c.paths.includes(path))) add([path]); }));
  for (const ev of ["dbc", "messages"]) s.add(can.on(ev, () => ctx.perf.invalidate(pane)));
  for (const ev of ["t", "range"]) s.add(ctx.clock.on(ev, () => ctx.perf.invalidate(pane)));
  s.add(ctx.theme.on(() => { colorCache.clear(); ctx.perf.invalidate(pane); }));
  build();
  return pane;
}

const colorCache = new Map();
function colorOf(can, path) {
  const c = can.parse(path)?.sig.color || "gray";
  return colorCache.get(c) || colorCache.set(c, c.replace(/(\d+(\.\d+)?)%\)$/, "45%)")).get(c);
}
