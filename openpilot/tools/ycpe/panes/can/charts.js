// cabana's charts, minimal until the shared plot pane lands: decoded signals over the shared x-range, merged by drag,
// as lines, steps or dots, with a value tip at the cursor
import {h} from "/dom.js";
import {canOf, subs} from "./common.js";

const DRAG = "application/x-ycpe-paths";
const PAD = {l: 46, r: 8, t: 6, b: 16};

export function create(ctx, state) {
  const can = canOf(ctx), s = subs();
  let charts = state.charts || [], cols = state.cols || 1, style = state.style || "line", cursor = null;
  const grid = h("div", {class: "can-charts-grid"});
  const colsSel = h("select", {class: "can-select", title: "Columns", onchange: (e) => { cols = +e.target.value; build(); }},
    [1, 2, 3].map((n) => h("option", {value: n, selected: n === cols}, `${n} col`)));
  const styleSel = h("select", {class: "can-select", title: "Style", onchange: (e) => { style = e.target.value; ctx.perf.invalidate(pane); }},
    ["line", "step", "scatter"].map((n) => h("option", {selected: n === style}, n)));
  const el = h("div", {class: "can-pane can-charts", ondragover: (e) => e.preventDefault(), ondrop: (e) => drop(e, null)},
    h("div", {class: "can-toolbar"}, h("span", {class: "can-muted"}, "Drop signals here, or drag a chart onto another to merge"),
      h("span", {class: "can-spacer"}), styleSel, colsSel,
      h("button", {class: "can-btn", onclick: () => { charts = []; build(); }}, "Clear")), grid);
  let views = [];
  const cache = new Map();  // chart -> decimated lines, keyed by range, size and series versions

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
      if (!src || src === chart) return;
      charts = charts.filter((c) => c !== src);
      add(src.paths, chart);
      return;
    }
    const paths = JSON.parse(e.dataTransfer.getData(DRAG) || "[]");
    if (paths.length) add(paths, chart);
  }

  function build() {
    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    views = charts.map((chart, i) => {
      const canvas = h("canvas", {class: "can-chart-canvas"});
      const legend = h("div", {class: "can-chart-legend", draggable: true, ondragstart: (e) => e.dataTransfer.setData("text/x-can-chart", String(i))},
        chart.paths.map((p) => h("span", {class: "can-chart-key", title: p}, h("i"), p.split("/").slice(-2).join("."),
          h("button", {class: "can-icon", title: "Remove", onclick: () => { chart.paths = chart.paths.filter((x) => x !== p); if (!chart.paths.length) charts.splice(charts.indexOf(chart), 1); build(); }}, "✕"))));
      const box = h("div", {class: "can-chart", ondragover: (e) => e.preventDefault(), ondrop: (e) => drop(e, chart)}, legend, canvas);
      canvas.addEventListener("pointermove", (e) => { cursor = {chart, x: e.offsetX}; ctx.perf.invalidate(pane); });
      canvas.addEventListener("pointerleave", () => { cursor = null; ctx.perf.invalidate(pane); });
      canvas.addEventListener("click", (e) => ctx.clock.seek(xToT(e.offsetX, canvas.clientWidth)));
      canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const [t0, t1] = ctx.clock.range, t = xToT(e.offsetX, canvas.clientWidth), z = Math.exp(e.deltaY * 0.002);
        ctx.clock.setRange(t - (t - t0) * z, t + (t1 - t) * z);
      }, {passive: false});
      return {chart, canvas, legend};
    });
    grid.replaceChildren(...views.map((v) => v.canvas.parentElement));
    ctx.perf.invalidate(pane);
  }

  const xToT = (x, w) => { const [t0, t1] = ctx.clock.range; return t0 + (x - PAD.l) / (w - PAD.l - PAD.r) * (t1 - t0); };

  // min/max per pixel over the range; recomputed only when the range, size or data change
  function lines(chart, w) {
    const [t0, t1] = ctx.clock.range, series = chart.paths.map((p) => can.series(p));
    const sig = `${t0},${t1},${w},${style},${series.map((x) => x?.version).join()}`;
    const hit = cache.get(chart);
    if (hit?.sig === sig) return hit;
    let min = Infinity, max = -Infinity;
    // per pixel column: [x, first, min, max, last]
    const out = series.map((sr) => {
      const pts = [];
      for (const c of sr?.chunks || []) {
        if (!c.t.length || c.t[0] > t1 || c.t[c.t.length - 1] < t0) continue;
        for (let i = Math.max(0, bisect(c.t, t0) - 1); i < c.t.length; i++) {
          const v = c.v[i];
          if (v !== v) continue;
          const x = Math.floor((c.t[i] - t0) / (t1 - t0) * w), p = pts.at(-1);
          if (p?.[0] === x) { p[2] = Math.min(p[2], v); p[3] = Math.max(p[3], v); p[4] = v; } else pts.push([x, v, v, v, v]);
          if (v < min) min = v;
          if (v > max) max = v;
          if (c.t[i] > t1) break;
        }
      }
      return pts;
    });
    if (min === max) { min -= 1; max += 1; }
    const r = {sig, out, min, max};
    cache.set(chart, r);
    return r;
  }

  const pane = {
    name: "can-charts", el,
    render() {
      const css = getComputedStyle(el), fg = css.getPropertyValue("--fg"), line = css.getPropertyValue("--line"), muted = css.getPropertyValue("--muted");
      const [t0, t1] = ctx.clock.range, dpr = devicePixelRatio;
      for (const {chart, canvas, legend} of views) {
        const W = canvas.clientWidth, H = canvas.clientHeight;
        if (!W || !H) continue;
        if (canvas.width !== W * dpr || canvas.height !== H * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; }
        const g = canvas.getContext("2d"), w = W - PAD.l - PAD.r, ph = H - PAD.t - PAD.b;
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, W, H);
        const {out, min, max} = lines(chart, w);
        const y = (v) => PAD.t + ph - (v - min) / (max - min) * ph;
        g.font = `10px ${css.getPropertyValue("--mono")}`;
        g.fillStyle = muted;
        g.strokeStyle = line;
        g.textAlign = "right";
        g.textBaseline = "middle";
        g.beginPath();
        for (let k = 0; k <= 3; k++) {
          const v = min + (max - min) * k / 3, yy = Math.round(y(v)) + 0.5;
          g.moveTo(PAD.l, yy);
          g.lineTo(PAD.l + w, yy);
          g.fillText(+v.toPrecision(4), PAD.l - 4, yy);
        }
        g.stroke();
        g.textAlign = "center";
        g.textBaseline = "top";
        for (let k = 0; k <= 4; k++) g.fillText((t0 + (t1 - t0) * k / 4).toFixed(1), PAD.l + w * k / 4, PAD.t + ph + 3);
        const colors = chart.paths.map((p) => colorOf(can, p));
        out.forEach((pts, j) => {
          g.strokeStyle = g.fillStyle = colors[j];
          g.lineWidth = 1.25;
          g.beginPath();
          pts.forEach(([x, first, lo, hi, last], k) => {
            const X = PAD.l + x + 0.5;
            if (style === "scatter") { g.fillRect(X - 1, y(hi) - 1, 2, Math.max(2, y(lo) - y(hi) + 2)); return; }
            if (k === 0) g.moveTo(X, y(first));
            else if (style === "step") g.lineTo(X, y(pts[k - 1][4]));
            g.lineTo(X, y(first));
            g.lineTo(X, y(lo));
            g.lineTo(X, y(hi));
            g.lineTo(X, y(last));
          });
          if (style !== "scatter") g.stroke();
        });
        legend.querySelectorAll("i").forEach((x, j) => { x.style.background = colors[j]; });
        // playhead and the value tip at the cursor
        const px = PAD.l + (ctx.clock.t - t0) / (t1 - t0) * w;
        g.strokeStyle = fg;
        g.beginPath();
        g.moveTo(Math.round(px) + 0.5, PAD.t);
        g.lineTo(Math.round(px) + 0.5, PAD.t + ph);
        g.stroke();
        if (cursor?.chart === chart && cursor.x >= PAD.l) {
          const t = xToT(cursor.x, W);
          g.strokeStyle = muted;
          g.setLineDash([3, 3]);
          g.beginPath();
          g.moveTo(cursor.x + 0.5, PAD.t);
          g.lineTo(cursor.x + 0.5, PAD.t + ph);
          g.stroke();
          g.setLineDash([]);
          const tips = chart.paths.map((p) => `${p.split("/").pop()} ${fmt(valueAt(can.series(p), t))}`);
          g.textAlign = cursor.x > W / 2 ? "right" : "left";
          g.textBaseline = "top";
          tips.forEach((tip, j) => { g.fillStyle = colors[j]; g.fillText(tip, cursor.x + (cursor.x > W / 2 ? -6 : 6), PAD.t + 2 + j * 12); });
          g.fillStyle = muted;
          g.fillText(`${t.toFixed(3)} s`, cursor.x + (cursor.x > W / 2 ? -6 : 6), PAD.t + 2 + tips.length * 12);
        }
      }
    },
    resize() { ctx.perf.invalidate(pane); },
    dispose() { s.dispose(); },
    save: () => ({charts, cols, style}),
  };
  s.add(can.on("chart", (path) => {
    if (!charts.some((c) => c.paths.includes(path))) add([path]);
  }));
  s.add(can.on("series", () => ctx.perf.invalidate(pane)));
  s.add(can.on("dbc", () => { cache.clear(); ctx.perf.invalidate(pane); }));
  s.add(ctx.clock.on("t", () => ctx.perf.invalidate(pane)));
  s.add(ctx.clock.on("range", () => ctx.perf.invalidate(pane)));
  s.add(ctx.theme.on(() => ctx.perf.invalidate(pane)));
  build();
  return pane;
}

const fmt = (v) => (v === undefined || Number.isNaN(v) ? "–" : +v.toPrecision(6));

function colorOf(can, path) {
  const p = can.parse(path);
  return p ? p.sig.color.replace(/(\d+(\.\d+)?)%\)$/, "45%)") : "gray";
}

function valueAt(series, t) {
  for (const c of series?.chunks || []) {
    if (!c.t.length || c.t[0] > t || c.t[c.t.length - 1] < t) continue;
    return c.v[Math.max(0, bisect(c.t, t + 1e-9) - 1)];
  }
}

function bisect(a, x) {
  let lo = 0, hi = a.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < x) lo = m + 1; else hi = m; }
  return lo;
}
