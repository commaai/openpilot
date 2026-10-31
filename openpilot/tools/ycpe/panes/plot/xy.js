// XY plot: y against x over the visible time range, with the playhead as a moving point. The pairing is computed
// in the transforms worker; the main thread draws at most a few thousand points per curve, on a base canvas that
// redraws only when the results or the view change. The playhead points go on an overlay.
import {h} from "/dom.js";
import {icon} from "/icons.js";
import {colors, fitCanvas, ticks, axes, fmtTick, fmtValue, nextColor, marker, clip} from "./draw.js";
import * as view from "./view.js";
import {menu} from "./menu.js";
import {exportPng} from "./export.js";
import {compute, lease} from "/data/transforms.js";

const L = 58, R = 10, T = 8, B = 20;

export function create(ctx, state = {}) {
  const {store, clock, perf} = ctx;
  let col = colors(ctx.theme);
  // a layout saved in the FFT mode there was: its curves (of one series each) and view go
  const fft = state.mode === "fft";
  const st = {curves: fft ? [] : state.curves || [], equal: !!state.equal, view: fft ? null : state.view || null};
  let pendingX = null, results = [], busy = false, stale = true, W = 0, H = 0, g = null, gOver = null, frame = null, drag = null;
  let dirty = true, timer = 0;

  const base = h("canvas"), cv = h("canvas", {class: "over", tabindex: 0});
  const legend = h("div", {class: "legend"});
  const hint = h("div", {class: "empty"});
  const el = h("div", {class: "plot-pane xy"}, base, cv, legend, hint);
  const pane = {name: "xy", el, render, resize, dispose, save};
  const invalidate = () => { dirty = true; perf.invalidate(pane); };
  const offs = [
    clock.on("range", refresh),
    clock.on("t", () => perf.invalidate(pane)),
    store.on("segment", () => { timer ||= setTimeout(() => { timer = 0; refresh(); }, view.RECOMPUTE_DEBOUNCE_MS); }),
    ctx.theme.on(() => { col = colors(ctx.theme); invalidate(); }),
    store.can?.on("series", (path) => { if (st.curves.some((c) => inputs(c).includes(path))) refresh(); }),
    // a new version of a custom series or a transform
    store.on("put", (path) => { if (st.curves.some((c) => inputs(c).includes(path))) refresh(); }),
  ].filter(Boolean);

  const inputs = (c) => [c.x, c.y];
  const name = (c) => `${c.y.slice(1)} vs ${c.x.slice(1)}`;

  // one request in flight; a change while busy asks again when it lands
  function refresh() {
    stale = true;
    if (busy) return;
    const curves = st.curves.filter((c) => inputs(c).every((p) => store.series(p)));
    lease(pane, store, st.curves.flatMap(inputs));  // the worker keeps these, and forgets what no pane reads
    showHint(st.curves.some((c) => inputs(c).some((p) => p.startsWith("/custom/") && !store.series(p))));
    if (!curves.length) { results = []; invalidate(); return; }
    busy = true;
    stale = false;
    const [t0, t1] = clock.range;
    Promise.all(curves.map((c) => compute(store, "xy", {x: c.x, y: c.y, t0, t1}, [c.x, c.y])))
      .then((rs) => { results = rs.map((r, k) => ({c: curves[k], ...r})); }, console.error)
      .finally(() => { busy = false; if (stale) refresh(); invalidate(); });
  }

  function addPaths(paths) {
    const q = pendingX ? [pendingX, ...paths] : [...paths];
    while (q.length >= 2) st.curves.push({x: q.shift(), y: q.shift(), color: nextColor(st.curves.map((c) => c.color))});
    pendingX = q[0] ?? null;
    rebuild();
  }

  // after a change to the curves: the legend anew, and the pairs computed again
  function rebuild() {
    legend.replaceChildren(...st.curves.map((c) => h("div", {class: "row"},
      h("span", {class: "swatch", style: `color:${c.color}`}), h("span", {class: "name", title: name(c)}, h("bdi", {}, name(c))), h("span"),
      h("button", {class: "x", title: "remove", "aria-label": `Remove ${name(c)}`, onclick: () => { st.curves.splice(st.curves.indexOf(c), 1); rebuild(); }}, icon("close", 12)))));
    legend.hidden = !st.curves.length;
    showHint(false);
    refresh();
  }

  // what to drop next, or that a custom series isn't there: they're computed by the time plots that show them
  function showHint(uncomputed) {
    hint.hidden = st.curves.length && !pendingX && !uncomputed;
    hint.textContent = uncomputed ? "custom series not computed: add it to a time plot"
      : pendingX ? `x = ${pendingX}, now drop the y series` : "drop an x series, then a y series (or two at once)";
  }

  // data bounds with 5% padding, widened to equal units per pixel when asked
  function bounds() {
    if (st.view) return st.view;
    let [x0, x1, y0, y1] = [Infinity, -Infinity, Infinity, -Infinity];
    for (const {bounds: [a, b, c, d]} of results) {
      [x0, x1, y0, y1] = [Math.min(x0, a), Math.max(x1, b), Math.min(y0, c), Math.max(y1, d)];
    }
    if (!(x1 >= x0)) [x0, x1, y0, y1] = [0, 1, 0, 1];
    const px = Math.max((x1 - x0) * 0.05, 1e-9), py = Math.max((y1 - y0) * 0.05, 1e-9);
    [x0, x1, y0, y1] = [x0 - px, x1 + px, y0 - py, y1 + py];
    if (st.equal) {
      const pw = W - L - R, ph = H - T - B, u = Math.max((x1 - x0) / pw, (y1 - y0) / ph);
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      [x0, x1, y0, y1] = [cx - u * pw / 2, cx + u * pw / 2, cy - u * ph / 2, cy + u * ph / 2];
    }
    return [x0, x1, y0, y1];
  }

  function render() {
    if (!g) return;
    if (dirty) drawBase();
    dirty = false;
    drawOverlay();
  }

  function drawBase() {
    const [x0, x1, y0, y1] = bounds(), r = {x: L, y: T, w: W - L - R, h: H - T - B};
    const X = (v) => r.x + (v - x0) / (x1 - x0) * r.w, Y = (v) => r.y + (y1 - v) / (y1 - y0) * r.h;
    frame = {x0, x1, y0, y1, r, X, Y};
    g.clearRect(0, 0, W, H);
    g.fillStyle = col.bg;
    g.fillRect(r.x, r.y, r.w, r.h);
    const xt = ticks(x0, x1, r.w / 90), yt = ticks(y0, y1, r.h / 34);
    axes(g, col, r, xt, X, (v) => fmtTick(v, xt.step), yt, Y);
    clip(g, r);
    g.lineWidth = 1.25;
    for (const {c, x, y} of results) {
      g.strokeStyle = c.color;
      g.beginPath();
      for (let i = 0; i < x.length; i++) g.lineTo(X(x[i]), Y(y[i]));
      g.stroke();
    }
    g.restore();
  }

  // the playhead's point on each XY curve, and the legend's values there
  function drawOverlay() {
    gOver.clearRect(0, 0, W, H);
    const {r, X, Y} = frame, g = gOver;
    clip(g, r);
    for (const res of results) {
      const px = store.valueAt(res.c.x, clock.t), py = store.valueAt(res.c.y, clock.t);
      if (typeof px === "number" && typeof py === "number") marker(g, X(px), Y(py), 4, res.c.color, col);
    }
    g.restore();
    legend.querySelectorAll(".row").forEach((row, k) => {
      const c = st.curves[k], v = `${sample(c.x)}, ${sample(c.y)}`;
      if (row.children[2].textContent !== v) row.children[2].textContent = v;
    });
  }

  const sample = (path) => fmtValue(store.valueAt(path, clock.t), store.meta(path));

  // ---- gestures: a local view (pan, wheel zoom); double-click returns to auto-fit

  const local = (e) => { const b = cv.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top]; };
  const at = ([x, y]) => [frame.x0 + (x - frame.r.x) / frame.r.w * (frame.x1 - frame.x0), frame.y1 - (y - frame.r.y) / frame.r.h * (frame.y1 - frame.y0)];

  cv.addEventListener("wheel", (e) => {
    if (!frame) return;
    e.preventDefault();
    const [x, y] = at(local(e)), f = Math.exp(e.deltaY * (e.deltaMode ? 33 : 1) * 0.002), [x0, x1, y0, y1] = bounds();
    st.view = [x - (x - x0) * f, x + (x1 - x) * f, y - (y - y0) * f, y + (y1 - y) * f];
    invalidate();
  }, {passive: false});
  cv.addEventListener("pointerdown", (e) => {
    if (!frame || e.button === 2) return;
    cv.setPointerCapture(e.pointerId);
    drag = {p: local(e), view: bounds()};
  });
  cv.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const [x, y] = local(e), [x0, x1, y0, y1] = drag.view;
    const dx = (x - drag.p[0]) / frame.r.w * (x1 - x0), dy = (y - drag.p[1]) / frame.r.h * (y1 - y0);
    st.view = [x0 - dx, x1 - dx, y0 + dy, y1 + dy];
    invalidate();
  });
  cv.addEventListener("pointerup", () => { drag = null; });
  const fit = () => { st.view = null; invalidate(); };
  cv.addEventListener("dblclick", fit);
  cv.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    menu(e.clientX, e.clientY, [
      {label: "Equal axes", checked: !!st.equal, fn: () => { st.equal = !st.equal; fit(); }},
      {label: "Swap X and Y", fn: () => { for (const c of st.curves) [c.x, c.y] = [c.y, c.x]; st.view = null; rebuild(); }},
      {label: "Fit", fn: fit},
      {label: "Export PNG", fn: () => exportPng([base, cv], st.curves.map((c) => ({color: c.color, name: name(c), value: ""})), col.bg, col.fg, col.font, "xy")},
      {label: "Remove all", disabled: !st.curves.length, fn: () => { st.curves = []; pendingX = null; rebuild(); }},
    ]);
  });

  view.acceptPaths(el, addPaths);

  function resize(w = el.clientWidth, hgt = el.clientHeight) {
    W = w;
    H = hgt;
    g = fitCanvas(base, W, H);
    gOver = fitCanvas(cv, W, H);
    invalidate();
  }

  function save() {
    return {curves: st.curves, ...(st.equal && {equal: true}), ...(st.view && {view: st.view})};
  }

  function dispose() {
    clearTimeout(timer);
    for (const off of offs) off();
    lease(pane, store);
  }

  rebuild();
  return pane;
}
