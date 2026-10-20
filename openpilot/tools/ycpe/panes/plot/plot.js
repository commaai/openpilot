// time plot: curves over the shared x-range with a hover tracker, a legend of values, state blocks for enums
// and text, and the view gestures. Two canvases: the base (grid, curves) redraws only when the view or the
// data changes; the overlay (playhead, tracker, zoom box) redraws on every cursor move. Per-frame work is
// O(pixels): curves come from store.decimate and state blocks from cached runs.
import {h} from "/dom.js";
import {colors, fitCanvas, ticks, fmtTime, fmtTick, fmtValue, strokeSeries, drawBlocks, nextColor, ellipsize} from "./draw.js";
import {blocks} from "./blocks.js";
import * as view from "./view.js";
import {menu, popup, curveEditor, close as closeMenu} from "./menu.js";
import {TRANSFORMS, derivedPath, derive, compute} from "/data/transforms.js";
import {evaluate} from "/data/custom.js";
import {download, exportPng} from "./export.js";

const L = 58, R = 8, T = 6, B = 20, STRIP = 18, DRAG = 4;
const STEPPED = new Set(["b", "e", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64"]);
export const PATHS_MIME = "application/x-ycpe-paths";


export function create(ctx, state = {}) {
  const {store, clock, perf} = ctx;
  let col = colors(ctx.theme);
  const st = {title: state.title || "", curves: [], y: {...state.y}, stats: !!state.stats};
  const base = h("canvas"), over = h("canvas", {class: "over", tabindex: 0});
  const legend = h("div", {class: "legend"});
  const empty = h("div", {class: "empty"}, "drag series here");
  const el = h("div", {class: "plot-pane"}, base, over, legend, empty);
  let W = 0, H = 0, gBase = null, gOver = null, dirty = true;
  let frame = null;    // geometry and prepared curves of the last base draw
  let gesture = null;  // the pointer gesture in progress
  let rows = [];       // legend rows, parallel to st.curves
  let statsTimer = 0, statsKey = "";
  const made = new Map();  // derived or custom path → input version it was computed from
  const busy = new Set();

  const pane = {
    name: "plot", el, render, resize, dispose, save, addCurves,
    get y() { return st.y; },
    setY(y) { st.y = {...y}; invalidate(); },
    overlay: () => perf.invalidate(pane),
  };
  const invalidate = () => { dirty = true; perf.invalidate(pane); };

  const offs = [
    clock.on("range", invalidate),
    clock.on("t", pane.overlay),
    store.on("segment", invalidate),
    store.on("paths", invalidate),
    ctx.theme.on(() => { col = colors(ctx.theme); invalidate(); }),
  ];
  view.join(pane);

  // ---- curves

  function addCurves(paths) {
    for (const path of paths) if (!st.curves.some((c) => c.path === path && !c.transform && !c.custom)) st.curves.push(normalize({path}));
    rebuildLegend();
    invalidate();
  }

  function normalize(c) {
    return {...c, color: c.color || nextColor(st.curves.map((x) => x.color))};
  }

  function removeCurve(c) {
    st.curves.splice(st.curves.indexOf(c), 1);
    rebuildLegend();
    invalidate();
  }

  // the store path a curve draws; transforms and custom series are (re)computed when their inputs change
  function source(c) {
    if (!c.transform && !c.custom) return c.path;
    const src = c.custom ? c.path : derivedPath(c.path, c.transform);
    const inputs = c.custom ? c.custom.inputs : [c.path];
    const version = inputs.map((p) => store.series(p)?.version).join();
    if (inputs.every((p) => store.series(p)) && made.get(src) !== version && !busy.has(src)) {
      busy.add(src);
      const job = c.custom ? evaluate(store, c.path, c.custom) : derive(store, c.path, c.transform);
      job.then(() => { made.set(src, version); c.error = null; }, (e) => { made.set(src, version); c.error = e.message; })
        .finally(() => { busy.delete(src); rebuildLegend(); invalidate(); });
    }
    return src;
  }

  const label = (c) => (c.label || c.path.replace(/^\//, "")) + (c.transform ? ` [${TRANSFORMS[c.transform.kind].label.toLowerCase()}]` : "");

  // ---- drawing

  function prepare(t0, t1, px) {
    const numeric = [], text = [];
    for (const c of st.curves) {
      if (c.hidden) continue;
      const src = source(c), s = store.series(src);
      if (!s) continue;
      const item = {c, src, s, shift: c.shift || 0, scale: c.scale ?? 1, offset: c.offset ?? 0};
      if (s.type === "t") text.push(item);
      else numeric.push(item);
    }
    // enums alone become rows of state blocks, like text; mixed with numbers they plot as steps
    const blockMode = numeric.length > 0 && numeric.every((it) => it.s.type === "e");
    const rows = blockMode ? [...numeric, ...text] : text;
    const lines = blockMode ? [] : numeric;
    for (const it of rows) it.blocks = blocks(it.s, t0 - it.shift, t1 - it.shift, px);
    for (const it of lines) {
      const span = t1 - t0, a = t0 - it.shift - span * 0.05, b = t1 - it.shift + span * 0.05;
      it.d = withEdges(store.decimate(it.src, a, b, Math.round(px * 1.1)), a, b, store.valueAt(it.src, a), store.valueAt(it.src, b));
      it.style = it.c.style || (STEPPED.has(it.s.type) ? "steps" : "line");
    }
    return {rows, lines, blockMode};
  }

  // held values at the query edges, so series sparser than the view still reach both sides
  function withEdges(d, a, b, va, vb) {
    if (!d) return d;
    const n = d.t.length, pre = typeof va === "number" && !(d.t[0] <= a), post = typeof vb === "number" && !(d.t[n - 1] >= b);
    if (!pre && !post) return d;
    const m = n + pre + post, t = new Float64Array(m), min = new Float64Array(m), max = new Float64Array(m);
    t.set(d.t, +pre);
    min.set(d.min, +pre);
    max.set(d.max, +pre);
    if (pre) [t[0], min[0], max[0]] = [a, va, va];
    if (post) [t[m - 1], min[m - 1], max[m - 1]] = [b, vb, vb];
    return {t, min, max};
  }

  function yRange(lines, t0, t1) {
    let lo = Infinity, hi = -Infinity;
    for (const {d, shift, scale, offset} of lines) {
      if (!d) continue;
      for (let i = 0; i < d.t.length; i++) {
        const t = d.t[i] + shift;
        if (t < t0 || t > t1 || d.min[i] !== d.min[i]) continue;
        const a = d.min[i] * scale + offset, b = d.max[i] * scale + offset;
        lo = Math.min(lo, a, b);
        hi = Math.max(hi, a, b);
      }
    }
    if (lo > hi) [lo, hi] = [0, 1];
    if (lines.length && lines.every((it) => it.s.type === "b")) [lo, hi] = [Math.min(lo, 0), Math.max(hi, 1)];
    const pad = hi > lo ? (hi - lo) * 0.05 : Math.max(Math.abs(lo) * 0.1, 0.1);
    lo = st.y.min ?? lo - pad;
    hi = st.y.max ?? hi + pad;
    return hi > lo ? [lo, hi] : [lo - 0.5, lo + 0.5];
  }

  function drawBase() {
    const g = gBase, [t0, t1] = clock.range;
    const px = L, pw = W - L - R, ph0 = H - T - B;
    if (pw < 4 || ph0 < 4) return;
    const p = prepare(t0, t1, pw);
    const stripH = p.blockMode ? ph0 / Math.max(1, p.rows.length) : Math.min(STRIP, ph0 / 4 / Math.max(1, p.rows.length));
    const py = T + (p.blockMode ? ph0 : p.rows.length * stripH), ph = T + ph0 - py;
    const [lo, hi] = p.blockMode ? [0, 1] : yRange(p.lines, t0, t1);
    const X = (t) => px + (t - t0) / (t1 - t0) * pw, Y = (v) => py + (hi - v) / (hi - lo) * ph;
    frame = {t0, t1, lo, hi, px, py, pw, ph, X, Y, ...p};

    g.clearRect(0, 0, W, H);
    g.fillStyle = col.bg;
    g.fillRect(px, T, pw, ph0);
    g.font = col.font;
    g.lineWidth = 1;

    // grid and tick labels
    const xt = ticks(t0, t1, pw / 90);
    g.strokeStyle = col.grid;
    g.beginPath();
    for (const t of xt.values) { const x = Math.round(X(t)) + 0.5; g.moveTo(x, T); g.lineTo(x, T + ph0); }
    const yt = p.blockMode ? {values: []} : ticks(lo, hi, ph / 34);
    for (const v of yt.values) { const y = Math.round(Y(v)) + 0.5; g.moveTo(px, y); g.lineTo(px + pw, y); }
    g.stroke();
    if (!p.blockMode && lo < 0 && hi > 0) {
      g.strokeStyle = col.zero;
      g.beginPath();
      g.moveTo(px, Math.round(Y(0)) + 0.5);
      g.lineTo(px + pw, Math.round(Y(0)) + 0.5);
      g.stroke();
    }
    g.fillStyle = col.text;
    g.textAlign = "center";
    g.textBaseline = "top";
    for (const t of xt.values) {
      const s = fmtTime(t, xt.step, t1 >= 120), w = g.measureText(s).width / 2;
      g.fillText(s, Math.min(X(t), W - w - 1), T + ph0 + 5);
    }
    g.textAlign = "right";
    g.textBaseline = "middle";
    for (const v of yt.values) g.fillText(fmtTick(v, yt.step), px - 6, Y(v));

    // state block rows, labeled in the gutter
    g.save();
    g.beginPath();
    g.rect(px, T, pw, ph0);
    g.clip();
    p.rows.forEach((it, k) => drawBlocks(g, it.blocks, (t) => X(t + it.shift), T + k * stripH, T + (k + 1) * stripH - 1, it.s.names, col));
    // curves
    for (const it of p.lines) {
      if (!it.d) continue;
      const {shift, scale, offset} = it;
      g.strokeStyle = g.fillStyle = it.c.color;
      g.globalAlpha = st.hl === undefined || st.curves[st.hl] === it.c ? 1 : 0.35;
      strokeSeries(g, it.d, (t) => X(t + shift), (v) => Y(v * scale + offset), it.style, st.curves[st.hl] === it.c ? 2.5 : 1.5);
    }
    g.globalAlpha = 1;
    drawInvalid(g, p.lines, X, py + ph);
    g.restore();

    g.fillStyle = col.text;
    g.textAlign = "right";
    g.textBaseline = "middle";
    p.rows.forEach((it, k) => g.fillText(ellipsize(g, label(it.c).split("/").pop(), px - 10), px - 6, T + (k + 0.5) * stripH));
  }

  // valid=false spans of the services behind the curves, as a band along the bottom
  function drawInvalid(g, lines, X, bottom) {
    const services = new Set(lines.filter((it) => it.src === it.c.path).map((it) => it.c.path.split("/")[1]));
    g.fillStyle = col.invalid;
    const [t0, t1] = clock.range;
    for (const svc of services) {
      const d = store.decimate(`/${svc}/valid`, t0, t1, Math.round(W / 2));
      if (!d) continue;
      const w = Math.max(2, (W - L - R) / Math.max(1, d.t.length));
      for (let i = 0; i < d.t.length; i++) if (d.min[i] === 0) g.fillRect(X(d.t[i]) - w / 2, bottom - 4, w, 4);
    }
  }

  function drawOverlay() {
    const g = gOver;
    g.clearRect(0, 0, W, H);
    if (!frame) return;
    const {px, pw, py, ph, X, Y, t0, t1} = frame, top = T, bottom = T + H - T - B;
    const vline = (t, color, dash) => {
      if (t < t0 || t > t1) return;
      const x = Math.round(X(t)) + 0.5;
      g.strokeStyle = color;
      g.setLineDash(dash);
      g.beginPath();
      g.moveTo(x, top);
      g.lineTo(x, bottom);
      g.stroke();
      g.setLineDash([]);
    };
    g.lineWidth = 1;
    vline(clock.t, col.playhead, []);
    const tt = view.getTracker();
    if (tt !== null) {
      vline(tt, col.tracker, [3, 3]);
      for (const it of frame.lines) {
        const v = store.valueAt(it.src, tt - it.shift);
        if (typeof v !== "number") continue;
        const y = Y(v * it.scale + it.offset);
        if (y < py - 1 || y > py + ph + 1) continue;
        g.fillStyle = col.bg;
        g.beginPath();
        g.arc(X(tt), y, 4.5, 0, 2 * Math.PI);
        g.fill();
        g.fillStyle = it.c.color;
        g.beginPath();
        g.arc(X(tt), y, 3, 0, 2 * Math.PI);
        g.fill();
      }
    }
    const box = gesture?.box;
    if (box) {
      g.fillStyle = col.box;
      g.strokeStyle = col.boxEdge;
      g.fillRect(box.x, box.y, box.w, box.h);
      g.strokeRect(box.x + 0.5, box.y + 0.5, box.w - 1, box.h - 1);
    }
  }

  // ---- legend

  function rebuildLegend() {
    rows = st.curves.map((c, k) => {
      const val = h("span", {class: "val"});
      const row = h("div", {class: `row ${c.hidden ? "off" : ""}`, title: c.error || "",
        onpointerenter: () => { st.hl = k; invalidate(); },
        onpointerleave: () => { st.hl = undefined; invalidate(); },
        oncontextmenu: (e) => { e.preventDefault(); editCurve(c, e); }},
      h("i", {style: `background:${c.color}`, title: "show/hide", onclick: () => { c.hidden = !c.hidden; rebuildLegend(); invalidate(); }}),
      h("span", {class: `name ${c.error ? "error" : ""}`, onclick: (e) => editCurve(c, e)}, label(c)),
      val,
      h("b", {class: "x", title: "remove", onclick: () => removeCurve(c)}, "×"));
      row.val = val;
      return row;
    });
    legend.replaceChildren(...rows);
    legend.hidden = !rows.length;
    empty.hidden = !!rows.length;
    statsKey = "";
  }

  const editCurve = (c, e) => curveEditor(e.clientX, e.clientY, c, {
    change: () => { rebuildLegend(); invalidate(); },
    remove: () => removeCurve(c),
  });

  function updateLegend() {
    const tt = view.getTracker() ?? clock.t;
    st.curves.forEach((c, k) => {
      const row = rows[k];
      if (!row) return;
      const s = store.series(c.transform ? derivedPath(c.path, c.transform) : c.path);
      let v = s ? store.valueAt(s.path ?? c.path, tt - (c.shift || 0)) : undefined;
      if (typeof v === "number" && s.type !== "e") v = v * (c.scale ?? 1) + (c.offset ?? 0);
      const text = fmtValue(v, s?.names);
      if (row.val.textContent !== text) row.val.textContent = text;
    });
    if (st.stats) updateStats();
  }

  // statistics over the visible range, computed in the worker shortly after the view settles
  function updateStats() {
    const [t0, t1] = clock.range, curves = frame?.lines.map((it) => ({path: it.src, shift: it.shift, scale: it.scale, offset: it.offset})) || [];
    const key = JSON.stringify([t0, t1, curves, curves.map((c) => store.series(c.path)?.version)]);
    if (key === statsKey) return;
    statsKey = key;
    clearTimeout(statsTimer);
    statsTimer = setTimeout(() => compute(store, "stats", {curves, t0, t1}, curves.map((c) => c.path)).then((res) => {
      for (const row of rows) row.querySelector(".stats")?.remove();
      res.forEach((r, k) => {
        const row = rows[st.curves.indexOf(frame.lines[k]?.c)];
        if (!r || !row) return;
        const f = (x) => fmtValue(+x.toPrecision(4));
        row.append(h("div", {class: "stats"}, `min ${f(r.min)}  max ${f(r.max)}  mean ${f(r.mean)}  std ${f(r.std)}  n ${r.count}  ${f(r.rate)} Hz`));
      });
    }, console.error), 120);
  }

  // ---- gestures

  const local = (e) => { const r = over.getBoundingClientRect(); return {x: e.clientX - r.left, y: e.clientY - r.top}; };
  const zone = ({x, y}) => x < L ? "y" : y > H - B ? "x" : "plot";
  const timeAt = (x) => frame.t0 + (x - frame.px) / frame.pw * (frame.t1 - frame.t0);
  const valueAt = (y) => frame.hi - (y - frame.py) / frame.ph * (frame.hi - frame.lo);
  const setRange = (a, b) => clock.setRange(a, b);

  over.addEventListener("pointerdown", (e) => {
    if (!frame) return;
    over.focus();
    over.setPointerCapture(e.pointerId);
    const p = local(e);
    gesture = {button: e.button, zone: zone(p), start: p, range: [...clock.range], lo: frame.lo, hi: frame.hi, moved: false};
  });

  over.addEventListener("pointermove", (e) => {
    if (!frame) return;
    const p = local(e);
    view.track(zone(p) === "plot" && !(gesture?.moved && gesture.box === undefined) ? Math.max(frame.t0, timeAt(p.x)) : null);
    if (!gesture) return;
    const dx = p.x - gesture.start.x, dy = p.y - gesture.start.y;
    if (!gesture.moved && Math.hypot(dx, dy) < DRAG) return;
    if (!gesture.moved) view.remember(clock);
    gesture.moved = true;
    const {zone: z, button, range: [a, b], lo, hi} = gesture;
    const panX = z === "x" || (z === "plot" && button !== 0), panY = z === "y" || (button !== 0 && st.y.min !== undefined);
    if (panX) {
      const dt = dx / frame.pw * (b - a);
      setRange(a - dt, b - dt);
    }
    if (panY) {
      const dv = dy / frame.ph * (hi - lo);
      st.y = {min: lo + dv, max: hi + dv};
      invalidate();
    }
    if (z === "plot" && button === 0) {
      const x0 = Math.max(frame.px, Math.min(p.x, gesture.start.x)), x1 = Math.min(frame.px + frame.pw, Math.max(p.x, gesture.start.x));
      const y0 = Math.max(frame.py, Math.min(p.y, gesture.start.y)), y1 = Math.min(frame.py + frame.ph, Math.max(p.y, gesture.start.y));
      // a flat box zooms time only, a narrow one zooms values only
      const xOnly = y1 - y0 < 12 || frame.blockMode, yOnly = !xOnly && x1 - x0 < 12;
      gesture.box = xOnly ? {x: x0, w: x1 - x0, y: frame.py, h: frame.ph, xOnly} : yOnly ? {x: frame.px, w: frame.pw, y: y0, h: y1 - y0, yOnly} : {x: x0, w: x1 - x0, y: y0, h: y1 - y0};
      pane.overlay();
    }
  });

  over.addEventListener("pointerup", (e) => {
    const g = gesture;
    gesture = null;
    if (!g || !frame) return;
    const p = local(e);
    if (!g.moved) {
      if (g.button === 0 && g.zone === "plot") clock.seek(timeAt(p.x));
      if (g.button === 2) paneMenu(e);
    } else if (g.box) {
      const {x, y, w, h: bh, xOnly, yOnly} = g.box;
      if (!yOnly && w > 2) setRange(timeAt(x), timeAt(x + w));
      if (!xOnly && bh > 2) st.y = {min: valueAt(y + bh), max: valueAt(y)};
      invalidate();
    }
    pane.overlay();
  });

  over.addEventListener("pointerleave", () => gesture || view.track(null));
  over.addEventListener("contextmenu", (e) => e.preventDefault());

  over.addEventListener("wheel", (e) => {
    if (!frame) return;
    e.preventDefault();
    const p = local(e), scale = e.deltaMode ? 33 : 1;
    const [a, b] = clock.range;
    if (zone(p) === "y" || e.shiftKey) {
      view.remember(clock, "wheel-y");
      const f = Math.exp((e.deltaY || e.deltaX) * scale * 0.002), v = valueAt(p.y);
      st.y = {min: v - (v - frame.lo) * f, max: v + (frame.hi - v) * f};
      invalidate();
    } else if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      view.remember(clock, "wheel-pan");
      const dt = e.deltaX * scale / frame.pw * (b - a);
      setRange(a + dt, b + dt);
    } else {
      view.remember(clock, "wheel");
      const f = Math.exp(e.deltaY * scale * 0.002), t = timeAt(p.x);
      setRange(t - (t - a) * f, t + (b - t) * f);
    }
  }, {passive: false});

  over.addEventListener("dblclick", (e) => {
    view.remember(clock);
    st.y = {};
    if (zone(local(e)) !== "y") setRange(0, clock.duration);
    invalidate();
  });

  over.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if (!(e.ctrlKey || e.metaKey) || (k !== "z" && k !== "y")) return;
    e.preventDefault();
    (k === "y" || e.shiftKey ? view.redo : view.undo)(clock);
  });

  function paneMenu(e) {
    const locked = st.y.min !== undefined || st.y.max !== undefined;
    menu(e.clientX, e.clientY, [
      {label: "Fit view", fn: () => { view.remember(clock); st.y = {}; setRange(0, clock.duration); invalidate(); }},
      {label: "Y auto-fit", checked: !locked, fn: () => { view.remember(clock); st.y = locked ? {} : {min: frame.lo, max: frame.hi}; invalidate(); }},
      {label: "Y limits…", fn: () => yLimits(e)},
      {label: "Undo view", fn: () => view.undo(clock)},
      {label: "Redo view", fn: () => view.redo(clock)},
      null,
      {label: "Statistics", checked: st.stats, fn: () => { st.stats = !st.stats; rebuildLegend(); perf.invalidate(pane); }},
      {label: "Export CSV", disabled: !frame.lines.length, fn: exportCsv},
      {label: "Export PNG", fn: () => exportPng([base, over], st.curves.filter((c) => !c.hidden).map((c) => ({color: c.color, name: label(c), value: rows[st.curves.indexOf(c)].val.textContent})),
        col.bg, col.fg, col.font, st.title || "plot")},
      null,
      {label: "Remove all curves", disabled: !st.curves.length, fn: () => { st.curves = []; rebuildLegend(); invalidate(); }},
    ]);
  }

  function yLimits(e) {
    const field = (k) => h("label", {}, k, h("input", {type: "number", step: "any", value: st.y[k] ?? "", placeholder: "auto",
      oninput: (ev) => { st.y = {...st.y, [k]: ev.target.value === "" ? undefined : +ev.target.value}; invalidate(); }}));
    popup(e.clientX, e.clientY, h("div", {class: "title"}, "Y limits"), field("max"), field("min"));
  }

  async function exportCsv() {
    const [t0, t1] = clock.range;
    const curves = frame.lines.map((it) => ({path: it.src, label: label(it.c), shift: it.shift, scale: it.scale, offset: it.offset}));
    const text = await compute(store, "csv", {curves, t0, t1}, curves.map((c) => c.path));
    download(new Blob([text], {type: "text/csv"}), `${st.title || "plot"}.csv`);
  }

  // ---- drop target for series dragged from the browser

  el.addEventListener("dragover", (e) => {
    if (!e.dataTransfer.types.includes(PATHS_MIME)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    el.classList.add("drop");
  });
  el.addEventListener("dragleave", () => el.classList.remove("drop"));
  el.addEventListener("drop", (e) => {
    el.classList.remove("drop");
    const data = e.dataTransfer.getData(PATHS_MIME);
    if (!data) return;
    e.preventDefault();
    addCurves(JSON.parse(data));
  });

  // ---- pane contract

  function render() {
    if (!gBase) return;
    if (dirty) {
      dirty = false;
      drawBase();
    }
    drawOverlay();
    updateLegend();
  }

  function resize(w = el.clientWidth, hgt = el.clientHeight) {
    W = w;
    H = hgt;
    gBase = fitCanvas(base, W, H);
    gOver = fitCanvas(over, W, H);
    invalidate();
  }

  function save() {
    const curves = st.curves.map(({error, ...c}) => c);
    return {title: st.title, curves, y: st.y, ...(st.stats && {stats: true})};
  }

  function dispose() {
    for (const off of offs) off?.();
    view.leave(pane);
    clearTimeout(statsTimer);
    closeMenu();
  }

  for (const c of state.curves || []) st.curves.push(normalize(c));
  rebuildLegend();
  return pane;
}
