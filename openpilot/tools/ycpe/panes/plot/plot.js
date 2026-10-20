// time plot: curves over the shared x-range with a hover tracker, a legend of values, state blocks for enums
// and text, and the view gestures. Two canvases: the base (grid, curves) redraws only when the view or the
// data changes; the overlay (playhead, tracker, zoom box) redraws on every cursor move. Per-frame work is
// O(pixels): curves come from store.decimate (cached on a bucket grid, so a pan only decimates the newly
// exposed edge) and state blocks from cached runs.
import {h} from "/dom.js";
import {colors, fitCanvas, ticks, axes, fmtTime, fmtValue, strokeSeries, drawBlocks, drawUnloaded, nextColor, ellipsize} from "./draw.js";
import {blocks} from "./blocks.js";
import {Decimator} from "./decimate.js";
import * as view from "./view.js";
import {menu, popup, curveEditor, close as closeMenu} from "./menu.js";
import {TRANSFORMS, derivedPath, derive, compute} from "/data/transforms.js";
import {evaluate} from "/data/custom.js";
import {download, exportPng} from "./export.js";
import {openConsole} from "./console.js";

const L = 58, R = 8, T = 6, B = 20, STRIP = 18, DRAG = 4;
const STEPPED = new Set(["b", "e", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64"]);

// the median spacing of a series' samples, from up to 16 gaps per chunk; cached per version
const spacing = new Map();  // path → {version, dt}
function medianInterval(path, s) {
  const hit = spacing.get(path);
  if (hit?.version === s.version) return hit.dt;
  const gaps = [];
  for (const {tt} of s.chunks) {
    const step = Math.max(1, (tt.length - 1) / 16);
    for (let k = 0; k < tt.length - 1; k += step) gaps.push(tt[Math.floor(k) + 1] - tt[Math.floor(k)]);
  }
  gaps.sort((a, b) => a - b);
  const dt = gaps.length ? gaps[gaps.length >> 1] : Infinity;
  spacing.set(path, {version: s.version, dt});
  return dt;
}

// lines break between points further apart than maxDt, or with unloaded time between them; called with
// increasing times, so the span pointer only moves forward
function gapTest(maxDt, spans) {
  let j = 0;
  return (a, b) => {
    if (b - a > maxDt) return true;
    while (j < spans.length && spans[j][1] <= a) j++;
    return j < spans.length && spans[j][0] < b;
  };
}


export function create(ctx, state = {}) {
  const {store, clock, perf} = ctx;
  let col = colors(ctx.theme);
  // range: the pane's own x-range when its x-axis is unlinked from the shared clock.range
  const st = {title: state.title || "", curves: [], y: {...state.y}, stats: !!state.stats, range: state.range || null};
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
  const decimator = new Decimator(store);

  const pane = {
    name: "plot", el, render, resize, dispose, save, addCurves, compare,
    has: (path) => st.curves.some((c) => c.path === path),
    get view() { return {y: {...st.y}, range: st.range && [...st.range]}; },
    setView({y, range}) { st.y = {...y}; st.range = range; invalidate(); },
    overlay: () => perf.invalidate(pane),
  };
  const invalidate = () => { dirty = true; perf.invalidate(pane); };

  const offs = [
    clock.on("range", invalidate),
    clock.on("t", pane.overlay),
    store.on("segment", invalidate),
    store.on("paths", invalidate),
    view.onUnloaded(invalidate),
    ctx.theme.on(() => { col = colors(ctx.theme); invalidate(); }),
    // decoded DBC signals arrive after they're asked for
    store.can?.on("series", (path) => { if (st.curves.some((c) => c.path === path || c.custom?.inputs.includes(path))) invalidate(); }),
  ].filter(Boolean);
  view.join(pane);

  // ---- curves

  function addCurves(paths) {
    for (const path of paths) if (!st.curves.some((c) => c.path === path && !c.transform && !c.custom)) st.curves.push(normalize({path}));
    rebuildLegend();
    invalidate();
  }

  function normalize(c) {
    return {...c, color: c.color || nextColor(st.curves.map((x) => x.color)),
      ...(c.custom?.untrusted && {error: "custom code from an imported layout: click to review and run it"})};
  }

  function removeCurve(c) {
    st.curves.splice(st.curves.indexOf(c), 1);
    rebuildLegend();
    invalidate();
  }

  // the store path a curve draws; transforms and custom series are (re)computed when their inputs change
  // custom code from an untrusted layout waits until the user has reviewed it in the console
  const inputsVersion = (inputs) => inputs.map((p) => store.series(p)?.version).join();
  function source(c) {
    if (!c.transform && !c.custom) return c.path;
    const src = c.custom ? c.path : derivedPath(c.path, c.transform);
    const inputs = c.custom ? c.custom.inputs : [c.path];
    const version = inputsVersion(inputs);
    if (!c.custom?.untrusted && inputs.every((p) => store.series(p)) && made.get(src) !== version && !busy.has(src)) {
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
      const d = decimator.get(it.src, it.s.version, a, b, Math.round(px * 1.1));
      // buckets may be empty here and there, so a gap is also wider than a couple of them
      it.gap = gapTest(Math.max(5 * medianInterval(it.src, it.s), 1, 2.5 * (d?.w || 0)), view.unloaded());
      it.d = withEdges(d, it.src, a, b);
      it.style = it.c.style || (STEPPED.has(it.s.type) ? "steps" : "line");
    }
    return {rows, lines, blockMode};
  }

  // held values in empty edge buckets, so series sparser than the view still reach both sides; only where the
  // edge's logs are loaded, since an unknown span must not show the last value
  function withEdges(d, path, a, b) {
    const n = d?.t.length, pre = n && d.min[0] !== d.min[0], post = n && d.min[n - 1] !== d.min[n - 1];
    if (!pre && !post) return d;
    const min = d.min.slice(), max = d.max.slice();
    for (const [i, t, held] of [[0, a, pre], [n - 1, b, post]]) {
      const v = held && !view.isUnloaded(t) ? store.valueAt(path, t) : undefined;
      if (typeof v === "number") min[i] = max[i] = v;
    }
    return {...d, min, max};
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
    const g = gBase, [t0, t1] = xRange();
    const px = L, pw = W - L - R, ph0 = H - T - B;
    if (pw < 4 || ph0 < 4) return;
    if (!st.curves.length) {  // the empty state is just the hint
      frame = null;
      return g.clearRect(0, 0, W, H);
    }
    const p = prepare(t0, t1, pw);
    const stripH = p.blockMode ? ph0 / Math.max(1, p.rows.length) : Math.min(STRIP, ph0 / 4 / Math.max(1, p.rows.length));
    const py = T + (p.blockMode ? ph0 : p.rows.length * stripH), ph = T + ph0 - py;
    const [lo, hi] = p.blockMode ? [0, 1] : yRange(p.lines, t0, t1);
    const X = (t) => px + (t - t0) / (t1 - t0) * pw, Y = (v) => py + (hi - v) / (hi - lo) * ph;
    frame = {t0, t1, lo, hi, px, py, pw, ph, X, Y, stripH, ...p};

    g.clearRect(0, 0, W, H);
    g.fillStyle = col.bg;
    g.fillRect(px, T, pw, ph0);

    const yt = p.blockMode ? {values: []} : ticks(lo, hi, ph / 34), xt = ticks(t0, t1, pw / 90);
    axes(g, col, {x: px, y: T, w: pw, h: ph0}, xt, X, (t) => fmtTime(t, xt.step, t1 >= 120), yt, Y);
    drawUnloaded(g, view.unloaded(), X, {x: px, y: T, w: pw, h: ph0}, col);

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
      g.setLineDash(it.c.dash ? [5, 3] : []);
      strokeSeries(g, it.d, (t) => X(t + shift), (v) => Y(v * scale + offset), it.style, st.curves[st.hl] === it.c ? 2.5 : 1.5, it.gap);
    }
    g.globalAlpha = 1;
    g.setLineDash([]);
    drawInvalid(g, p.lines, X, py + ph);
    decimator.sweep();
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
    const [t0, t1] = xRange();
    for (const svc of services) {
      const path = `/${svc}/valid`, d = decimator.get(path, store.series(path)?.version, t0, t1, Math.round(W / 2));
      if (!d) continue;
      const w = Math.max(2, X(d.w) - X(0));
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
      const row = h("div", {class: `row ${c.hidden ? "off" : ""}`, title: c.error || label(c),
        onpointerenter: () => { st.hl = k; invalidate(); },
        onpointerleave: () => { st.hl = undefined; invalidate(); },
        oncontextmenu: (e) => { e.preventDefault(); editCurve(c, e); }},
      h("i", {class: c.dash ? "dash" : "", style: `color:${c.color}`, title: "show/hide", onclick: () => { c.hidden = !c.hidden; rebuildLegend(); invalidate(); }}),
      h("span", {class: `name ${c.error ? "error" : ""}`, onclick: (e) => (c.custom ? editCode : editCurve)(c, e)}, h("bdi", {}, label(c))),
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
    code: c.custom && (() => editCode(c, e)),
    actions: c.custom || c.transform ? [] : (store.overlays || [])
      .filter((o) => !c.path.startsWith(o.prefix) && store.series(o.prefix + c.path))
      .map((o) => ({label: `Compare with ${o.name}`, fn: () => compare(c.path, o.prefix)})),
  });

  // an overlay's run of path (a replay) next to the original: dashed in the original's color, plus their
  // difference in a color of its own (hidden until toggled on); t, if given, is seeked to and brought into view
  function compare(path, prefix, t) {
    const name = (store.overlays || []).find((o) => o.prefix === prefix)?.name || prefix.split("/").pop();
    let orig = st.curves.find((c) => c.path === path && !c.transform && !c.custom);
    if (!orig) st.curves.push(orig = normalize({path}));
    const replay = prefix + path, tail = path.slice(1);
    if (!st.curves.some((c) => c.path === replay)) st.curves.push({path: replay, color: orig.color, dash: true, label: `${name}: ${tail}`});
    const diff = `/custom/${`${name}-${tail}`.replace(/\W+/g, "_")}`;
    if (!st.curves.some((c) => c.path === diff)) {
      st.curves.push(normalize({path: diff, label: `${name} − original: ${tail}`, hidden: true,
        custom: {inputs: [replay, path], globals: "", code: "return value - v1;"}}));
    }
    if (t !== undefined) {
      const [a, b] = xRange();
      if (t < a || t > b) setRange(t - (b - a) / 2, t + (b - a) / 2);
      clock.seek(t);
    }
    rebuildLegend();
    invalidate();
  }

  // creates (c undefined) or edits a custom series; the curve changes only once the code has run
  const editCode = (c, e) => openConsole(e.clientX, e.clientY, c ? {name: c.label, ...c.custom} : {inputs: st.curves.slice(0, 1).map((x) => x.path)}, async (spec) => {
    const path = `/custom/${spec.name}`, custom = {inputs: spec.inputs, globals: spec.globals, code: spec.code};
    await evaluate(store, path, custom);
    made.set(path, inputsVersion(custom.inputs));
    if (c) Object.assign(c, {path, label: spec.name, custom, error: null});
    else st.curves.push(normalize({path, label: spec.name, custom}));
    rebuildLegend();
    invalidate();
  });

  function updateLegend() {
    const tt = view.getTracker() ?? clock.t;
    st.curves.forEach((c, k) => {
      const row = rows[k];
      if (!row) return;
      const s = store.series(c.transform ? derivedPath(c.path, c.transform) : c.path);
      let v = s ? store.valueAt(s.path ?? c.path, tt - (c.shift || 0)) : undefined;
      if (typeof v === "number" && s.type !== "e") v = v * (c.scale ?? 1) + (c.offset ?? 0);
      const text = fmtValue(v, s?.names).slice(0, 100);
      if (row.val.textContent !== text) row.val.textContent = text;
    });
    if (st.stats) updateStats();
  }

  // statistics over the visible range, computed in the worker shortly after the view settles
  function updateStats() {
    const [t0, t1] = xRange(), curves = frame?.lines.map((it) => ({path: it.src, shift: it.shift, scale: it.scale, offset: it.offset})) || [];
    const key = JSON.stringify([t0, t1, curves, curves.map((c) => store.series(c.path)?.version)]);
    if (key === statsKey) return;
    statsKey = key;
    clearTimeout(statsTimer);
    statsTimer = setTimeout(() => compute(store, "stats", {curves, t0, t1}, curves.map((c) => c.path)).then((res) => {
      for (const row of rows) row.querySelector(".stats")?.remove();
      res.forEach((r, k) => {
        const row = rows[st.curves.indexOf(frame.lines[k]?.c)];
        if (!r || !row) return;
        const f = fmtValue;
        row.append(h("div", {class: "stats"}, `min ${f(r.min)}  max ${f(r.max)}  mean ${f(r.mean)}  std ${f(r.std)}  n ${r.count}  ${f(r.rate)} Hz`));
      });
    }, console.error), 120);
  }

  // ---- gestures

  const local = (e) => { const r = over.getBoundingClientRect(); return {x: e.clientX - r.left, y: e.clientY - r.top}; };
  const zone = ({x, y}) => x < L ? "y" : y > H - B ? "x" : "plot";
  const timeAt = (x) => frame.t0 + (x - frame.px) / frame.pw * (frame.t1 - frame.t0);

  // the state block under p, as tooltip text: its value, span and duration
  function blockAt(p) {
    const k = Math.floor((p.y - T) / frame.stripH), it = frame.rows[k];
    if (!it || p.x < frame.px) return "";
    const t = timeAt(p.x) - it.shift, b = it.blocks.find((b) => b.a <= t && t < b.b);
    if (!b) return "";
    const what = b.mixed ? "several states (zoom in)" : fmtValue(b.v, it.s.names) + (it.s.names ? ` (${b.v})` : "");
    return `${label(it.c)}: ${what}\n${b.a.toFixed(2)} s → ${b.b.toFixed(2)} s, ${(b.b - b.a).toFixed(2)} s`;
  }
  const valueAt = (y) => frame.hi - (y - frame.py) / frame.ph * (frame.hi - frame.lo);
  const xRange = () => st.range || clock.range;
  function setRange(a, b) {
    if (!st.range) return clock.setRange(a, b);
    const span = Math.max(0.01, Math.min(clock.duration, b - a)), t0 = Math.max(0, Math.min(clock.duration - span, a));
    st.range = [t0, t0 + span];
    invalidate();
  }

  over.addEventListener("pointerdown", (e) => {
    view.touch(pane);
    if (!frame) return;
    over.focus();
    over.setPointerCapture(e.pointerId);
    const p = local(e);
    gesture = {button: e.button, zone: zone(p), start: p, range: [...xRange()], lo: frame.lo, hi: frame.hi, moved: false};
  });

  over.addEventListener("pointermove", (e) => {
    if (!frame) return;
    const p = local(e);
    view.track(zone(p) === "plot" && !(gesture?.moved && gesture.box === undefined) ? Math.max(frame.t0, timeAt(p.x)) : null);
    const tip = gesture ? "" : blockAt(p);
    if (over.title !== tip) over.title = tip;
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
      // a flat box (or alt) zooms time only, a narrow one (or shift) zooms values only
      const xOnly = e.altKey || y1 - y0 < 12 || frame.blockMode, yOnly = !xOnly && (e.shiftKey || x1 - x0 < 12);
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
  // a legend highlight can outlive its pointerleave when a popup opens over the row
  over.addEventListener("pointerenter", () => { if (st.hl !== undefined) { st.hl = undefined; invalidate(); } });
  over.addEventListener("contextmenu", (e) => e.preventDefault());

  over.addEventListener("wheel", (e) => {
    if (!frame) return;
    e.preventDefault();
    const p = local(e), scale = e.deltaMode ? 33 : 1;
    const [a, b] = xRange();
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
      {label: "Link x-axis", checked: !st.range, fn: () => { st.range = st.range ? null : [...clock.range]; invalidate(); }},
      {label: "Undo view", fn: () => view.undo(clock)},
      {label: "Redo view", fn: () => view.redo(clock)},
      null,
      {label: "Statistics", checked: st.stats, fn: () => { st.stats = !st.stats; rebuildLegend(); perf.invalidate(pane); }},
      {label: "Export CSV", disabled: !frame.lines.length, fn: exportCsv},
      {label: "Export PNG", fn: () => exportPng([base, over], st.curves.filter((c) => !c.hidden).map((c) => ({color: c.color, name: label(c), value: rows[st.curves.indexOf(c)].val.textContent})),
        col.bg, col.fg, col.font, st.title || "plot")},
      null,
      {label: "Custom series…", fn: () => editCode(undefined, e)},
      {label: "Remove all curves", disabled: !st.curves.length, fn: () => { st.curves = []; rebuildLegend(); invalidate(); }},
    ]);
  }

  function yLimits(e) {
    const field = (k) => h("label", {}, k, h("input", {type: "number", step: "any", value: st.y[k] ?? "", placeholder: "auto",
      oninput: (ev) => { st.y = {...st.y, [k]: ev.target.value === "" ? undefined : +ev.target.value}; invalidate(); }}));
    popup(e.clientX, e.clientY, h("div", {class: "title"}, "Y limits"), field("max"), field("min"));
  }

  async function exportCsv() {
    const [t0, t1] = xRange();
    const curves = frame.lines.map((it) => ({path: it.src, label: label(it.c), shift: it.shift, scale: it.scale, offset: it.offset}));
    const text = await compute(store, "csv", {curves, t0, t1}, curves.map((c) => c.path));
    download(new Blob([text], {type: "text/csv"}), `${st.title || "plot"}.csv`);
  }

  view.acceptPaths(el, addCurves);

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
    el.classList.toggle("narrow", W < 260);  // legend names only
    invalidate();
  }

  function save() {
    const curves = st.curves.map(({error, ...c}) => c);
    return {title: st.title, curves, y: st.y, ...(st.stats && {stats: true}), ...(st.range && {range: st.range})};
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
