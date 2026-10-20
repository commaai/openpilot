// time plot: curves over the shared x-range with a hover tracker, a legend of values, state blocks for enums
// and text, and the view gestures. Two canvases: the base (grid, curves) redraws only when the view or the
// data changes; the overlay (playhead, tracker, zoom box) redraws on every cursor move. Per-frame work is
// O(pixels): curves come from store.decimate (cached on a bucket grid, so a pan only decimates the newly
// exposed edge) and state blocks from cached runs.
import {h} from "/dom.js";
import {icon} from "/icons.js";
import {colors, fitCanvas, ticks, axes, fmtTick, fmtTime, fmtValue, strokeSeries, drawBlocks, drawUnloaded, nextColor, ellipsize} from "./draw.js";
import {blocks} from "./blocks.js";
import {Decimator} from "./decimate.js";
import * as view from "./view.js";
import {menu, popup, curveEditor, close as closeMenu} from "./menu.js";
import {TRANSFORMS, derivedPath, derive, compute, transformInputs} from "/data/transforms.js";
import {evaluate} from "/data/custom.js";
import {download, exportPng} from "./export.js";
import {openConsole} from "./console.js";
import {customsOf, plots, specKey} from "/layouts/import.js";

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
  const empty = h("div", {class: "empty"});
  const el = h("div", {class: "plot-pane"}, base, over, legend, empty);
  let W = 0, H = 0, gBase = null, gOver = null, dirty = true;
  let frame = null;    // geometry and prepared curves of the last base draw
  let gesture = null;  // the pointer gesture in progress
  const pointers = new Map();  // pointer id → where it is, while it's down (two make a pinch)
  let rows = [];       // legend rows, parallel to st.curves
  let statsTimer = 0, statsKey = "", recomputeTimer = 0;
  const made = new Map();  // derived or custom path → the recipe it was computed from
  const busy = new Set();
  const decimator = new Decimator(store, () => invalidate());

  const pane = {
    name: "plot", el, store, render, resize, dispose, save, addCurves, compare, setCustom,
    has: (path) => st.curves.some((c) => c.path === path),
    get view() { return {y: {...st.y}, range: st.range && [...st.range]}; },
    setView({y, range}) { st.y = {...y}; st.range = range; invalidate(); },
    get curves() { return structuredClone(st.curves); },
    setCurves(curves) {
      st.curves = curves;
      for (const c of curves) if (c.custom) share(c);
      rebuildLegend();
      invalidate();
    },
    overlay: () => perf.invalidate(pane),
  };
  const invalidate = () => { dirty = true; perf.invalidate(pane); };

  const offs = [
    clock.on("range", invalidate),
    clock.on("t", pane.overlay),
    store.on("segment", invalidate),
    store.on("paths", invalidate),
    store.on("put", (path) => { if (st.curves.some((c) => c.path === path)) invalidate(); }),  // a custom series computed by another plot
    store.on("progress", hint),
    view.onUnloaded(invalidate),
    ctx.theme.on(() => { col = colors(ctx.theme); invalidate(); }),
    // decoded DBC signals arrive after they're asked for
    store.can?.on("series", (path) => { if (st.curves.some((c) => c.path === path || c.custom?.inputs.includes(path))) invalidate(); }),
    store.can?.on("moved", followDbc),
  ].filter(Boolean);
  view.join(pane);

  // ---- curves

  function addCurves(paths) {
    const added = paths.filter((path, k) => paths.indexOf(path) === k && !st.curves.some((c) => c.path === path && !c.transform && !c.custom));
    if (!added.length) return;
    view.rememberCurves(clock, pane);
    for (const path of added) st.curves.push(normalize({path}));
    rebuildLegend();
    invalidate();
  }

  function normalize(c) {
    return {...c, color: c.color || nextColor(st.curves.map((x) => x.color)),
      ...(c.custom?.untrusted && {error: "custom code from an imported layout: click to review and run it"})};
  }

  // curves and custom series' inputs follow a renamed DBC signal or message, and go with a removed one
  function followDbc(move) {
    const to = (path) => path.startsWith("/dbc/") ? move(path) : path;
    const follow = (c) => c.custom ? {...c, custom: {...c.custom, inputs: c.custom.inputs.map(to)}} : {...c, path: to(c.path)};
    const curves = st.curves.map(follow).filter((c) => c.path && !c.custom?.inputs.includes(null));
    if (JSON.stringify(curves) === JSON.stringify(st.curves)) return;
    view.rememberCurves(clock, pane);
    st.curves = curves;
    rebuildLegend();
    invalidate();
  }

  // removals can be undone in the plot, so it takes the focus
  function removeCurves(cs) {
    view.rememberCurves(clock, pane);
    st.curves = st.curves.filter((c) => !cs.includes(c));
    rebuildLegend();
    invalidate();
    over.focus({preventScroll: true});
  }

  // the store path a curve draws. Custom series and transforms are recomputed in workers when their inputs
  // change, off the draw and a moment later (inputs that change together, as segments arrive, recompute once);
  // custom code from an untrusted layout waits until the user has reviewed it in the console. A recipe is the inputs'
  // versions and a custom series' code, so an undone edit of the code computes again
  const recipe = (c, out, inputs) => (out === c.path && c.custom ? `${specKey(c.custom)}|` : "") + inputs.map((p) => store.series(p)?.version).join();
  const stale = (c, out, inputs) => !c.custom?.untrusted && !busy.has(out) && inputs.every((p) => store.series(p)) && made.get(out) !== recipe(c, out, inputs);
  function source(c) {
    let src = c.path;
    steps(c, (out, inputs) => {
      src = out;
      if (stale(c, out, inputs)) recomputeTimer ||= setTimeout(recompute, made.has(out) ? view.RECOMPUTE_DEBOUNCE_MS : 0);
    });
    return src;
  }

  // the computed series behind a curve, in order: fn(store path, its inputs, run) for its custom series, then for
  // its transform (of that, or of its path)
  function steps(c, fn) {
    if (c.custom) fn(c.path, c.custom.inputs, () => evaluate(store, c.path, c.custom));
    if (c.transform) fn(derivedPath(c.path, c.transform), transformInputs(c.path, c.transform), () => derive(store, c.path, c.transform));
  }

  function recompute() {
    recomputeTimer = 0;
    for (const c of st.curves) {
      if (c.hidden) continue;
      steps(c, (out, inputs, run) => {
        if (!stale(c, out, inputs)) return;
        const from = recipe(c, out, inputs);
        busy.add(out);
        run().then(() => { made.set(out, from); c.error = null; }, (e) => { made.set(out, from); c.error = e.message; })
          .finally(() => { busy.delete(out); rebuildLegend(); invalidate(); });
      });
    }
  }

  // the empty state: where to drop series, or why there's nothing to plot yet
  function hint() {
    const failure = store.failure;
    const text = failure ? `Couldn't load the logs: ${failure}` : store.loading ? "Loading logs…" : "Drag series here, or double-click one in Series · right-click for options";
    if (empty.textContent !== text) empty.textContent = text;
    empty.hidden = !failure && !!rows.length;
  }

  // a curve's name, with its transform, scale, offset and time shift where it has them
  const signed = (x) => `${x < 0 ? "−" : "+"}${Math.abs(x)}`;
  const label = (c) => (c.label || c.path.replace(/^\//, "")) + (c.transform ? ` [${TRANSFORMS[c.transform.kind].label.toLowerCase()}]` : "") +
    ((c.scale ?? 1) !== 1 ? ` ×${c.scale}` : "") + (c.offset ? ` ${signed(c.offset)}` : "") + (c.shift ? ` (${signed(c.shift)} s)` : "");

  // ---- drawing

  // null when frame f's budget ran out first (the curves done so far stay cached)
  function prepare(t0, t1, px, f) {
    const numeric = [], text = [];
    for (const c of st.curves) {
      if (c.hidden) continue;
      if (view.left(f) <= 0) return null;  // a series' first use unpacks it
      const src = source(c), s = store.series(src);
      if (!s) continue;
      const item = {c, src, s, shift: c.shift || 0, scale: c.scale ?? 1, offset: c.offset ?? 0};
      if (s.type === "t") text.push(item);
      else numeric.push(item);
    }
    // states alone become rows of state blocks, like text; mixed with numbers they plot as steps
    const blockMode = numeric.length > 0 && numeric.every((it) => it.s.type === "e" || it.s.states);
    const rows = blockMode ? [...numeric, ...text] : text;
    const lines = blockMode ? [] : numeric;
    for (const it of rows) it.blocks = blocks(it.s, t0 - it.shift, t1 - it.shift, px);
    for (const it of lines) {
      if (view.left(f) <= 0) return null;
      const span = t1 - t0, a = t0 - it.shift - span * 0.05, b = t1 - it.shift + span * 0.05;
      const d = decimator.get(it.src, it.s.version, a, b, Math.round(px * 1.1));
      // buckets may be empty here and there, so a gap is also wider than a couple of them
      it.gap = gapTest(Math.max(5 * medianInterval(it.src, it.s), 1, 2.5 * (d?.w || 0)), view.unloaded());
      it.d = withEdges(d, it.src, a, b);
      it.style = it.c.style || (STEPPED.has(it.s.type) || it.s.names ? "steps" : "line");  // described values step like enums
    }
    // valid=false spans of the services behind the plain curves; none where valid is always true
    const invalid = [];
    for (const svc of new Set(lines.filter((it) => it.src === it.c.path).map((it) => it.c.path.split("/")[1]))) {
      if (view.left(f) <= 0) return null;
      const path = `/${svc}/valid`;
      if (store.constant(path) === 1) continue;
      const d = decimator.get(path, store.series(path)?.version, t0, t1, Math.round(W / 2));
      if (d) invalid.push(d);
    }
    return {rows, lines, blockMode, invalid};
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

  // false when it ran out of frame f's budget and left the canvas as it was
  function drawBase(f) {
    const g = gBase, [t0, t1] = xRange();
    const px = L, pw = W - L - R, ph0 = H - T - B;
    if (pw < 4 || ph0 < 4) return true;
    if (!st.curves.length) {  // the empty state is just the hint
      frame = null;
      g.clearRect(0, 0, W, H);
      return true;
    }
    const p = prepare(t0, t1, pw, f);
    if (!p) return false;
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
    p.rows.forEach((it, k) => drawBlocks(g, it.blocks, (t) => X(t + it.shift), T + k * stripH, T + (k + 1) * stripH - 1, it.s, col));
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
    drawInvalid(g, p.invalid, X, py + ph);
    decimator.sweep();
    g.restore();

    g.fillStyle = col.text;
    g.textAlign = "right";
    g.textBaseline = "middle";
    p.rows.forEach((it, k) => g.fillText(ellipsize(g, label(it.c).split("/").pop(), px - 10), px - 6, T + (k + 0.5) * stripH));
    drawUnit(g, p.lines, yt, px, py + ph / 2, ph);
    return true;
  }

  // the y axis' unit, as cabana: when the curves share one and none is scaled, along the gutter left of the tick
  // labels if it fits there
  function drawUnit(g, lines, yt, right, y, len) {
    const units = new Set(lines.map((it) => it.scale === 1 && !it.offset ? it.s.unit : undefined)), [unit] = units;
    if (units.size !== 1 || !unit) return;
    const x = right - 14 - Math.max(0, ...yt.values.map((v) => g.measureText(fmtTick(v, yt.step)).width));
    if (x < 7) return;
    g.save();
    g.translate(x, y);
    g.rotate(-Math.PI / 2);
    g.textAlign = "center";
    g.fillText(ellipsize(g, unit, len - 8), 0, 0);
    g.restore();
  }

  // valid=false spans, as a band along the bottom
  function drawInvalid(g, invalid, X, bottom) {
    g.fillStyle = col.invalid;
    for (const d of invalid) {
      const w = Math.max(2, X(d.w) - X(0));
      for (let i = 0; i < d.t.length; i++) if (d.min[i] === 0) g.fillRect(X(d.t[i]) - w / 2, bottom - 4, w, 4);
    }
  }

  function drawOverlay() {
    const g = gOver;
    g.clearRect(0, 0, W, H);
    if (!frame) return;
    const {px, pw, py, ph, X, Y, t0, t1} = frame, top = T, bottom = H - B;
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

  // where a popup for click e opens: at the pointer, or under the control when it came from a key
  const from = (e) => e.detail ? e : (({left, bottom}) => ({clientX: left, clientY: bottom}))(e.currentTarget.getBoundingClientRect());

  function rebuildLegend() {
    // the keyboard focus stays on the same control of the same row
    const f = document.activeElement, at = legend.contains(f) && [rows.indexOf(f.parentElement), [...f.parentElement.children].indexOf(f)];
    rows = st.curves.map((c, k) => {
      const val = h("span", {class: "val"}), name = label(c);
      const row = h("div", {class: `row ${c.hidden ? "off" : ""}`, title: c.error || name,
        onpointerenter: () => { st.hl = k; invalidate(); },
        onpointerleave: () => { st.hl = undefined; invalidate(); },
        oncontextmenu: (e) => { e.preventDefault(); editCurve(c, e); }},
      h("button", {class: `swatch ${c.dash ? "dash" : ""}`, style: `color:${c.color}`, title: "show/hide", "aria-label": `${c.hidden ? "Show" : "Hide"} ${name}`,
        onclick: () => { view.rememberCurves(clock, pane); c.hidden = !c.hidden; rebuildLegend(); invalidate(); }}),
      h("button", {class: `name ${c.error ? "error" : ""}`, "aria-label": `Edit ${name}`, onclick: (e) => (c.custom ? editCode : editCurve)(c, from(e))}, h("bdi", {}, name)),
      val,
      h("button", {class: "x", title: "remove", "aria-label": `Remove ${name}`, onclick: () => removeCurves([c])}, icon("close", 12)));
      row.val = val;
      return row;
    });
    legend.replaceChildren(...rows);
    if (at) (rows[at[0]] || rows.at(-1))?.children[at[1]]?.focus();
    legend.hidden = !rows.length;
    hint();
    statsKey = "";
  }

  // the edits of one opening are one undo entry
  function editCurve(c, e) {
    let before = pane.curves;
    curveEditor(e.clientX, e.clientY, c, {
      change: () => {
        if (before) view.rememberCurves(clock, pane, before);
        before = null;
        rebuildLegend();
        invalidate();
      },
      remove: () => removeCurves([c]),
      code: c.custom && (() => editCode(c, e)),
      actions: c.custom || c.transform ? [] : (store.overlays || [])
        .filter((o) => !c.path.startsWith(o.prefix) && store.series(o.prefix + c.path))
        .map((o) => ({label: `Compare with ${o.name}`, fn: () => compare(c.path, o.prefix)})),
    });
  }

  // an overlay's run of path (a replay) next to the original: dashed in the original's color, plus their
  // difference in a color of its own (hidden until toggled on); t, if given, is seeked to and brought into view
  function compare(path, prefix, t) {
    const name = (store.overlays || []).find((o) => o.prefix === prefix)?.name || prefix.split("/").pop();
    view.rememberCurves(clock, pane);
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

  // creates (c undefined) or edits a custom series; the curve changes only once the code has run. A name is one
  // piece of code across the layout, since other custom series read it by name
  const editCode = (c, e) => openConsole(e.clientX, e.clientY, c ? {name: c.path.slice("/custom/".length), ...c.custom} : {inputs: st.curves.slice(0, 1).map((x) => x.path)}, async (spec) => {
    const path = `/custom/${spec.name}`, custom = {inputs: spec.inputs, globals: spec.globals, code: spec.code};
    const other = path !== c?.path && ctx.layout && customsOf(ctx.layout.save()).get(path);
    if (other && other !== specKey(custom)) throw new Error(`another custom series is named ${spec.name}: pick another name`);
    await evaluate(store, path, custom);
    made.set(path, recipe({path, custom}, path, custom.inputs));
    view.rememberCurves(clock, pane);
    if (c) Object.assign(c, {path, label: spec.name, custom, error: null});
    else st.curves.push(normalize({path, label: spec.name, custom}));
    share({path, custom});
    rebuildLegend();
    invalidate();
  });

  // the other curves named like custom series c, in the plots on screen and the layout's other tabs, take its code
  function share(c) {
    view.each((p) => p.store === store && p.setCustom?.(c.path, c.custom));
    if (ctx.layout) plots(ctx.layout.model, (curves) => { for (const x of curves) if (x.path === c.path && x.custom) x.custom = structuredClone(c.custom); });
  }

  // another curve's code for the custom series at path, which that curve computes into the store
  function setCustom(path, custom) {
    const cs = st.curves.filter((c) => c.path === path && c.custom && specKey(c.custom) !== specKey(custom));
    if (!cs.length) return;
    for (const c of cs) Object.assign(c, {custom: structuredClone(custom), error: null});
    made.set(path, recipe(cs[0], path, custom.inputs));
    rebuildLegend();
    invalidate();
  }

  function updateLegend() {
    const tt = view.getTracker() ?? clock.t;
    st.curves.forEach((c, k) => {
      const row = rows[k];
      if (!row) return;
      const path = c.transform ? derivedPath(c.path, c.transform) : c.path, m = store.meta(path);
      let v = m ? store.valueAt(path, tt - (c.shift || 0)) : undefined, fm = m;
      // names stand for the sample as it is; other numbers are scaled, and keep their unit only when they aren't
      const plain = (c.scale ?? 1) === 1 && !c.offset;
      if (typeof v === "number" && m.type !== "e" && m.names?.[v] === undefined && !plain) {
        v = v * (c.scale ?? 1) + (c.offset ?? 0);
        fm = null;
      }
      const text = fmtValue(v, fm).slice(0, 100);
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
        row.append(h("div", {class: "stats"},
          `min ${fmtValue(r.min)}  max ${fmtValue(r.max)}  mean ${fmtValue(r.mean)}  std ${fmtValue(r.std)}  n ${r.count}  ${fmtValue(r.rate)} Hz`));
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
    const name = b.mixed ? undefined : it.s.names?.[b.v];
    const what = b.mixed ? "several states (zoom in)" : name !== undefined ? `${name} (${b.v})` : fmtValue(b.v, it.s);
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

  // a press: left drags a zoom box, others pan; one finger pans, two pinch time around their middle. The mouse's
  // back button undoes the view, as in jotpluggler
  over.addEventListener("pointerdown", (e) => {
    view.touch(pane);
    if (!frame) return;
    if (e.button === 3) return view.undo(clock);
    over.focus();
    over.setPointerCapture(e.pointerId);
    const p = local(e);
    pointers.set(e.pointerId, p);
    if (pointers.size === 2) {
      const [f0, f1] = pointers.values(), [a, b] = xRange();
      view.remember(clock);
      gesture = {pinch: true, moved: true, span: b - a, apart: Math.max(10, Math.abs(f1.x - f0.x)), at: timeAt((f0.x + f1.x) / 2)};
      return pane.overlay();
    }
    if (pointers.size > 2) return;
    gesture = {button: e.button, touch: e.pointerType === "touch", zone: zone(p), start: p, range: [...xRange()], lo: frame.lo, hi: frame.hi, moved: false};
  });
  over.addEventListener("mouseup", (e) => { if (e.button === 3) e.preventDefault(); });  // else the page goes back

  over.addEventListener("pointermove", (e) => {
    if (!frame) return;
    const p = local(e);
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, p);
    if (gesture?.pinch) {
      const [f0, f1] = pointers.values(), span = gesture.span * gesture.apart / Math.max(10, Math.abs(f1.x - f0.x));
      const a = gesture.at - ((f0.x + f1.x) / 2 - frame.px) / frame.pw * span;
      return setRange(a, a + span);
    }
    view.track(zone(p) === "plot" && !(gesture?.moved && gesture.box === undefined) ? Math.max(frame.t0, timeAt(p.x)) : null);
    const tip = gesture ? "" : blockAt(p);
    if (over.title !== tip) over.title = tip;
    if (!gesture) return;
    const dx = p.x - gesture.start.x, dy = p.y - gesture.start.y;
    if (!gesture.moved && Math.hypot(dx, dy) < DRAG) return;
    if (!gesture.moved) view.remember(clock);
    gesture.moved = true;
    const {zone: z, button, touch, range: [a, b], lo, hi} = gesture;
    const panX = z === "x" || (z === "plot" && (button !== 0 || touch)), panY = z === "y" || (button !== 0 && st.y.min !== undefined);
    if (panX) {
      const dt = dx / frame.pw * (b - a);
      setRange(a - dt, b - dt);
    }
    if (panY) {
      const dv = dy / frame.ph * (hi - lo);
      st.y = {min: lo + dv, max: hi + dv};
      invalidate();
    }
    if (z === "plot" && button === 0 && !touch) {
      const x0 = Math.max(frame.px, Math.min(p.x, gesture.start.x)), x1 = Math.min(frame.px + frame.pw, Math.max(p.x, gesture.start.x));
      const y0 = Math.max(frame.py, Math.min(p.y, gesture.start.y)), y1 = Math.min(frame.py + frame.ph, Math.max(p.y, gesture.start.y));
      // a flat box (or alt) zooms time only, a narrow one (or shift) zooms values only
      const xOnly = e.altKey || y1 - y0 < 12 || frame.blockMode, yOnly = !xOnly && (e.shiftKey || x1 - x0 < 12);
      gesture.box = xOnly ? {x: x0, w: x1 - x0, y: frame.py, h: frame.ph, xOnly} : yOnly ? {x: frame.px, w: frame.pw, y: y0, h: y1 - y0, yOnly} : {x: x0, w: x1 - x0, y: y0, h: y1 - y0};
      pane.overlay();
    }
  });

  over.addEventListener("pointerup", (e) => {
    pointers.delete(e.pointerId);
    const g = gesture;
    gesture = null;
    if (!g || g.pinch || !frame) return;  // the finger left of a pinch doesn't start a gesture
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

  over.addEventListener("pointercancel", (e) => {
    pointers.delete(e.pointerId);
    gesture = null;
    pane.overlay();
  });
  over.addEventListener("pointerleave", () => gesture || view.track(null));
  // a legend highlight can outlive its pointerleave when a popup opens over the row
  over.addEventListener("pointerenter", () => { if (st.hl !== undefined) { st.hl = undefined; invalidate(); } });
  // the context menu key (or shift+F10) opens the plot's menu at its middle; a right click opens it on release,
  // or here in an empty plot (which has no gestures); a long press opens it here, unless the finger has moved
  over.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    const r = over.getBoundingClientRect();
    if (e.button === -1) paneMenu({clientX: r.left + r.width / 2, clientY: r.top + r.height / 2});
    else if (!frame) paneMenu(e);
    else if (e.pointerType === "touch" && gesture && !gesture.moved) {
      gesture = null;  // so lifting the finger doesn't seek
      paneMenu(e);
    }
  });

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

  // the view's and the curves' undo, from the plot or its legend
  el.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if (!(e.ctrlKey || e.metaKey) || (k !== "z" && k !== "y")) return;
    e.preventDefault();
    (k === "y" || e.shiftKey ? view.redo : view.undo)(clock);
  });

  function paneMenu(e) {
    const locked = st.y.min !== undefined || st.y.max !== undefined;
    menu(e.clientX, e.clientY, [
      {label: "Fit view", fn: () => { view.remember(clock); st.y = {}; setRange(0, clock.duration); invalidate(); }},
      {label: "Y auto-fit", checked: !locked, disabled: !frame && !locked, fn: () => { view.remember(clock); st.y = locked ? {} : {min: frame.lo, max: frame.hi}; invalidate(); }},
      {label: "Y limits…", fn: () => yLimits(e)},
      {label: "Link x-axis", checked: !st.range, fn: () => { st.range = st.range ? null : [...clock.range]; invalidate(); }},
      {label: "Undo", fn: () => view.undo(clock)},
      {label: "Redo", fn: () => view.redo(clock)},
      null,
      {label: "Statistics", checked: st.stats, fn: () => { st.stats = !st.stats; rebuildLegend(); perf.invalidate(pane); }},
      {label: "Export CSV", disabled: !frame?.lines.length, fn: exportCsv},
      {label: "Export PNG", fn: () => exportPng([base, over], st.curves.filter((c) => !c.hidden).map((c) => ({color: c.color, name: label(c), value: rows[st.curves.indexOf(c)].val.textContent})),
        col.bg, col.fg, col.font, st.title || "plot")},
      null,
      {label: "Custom series…", fn: () => editCode(undefined, e)},
      {label: "Remove all curves", disabled: !st.curves.length, fn: () => removeCurves(st.curves)},
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

  // base redraws share the frame's budget: a pane starts only with MIN_MS of it left (else it keeps its picture
  // this frame), and one that runs out between curves finishes on a later one
  function render(f) {
    if (!gBase) return;
    if (dirty && view.left(f) >= view.MIN_MS) dirty = !drawBase(f);
    if (dirty) perf.invalidate(pane);
    drawOverlay();
    updateLegend();
  }

  function resize(w = el.clientWidth, hgt = el.clientHeight) {
    W = w;
    H = hgt;
    gBase = fitCanvas(base, W, H);
    gOver = fitCanvas(over, W, H);
    el.classList.toggle("narrow", W < 260);  // legend names only
    el.classList.toggle("tiny", W < 170);  // legend swatches only (names on hover)
    invalidate();
  }

  function save() {
    const curves = st.curves.map(({error, ...c}) => c);
    return {title: st.title, curves, y: st.y, ...(st.stats && {stats: true}), ...(st.range && {range: st.range})};
  }

  function dispose() {
    for (const off of offs) off();
    view.leave(pane, clock);
    clearTimeout(statsTimer);
    clearTimeout(recomputeTimer);
    decimator.dispose();
    closeMenu();
  }

  for (const c of state.curves || []) st.curves.push(normalize(c));
  rebuildLegend();
  return pane;
}
