// shared by every plot pane: the hover tracker, the view's and the curves' entries in the workspace's undo history,
// the series drop target, and the time whose logs aren't loaded
import {SEGMENT} from "/api.js";
import {listeners} from "/dom.js";
import {undoHistory} from "/history.js";
import {status} from "/status.js";

export const PATHS_MIME = "application/x-ycpe-paths";
const panes = new Set();  // {overlay(), view, setView(view), curves, setCurves(curves)}: view is a pane's own part ({y, range})
let tracker = null;

let touched = null;  // the plot pane the user last pressed in
const curvesSubs = listeners();  // a plot's curves changed, or a plot came or went

// a copy of a plot (a duplicated tab) takes an id of its own
export function join(p) {
  if (byId(p.id)) p.id = crypto.randomUUID().slice(0, 8);
  panes.add(p);
}
export function leave(p) {
  panes.delete(p);
  if (touched === p) touched = null;
  curvesSubs.emit();
}
// history entries name plots by id: every layout edit makes the panes anew, and undoing a close makes the plot again
const byId = (id) => [...panes].find((p) => p.id === id);
export function touch(p) { touched = p; }
// the plot a series is sent to without a drag, if it's on screen (verbs.js)
export const lastTouched = () => touched;
export function each(fn) { for (const p of panes) fn(p); }
// fn() after any plot's curves change (what the CAN panes' plot buttons show)
export const onCurves = curvesSubs.on;
export const curvesChanged = () => curvesSubs.emit();

// the hover time, mirrored in every pane; null when no pane is hovered
export function getTracker() { return tracker; }
export function track(t) {
  if (t === tracker) return;
  tracker = t;
  for (const p of panes) p.overlay();
}

// route time whose logs aren't in memory, as sorted [t0, t1] spans (segment n is [60n, 60n + 60)); none
// when nothing publishes log status. A segment that was ready and is being parsed again (a newly requested
// list) keeps its data meanwhile, so it still counts as loaded. onUnloaded(fn) runs fn when the spans change.
let unloadedSpans = [], unloadedKey = "", logs = null, held = [];
const unloadedSubs = listeners();
status.on(() => {
  const src = status.kinds().find((k) => k.name === "logs"), segs = src?.segs || [];
  if (src !== logs) [logs, held] = [src, []];
  segs.forEach((s, n) => { held[n] = s.state === "ready" || (held[n] && (s.state === "loading" || s.state === "queued")); });
  const key = held.map(Number).join("");
  if (key === unloadedKey) return;
  unloadedKey = key;
  unloadedSpans = [];
  segs.forEach((s, n) => {
    if (held[n]) return;
    const last = unloadedSpans.at(-1);
    if (last?.[1] === n * SEGMENT) last[1] += SEGMENT;
    else unloadedSpans.push([n * SEGMENT, (n + 1) * SEGMENT]);
  });
  unloadedSubs.emit();
});
export const unloaded = () => unloadedSpans;
export const onUnloaded = unloadedSubs.on;
export const isUnloaded = (t) => unloadedSpans.some(([a, b]) => t >= a && t < b);

// base redraws share the frame's budget (perf.left), so opening a layout or a segment landing spreads over a few
// frames instead of stalling one. A redraw starts with MIN_MS left
export const MIN_MS = 1.5;

// ms that inputs changing together (segments arriving) wait to recompute derived series as one
export const RECOMPUTE_DEBOUNCE_MS = 200;

// a wheel gesture latches, as the browser's own scrolling does: it goes on to what it started on, a plot (zooming it)
// or the page (scrolling it), until LATCH_MS pass without a wheel event. A page scroll that slides a plot under the
// pointer goes on scrolling, and a zoom that drifts onto an axis goes on zooming.
const LATCH_MS = 250;
let latch = null, latchAt = -Infinity;  // the plot the gesture is zooming, or null: the page's
addEventListener("wheel", (e) => { if (!e.defaultPrevented) [latch, latchAt] = [null, e.timeStamp]; }, {passive: true});

// whether plot takes wheel event e (then it's the plot's alone): inside, over its plot area, starts a gesture
export function takesWheel(e, plot, inside) {
  const held = e.timeStamp - latchAt < LATCH_MS ? latch : undefined;
  if (held === undefined ? !inside : held !== plot) return false;
  e.preventDefault();
  [latch, latchAt] = [plot, e.timeStamp];
  return true;
}

// changes with the same key within 400 ms (a wheel burst, one drag) are one entry, while it's still the newest
const bursts = new WeakMap();  // clock → {key, at, entry} of the last view change

// call before a view change: the x-range and every plot's view go in the history
export function remember(clock, key = null) {
  const now = performance.now(), last = bursts.get(clock), hist = undoHistory(clock);
  if (key !== null && key === last?.key && now - last.at < 400 && hist.past.at(-1) === last.entry) {
    last.at = now;
    return;
  }
  const entry = viewEntry(clock);
  bursts.set(clock, {key, at: now, entry});
  hist.push(entry);
}

// call before a change to p's curves (curves: as they were, if they've changed already)
export function rememberCurves(clock, p, label, curves = p.curves) {
  undoHistory(clock).push(curvesEntry(p.id, label, curves));
}

function viewEntry(clock) {
  const range = [...clock.range], views = new Map([...panes].map((p) => [p.id, p.view]));
  return {label: "zoom", restore() {
    const now = viewEntry(clock);
    clock.setRange(...range);
    for (const [id, v] of views) byId(id)?.setView(v);
    return now;
  }};
}

function curvesEntry(id, label, curves) {
  return {label, restore() {
    const p = byId(id), now = curvesEntry(id, label, p ? p.curves : curves);
    p?.setCurves(curves);
    return now;
  }};
}

export const undo = (clock) => undoHistory(clock).undo();
export const redo = (clock) => undoHistory(clock).redo();

// el accepts series dragged from the browser; fn(paths)
export function acceptPaths(el, fn) {
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
    fn(JSON.parse(data));
  });
}
