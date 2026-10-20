// shared by every plot pane: the hover tracker, undo/redo of the view and the curves, the series drop target, and
// the time whose logs aren't loaded
import {status} from "/status.js";

export const PATHS_MIME = "application/x-ycpe-paths";
const panes = new Set();  // {overlay(), view, setView(view), curves, setCurves(curves)}: view is a pane's own part ({y, range})
let tracker = null;

let touched = null;  // the plot pane the user last pressed in

export function join(p) { panes.add(p); }
export function leave(p) {
  panes.delete(p);
  if (touched === p) touched = null;
}
export function touch(p) { touched = p; }
export function each(fn) { for (const p of panes) fn(p); }

// the time plot a series is sent to without a drag: the one last used, else the first on the page
function target() {
  const live = [...panes].filter((p) => p.compare && p.el.isConnected);
  if (live.includes(touched)) return touched;
  return live.sort((a, b) => a.el.compareDocumentPosition(b.el) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1)[0];
}

// adds paths to that plot; false when no plot is mounted
export function addToPlot(paths) {
  const p = target();
  p?.addCurves(paths);
  return !!p;
}

// shows an overlay's run of path (prefix + path, e.g. a replay) against the original in a mounted time plot:
// the one already showing path, else the one last used, else any; seeks to t if given. False when no plot is
// mounted.
export function compare(path, prefix, {t} = {}) {
  const p = [...panes].find((p) => p.compare && p.el.isConnected && p.has(path)) || target();
  p?.compare(path, prefix, t);
  return !!p;
}

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
const unloadedSubs = new Set();
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
    if (last?.[1] === n * 60) last[1] += 60;
    else unloadedSpans.push([n * 60, n * 60 + 60]);
  });
  for (const fn of unloadedSubs) fn();
});
export const unloaded = () => unloadedSpans;
export function onUnloaded(fn) { unloadedSubs.add(fn); return () => unloadedSubs.delete(fn); }
export const isUnloaded = (t) => unloadedSpans.some(([a, b]) => t >= a && t < b);

// base redraws share BUDGET_MS a frame, so opening a layout or a segment landing spreads over a few frames
// instead of stalling one: the time left in frame f, from its first redraw. A redraw starts with MIN_MS left
const BUDGET_MS = 4;
export const MIN_MS = 1.5;
let budget = {frame: null, end: 0};
export function left(f) {
  if (budget.frame !== f) budget = {frame: f, end: performance.now() + BUDGET_MS};
  return budget.end - performance.now();
}

// undo history, one per workspace (its clock), so it goes with it. An entry is the x-range and every plot's
// view, or one plot's curves
const histories = new WeakMap();  // clock → {past, future, key, at}
function history(clock) {
  if (!histories.has(clock)) histories.set(clock, {past: [], future: [], key: null, at: 0});
  return histories.get(clock);
}

function push(hist, entry) {
  hist.past.push(entry);
  if (hist.past.length > 100) hist.past.shift();
  hist.future.length = 0;
}

// call before a view change; changes with the same key within 400 ms (a wheel burst, one drag) are one entry
export function remember(clock, key = null) {
  const hist = history(clock), now = performance.now();
  if (key !== null && key === hist.key && now - hist.at < 400) {
    hist.at = now;
    return;
  }
  [hist.key, hist.at] = [key, now];
  push(hist, snapshot(clock));
}

// call before a change to p's curves (curves: as they were, if they've changed already)
export function rememberCurves(clock, p, curves = p.curves) {
  const hist = history(clock);
  hist.key = null;
  push(hist, {pane: p, curves});
}

const snapshot = (clock) => ({range: [...clock.range], views: new Map([...panes].map((p) => [p, p.view]))});

// restores entry s; → the entry that undoes that
function restore(clock, s) {
  if (s.pane) {
    const now = {pane: s.pane, curves: s.pane.curves};
    if (panes.has(s.pane)) s.pane.setCurves(s.curves);
    return now;
  }
  const now = snapshot(clock);
  clock.setRange(...s.range);
  for (const [p, v] of s.views) if (panes.has(p)) p.setView(v);
  return now;
}

export function undo(clock) {
  const hist = history(clock), s = hist.past.pop();
  if (!s) return;
  hist.future.push(restore(clock, s));
  hist.key = null;
}

export function redo(clock) {
  const hist = history(clock), s = hist.future.pop();
  if (!s) return;
  hist.past.push(restore(clock, s));
  hist.key = null;
}

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
