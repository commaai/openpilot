// shared by every plot pane: the hover tracker, undo/redo of the view, the series drop target, and the time
// whose logs aren't loaded
import {status} from "/status.js";

export const PATHS_MIME = "application/x-ycpe-paths";
const panes = new Set();  // {overlay(), view, setView(view)}: view is a pane's own part ({y, range})
const past = [], future = [];
let tracker = null, lastKey = null, lastAt = 0;

let touched = null;  // the plot pane the user last pressed in

export function join(p) { panes.add(p); }
export function leave(p) { panes.delete(p); }
export function touch(p) { touched = p; }

// the time plot a series is sent to without a drag: the one last used, else any mounted one
function target() {
  const live = [...panes].filter((p) => p.compare && p.el.isConnected);
  return live.includes(touched) ? touched : live[0];
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
// instead of stalling one: the time left in frame f, from its first redraw
const BUDGET_MS = 4;
let budget = {frame: null, end: 0};
export function left(f) {
  if (budget.frame !== f) budget = {frame: f, end: performance.now() + BUDGET_MS};
  return budget.end - performance.now();
}

// call before a view change; changes with the same key within 400 ms (a wheel burst, one drag) are one entry
export function remember(clock, key = null) {
  const now = performance.now();
  if (key !== null && key === lastKey && now - lastAt < 400) {
    lastAt = now;
    return;
  }
  lastKey = key;
  lastAt = now;
  past.push(snapshot(clock));
  if (past.length > 100) past.shift();
  future.length = 0;
}

const snapshot = (clock) => ({range: [...clock.range], views: new Map([...panes].map((p) => [p, p.view]))});

function restore(clock, s) {
  clock.setRange(...s.range);
  for (const [p, v] of s.views) if (panes.has(p)) p.setView(v);
}

export function undo(clock) {
  const s = past.pop();
  if (!s) return;
  future.push(snapshot(clock));
  restore(clock, s);
  lastKey = null;
}

export function redo(clock) {
  const s = future.pop();
  if (!s) return;
  past.push(snapshot(clock));
  restore(clock, s);
  lastKey = null;
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
