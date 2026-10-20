// shared by every plot pane: the hover tracker, undo/redo of the view, and the series drop target

export const PATHS_MIME = "application/x-ycpe-paths";
const panes = new Set();  // {overlay(), view, setView(view)}: view is a pane's own part ({y, range})
const past = [], future = [];
let tracker = null, lastKey = null, lastAt = 0;

export function join(p) { panes.add(p); }
export function leave(p) { panes.delete(p); }

// the hover time, mirrored in every pane; null when no pane is hovered
export function getTracker() { return tracker; }
export function track(t) {
  if (t === tracker) return;
  tracker = t;
  for (const p of panes) p.overlay();
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
