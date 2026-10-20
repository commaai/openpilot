// the frame scheduler and the perf HUD. Every pane renders through invalidate(), so each frame's work is
// timed per pane; the HUD in the top bar is always on, and its panel shows where the time went.
import {h} from "/dom.js";

const HISTORY = 600;             // frames kept for the panel and the harness (~10 s at 60 Hz)
const BUDGET_MS = 4;             // main-thread render budget per frame

const dirty = new Set();
const ticks = new Set();
const frames = [];               // {t, interval, work, panes: {name: ms}}
const longFrames = [];           // long-animation-frame entries, newest last
const dropped = [];              // the frames that came late, newest last
const marks = {};
let last = 0, drops = 0, frameNo = 0;
let refresh = 1000 / 60;         // learned display interval

export const perf = {
  frames, longFrames, dropped, marks,
  get drops() { return drops; },

  // pane.render(frame) runs once on the next frame
  invalidate(pane) { dirty.add(pane); },
  // fn(frame) runs every frame until removed (playback, animations)
  tick(fn) { ticks.add(fn); return () => ticks.delete(fn); },
  measure(name, fn) {
    const t = performance.now();
    try { return fn(); } finally { add(current, name, performance.now() - t); }
  },
  mark(name, value) { marks[name] = value; },
};
window.__perf = perf;

let current = null;
function add(f, name, ms) {
  if (f) f.panes[name] = (f.panes[name] || 0) + ms;
}

function loop(now) {
  requestAnimationFrame(loop);
  const interval = last ? now - last : refresh;
  last = now;
  if (interval < refresh * 1.2) refresh += (interval - refresh) * 0.05;
  const frame = current = {t: now, n: frameNo++, interval, work: 0, panes: {}};
  if (interval > refresh * 1.5 && !document.hidden) {
    drops += Math.round(interval / refresh) - 1;
    dropped.push(frame);
    if (dropped.length > 50) dropped.shift();
  }

  const start = performance.now();
  for (const fn of ticks) fn(frame);
  const batch = [...dirty];
  dirty.clear();
  for (const pane of batch) {
    const t = performance.now();
    try { pane.render(frame); } catch (e) { console.error(pane.name, e); }
    add(frame, pane.name || "?", performance.now() - t);
  }
  frame.work = performance.now() - start;
  frames.push(frame);
  if (frames.length > HISTORY) frames.shift();
  current = null;
}
requestAnimationFrame(loop);
document.addEventListener("visibilitychange", () => { last = 0; });

if (PerformanceObserver.supportedEntryTypes?.includes("long-animation-frame")) {
  new PerformanceObserver((list) => {
    for (const e of list.getEntries()) {
      // layout: style + layout + paint time at the frame's end, where DOM churn shows up
      longFrames.push({t: e.startTime, duration: e.duration, blocking: e.blockingDuration,
        layout: e.styleAndLayoutStart ? e.startTime + e.duration - e.styleAndLayoutStart : 0,
        scripts: e.scripts.map((s) => `${s.sourceFunctionName || s.invoker || "?"} ${s.sourceURL.split("/").pop()}:${s.sourceCharPosition} ${s.duration.toFixed(0)}ms`)});
      if (longFrames.length > 50) longFrames.shift();
    }
  }).observe({type: "long-animation-frame", buffered: false});
}

function stats(n) {
  const recent = frames.slice(-n);
  const span = recent.reduce((s, f) => s + f.interval, 0);
  return {fps: span ? recent.length * 1000 / span : 0, max: recent.reduce((m, f) => Math.max(m, f.work), 0)};
}

// the top-bar readout; text updates 4x a second so the HUD itself costs nothing
// extra: more readouts inside the HUD (memory and disk come from status.js)
export function hud(...extra) {
  const fps = h("span", {class: "hud-fps"});
  const ms = h("span", {class: "hud-ms"});
  const dropEl = h("span", {class: "hud-drops"});
  const spark = h("canvas", {class: "hud-spark", width: 40, height: 18});
  const el = h("button", {class: "hud", title: "frame rate · worst render time this second · dropped frames · memory · disk (click for details)"}, spark, fps, ms, dropEl, ...extra);
  let panel = null;
  el.addEventListener("click", () => {
    if (panel) { panel.close(); panel = null; } else panel = perfPanel(() => { panel = null; });
  });
  const sparkPane = {name: "hud", render() {
    const ctx = spark.getContext("2d");
    const w = spark.width, hgt = spark.height;
    ctx.clearRect(0, 0, w, hgt);
    const recent = frames.slice(-w);
    for (let i = 0; i < recent.length; i++) {
      const f = recent[i];
      const v = Math.min(hgt, f.work / (BUDGET_MS * 2) * hgt);
      ctx.fillStyle = f.work > BUDGET_MS || f.interval > refresh * 1.5 ? "#e5484d" : "#3fb950";
      ctx.fillRect(w - recent.length + i, hgt - Math.max(1, v), 1, Math.max(1, v));
    }
  }};
  setInterval(() => {
    const s = stats(60);
    fps.textContent = `${Math.round(s.fps)} fps`;
    ms.textContent = `${s.max.toFixed(1)} ms`;
    dropEl.textContent = `${drops} drop${drops === 1 ? "" : "s"}`;
    el.classList.toggle("bad", s.max > BUDGET_MS * 2 || s.fps < 55 && !document.hidden);
    perf.invalidate(sparkPane);
  }, 250);
  return el;
}

// the detail panel: frame-time chart, per-pane cost, long frames, counters
function perfPanel(onClose) {
  const chart = h("canvas", {class: "perf-chart"});
  const table = h("div", {class: "perf-table"});
  const longs = h("div", {class: "perf-longs"});
  const el = h("div", {class: "perf-panel"},
    h("div", {class: "perf-head"}, h("b", {}, "Performance"),
      h("button", {class: "link", onclick: () => { drops = 0; longFrames.length = dropped.length = 0; }}, "reset"),
      h("button", {class: "link", onclick: () => close()}, "close")),
    chart, table, longs);
  document.body.append(el);
  const pane = {name: "perf-panel", render() {
    const dpr = devicePixelRatio, w = chart.clientWidth, hgt = chart.clientHeight;
    if (chart.width !== w * dpr) { chart.width = w * dpr; chart.height = hgt * dpr; }
    const ctx = chart.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, hgt);
    const scale = hgt / 33;
    const bw = w / HISTORY;
    frames.forEach((f, i) => {
      const x = w - (frames.length - i) * bw;
      ctx.fillStyle = f.interval > refresh * 1.5 ? "rgba(229,72,77,.35)" : "rgba(128,128,128,.25)";
      ctx.fillRect(x, hgt - Math.min(hgt, f.interval * scale), bw, Math.min(hgt, f.interval * scale));
      ctx.fillStyle = f.work > BUDGET_MS ? "#e5484d" : "#3fb950";
      ctx.fillRect(x, hgt - Math.min(hgt, f.work * scale), bw, Math.max(1, f.work * scale));
    });
    ctx.fillStyle = "rgba(128,128,128,.8)";
    ctx.font = "10px ui-monospace, monospace";
    for (const ms of [BUDGET_MS, 16.7]) {
      ctx.fillRect(0, hgt - ms * scale, w, 1);
      ctx.fillText(`${ms} ms`, 4, hgt - ms * scale - 3);
    }
  }};
  const update = () => {
    perf.invalidate(pane);
    const totals = {};
    const recent = frames.slice(-120);
    for (const f of recent) for (const [k, v] of Object.entries(f.panes)) (totals[k] ||= []).push(v);
    const rows = Object.entries(totals).map(([k, v]) => [k, v.reduce((a, b) => a + b, 0) / recent.length, Math.max(...v)]).sort((a, b) => b[1] - a[1]);
    table.replaceChildren(h("div", {class: "perf-row head"}, h("span", {}, "pane"), h("span", {}, "avg ms"), h("span", {}, "max ms")),
      ...rows.map(([k, avg, max]) => h("div", {class: "perf-row"}, h("span", {}, k), h("span", {}, avg.toFixed(2)), h("span", {class: max > BUDGET_MS ? "bad" : ""}, max.toFixed(2)))),
      ...Object.entries(marks).map(([k, v]) => h("div", {class: "perf-row"}, h("span", {}, k), h("span", {}, String(v)), h("span"))));
    longs.replaceChildren(h("div", {class: "muted"}, `${longFrames.length} long frames`),
      ...longFrames.slice(-6).reverse().map((l) => h("div", {class: "mono small"}, `${l.duration.toFixed(0)} ms (layout ${l.layout.toFixed(0)})  ${l.scripts.slice(0, 2).join(" · ")}`)));
  };
  const timer = setInterval(update, 500);
  update();
  function close() { clearInterval(timer); el.remove(); onClose(); }
  return {close};
}
