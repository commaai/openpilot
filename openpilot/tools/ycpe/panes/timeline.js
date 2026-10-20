// the drive timeline: engaged / override / alert spans and bookmarks over sprite thumbnails.
// strip() is the static card version; create() is the pane, with segment-load progress and scrubbing on the clock.
import {api, routeEvents} from "../api.js";
import {h} from "../dom.js";

const THUMB_HEIGHT = 20;
const THUMB_WIDTH = THUMB_HEIGHT * 128 / 96;  // one 128x96 sprite.jpg per segment
const SPANS = 12, PROGRESS = 3;
const COLORS = {engaged: "#178645", override: "#919b95", alert: "#da6f25", critical: "#c92231", bookmark: "#e3d756"};
const SEGMENT_COLORS = {queued: "#2b3135", loading: "#57a9e3", done: "#5f6b72", error: "#c92231"};
const DRIVING = "#175886";

export function routeDuration(route) {
  return route.end_time_utc_millis - route.start_time_utc_millis;
}

export function segmentNumber(route, offsetMs) {
  const starts = route.segment_start_times;
  if (!starts) return Math.floor(offsetMs / 60000);
  const t = route.start_time_utc_millis + offsetMs;
  let n = 0;
  for (let i = 0; i < starts.length && starts[i] <= t; i++) n = route.segment_numbers ? route.segment_numbers[i] : i;
  return n;
}

function thumbnails(route, el, width) {
  const count = Math.ceil(width / THUMB_WIDTH);
  if (!(count > 0)) return;
  const duration = routeDuration(route);
  el.replaceChildren(...Array.from({length: count}, (_, i) => h("div", {class: "thumb",
    style: `width:${THUMB_WIDTH}px;height:${THUMB_HEIGHT}px;background-image:url(${api.segmentAsset(route, segmentNumber(route, (i + 0.5) / count * duration), "sprite.jpg")})`})));
}

// a static strip for drive cards, loaded when it scrolls near the viewport; onSpans gets the events once loaded
export function strip(route, onSpans) {
  const duration = routeDuration(route);
  const pct = (ms) => `${(100 * ms / duration).toFixed(3)}%`;
  const spansEl = h("div", {class: "spans"});
  const thumbsEl = h("div", {class: "thumbs"});
  const el = h("div", {class: "timeline"}, spansEl, h("div", {class: "status-gradient"}), thumbsEl);
  const io = new IntersectionObserver((entries) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    io.disconnect();
    thumbnails(route, thumbsEl, thumbsEl.clientWidth);
    routeEvents(route).then((spans) => {
      spansEl.replaceChildren(...spans.map((s) => h("div", {class: `span ${s.kind}`, style: `left:${pct(s.start)};width:${pct(Math.max(s.end - s.start, duration / 1000))}`})));
      onSpans?.(spans);
    }).catch(console.error);
  }, {rootMargin: "300px"});
  io.observe(el);
  return el;
}

// pane: ctx.route is the API route record; times are route seconds
export function create(ctx) {
  const {clock, route, store, perf} = ctx;
  const duration = routeDuration(route) / 1000;
  const canvas = h("canvas", {class: "tl-canvas"});
  const thumbsEl = h("div", {class: "thumbs"});
  const range = h("div", {class: "tl-range", hidden: true});
  const playhead = h("div", {class: "tl-playhead"});
  const bead = h("div", {class: "bead", hidden: true});
  const el = h("div", {class: "tl-pane"}, canvas, thumbsEl, range, playhead, bead);
  let spans = [], width = 0, dirty = true, scrubbing = false;

  const pane = {name: "timeline", el, render, resize, dispose, save: () => ({})};
  const redraw = () => { dirty = true; perf.invalidate(pane); };
  const unsubs = [
    clock.on("t", () => perf.invalidate(pane)),
    clock.on("range", () => perf.invalidate(pane)),
    store?.on("segment", redraw),
    store?.on("progress", redraw),
  ];
  routeEvents(route).then((s) => { spans = s; redraw(); }).catch(console.error);

  const xAt = (clientX) => Math.max(0, Math.min(width, clientX - el.getBoundingClientRect().left));
  const xToT = (clientX) => xAt(clientX) / width * duration;
  el.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    scrubbing = true;
    el.setPointerCapture(e.pointerId);
    clock.seek(xToT(e.clientX));
  });
  el.addEventListener("pointermove", (e) => {
    const t = xToT(e.clientX);
    if (scrubbing) clock.seek(t);
    bead.hidden = false;
    bead.style.transform = `translateX(${Math.max(0, Math.min(width - 90, xAt(e.clientX) - 45))}px)`;
    bead.textContent = `${segmentNumber(route, t * 1000)}, ${new Date(route.start_time_utc_millis + t * 1000).toLocaleTimeString(undefined, {hour12: false})}`;
  });
  el.addEventListener("pointerup", () => { scrubbing = false; });
  el.addEventListener("pointerleave", () => { bead.hidden = true; });

  function draw() {
    const dpr = devicePixelRatio, H = SPANS + (store ? PROGRESS : 0);
    canvas.width = Math.round(width * dpr);
    canvas.height = H * dpr;
    canvas.style.height = `${H}px`;
    const g = canvas.getContext("2d");
    g.scale(dpr, dpr);
    const x = (t) => t / duration * width;
    g.fillStyle = DRIVING;
    g.fillRect(0, 0, width, SPANS);
    for (const s of spans) {
      g.fillStyle = COLORS[s.kind];
      const x0 = x(s.start / 1000);
      g.fillRect(x0, 0, Math.max(1, x(s.end / 1000) - x0), s.kind === "bookmark" ? SPANS / 2 : SPANS);
    }
    for (const seg of store?.route.segments || []) {
      g.fillStyle = SEGMENT_COLORS[seg.status] || SEGMENT_COLORS.queued;
      g.fillRect(x(seg.t0), SPANS, Math.max(1, x(seg.t1) - x(seg.t0) - 1), PROGRESS);
    }
  }

  function render() {
    if (dirty && width) { draw(); dirty = false; }
    playhead.style.transform = `translateX(${(Math.min(clock.t, duration) / duration * width).toFixed(1)}px)`;
    const [a, b] = clock.range;
    range.hidden = a <= 0 && b >= duration;
    if (!range.hidden) range.style.cssText = `left:${a / duration * 100}%;width:${(b - a) / duration * 100}%`;
  }

  function resize(w) {
    if (w === width) return;
    width = w;
    thumbnails(route, thumbsEl, w);
    redraw();
  }

  function dispose() {
    for (const off of unsubs) off?.();
  }

  return pane;
}
