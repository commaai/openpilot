// connect's drive timeline: a 12px strip of engaged / override / alert spans over 20px of sprite thumbnails,
// and optionally a 44px ruler that follows and seeks a video
import {api, routeEvents} from "/api.js";
import {h} from "/dom.js";

const THUMB_HEIGHT = 20;
const THUMB_WIDTH = THUMB_HEIGHT * 128 / 96;  // one 128x96 sprite.jpg per segment

export function routeDuration(route) {
  return route.end_time_utc_millis - route.start_time_utc_millis;
}

export function segmentNumber(route, offsetMs) {
  const t = route.start_time_utc_millis + offsetMs;
  const starts = route.segment_start_times || [];
  let n = 0;
  for (let i = 0; i < starts.length; i++) if (starts[i] <= t) n = route.segment_numbers ? route.segment_numbers[i] : i;
  return n;
}

function thumbnails(route, el) {
  const count = Math.ceil(el.clientWidth / THUMB_WIDTH);
  if (!Number.isFinite(count) || count <= 0) return;
  const duration = routeDuration(route);
  const frag = document.createDocumentFragment();
  for (let i = 0; i < count; i++) {
    const seg = segmentNumber(route, (i + 0.5) / count * duration);
    frag.append(h("div", {class: "thumb", style: `width:${THUMB_WIDTH}px;height:${THUMB_HEIGHT}px;background-image:url(${api.segmentAsset(route, seg, "sprite.jpg")})`}));
  }
  el.replaceChildren(frag);
}

export function timeline(route, {video = null, ruler = false, lazy = true} = {}) {
  const duration = routeDuration(route);
  const pct = (ms) => `${(100 * ms / duration).toFixed(3)}%`;
  const spansEl = h("div", {class: "spans"});
  const thumbsEl = h("div", {class: "thumbs"});
  const el = h("div", {class: `timeline ${ruler ? "has-ruler" : ""}`}, spansEl, h("div", {class: "status-gradient"}), thumbsEl);

  const load = () => {
    thumbnails(route, thumbsEl);
    routeEvents(route).then((spans) => spansEl.replaceChildren(
      ...spans.map((s) => h("div", {class: `span ${s.kind}`, style: `left:${pct(s.start)};width:${pct(Math.max(s.end - s.start, duration / 1000))}`})))).catch(console.error);
  };
  if (lazy && "IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => { if (entries.some((e) => e.isIntersecting)) { io.disconnect(); load(); } }, {rootMargin: "300px"});
    io.observe(el);
  } else {
    requestAnimationFrame(load);
  }

  if (ruler && video) {
    const remaining = h("div", {class: "remaining"});
    const bead = h("div", {class: "bead", hidden: true});
    const rulerEl = h("div", {class: "ruler"}, remaining, bead);
    const offsetAt = (clientX) => {
      const r = rulerEl.getBoundingClientRect();
      return Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * duration;
    };
    rulerEl.addEventListener("click", (e) => { video.currentTime = offsetAt(e.clientX) / 1000; });
    rulerEl.addEventListener("pointermove", (e) => {
      const r = rulerEl.getBoundingClientRect();
      const off = offsetAt(e.clientX);
      bead.hidden = false;
      bead.style.left = `${Math.max(0, Math.min(r.width - 90, e.clientX - r.left - 45))}px`;
      bead.textContent = `${segmentNumber(route, off)}, ${new Date(route.start_time_utc_millis + off).toLocaleTimeString(undefined, {hour12: false})}`;
    });
    rulerEl.addEventListener("pointerleave", () => { bead.hidden = true; });
    const follow = () => {
      const p = Math.min(1, video.currentTime * 1000 / duration);
      remaining.style.left = `${(100 * p).toFixed(2)}%`;
      remaining.style.width = `${(100 * (1 - p)).toFixed(2)}%`;
    };
    video.addEventListener("timeupdate", follow);
    video.addEventListener("seeking", follow);
    el.append(rulerEl);
  }
  return el;
}
