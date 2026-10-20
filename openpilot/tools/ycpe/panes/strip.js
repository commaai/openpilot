// a drive at a glance, as connect draws it: its engagement spans over its thumbnails, one sprite per segment repeated
// across the segment's width. Spans and thumbnails are laid out in % of an inner track, so showing part of the drive
// (zoomed in) only resizes and moves the track: nothing reads layout. The device page's drive rows and the connect
// view's timeline both use it; their stylesheets size the bands.
import {SEGMENT, api, routeEvents, segmentAt} from "/api.js";
import {h} from "/dom.js";

const MAX_THUMBS = 48;  // sprites per strip at most: a long drive shows every few segments' middle one

// duration in route seconds; fill() loads the spans and thumbnails (a list fills its rows as they near the viewport)
export function strip(route, duration) {
  const spans = h("div", {class: "strip-spans"});
  const thumbs = h("div", {class: "strip-thumbs"});
  const track = h("div", {class: "strip-track"}, spans, thumbs);
  const el = h("div", {class: "strip"}, track);
  const pct = (s) => `${(100 * s / duration).toFixed(3)}%`;
  return {
    el,
    // the spans, once they're in
    fill() {
      const slots = Math.max(1, Math.ceil(duration / SEGMENT)), step = Math.ceil(slots / MAX_THUMBS);
      if (route.url) {
        thumbs.replaceChildren(...Array.from({length: Math.ceil(slots / step)}, (_, i) => {
          const a = i * step * SEGMENT, b = Math.min(duration, a + step * SEGMENT);
          const n = segmentAt(route, Math.min(slots - 1, i * step + (step >> 1)));
          return h("span", {style: `left:${pct(a)};width:${pct(b - a)};background-image:url(${api.segmentAsset(route, n, "sprite.jpg")})`});
        }));
      }
      return routeEvents(route).then((list) => {
        spans.replaceChildren(...list.map((s) => h("span", {class: s.kind,
          style: `left:${pct(s.start / 1000)};width:${pct(Math.max(s.end - s.start, duration * 2) / 1000)}`})));
        return list;
      });
    },
    // only [t0, t1] in view
    show(t0, t1) {
      const span = Math.max(0.01, t1 - t0);
      track.style.width = `${(100 * duration / span).toFixed(3)}%`;
      track.style.left = `${(-100 * t0 / span).toFixed(3)}%`;
    },
  };
}
