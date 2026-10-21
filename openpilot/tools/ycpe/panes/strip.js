// a drive at a glance, as connect draws it: its engagement spans over its thumbnails, one sprite per segment repeated
// across the segment's width. Spans and thumbnails are laid out in % of an inner track, so showing part of the drive
// (zoomed in) only resizes and moves the track: nothing reads layout. The device page's drive rows and the connect
// view's timeline both use it; their stylesheets size the bands.
import {SEGMENT, api, routeEvents, segmentAt} from "/api.js";
import {h} from "/dom.js";

const MAX_THUMBS = 48;  // sprites per strip at most: a long drive shows every few segments' middle one

// the sprites a strip shows: [start, end] in route seconds and the sprite's url
function sprites(route, duration) {
  if (!route.url) return [];
  const slots = Math.max(1, Math.ceil(duration / SEGMENT)), step = Math.ceil(slots / MAX_THUMBS);
  return Array.from({length: Math.ceil(slots / step)}, (_, i) => [i * step * SEGMENT, Math.min(duration, (i + 1) * step * SEGMENT),
    api.segmentAsset(route, segmentAt(route, Math.min(slots - 1, i * step + (step >> 1))), "sprite.jpg")]);
}

// a sprite fetched and decoded, kept so it paints at once wherever it's shown next. Ahead of the events: those are the
// strip's thin band, the sprites most of it. A missing one is done too (it paints nothing)
const decoded = new Map();
function decode(url) {
  if (!decoded.has(url)) {
    const img = new Image();
    img.fetchPriority = "high";
    img.src = url;
    decoded.set(url, img.decode().catch(() => {}));
    if (decoded.size > 2000) decoded.delete(decoded.keys().next().value);
  }
  return decoded.get(url);
}

// what the route's strip shows, loaded ahead (for a device page about to open)
export function warm(route, duration) {
  return Promise.all([routeEvents(route).catch(() => {}), ...sprites(route, duration).map(([, , url]) => decode(url))]);
}

// duration in route seconds; fill() loads the spans and thumbnails (a list fills its rows as they near the viewport) and
// puts each band in whole as it comes, fading in if the strip is already shown, so a strip never fills in piece by piece
export function strip(route, duration) {
  const spans = h("div", {class: "strip-spans"});
  const thumbs = h("div", {class: "strip-thumbs"});
  const track = h("div", {class: "strip-track"}, spans, thumbs);
  const el = h("div", {class: "strip"}, track);
  const pct = (s) => `${(100 * s / duration).toFixed(3)}%`;
  const put = (band, children) => {
    band.classList.toggle("fade", el.isConnected);
    band.replaceChildren(...children);
  };
  const putThumbs = (list) => put(thumbs, list.map(([a, b, url]) => h("span", {style: `left:${pct(a)};width:${pct(b - a)};background-image:url(${url})`})));
  const putSpans = (events) => put(spans, events.map((s) => h("span", {class: s.kind,
    style: `left:${pct(s.start / 1000)};width:${pct(Math.max(s.end - s.start, duration * 2) / 1000)}`})));
  return {
    el,
    // the spans, once they and the thumbnails are in; neither band before `hold` (the rest of a list's first screen in)
    fill(hold) {
      const list = sprites(route, duration);
      const thumbsIn = Promise.all([hold, ...list.map(([, , url]) => decode(url))]).then(() => putThumbs(list));
      const spansIn = Promise.all([routeEvents(route), hold]).then(([events]) => (putSpans(events), events));
      return Promise.all([spansIn, thumbsIn]).then(([events]) => events);
    },
    // the thumbnails decoded, ready to go in
    decoded: () => Promise.all(sprites(route, duration).map(([, , url]) => decode(url))),
    // only [t0, t1] in view
    show(t0, t1) {
      const span = Math.max(0.01, t1 - t0);
      track.style.width = `${(100 * duration / span).toFixed(3)}%`;
      track.style.left = `${(-100 * t0 / span).toFixed(3)}%`;
    },
  };
}
