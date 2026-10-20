// canvas maps from Mapbox raster tiles: the device location on the device page, and the map pane (GPS trace
// colored by engagement, a playhead marker, click to seek). Tiles decode off the main thread and every draw goes
// through the frame scheduler, so the map never costs a frame. The trace comes from coords.json, or from the
// store's GPS once logs are loading.
import {MAPBOX_TOKEN} from "../geocode.js";
import {routeCoords, routeEvents} from "../api.js";
import {h, css} from "../dom.js";
import {perf} from "../perf.js";
import {theme} from "../theme.js";

await css(new URL("./map.css", import.meta.url));

const STYLES = {dark: "commaai/cjj4yzqk201c52ss60ebmow0w", light: "mapbox/light-v10"};
const TILE = 256;           // css px per tile; images are @2x
const CACHE = 96;           // decoded tiles kept (up to 1 MB each)
const FETCHES = 8;          // tile requests started per frame (~0.15 ms each)
const MIN_Z = 1, MAX_Z = 19;
const GPS = ["gpsLocationExternal", "gpsLocation"];
const BUCKETS = 2000;
const PICK_PX = 24;         // clicks this close to the trace seek

// web mercator in unit coordinates: the world is [0, 1) in x and y
const unitX = (lng) => (lng + 180) / 360;
const unitY = (lat) => { const s = Math.sin(lat * Math.PI / 180); return 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI); };

// device px per tile: images are decoded at this size, as a scaled drawImage of an image scales it on the CPU
const tilePx = () => Math.round(TILE * Math.min(2, devicePixelRatio));

// decoded tiles shared by every map, least recently used first. A frame starts at most left.fetches requests.
const tiles = new Map();
function tile(style, z, x, y, onLoad, left) {
  const key = `${style}/${z}/${x}/${y}`;
  let t = tiles.get(key);
  if (t) { tiles.delete(key); tiles.set(key, t); return t.bitmap; }
  if (!left.fetches) return left.more = true, null;
  left.fetches--;
  tiles.set(key, t = {bitmap: null});
  fetch(`https://api.mapbox.com/styles/v1/${style}/tiles/${TILE}/${z}/${x}/${y}@2x?access_token=${MAPBOX_TOKEN}`)
    .then((r) => r.ok ? r.blob() : Promise.reject(r.status))
    .then((b) => { const px = tilePx(); return createImageBitmap(b, px < 2 * TILE ? {resizeWidth: px, resizeHeight: px, resizeQuality: "high"} : {}); })
    .then((bitmap) => { t.bitmap = bitmap; onLoad(); })
    .catch(() => tiles.delete(key));
  for (const [k, old] of tiles) {
    if (tiles.size <= CACHE) break;
    old.bitmap?.close();
    tiles.delete(k);
  }
  return null;
}

// a loaded tile or the nearest loaded ancestor, as [bitmap, sx, sy, size] in image pixels
function covering(style, z, x, y) {
  for (let up = 0; up <= Math.min(z, 5); up++) {
    const t = tiles.get(`${style}/${z - up}/${x >> up}/${y >> up}`)?.bitmap;
    if (!t) continue;
    const size = t.width >> up, mask = (1 << up) - 1;
    return [t, (x & mask) * size, (y & mask) * size, size];
  }
  return null;
}

class TileMap {
  constructor(el, {interactive = true} = {}) {
    this.name = "map";
    this.el = el;
    this.canvas = el.appendChild(h("canvas", {class: "tilemap"}));
    this.ctx = this.canvas.getContext("2d");
    this.w = this.h = 0;
    this.cx = this.cy = 0.5;
    this.z = 3;
    this.overlay = null;      // fn(ctx) drawing on top, in css px
    if (interactive) this.interact();
    el.append(h("div", {class: "tilemap-credit"},
      h("a", {href: "https://www.mapbox.com/about/maps/", target: "_blank"}, "© Mapbox"), " ",
      h("a", {href: "https://www.openstreetmap.org/copyright", target: "_blank"}, "© OpenStreetMap")));
  }

  invalidate() { perf.invalidate(this); }

  resize(w, h) {
    const dpr = devicePixelRatio;
    this.w = w;
    this.h = h;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.invalidate();
  }

  get scale() { return TILE * 2 ** this.z; }  // css px per world
  toScreen(ux, uy) { return [(ux - this.cx) * this.scale + this.w / 2, (uy - this.cy) * this.scale + this.h / 2]; }
  toUnit(px, py) { return [(px - this.w / 2) / this.scale + this.cx, (py - this.h / 2) / this.scale + this.cy]; }

  view(lng, lat, z) {
    this.cx = unitX(lng);
    this.cy = unitY(lat);
    this.z = z;
    this.invalidate();
  }

  // frames unit-coordinate bounds with a margin in css px
  fit(x0, y0, x1, y1, pad = 30) {
    this.cx = (x0 + x1) / 2;
    this.cy = (y0 + y1) / 2;
    const fx = (this.w - 2 * pad) / TILE / Math.max(x1 - x0, 1e-9), fy = (this.h - 2 * pad) / TILE / Math.max(y1 - y0, 1e-9);
    this.z = Math.max(MIN_Z, Math.min(16, Math.log2(Math.min(fx, fy))));
    this.invalidate();
  }

  zoomAt(px, py, z) {
    const [ux, uy] = this.toUnit(px, py);
    this.z = Math.max(MIN_Z, Math.min(MAX_Z, z));
    const [nx, ny] = this.toUnit(px, py);
    this.cx += ux - nx;
    this.cy += uy - ny;
    this.invalidate();
  }

  interact() {
    const c = this.canvas;
    const at = (e) => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.zoomAt(...at(e), this.z - e.deltaY * (e.deltaMode ? 0.05 : 0.002));
    }, {passive: false});
    c.addEventListener("dblclick", (e) => this.zoomAt(...at(e), this.z + 1));
    c.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      c.setPointerCapture(e.pointerId);
      const [x0, y0] = at(e), cx = this.cx, cy = this.cy;
      let moved = false;
      c.onpointermove = (ev) => {
        const [x, y] = at(ev);
        moved ||= Math.hypot(x - x0, y - y0) > 3;
        if (!moved) return;
        this.cx = cx - (x - x0) / this.scale;
        this.cy = cy - (y - y0) / this.scale;
        c.classList.add("panning");
        this.invalidate();
      };
      c.onpointerup = (ev) => {
        c.onpointermove = c.onpointerup = null;
        c.classList.remove("panning");
        if (!moved) this.onClick?.(...at(ev));
      };
    });
  }

  render() {
    const {ctx, w, h} = this;
    if (!w || !h) return;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    ctx.fillStyle = theme.color("--bg");
    ctx.fillRect(0, 0, w, h);

    const style = STYLES[theme.name] || STYLES.dark;
    const tz = Math.max(0, Math.min(MAX_Z, Math.round(this.z))), n = 2 ** tz;
    const size = this.scale / n;  // css px per tile at this zoom
    const [ux0, uy0] = this.toUnit(0, 0), [ux1, uy1] = this.toUnit(w, h);
    const onLoad = () => this.invalidate(), left = {fetches: FETCHES, more: false};
    for (let ty = Math.max(0, Math.floor(uy0 * n)); ty <= Math.min(n - 1, Math.floor(uy1 * n)); ty++) {
      for (let tx = Math.floor(ux0 * n); tx <= Math.floor(ux1 * n); tx++) {
        const wx = ((tx % n) + n) % n;
        tile(style, tz, wx, ty, onLoad, left);
        const src = covering(style, tz, wx, ty);
        if (!src) continue;
        const [x, y] = this.toScreen(tx / n, ty / n);
        // whole pixels on both edges, so neighbors meet without seams
        const dx = Math.round(x), dy = Math.round(y);
        ctx.drawImage(src[0], src[1], src[2], src[3], src[3], dx, dy, Math.round(x + size) - dx, Math.round(y + size) - dy);
      }
    }
    if (left.more) this.invalidate();  // the other requests next frame
    this.overlay?.(ctx);
  }

  dispose() {
    this.overlay = this.onClick = null;
  }
}

export function deviceMap(locationPromise) {
  const el = h("div", {class: "device-map"});
  const map = new TileMap(el, {interactive: false});
  new ResizeObserver(([e]) => map.resize(e.contentRect.width, e.contentRect.height)).observe(el);
  Promise.resolve(locationPromise).then((loc) => {
    if (!loc?.lat) return map.view(-117.16, 32.72, 5);
    map.view(loc.lng, loc.lat, 13);
    map.overlay = (ctx) => dot(ctx, ...map.toScreen(unitX(loc.lng), unitY(loc.lat)), 6, theme.color("--accent"));
  });
  return el;
}

function dot(ctx, x, y, r, fill) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = theme.color("--bg");
  ctx.stroke();
}

// the store's GPS columns as decimated {t, lat, lng} points: O(buckets), not O(samples)
function storeTrack(store, duration) {
  const svc = GPS.find((s) => store.series(`/${s}/latitude`));
  if (!svc) return null;
  const lat = store.decimate(`/${svc}/latitude`, 0, duration, BUCKETS), lng = store.decimate(`/${svc}/longitude`, 0, duration, BUCKETS);
  const points = [];
  for (let i = 0; i < lat.t.length; i++) {
    const y = (lat.min[i] + lat.max[i]) / 2, x = (lng.min[i] + lng.max[i]) / 2;
    if (y === y && x === x && (x || y)) points.push({t: lat.t[i], lat: y, lng: x});
  }
  return points;
}

// the trace in unit coordinates with each point's engagement, ready to draw every frame
function trace(points, spans) {
  const engaged = spans.filter((s) => s.kind === "engaged");
  const n = points.length, t = new Float64Array(n), x = new Float64Array(n), y = new Float64Array(n), on = new Uint8Array(n);
  let k = 0;
  points.forEach((p, i) => {
    const ms = p.t * 1000;
    while (k < engaged.length && engaged[k].end < ms) k++;
    t[i] = p.t;
    x[i] = unitX(p.lng);
    y[i] = unitY(p.lat);
    on[i] = k < engaged.length && engaged[k].start <= ms;
  });
  return {n, t, x, y, on};
}

function nearest(tr, t) {
  let lo = 0, hi = tr.n - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (tr.t[mid] < t) lo = mid + 1; else hi = mid; }
  return lo;
}

export function create(ctx) {
  const {clock, route} = ctx;
  const empty = h("p", {class: "map-empty muted", hidden: true}, "No GPS for this route");
  const el = h("div", {class: "map-pane"}, empty);
  const map = new TileMap(el);
  map.view(route.start_lng || -117.16, route.start_lat || 32.72, 12);
  let tr = null, spans = [], fromStore = false, fitted = false, pending = null, disposed = false;

  map.overlay = (g) => {
    if (!tr?.n) return;
    const sx = map.scale, ox = map.w / 2 - map.cx * sx, oy = map.h / 2 - map.cy * sx;
    g.lineWidth = 4;
    g.lineCap = g.lineJoin = "round";
    const colors = [theme.color("--accent"), theme.color("--engaged")];
    for (let i = 0; i < tr.n - 1;) {
      const on = tr.on[i];
      g.beginPath();
      g.moveTo(tr.x[i] * sx + ox, tr.y[i] * sx + oy);
      while (++i < tr.n && tr.on[i] === on) g.lineTo(tr.x[i] * sx + ox, tr.y[i] * sx + oy);
      if (i < tr.n) g.lineTo(tr.x[i] * sx + ox, tr.y[i] * sx + oy);
      g.strokeStyle = colors[on];
      g.stroke();
    }
    const i = nearest(tr, clock.t);
    const [px, py] = map.toScreen(tr.x[i], tr.y[i]);
    // follow the playhead off the edge while playing
    if (clock.playing && (px < 0 || py < 0 || px > map.w || py > map.h)) {
      map.cx = tr.x[i];
      map.cy = tr.y[i];
      return map.invalidate();
    }
    dot(g, px, py, 6, theme.color("--fg"));
  };

  map.onClick = (px, py) => {
    if (!tr?.n) return;
    let best = -1, bestD = PICK_PX ** 2;
    for (let i = 0; i < tr.n; i++) {
      const [x, y] = map.toScreen(tr.x[i], tr.y[i]);
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
    if (best >= 0) clock.seek(tr.t[best]);
  };

  function fit() {
    if (fitted || !tr?.n || !map.w) return;
    fitted = true;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < tr.n; i++) {
      x0 = Math.min(x0, tr.x[i]); x1 = Math.max(x1, tr.x[i]);
      y0 = Math.min(y0, tr.y[i]); y1 = Math.max(y1, tr.y[i]);
    }
    map.fit(x0, y0, x1, y1);
  }

  function update(points) {
    tr = trace(points, spans);
    empty.hidden = tr.n > 0;
    fit();
    map.invalidate();
  }

  const events = routeEvents(route).then((s) => { spans = s; }, () => {});
  Promise.all([events, routeCoords(route).catch(() => [])]).then(([, coords]) => { if (!fromStore && !disposed) update(coords); });

  // once logs are loading, rebuild from the store's GPS at most every 2 s
  const duration = (route.end_time_utc_millis - route.start_time_utc_millis) / 1000;
  const unsubs = [clock.on("t", () => map.invalidate()), theme.on(() => map.invalidate())];
  ctx.onStore((store) => !disposed && unsubs.push(store.on("segment", () => {
    pending ??= setTimeout(() => {
      pending = null;
      const track = storeTrack(store, duration);
      if (track?.length) { fromStore = true; events.then(() => !disposed && update(track)); }
    }, 2000);
  })));

  return {
    name: "map", el,
    render: () => map.render(),
    resize: (w, h) => { map.resize(w, h); fit(); },
    dispose() {
      disposed = true;
      for (const off of unsubs) off?.();
      clearTimeout(pending);
      map.dispose();
    },
    save: () => ({}),
  };
}
