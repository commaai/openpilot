// canvas maps from Mapbox raster tiles: the device page's (where the car was last), and the map pane (GPS trace
// colored by engagement, a playhead marker, click to seek). Tiles decode off the main thread and every draw goes
// through the frame scheduler, so the map never costs a frame. The trace comes from coords.json, or from the
// store's GPS once logs are loading.
import {MAPBOX_TOKEN, reverseLookup} from "/geocode.js";
import {routeCoords, routeEvents} from "/api.js";
import {lowerBound} from "/data/bounds.js";
import {h, css, fitCanvas, fmt} from "/dom.js";
import {icon} from "/icons.js";
import {perf} from "/perf.js";
import {theme} from "/theme.js";

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
// Each render stamps the tiles it draws (left.stamp), and eviction after it keeps those: a big map needs more than
// CACHE, and evicting what's on screen would refetch it forever.
const tiles = new Map();
let renders = 0;
function tile(style, z, x, y, onLoad, left) {
  const key = `${style}/${z}/${x}/${y}`;
  let t = tiles.get(key);
  if (t) { tiles.delete(key); tiles.set(key, t); t.used = left.stamp; return t.bitmap; }
  if (!left.fetches) return left.more = true, null;
  left.fetches--;
  tiles.set(key, t = {bitmap: null, used: left.stamp});
  t.loaded = fetch(`https://api.mapbox.com/styles/v1/${style}/tiles/${TILE}/${z}/${x}/${y}@2x?access_token=${MAPBOX_TOKEN}`)
    .then((r) => r.ok ? r.blob() : Promise.reject(r.status))
    .then((b) => { const px = tilePx(); return createImageBitmap(b, px < 2 * TILE ? {resizeWidth: px, resizeHeight: px, resizeQuality: "high"} : {}); })
    .then((bitmap) => {
      if (tiles.get(key) !== t) return bitmap.close();  // evicted while loading
      t.bitmap = bitmap;
      onLoad();
    })
    .catch(() => tiles.get(key) === t && tiles.delete(key));
  return null;
}

function evict(keep, stamp) {
  for (const [k, t] of tiles) {
    if (tiles.size <= keep) break;
    if (t.used === stamp) continue;
    t.bitmap?.close();
    tiles.delete(k);
  }
}

// a loaded tile or the nearest loaded ancestor, as [bitmap, sx, sy, size] in image pixels
function covering(style, z, x, y, stamp) {
  for (let up = 0; up <= Math.min(z, 5); up++) {
    const t = tiles.get(`${style}/${z - up}/${x >> up}/${y >> up}`);
    if (!t?.bitmap) continue;
    t.used = stamp;
    const size = t.bitmap.width >> up, mask = (1 << up) - 1;
    return [t.bitmap, (x & mask) * size, (y & mask) * size, size];
  }
  return null;
}

class TileMap {
  constructor(el) {
    this.name = "map";
    this.el = el;
    this.canvas = el.appendChild(h("canvas", {class: "tilemap"}));
    this.w = this.h = 0;
    this.cx = this.cy = 0.5;
    this.z = 3;
    this.overlay = null;      // fn(ctx) drawing on top, in css px
    this.dark = false;        // connect's dark style whatever the theme (user mode)
    this.touched = 0;         // when the user last panned or zoomed it
    this.interact();
    el.append(h("div", {class: "tilemap-credit"},
      h("a", {href: "https://www.mapbox.com/about/maps/", target: "_blank"}, "© Mapbox"), " ",
      h("a", {href: "https://www.openstreetmap.org/copyright", target: "_blank"}, "© OpenStreetMap")));
  }

  invalidate() { perf.invalidate(this); }

  resize(w, h) {
    this.w = w;
    this.h = h;
    this.invalidate();
  }

  get scale() { return TILE * 2 ** this.z; }  // css px per world
  toScreen(ux, uy) { return [(ux - this.cx) * this.scale + this.w / 2, (uy - this.cy) * this.scale + this.h / 2]; }
  toUnit(px, py, w = this.w, h = this.h) { return [(px - w / 2) / this.scale + this.cx, (py - h / 2) / this.scale + this.cy]; }

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
    this.touched = performance.now();
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

    // one pointer pans; two pan and pinch-zoom around their midpoint, keeping the point under it there. Where the page
    // scrolls under the map (map.css: touch-action pan-y on a phone), one finger is the page's and the map takes two.
    const down = new Map();  // pointerId → [x, y]
    let from = null, moved = false, scrolls = false;  // from: the gesture's midpoint, spread, unit point and zoom
    const spread = () => {
      const [[x0, y0], [x1, y1] = [x0, y0]] = down.values();
      return [(x0 + x1) / 2, (y0 + y1) / 2, Math.hypot(x1 - x0, y1 - y0)];
    };
    const start = () => {
      const [x, y, d] = spread();
      from = {x, y, d, u: this.toUnit(x, y), z: this.z};
    };
    c.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      c.setPointerCapture(e.pointerId);
      down.set(e.pointerId, at(e));
      moved = down.size > 1;  // two pointers never click
      scrolls = e.pointerType === "touch" && getComputedStyle(c).touchAction === "pan-y";
      start();
    });
    c.addEventListener("pointermove", (e) => {
      if (!down.size) return this.onHover?.(...at(e));
      if (!down.has(e.pointerId)) return;
      down.set(e.pointerId, at(e));
      const [x, y, d] = spread();
      moved ||= Math.hypot(x - from.x, y - from.y) > 3;
      if (!moved || scrolls && down.size < 2) return;
      this.z = Math.max(MIN_Z, Math.min(MAX_Z, from.z + (from.d && down.size > 1 ? Math.log2(d / from.d) : 0)));
      this.cx = from.u[0] - (x - this.w / 2) / this.scale;
      this.cy = from.u[1] - (y - this.h / 2) / this.scale;
      c.classList.add("panning");
      this.touched = performance.now();
      this.invalidate();
    });
    const up = (e) => {
      if (!down.delete(e.pointerId)) return;
      if (down.size) return start();  // the other pointer carries on from here
      c.classList.remove("panning");
      if (!moved && e.type === "pointerup") this.onClick?.(...at(e));
    };
    c.addEventListener("pointerup", up);
    c.addEventListener("pointercancel", up);
    c.addEventListener("pointerleave", () => this.onHover?.(-1, -1));
  }

  // fn(style, z, x, y, tx, ty) for each tile a w × h view shows, x wrapped around the world
  eachTile(w, h, fn) {
    const style = this.dark ? STYLES.dark : STYLES[theme.name] || STYLES.dark;
    const tz = Math.max(0, Math.min(MAX_Z, Math.round(this.z))), n = 2 ** tz;
    const [ux0, uy0] = this.toUnit(0, 0, w, h), [ux1, uy1] = this.toUnit(w, h, w, h);
    for (let ty = Math.max(0, Math.floor(uy0 * n)); ty <= Math.min(n - 1, Math.floor(uy1 * n)); ty++) {
      for (let tx = Math.floor(ux0 * n); tx <= Math.floor(ux1 * n); tx++) fn(style, tz, ((tx % n) + n) % n, ty, tx, ty);
    }
  }

  // the view's tiles at a size it isn't laid out at yet, fetched and decoded
  preload(w, h) {
    const left = {fetches: Infinity, stamp: renders}, loading = [];
    this.eachTile(w, h, (style, z, x, y) => {
      tile(style, z, x, y, () => {}, left);
      loading.push(tiles.get(`${style}/${z}/${x}/${y}`)?.loaded);
    });
    return Promise.allSettled(loading);
  }

  render() {
    const {w, h} = this;
    if (!w || !h) return;
    const ctx = fitCanvas(this.canvas, w, h);
    ctx.fillStyle = this.dark ? "#1d2225" : theme.color("--bg");
    ctx.fillRect(0, 0, w, h);

    const onLoad = () => this.invalidate(), left = {fetches: FETCHES, more: false, stamp: ++renders};
    let drawn = 0;
    this.eachTile(w, h, (style, z, x, y, tx, ty) => {
      tile(style, z, x, y, onLoad, left);
      drawn++;
      const src = covering(style, z, x, y, left.stamp);
      if (!src) return;
      const n = 2 ** z, size = this.scale / n;  // css px per tile at this zoom
      const [sx, sy] = this.toScreen(tx / n, ty / n);
      // whole pixels on both edges, so neighbors meet without seams
      const dx = Math.round(sx), dy = Math.round(sy);
      ctx.drawImage(src[0], src[1], src[2], src[3], src[3], dx, dy, Math.round(sx + size) - dx, Math.round(sy + size) - dy);
    });
    evict(Math.max(CACHE, 2 * drawn), left.stamp);  // room for this view and the one after a pan or zoom
    if (left.more) this.invalidate();  // the other requests next frame
    this.overlay?.(ctx);
  }

  dispose() {
    this.overlay = this.onClick = this.onHover = null;
  }
}

const PIN = 30;  // css px from the car pin's point to its top
const TIME = new Intl.DateTimeFormat(undefined, {hour: "numeric", minute: "2-digit"});

// the device's last location, as connect's map has it: pan and zoom, the car's pin (pointed at, it says when it was
// there; tapped, a card with the address and a link to the maps app), and your own location on asking. User mode's
// (connect) has connect's zoom and pin (drawPin), and doesn't ask
export function deviceMap(device, locationPromise, {connect = false, drawPin = null} = {}) {
  const tip = h("div", {class: "map-tip", hidden: true});
  const card = h("div", {class: "map-card", hidden: true});
  const locate = !connect && navigator.geolocation && h("button", {class: "map-btn", title: "Show my location", "aria-label": "Show my location", onclick: () => findMe()}, icon("myLocation", 18));
  const el = h("div", {class: "device-map"});
  const map = new TileMap(el);
  el.append(tip, card, locate || "");
  const at = (lng, lat) => ({lng, lat, x: unitX(lng), y: unitY(lat)});
  let car = null, me = null;
  const sized = new ResizeObserver(([e]) => {
    const first = !map.w;
    map.resize(e.contentRect.width, e.contentRect.height);
    if (first) map.render();  // before its first paint, with the tiles preloaded: it's never shown empty
  });
  sized.observe(el);
  const offTheme = theme.on(() => map.invalidate());
  el.dispose = () => { sized.disconnect(); offTheme(); map.dispose(); };

  map.overlay = (g) => {
    if (me) dot(g, ...map.toScreen(me.x, me.y), 6, "#31a1ee");
    if (!car) return;
    const [x, y] = map.toScreen(car.x, car.y);
    if (drawPin) drawPin(g, x, y); else pin(g, x, y, theme.color("--accent"));
    tip.style.translate = `${Math.round(x)}px ${Math.round(y - PIN - 4)}px`;  // map.css centers it above
  };
  const onPin = (px, py) => {
    if (!car) return false;
    const [x, y] = map.toScreen(car.x, car.y);
    return Math.abs(px - x) <= 12 && py <= y + 2 && py >= y - PIN - 2;
  };
  map.onHover = (px, py) => {
    const over = onPin(px, py);
    if (over && tip.hidden && car.time) tip.replaceChildren(`${TIME.format(car.time)},`, h("br"), fmt.ago(car.time / 1000));
    tip.hidden = !over || !car.time;
    map.canvas.classList.toggle("pointing", over);
  };
  map.onClick = (px, py) => { if (onPin(px, py)) showCard(); };

  function showCard() {
    const name = fmt.deviceName(device), {lat, lng} = car;
    // Apple Maps on an iPhone or iPad, Google's everywhere else, as connect links them
    const maps = /iphone|ipad|ipod/i.test(navigator.userAgent) ? `https://maps.apple.com/?ll=${lat},${lng}&q=${encodeURIComponent(name)}` : `https://maps.google.com/?q=${lat},${lng}`;
    const address = h("div", {class: "small muted"}, `${lat.toFixed(5)}, ${lng.toFixed(5)}`);
    card.replaceChildren(
      h("div", {class: "map-card-head"},
        h("div", {}, h("b", {}, name), car.time && h("div", {class: "small muted"}, fmt.ago(car.time / 1000))),
        h("a", {class: "button small", href: maps, target: "_blank", rel: "noopener"}, "Open in maps"),
        h("button", {class: "icon-btn", title: "Close", "aria-label": "Close", onclick: () => { card.hidden = true; }}, icon("close", 16))),
      address);
    card.hidden = false;
    reverseLookup(lng, lat).then((loc) => { if (loc?.address) address.textContent = loc.address; });
  }

  // asked for only on the button's press; the view takes in both you and the car
  function findMe() {
    locate.disabled = true;
    navigator.geolocation.getCurrentPosition((pos) => {
      locate.disabled = false;
      locate.title = "Show my location";
      me = at(pos.coords.longitude, pos.coords.latitude);
      if (!car) return map.view(me.lng, me.lat, 13);
      map.fit(Math.min(me.x, car.x), Math.min(me.y, car.y), Math.max(me.x, car.x), Math.max(me.y, car.y), 40);
    }, (err) => {
      locate.disabled = false;
      locate.title = `Couldn't find your location: ${err.message || "not allowed"}`;
    }, {enableHighAccuracy: true, timeout: 15000});
  }

  // ready once the view's tiles are in, at the size of the map this one replaces
  el.ready = Promise.resolve(locationPromise).then((loc) => {
    if (!loc?.lat) map.view(-117.16, 32.72, 5);
    else {
      car = {...at(loc.lng, loc.lat), time: loc.time};
      // connect fits the car and a point 0.01° northwest of it at its zoom cap, so the car sits off center
      if (connect) map.view(loc.lng - 0.005, loc.lat + 0.005, 11); else map.view(loc.lng, loc.lat, 13);
    }
    const shown = document.querySelector(".device-map");
    return shown && map.preload(shown.clientWidth, shown.clientHeight);
  });
  return el;
}

// a map pin with its point at (x, y)
function pin(ctx, x, y, fill) {
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.bezierCurveTo(x - 2, y - 8, x - 10, y - 13, x - 10, y - 20);
  ctx.arc(x, y - 20, 10, Math.PI, 0);
  ctx.bezierCurveTo(x + 10, y - 13, x + 2, y - 8, x, y);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = theme.color("--bg");
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, y - 20, 4, 0, 2 * Math.PI);
  ctx.fillStyle = theme.color("--bg");
  ctx.fill();
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

// an arrowhead pointing at angle (radians, clockwise from east)
function arrow(ctx, x, y, angle, fill) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.beginPath();
  ctx.moveTo(9, 0);
  ctx.lineTo(-6, 7);
  ctx.lineTo(-3, 0);
  ctx.lineTo(-6, -7);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.strokeStyle = theme.color("--bg");
  ctx.stroke();
  ctx.restore();
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

export function create(ctx) {
  const {clock, route} = ctx;
  const empty = h("p", {class: "map-empty muted", hidden: true}, "No GPS for this route");
  const el = h("div", {class: "map-pane"}, empty);
  const map = new TileMap(el);
  map.view(route.start_lng || -117.16, route.start_lat || 32.72, ctx.user ? 14 : 12);
  let tr = null, spans = [], fromStore = false, fitted = false, pending = null, disposed = false;

  // the playhead between the trace's points, in unit coordinates
  const playhead = () => {
    const i = lowerBound(tr.t, clock.t, 0, tr.n - 1), a = Math.max(0, i - 1);
    const f = i > a ? Math.max(0, Math.min(1, (clock.t - tr.t[a]) / (tr.t[i] - tr.t[a]))) : 0;
    return [tr.x[a] + f * (tr.x[i] - tr.x[a]), tr.y[a] + f * (tr.y[i] - tr.y[a]), i];
  };
  // user mode's is connect's DriveMap: its dark style, a grey line, a blue dot at the playhead, and the view at zoom 14
  // on the dot unless the map was moved in the last 5 s
  const connectMap = (g) => {
    const sx = map.scale, ox = map.w / 2 - map.cx * sx, oy = map.h / 2 - map.cy * sx;
    g.lineWidth = 8;
    g.lineCap = g.lineJoin = "round";
    g.beginPath();
    for (let i = 0; i < tr.n; i++) g[i ? "lineTo" : "moveTo"](tr.x[i] * sx + ox, tr.y[i] * sx + oy);
    g.strokeStyle = "#888";
    g.stroke();
    const [px, py] = map.toScreen(...playhead());
    g.beginPath();
    g.arc(px, py, 10, 0, 2 * Math.PI);
    g.fillStyle = "#007cbf";
    g.fill();
  };
  let followed = "";  // as connect, it moves only when the dot does
  const follow = () => {
    map.dark = !!ctx.user;
    if (!ctx.user || !tr?.n) return;
    const [ux, uy] = playhead();
    if (followed === `${ux} ${uy}`) return;
    followed = `${ux} ${uy}`;
    if (performance.now() - map.touched > 5000) [map.cx, map.cy] = [ux, uy];
  };

  map.overlay = (g) => {
    if (!tr?.n) return;
    if (ctx.user) return connectMap(g);
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
    // the playhead between the trace's points, pointing along the GPS bearing once logs are loading, else the trace
    const [ux, uy, i] = playhead();
    const [px, py] = map.toScreen(ux, uy);
    // follow the playhead off the edge while playing
    if (clock.playing && (px < 0 || py < 0 || px > map.w || py > map.h)) {
      map.cx = ux;
      map.cy = uy;
      return map.invalidate();
    }
    const store = ctx.peekStore(), bearing = store ? GPS.map((s) => store.valueAt(`/${s}/bearingDeg`, clock.t)).find((v) => v !== undefined) : undefined;
    const j = Math.max(1, Math.min(tr.n - 1, i));  // the trace's step under the playhead, for its direction
    const angle = bearing !== undefined ? (bearing - 90) * Math.PI / 180 : Math.atan2(tr.y[j] - tr.y[j - 1], tr.x[j] - tr.x[j - 1]) || 0;
    arrow(g, px, py, angle, theme.color("--fg"));
  };

  map.onClick = (px, py) => {
    if (!tr?.n || ctx.user) return;  // connect's map doesn't seek
    let best = -1, bestD = PICK_PX ** 2;
    for (let i = 0; i < tr.n; i++) {
      const [x, y] = map.toScreen(tr.x[i], tr.y[i]);
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
    if (best >= 0) clock.seek(tr.t[best]);
  };

  function fit() {
    if (fitted || !tr?.n || !map.w || ctx.user) return;
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
  unsubs.push(ctx.onStore((store) => !disposed && unsubs.push(store.on("segment", () => {
    pending ??= setTimeout(() => {
      pending = null;
      const track = storeTrack(store, route.live ? store.route.duration : duration);  // a live session grows
      if (track?.length) { fromStore = true; events.then(() => !disposed && update(track)); }
    }, 2000);
  }))));

  return {
    name: "map", el,
    render: () => { follow(); map.render(); },
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
