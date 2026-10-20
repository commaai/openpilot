// mapbox maps: the device location on the device page, and the map pane (GPS trace colored by engagement,
// a playhead marker, click to seek). The trace comes from coords.json, or from the store's GPS once it's loaded.
import {MAPBOX_TOKEN} from "../geocode.js";
import {routeCoords, routeEvents} from "../api.js";
import {h, css} from "../dom.js";
import {theme} from "../theme.js";

const STYLES = {dark: "mapbox://styles/commaai/cjj4yzqk201c52ss60ebmow0w", light: "mapbox://styles/mapbox/light-v10"};
const GPS = ["gpsLocationExternal", "gpsLocation"];
const BUCKETS = 2000;

function mapbox(container, opts) {
  if (!window.mapboxgl) return null;
  mapboxgl.accessToken = MAPBOX_TOKEN;
  return new mapboxgl.Map({container, style: STYLES[theme.name] || STYLES.dark, attributionControl: false, ...opts});
}

export function deviceMap(locationPromise) {
  const el = h("div", {class: "device-map"});
  Promise.resolve(locationPromise).then((location) => {
    const center = location?.lat ? [location.lng, location.lat] : [-117.16, 32.72];
    const map = mapbox(el, {center, zoom: location?.lat ? 13 : 5, interactive: false});
    if (!map) return;
    el._map = map;
    if (location?.lat) new mapboxgl.Marker({color: theme.color("--accent")}).setLngLat(center).addTo(map);
  });
  return el;
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

// one LineString per run of engaged or not, so mapbox colors by property
function features(points, spans) {
  const engaged = spans.filter((s) => s.kind === "engaged");
  const out = [];
  let k = 0, run = null;
  for (const p of points) {
    const ms = p.t * 1000;
    while (k < engaged.length && engaged[k].end < ms) k++;
    const on = k < engaged.length && engaged[k].start <= ms;
    if (run?.properties.engaged !== on) {
      run = {type: "Feature", properties: {engaged: on}, geometry: {type: "LineString", coordinates: run ? [run.geometry.coordinates.at(-1)] : []}};
      out.push(run);
    }
    run.geometry.coordinates.push([p.lng, p.lat]);
  }
  return {type: "FeatureCollection", features: out};
}

function nearest(points, t) {
  let lo = 0, hi = points.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (points[mid].t < t) lo = mid + 1; else hi = mid; }
  return points[lo];
}

export function create(ctx) {
  css(new URL("./map.css", import.meta.url));
  const {clock, route, perf} = ctx;
  const empty = h("p", {class: "map-empty muted", hidden: true}, "No GPS for this route");
  const el = h("div", {class: "map-pane"}, empty);
  const map = mapbox(el, {center: [route.start_lng || -117.16, route.start_lat || 32.72], zoom: 12});
  const marker = map && new mapboxgl.Marker({color: theme.color("--fg"), scale: 0.6});
  let points = [], spans = [], fromStore = false, fitted = false, shown = null, pending = null, styled = false, disposed = false;

  const pane = {name: "map", el, render, resize: () => map?.resize(), dispose, save: () => ({})};

  function draw() {
    if (!styled || !points.length) return;
    const data = features(points, spans);
    if (map.getSource("trace")) return map.getSource("trace").setData(data);
    map.addSource("trace", {type: "geojson", data});
    map.addLayer({id: "trace", type: "line", source: "trace", layout: {"line-cap": "round", "line-join": "round"},
      paint: {"line-color": ["case", ["get", "engaged"], theme.color("--engaged"), theme.color("--accent")], "line-width": 4}});
  }

  function update(next) {
    points = next;
    empty.hidden = points.length > 0;
    if (!map || !points.length) return;
    if (!fitted) {
      fitted = true;
      map.fitBounds(points.reduce((b, p) => b.extend([p.lng, p.lat]), new mapboxgl.LngLatBounds()), {padding: 30, duration: 0});
      marker.setLngLat([points[0].lng, points[0].lat]).addTo(map);
    }
    draw();
    shown = null;
    perf.invalidate(pane);
  }

  map?.on("style.load", () => { styled = true; draw(); });
  const events = routeEvents(route).then((s) => { spans = s; }, () => {});
  Promise.all([events, routeCoords(route).catch(() => [])]).then(([, coords]) => { if (!fromStore) update(coords); });

  // once logs are loading, rebuild from the store's GPS at most every 2 s
  const duration = (route.end_time_utc_millis - route.start_time_utc_millis) / 1000;
  const unsubs = [clock.on("t", () => perf.invalidate(pane)), theme.on(() => { styled = false; map?.setStyle(STYLES[theme.name]); })];
  ctx.onStore((store) => !disposed && unsubs.push(store.on("segment", () => {
    pending ??= setTimeout(() => {
      pending = null;
      const track = storeTrack(store, duration);
      if (track?.length) { fromStore = true; events.then(() => update(track)); }
    }, 2000);
  })));

  map?.on("click", (e) => {
    if (!points.length) return;
    const {lng, lat} = e.lngLat;
    let best = points[0], bestD = Infinity;
    for (const p of points) {
      const d = (p.lng - lng) ** 2 + (p.lat - lat) ** 2;
      if (d < bestD) { bestD = d; best = p; }
    }
    clock.seek(best.t);
  });

  function render() {
    if (!marker || !points.length) return;
    const p = nearest(points, clock.t);
    if (p !== shown) { shown = p; marker.setLngLat([p.lng, p.lat]); }
  }

  function dispose() {
    disposed = true;
    for (const off of unsubs) off?.();
    clearTimeout(pending);
    map?.remove();
  }

  return pane;
}
