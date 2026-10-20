// mapbox maps: the device location header, and the drive map pane (GPS trace colored by engagement, playhead marker)
import {MAPBOX_TOKEN, MAPBOX_STYLE} from "../geocode.js";
import {routeCoords, routeEvents} from "../api.js";
import {h} from "../dom.js";

const ENGAGED = "#1da756", DRIVING = "#57a9e3";
const GPS = ["gpsLocationExternal", "gpsLocation"];
const BUCKETS = 2000;

function mapbox(container, opts) {
  if (!window.mapboxgl) return null;
  mapboxgl.accessToken = MAPBOX_TOKEN;
  return new mapboxgl.Map({container, style: MAPBOX_STYLE, attributionControl: false, ...opts});
}

export function deviceMap(locationPromise) {
  const el = h("div", {class: "device-map"});
  Promise.resolve(locationPromise).then((location) => {
    const center = location?.lat ? [location.lng, location.lat] : [-117.16, 32.72];
    const map = mapbox(el, {center, zoom: location?.lat ? 13 : 5, interactive: false});
    if (!map) return;
    el._map = map;
    if (location?.lat) new mapboxgl.Marker({color: "#57a9e3"}).setLngLat(center).addTo(map);
  });
  return el;
}

// the store's GPS columns as decimated {t, lat, lng} points, O(buckets)
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
    const color = k < engaged.length && engaged[k].start <= ms ? ENGAGED : DRIVING;
    if (run?.properties.color !== color) {
      const start = run ? [run.geometry.coordinates.at(-1)] : [];
      run = {type: "Feature", properties: {color}, geometry: {type: "LineString", coordinates: start}};
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

// pane: ctx.route is the API route record; GPS from the store once it has some, coords.json until then
export function create(ctx) {
  const {clock, route, store, perf} = ctx;
  const duration = (route.end_time_utc_millis - route.start_time_utc_millis) / 1000;
  const empty = h("p", {class: "muted map-empty", hidden: true}, "No location data");
  const el = h("div", {class: "map-pane"}, empty);
  const map = mapbox(el, {center: [route.start_lng || -117.16, route.start_lat || 32.72], zoom: 12});
  const marker = map && new mapboxgl.Marker({color: "#fff", scale: 0.6});
  let points = [], spans = [], fromStore = false, fitted = false, shown = null, pending = null;

  const pane = {name: "map", el, render, resize, dispose, save: () => ({})};
  const loaded = map ? new Promise((resolve) => map.on("load", resolve)) : Promise.resolve();

  async function update(next) {
    points = next;
    empty.hidden = points.length > 0;
    await loaded;
    if (!map || !points.length) return;
    const data = features(points, spans);
    if (map.getSource("trace")) map.getSource("trace").setData(data);
    else {
      map.addSource("trace", {type: "geojson", data});
      map.addLayer({id: "trace", type: "line", source: "trace", layout: {"line-cap": "round", "line-join": "round"},
        paint: {"line-color": ["get", "color"], "line-width": 4}});
    }
    if (!fitted) {
      fitted = true;
      const bounds = points.reduce((b, p) => b.extend([p.lng, p.lat]), new mapboxgl.LngLatBounds());
      map.fitBounds(bounds, {padding: 40, duration: 0});
    }
    marker.setLngLat([points[0].lng, points[0].lat]).addTo(map);
    shown = null;
    perf.invalidate(pane);
  }

  const events = routeEvents(route).catch(() => []);
  Promise.all([events, routeCoords(route).catch(() => [])]).then(([s, coords]) => { spans = s; if (!fromStore) update(coords); });
  // rebuild from the store at most every 2 s while segments stream in
  const fromColumns = () => {
    pending ??= setTimeout(() => {
      pending = null;
      const track = storeTrack(store, duration);
      if (track?.length) { fromStore = true; events.then(() => update(track)); }
    }, 2000);
  };
  const unsubs = [clock.on("t", () => perf.invalidate(pane)), store?.on("segment", fromColumns)];

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

  function resize() {
    map?.resize();
  }

  function dispose() {
    for (const off of unsubs) off?.();
    clearTimeout(pending);
    map?.remove();
  }

  return pane;
}
