// mapbox maps: the device header and the drive map
import {MAPBOX_TOKEN, MAPBOX_STYLE} from "/geocode.js";
import {h} from "/dom.js";

function create(container, opts) {
  if (!window.mapboxgl) return null;
  mapboxgl.accessToken = MAPBOX_TOKEN;
  return new mapboxgl.Map({container, style: MAPBOX_STYLE, attributionControl: false, ...opts});
}

export function deviceMap(locationPromise) {
  const el = h("div", {class: "device-map"});
  Promise.resolve(locationPromise).then((location) => {
    const center = location && location.lat ? [location.lng, location.lat] : [-117.16, 32.72];
    const map = create(el, {center, zoom: location && location.lat ? 13 : 5, interactive: false});
    if (!map) return;
    el._map = map;
    if (location && location.lat) new mapboxgl.Marker({color: "#57a9e3"}).setLngLat(center).addTo(map);
  });
  return el;
}

// coords: [{t (route seconds), lng, lat}], the marker follows the video
export function driveMap(coords, video) {
  const el = h("div", {class: "drive-map"});
  requestAnimationFrame(() => {
    const line = coords.map((c) => [c.lng, c.lat]);
    if (!line.length) { el.append(h("p", {class: "muted center"}, "No location data")); return; }
    const map = create(el, {center: line[0], zoom: 14});
    if (!map) return;
    el._map = map;
    const bounds = line.reduce((b, p) => b.extend(p), new mapboxgl.LngLatBounds(line[0], line[0]));
    map.fitBounds(bounds, {padding: 40, duration: 0});
    map.on("load", () => {
      map.addSource("route", {type: "geojson", data: {type: "Feature", geometry: {type: "LineString", coordinates: line}}});
      map.addLayer({id: "route", type: "line", source: "route", paint: {"line-color": "#888", "line-width": 4}});
    });
    const marker = new mapboxgl.Marker({color: "#57a9e3"}).setLngLat(line[0]).addTo(map);
    video.addEventListener("timeupdate", () => {
      const t = video.currentTime;
      let lo = 0, hi = coords.length - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (coords[mid].t < t) lo = mid + 1; else hi = mid; }
      marker.setLngLat([coords[lo].lng, coords[lo].lat]);
    });
  });
  return el;
}
