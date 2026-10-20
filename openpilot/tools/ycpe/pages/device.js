import {api} from "/api.js";
import {h, fmt} from "/dom.js";
import {icon} from "/icons.js";
import {isOnline, link} from "/app.js";
import {reverseLookup} from "/geocode.js";
import {deviceMap} from "/map.js";
import {timeline, routeDuration} from "/timeline.js";

const DAY = 86400000;
const HOUR = 3600000;

// the first 14-day window ends at the next hour boundary so prefetch and page share a cache key
const firstWindowEnd = () => Math.ceil(Date.now() / HOUR) * HOUR;

export function prefetchDevice(dongleId) {
  const end = firstWindowEnd();
  api.routes(dongleId, end - 14 * DAY, end).catch(() => {});
  api.deviceLocation(dongleId).catch(() => {});
}

function placeCell(lng, lat) {
  const cell = h("div", {class: "cell"});
  reverseLookup(lng, lat).then((loc) => {
    if (loc) { cell.replaceChildren(h("div", {class: "first"}, loc.place), h("div", {}, loc.details)); cell.classList.add("enter"); }
  });
  return cell;
}

export function driveCard(r) {
  const path = `/${r.dongle_id}/${r.fullname.split("|")[1]}`;
  const start = new Date(r.start_time_utc_millis), end = new Date(r.end_time_utc_millis);
  const sameYear = start.getFullYear() === new Date().getFullYear();
  const date = start.toLocaleDateString(undefined, {weekday: "long", month: "short", day: "numeric", ...(sameYear ? {} : {year: "numeric"})});
  const hm = (d) => d.toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit", hour12: false});
  const duration = routeDuration(r);
  return h("a", {class: "drive", href: path, onclick: link(path)},
    h("div", {class: "drive-head"},
      h("div", {class: "cell date"}, h("div", {class: "first"}, date), h("div", {}, `${hm(start)} to ${hm(end)}`)),
      h("div", {class: "cell dur"}, h("div", {class: "first"}, fmt.minutes(duration)), h("div", {}, fmt.miles(r.distance ?? 0))),
      placeCell(r.start_lng, r.start_lat),
      placeCell(r.end_lng, r.end_lat),
      h("div", {class: "cell arrow"}, icon("chevron", 32))),
    timeline(r));
}

export async function devicePage(dongleId, devices) {
  const device = (devices || []).find((d) => d.dongle_id === dongleId) || await api.device(dongleId);
  const location = api.deviceLocation(dongleId).catch(() => null);
  const list = h("div", {class: "drives"});
  const more = h("button", {class: "button"}, "Load older drives");
  const empty = h("p", {class: "muted center", hidden: true}, "No drives in this period.");
  let end = firstWindowEnd();
  const seen = new Set();

  async function loadMore() {
    more.disabled = true;
    const start = end - 14 * DAY;
    const routes = await api.routes(dongleId, start, end);
    for (const r of routes) {
      if (seen.has(r.fullname)) continue;
      seen.add(r.fullname);
      api.route.prime(r, r.fullname);
      list.append(driveCard(r));
    }
    end = start;
    more.disabled = false;
    if (!routes.length && !list.children.length && Date.now() - end < 180 * DAY) return loadMore();
    empty.hidden = list.children.length > 0;
  }
  loadMore();

  const online = isOnline(device);
  return h("div", {class: "device-page"},
    deviceMap(location),
    h("div", {class: "device-head"},
      h("h2", {}, device.alias || dongleId),
      h("div", {class: "device-buttons"},
        h("span", {class: `pill ${online ? "online" : ""}`}, online ? "device online" : "device offline"))),
    h("div", {class: "page"}, list, empty, h("p", {class: "center"}, more)));
}
