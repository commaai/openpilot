// the device page: location map, header, and tabs for drives, preserved drives, users and logs
import {api, engagedFraction} from "../api.js";
import {h, fmt} from "../dom.js";
import {icon} from "../icons.js";
import {isOnline, link} from "../app.js";
import {reverseLookup} from "../geocode.js";
import {deviceMap} from "../panes/map.js";
import {strip, routeDuration} from "../panes/timeline.js";
import {deviceName} from "./devices.js";
import {usersSection} from "./users.js";
import {logsSection} from "./logs.js";

const DAY = 86400000;
const HOUR = 3600000;
const WINDOW = 14 * DAY;
const LIMIT = 50;

// the first window ends at the next hour boundary so prefetch and page share a cache key
const firstWindowEnd = () => Math.ceil(Date.now() / HOUR) * HOUR;

export function prefetchDevice(dongleId) {
  const end = firstWindowEnd();
  api.routes(dongleId, end - WINDOW, end).catch(() => {});
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
  const engaged = h("div", {class: "first"}, "–");
  return h("a", {class: "drive", href: path, onclick: link(path)},
    h("div", {class: "drive-head"},
      h("div", {class: "cell date"}, h("div", {class: "first"}, date), h("div", {}, `${hm(start)} to ${hm(end)}`)),
      h("div", {class: "cell dur"}, h("div", {class: "first"}, fmt.minutes(duration)), h("div", {}, fmt.miles(r.distance ?? 0))),
      h("div", {class: "cell eng"}, engaged, h("div", {}, "engaged")),
      placeCell(r.start_lng, r.start_lat),
      placeCell(r.end_lng, r.end_lat),
      h("div", {class: "cell arrow"}, icon("chevron", 32))),
    strip(r, (spans) => { engaged.textContent = `${Math.round(100 * engagedFraction(spans, duration))}%`; }));
}

// newest first, a window at a time; a full window pages on from its oldest drive
function drivesSection(dongleId) {
  const list = h("div", {class: "drives"});
  const more = h("button", {class: "button", onclick: loadMore}, "Load older drives");
  const empty = h("p", {class: "muted center", hidden: true}, "No drives in this period.");
  let end = firstWindowEnd();
  const seen = new Set();

  async function loadMore() {
    more.disabled = true;
    const start = end - WINDOW;
    const routes = await api.routes(dongleId, start, end, LIMIT).catch(() => []);
    for (const r of routes) {
      if (seen.has(r.fullname)) continue;
      seen.add(r.fullname);
      api.route.prime(r, r.fullname);
      list.append(driveCard(r));
    }
    end = routes.length === LIMIT ? Math.min(...routes.map((r) => r.start_time_utc_millis)) : start;
    more.disabled = false;
    more.textContent = `Load drives before ${new Date(end).toLocaleDateString()}`;
    if (!routes.length && !list.children.length && Date.now() - end < 180 * DAY) return loadMore();
    empty.hidden = list.children.length > 0;
  }
  loadMore();
  return h("div", {}, list, empty, h("p", {class: "center"}, more));
}

function preservedSection(dongleId) {
  const el = h("div", {class: "drives"}, h("p", {class: "muted"}, "Loading…"));
  api.preservedRoutes(dongleId).then((routes) => {
    for (const r of routes) {
      r.start_time_utc_millis ??= Date.parse(r.start_time + "Z");
      r.end_time_utc_millis ??= Date.parse(r.end_time + "Z");
    }
    el.replaceChildren(...routes.length ? routes.map(driveCard) : [h("p", {class: "muted center"}, "No preserved drives.")]);
  }).catch((err) => el.replaceChildren(h("p", {class: "error"}, String(err))));
  return el;
}

function stats(dongleId) {
  const el = h("div", {class: "device-stats muted small"});
  api.deviceStats(dongleId).then(({all, week}) => el.replaceChildren(
    h("span", {}, `${all.routes} drives · ${Math.round(all.distance)} mi · ${Math.round(all.minutes / 60)} h`),
    h("span", {}, `this week ${week.routes} drives · ${Math.round(week.distance)} mi`))).catch(() => {});
  return el;
}

export async function devicePage(dongleId, devices) {
  const device = (devices || []).find((d) => d.dongle_id === dongleId) || await api.device(dongleId);
  const tabs = {
    drives: ["Drives", () => drivesSection(dongleId)],
    preserved: ["Preserved", () => preservedSection(dongleId)],
    users: ["Users", () => usersSection(device)],
    bootlogs: ["Boot logs", () => logsSection(dongleId, "boot")],
    crashlogs: ["Crash logs", () => logsSection(dongleId, "crash")],
  };
  const built = {};
  const body = h("div", {class: "page"});
  const pills = h("div", {class: "pills"});
  const select = (key) => {
    built[key] ||= tabs[key][1]();
    body.replaceChildren(built[key]);
    for (const p of pills.children) p.classList.toggle("dim", p.dataset.tab !== key);
    const url = new URL(location.href);
    if (key === "drives") url.searchParams.delete("tab"); else url.searchParams.set("tab", key);
    history.replaceState(history.state, "", url);
  };
  pills.append(...Object.entries(tabs).map(([key, [label]]) => h("div", {class: "pill-item", "data-tab": key, onclick: () => select(key)}, label)));
  const initial = new URLSearchParams(location.search).get("tab");
  select(tabs[initial] ? initial : "drives");

  const online = isOnline(device);
  return h("div", {class: "device-page"},
    deviceMap(api.deviceLocation(dongleId).catch(() => null)),
    h("div", {class: "device-head"},
      h("div", {},
        h("h2", {}, deviceName(device)),
        h("div", {class: "muted small"}, h("span", {class: "mono"}, dongleId), device.openpilot_version ? ` · openpilot ${device.openpilot_version}` : "",
          online ? "" : ` · last seen ${fmt.ago(device.last_athena_ping)}`),
        stats(dongleId)),
      h("span", {class: `pill ${online ? "online" : ""}`}, online ? "device online" : "device offline")),
    h("div", {class: "device-tabs"}, pills),
    body);
}
