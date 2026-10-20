// the device page: header with location map, then tabs for drives, preserved drives, users and logs
import {api, engagedFraction, routeEvents} from "../api.js";
import {h, fmt} from "../dom.js";
import {isOnline, link} from "../app.js";
import {reverseLookup} from "../geocode.js";
import {deviceMap} from "../panes/map.js";
import {deviceName} from "./home.js";
import {usersSection} from "./users.js";
import {logsSection} from "./logs.js";

const DAY = 86400000, HOUR = 3600000;
const WINDOW = 14 * DAY, LIMIT = 50;
const THUMB_W = 24;  // 18 px tall sprite thumbnails, 4:3

// the first window ends at the next hour boundary so prefetch and page share a cache key
const firstWindowEnd = () => Math.ceil(Date.now() / HOUR) * HOUR;

export function prefetchDevice(dongleId) {
  const end = firstWindowEnd();
  api.routes(dongleId, end - WINDOW, end, LIMIT).catch(() => {});
  api.deviceLocation(dongleId).catch(() => {});
}

const duration = (r) => r.end_time_utc_millis - r.start_time_utc_millis;

function segmentAt(r, ms) {
  const starts = r.segment_start_times;
  if (!starts) return Math.min(r.maxqlog ?? 0, Math.floor(ms / 60000));
  let n = 0;
  for (let i = 0; i < starts.length && starts[i] <= r.start_time_utc_millis + ms; i++) n = r.segment_numbers?.[i] ?? i;
  return n;
}

// engagement spans over sprite thumbnails, filled in when the row nears the viewport
function strip(r, onSpans) {
  const spans = h("div", {class: "strip-spans"});
  const thumbs = h("div", {class: "strip-thumbs"});
  const el = h("div", {class: "strip"}, spans, thumbs);
  const io = new IntersectionObserver((entries) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    io.disconnect();
    const d = duration(r), n = Math.max(1, Math.floor(thumbs.clientWidth / THUMB_W));
    const urls = Array.from({length: n}, (_, i) => api.segmentAsset(r, segmentAt(r, (i + 0.5) / n * d), "sprite.jpg"));
    thumbs.replaceChildren(...urls.map((u) => h("span", {style: `background-image:url(${u})`})));
    routeEvents(r).then((list) => {
      const pct = (ms) => `${(100 * ms / d).toFixed(2)}%`;
      spans.replaceChildren(...list.map((s) => h("span", {class: s.kind, style: `left:${pct(s.start)};width:${pct(Math.max(s.end - s.start, d / 500))}`})));
      onSpans(list);
    }).catch(() => {});
  }, {rootMargin: "400px"});
  io.observe(el);
  return el;
}

function place(lng, lat) {
  const el = h("span", {}, "–");
  reverseLookup(lng, lat).then((loc) => {
    if (!loc) return;
    el.textContent = loc.place || loc.details;
    el.title = [loc.place, loc.details].filter(Boolean).join(", ");
  });
  return el;
}

function driveRow(r) {
  const path = `/${r.fullname.replace("|", "/")}`;
  const start = new Date(r.start_time_utc_millis);
  const engaged = h("td", {class: "num"}, "");
  const hm = (d) => d.toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit", hour12: false});
  return h("tr", {class: "clickable", onclick: (e) => { if (!e.target.closest("a")) link(path)(e); }},
    h("td", {class: "nowrap"}, h("a", {href: path, onclick: link(path)},
      start.toLocaleDateString(undefined, {weekday: "short", month: "short", day: "numeric",
        ...(start.getFullYear() === new Date().getFullYear() ? {} : {year: "numeric"})})),
      h("span", {class: "muted"}, ` ${hm(start)}–${hm(new Date(r.end_time_utc_millis))}`)),
    h("td", {class: "num"}, fmt.minutes(duration(r))),
    h("td", {class: "num"}, fmt.miles(r.distance ?? 0)),
    engaged,
    h("td", {class: "places"}, place(r.start_lng, r.start_lat), h("span", {class: "muted"}, " → "), place(r.end_lng, r.end_lat)),
    h("td", {class: "strip-cell"}, strip(r, (spans) => { engaged.textContent = `${Math.round(100 * engagedFraction(spans, duration(r)))}%`; })),
    h("td", {class: "muted small nowrap"}, r.version || ""));
}

function drivesTable(rows) {
  return h("table", {class: "list drives"},
    h("thead", {}, h("tr", {}, ["drive", "time", "distance", "engaged", "from → to", "timeline", "version"].map((c) => h("th", {}, c)))),
    rows);
}

// newest first, a window at a time; a full window continues from its oldest drive
function drivesSection(dongleId) {
  const body = h("tbody");
  const more = h("button", {class: "button", onclick: loadMore}, "Older drives");
  const status = h("p", {class: "muted small"});
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
      body.append(driveRow(r));
    }
    end = routes.length === LIMIT ? Math.min(...routes.map((r) => r.start_time_utc_millis)) : start;
    more.disabled = false;
    if (!routes.length && !body.children.length && Date.now() - end < 180 * DAY) return loadMore();
    status.textContent = `${body.children.length} drives since ${new Date(end).toLocaleDateString()}`;
  }
  loadMore();
  return h("div", {}, drivesTable(body), h("div", {class: "list-foot"}, status, more));
}

function preservedSection(dongleId) {
  const el = h("div", {}, h("p", {class: "muted"}, "Loading…"));
  api.preservedRoutes(dongleId).then((routes) => {
    for (const r of routes) {
      r.start_time_utc_millis ??= Date.parse(r.start_time + "Z");
      r.end_time_utc_millis ??= Date.parse(r.end_time + "Z");
    }
    el.replaceChildren(routes.length ? drivesTable(h("tbody", {}, routes.map(driveRow))) : h("p", {class: "muted"}, "No preserved drives."));
  }).catch((err) => el.replaceChildren(h("p", {class: "error"}, err.message)));
  return el;
}

function stats(dongleId) {
  const el = h("span", {class: "muted"});
  api.deviceStats(dongleId).then(({all, week}) => {
    el.textContent = `${all.routes} drives · ${Math.round(all.distance)} mi · ${Math.round(all.minutes / 60)} h — this week ${week.routes} drives · ${Math.round(week.distance)} mi`;
  }).catch(() => {});
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
  const body = h("div", {class: "tab-body"});
  const bar = h("nav", {class: "page-tabs"});
  const select = (key) => {
    built[key] ||= tabs[key][1]();
    body.replaceChildren(built[key]);
    for (const b of bar.children) b.classList.toggle("on", b.dataset.tab === key);
    const url = new URL(location.href);
    if (key === "drives") url.searchParams.delete("tab"); else url.searchParams.set("tab", key);
    history.replaceState(history.state, "", url);
  };
  bar.append(...Object.entries(tabs).map(([key, [label]]) => h("button", {class: "tab", "data-tab": key, onclick: () => select(key)}, label)));
  const initial = new URLSearchParams(location.search).get("tab");
  select(tabs[initial] ? initial : "drives");

  const online = isOnline(device);
  return h("div", {class: "page device-page"},
    h("div", {class: "device-head"},
      h("div", {class: "device-info"},
        h("h2", {}, h("span", {class: `dot ${online ? "online" : ""}`}), deviceName(device)),
        h("div", {class: "kv"},
          h("span", {class: "mono"}, dongleId),
          device.device_type && h("span", {}, device.device_type),
          device.openpilot_version && h("span", {}, `openpilot ${device.openpilot_version}`),
          h("span", {}, online ? "online" : `last seen ${fmt.ago(device.last_athena_ping)}`)),
        stats(dongleId)),
      deviceMap(api.deviceLocation(dongleId).catch(() => null))),
    bar, body);
}
