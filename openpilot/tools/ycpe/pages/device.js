// the device page: header with location map, then tabs for drives, preserved drives, uploads, users, logs and settings
import {api, auth, engagedFraction, routeEvents} from "../api.js";
import {h, fmt} from "../dom.js";
import {devicesChanged, go, isOnline, link} from "../app.js";
import {perf} from "../perf.js";
import {reverseLookup} from "../geocode.js";
import {deviceMap} from "../panes/map.js";
import {deviceName} from "./home.js";
import {usersSection} from "./users.js";
import {logsSection} from "./logs.js";
import {deviceUploads} from "./files.js";
import {settingsSection} from "./settings.js";

const DAY = 86400000, HOUR = 3600000;
const WINDOW = 14 * DAY, LIMIT = 50;
const THUMBS = 16;  // sprite thumbnails per strip

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

// engagement spans over sprite thumbnails, filled in when the row nears the viewport. One observer for every
// strip, and a fixed thumbnail count, so filling one never reads layout.
const fillers = new WeakMap();
const queue = [];
let draining = null;
const nearby = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    nearby.unobserve(e.target);
    queue.push(fillers.get(e.target));
  }
  // ~2 ms of strips per frame, so a fast scroll through hundreds of drives stays smooth
  draining ??= perf.tick(() => {
    const end = performance.now() + 2;
    while (queue.length && performance.now() < end) queue.shift()();
    if (!queue.length) { draining(); draining = null; }
  }, "drives");
}, {rootMargin: "400px"});

function strip(r, onSpans, onVisible) {
  const spans = h("div", {class: "strip-spans"});
  const thumbs = h("div", {class: "strip-thumbs"});
  const el = h("div", {class: "strip"}, thumbs, spans);
  fillers.set(el, () => {
    onVisible();
    const d = duration(r);
    thumbs.replaceChildren(...Array.from({length: THUMBS}, (_, i) =>
      h("span", {style: `background-image:url(${api.segmentAsset(r, segmentAt(r, (i + 0.5) / THUMBS * d), "sprite.jpg")})`})));
    routeEvents(r).then((list) => {
      const pct = (ms) => `${(100 * ms / d).toFixed(2)}%`;
      spans.replaceChildren(...list.map((s) => h("span", {class: s.kind, style: `left:${pct(s.start)};width:${pct(Math.max(s.end - s.start, d / 500))}`})));
      onSpans(list);
    }).catch(() => {});
  });
  nearby.observe(el);
  return el;
}

function place(el, lng, lat) {
  reverseLookup(lng, lat).then((loc) => {
    if (!loc) return;
    el.textContent = loc.place || loc.details;
    el.title = [loc.place, loc.details].filter(Boolean).join(", ");
  });
}

const HM = new Intl.DateTimeFormat(undefined, {hour: "2-digit", minute: "2-digit", hour12: false});
const DAY_FMT = new Intl.DateTimeFormat(undefined, {weekday: "short", month: "short", day: "numeric"});
const OLD_DAY_FMT = new Intl.DateTimeFormat(undefined, {weekday: "short", month: "short", day: "numeric", year: "numeric"});

function driveRow(r) {
  const path = `/${r.fullname.replace("|", "/")}`;
  const start = new Date(r.start_time_utc_millis);
  const engaged = h("span", {class: "num"});
  const from = h("span", {}, "–"), to = h("span", {}, "–");
  return h("a", {class: "drive-row", href: path, onclick: link(path)},
    h("span", {}, h("b", {}, (start.getFullYear() === new Date().getFullYear() ? DAY_FMT : OLD_DAY_FMT).format(start)),
      h("span", {class: "muted"}, ` ${HM.format(start)}–${HM.format(r.end_time_utc_millis)}`)),
    h("span", {class: "num"}, fmt.minutes(duration(r))),
    h("span", {class: "num"}, fmt.miles(r.distance ?? 0)),
    engaged,
    h("span", {}, from, h("span", {class: "muted"}, " → "), to),
    strip(r, (spans) => { engaged.textContent = `${Math.round(100 * engagedFraction(spans, duration(r)))}%`; },
      () => { place(from, r.start_lng, r.start_lat); place(to, r.end_lng, r.end_lat); }),
    h("span", {class: "muted"}, r.version || ""));
}

const drivesHead = () => h("div", {class: "drive-row head"}, ["drive", "time", "distance", "engaged", "from → to", "timeline", "version"].map((c) => h("span", {}, c)));

// newest first, a window at a time; a full window continues from its oldest drive. A date restarts the list from
// the end of that day, like connect's filter.
function drivesSection(dongleId) {
  const body = h("div", {class: "drives"});
  const more = h("button", {class: "button", onclick: () => loadMore()}, "Older drives");
  const foot = h("div", {class: "list-foot", hidden: true}, h("span"), more);  // shown with the first drives, so nothing shifts
  const status = h("p", {class: "muted small"});
  const before = h("input", {type: "date", class: "field", title: "Drives up to this day", onchange: () => {
    end = top = before.value ? new Date(`${before.value}T00:00`).getTime() + DAY : firstWindowEnd();
    body.replaceChildren();
    seen.clear();
    loadMore(++gen);
  }});
  let end = firstWindowEnd(), top = end, gen = 0;
  const seen = new Set();

  async function loadMore(g = gen) {
    more.disabled = true;
    if (!body.children.length) status.textContent = "Loading…";
    const start = end - WINDOW;
    const routes = await api.routes(dongleId, start, end, LIMIT).catch(() => []);
    if (g !== gen) return;
    for (const r of routes) {
      if (seen.has(r.fullname)) continue;
      seen.add(r.fullname);
      api.route.prime(r, r.fullname);
      body.append(driveRow(r));
    }
    end = routes.length === LIMIT ? Math.min(...routes.map((r) => r.start_time_utc_millis)) : start;
    more.disabled = false;
    foot.hidden = false;
    if (!routes.length && !body.children.length && top - end < 180 * DAY) return loadMore(g);
    const since = new Date(end).toLocaleDateString();
    status.textContent = body.children.length ? `${body.children.length} drives since ${since}` : `No drives since ${since}`;
  }
  loadMore();
  return h("div", {},
    h("div", {class: "list-foot"}, status, h("label", {class: "muted small"}, "drives before ", before)),
    drivesHead(), body, foot);
}

function preservedSection(dongleId) {
  const el = h("div", {}, h("p", {class: "muted"}, "Loading…"));
  api.preservedRoutes(dongleId).then((routes) => {
    for (const r of routes) {
      r.start_time_utc_millis ??= Date.parse(r.start_time + "Z");
      r.end_time_utc_millis ??= Date.parse(r.end_time + "Z");
    }
    el.replaceChildren(...routes.length ? [drivesHead(), h("div", {class: "drives"}, routes.map(driveRow))] : [h("p", {class: "muted"}, "No preserved drives.")]);
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

// going back to a device page (from a drive, say) shows it as it was: the drives loaded, the open tab, the scroll. Any
// other way there builds it afresh, and a kept page is dropped a few minutes after it was built, or on signing in or out.
// Pages leave on a navigation, which says where they were scrolled.
const KEEP = 5 * 60e3;
const kept = new Map();  // dongle id → {el, tab, select, scroll, until}
let traverse = false;  // the navigation under way goes back or forward through history
auth.on(() => kept.clear());
globalThis.navigation?.addEventListener("navigate", (e) => {
  traverse = e.navigationType === "traverse";
  for (const k of kept.values()) if (k.el.isConnected) k.scroll = k.el.parentElement.scrollTop;
});

function restore(k) {
  const tab = new URLSearchParams(location.search).get("tab") || "drives";
  const scroll = tab === k.tab ? k.scroll : 0;
  k.select(tab);
  // before the first frame that shows it
  requestAnimationFrame(() => { if (k.el.isConnected) k.el.parentElement.scrollTop = scroll; });
  return k.el;
}

export async function devicePage(dongleId, devices) {
  for (const [id, k] of kept) if (k.until < Date.now()) kept.delete(id);
  const back = traverse;
  traverse = false;  // a render without a navigation (a retry, the device list changing) builds afresh
  if (back && kept.has(dongleId)) return restore(kept.get(dongleId));
  const device = (devices || []).find((d) => d.dongle_id === dongleId) || await api.device(dongleId);
  // the account lost the device (unpaired, or access removed)
  const forget = () => {
    kept.delete(dongleId);
    api.device.forget(dongleId);
    devicesChanged();
  };
  const tabs = {
    drives: ["Drives", () => drivesSection(dongleId)],
    preserved: ["Preserved", () => preservedSection(dongleId)],
    uploads: ["Uploads", () => deviceUploads(device)],
    users: ["Users", () => usersSection(device, forget)],
    bootlogs: ["Boot logs", () => logsSection(dongleId, "boot")],
    crashlogs: ["Crash logs", () => logsSection(dongleId, "crash")],
    settings: ["Settings", () => settingsSection(device, () => { forget(); go("/"); })],
  };
  const built = {};
  const keep = {tab: null, scroll: 0};
  const body = h("div", {class: "tab-body", role: "tabpanel"});
  const bar = h("nav", {class: "page-tabs", role: "tablist"});
  const select = (key) => {
    if (!tabs[key]) key = "drives";
    keep.tab = key;
    built[key] ||= tabs[key][1]();
    body.replaceChildren(built[key]);
    built[key].resume?.();
    for (const b of bar.children) {
      b.classList.toggle("on", b.dataset.tab === key);
      b.setAttribute("aria-selected", b.dataset.tab === key);
    }
    const url = new URL(location.href);
    if (key === "drives") url.searchParams.delete("tab"); else url.searchParams.set("tab", key);
    history.replaceState(history.state, "", url);
  };
  bar.append(...Object.entries(tabs).map(([key, [label]]) => h("button", {class: "tab", role: "tab", "data-tab": key, onclick: () => select(key)}, label)));
  select(new URLSearchParams(location.search).get("tab"));

  const online = isOnline(device);
  const el = h("div", {class: "page device-page"},
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
  kept.set(dongleId, Object.assign(keep, {el, select, until: Date.now() + KEEP}));
  return el;
}
