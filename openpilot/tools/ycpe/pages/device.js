// the device page: header with location map, then tabs for drives, preserved drives, uploads, users, logs and settings
import {api, auth, engagedFraction} from "../api.js";
import {h, fmt} from "../dom.js";
import {icon} from "../icons.js";
import {devicesChanged, go, isOnline, link, onlineMark} from "../app.js";
import {perf} from "../perf.js";
import {reverseLookup} from "../geocode.js";
import {deviceMap} from "../panes/map.js";
import {strip} from "../panes/strip.js";
import {deviceName} from "./home.js";
import {usersSection} from "./users.js";
import {logsSection} from "./logs.js";
import {deviceUploads} from "./files.js";
import {settingsSection} from "./settings.js";

const DAY = 86400000, HOUR = 3600000;
const WINDOW = 14 * DAY, LIMIT = 50, QUIET = 6;

// the first window ends at the next hour boundary so prefetch and page share a cache key
const firstWindowEnd = () => Math.ceil(Date.now() / HOUR) * HOUR;

export function prefetchDevice(dongleId) {
  const end = firstWindowEnd();
  api.routes(dongleId, end - WINDOW, end, LIMIT).catch(() => {});
  api.deviceLocation(dongleId).catch(() => {});
}

const duration = (r) => r.end_time_utc_millis - r.start_time_utc_millis;

// a drive's strip (panes/strip.js), filled in when its row nears the viewport. One observer for every strip, so
// filling one never reads layout.
const fillers = new WeakMap();
const queue = [];
let draining = null;
const nearby = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    nearby.unobserve(e.target);
    queue.push(fillers.get(e.target));
  }
  // strips until 2 ms of the frame's budget are left, so a fast scroll through hundreds of drives stays smooth
  draining ??= perf.tick(() => {
    while (queue.length) {
      queue.shift()();
      if (perf.left() <= 2) break;
    }
    if (!queue.length) { draining(); draining = null; }
  }, "drives");
}, {rootMargin: "400px"});

function driveStrip(r, onSpans, onVisible) {
  const s = strip(r, duration(r) / 1000);
  fillers.set(s.el, () => {
    onVisible();
    s.fill().then(onSpans, () => {});
  });
  nearby.observe(s.el);
  return s.el;
}

// the neighborhood, and under it (on a phone's card) the city
function place(el, lng, lat) {
  reverseLookup(lng, lat).then((loc) => {
    if (!loc) return;
    el.replaceChildren(h("span", {}, loc.place || loc.details), loc.place && h("span", {class: "dr-detail"}, loc.details));
    el.title = [loc.place, loc.details].filter(Boolean).join(", ");
  });
}

const HM = new Intl.DateTimeFormat(undefined, {hour: "2-digit", minute: "2-digit", hour12: false});
const DAY_FMT = new Intl.DateTimeFormat(undefined, {weekday: "short", month: "short", day: "numeric"});
const OLD_DAY_FMT = new Intl.DateTimeFormat(undefined, {weekday: "short", month: "short", day: "numeric", year: "numeric"});

// a row of the drives table, or on a phone (pages.css) a card as connect's: when and how long over where from and to,
// the strip along the bottom
function driveRow(r) {
  const path = `/${r.fullname.replace("|", "/")}`;
  const start = new Date(r.start_time_utc_millis);
  const engaged = h("span", {class: "num dr-engaged"});
  const from = h("span", {class: "dr-place dr-from"}), to = h("span", {class: "dr-place dr-to"});
  return h("a", {class: "drive-row", href: path, onclick: link(path)},
    h("span", {class: "dr-when"}, h("b", {}, (start.getFullYear() === new Date().getFullYear() ? DAY_FMT : OLD_DAY_FMT).format(start)), " ",
      h("span", {class: "dr-hours muted"}, HM.format(start), h("span", {class: "dr-dash"}), HM.format(r.end_time_utc_millis))),
    h("span", {class: "num dr-dur"}, fmt.minutes(duration(r))),
    h("span", {class: "num dr-dist"}, fmt.miles(r.distance ?? 0)),
    engaged,
    h("span", {class: "dr-places"}, from, h("span", {class: "dr-arrow muted"}, " → "), to),
    driveStrip(r, (spans) => { engaged.textContent = `${Math.round(100 * engagedFraction(spans, duration(r)))}%`; },
      () => { place(from, r.start_lng, r.start_lat); place(to, r.end_lng, r.end_lat); }),
    h("span", {class: "muted dr-version"}, r.version || ""));
}

const drivesHead = () => h("div", {class: "drive-row head"}, ["drive", "time", "distance", "engaged", "from → to", "timeline", "version"].map((c) => h("span", {}, c)));

// newest first, a window at a time; a full window continues from its oldest drive. The next window loads as the end
// of the list scrolls near, until QUIET windows in a row had none (then the button asks for more). A date restarts the
// list from the end of that day, like connect's filter.
function drivesSection(dongleId) {
  const body = h("div", {class: "drives"});
  const more = h("button", {class: "button more", onclick: () => loadMore()}, "Older drives");
  const foot = h("div", {class: "list-foot", hidden: true}, h("span"), more);  // shown with the first drives, so nothing shifts
  const status = h("p", {class: "muted small"});
  const before = h("input", {type: "date", class: "field", title: "Drives up to this day", onchange: () => {
    end = top = before.value ? new Date(`${before.value}T00:00`).getTime() + DAY : firstWindowEnd();
    quiet = 0;
    body.replaceChildren();
    seen.clear();
    loadMore(++gen);
  }});
  let end = firstWindowEnd(), top = end, gen = 0, quiet = 0;
  const seen = new Set();
  const near = new IntersectionObserver(([e]) => { if (e.isIntersecting && !more.disabled && quiet < QUIET) loadMore(); }, {rootMargin: "800px 0px"});
  near.observe(more);

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
    quiet = routes.length ? 0 : quiet + 1;
    more.disabled = false;
    foot.hidden = false;
    // observed afresh, it says whether the end is still near
    near.unobserve(more);
    near.observe(more);
    if (!routes.length && !body.children.length && top - end < 180 * DAY) return loadMore(g);
    const since = new Date(end).toLocaleDateString();
    status.textContent = body.children.length ? `${body.children.length} drives since ${since}` : `No drives since ${since}`;
  }
  loadMore();
  // a phone shows connect's Filter button for the date instead (pages.css)
  const filter = h("button", {class: "button filter", onclick: () => before.showPicker()}, icon("filter", 20), "Filter");
  return h("div", {},
    h("div", {class: "list-foot"}, status, h("label", {class: "muted small"}, "drives before ", before), filter),
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

// a phone shows only the totals, their numbers bold as connect's (pages.css)
function stats(dongleId) {
  const el = h("span", {class: "stats muted"});
  const stat = (n, unit) => h("span", {}, h("b", {}, n), ` ${unit}`);
  api.deviceStats(dongleId).then(({all, week}) => {
    el.replaceChildren(stat(all.routes, "drives"), " · ", stat(Math.round(all.distance), "mi"), " · ", stat(Math.round(all.minutes / 60), "h"),
      h("span", {class: "stats-week"}, ` — this week ${week.routes} drives · ${Math.round(week.distance)} mi`));
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
  const body = h("div", {class: "tab-body", role: "tabpanel", id: "device-tabpanel"});
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
    body.setAttribute("aria-labelledby", `device-tab-${key}`);
    // a phone's strip scrolls: bring the tab into it, once the page is shown
    const reveal = () => bar.querySelector(".on").scrollIntoView({inline: "nearest", block: "nearest"});
    if (bar.isConnected) reveal(); else requestAnimationFrame(reveal);
    const url = new URL(location.href);
    if (key === "drives") url.searchParams.delete("tab"); else url.searchParams.set("tab", key);
    history.replaceState(history.state, "", url);
  };
  bar.append(...Object.entries(tabs).map(([key, [label]]) => h("button", {class: "tab", role: "tab", id: `device-tab-${key}`,
    "aria-controls": "device-tabpanel", "data-tab": key, onclick: () => select(key)}, label)));
  select(new URLSearchParams(location.search).get("tab"));

  const online = isOnline(device);
  const el = h("div", {class: "page device-page"},
    h("div", {class: "device-head"},
      h("div", {class: "device-info"},
        h("h2", {}, onlineMark(device), deviceName(device), onlineMark(device, "pill")),
        h("div", {class: "kv"},
          h("span", {class: "mono"}, dongleId),
          device.device_type && h("span", {}, device.device_type),
          device.openpilot_version && h("span", {}, `openpilot ${device.openpilot_version}`),
          onlineMark(device, "seen"),
          online && h("a", {href: `/${dongleId}/live`, onclick: link(`/${dongleId}/live`), title: "Stream its cameras and messages"}, "live")),
        stats(dongleId)),
      deviceMap(api.deviceLocation(dongleId).catch(() => null))),
    bar, body);
  kept.set(dongleId, Object.assign(keep, {el, select, until: Date.now() + KEEP}));
  return el;
}
