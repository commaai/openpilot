// the device page: header with location map, then tabs for drives, preserved drives, uploads, users, logs and settings
import {api, auth, engagedFraction, versionAtLeast} from "../api.js";
import {h, fmt} from "../dom.js";
import {icon} from "../icons.js";
import {devicesChanged, go, isOnline, link, onlineMark, toast} from "../app.js";
import {perf} from "../perf.js";
import {reverseLookup} from "../geocode.js";
import {parseRemote} from "../data/github.js";
import {deviceMap} from "../panes/map.js";
import {strip} from "../panes/strip.js";
import {canManage, deviceName} from "./home.js";
import {usersSection} from "./users.js";
import {logsSection} from "./logs.js";
import {deviceUploads} from "./files.js";
import {settingsSection} from "./settings.js";

const DAY = 86400000, HOUR = 3600000, LIMIT = 50;
const DETAILS_KEY = "ycpe_drive_details";

// the first page ends at the next hour boundary so prefetch and page share a cache key
const firstPageEnd = () => Math.ceil(Date.now() / HOUR) * HOUR;

export function prefetchDevice(dongleId) {
  api.routes(dongleId, undefined, firstPageEnd(), LIMIT).catch(() => {});
  api.deviceLocation(dongleId).catch(() => {});
}

const duration = (r) => r.end_time_utc_millis - r.start_time_utc_millis;
// a preserved drive's times come as strings
function timed(r) {
  r.start_time_utc_millis ??= Date.parse(r.start_time + "Z");
  r.end_time_utc_millis ??= Date.parse(r.end_time + "Z");
  return r;
}

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
// the strip along the bottom. Under it, shown with the list's details, what useradmin lists of
// it
function driveRow(r, owner) {
  const path = `/${r.fullname.replace("|", "/")}`;
  const start = new Date(r.start_time_utc_millis);
  const engaged = h("span", {class: "num dr-engaged"});
  const from = h("span", {class: "dr-place dr-from"}), to = h("span", {class: "dr-place dr-to"});
  return h("a", {class: "drive-row", href: path, onclick: link(path)},
    h("span", {class: "dr-when"}, h("b", {}, (start.getFullYear() === new Date().getFullYear() ? DAY_FMT : OLD_DAY_FMT).format(start)), " ",
      h("span", {class: "dr-hours muted"}, HM.format(start), h("span", {class: "dr-dash"}), HM.format(r.end_time_utc_millis))),
    h("span", {class: "num dr-dur"}, fmt.minutes(duration(r))),
    h("span", {class: "num dr-dist"}, fmt.distance(r.distance ?? 0)),
    engaged,
    h("span", {class: "dr-places"}, from, h("span", {class: "dr-arrow muted"}, " → "), to),
    driveStrip(r, (spans) => { engaged.textContent = `${Math.round(100 * engagedFraction(spans, duration(r)))}%`; },
      () => { place(from, r.start_lng, r.start_lat); place(to, r.end_lng, r.end_lat); }),
    h("span", {class: "muted dr-version"}, r.version || ""),
    driveDetails(r, owner));
}

const UPLOADED = new Intl.DateTimeFormat(undefined, {month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false});

// the car, where its openpilot came from (comma's own in green, as useradmin marks it), how many segments, when it was
// uploaded, the route's name, and whether it's preserved
function driveDetails(r, owner) {
  const repo = parseRemote(r.git_remote) || r.git_remote;
  const preserved = h("span", {class: "dr-preserve", "data-preserve": r.fullname, "data-owner": owner ? "1" : ""});
  markPreserved(preserved, changed.get(r.fullname) ?? !!r.is_preserved);
  return h("span", {class: "dr-more muted"},
    r.platform && h("span", {}, r.platform),
    repo && h("span", {}, h("span", {class: repo === "commaai/openpilot" ? "official" : ""}, repo), r.git_branch && ` ${r.git_branch}`),
    r.maxqlog != null && h("span", {}, `${r.maxqlog + 1} segments`),
    r.create_time && h("span", {}, `uploaded ${UPLOADED.format(r.create_time * 1000)}`),
    h("span", {class: "mono", title: r.fullname}, r.fullname.split("|")[1]),
    preserved);
}

// a preserved drive outlives the device's retention. Only its owner can preserve, and only so many (useradmin's 403).
// What was changed here holds over the cached lists' records: route name → preserved
const changed = new Map();
function markPreserved(el, on) {
  const toggle = el.dataset.owner && h("button", {class: "link", onclick: (e) => {
    e.preventDefault();  // not the row's link
    e.stopPropagation();
    preserve(el.dataset.preserve, !on, e.currentTarget);
  }}, on ? "Unpreserve" : "Preserve");
  el.replaceChildren(on ? h("span", {class: "tag"}, "preserved") : "", toggle || "");
}

async function preserve(name, on, button) {
  button.disabled = true;
  try {
    await api.setRoutePreserved(name, on);
  } catch (err) {
    button.disabled = false;
    return toast(err.status !== 403 ? `Couldn't ${on ? "preserve" : "unpreserve"} the drive: ${err.message}`
      : on ? "Maximum number of drives have been preserved. Unpreserve a drive and try again."
      : "Only whoever preserved this drive can unpreserve it.");
  }
  api.route.forget(name);  // the drive view asks again
  changed.set(name, on);
  for (const el of document.querySelectorAll(`[data-preserve="${name}"]`)) markPreserved(el, on);
}

const drivesHead = () => h("div", {class: "drive-row head"}, ["drive", "time", "distance", "engaged", "from → to", "timeline", "version"].map((c) => h("span", {}, c)));

// a date input's value for a time
const isoDay = (t) => { const d = new Date(t); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; };
const DATE = new Intl.DateTimeFormat(undefined, {month: "short", day: "numeric", year: "numeric"});
const nDrives = (n) => `${n} drive${n === 1 ? "" : "s"}`;

// newest first, LIMIT at a time: each page is the drives that started before the last one's oldest, and the next loads
// as the end of the list scrolls near. A date range (connect's filter: days within the last year) starts the list over
// from the end of its last day, down to the start of its first.
function drivesSection(device) {
  const dongleId = device.dongle_id;
  const body = h("div", {class: "drives"});
  const more = h("button", {class: "button more", onclick: () => loadMore()}, "Older drives");
  const foot = h("div", {class: "list-foot", hidden: true}, h("span"), more);  // shown with the first drives, so nothing shifts
  const status = h("p", {class: "muted small"});
  const today = isoDay(Date.now()), yearAgo = isoDay(Date.now() - 365 * DAY);
  const from = h("input", {type: "date", class: "field", min: yearAgo, max: today, "aria-label": "From", onchange: () => {
    to.min = from.value || yearAgo;
    if (to.value && to.value < from.value) to.value = from.value;
    restart();
  }});
  const to = h("input", {type: "date", class: "field", min: yearAgo, max: today, "aria-label": "To", onchange: () => {
    if (to.value && to.value < from.value) to.value = from.value;
    restart();
  }});
  const clear = h("button", {class: "icon-btn", title: "All drives", "aria-label": "Show all drives", hidden: true, onclick: () => {
    from.value = to.value = "";
    to.min = yearAgo;
    restart();
  }}, icon("close", 14));
  const details = h("input", {type: "checkbox", checked: localStorage.getItem(DETAILS_KEY) === "1", onchange: () => {
    localStorage.setItem(DETAILS_KEY, details.checked ? "1" : "");
    body.classList.toggle("details", details.checked);
  }});
  body.classList.toggle("details", details.checked);
  // a phone shows them under connect's Filter button (pages.css)
  const opts = h("span", {class: "drive-opts muted small"}, h("label", {}, "from", from), h("label", {}, "to", to), clear, h("label", {}, details, "details"));
  const filter = h("button", {class: "button filter", "aria-expanded": "false", onclick: () => filter.setAttribute("aria-expanded", opts.classList.toggle("open"))},
    icon("filter", 20), "Filter");
  let range = null, end = firstPageEnd(), gen = 0, done = false;
  const seen = new Set();
  const near = new IntersectionObserver(([e]) => { if (e.isIntersecting && !more.disabled && !done) loadMore(); }, {rootMargin: "800px 0px"});
  near.observe(more);

  function restart() {
    range = from.value || to.value ? [from.value ? new Date(`${from.value}T00:00`).getTime() : undefined,
      to.value ? new Date(`${to.value}T23:59:59.999`).getTime() : firstPageEnd()] : null;
    end = range ? range[1] : firstPageEnd();
    clear.hidden = !range;
    done = false;
    body.replaceChildren();
    seen.clear();
    loadMore(++gen);
  }

  async function loadMore(g = gen) {
    more.disabled = true;
    if (!body.children.length) status.textContent = "Loading…";
    let routes, owner;
    try {
      [routes, owner] = await Promise.all([api.routes(dongleId, range?.[0], end, LIMIT), canManage(device)]);
    } catch (err) {
      if (g !== gen) return;
      status.textContent = `Couldn't load drives: ${err.message}`;
      more.disabled = foot.hidden = false;  // its button tries again
      return;
    }
    if (g !== gen) return;
    const fresh = routes.filter((r) => !seen.has(r.fullname));
    for (const r of fresh) {
      seen.add(r.fullname);
      api.route.prime(r, r.fullname);
      body.append(driveRow(r, owner));
    }
    done = routes.length < LIMIT || !fresh.length;
    if (fresh.length) end = Math.min(...fresh.map((r) => r.start_time_utc_millis));
    more.disabled = false;
    more.hidden = done;
    foot.hidden = false;
    // observed afresh, it says whether the end is still near
    near.unobserve(more);
    near.observe(more);
    const n = body.children.length;
    const span = !range ? "" : range[0] ? ` from ${DATE.format(range[0])} to ${DATE.format(range[1])}` : ` up to ${DATE.format(range[1])}`;
    status.textContent = !n ? `No drives${span}` : done ? `${nDrives(n)}${span}` : `${nDrives(n)} since ${DATE.format(end)}`;
  }
  loadMore();
  return h("div", {},
    h("div", {class: "list-foot"}, status, opts, filter),
    drivesHead(), body, foot);
}

// with their details, as useradmin lists them; asked for again each time the tab is shown, as drives come and go
function preservedSection(device) {
  const el = h("div", {}, h("p", {class: "muted"}, "Loading…"));
  const load = () => Promise.all([api.preservedRoutes(device.dongle_id), canManage(device)]).then(([routes, owner]) => {
    el.replaceChildren(...routes.length ? [drivesHead(), h("div", {class: "drives details"}, routes.map((r) => driveRow(timed({...r, is_preserved: true}), owner)))]
      : [h("p", {class: "muted"}, "No preserved drives.")]);
  }).catch((err) => el.replaceChildren(h("p", {class: "error"}, err.message)));
  el.resume = load;
  return el;
}

// a phone shows only the totals, their numbers bold as connect's (pages.css)
function stats(dongleId) {
  const el = h("span", {class: "stats muted"});
  const stat = (n, unit) => h("span", {}, h("b", {}, n), ` ${unit}`);
  api.deviceStats(dongleId).then(({all, week}) => {
    const [dist, unit] = fmt.distanceIn(all.distance), [weekDist] = fmt.distanceIn(week.distance);
    el.replaceChildren(stat(all.routes, "drives"), " · ", stat(Math.round(dist), unit), " · ", stat(Math.round(all.minutes / 60), "h"),
      h("span", {class: "stats-week"}, ` — this week ${week.routes} drives · ${Math.round(weekDist)} ${unit}`));
  }).catch(() => {});
  return el;
}

// the car's battery, as connect reads it from an online device (peripheralState is in millivolts); red under 11 V
function battery(device) {
  const el = h("span", {class: "battery", title: "Car battery", hidden: true});
  if (!isOnline(device)) return el;
  api.athena(device.dongle_id, "getMessage", {service: "peripheralState", timeout: 5000}).then((r) => r?.peripheralState?.voltage / 1000, () => 0).then((v) => {
    el.replaceChildren(icon("battery", 14), v ? `${v.toFixed(1)} V` : "N/A");
    el.className = `battery ${!v ? "" : v < 11 ? "low" : "ok"}`;
    el.hidden = false;
  });
  return el;
}

// devices older than 0.11.2 can't stream, so connect has them take a still from each camera instead
function snapshotButton(device, panel) {
  if (versionAtLeast(device, "0.11.2")) return null;
  const online = isOnline(device);
  const label = online ? "Take snapshot" : "Device offline";
  const b = h("button", {class: "icon-btn", title: label, "aria-label": label, disabled: !online, onclick: async () => {
    b.disabled = true;
    b.replaceChildren(h("span", {class: "spinner"}));
    try {
      let r = await api.athena(device.dongle_id, "takeSnapshot");
      if (!r?.jpegBack && !r?.jpegFront) r = await api.athena(device.dongle_id, "takeSnapshot");  // the cameras may be starting
      if (!r?.jpegBack && !r?.jpegFront) throw new Error("unable to fetch snapshot");
      panel.replaceChildren(snapshot(r.jpegBack, false), snapshot(r.jpegFront, true));
    } catch (err) {
      panel.replaceChildren(h("p", {class: "small error"}, err.message.includes("Device not registered") ? "device offline" : err.message));
    }
    panel.append(h("button", {class: "icon-btn", title: "Close", "aria-label": "Close the snapshot", onclick: () => { panel.hidden = true; }}, icon("close", 16)));
    panel.hidden = false;
    b.disabled = false;
    b.replaceChildren(icon("camera", 16));
  }}, icon("camera", 16));
  return b;
}

const snapshot = (jpeg, interior) => jpeg ? h("img", {src: `data:image/jpeg;base64,${jpeg}`, alt: interior ? "Interior camera" : "Road camera"})
  : h("div", {class: "snapshot-missing"}, h("b", {}, interior ? "Interior snapshot not available" : "Snapshot not available"),
    interior && h("p", {}, "Enable “Record and Upload Driver Camera” on your device for interior camera snapshots"));

// going back to a device page (from a drive, say) shows it as it was: the drives loaded, the open tab, the scroll. Any
// other way there builds it afresh, and a kept page is dropped a few minutes after it was built, or on signing in or out.
// Pages leave on a navigation, which says where they were scrolled.
const KEEP = 5 * 60e3;
const kept = new Map();  // dongle id → {el, tab, select, scroll, until, dispose}
let traverse = false;  // the navigation under way goes back or forward through history
// a page dropped, or replaced by one built afresh, lets go of its map
const drop = (id) => { kept.get(id)?.dispose(); kept.delete(id); };
auth.on(() => { for (const id of kept.keys()) drop(id); });
globalThis.navigation?.addEventListener("navigate", (e) => {
  traverse = e.navigationType === "traverse";
  for (const k of kept.values()) if (k.el.isConnected) k.scroll = k.el.parentElement.scrollTop;
});

function restore(k) {
  const tab = new URLSearchParams(location.search).get("tab") || "drives";
  const scroll = tab === k.tab ? k.scroll : 0;
  k.select(tab);
  // before the first frame that shows it
  perf.invalidate({name: "device-scroll", render: () => { if (k.el.isConnected) k.el.parentElement.scrollTop = scroll; }});
  return k.el;
}

export async function devicePage(dongleId, devices) {
  for (const [id, k] of kept) if (k.until < Date.now()) drop(id);
  const back = traverse;
  traverse = false;  // a render without a navigation (a retry, the device list changing) builds afresh
  if (back && kept.has(dongleId)) return restore(kept.get(dongleId));
  const device = (devices || []).find((d) => d.dongle_id === dongleId) || await api.device(dongleId);
  // the account lost the device (unpaired, or access removed)
  const forget = () => {
    drop(dongleId);
    api.device.forget(dongleId);
    devicesChanged();
  };
  const tabs = {
    drives: ["Drives", () => drivesSection(device)],
    preserved: ["Preserved", () => preservedSection(device)],
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
    if (bar.isConnected) reveal(); else perf.invalidate({name: "device-tabs", render: reveal});
    const url = new URL(location.href);
    if (key === "drives") url.searchParams.delete("tab"); else url.searchParams.set("tab", key);
    history.replaceState(history.state, "", url);
  };
  bar.append(...Object.entries(tabs).map(([key, [label]]) => h("button", {class: "tab", role: "tab", id: `device-tab-${key}`,
    "aria-controls": "device-tabpanel", "data-tab": key, onclick: () => select(key)}, label)));
  select(new URLSearchParams(location.search).get("tab"));

  const online = isOnline(device), map = deviceMap(device, api.deviceLocation(dongleId).catch(() => null));
  const snapshots = h("div", {class: "snapshot", hidden: true});
  const actions = h("span", {class: "device-actions"}, battery(device), snapshotButton(device, snapshots));
  import("../panes/clips.js").then(({deviceClips}) => actions.prepend(deviceClips(device)));  // loaded after the page
  const el = h("div", {class: "page device-page"},
    h("div", {class: "device-head"},
      h("div", {class: "device-info"},
        h("h2", {}, onlineMark(device), deviceName(device), onlineMark(device, "pill")),
        actions,
        h("div", {class: "kv"},
          h("span", {class: "mono"}, dongleId),
          device.device_type && h("span", {}, fmt.deviceType(device.device_type)),
          device.openpilot_version && h("span", {}, `openpilot ${device.openpilot_version}`),
          onlineMark(device, "seen"),
          online && h("a", {href: `/${dongleId}/live`, onclick: link(`/${dongleId}/live`), title: "Stream its cameras and messages"}, "live")),
        stats(dongleId)),
      map),
    snapshots, bar, body);
  drop(dongleId);
  kept.set(dongleId, Object.assign(keep, {el, select, until: Date.now() + KEEP, dispose: map.dispose}));
  return el;
}
