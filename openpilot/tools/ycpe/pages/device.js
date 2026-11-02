// the device page: header with location map, then tabs for uploads, users, logs and settings; its drives are the
// navigator's (nav.js), whose rows' strips and preserved tags come from here. In user mode it's connect's dashboard: the
// map, the device's name and buttons, its totals and the drives as cards
import {api, auth, memo, versionAtLeast} from "../api.js";
import {h, fmt, isMetric} from "../dom.js";
import {icon} from "../icons.js";
import {go, isOnline, link, onlineMark} from "../app.js";
import {deviceRecord, deviceLeft, onDeviceChange} from "../devices.js";
import {perf} from "../perf.js";
import {reverseLookup} from "../geocode.js";
import {deviceMap} from "../panes/map.js";
import {strip, warm} from "../panes/strip.js";
import {deviceName} from "./home.js";
import {usersSection} from "./users.js";
import {logsSection} from "./logs.js";
import {deviceUploads} from "./files.js";
import {settingsSection} from "./settings.js";
import {cnModal, cnTip, modeIcon} from "./cnui.js";
import {userMode} from "./connect/mode.js";
import {cnIcon, drawPinCar} from "./connect/icons.js";

const DAY = 86400000, HOUR = 3600000, LIMIT = 50;
const READY_MS = 600;  // the longest a device page waits for its map, stats and drives before it's shown

// the first page ends at the next hour boundary so prefetch and page share a cache key
const firstPageEnd = () => Math.ceil(Date.now() / HOUR) * HOUR;

// about how many drives the page opens with in view, under the map and the device's head, a card's or a row's height
// each: those are filled before it's shown
const firstScreen = (user) => Math.ceil((innerHeight - 200) / (user ? 160 : 60));

// on a pointer over the device's link (or over the switch to the other mode: user false): the page's data, and once the
// pointer stays a moment, its first drives' strips and places (not those of every device a sweep down the list passes)
let pointed = null;
export function prefetchDevice(dongleId, user = userMode()) {
  const routes = api.routes(dongleId, user ? firstPageEnd() - 365 * DAY : undefined, firstPageEnd(), LIMIT);
  routes.catch(() => {});
  api.deviceLocation(dongleId).catch(() => {});
  api.deviceStats(dongleId).catch(() => {});
  pointed = dongleId;
  setTimeout(() => pointed === dongleId && routes.then((list) => {
    for (const r of list.slice(0, firstScreen(user))) {
      warm(r, duration(r) / 1000);
      reverseLookup(r.start_lng, r.start_lat);
      reverseLookup(r.end_lng, r.end_lat);
    }
  }, () => {}), 100);
}

const duration = (r) => r.end_time_utc_millis - r.start_time_utc_millis;

// a drive's strip (panes/strip.js) and places, filled in when its row nears the viewport. One observer for every
// strip, so filling one never reads layout. The sprites and events come from a host that serves six at a time, so the
// rows further down wait for the first screen's (up to OPENING_MS) rather than share it
const OPENING_MS = 2000;
const fillers = new WeakMap();
const queue = [];
let draining = null, opening = Promise.resolve();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// root: the list's own scroller, if it has one (the margin reaches past the viewport's edges only)
export const nearObserver = (root = null) => new IntersectionObserver((entries, observer) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    observer.unobserve(e.target);
    queue.push(fillers.get(e.target));
  }
  // strips until 2 ms of the frame's budget are left, so a fast scroll through hundreds of drives stays smooth
  opening.then(() => draining ??= perf.tick(() => {
    while (queue.length) {
      queue.shift()();
      if (perf.left() <= 2) break;
    }
    if (!queue.length) { draining(); draining = null; }
  }, "drives"));
}, {root, rootMargin: "400px"});
const nearby = nearObserver();

// onPlaces(from, to) puts the places in, onSpans(spans) whatever the row shows of the spans. A drive in the first
// screen fills at once, holding what it loaded until `hold`: the rest of the screen's thumbnails and places are in too,
// so they come in together (not the spans, whose events on a long drive take much longer). Its strip's loaded says
// when it has those, shown when they're in the row. observer: the list's (nearObserver), if it scrolls itself
export function driveStrip(r, onSpans, onPlaces, hold, observer = nearby) {
  const s = strip(r, duration(r) / 1000);
  const fill = () => {
    const places = Promise.all([reverseLookup(r.start_lng, r.start_lat), reverseLookup(r.end_lng, r.end_lat)]);
    const loaded = Promise.all([s.decoded(), places]);
    s.fill(hold).then(onSpans, () => {});
    const shown = Promise.all([places, hold && loaded.then(() => hold)]).then(([[from, to]]) => onPlaces(from, to)).catch(() => {});
    return {loaded, shown};
  };
  if (hold) return Object.assign(s.el, fill());
  fillers.set(s.el, fill);
  observer.observe(s.el);
  return s.el;
}

// as connect writes them, in English whatever the locale: "Sunday, Oct 18" (on a phone "Sun, Oct 18"), "10:00"
const CN_DAY = Object.fromEntries(["long", "short"].flatMap((weekday) => [[weekday, new Intl.DateTimeFormat("en-US", {weekday, month: "short", day: "numeric"})],
  [`${weekday}-year`, new Intl.DateTimeFormat("en-US", {weekday, month: "short", day: "numeric", year: "numeric"})]]));
const cnDay = (weekday, t) => CN_DAY[new Date(t).getFullYear() === new Date().getFullYear() ? weekday : `${weekday}-year`].format(t);
const CN_TIME = new Intl.DateTimeFormat("en-US", {hour: "2-digit", minute: "2-digit", hourCycle: "h23"});
const cnTime = (t) => CN_TIME.format(t);
const cnDuration = (ms) => { const hr = Math.floor(ms / HOUR) % 24, min = Math.floor(ms / 60000) % 60; return `${hr > 0 ? `${hr} hr ` : ""}${min} min`; };
const cnDistance = (mi) => { const [v, unit] = fmt.distanceIn(mi); return `${+v.toFixed(1)} ${unit}`; };
const lines = (cls, first, second) => h("div", {class: cls}, h("p", {class: "cn-b"}, first), h("p", {}, second));

// the drive view's modules and hls.js, as a pointer comes over a drive: the first one opened waits on nothing cold
function prefetchDrive() {
  import("../panes/connect.js").catch(() => {});
  import("../panes/camera.js").then((m) => m.prefetch(), () => {});
}

// connect's DriveListItem: when, how long and how far, from and to, an arrow (wider than a phone), and the drive's strip
// along the bottom. A first screen's card says when its strip's loaded and shown (driveStrip)
function driveCard(r, hold) {
  const path = `/connect/${r.fullname.replace("|", "/")}`;
  const from = lines("cn-c-from", "", ""), to = lines("cn-c-to", "", "");
  const day = h("p", {class: "cn-b"}, h("span", {class: "cn-wide"}, cnDay("long", r.start_time_utc_millis)),
    h("span", {class: "cn-narrow"}, cnDay("short", r.start_time_utc_millis)));
  const s = driveStrip(r, () => {}, (a, b) => { cnPlace(from, a); cnPlace(to, b); }, hold);
  return Object.assign(h("a", {class: "cn-drive", href: path, onclick: link(path), onpointerenter: prefetchDrive},
    h("div", {class: "cn-drive-head"},
      h("div", {class: "cn-drive-grid"},
        h("div", {class: "cn-c-date"}, day, h("p", {}, `${cnTime(r.start_time_utc_millis)} to ${cnTime(r.end_time_utc_millis)}`)),
        lines("cn-c-dur", cnDuration(duration(r)), cnDistance(r.distance ?? 0)),
        from, to,
        h("div", {class: "cn-c-arrow"}, cnIcon("arrow", 32, "100%")))),
    s), {loaded: s.loaded, shown: s.shown});
}
function cnPlace(el, loc) {
  if (!loc) return;
  el.firstChild.textContent = loc.place || "";
  el.lastChild.textContent = loc.details || "";
}

// a preserved drive outlives the device's retention; either mode's route menu preserves one. What was changed there
// holds over the cached lists' records: route name → preserved. A [data-preserve] element is its route's tag
const changed = new Map();
export const isPreserved = (r) => changed.get(r.fullname) ?? !!r.is_preserved;
export const markPreserved = (el, on) => el.replaceChildren(on ? h("span", {class: "tag"}, "preserved") : "");

// whether the drive is preserved as the device's preserved list says, which connect asks rather than trust the record
const listed = memo((name) => api.preservedRoutes(name.split("|")[0]).then((list) => Array.isArray(list) && list.some((r) => r.fullname === name)));
export const preservedNow = async (name) => changed.get(name) ?? listed(name);

// only its owner can, and only so many (useradmin's 403); route is the drive's record, which says so from then on
export async function setPreserved(route, on) {
  const name = route.fullname;
  if (!(await api.setRoutePreserved(name, on))?.success) throw new Error("unable to update");
  route.is_preserved = on;
  api.route.forget(name);  // the drive view asks again
  changed.set(name, on);
  for (const el of document.querySelectorAll(`[data-preserve="${name}"]`)) markPreserved(el, on);
}

// a date input's value for a time
export const isoDay = (t) => { const d = new Date(t); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; };

// connect's DriveList, user mode's: its totals and Filter over the cards, the drives of the last year unless filtered
// (connect's filter: days within the last year), newest first, LIMIT at a time: each page is the drives that started
// before the last one's oldest, and the next loads as the end of the list scrolls near. Then connect's word on what's
// left. Developer mode's drives are the navigator's (nav.js)
function drivesSection(device) {
  const dongleId = device.dongle_id;
  const body = h("div", {class: "cn-drives"});
  const foot = h("div", {class: "cn-more"});
  const status = h("p");
  const filter = h("button", {class: "cn-filter", onclick: () => timeSelect(range, (r) => restart(r))}, icon("filter", 20), h("span", {}, "Filter"));
  let range = defaultRange(), end = range[1], gen = 0, done = false, loading = false;
  const seen = new Set();
  const near = new IntersectionObserver(([e]) => { if (e.isIntersecting && !loading && !done) loadMore(); }, {rootMargin: "800px 0px"});
  near.observe(foot);

  function restart(r) {
    range = r;
    end = range[1];
    done = false;
    body.replaceChildren();
    seen.clear();
    loadMore(++gen);
  }

  async function loadMore(g = gen) {
    loading = true;
    if (!body.children.length) status.textContent = "Loading...";
    let routes;
    try {
      routes = await api.routes(dongleId, range[0], end, LIMIT);
    } catch (err) {
      if (g !== gen) return;
      status.textContent = `Couldn't load drives: ${err.message}`;
      loading = foot.hidden = false;
      return;
    }
    if (g !== gen) return;
    const fresh = routes.filter((r) => !seen.has(r.fullname));
    // the drives in the first screen come in together, once they're all loaded (or after OPENING_MS)
    const first = firstScreen(true), loaded = [], shown = [];
    let release;
    const hold = body.children.length < first && new Promise((r) => release = r);
    for (const r of fresh) {
      seen.add(r.fullname);
      api.route.prime(r, r.fullname);
      const now = body.children.length < first && hold;
      body.append(driveCard(r, now));
      if (now) {
        loaded.push(body.lastChild.loaded);
        shown.push(body.lastChild.shown);
      }
    }
    if (hold) opening = Promise.race([Promise.allSettled(loaded), sleep(OPENING_MS)]).then(release);
    done = routes.length < LIMIT || !fresh.length;
    if (fresh.length) end = Math.min(...fresh.map((r) => r.start_time_utc_millis));
    loading = false;
    foot.hidden = false;
    // observed afresh, it says whether the end is still near
    near.unobserve(foot);
    near.observe(foot);
    // connect says there are no more past five drives, once there aren't
    const n = body.children.length;
    status.textContent = !n ? "No routes found in selected time range." : !done ? "Loading..." : n > 5 ? "There are no more routes found in selected time range." : "";
    status.parentElement.className = !n ? "cn-empty" : "cn-end";
    // ready once the first screen's thumbnails and places are in
    return Promise.all(shown);
  }
  const ready = loadMore();
  return Object.assign(h("div", {class: "cn-drivelist"},
    h("div", {class: "cn-drivelist-head"}, device.shared ? h("div") : cnStats(dongleId), filter),
    body, foot, h("div", {class: "cn-empty"}, status)), {ready});
}

// connect's: from the start of the hour after this one, back a year
const defaultRange = () => [firstPageEnd() - 365 * DAY, firstPageEnd()];

// connect's TimeSelect: a start and an end day within the last year, kept on Save
async function timeSelect(range, onSave) {
  const [t0, t1] = range || defaultRange();
  const today = isoDay(Date.now()), yearAgo = isoDay(Date.now() - 365 * DAY);
  const start = h("input", {type: "date", min: yearAgo, max: today, value: isoDay(t0), onchange: () => {
    if (!start.value) return;
    end.min = start.value;
    if (end.value < start.value) end.value = start.value;
  }});
  const end = h("input", {type: "date", min: isoDay(t0), max: today, value: isoDay(t1), onchange: () => {
    if (end.value && end.value < start.value) end.value = start.value;
  }});
  const field = (label, input) => h("label", {class: "cn-ts-field"}, h("span", {}, label), input);
  const {el, paper} = cnModal("Filter drives", "cn-ts");
  paper.append(
    h("div", {class: "cn-ts-fields"}, field("Start date:", start), field("End date:", end)),
    h("hr"),
    h("div", {class: "cn-ts-buttons"},
      h("button", {class: "cnui-btn grey", onclick: () => el.close()}, "Cancel"),
      h("button", {class: "cnui-btn light", onclick: () => {
        el.close();
        onSave([new Date(`${start.value}T00:00`).getTime(), new Date(`${end.value}T23:59:59.999`).getTime()]);
      }}, "Save")));
}

// connect's totals: distance, drives and hours, the numbers bold
function cnStats(dongleId) {
  const el = h("div", {class: "cn-stats"});
  const stat = (n, unit) => h("div", {}, h("p", {class: "cn-b"}, n), h("p", {}, unit));
  el.ready = api.deviceStats(dongleId).then(({all}) => {
    el.replaceChildren(stat(Math.round(fmt.distanceIn(all.distance)[0]), isMetric() ? "kilometers" : "miles"),
      stat(all.routes, "drives"), stat(Math.round(all.minutes / 60), "hours"));
  }).catch(() => {});
  return el;
}

// the device's totals, on its page and over its drives in the navigator
export function stats(dongleId) {
  const el = h("span", {class: "stats muted"});
  const stat = (n, unit) => h("span", {}, `${n} ${unit}`);
  el.ready = api.deviceStats(dongleId).then(({all, week}) => {
    const [dist, unit] = fmt.distanceIn(all.distance), [weekDist] = fmt.distanceIn(week.distance);
    el.replaceChildren(stat(all.routes, "drives"), " · ", stat(Math.round(dist), unit), " · ", stat(Math.round(all.minutes / 60), "h"),
      h("span", {class: "stats-week"}, ` — this week ${week.routes} drives · ${Math.round(weekDist)} ${unit}`));
  }).catch(() => {});
  return el;
}

// the car's battery in volts, as connect reads it from an online device (peripheralState is in millivolts); 0 when
// it doesn't say
const carVolts = (device) => api.athena(device.dongle_id, "getMessage", {service: "peripheralState", timeout: 5000})
  .then((r) => r?.peripheralState?.voltage / 1000, () => 0);

// red under 11 V
function battery(device) {
  const el = h("span", {class: "battery", title: "Car battery", hidden: true});
  if (!isOnline(device)) return el;
  carVolts(device).then((v) => {
    el.replaceChildren(icon("battery", 14), v ? `${v.toFixed(1)} V` : "N/A");
    el.className = `battery ${!v ? "" : v < 11 ? "low" : "ok"}`;
    el.hidden = false;
  });
  return el;
}

// devices older than 0.11.2 can't stream, so connect has them take a still from each camera instead. User mode's is
// connect's pill
function snapshotButton(device, panel, user = false) {
  if (versionAtLeast(device, "0.11.2")) return null;
  const online = isOnline(device);
  const label = online ? "Take snapshot" : "Device offline";
  const glyph = () => modeIcon("camera", 16, "camera", 24);
  const b = h("button", {class: user ? "cn-pill" : "icon-btn", title: user ? null : label, "aria-label": label,
    disabled: !online, onclick: async () => {
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
    b.replaceChildren(glyph());
  }}, glyph());
  return user ? cnTip(b, "Take snapshot", "pill") : b;
}

// connect's DeviceInfo: the name (after commacare's badge), then clips, the livestream (a snapshot before 0.11.2) and
// the car's battery, or "device offline" with when it was last seen
function deviceInfo(device, snapshots) {
  const online = isOnline(device), live = `/${device.dongle_id}/live`;
  const buttons = h("div", {class: "cn-info-btns"},
    versionAtLeast(device, "0.11.2") ? cnTip(h("button", {class: "cn-pill", disabled: !online, "aria-label": "Livestream", onclick: link(live)}, cnIcon("livestream")), "Livestream", "pill")
      : snapshotButton(device, snapshots, true),
    cnBattery(device));
  // connect's scissors on the clips track's button, a pill as the others
  import("../panes/clips.js").then(({deviceClips}) => {
    const b = deviceClips(device);
    b.classList.replace("icon-btn", "cn-pill");
    cnTip(b, b.title, "pill");
    b.removeAttribute("title");
    b.replaceChildren(cnIcon("cut"));
    buttons.prepend(b);
  });
  const name = fmt.deviceName(device);
  return h("div", {class: "cn-info"}, h("div", {class: "cn-info-row"},
    h("div", {class: "cn-info-name"},
      device.commacare && h("img", {src: new URL("../icons/commacare.webp", import.meta.url), alt: "commacare", width: 22}),
      h("h2", {}, name.length > 16 ? `${name.slice(0, 16)}...` : name)),
    buttons));
}

const PING = new Intl.DateTimeFormat("en-US", {month: "short", day: "numeric", year: "numeric"});
const PING_TIME = new Intl.DateTimeFormat("en-US", {hour: "numeric", minute: "2-digit"});
function cnBattery(device) {
  if (!isOnline(device)) {
    const t = device.last_athena_ping * 1000;
    return cnTip(h("div", {class: "cn-battery off"}, h("p", {}, "device offline")),
      t ? `Last ping on ${PING.format(t)} at ${PING_TIME.format(t)}` : "no ping received from device", "pill");
  }
  const text = h("p", {class: "cn-volts"}, "N/A"), el = h("div", {class: "cn-battery"}, cnIcon("carBattery", 20), text);
  carVolts(device).then((v) => {
    if (!v) return;
    text.textContent = `${v.toFixed(1)}\u00a0V`;
    el.classList.add(v < 11 ? "low" : "ok");
  });
  return el;
}

const snapshot = (jpeg, interior) => jpeg ? h("img", {src: `data:image/jpeg;base64,${jpeg}`, alt: interior ? "Interior camera" : "Road camera"})
  : h("div", {class: "snapshot-missing"}, h("b", {}, interior ? "Interior snapshot not available" : "Snapshot not available"),
    interior && h("p", {}, "Enable “Record and Upload Driver Camera” on your device for interior camera snapshots"));

// the device's tabs: its uploads, users, boot and crash logs and settings (its drives are the navigator's), and in user
// mode first its drives, all connect's dashboard shows. The device page shows them under its head; the device pane
// (panes/device.js) beside a route. onSelect(key) is told of each tab shown; a key that names none is the first
export const DEVICE_TABS = {uploads: "Uploads", users: "Users", logs: "Logs", settings: "Settings"};

export function deviceTabs(device, onSelect = () => {}) {
  const dongleId = device.dongle_id, user = userMode();
  // the account lost the device (unpaired, or access removed)
  const forget = () => deviceLeft(dongleId);
  const sections = {
    ...user && {drives: () => drivesSection(device)},
    uploads: () => deviceUploads(device),
    users: () => usersSection(device, forget),
    logs: () => h("div", {}, logsSection(dongleId, "boot"), logsSection(dongleId, "crash")),
    settings: () => settingsSection(device, () => go("/")),
  };
  const built = {};
  const id = `device-tabpanel-${++tabsMade}`;
  const body = h("div", {class: "tab-body", role: "tabpanel", id});
  const bar = h("nav", {class: "page-tabs", role: "tablist"});
  const first = Object.keys(sections)[0];
  const select = (key) => {
    if (!sections[key]) key = first;
    built[key] ||= sections[key]();
    body.replaceChildren(built[key]);
    built[key].resume?.();
    for (const b of bar.children) {
      b.classList.toggle("on", b.dataset.tab === key);
      b.setAttribute("aria-selected", b.dataset.tab === key);
    }
    body.setAttribute("aria-labelledby", `${id}-${key}`);
    // a phone's strip scrolls: bring the tab into it, once the page is shown (user mode has no tabs: the gear opens settings)
    const reveal = () => bar.querySelector(".on").scrollIntoView({inline: "nearest", block: "nearest"});
    if (bar.isConnected) reveal(); else if (!user) perf.invalidate({name: "device-tabs", render: reveal});
    onSelect(key);
  };
  bar.append(...Object.keys(sections).map((key) => h("button", {class: "tab", role: "tab", id: `${id}-${key}`,
    "aria-controls": id, "data-tab": key, onclick: () => select(key)}, DEVICE_TABS[key] ?? "Drives")));
  return {bar, body, select, first};
}
let tabsMade = 0;

// going back to a device page (from a drive, say) shows it as it was: the open tab, the scroll (user mode: the drives). Any
// other way there builds it afresh, and a kept page is dropped a few minutes after it was built, or on signing in or out.
// Pages leave on a navigation, which says where they were scrolled.
const KEEP = 5 * 60e3;
const kept = new Map();  // dongle id → {el, tab, select, scroll, until, dispose}
let traverse = false;  // the navigation under way goes back or forward through history
// a page dropped, or replaced by one built afresh, lets go of its map
const drop = (id) => { kept.get(id)?.dispose(); kept.delete(id); };
auth.on(() => { for (const id of kept.keys()) drop(id); });
// a device that changed (renamed, gone) is built afresh next time, its kept page dropped then (it may still be shown)
onDeviceChange((id) => { if (kept.has(id)) kept.get(id).until = 0; });
globalThis.navigation?.addEventListener("navigate", (e) => {
  traverse = e.navigationType === "traverse";
  for (const k of kept.values()) if (k.el.isConnected) k.scroll = k.el.parentElement.scrollTop;
});

function restore(k) {
  const was = k.tab;
  k.select(new URLSearchParams(location.search).get("tab"));
  const scroll = k.tab === was ? k.scroll : 0;
  // before the first frame that shows it
  perf.invalidate({name: "device-scroll", render: () => { if (k.el.isConnected) k.el.parentElement.scrollTop = scroll; }});
  return k.el;
}

// shown once its map, stats and first drives are in (or after READY_MS), so the page it replaces stays until then
// rather than flashing an empty one
export async function devicePage(dongleId) {
  for (const [id, k] of kept) if (k.until < Date.now()) drop(id);
  const back = traverse;
  traverse = false;  // a render without a navigation (a retry, the device list changing) builds afresh
  if (back && kept.get(dongleId)?.user === userMode()) return restore(kept.get(dongleId));
  const device = await deviceRecord(dongleId);
  const user = userMode();
  const keep = {tab: null, scroll: 0};
  // the open tab rides in the URL's ?tab=
  const {bar, body, select, first} = deviceTabs(device, (key) => {
    keep.tab = key;
    const url = new URL(location.href);
    if (key === first) url.searchParams.delete("tab"); else url.searchParams.set("tab", key);
    history.replaceState(history.state, "", url);
  });
  select(new URLSearchParams(location.search).get("tab"));

  const online = isOnline(device), map = deviceMap(device, api.deviceLocation(dongleId).catch(() => null), user ? {connect: true, drawPin: drawPinCar} : undefined);
  const snapshots = h("div", {class: "snapshot", hidden: true});
  if (user) return keepPage(dongleId, keep, select, map, h("div", {class: "cn-dash"}, map, deviceInfo(device, snapshots), snapshots, body));
  const actions = h("span", {class: "device-actions"}, battery(device), snapshotButton(device, snapshots));
  import("../panes/clips.js").then(({deviceClips}) => actions.prepend(deviceClips(device)));  // loaded after the page
  const el = h("div", {class: "page device-page"},
    h("div", {class: "device-head"},
      h("div", {class: "device-info"},
        h("h2", {}, onlineMark(device), deviceName(device)),
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
  return keepPage(dongleId, keep, select, map, el);
}

async function keepPage(dongleId, keep, select, map, el) {
  drop(dongleId);
  kept.set(dongleId, Object.assign(keep, {el, select, user: userMode(), until: Date.now() + KEEP, dispose: map.dispose}));
  const ready = [map.ready, el.querySelector(".stats, .cn-stats")?.ready, el.querySelector(".tab-body")?.firstChild?.ready];
  await Promise.race([Promise.allSettled(ready), sleep(READY_MS)]);
  await new Promise(requestAnimationFrame);  // in the frame after the one its drives were built in, not sharing it
  return el;
}
