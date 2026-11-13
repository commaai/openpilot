// the navigator: developer mode's left sidebar, always there. One box on top (the omnibox: Enter opens whatever it
// names, typing filters the tree), then the tree: the account's devices (their totals, a filter of their drives by day
// or preserved, their panels: settings, users, uploads, logs …) → their drives, each a dense row with its strip; then
// what streams live, logs on this machine, and the routes opened lately. It follows the open place: the device and
// drive shown are marked and opened.
import {api, auth, engagedFraction, utcTimes} from "/api.js";
import {h, fmt} from "/dom.js";
import {icon} from "/icons.js";
import {daemon, needsDaemon} from "/daemon.js";
import {isOnline, link, onlineMark} from "/app.js";
import {onDeviceChange} from "/devices.js";
import {omnibox, recentRoutes, routePath} from "/pages/omnibox.js";
import {loginPath} from "/pages/login.js";
import {prefetchDevice, driveStrip, nearObserver, isPreserved, markPreserved, stats, isoDay} from "/pages/device.js";

const LIMIT = 50;
const DEMO_ROUTE = "5beb9b58bd12b691|0000010a--a51155e496";
const WHEN = new Intl.DateTimeFormat(undefined, {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false});
// a device's panels, as the device page's tabs (pages/device.js), then the pages of its own
const PANELS = [["settings", "settings"], ["users", "users"], ["uploads", "uploads"], ["logs", "logs"]];

// hooks: devices() and me() the account's; openDevice(dongle, tab) a device panel; openRoute(name, t) a drive (at a
// time); openFiles(files) logs from this machine; search(path) where the omnibox goes
export function sidebar(hooks) {
  const search = omnibox({go: hooks.search, me: hooks.me, onMiss: hooks.onMiss});
  const input = search.querySelector("input");
  const devices = h("div", {class: "nav-devices"});
  const live = h("div"), local = h("div"), recent = h("div");
  const tree = h("div", {class: "nav-tree"},
    section("Devices", devices), section("Live", live), section("Local", local), section("Recent", recent));
  const el = h("nav", {class: "navigator", "aria-label": "Navigator"}, h("div", {class: "nav-search"}, search), tree);
  const strips = nearObserver(tree);  // the drives' strips fill as they near the tree's view
  let groups = new Map();  // dongle id → its device group, kept across redraws (its drives, what's open)
  let shown = {dongle: null, route: null, path: ""};
  let drawnFor;  // the devices list the tree was drawn from

  // a device with its totals, tools and drives; the drives load as it's first opened, newest first a page at a time
  // (the filter's: from its last day, or the preserved ones all at once)
  function deviceGroup(device) {
    const dongle = device.dongle_id, mine = !!device.device_type;
    const kids = h("div", {class: "nav-kids"});
    const drives = h("div");
    const more = h("button", {class: "nav-row nav-more", hidden: true, onclick: () => load()}, "older drives…");
    const twisty = h("button", {class: "nav-twisty", "aria-label": "Show drives", onclick: (e) => { inside(e); toggle(); }}, icon("chevron", 12));
    const label = h("span", {class: "nav-label"});
    const row = h("a", {class: "nav-row nav-device", href: `/${dongle}`, onclick: link(`/${dongle}`), onpointerenter: () => mine && prefetchDevice(dongle)},
      twisty, mine ? onlineMark(device) : h("span", {class: "dot"}), label, device.alias !== dongle && h("span", {class: "nav-aside mono"}, dongle.slice(0, 6)));
    const named = () => {
      label.textContent = fmt.deviceName(device);
      row.dataset.text = `${label.textContent} ${dongle}`.toLowerCase();
    };
    named();
    // the account's devices have their totals (once opened) and filter, panels, live stream, prime and clips (the
    // scissors, once it says it cuts them)
    const which = mine ? driveFilter(() => restart()) : null;  // which drives
    const head = mine && h("div", {class: "nav-device-head"}, which.button);
    const tools = mine && h("div", {class: "nav-tools"},
      PANELS.map(([tab, label]) => h("button", {class: "nav-tool", onclick: () => hooks.openDevice(dongle, tab)}, label)),
      isOnline(device) && h("a", {class: "nav-tool", href: `/${dongle}/live`, onclick: link(`/${dongle}/live`)}, "live"),
      device.is_owner && h("a", {class: "nav-tool", href: `/connect/${dongle}/prime`, onclick: link(`/connect/${dongle}/prime`)}, device.prime ? "prime" : "get prime"));
    if (tools) import("/panes/clips.js").then(({deviceClips}) => tools.append(deviceClips(device)));
    kids.append(...[head, which?.el, tools, drives, more].filter(Boolean));
    const el = h("div", {class: "nav-group", role: "group"}, row, kids);
    const seen = new Map();  // route name → its drive group
    const firstEnd = () => which?.until() ?? Math.ceil(Date.now() / 3600000) * 3600000;
    let end = firstEnd(), gen = 0, loading = null, open = false;

    function toggle(to = !open) {
      open = to;
      el.classList.toggle("open", open);
      twisty.setAttribute("aria-expanded", open);
      if (open && head && !head.querySelector(".stats")) head.prepend(stats(dongle));
      if (open && !seen.size && !loading) load();
    }
    function add(r) {
      if (seen.has(r.fullname)) return seen.get(r.fullname);
      const d = driveRow(r);
      seen.set(r.fullname, d);
      drives.append(d.el);
      return d;
    }
    const page = () => which.preserved()
      ? api.preservedRoutes(dongle).then((routes) => routes.map((r) => utcTimes({...r, is_preserved: true})))
      : api.routes(dongle, which.since(), end, LIMIT);
    // the account's devices only (another's would 403)
    async function load() {
      if (!mine || !auth.isAuthenticated()) return;
      const g = gen;
      more.disabled = true;
      loading = page().then((routes) => {
        if (g !== gen) return;
        for (const r of routes) {
          api.route.prime(r, r.fullname);
          add(r);
        }
        if (routes.length) end = Math.min(...routes.map((r) => r.start_time_utc_millis));
        more.hidden = which.preserved() || routes.length < LIMIT;
        sortDrives();
      }, (e) => g === gen && drives.append(h("p", {class: "nav-note error"}, `Couldn't load drives: ${e.message}`)));
      await loading;
      loading = null;
      more.disabled = false;
      filter();
      mark();
    }
    // the drives anew, as the filter has them
    function restart() {
      gen++;
      seen.clear();
      drives.replaceChildren();
      more.hidden = true;
      end = firstEnd();
      load();
    }
    // a drive opened from elsewhere joins the list in its place
    const sortDrives = () => drives.append(...[...seen.values()].sort((a, b) => b.start - a.start).map((d) => d.el));
    return {el, row, dongle, known: mine, toggle, named, drives: seen, show(route) {
      toggle(true);
      if (route && !seen.has(route.fullname)) { add(route); sortDrives(); }
    }};
  }

  // a drive: when, how long, how far and how much engaged, from where to where on which openpilot (and whether it's
  // preserved), over its strip, filled as it nears the tree's view; its tooltip has the rest
  function driveRow(r) {
    const name = r.fullname, start = r.start_time_utc_millis, duration = r.end_time_utc_millis - start;
    const path = routePath(name);
    const engaged = h("span"), places = h("span", {class: "nav-places"}), tag = h("span", {"data-preserve": name});
    markPreserved(tag, isPreserved(r));
    const strip = driveStrip(r, (spans) => { engaged.textContent = ` · ${Math.round(100 * engagedFraction(spans, duration))}%`; }, (from, to) => {
      if (!from && !to) return;
      places.textContent = `${placeName(from)} → ${placeName(to)}`;
      row.dataset.text += ` ${places.textContent.toLowerCase()}`;
    }, false, strips);
    const about = [name, r.platform, r.git_remote && `${r.git_remote} ${r.git_branch || ""}`, r.maxqlog != null && `${r.maxqlog + 1} segments`];
    const row = h("a", {class: "nav-row nav-drive", href: path, onclick: link(path, () => hooks.openRoute(name)),
      "data-text": `${WHEN.format(start)} ${name} ${r.platform || ""} ${r.version || ""}`.toLowerCase(), title: about.filter(Boolean).join("\n")},
      h("span", {class: "nav-label"}, WHEN.format(start)),
      h("span", {class: "nav-aside"}, `${fmt.minutes(duration)} · ${fmt.distance(r.distance ?? 0)}`, engaged),
      places, h("span", {class: "nav-version"}, tag, r.version), strip);
    return {el: row, start};
  }

  // the account's devices, and the open drive's device when it isn't one of them (a public drive): that one by its id
  function drawDevices() {
    const list = hooks.devices(), old = groups;
    const stranger = shown.dongle && !list?.some((d) => d.dongle_id === shown.dongle);
    drawnFor = list;
    // a group drawn before the device's record was in (by its id alone) is drawn again from it
    const kept = (d) => old.get(d.dongle_id)?.known || !d.device_type ? old.get(d.dongle_id) : null;
    groups = new Map([...list || [], ...stranger ? [{dongle_id: shown.dongle, alias: shown.dongle}] : []]
      .map((d) => [d.dongle_id, kept(d) ?? deviceGroup(d)]));
    devices.replaceChildren(...[...groups.values()].map((g) => g.el));
    if (!auth.isAuthenticated()) devices.append(h("p", {class: "nav-note muted"}, h("a", {href: loginPath(), onclick: link(loginPath())}, "Sign in"), " to see your devices"));
    else devices.append(h("button", {class: "nav-row", onclick: () => import("/pages/pair.js").then((m) => m.openPair())},
      icon("add", 12), h("span", {class: "nav-label"}, "add device…")));
  }

  function drawLive() {
    const source = (path, on, label, detail, why) => why
      ? h("div", {class: "nav-row off", title: why}, h("span", {class: "dot"}), h("span", {class: "nav-label"}, label))
      : h("a", {class: `nav-row ${path.split("?")[0] === shown.live ? "on" : ""}`, href: path, onclick: link(path), title: detail, "data-text": label.toLowerCase()},
        h("span", {class: `dot ${on ? "online" : ""}`}), h("span", {class: "nav-label"}, label));
    live.replaceChildren(
      source("/live/local", daemon.present, "openpilot here", "a webrtcd on this computer: openpilot, the simulator or tools/replay",
        daemon.present ? null : needsDaemon("Streaming from this computer")),
      source("/live/panda", !!navigator.usb, "panda over USB", "CAN and panda health, receive only",
        navigator.usb ? null : "needs WebUSB, which this browser doesn't have"),
      source(`/live/replay?route=${encodeURIComponent(DEMO_ROUTE)}`, true, "demo replay", "a demo route's log played back as if it were live"),
      ...(hooks.devices() || []).filter(isOnline).map((d) => source(`/${d.dongle_id}/live`, true, fmt.deviceName(d), "stream through athena")));
  }

  function drawLocal() {
    const picker = (label, attrs) => {
      const file = h("input", {type: "file", multiple: true, hidden: true, ...attrs, onchange: () => hooks.openFiles([...file.files])});
      return h("label", {class: "nav-row", title: "rlog, qlog and camera files; or drop them anywhere"}, icon("upload", 12), h("span", {class: "nav-label"}, label), file);
    };
    const open = shown.loose && h("a", {class: "nav-row on", href: shown.path}, icon("check", 12), h("span", {class: "nav-label mono"}, shown.loose));
    local.replaceChildren(...[picker("open files…"), picker("open a folder…", {webkitdirectory: true}), open].filter(Boolean));
  }

  function drawRecent() {
    const alias = (dongle) => { const d = hooks.devices()?.find((d) => d.dongle_id === dongle); return d ? fmt.deviceName(d) : dongle.slice(0, 6); };
    recent.replaceChildren(...recentRoutes().slice(0, 8).map((r) => h("a", {class: "nav-row", href: routePath(r.name),
      onclick: link(routePath(r.name), () => hooks.openRoute(r.name)), "data-text": `${r.name} ${alias(r.name.split("|")[0])}`.toLowerCase(), title: r.name},
      h("span", {class: "nav-label"}, WHEN.format(r.start)), h("span", {class: "nav-aside"}, alias(r.name.split("|")[0])))));
  }

  // what's open marked (and opened to)
  function mark() {
    for (const a of devices.querySelectorAll(".nav-row.on")) a.classList.remove("on");
    const g = groups.get(shown.dongle);
    if (!g) return;
    if (!shown.route) return g.row.classList.add("on");
    const d = g.drives.get(shown.route.fullname);
    if (!d) return;
    d.el.classList.add("on");
  }

  // the typed text keeps the rows that have it: a device that has it keeps its drives, a drive keeps its device
  function filter() {
    const q = input.value.trim().toLowerCase();
    const has = (row) => !q || row.dataset.text?.includes(q);
    for (const g of groups.values()) {
      const all = has(g.row);
      for (const d of g.drives.values()) d.el.hidden = !all && !has(d.el);
      g.el.hidden = !all && [...g.drives.values()].every((d) => d.el.hidden);
    }
    for (const list of [live, recent]) {
      for (const row of list.children) row.hidden = !has(row);
      list.parentElement.hidden = [...list.children].every((row) => row.hidden);
    }
  }
  input.addEventListener("input", filter);
  // a device renamed (in either mode's settings) is named anew
  onDeviceChange((dongle) => {
    groups.get(dongle)?.named();
    filter();
  });

  function draw() {
    if (drawnFor !== hooks.devices() || shown.dongle && !groups.has(shown.dongle)) drawDevices();
    drawLive();
    drawLocal();
    drawRecent();
    filter();
  }
  draw();

  return {
    el,
    // the place shown: {dongle, route (its record), loose (local logs' name), live (a session's path), path}
    show(place) {
      const moved = place.route?.fullname !== shown.route?.fullname || place.dongle !== shown.dongle;
      shown = {...shown, ...place};
      draw();
      // opened to once per place (and again when the device's group is drawn anew, its record in)
      const g = groups.get(shown.dongle), key = shown.route?.fullname ?? "";
      if (g && g.revealed !== key) {
        g.revealed = key;
        g.show(shown.route);
      }
      mark();
      if (moved) el.querySelector(".nav-row.on")?.scrollIntoView({block: "nearest"});
    },
    focus: () => input.focus(),
  };
}

const section = (title, body) => h("section", {class: "nav-section"}, h("h4", {}, title), body);
const placeName = (loc) => loc?.place || loc?.details || "?";

// a device's drive filter, opened from its funnel: the days from and to, or only the preserved drives. onChange() as
// it changes
function driveFilter(onChange) {
  const today = isoDay(Date.now());
  const from = h("input", {type: "date", class: "field", max: today, "aria-label": "From", onchange: () => { to.min = from.value; changed(); }});
  const to = h("input", {type: "date", class: "field", max: today, "aria-label": "To", onchange: () => changed()});
  const preserved = h("input", {type: "checkbox", onchange: () => changed()});
  const clear = h("button", {class: "icon-btn", title: "All drives", "aria-label": "Show all drives", hidden: true, onclick: () => {
    from.value = to.value = to.min = "";
    preserved.checked = false;
    changed();
  }}, icon("close", 12));
  const el = h("div", {class: "nav-filter", hidden: true}, h("label", {}, "from", from), h("label", {}, "to", to), h("label", {}, preserved, "preserved"), clear);
  const button = h("button", {class: "icon-btn", title: "Filter the drives", "aria-label": "Filter the drives", "aria-expanded": "false", onclick: () => {
    el.hidden = !el.hidden;
    button.setAttribute("aria-expanded", !el.hidden);
  }}, icon("filter", 14));
  function changed() {
    clear.hidden = !from.value && !to.value && !preserved.checked;
    button.classList.toggle("on", !clear.hidden);
    onChange();
  }
  const day = (input, last) => input.value ? new Date(`${input.value}T${last ? "23:59:59.999" : "00:00"}`).getTime() : undefined;
  return {el, button, since: () => day(from), until: () => day(to, true), preserved: () => preserved.checked};
}

// a click on a row's twisty: not the row's link
function inside(e) {
  e.preventDefault();
  e.stopPropagation();
}
