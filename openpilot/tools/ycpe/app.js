import {api, auth} from "/api.js";
import {daemon} from "/daemon.js";
import {localBadge} from "/local.js";
import {h} from "/dom.js";
import {logo, icon} from "/icons.js";
import {perf, hud} from "/perf.js";
import {theme} from "/theme.js";
import {createClock, RATES} from "/clock.js";
import {Layout, checkLayout} from "/layout.js";
import {PANES} from "/panes.js";
import {VIEWS} from "/views.js";
import {playbar} from "/playbar.js";
import {status, loadPill, footprint} from "/status.js";
import {Store} from "/data/store.js";
import {logSchema} from "/data/schemas.js";
import {routeRecord, routeKey, routeFromPath, routeLogs, localRoute, urlRoute, droppedFiles, watchRouteStart} from "/data/sources.js";
import {devicePage, prefetchDevice} from "/pages/device.js";
import {homePage} from "/pages/home.js";
import {loginPage, authPage, signInCard, loginPath} from "/pages/login.js";
import {omnibox, rememberRoute} from "/pages/omnibox.js";
import {localPage, pullCard} from "/pages/local.js";
import {create as routesPane} from "/panes/routes.js";
import {layoutPicker, loadLayout, mergeLayout, acceptLayoutFiles} from "/layouts/picker.js";

const DONGLE_RE = /^[0-9a-f]{16}$/i;

export function go(path, replace = false) {
  history[replace ? "replaceState" : "pushState"](null, "", path);
  render();
}

export function link(path) {
  return (e) => {
    if (e.metaKey || e.ctrlKey || e.button !== 0) return;
    e.preventDefault();
    go(path);
  };
}

export const isOnline = (d) => d.last_athena_ping && Date.now() / 1000 - d.last_athena_ping < 120;

// the omnibox (pages/omnibox.js); switching routes keeps the open view
function searchBox() {
  return omnibox({
    go: (path) => go(ws ? path.replace(/^(\/[0-9a-f]{16}\/[^/?]+(?:\/\d+\/\d+)?)(?=\?|$)/, `$1/${ws.view}`) : path),
    devices: () => state.devices,
    onMiss: () => toast("Enter a route name, a dongle id, or a connect, useradmin or cabana URL"),
  });
}

function toast(text) {
  const el = h("div", {class: "toast"}, text);
  document.body.append(el);
  setTimeout(() => el.remove(), 3000);
}

const state = {me: null, devices: null};
let ws = null;  // the open route workspace

// ---- top bar ----

// a popover under anchor; clicking the anchor again, clicking or focusing elsewhere or Escape closes it. Null when it
// was open. Focus moves into it (arrows step through a menu's rows) and back to the anchor when it closes.
function menu(anchor, build) {
  const open = document.querySelector(".popover");
  closePopover();
  if (open?._anchor === anchor) return null;
  const close = () => {
    if (!el.isConnected) return;
    const focused = el.contains(document.activeElement);
    el.remove();
    anchor.setAttribute("aria-expanded", "false");
    document.removeEventListener("pointerdown", outside);
    document.removeEventListener("focusin", outside);
    document.removeEventListener("keydown", escape);
    if (focused) anchor.focus();
  };
  const outside = (e) => { if (!el.contains(e.target) && !anchor.contains(e.target)) close(); };  // a click or focus
  const escape = (e) => { if (e.key === "Escape") close(); };
  const el = h("div", {class: "popover"}, build(close));
  el._anchor = anchor;
  el._close = close;
  const rows = () => [...el.querySelectorAll(".menu-list > :is(button, a)")];
  if (el.querySelector(".menu-list")) {
    el.setAttribute("role", "menu");
    for (const row of rows()) row.setAttribute("role", "menuitem");
  }
  el.addEventListener("keydown", (e) => {
    if (el.getAttribute("role") !== "menu" || (e.key !== "ArrowDown" && e.key !== "ArrowUp")) return;
    e.preventDefault();
    e.stopPropagation();
    const all = rows(), i = all.indexOf(document.activeElement);
    all[(i + (e.key === "ArrowDown" ? 1 : all.length - 1)) % all.length]?.focus();
  });
  anchor.setAttribute("aria-haspopup", "true");
  anchor.setAttribute("aria-expanded", "true");
  const r = anchor.getBoundingClientRect();
  el.style.left = `${Math.min(r.left, innerWidth - 340)}px`;
  el.style.top = `${r.bottom + 4}px`;
  document.body.append(el);
  setTimeout(() => {
    document.addEventListener("pointerdown", outside);
    document.addEventListener("focusin", outside);
    document.addEventListener("keydown", escape);
    el.querySelector("button, a[href], input, select")?.focus();  // once the caller has filled it
  });
  return el;
}

// the open popover, ours or a pane's (they share the class)
function closePopover() {
  const open = document.querySelector(".popover");
  if (open?._close) open._close(); else open?.remove();
}

const bar = {
  device: h("button", {class: "crumb", title: "Devices"}),
  route: h("button", {class: "crumb", title: "Routes on this device"}),
  views: h("nav", {class: "views"}),
  tabs: h("nav", {class: "tabs"}),
};

function topBar() {
  const search = searchBox();
  bar.tabs.addEventListener("wheel", (e) => {
    if (e.deltaX || !e.deltaY) return;
    e.preventDefault();
    bar.tabs.scrollLeft += e.deltaY;
    tabsScrolled = true;
  }, {passive: false});
  // the bar's other items settle after the tabs: keep the active tab in view, unless the user scrolled the strip
  new ResizeObserver(() => {
    const on = bar.tabs.querySelector(".tab.on")?.getBoundingClientRect(), r = bar.tabs.getBoundingClientRect();
    if (on && !tabsScrolled && (on.left < r.left || on.right > r.right)) showActiveTab();
  }).observe(bar.tabs);
  bar.device.addEventListener("click", () => menu(bar.device, (close) => h("div", {class: "menu-list"},
    (state.devices || []).map((d) => h("a", {class: "menu-row", href: `/${d.dongle_id}`, onclick: (e) => { close(); link(`/${d.dongle_id}`)(e); },
      onpointerenter: () => prefetchDevice(d.dongle_id)},
      h("span", {class: `dot ${isOnline(d) ? "online" : ""}`}), h("span", {}, d.alias || d.dongle_id), h("span", {class: "muted mono"}, d.dongle_id.slice(0, 6)))),
    !state.devices?.length && h("div", {class: "menu-row muted"}, auth.isAuthenticated() ? "No devices" : "Sign in to see your devices"))));
  bar.route.addEventListener("click", () => {
    if (!ws) return;
    const pop = menu(bar.route, () => h("div", {class: "route-pop"}));
    pop?.firstChild.append(routesPane({route: ws.route, openRoute: (name) => { pop._close(); ws.openRoute(name); }}).el);
  });
  const account = h("button", {class: "tbtn", title: "Account"}, icon("account", 16));
  account.addEventListener("click", () => menu(account, (close) => h("div", {class: "menu-list"},
    state.me && h("div", {class: "menu-row muted"}, state.me.username || state.me.email || "signed in"),
    auth.viaDaemon() && h("div", {class: "menu-row muted"}, "signed in via local daemon"),
    !auth.isAuthenticated() && daemon.hasJwt &&h("button", {class: "menu-row", onclick: () => { close(); auth.useDaemon(); }}, "Sign in via local daemon"),
    auth.isAuthenticated()
      ? h("button", {class: "menu-row", onclick: () => { close(); auth.signOut(); }}, "Sign out")
      : h("a", {class: "menu-row", href: loginPath(), onclick: (e) => { close(); link(loginPath())(e); }}, "Sign in"))));
  const themeBtn = h("button", {class: "tbtn", title: "Light/dark (T)", onclick: () => theme.toggle()}, icon("theme", 16));
  const helpBtn = h("button", {class: "tbtn", title: "Keyboard and mouse help (?)", onclick: help}, icon("help", 16));
  return h("header", {class: "topbar"},
    h("a", {class: "brand", href: "/", onclick: link("/"), title: "YCPE"}, logo(20), h("span", {}, "ycpe")),
    bar.device, h("span", {class: "sep"}, "›"), bar.route, search, bar.views, bar.tabs,
    h("span", {class: "grow"}), loadPill(), localBadge(), hud(footprint()), helpBtn, themeBtn, account);
}

function updateTop() {
  const dongle = ws?.dongle || location.pathname.split("/")[1];
  const device = state.devices?.find((d) => d.dongle_id === dongle);
  bar.device.textContent = device ? device.alias || dongle : DONGLE_RE.test(dongle || "") ? dongle : "devices";
  bar.route.hidden = bar.route.previousSibling.hidden = !ws;
  bar.views.hidden = bar.tabs.hidden = !ws;
  if (!ws) {  // the buttons' handlers would keep the closed workspace alive
    bar.views.replaceChildren();
    return bar.tabs.replaceChildren();
  }
  const start = new Date(ws.route.start_time_utc_millis);  // 0 for local files without a timestamp: show the name
  bar.route.textContent = ws.route.start_time_utc_millis
    ? `${start.toLocaleDateString(undefined, {month: "short", day: "numeric"})} ${start.toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit", hour12: false})}`
    : ws.route.fullname.split("|")[1];
  bar.route.title = `${ws.route.fullname} · ${[ws.route.platform, ws.route.version].filter(Boolean).join(" · ")}`;
  bar.views.replaceChildren(...Object.entries(VIEWS).map(([k, v], i) => h("button", {class: `view ${k === ws.view ? "on" : ""}`, title: `${v.title} (${i + 1})`, onclick: () => ws.setView(k)}, v.title)));
  const layout = ws.layout;
  if (!layout) return bar.tabs.replaceChildren();  // the view is still waiting on its files
  bar.tabs.replaceChildren(...layout.model.tabs.map((t, i) => h("button", {class: `tab ${i === layout.model.active ? "on" : ""}`,
    onclick: (e) => {  // in place, so a double-click lands on the same button
      if (i === layout.model.active) return;
      layout.selectTab(i);
      for (const b of bar.tabs.children) b.classList.toggle("on", b === e.currentTarget);
    },
    ondblclick: (e) => renameTab(e.currentTarget, i, t.name),
    oncontextmenu: (e) => { e.preventDefault(); tabMenu(e.currentTarget, i, t.name); },
    onauxclick: (e) => { if (e.button === 1) { layout.closeTab(i); updateTop(); } }}, t.name,
    layout.model.tabs.length > 1 && h("span", {class: "x", title: "Close tab", onclick: (e) => { e.stopPropagation(); layout.closeTab(i); updateTop(); }}, icon("close", 12)))),
  h("button", {class: "tab add", title: "New tab", onclick: () => { layout.addTab(); updateTop(); }}, "+"),
  h("button", {class: "tab add", title: "Open, import or export layouts", onclick: (e) => layoutMenu(e.currentTarget)}, "⋯"));
  // scrolled to on the next frame, when it lays out anyway, and only when the active tab or the tab set changed
  const shown = `${ws.view} ${layout.model.active} ${layout.model.tabs.map((t) => t.name).join("|")}`;
  if (shown !== tabsShown) {
    tabsShown = shown;
    tabsScrolled = false;
    requestAnimationFrame(showActiveTab);
  }
}

let tabsShown = "";        // the tabs and active tab the strip last scrolled to
let tabsScrolled = false;  // the user scrolled the strip since

// too many tabs scroll: show the buttons at the end if the active tab still fits, else the active tab
function showActiveTab() {
  if (bar.tabs.scrollWidth <= bar.tabs.clientWidth) return;
  bar.tabs.lastChild?.scrollIntoView({block: "nearest", inline: "nearest"});
  bar.tabs.querySelector(".tab.on")?.scrollIntoView({block: "nearest", inline: "nearest"});
}

function layoutMenu(anchor) {
  menu(anchor, (close) => layoutPicker({current: () => ws.layout.save(), apply: (m) => ws.applyLayout(m), close, name: ws.view}));
}

function tabMenu(tabEl, i, name) {
  const layout = ws.layout;
  const item = (label, fn) => h("button", {class: "menu-row", onclick: () => { closePopover(); fn(); updateTop(); }}, label);
  menu(tabEl, () => h("div", {class: "menu-list"},
    item("Rename", () => setTimeout(() => renameTab(bar.tabs.children[i], i, name))),
    item("Duplicate", () => layout.addTab(`${name} copy`, structuredClone(layout.save().tabs[i].root))),
    layout.model.tabs.length > 1 && item("Close", () => layout.closeTab(i)),
    item("Open layout…", () => setTimeout(() => layoutMenu(bar.tabs.children[i]))),
    item(`Reset ${VIEWS[ws.view].title} layout`, () => ws.resetLayout())));
}

// double-click a tab to rename it in place
function renameTab(tabEl, i, name) {
  const layout = ws.layout;
  const input = h("input", {class: "tab-edit", value: name});
  let open = true;
  const done = (save) => {
    if (!open) return;
    open = false;
    if (save && input.value.trim()) layout.renameTab(i, input.value.trim());
    updateTop();
  };
  input.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") done(true); else if (e.key === "Escape") done(false); });
  input.addEventListener("blur", () => done(true));
  tabEl.replaceWith(input);  // not inside it: the tab button would take the input's clicks and keys
  input.focus();
  input.select();
}

// ---- route workspace: one clock, one store, one layout per view ----

class Workspace {
  constructor(route, files, view, {t = 0, repeat = 1}) {
    this.route = route;
    this.dongle = route.fullname.split("|")[0];
    this.files = null;  // each segment's log from the route's source (data/sources.js); the drive view mounts without it
    this.filesReady = files.then((f) => { this.files = f; });
    this.filesReady.catch(() => {});  // a view that needs the logs shows the failure; until then it's not an error
    this.repeat = repeat;
    this.store = null;
    this.storeSubs = new Set();
    this.layouts = {};
    this.clock = createClock(Math.max(1, (route.end_time_utc_millis - route.start_time_utc_millis) / 1000 * repeat));
    this.clock.seek(t);
    this.main = h("div", {class: "workspace"});
    this.playbar = playbar(this.ctx());
    this.el = h("div", {class: "ws"}, this.main, this.playbar.el);
    this.offs = [this.clock.on("t", () => this.syncUrl()), this.clock.on("play", () => this.syncUrl())];
    // .json/.xml layouts dropped on the workspace open as new tabs
    acceptLayoutFiles(this.main, (f) => this.openLayout(f));
    this.setView(view);
  }

  ctx() {
    const self = this;
    return {
      route: this.route, clock: this.clock, perf, theme,
      get store() { return self.ensureStore(); },
      peekStore: () => this.store,
      // fn(store) once the logs load; returns its unsubscribe
      onStore: (fn) => {
        if (this.store) { fn(this.store); return () => {}; }
        this.storeSubs.add(fn);
        return () => this.storeSubs.delete(fn);
      },
      openRoute: (name) => this.openRoute(name),
    };
  }

  // logs load only once something needs them (the drive view doesn't); null until the file list is in
  ensureStore() {
    if (this.store || !this.files) return this.store;
    this.store = new Store(this.route.fullname, this.files, {t: this.clock.t, repeat: this.repeat});
    this.offs.push(watchRouteStart(this.route, this.store));
    const schema = logSchema(this.store, this.route);  // switches to the log's own schema for forks and unknown fields
    this.offs.push(() => schema.dispose(), () => status.note("schema", ""),
      schema.on((s) => status.note("schema", s.active === "log" && s.fork ? `${s.label} schema` : "")));
    // the store reprioritizes and reports buffering as the playhead moves; it publishes its own load status
    this.offs.push(this.clock.on("t", () => this.store.focus(this.clock.t)));
    this.store.on("progress", () => {
      if (this.store.route.duration) this.clock.setDuration(this.store.route.duration);
    });
    for (const fn of this.storeSubs) fn(this.store);
    this.storeSubs.clear();  // they've had it; holding them would hold every pane tree rendered since
    return this.store;
  }

  get layout() { return this.layouts[this.view]; }

  setView(view) {
    if (!VIEWS[view]) view = "drive";
    if (view !== this.view) this.persist();
    this.view = view;
    this.el.dataset.view = view;
    if (!this.layouts[view]) {
      const model = savedLayout(view);
      if (usesData(model) && !this.files) {
        this.filesReady.then(() => { if (this.view === view) this.setView(view); }, (e) => this.main.replaceChildren(errorPage(e)));
        this.main.replaceChildren(h("div", {class: "ws-wait"}, h("span", {class: "spinner"}), "Fetching the route's file list…"));
        return updateTop();
      }
      if (usesData(model)) this.ensureStore();
      this.layouts[view] = new Layout(this.ctx(), model, (m) => {
        localStorage.setItem(`ycpe_layout_${view}`, JSON.stringify(m));
        if (usesData(m)) this.ensureStore();
      });
    }
    this.main.replaceChildren(this.layout.el);
    this.layout.invalidateAll();
    this.syncUrl(true);
    updateTop();
  }

  // install a whole new model for the view (null: back to the default)
  applyLayout(model) {
    if (model) localStorage.setItem(`ycpe_layout_${this.view}`, JSON.stringify(model));
    else localStorage.removeItem(`ycpe_layout_${this.view}`);
    this.layout?.dispose();
    delete this.layouts[this.view];
    this.setView(this.view);
  }

  resetLayout() { this.applyLayout(null); }

  // panes change their own state (curves, filters) without an edit to the layout; saved whenever they could go away
  persist() {
    for (const [view, layout] of Object.entries(this.layouts)) localStorage.setItem(`ycpe_layout_${view}`, JSON.stringify(layout.save()));
  }

  // a layout file, URL or text (ours, jotpluggler's or PlotJuggler's), opened as new tabs
  async openLayout(src) {
    try {
      const {layout, warnings} = await loadLayout(src);
      this.applyLayout(mergeLayout(this.layout?.save() ?? savedLayout(this.view), layout, "tab"));
      if (warnings.length) toast(`Layout opened with ${warnings.length} warning${warnings.length > 1 ? "s" : ""}: ${warnings[0]}${warnings.length > 1 ? " …" : ""}`);
    } catch (e) {
      toast(`Couldn't open the layout: ${e.message}`);
    }
  }

  // a connect time range: the playhead at its start, and only it in view
  showRange([t0, t1]) {
    this.clock.seek(t0);
    this.clock.setRange(t0, t1);
  }

  openRoute(name) {
    go(`/${name.replace("|", "/")}/${this.view}`);
  }

  syncUrl(now = false) {
    clearTimeout(this.urlTimer);
    const write = () => {
      if (this.disposed) return;  // left meanwhile
      const q = new URLSearchParams(location.search);
      q.set("t", this.clock.t.toFixed(1));
      q.delete("layout");  // opened once, then it's part of the saved layout
      if (this.clock.t === 0) q.delete("t");
      const path = `/${routeKey(this.route).replace("|", "/")}/${this.view}`;
      history.replaceState(null, "", `${path}${q.size ? `?${q}` : ""}`);
    };
    if (now) write(); else this.urlTimer = setTimeout(write, 400);
  }

  dispose() {
    this.disposed = true;
    clearTimeout(this.urlTimer);
    for (const off of this.offs) off();
    this.playbar.dispose();
    this.clock.pause();
    this.persist();
    for (const l of Object.values(this.layouts)) l.dispose();
    this.store?.dispose();
  }
}

// the view's saved layout, else its default; one that doesn't read as a layout anymore is dropped
function savedLayout(view) {
  const key = `ycpe_layout_${view}`;
  try {
    const saved = JSON.parse(localStorage.getItem(key) || "null");
    if (saved?.tabs?.length) return checkLayout(saved);
  } catch {
    localStorage.removeItem(key);
    toast(`The saved ${VIEWS[view].title} layout couldn't be read, so it's back to the default`);
  }
  return VIEWS[view].layout;
}

const usesData = (model) => JSON.stringify(model.tabs).match(/"pane":"([^"]+)"/g)?.some((m) => PANES[m.slice(8, -1)]?.data);

// ---- keyboard ----

// undo/redo of the layout, on window so it runs last: panes take the keys first for their own undo (a plot's view, the
// CAN view's DBC edits) and mark them handled
addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (!(e.ctrlKey || e.metaKey) || (k !== "z" && k !== "y") || e.defaultPrevented || !ws?.layout) return;
  if (e.target.closest?.("input, textarea, select, [contenteditable]")) return;
  e.preventDefault();
  if (k === "y" || e.shiftKey) ws.layout.redo(); else ws.layout.undo();
  updateTop();
});

// the global keys, for whatever a pane didn't handle itself
document.addEventListener("keydown", (e) => {
  if (e.defaultPrevented || e.target.closest?.("input, textarea, select, [contenteditable]") || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "/") { e.preventDefault(); document.querySelector(".omnibox").focus(); return; }
  if (e.key === "t") return theme.toggle();
  if (e.key === "?") return help();
  if (!ws || e.target.closest?.("dialog")) return;
  const c = ws.clock, step = e.shiftKey ? 10 : 1;
  const keys = {
    " ": () => c.toggle(),
    ArrowLeft: () => c.seek(c.t - step),
    ArrowRight: () => c.seek(c.t + step),
    ",": () => c.seek(c.t - 0.05),
    ".": () => c.seek(c.t + 0.05),
    "[": () => c.setRate(RATES[Math.max(0, RATES.indexOf(c.rate) - 1)]),
    "]": () => c.setRate(RATES[Math.min(RATES.length - 1, RATES.indexOf(c.rate) + 1)]),
    l: () => c.setLoop(c.loop ? null : [...c.range]),
    f: () => c.setRange(0, c.duration),
    Home: () => c.seek(0),
    End: () => c.seek(c.duration),
  };
  Object.keys(VIEWS).forEach((v, i) => { keys[String(i + 1)] = () => ws.setView(v); });
  if (keys[e.key]) { e.preventDefault(); keys[e.key](); }
});

// keys are space separated; "+" joins a combination
const SHORTCUTS = [
  ["space", "play / pause"], ["← →", "seek 1 s (shift: 10 s)"], [", .", "step 50 ms"], ["[ ]", "slower / faster"],
  ["l", "loop the visible range"], ["f", "show the whole route"], ["Home End", "start / end"], ["1 2 3 4", "Drive / Plot / CAN / Files"],
  ["/", "search routes and devices"], ["t", "light / dark"], ["?", "this help"],
  ["ctrl+z", "undo (shift: redo): in a plot its view, in CAN the DBC edits, else the layout"],
];
const GESTURES = [
  ["double-click a series", "add it to a plot"], ["drag in a plot", "zoom to the box (alt: time only, shift: values only)"],
  ["right-drag in a plot", "pan"], ["right-click a plot", "options, custom series, export"], ["double-click a plot", "fit"],
  ["drag a pane header", "move the pane (edges dock, center swaps)"], ["double-click a divider", "even out the split"],
  ["wheel on the timeline", "zoom the visible range"], ["drop rlog or qlog files on the page", "open them"],
];

// a modal: Escape, the close button, a click outside or ? again closes it
function help() {
  const open = document.querySelector(".help");
  if (open) return open.close();
  const keys = (k) => k.split(" ").map((x) => x.split("+").map((y, i) => [i > 0 && "+", h("kbd", {}, y)]));
  const el = h("dialog", {class: "help", "aria-labelledby": "help-title", onclose: () => el.remove(), onclick: (e) => { if (e.target === el) el.close(); }},
    h("div", {class: "help-card"},
      h("div", {class: "help-head"}, h("h2", {id: "help-title"}, "Keyboard and mouse"), h("button", {class: "link", onclick: () => el.close()}, "close")),
      h("dl", {}, SHORTCUTS.map(([k, v]) => [h("dt", {}, keys(k)), h("dd", {}, v)]),
        GESTURES.map(([g, v]) => [h("dt", {class: "muted"}, g), h("dd", {}, v)]))));
  document.body.append(el);
  el.showModal();
}

// ---- routing ----

function closeWorkspace() {
  ws?.dispose();
  ws = null;
  document.body.classList.remove("in-ws");
}

let navId = 0;
async function render() {
  const id = ++navId;
  const main = document.getElementById("main");
  const parts = location.pathname.split("/").filter(Boolean);
  const q = new URLSearchParams(location.search);
  closePopover();

  if (parts[0] === "auth" || parts[0] === "login") {
    closeWorkspace();
    updateTop();
    return main.replaceChildren((parts[0] === "auth" ? authPage : loginPage)(q));
  }
  if (auth.isAuthenticated() && !state.devices) {
    [state.me, state.devices] = await Promise.all([api.me().catch(() => null), api.devices().catch(() => [])]);
  }

  if (openLoose(parts, q, main)) return;
  const target = parts.length >= 2 && DONGLE_RE.test(parts[0]) ? routeFromPath(parts) : null;  // "/<dongle>/<log>[/2:5][/q][/<view>]"
  const routeName = target?.name;
  if (routeName) {
    if (ws && routeKey(ws.route) === routeName) {
      if (target.view && target.view !== ws.view) ws.setView(target.view);
      if (target.range) ws.showRange(target.range);
      return;
    }
    document.body.classList.add("loading");
    try {
      const record = routeRecord(routeName);
      const files = routeLogs(routeName, record);
      files.catch(() => {});
      const route = await record;
      if (id !== navId) return;
      ws?.dispose();
      ws = new Workspace(route, files, target.view || "drive", {t: +q.get("t") || 0, repeat: Math.max(1, +q.get("repeat") || 1)});
      if (target.range) ws.showRange(target.range);
      rememberRoute(route);
      if (q.has("layout")) ws.openLayout(q.get("layout"));
      document.body.classList.add("in-ws");
      main.replaceChildren(ws.el);
    } catch (e) {
      const card = [403, 404].includes(e.status) && await pullCard(routeName, render).catch(() => null);  // only on the LAN device
      if (id === navId) {
        closeWorkspace();
        main.replaceChildren(card || errorPage(e));
      }
    } finally {
      document.body.classList.remove("loading");
    }
    return updateTop();
  }

  closeWorkspace();
  updateTop();
  if (DONGLE_RE.test(parts[0] || "")) {
    const page = await devicePage(parts[0], state.devices).catch(errorPage);
    if (id === navId) main.replaceChildren(page);
  } else if (state.devices?.length === 1) {
    go(`/${state.devices[0].dongle_id}`, true);
  } else {
    main.replaceChildren(homePage(state.devices));
  }
}

// logs outside the API (data/sources.js): files dropped anywhere on the page, or a direct link (?url=). False when the
// path is neither.
let dropped = null;
function openLoose(parts, q, main) {
  const loose = q.has("url") ? urlRoute(q.get("url")) : dropped?.route.fullname === `${parts[0]}|${parts[1]}` ? dropped : null;
  if (!loose && parts[0] !== "local") return false;
  if (loose && ws?.route.fullname === loose.route.fullname) {
    if (parts[2] && parts[2] !== ws.view) ws.setView(parts[2]);
    return true;
  }
  ws?.dispose();
  ws = loose && new Workspace(loose.route, Promise.resolve(loose.urls), parts[2] || "plot", {t: +q.get("t") || 0});
  document.body.classList.toggle("in-ws", !!ws);
  main.replaceChildren(ws ? ws.el : localPage(openFiles));
  updateTop();
  return true;
}
function openFiles(files) {
  dropped = localRoute(files);
  if (dropped) go(`/${dropped.route.fullname.replace("|", "/")}/plot`);
  else toast("No rlog, qlog or camera files there");
}
// files dropped anywhere a pane didn't take them (layouts on the workspace, say) open as a local route
const carriesFiles = (e) => e.dataTransfer?.types.includes("Files");
document.addEventListener("dragover", (e) => { if (carriesFiles(e)) e.preventDefault(); });
document.addEventListener("drop", async (e) => {
  if (e.defaultPrevented || !carriesFiles(e)) return;
  e.preventDefault();
  openFiles(await droppedFiles(e.dataTransfer));
});

function errorPage(e) {
  console.error(e);
  if (e.status === 401 || e.status === 403) return signInCard();
  const msg = e.status === 404 ? "Not found." : String(e.message || e);
  return h("div", {class: "page-msg"}, h("h2", {}, msg), h("p", {class: "muted"}, "Try a route name or dongle id in the box above."));
}

document.querySelector("#app > .topbar").replaceWith(topBar());
addEventListener("ycpe:route-start", (e) => { if (ws?.route === e.detail) updateTop(); });
window.addEventListener("popstate", render);
addEventListener("pagehide", () => ws?.persist());
document.addEventListener("visibilitychange", () => { if (document.hidden) ws?.persist(); });
// signing in or out re-renders the current page under the new account; /auth navigates on by itself (rendering it
// again would spend the one-time code twice). A token the API refused leaves an open route as it is: it's loaded.
auth.on((reason) => {
  state.me = state.devices = null;
  if (location.pathname.startsWith("/auth")) return;
  if (reason === "expired") toast("Your session expired. Sign in again to see private routes and devices.");
  if (reason === "expired" && ws) return updateTop();
  closeWorkspace();
  render();
});
render();
