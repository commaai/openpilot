import {api, auth} from "/api.js";
import {daemon} from "/daemon.js";
import {localBadge} from "/local.js";
import {h, parseClock} from "/dom.js";
import {logo, icon} from "/icons.js";
import {perf, hud} from "/perf.js";
import {theme} from "/theme.js";
import {createClock, RATES} from "/clock.js";
import {Layout, checkLayout} from "/layout.js";
import {PANES} from "/panes.js";
import {VIEWS, LIVE_VIEWS} from "/views.js";
import {playbar} from "/playbar.js";
import {status, loadPill, footprint} from "/status.js";
import {Store} from "/data/store.js";
import {logSchema} from "/data/schemas.js";
import {routeRecord, routeKey, routeFromPath, routeLogs, localRoute, urlRoute, droppedFiles, watchRouteStart} from "/data/sources.js";
import {devicePage, prefetchDevice} from "/pages/device.js";
import {homePage} from "/pages/home.js";
import {loginPage, authPage, signInCard, loginPath} from "/pages/login.js";
import {omnibox, rememberRoute, resolve} from "/pages/omnibox.js";
import {localPage, pullCard} from "/pages/local.js";
import {create as routesPane} from "/panes/routes.js";
import {layoutPicker, loadLayout, mergeLayout, acceptLayoutFiles} from "/layouts/picker.js";
import {uniqueCustoms} from "/layouts/import.js";

const DONGLE_RE = /^[0-9a-f]{16}$/i;
const LOG_RE = /^[0-9a-f]{8}--[0-9a-f]{10}$/i;

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
  const el = h("div", {class: "toast", role: "status"}, text);
  document.body.append(el);
  setTimeout(() => el.remove(), 3000);
}

const state = {me: null, devices: null, devicesError: null};
// a device left the account (unpaired): the next page asks for the list again
export const devicesChanged = () => { state.devices = null; };
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
  // under the anchor, inside the window
  const r = anchor.getBoundingClientRect();
  el.style.top = `${r.bottom + 4}px`;
  document.body.append(el);
  el.style.left = `${Math.max(8, Math.min(r.left, innerWidth - el.offsetWidth - 8))}px`;
  setTimeout(() => {
    document.addEventListener("pointerdown", outside);
    document.addEventListener("focusin", outside);
    document.addEventListener("keydown", escape);
    (el.querySelector(".menu-list > .on") || el.querySelector("button, a[href], input, select"))?.focus();  // once the caller has filled it
  });
  return el;
}

// the open popover, ours or a pane's (they share the class)
function closePopover() {
  const open = document.querySelector(".popover");
  if (open?._close) open._close(); else open?.remove();
}

// shown once updateTop() names them
const crumb = (title) => h("button", {class: "crumb", title, hidden: true}, h("span"), icon("down", 16));
const bar = {
  device: crumb("Devices"),
  route: crumb("Routes on this device"),
  views: h("nav", {class: "views"}),
  tabs: h("nav", {class: "tabs", "aria-label": "Layouts"}),  // the tab list, then New tab and the layouts menu
  tablist: h("span", {class: "tablist", role: "tablist", "aria-label": "Layout tabs"}),
};

function topBar() {
  const search = searchBox();
  bar.tabs.addEventListener("wheel", (e) => {
    if (e.deltaX || !e.deltaY) return;
    e.preventDefault();
    bar.tabs.scrollLeft += e.deltaY;
    tabsScrolled = true;
  }, {passive: false});
  tabsObserver.observe(bar.tabs);
  // arrows move between the tabs (Enter or space opens one); Delete closes the focused one
  bar.tablist.addEventListener("keydown", (e) => {
    const tabs = [...bar.tablist.querySelectorAll("[role=tab]")], i = tabs.indexOf(e.target);
    const to = {ArrowLeft: i - 1, ArrowRight: i + 1, Home: 0, End: tabs.length - 1}[e.key];
    if (i < 0 || (to === undefined && e.key !== "Delete")) return;
    e.preventDefault();
    e.stopPropagation();  // not the playhead's arrows
    if (to !== undefined) return tabs[(to + tabs.length) % tabs.length].focus();
    if (tabs.length > 1) {
      ws.layout.closeTab(i);
      updateTop();
    }
  });
  bar.device.addEventListener("click", () => menu(bar.device, (close) => h("div", {class: "menu-list"},
    (state.devices || []).map((d) => h("a", {class: `menu-row ${d.dongle_id === (ws?.dongle || location.pathname.split("/")[1]) ? "on" : ""}`,
      href: `/${d.dongle_id}`, onclick: (e) => { close(); link(`/${d.dongle_id}`)(e); },
      onpointerenter: () => prefetchDevice(d.dongle_id)},
      h("span", {class: `dot ${isOnline(d) ? "online" : ""}`}), h("span", {}, d.alias || d.dongle_id), h("span", {class: "muted mono"}, d.dongle_id.slice(0, 6)))),
    !state.devices?.length && h("div", {class: "menu-row muted"},
      !auth.isAuthenticated() ? "Sign in to see your devices" : state.devicesError ? "Couldn't load devices" : "No devices"))));
  bar.route.addEventListener("click", () => {
    if (!ws || ws.session) return;
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
    bar.device, h("span", {class: "sep", hidden: true}, "›"), bar.route, search, bar.views, bar.tabs,
    h("span", {class: "grow"}), loadPill(), localBadge(), hud(footprint()), helpBtn, themeBtn, account);
}

function updateTop() {
  // the focused control of the bar, focused again once rebuilt (or the active tab, after a tab closed)
  const focused = document.activeElement?.closest?.(".topbar [data-key]")?.dataset.key;
  const dongle = ws?.dongle || location.pathname.split("/")[1];
  const device = state.devices?.find((d) => d.dongle_id === dongle);
  bar.device.firstChild.textContent = device ? device.alias || dongle : DONGLE_RE.test(dongle || "") ? dongle : "devices";
  bar.device.hidden = false;
  bar.route.hidden = bar.route.previousElementSibling.hidden = !ws;
  bar.views.hidden = bar.tabs.hidden = !ws;
  if (!ws) return bar.tabs.replaceChildren();  // the tabs' handlers would keep the closed workspace alive
  const start = new Date(ws.route.start_time_utc_millis);  // 0 for local files without a timestamp: show the name
  // a live session: the device crumb names a device's, else the source does
  bar.route.firstChild.textContent = ws.session ? (DONGLE_RE.test(ws.dongle) ? "live" : `live · ${ws.dongle}`) : ws.route.start_time_utc_millis
    ? `${start.toLocaleDateString(undefined, {month: "short", day: "numeric"})} ${start.toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit", hour12: false})}`
    : ws.route.fullname.split("|")[1];
  bar.route.title = ws.session ? ws.session.label : `${ws.route.fullname} · ${[ws.route.platform, ws.route.version].filter(Boolean).join(" · ")}`;
  // made once per set of views (a live session has its own) and kept, so a focused one stays focused: focusing a new
  // one would lay the new view out in the click
  if (viewsShown !== ws.views) {
    viewsShown = ws.views;
    bar.views.replaceChildren(...Object.entries(ws.views).map(([k, v], i) => h("button", {class: "view", "data-key": `view:${k}`,
      title: `${v.title} (${i + 1})`, onclick: () => ws?.setView(k)}, v.title)));
  }
  for (const b of bar.views.children) {
    const on = b.dataset.key === `view:${ws.view}`;
    b.classList.toggle("on", on);
    if (on) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
  }
  const layout = ws.layout;
  if (!layout) {  // the view is still waiting on its files
    bar.tabs.replaceChildren();
    return refocus(focused);
  }
  const select = (tab) => {
    for (const t of bar.tablist.children) {
      t.classList.toggle("on", t === tab);
      t.querySelector("[role=tab]")?.setAttribute("aria-selected", t === tab);
      t.querySelector("[role=tab]")?.setAttribute("tabindex", t === tab ? 0 : -1);
    }
  };
  bar.tablist.replaceChildren(...layout.model.tabs.map((t, i) => h("span", {class: `tab ${i === layout.model.active ? "on" : ""}`,
    onclick: (e) => {  // in place, so a double-click lands on the same tab
      if (i === layout.model.active) return;
      layout.selectTab(i);
      select(e.currentTarget);
    },
    ondblclick: (e) => { if (!e.target.closest(".x")) renameTab(e.currentTarget, i, t.name); },
    oncontextmenu: (e) => { e.preventDefault(); tabMenu(e.currentTarget.firstChild, i, t.name); },
    onauxclick: (e) => { if (e.button === 1) { layout.closeTab(i); updateTop(); } }},
    h("button", {class: "tab-name", role: "tab", "aria-selected": String(i === layout.model.active), tabindex: i === layout.model.active ? 0 : -1,
      "data-key": `tab:${i}`}, t.name),
    // for the mouse: a tab list holds only tabs, and the keyboard closes one with Delete
    layout.model.tabs.length > 1 && h("button", {class: "x", title: "Close tab (Delete)", tabindex: -1, "aria-hidden": "true",
      // not on a double-click's second click: that's "+" (or another close button) clicked again after the strip moved
      onclick: (e) => { e.stopPropagation(); if (e.detail < 2) { layout.closeTab(i); updateTop(); } }}, icon("close", 12)))));
  bar.tabs.replaceChildren(bar.tablist,
    h("button", {class: "tab add", title: "New tab", "data-key": "add", onclick: () => { layout.addTab(); updateTop(); }}, icon("add", 16)),
    h("button", {class: "tab add", title: "Layouts: open, import, export, share", "data-key": "layouts", onclick: (e) => layoutMenu(e.currentTarget)}, icon("more_horiz", 16)));
  refocus(focused);
  // scrolled to once laid out (tabsObserver), and only when the active tab or the tab set changed
  const shown = `${ws.view} ${layout.model.active} ${layout.model.tabs.map((t) => t.name).join("|")}`;
  if (shown !== tabsShown) {
    tabsShown = shown;
    tabsScrolled = false;
    tabsMoved = true;
    tabsObserver.observe(bar.tabs.querySelector(".tab.on"));  // reported once it's laid out
  }
}

function refocus(key) {
  if (!key) return;
  const el = bar.views.querySelector(`[data-key="${key}"]`) || bar.tabs.querySelector(`[data-key="${key}"]`) || bar.tablist.querySelector(".tab.on [role=tab]");
  if (el !== document.activeElement) el?.focus();
}

let viewsShown = null;     // the set of views the bar has buttons for
let tabsShown = "";        // the tabs and active tab the strip last scrolled to
let tabsScrolled = false;  // the user scrolled the strip since
let tabsMoved = false;     // the tabs changed since the strip scrolled to them

// scrolls the strip to the active tab after the tabs change, and when the bar's other items settle after them (unless
// the user scrolled it). It runs after layout, so its reads never force one.
const tabsObserver = new ResizeObserver(() => {
  if (tabsMoved) {
    tabsMoved = false;
    tabsObserver.disconnect();  // the tabs it was told of are gone or shown
    tabsObserver.observe(bar.tabs);
    return showActiveTab();
  }
  const on = bar.tabs.querySelector(".tab.on")?.getBoundingClientRect(), r = bar.tabs.getBoundingClientRect();
  if (on && !tabsScrolled && (on.left < r.left || on.right > r.right)) showActiveTab();
});

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
    item("Rename", () => setTimeout(() => renameTab(bar.tablist.children[i], i, name))),
    item("Duplicate", () => layout.addTab(`${name} copy`, structuredClone(layout.save().tabs[i].root))),
    layout.model.tabs.length > 1 && item("Close", () => layout.closeTab(i)),
    item("Open layout…", () => setTimeout(() => layoutMenu(bar.tablist.children[i]))),
    item(`Reset ${ws.views[ws.view].title} layout`, () => ws.resetLayout())));
}

// double-click a tab to rename it in place
function renameTab(tabEl, i, name) {
  const layout = ws.layout;
  const input = h("input", {class: "tab-edit", value: name, "data-key": `tab:${i}`});
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
  // session: a live one (data/live.js), whose route record this is
  constructor(route, files, view, {t = 0, repeat = 1, session = null}) {
    this.route = route;
    this.session = session;
    this.views = session ? LIVE_VIEWS : VIEWS;
    this.dongle = session ? route.dongle_id : route.fullname.split("|")[0];
    // each segment's log from the route's source (data/sources.js); the drive view mounts without it
    this.files = session ? [] : null;
    this.filesReady = files.then((f) => { this.files = f; });
    this.filesReady.catch(() => {});  // a view that needs the logs shows the failure; until then it's not an error
    this.repeat = repeat;
    this.href = location.pathname + location.search;  // the URL it shows, as syncUrl last wrote it
    this.store = null;
    this.storeSubs = new Set();
    this.layouts = {};
    this.written = {};  // view → its layout as this tab last saved it, or had it when another tab saved one
    this.clock = session ? createClock(0, {live: true}) : createClock(Math.max(1, (route.end_time_utc_millis - route.start_time_utc_millis) / 1000 * repeat));
    if (!session) this.clock.seek(t);
    this.main = h("div", {class: "workspace"});
    this.playbar = playbar(this.ctx());
    this.el = h("div", {class: "ws"}, this.main, this.playbar.el);
    this.offs = ["t", "play", "range"].map((ev) => this.clock.on(ev, () => this.syncUrl()));
    // another browser tab saved a layout: this one's copy only replaces it once it changes
    const stored = (e) => {
      const view = Object.keys(this.layouts).find((v) => e.key === `ycpe_layout_${this.layoutKey(v)}`);
      if (view) this.written[view] = JSON.stringify(this.layouts[view].save());
    };
    addEventListener("storage", stored);
    this.offs.push(() => removeEventListener("storage", stored));
    // .json/.xml layouts dropped on the workspace open as new tabs
    acceptLayoutFiles(this.main, (f) => this.openLayout(f));
    if (session) this.ensureStore();
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
      // for links: this route at the playhead in a view (url), and a view's layout as it is now
      url: (view) => this.url(view),
      layoutOf: (view) => this.layouts[view]?.save() ?? savedLayout(this.layoutKey(view), this.views[view]),
    };
  }

  // logs load only once something needs them (the drive view doesn't); null until the file list is in. A live
  // session's store is there from the start, and moves the clock's head and window
  ensureStore() {
    if (!this.store && this.session) {
      this.store = this.session.store;
      this.offs.push(this.store.on("live", (t) => this.clock.setEdge(t)), this.store.on("window", (s) => this.clock.setStart(s)));
      for (const fn of this.storeSubs) fn(this.store);
      this.storeSubs.clear();
    }
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

  // the view's layout is saved under this name
  layoutKey(view) { return this.views[view].key || view; }

  setView(view) {
    if (!this.views[view]) view = this.session ? "plot" : "drive";
    if (view !== this.view) this.persist();
    this.view = view;
    this.el.dataset.view = view;
    if (!this.layouts[view]) {
      const model = savedLayout(this.layoutKey(view), this.views[view]);
      if (usesData(model) && !this.files) {
        this.filesReady.then(() => { if (!this.disposed && this.view === view) this.setView(view); },
          (e) => { if (!this.disposed) this.main.replaceChildren(errorPage(e)); });
        this.main.replaceChildren(h("div", {class: "ws-wait"}, h("span", {class: "spinner"}), "Fetching the route's file list…"));
        return updateTop();
      }
      if (usesData(model)) this.ensureStore();
      this.layouts[view] = new Layout(this.ctx(), model, (m) => {
        localStorage.setItem(`ycpe_layout_${this.layoutKey(view)}`, this.written[view] = JSON.stringify(m));
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
    if (model) localStorage.setItem(`ycpe_layout_${this.layoutKey(this.view)}`, JSON.stringify(checkLayout(model)));
    else localStorage.removeItem(`ycpe_layout_${this.layoutKey(this.view)}`);
    this.layout?.dispose();
    delete this.layouts[this.view];
    this.setView(this.view);
  }

  resetLayout() { this.applyLayout(null); }

  // panes change their own state (curves, filters) without an edit to the layout; saved whenever they could go away,
  // if they changed
  persist() {
    for (const [view, layout] of Object.entries(this.layouts)) {
      const json = JSON.stringify(layout.save());
      if (json !== this.written[view]) localStorage.setItem(`ycpe_layout_${this.layoutKey(view)}`, this.written[view] = json);
    }
  }

  // a layout file, URL or text (ours, jotpluggler's or PlotJuggler's), opened as new tabs
  async openLayout(src) {
    try {
      const {layout, warnings} = await loadLayout(src);
      this.applyLayout(mergeLayout(this.layout?.save() ?? savedLayout(this.layoutKey(this.view), this.views[this.view]), layout, "tab"));
      if (warnings.length) toast(`Layout opened with ${warnings.length} warning${warnings.length > 1 ? "s" : ""}: ${warnings[0]}${warnings.length > 1 ? " …" : ""}`);
    } catch (e) {
      toast(`Couldn't open the layout: ${e.message}`);
    }
  }

  // a connect time range: only it in view, the playhead at its start (or where the URL's t put it)
  showRange([t0, t1], seek = true) {
    if (seek) this.clock.seek(t0);
    this.clock.setRange(t0, t1);
  }

  openRoute(name) {
    go(`/${name.replace("|", "/")}/${this.view}`);
  }

  syncUrl(now = false) {
    clearTimeout(this.urlTimer);
    const write = () => {
      if (this.disposed) return;  // left meanwhile
      history.replaceState(null, "", this.href = this.url());
    };
    if (now) write(); else this.urlTimer = setTimeout(write, 400);
  }

  // the path and query of the route at the playhead, in view
  url(view = this.view) {
    // a live session: the source's path and options (a time there means nothing next time)
    if (this.session) return `${this.session.path}/${view}${location.search}`;
    const q = new URLSearchParams(location.search);
    q.set("t", this.clock.t.toFixed(1));
    q.delete("layout");  // opened once, then it's part of the saved layout
    if (this.clock.t === 0) q.delete("t");
    if (this.repeat > 1) q.set("repeat", this.repeat); else q.delete("repeat");  // as clamped
    // a window narrower than the route stays in the path as connect's "/<start>/<end>" (whole seconds; not on a slice,
    // which that form can't follow)
    const [r0, r1] = this.clock.range;
    const span = !this.route.range && (r0 > 0 || r1 < this.clock.duration) ? `/${Math.floor(r0)}/${Math.ceil(r1)}` : "";
    return `/${routeKey(this.route).replace("|", "/")}${span}/${view}${q.size ? `?${q}` : ""}`;
  }

  dispose() {
    this.disposed = true;
    clearTimeout(this.urlTimer);
    for (const off of this.offs) off();
    this.playbar.dispose();
    this.clock.pause();
    this.clock.dispose();
    this.persist();
    for (const l of Object.values(this.layouts)) l.dispose();
    if (this.session) this.session.stop();
    else this.store?.dispose();
  }
}

// the view's layout saved under name, else the view's default; one that doesn't read as a layout anymore is dropped
function savedLayout(name, view) {
  const key = `ycpe_layout_${name}`;
  try {
    const saved = JSON.parse(localStorage.getItem(key) || "null");
    // custom series named alike were saved with different code before names had to be unique: they're renamed
    if (saved?.tabs?.length) return uniqueCustoms(checkLayout(saved));
  } catch {
    localStorage.removeItem(key);
    toast(`The saved ${view.title} layout couldn't be read, so it's back to the default`);
  }
  return view.layout;
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

// ctrl+f finds a series, as in jotpluggler: the Series filter, when the open tab shows one (the browser's find can't
// see into the tree or the plots)
addEventListener("keydown", (e) => {
  if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "f" || e.shiftKey || e.altKey || e.defaultPrevented) return;
  const filter = ws?.layout?.el.querySelector(".browser .filter");
  if (!filter) return;
  e.preventDefault();
  filter.focus();
  filter.select();
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
    f: () => c.setRange(c.start, c.duration),
    Home: () => c.seek(c.start),
    End: () => c.seek(c.duration),  // live: following again
  };
  Object.keys(ws.views).forEach((v, i) => { keys[String(i + 1)] = () => ws.setView(v); });
  if (keys[e.key]) { e.preventDefault(); keys[e.key](); }
});

// keys are space separated; "+" joins a combination
const SHORTCUTS = [
  ["space", "play / pause"], ["← →", "seek 1 s (shift: 10 s)"], [", .", "step 50 ms"], ["[ ]", "slower / faster"],
  ["l", "loop the visible range"], ["f", "show the whole route"], ["Home End", "start / end (live: follow)"], ["1 2 3 4", "Drive / Plot / CAN / Files"],
  ["/", "search routes and devices"], ["ctrl+f", "filter the series"], ["t", "light / dark"], ["?", "this help"],
  ["ctrl+z", "undo (shift: redo): in a plot its view, in CAN the DBC edits, else the layout"],
];
const GESTURES = [
  ["click in a plot or the timeline", "seek there"], ["double-click a series", "add it to a plot"],
  ["drag in a plot", "zoom to the box (alt: time only, shift: values only)"], ["right-drag in a plot", "pan"],
  ["wheel in a plot", "zoom time (shift or on the y-axis: values)"], ["right-click a plot", "options, custom series, export"],
  ["right-click a legend entry", "color, transform, remove that curve"], ["double-click a plot", "fit"],
  ["drag a pane header", "move the pane (edges dock, center swaps)"], ["double-click a divider", "even out the split"],
  ["wheel on the timeline", "zoom the visible range"], ["drag the strip above the timeline", "pan / resize the visible window"],
  ["click the playback time", "type a time to go to (3:12 or 192)"], ["drop rlog or qlog files on the page", "open them"],
];
const TOUCHES = [
  ["tap a plot or the timeline", "seek there"], ["double-tap a series", "add it to a plot"], ["long-press a plot", "options, custom series, export"],
  ["drag in a plot", "pan"], ["pinch a plot", "zoom time"], ["drag a plot's y-axis", "pan the values"],
  ["drag the strip above the timeline", "pan / resize the visible window"], ["tap the playback time", "type a time to go to (3:12 or 192)"],
];

// a modal: Escape, the close button, a click outside or ? again closes it. On a touch screen, the touch gestures
function help() {
  const open = document.querySelector(".help");
  if (open) return open.close();
  const touch = matchMedia("(pointer: coarse)").matches;
  const keys = (k) => k.split(" ").map((x) => x.split("+").map((y, i) => [i > 0 && "+", h("kbd", {}, y)]));
  const el = h("dialog", {class: "help", "aria-labelledby": "help-title", onclose: () => el.remove(), onclick: (e) => { if (e.target === el) el.close(); }},
    h("div", {class: "help-card"},
      h("div", {class: "help-head"}, h("h2", {id: "help-title"}, touch ? "Touch and keyboard" : "Keyboard and mouse"), h("button", {class: "link", onclick: () => el.close()}, "close")),
      h("dl", {}, (touch ? TOUCHES : []).map(([g, v]) => [h("dt", {class: "muted"}, g), h("dd", {}, v)]),
        SHORTCUTS.map(([k, v]) => [h("dt", {}, keys(k)), h("dd", {}, v)]),
        (touch ? [] : GESTURES).map(([g, v]) => [h("dt", {class: "muted"}, g), h("dd", {}, v)]))));
  document.body.append(el);
  el.showModal();
}

// ---- routing ----

function closeWorkspace() {
  ws?.dispose();
  ws = null;
  document.body.classList.remove("in-ws");
}

// the path still names the open workspace's route (or loose logs), whatever the view
const showsWorkspace = (parts) => ws.session ? `/${parts.join("/")}/`.startsWith(`${ws.session.path}/`) : [parts.length >= 2 && routeFromPath(parts)?.name, `${parts[0]}|${parts[1]}`].includes(routeKey(ws.route));

// a page that throws shows the error rather than its spinner forever
function render() {
  return renderPage().catch((e) => {
    document.body.classList.remove("loading");
    document.getElementById("main").replaceChildren(errorPage(e));
  });
}

let navId = 0;
async function renderPage() {
  const id = ++navId;
  const main = document.getElementById("main");
  const parts = location.pathname.split("/").filter(Boolean);
  const q = new URLSearchParams(location.search);
  closePopover();
  // the API names devices and routes in lowercase: a typed /5BEB…/0000010A--… is the same route
  const lower = parts.map((p, i) => i === 0 && DONGLE_RE.test(p) || i === 1 && LOG_RE.test(p) ? p.toLowerCase() : p);
  if (DONGLE_RE.test(parts[0] || "") && lower.join("/") !== parts.join("/")) return go(`/${lower.join("/")}${location.search}`, true);
  // leaving the route with unsaved DBC edits asks first, as closing the browser tab does; staying puts its URL back
  if (ws?.store?.can.dbcs?.dirty && !showsWorkspace(parts) && !confirm("The DBC files have unsaved changes. Leave and discard them?")) {
    return history.pushState(null, "", ws.href);
  }

  if (parts[0] === "auth" || parts[0] === "login") {
    closeWorkspace();
    updateTop();
    return main.replaceChildren((parts[0] === "auth" ? authPage : loginPage)(q));
  }
  if (auth.isAuthenticated() && !state.devices) {
    updateTop();
    document.body.classList.add("loading");
    main.replaceChildren(h("div", {class: "ws-wait"}, h("span", {class: "spinner"}), "Loading devices…"));
    // a failed list stays null, so the next page asks again
    [state.me, state.devices] = await Promise.all([api.me().catch(() => null),
      api.devices().then((d) => { state.devicesError = null; return d; }, (e) => { state.devicesError = e; return null; })]);
    document.body.classList.remove("loading");
  }

  if (parts[0] === "live" || (parts[1] === "live" && DONGLE_RE.test(parts[0]))) return openLiveSession(parts, q, main, id);
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
      // ?repeat=n (a load test): the logs n times over, at most 50
      const repeat = Math.min(50, Math.max(1, Math.floor(+q.get("repeat")) || 1));
      const t = urlTime(q);
      ws = new Workspace(route, files, target.view || "drive", {t: t ?? 0, repeat});
      if (target.range) ws.showRange(target.range, t === null);
      rememberRoute(route);
      if (q.has("layout")) ws.openLayout(q.get("layout"));
      document.body.classList.add("in-ws");
      main.replaceChildren(ws.el);
    } catch (e) {
      const card = [403, 404].includes(e.status) && await pullCard(routeName, render).catch(() => null);  // only on the LAN device
      if (id === navId) {
        closeWorkspace();
        main.replaceChildren(card || errorPage(e, {route: routeName}));
      }
    } finally {
      document.body.classList.remove("loading");
    }
    return updateTop();
  }

  // whatever else the omnibox reads goes where it names: connect's "/<dongle>/<start ms>/<end ms>", "/<dongle>|<log id>"
  if (parts.length >= 2 || parts.length && !DONGLE_RE.test(parts[0])) {
    const path = await resolve(location.pathname.slice(1) + location.search).catch(() => null);
    if (id !== navId) return;
    if (path && path !== location.pathname + location.search) return go(path, true);
  }
  closeWorkspace();
  updateTop();
  if (DONGLE_RE.test(parts[0] || "")) {
    const page = await devicePage(parts[0], state.devices).catch((e) => errorPage(e, {device: parts[0]}));
    if (id === navId) main.replaceChildren(page);
  } else if (parts.length) {
    main.replaceChildren(h("div", {class: "page-msg"}, h("h2", {}, `Nothing at ${location.pathname}`),
      h("p", {class: "muted"}, "Try a route name or dongle id in the box above, or ", h("a", {href: "/", onclick: link("/")}, "go home"), ".")));
  } else if (state.devicesError) {
    main.replaceChildren(errorPage(state.devicesError, {retry: true}));
  } else if (state.devices?.length === 1) {
    go(`/${state.devices[0].dongle_id}`, true);
  } else {
    main.replaceChildren(homePage(state.devices));
  }
}

// a live session (data/live.js): /live lists the sources; /live/local (this computer's webrtcd), /live/panda,
// /live/replay?route=&rate= (a log played back as if live) and /<dongle>/live (a device through athena), each with an
// optional view after it. Loaded once one's opened, not before the first paint
async function openLiveSession(parts, q, main, id) {
  const [{openLive}, {livePage}] = await Promise.all([import("/data/live.js"), import("/pages/live.js")]);
  if (id !== navId) return;
  const dongle = parts[0] === "live" ? null : parts[0], kind = dongle ? "device" : parts[1];
  const path = dongle ? `/${dongle}/live` : `/live/${kind}`, view = parts[2];
  if (!kind) {
    closeWorkspace();
    updateTop();
    return main.replaceChildren(livePage(state.devices));
  }
  if (ws?.session?.path === path) {
    if (view && view !== ws.view) ws.setView(view);
    return;
  }
  const [module, opts] = {
    device: ["webrtc", {dongle}], local: ["webrtc", {local: true}], panda: ["panda", {}],
    replay: ["replay", {route: q.get("route") || "", rate: +q.get("rate") || 1}],
  }[kind] || [];
  closeWorkspace();
  updateTop();
  if (!module) return main.replaceChildren(errorPage({status: 404}));
  let session;
  try {
    session = await openLive(module, opts, {key: dongle || kind, path, window: +q.get("window") || undefined});
  } catch (e) {
    return id === navId && main.replaceChildren(errorPage(new Error(`Couldn't open the ${kind} source: ${e.message}`)));
  }
  if (id !== navId) return session.stop();
  ws = new Workspace(session.route, Promise.resolve([]), view || "plot", {session});
  window.__live = ws;  // for the harness
  document.body.classList.add("in-ws");
  main.replaceChildren(ws.el);
  updateTop();
}

// ?t=, in seconds or as the playbar shows it (3:12); null when there's none, or it isn't a time (which is said)
function urlTime(q) {
  const t = q.has("t") ? parseClock(q.get("t")) : null;
  if (Number.isNaN(t)) toast(`Couldn't read t=${q.get("t")}: give seconds or m:ss`);
  return Number.isNaN(t) ? null : t;
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
  ws = loose && new Workspace(loose.route, Promise.resolve(loose.urls), parts[2] || "plot", {t: urlTime(q) ?? 0});
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

// route or device: what the page was for, to say what wasn't found; retry: a button that renders the page again
function errorPage(e, {route, device, retry} = {}) {
  console.error(e);
  if (e.status === 401 || e.status === 403) return signInCard();
  const [msg, why] = e.status !== 404 ? [String(e.message || e), ""]
    : route ? [`Route ${route.replace("|", "/")} not found`, "Check the name, or it may not be uploaded or shared with you. "]
    : device ? [`Device ${device} not found`, "It may not be shared with this account. "]
    : ["Not found.", ""];
  return h("div", {class: "page-msg"}, h("h2", {}, msg), h("p", {class: "muted"}, why, "Try a route name or dongle id in the box above."),
    retry && h("button", {class: "button", onclick: () => render()}, "Retry"));
}

document.querySelector("#app > .topbar").replaceWith(topBar());
addEventListener("ycpe:route-start", (e) => { if (ws?.route === e.detail) updateTop(); });
window.addEventListener("popstate", render);
addEventListener("pagehide", () => ws?.persist());
document.addEventListener("visibilitychange", () => { if (document.hidden) ws?.persist(); });
// signing in or out re-renders the current page under the new account; /auth navigates on by itself (rendering it
// again would spend the one-time code twice). A token the API refused leaves an open route as it is: it's loaded.
auth.on((reason) => {
  state.me = state.devices = state.devicesError = null;
  if (location.pathname.startsWith("/auth")) return;
  if (reason === "expired") toast("Your session expired. Sign in again to see private routes and devices.");
  if (reason === "expired" && ws) return updateTop();
  closeWorkspace();
  render();
});
render();
