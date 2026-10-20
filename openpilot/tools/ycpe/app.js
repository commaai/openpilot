import {api, auth} from "/api.js";
import {h} from "/dom.js";
import {logo} from "/icons.js";
import {perf, hud} from "/perf.js";
import {theme} from "/theme.js";
import {createClock, RATES} from "/clock.js";
import {Layout} from "/layout.js";
import {PANES} from "/panes.js";
import {VIEWS} from "/views.js";
import {playbar} from "/playbar.js";
import {loadPill, footprint} from "/status.js";
import {Store} from "/data/store.js";
import {routeLogs} from "/data/sources.js";
import {devicePage, prefetchDevice} from "/pages/device.js";
import {homePage} from "/pages/home.js";
import {loginPage, authPage, signInCard, loginPath} from "/pages/login.js";
import {omnibox, rememberRoute} from "/pages/omnibox.js";
import {create as routesPane} from "/panes/routes.js";

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
    go: (path) => go(ws ? path.replace(/^(\/[0-9a-f]{16}\/[^/?]+)(?=\?|$)/, `$1/${ws.view}`) : path),
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

function menu(anchor, build) {
  document.querySelector(".popover")?.remove();
  const el = h("div", {class: "popover"}, build(() => el.remove()));
  const r = anchor.getBoundingClientRect();
  el.style.left = `${Math.min(r.left, innerWidth - 340)}px`;
  el.style.top = `${r.bottom + 4}px`;
  document.body.append(el);
  setTimeout(() => document.addEventListener("pointerdown", function off(e) {
    if (!el.contains(e.target)) { el.remove(); document.removeEventListener("pointerdown", off); }
  }));
  return el;
}

const bar = {
  device: h("button", {class: "crumb", title: "Devices"}),
  route: h("button", {class: "crumb", title: "Routes on this device"}),
  views: h("nav", {class: "views"}),
  tabs: h("nav", {class: "tabs"}),
};

function topBar() {
  const search = searchBox();
  bar.device.addEventListener("click", () => menu(bar.device, (close) => h("div", {class: "menu-list"},
    (state.devices || []).map((d) => h("a", {class: "menu-row", href: `/${d.dongle_id}`, onclick: (e) => { close(); link(`/${d.dongle_id}`)(e); },
      onpointerenter: () => prefetchDevice(d.dongle_id)},
      h("span", {class: `dot ${isOnline(d) ? "online" : ""}`}), h("span", {}, d.alias || d.dongle_id), h("span", {class: "muted mono"}, d.dongle_id.slice(0, 6)))),
    !state.devices?.length && h("div", {class: "menu-row muted"}, auth.isAuthenticated() ? "No devices" : "Sign in to see your devices"))));
  bar.route.addEventListener("click", () => {
    if (!ws) return;
    const pop = menu(bar.route, () => h("div", {class: "route-pop"}));
    pop.firstChild.append(routesPane({route: ws.route, openRoute: (name) => { pop.remove(); ws.openRoute(name); }}).el);
  });
  const account = h("button", {class: "tbtn", title: "Account"}, "●");
  account.addEventListener("click", () => menu(account, (close) => h("div", {class: "menu-list"},
    state.me && h("div", {class: "menu-row muted"}, state.me.username || state.me.email || "signed in"),
    auth.isAuthenticated()
      ? h("button", {class: "menu-row", onclick: () => { close(); auth.signOut(); }}, "Sign out")
      : h("a", {class: "menu-row", href: loginPath(), onclick: (e) => { close(); link(loginPath())(e); }}, "Sign in"))));
  const themeBtn = h("button", {class: "tbtn", title: "Light/dark (T)", onclick: () => theme.toggle()}, "◐");
  return h("header", {class: "topbar"},
    h("a", {class: "brand", href: "/", onclick: link("/"), title: "YCPE"}, logo(20), h("span", {}, "ycpe")),
    bar.device, h("span", {class: "sep"}, "›"), bar.route, search, bar.views, bar.tabs,
    h("span", {class: "grow"}), loadPill(), hud(footprint()), themeBtn, account);
}

function updateTop() {
  const dongle = ws?.dongle || location.pathname.split("/")[1];
  const device = state.devices?.find((d) => d.dongle_id === dongle);
  bar.device.textContent = device ? device.alias || dongle : DONGLE_RE.test(dongle || "") ? dongle : "devices";
  bar.route.hidden = bar.route.previousSibling.hidden = !ws;
  bar.views.hidden = bar.tabs.hidden = !ws;
  if (!ws) return;
  const start = new Date(ws.route.start_time_utc_millis);
  bar.route.textContent = `${start.toLocaleDateString(undefined, {month: "short", day: "numeric"})} ${start.toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit", hour12: false})}`;
  bar.route.title = `${ws.route.fullname} · ${[ws.route.platform, ws.route.version].filter(Boolean).join(" · ")}`;
  bar.views.replaceChildren(...Object.entries(VIEWS).map(([k, v], i) => h("button", {class: `view ${k === ws.view ? "on" : ""}`, title: `${v.title} (${i + 1})`, onclick: () => ws.setView(k)}, v.title)));
  const layout = ws.layout;
  if (!layout) return bar.tabs.replaceChildren();  // the view is still waiting on its files
  bar.tabs.replaceChildren(...layout.model.tabs.map((t, i) => h("button", {class: `tab ${i === layout.model.active ? "on" : ""}`,
    onclick: () => { layout.selectTab(i); updateTop(); },
    ondblclick: (e) => renameTab(e.currentTarget, i, t.name),
    oncontextmenu: (e) => { e.preventDefault(); tabMenu(e.currentTarget, i, t.name); },
    onauxclick: (e) => { if (e.button === 1) { layout.closeTab(i); updateTop(); } }}, t.name,
    layout.model.tabs.length > 1 && h("span", {class: "x", title: "Close tab", onclick: (e) => { e.stopPropagation(); layout.closeTab(i); updateTop(); }}, "×"))),
  h("button", {class: "tab add", title: "New tab", onclick: () => { layout.addTab(); updateTop(); }}, "+"));
}

function tabMenu(tabEl, i, name) {
  const layout = ws.layout;
  const item = (label, fn) => h("button", {class: "menu-row", onclick: () => { document.querySelector(".popover")?.remove(); fn(); updateTop(); }}, label);
  menu(tabEl, () => h("div", {class: "menu-list"},
    item("Rename", () => setTimeout(() => renameTab(bar.tabs.children[i], i, name))),
    item("Duplicate", () => layout.addTab(`${name} copy`, structuredClone(layout.save().tabs[i].root))),
    layout.model.tabs.length > 1 && item("Close", () => layout.closeTab(i)),
    item(`Reset ${VIEWS[ws.view].title} layout`, () => ws.resetLayout())));
}

// double-click a tab to rename it in place
function renameTab(tabEl, i, name) {
  const input = h("input", {class: "tab-edit", value: name});
  const done = (save) => {
    if (save && input.value.trim()) ws.layout.renameTab(i, input.value.trim());
    updateTop();
  };
  input.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") done(true); else if (e.key === "Escape") done(false); });
  input.addEventListener("blur", () => done(true));
  tabEl.replaceChildren(input);
  input.select();
}

// ---- route workspace: one clock, one store, one layout per view ----

class Workspace {
  constructor(route, files, view, {t = 0, repeat = 1}) {
    this.route = route;
    this.files = null;  // each segment's log from the route's source (data/sources.js); the drive view mounts without it
    this.files = null;  // the file list only matters once logs load; the drive view mounts without it
    this.filesReady = files.then((f) => { this.files = f; });
    this.repeat = repeat;
    this.store = null;
    this.storeSubs = new Set();
    this.layouts = {};
    this.clock = createClock(Math.max(1, (route.end_time_utc_millis - route.start_time_utc_millis) / 1000 * repeat));
    this.clock.seek(t);
    this.main = h("div", {class: "workspace"});
    this.el = h("div", {class: "ws"}, this.main, playbar(this.ctx()));
    this.offs = [this.clock.on("t", () => this.syncUrl()), this.clock.on("play", () => this.syncUrl())];
    this.setView(view);
  }

  ctx() {
    const self = this;
    return {
      route: this.route, clock: this.clock, perf, theme,
      get store() { return self.ensureStore(); },
      peekStore: () => this.store,
      onStore: (fn) => { this.storeSubs.add(fn); if (this.store) fn(this.store); },
      openRoute: (name) => this.openRoute(name),
    };
  }

  // logs load only once something needs them (the drive view doesn't); null until the file list is in
  ensureStore() {
    if (this.store || !this.files) return this.store;
    this.store = new Store(this.route.fullname, this.files, {t: this.clock.t, repeat: this.repeat});
    // the store reprioritizes and reports buffering as the playhead moves; it publishes its own load status
    this.offs.push(this.clock.on("t", () => this.store.focus(this.clock.t)));
    this.store.on("progress", () => {
      if (this.store.route.duration) this.clock.setDuration(this.store.route.duration);
    });
    for (const fn of this.storeSubs) fn(this.store);
    return this.store;
  }

  get layout() { return this.layouts[this.view]; }

  setView(view) {
    if (!VIEWS[view]) view = "drive";
    this.view = view;
    this.el.dataset.view = view;
    if (!this.layouts[view]) {
      const saved = JSON.parse(localStorage.getItem(`ycpe_layout_${view}`) || "null");
      const model = saved?.tabs?.length ? saved : VIEWS[view].layout;
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

  resetLayout() {
    localStorage.removeItem(`ycpe_layout_${this.view}`);
    this.layout.dispose();
    delete this.layouts[this.view];
    this.setView(this.view);
  }

  openRoute(name) {
    go(`/${name.replace("|", "/")}/${this.view}`);
  }

  syncUrl(now = false) {
    clearTimeout(this.urlTimer);
    const write = () => {
      const q = new URLSearchParams(location.search);
      q.set("t", this.clock.t.toFixed(1));
      if (this.clock.t === 0) q.delete("t");
      const path = `/${this.route.fullname.replace("|", "/")}/${this.view}`;
      history.replaceState(null, "", `${path}${q.size ? `?${q}` : ""}`);
    };
    if (now) write(); else this.urlTimer = setTimeout(write, 400);
  }

  dispose() {
    for (const off of this.offs) off();
    this.clock.pause();
    for (const l of Object.values(this.layouts)) l.dispose();
    this.store?.dispose();
  }
}

const usesData = (model) => JSON.stringify(model.tabs).match(/"pane":"([^"]+)"/g)?.some((m) => PANES[m.slice(8, -1)]?.data);

// ---- keyboard ----

document.addEventListener("keydown", (e) => {
  if (e.target.closest?.("input, textarea, select, [contenteditable]") || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "/") { e.preventDefault(); document.querySelector(".omnibox").focus(); return; }
  if (e.key === "t") return theme.toggle();
  if (e.key === "?") return help();
  if (!ws) return;
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

const SHORTCUTS = [
  ["space", "play / pause"], ["← →", "seek 1 s (shift: 10 s)"], [", .", "step 50 ms"], ["[ ]", "slower / faster"],
  ["l", "loop the visible range"], ["f", "show the whole route"], ["Home End", "start / end"], ["1 2 3", "Drive / Plot / CAN"],
  ["/", "search routes and devices"], ["t", "light / dark"], ["?", "this help"],
  ["drag a pane header", "move the pane (edges dock, center swaps)"], ["double-click a divider", "even out the split"],
  ["wheel on the timeline", "zoom the visible range"],
];

function help() {
  if (document.querySelector(".help")) return document.querySelector(".help").remove();
  const el = h("div", {class: "help", onclick: () => el.remove()},
    h("div", {class: "help-card", onclick: (e) => e.stopPropagation()},
      h("b", {}, "Keyboard and mouse"),
      h("dl", {}, SHORTCUTS.map(([k, v]) => [h("dt", {}, k.split(" ").map((x) => h("kbd", {}, x))), h("dd", {}, v)]))));
  document.body.append(el);
  addEventListener("keydown", function off(e) { if (e.key === "Escape") { el.remove(); removeEventListener("keydown", off); } });
}

// ---- routing ----

let navId = 0;
async function render() {
  const id = ++navId;
  const main = document.getElementById("main");
  const parts = location.pathname.split("/").filter(Boolean);
  const q = new URLSearchParams(location.search);
  document.querySelector(".popover")?.remove();

  if (parts[0] === "auth" || parts[0] === "login") {
    ws?.dispose();
    ws = null;
    document.body.classList.remove("in-ws");
    updateTop();
    return main.replaceChildren((parts[0] === "auth" ? authPage : loginPage)(q));
  }
  if (auth.isAuthenticated() && !state.devices) {
    [state.me, state.devices] = await Promise.all([api.me().catch(() => null), api.devices().catch(() => [])]);
  }

  const routeName = parts.length >= 2 && DONGLE_RE.test(parts[0]) ? `${parts[0]}|${parts[1]}` : null;
  if (routeName) {
    if (ws?.route.fullname === routeName) {
      if (parts[2] && parts[2] !== ws.view) ws.setView(parts[2]);
      return;
    }
    document.body.classList.add("loading");
    try {
      const files = routeLogs(routeName);
      files.catch(() => {});
      const route = await api.route(routeName);
      if (id !== navId) return;
      ws?.dispose();
      ws = new Workspace(route, files, parts[2] || "drive", {t: +q.get("t") || 0, repeat: Math.max(1, +q.get("repeat") || 1)});
      rememberRoute(route);
      document.body.classList.add("in-ws");
      main.replaceChildren(ws.el);
    } catch (e) {
      main.replaceChildren(errorPage(e));
    } finally {
      document.body.classList.remove("loading");
    }
    return updateTop();
  }

  ws?.dispose();
  ws = null;
  document.body.classList.remove("in-ws");
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

function errorPage(e) {
  console.error(e);
  if (e.status === 401 || e.status === 403) return signInCard();
  const msg = e.status === 403 ? "You don't have access to this." : e.status === 404 ? "Not found." : String(e.message || e);
  return h("div", {class: "page-msg"}, h("h2", {}, msg), h("p", {class: "muted"}, "Try a route name or dongle id in the box above."));
}

document.querySelector("#app > .topbar").replaceWith(topBar());
window.addEventListener("popstate", render);
// signing in or out re-renders the current page under the new account; /auth navigates on by itself (rendering it
// again would spend the one-time code twice)
auth.on(() => {
  state.me = state.devices = null;
  if (location.pathname.startsWith("/auth")) return;
  ws?.dispose();
  ws = null;
  render();
});
render();
