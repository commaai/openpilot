import {api, auth} from "/api.js";
import {h, clear} from "/dom.js";
import {icon} from "/icons.js";
import {devicePage, prefetchDevice} from "/pages/device.js";
import {routePage} from "/pages/route.js";

const ROUTE_RE = /^([0-9a-f]{16})[|/_]([0-9a-f]+--[0-9a-f]+)$/i;
const DONGLE_RE = /^[0-9a-f]{16}$/i;

export function go(path) {
  history.pushState(null, "", path);
  render();
}

export function link(path) {
  return (e) => {
    if (e.metaKey || e.ctrlKey || e.button !== 0) return;
    e.preventDefault();
    go(path);
  };
}

// a route name or dongle id typed anywhere lands on its page
function onebox(text) {
  const s = text.trim();
  const m = s.match(ROUTE_RE);
  if (m) return go(`/${m[1]}/${m[2]}`);
  if (DONGLE_RE.test(s)) return go(`/${s}`);
  alert("Enter a route name (dongle|time) or a dongle id");
}

export const isOnline = (d) => d.last_athena_ping && Date.now() / 1000 - d.last_athena_ping < 120;

const state = {me: null, devices: null};

function setDrawer(open) {
  document.body.classList.toggle("drawer-open", open);
}

// the header is built once; the account menu is built on open so it always reflects auth state
function header() {
  const search = h("input", {type: "search", class: "onebox", placeholder: "route name or dongle id",
    onkeydown: (e) => { if (e.key === "Enter") { onebox(e.target.value); e.target.value = ""; } }});
  const account = h("div", {class: "account"});
  const closeMenu = () => account.querySelector(".menu")?.remove();
  const openMenu = () => account.append(h("div", {class: "menu"},
    state.me ? h("div", {class: "menu-item muted"}, state.me.username) : null,
    state.me ? h("div", {class: "menu-item", onclick: () => auth.logOut()}, "Sign out") : h("div", {class: "menu-item", onclick: () => go("/")}, "Sign in")));
  account.append(h("button", {class: "icon-button", "aria-label": "Account",
    onclick: (e) => { e.stopPropagation(); account.querySelector(".menu") ? closeMenu() : openMenu(); }}, icon("account", 28)));
  document.addEventListener("click", closeMenu);
  return [
    h("button", {class: "icon-button hamburger", "aria-label": "Menu", onclick: () => setDrawer(!document.body.classList.contains("drawer-open"))}, icon("menu")),
    h("a", {class: "brand", href: "/", onclick: link("/")}, h("img", {src: "/images/comma-white.png", alt: "comma"}), h("span", {}, "connect")),
    search,
    account,
  ];
}

let sidebarFor = null;
function sidebar(selected) {
  const nav = document.getElementById("sidebar");
  if (sidebarFor !== state.devices) {
    sidebarFor = state.devices;
    clear(nav).append(...(state.devices || []).map((d) => h("a", {class: "device-item", href: `/${d.dongle_id}`, "data-dongle": d.dongle_id,
      onclick: (e) => { setDrawer(false); link(`/${d.dongle_id}`)(e); }, onpointerenter: () => prefetchDevice(d.dongle_id)},
      h("span", {class: `dot ${isOnline(d) ? "online" : ""}`}),
      h("span", {}, h("div", {class: "name"}, d.alias || d.dongle_id), h("div", {class: "muted small"}, d.dongle_id)))));
  }
  for (const a of nav.children) a.classList.toggle("selected", a.dataset.dongle === selected);
  document.body.classList.toggle("has-sidebar", !!state.devices && state.devices.length > 0);
}

function loginPage() {
  return h("div", {class: "login"},
    h("img", {src: "/images/comma-white.png", alt: "comma", class: "login-logo"}),
    h("h1", {}, "connect"),
    ["google", "github", "apple"].map((p) => h("a", {class: "button wide", href: auth.loginUrl(p)}, `Sign in with ${p[0].toUpperCase() + p.slice(1)}`)),
    h("p", {class: "muted"}, "Or open a public route by name in the search box."));
}

async function authCallback() {
  const q = new URLSearchParams(location.search);
  try {
    await auth.exchangeCode(q.get("code"), q.get("provider"));
    history.replaceState(null, "", "/");
  } catch (e) {
    return h("p", {class: "error"}, String(e));
  }
  return render();
}

// built pages are kept (with their scroll position) so back/forward and revisits swap in instantly;
// the old page stays up while a new one loads, and the progress bar only appears if that takes a moment
const PAGE_CACHE = 8;
const PAGE_TTL = 5 * 60000;
const pages = new Map();
let current = null;
let navId = 0;

function cachePage(path, el) {
  pages.delete(path);
  pages.set(path, {el, scrollY: 0, time: Date.now()});
  for (const [k, v] of pages) {
    if (pages.size <= PAGE_CACHE) break;
    pages.delete(k);
    v.el.cleanup?.();
  }
  return pages.get(path);
}

function show(entry, fresh) {
  if (current && current !== entry) {
    current.scrollY = window.scrollY;
    for (const v of current.el.querySelectorAll("video")) v.pause();
  }
  current = entry;
  entry.el.classList.toggle("enter", fresh);
  document.getElementById("main").replaceChildren(entry.el);
  window.scrollTo(0, entry.scrollY);
  for (const m of entry.el.querySelectorAll(".device-map, .drive-map")) m._map?.resize();
}

async function render() {
  const id = ++navId;
  const main = document.getElementById("main");
  const path = location.pathname.replace(/\/+$/, "") || "/";
  setDrawer(false);
  const parts = path.split("/").filter(Boolean);
  const dongleId = DONGLE_RE.test(parts[0] || "") ? parts[0] : null;

  const cached = pages.get(path);
  if (cached && Date.now() - cached.time < PAGE_TTL) {
    sidebar(dongleId);
    return show(cached, false);
  }

  const slow = setTimeout(() => document.body.classList.add("loading"), 150);
  let page, cacheable = false;
  try {
    if (path === "/auth") return main.replaceChildren(await authCallback());
    if (auth.isAuthenticated() && !state.me) {
      [state.me, state.devices] = await Promise.all([api.me().catch(() => null), api.devices().catch(() => [])]);
    }
    sidebar(dongleId);
    if (dongleId && parts.length === 2) page = await routePage(`${dongleId}|${parts[1]}`, dongleId, state.devices);
    else if (dongleId) page = await devicePage(dongleId, state.devices);
    else if (!auth.isAuthenticated()) page = loginPage();
    else if (state.devices.length) { history.replaceState(null, "", `/${state.devices[0].dongle_id}`); return render(); }
    else page = h("div", {class: "page"}, h("h2", {}, "No devices"), h("p", {class: "muted"}, "Pair a device from its setup screen."));
    cacheable = !!dongleId;
  } catch (e) {
    console.error(e);
    page = h("p", {class: "error center"}, e.status === 403 ? "You don't have access to this." : e.status === 404 ? "Not found." : String(e));
  } finally {
    clearTimeout(slow);
    if (id === navId) document.body.classList.remove("loading");
  }
  if (id !== navId) { page.cleanup?.(); return; }  // superseded by a newer navigation
  show(cacheable ? cachePage(path, page) : {el: page, scrollY: 0}, true);
}

history.scrollRestoration = "manual";
document.getElementById("header").append(...header().filter(Boolean));
document.getElementById("backdrop").addEventListener("click", () => setDrawer(false));
window.addEventListener("popstate", render);
render();
