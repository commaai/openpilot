import {api, auth} from "/api.js";
import {h, clear} from "/dom.js";
import {icon} from "/icons.js";
import {devicePage} from "/pages/device.js";
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
      onclick: (e) => { setDrawer(false); link(`/${d.dongle_id}`)(e); }},
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

// the old page stays up while the next one loads; a progress bar shows instead of a "Loading…" flash
let navId = 0;
async function render() {
  const id = ++navId;
  const main = document.getElementById("main");
  const path = location.pathname.replace(/\/+$/, "") || "/";
  document.body.classList.add("loading");
  setDrawer(false);
  let page;
  try {
    if (path === "/auth") return main.replaceChildren(await authCallback());
    if (auth.isAuthenticated() && !state.me) {
      [state.me, state.devices] = await Promise.all([api.me().catch(() => null), api.devices().catch(() => [])]);
    }
    const parts = path.split("/").filter(Boolean);
    const dongleId = DONGLE_RE.test(parts[0] || "") ? parts[0] : null;
    sidebar(dongleId);

    if (dongleId && parts.length === 2) page = await routePage(`${dongleId}|${parts[1]}`, dongleId, state.devices);
    else if (dongleId) page = await devicePage(dongleId, state.devices);
    else if (!auth.isAuthenticated()) page = loginPage();
    else if (state.devices.length) { history.replaceState(null, "", `/${state.devices[0].dongle_id}`); return render(); }
    else page = h("div", {class: "page"}, h("h2", {}, "No devices"), h("p", {class: "muted"}, "Pair a device from its setup screen."));
  } catch (e) {
    console.error(e);
    page = h("p", {class: "error center"}, e.status === 403 ? "You don't have access to this." : e.status === 404 ? "Not found." : String(e));
  } finally {
    if (id === navId) document.body.classList.remove("loading");
  }
  if (id !== navId) return;  // superseded by a newer navigation
  page.classList.add("enter");
  main.replaceChildren(page);
  window.scrollTo(0, 0);
}

document.getElementById("header").append(...header().filter(Boolean));
document.getElementById("backdrop").addEventListener("click", () => setDrawer(false));
window.addEventListener("popstate", render);
render();
