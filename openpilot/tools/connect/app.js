import {api, auth} from "/api.js";
import {h, clear} from "/dom.js";
import {devicesPage} from "/pages/devices.js";
import {devicePage} from "/pages/device.js";
import {routePage} from "/pages/route.js";

const ROUTE_RE = /^([0-9a-f]{16})[|/_]([0-9a-f]+--[0-9a-f]+)$/i;
const DONGLE_RE = /^[0-9a-f]{16}$/i;

// a route name typed or pasted anywhere lands on its page; a dongle id on its device
export function navigateOnebox(text) {
  const s = text.trim();
  const m = s.match(ROUTE_RE);
  if (m) return go(`/${m[1]}/${m[2]}`);
  if (DONGLE_RE.test(s)) return go(`/${s}`);
  alert("Enter a route name (dongle|time) or a dongle id");
}

export function go(path) {
  history.pushState(null, "", path);
  render();
}

function header(me) {
  const box = h("input", {type: "search", placeholder: "route name or dongle id", onkeydown: (e) => { if (e.key === "Enter") navigateOnebox(e.target.value); }});
  return [
    h("a", {href: "/", class: "brand", onclick: link("/")}, "comma connect"),
    h("form", {class: "onebox", onsubmit: (e) => { e.preventDefault(); navigateOnebox(box.value); }}, box),
    me ? h("span", {class: "user"}, me.username, " ", h("a", {href: "#", onclick: (e) => { e.preventDefault(); auth.logOut(); }}, "sign out")) : null,
  ];
}

export function link(path) {
  return (e) => {
    if (e.metaKey || e.ctrlKey || e.button !== 0) return;
    e.preventDefault();
    go(path);
  };
}

function loginPage() {
  return h("div", {class: "login"},
    h("h1", {}, "comma connect"),
    ["google", "github", "apple"].map((p) => h("a", {class: "button", href: auth.loginUrl(p)}, `Sign in with ${p[0].toUpperCase() + p.slice(1)}`)),
    h("p", {class: "muted"}, "Or open a public route by name above."));
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

let me = null;

async function render() {
  const main = clear(document.getElementById("main"));
  const path = location.pathname.replace(/\/+$/, "") || "/";
  main.append(h("p", {class: "muted"}, "loading…"));
  try {
    if (path === "/auth") return main.replaceChildren(await authCallback());
    if (auth.isAuthenticated() && !me) me = await api.me().catch(() => null);
    clear(document.getElementById("header")).append(...header(me).filter(Boolean));

    const parts = path.split("/").filter(Boolean);
    let page;
    if (parts.length === 2 && DONGLE_RE.test(parts[0])) page = await routePage(`${parts[0]}|${parts[1]}`, parts[0]);
    else if (parts.length === 1 && DONGLE_RE.test(parts[0])) page = await devicePage(parts[0]);
    else if (!auth.isAuthenticated()) page = loginPage();
    else page = await devicesPage();
    main.replaceChildren(page);
  } catch (e) {
    console.error(e);
    main.replaceChildren(h("p", {class: "error"}, e.status === 403 ? "You don't have access to this." : e.status === 404 ? "Not found." : String(e)));
  }
}

window.addEventListener("popstate", render);
render();
