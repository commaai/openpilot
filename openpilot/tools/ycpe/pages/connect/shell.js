// user mode's frame, as connect's explorer: its AppHeader (menu or logo, "connect", then the referral gift and the
// account menu, with Developer mode before them) and, beside the page, its DeviceList (a drawer from the menu button up
// to 1080px wide, as connect's). The pages inside are the route browser's own; the other tracks' pairing, prime and clips
// mount where connect shows them, once their modules are there.
import {h, css, fmt} from "../../dom.js";
import {auth} from "../../api.js";
import {daemon} from "../../daemon.js";
import {icon} from "../../icons.js";
import {REFERRALS_SEEN, go, link, onlineMark, switchMode} from "../../app.js";
import {loginPath} from "../login.js";
import {prefetchDevice} from "../device.js";
import {cnIcon} from "./icons.js";

await css(new URL("./connect.css", import.meta.url));

// another track's export called with args, or null while its module isn't merged
export const optional = (path, name, ...args) => import(path).then((m) => m[name]?.(...args) ?? null, () => null);

// connect's own mark, as it draws it
const commaLogo = () => h("img", {src: new URL("../../icons/comma-white.png", import.meta.url), alt: "comma", width: 18.9, height: 34});

const LAST_KEY = "ycpe_cn_device";  // the device shown last, which /connect opens on
export const lastDevice = (devices) => devices.find((d) => d.dongle_id === localStorage.getItem(LAST_KEY)) || devices[0];

// connect's order: the account's own devices first, then named ones by name, then the rest by dongle id
const order = (a, b) => (b.is_owner ? 1 : 0) - (a.is_owner ? 1 : 0) || (a.alias && b.alias ? a.alias.localeCompare(b.alias)
  : !a.alias && !b.alias ? a.dongle_id.localeCompare(b.dongle_id) : !!b.alias - !!a.alias);

const list = h("div", {class: "cn-list"});
const side = h("nav", {class: "cn-side", "aria-label": "Devices"},
  h("a", {class: "cn-side-head", href: "/connect", onclick: link("/connect")}, commaLogo(), h("span", {}, "connect")), list);
const scrim = h("div", {class: "cn-scrim", onclick: () => drawer(false)});
const content = h("div", {class: "cn-content"});
const el = h("div", {class: "cn-shell"}, side, scrim, content);

const menu = h("button", {class: "cn-hbtn cn-menu", "aria-label": "menu", "aria-expanded": "false", onclick: () => drawer(!el.classList.contains("open"))}, icon("menu", 24));
const logoLink = h("a", {class: "cn-logo", href: "/connect"}, commaLogo());
const brand = h("a", {class: "cn-brand", href: "/connect"}, "connect");
for (const a of [logoLink, brand]) a.addEventListener("click", (e) => link(a.getAttribute("href"))(e));
const dot = h("span", {class: "cn-gift-dot", "aria-label": "New referrals"});
const gift = h("a", {class: "cn-hbtn cn-gift", href: "/connect/referrals", "aria-label": "referrals", onclick: (e) => {
  localStorage.setItem(REFERRALS_SEEN, "1");
  dot.hidden = true;
  link(location.pathname === "/connect/referrals" ? brand.getAttribute("href") : "/connect/referrals")(e);
}}, cnIcon("giftOpen", 28), icon("gift", 28), dot);
const account = h("button", {class: "cn-hbtn cn-account", "aria-label": "account menu", "aria-haspopup": "true", "aria-expanded": "false", onclick: () => accountMenu()},
  cnIcon("account", 34));
const accountWrap = h("div", {class: "cn-account-wrap"}, account);
export const header = h("header", {class: "cn-header"},
  h("div", {class: "cn-title"}, menu, logoLink, brand),
  h("div", {class: "cn-actions"},
    h("button", {class: "cn-devmode", title: "Developer mode: this place with plots, CAN, files and logs", "aria-label": "Developer mode", onclick: switchMode},
      icon("code", 18), h("span", {}, "Developer mode")),
    gift, accountWrap));

// connect's AccountMenu: who's signed in and the build, then referrals, useradmin and logging out. Signed out, the
// button goes to sign in, as connect's goes to its home page
async function accountMenu() {
  if (closeMenu()) return;
  if (!auth.isAuthenticated()) return go(loginPath());
  const me = await auth.user();
  if (!me || accountWrap.querySelector(".cn-menu-panel")) return;
  const close = () => closeMenu();
  const c = daemon.caps?.commit;
  const row = (attrs, text) => h(attrs.href ? "a" : "button", {class: "cn-menu-row", role: "menuitem", ...attrs}, text);
  const panel = h("div", {class: "cn-menu-panel", role: "menu"},
    h("div", {class: "cn-menu-head"},
      h("b", {}, me.email || me.username || ""),
      h("span", {}, me.user_id),
      h("span", {}, "Version: ", c ? [h("a", {href: `https://github.com/commaai/openpilot/commit/${c.sha}`, target: "_blank", rel: "noreferrer"}, c.sha.slice(0, 7)),
        `, ${fmt.ago(c.time)}`] : "dev")),
    h("div", {class: "cn-menu-line"}),
    row({href: "/connect/referrals", onclick: (e) => { close(); link("/connect/referrals")(e); }}, "Referrals"),
    row({href: "https://useradmin.comma.ai/", target: "_blank", rel: "noreferrer", onclick: close}, "Manage account"),
    row({onclick: () => { close(); auth.signOut(); }}, "Log out"));
  accountWrap.append(h("div", {class: "cn-menu-scrim", onclick: close}), panel);
  account.setAttribute("aria-expanded", "true");
}
function closeMenu() {
  const open = accountWrap.querySelectorAll(".cn-menu-scrim, .cn-menu-panel");
  for (const e of open) e.remove();
  account.setAttribute("aria-expanded", "false");
  return open.length > 0;
}
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && closeMenu()) account.focus(); });

// the drawer is modal, as connect's: the page under it is inert until a tap beside it or Escape
function drawer(open) {
  el.classList.toggle("open", open);
  menu.setAttribute("aria-expanded", open);
  content.inert = header.inert = open;
  if (open) list.querySelector("a")?.focus({preventScroll: true});
}
side.addEventListener("keydown", (e) => { if (e.key === "Escape" && el.classList.contains("open")) { drawer(false); menu.focus(); } });

let shown = null;  // the devices and selection the list was drawn for
// the page's container, with the device list for the account's devices (none: the page has the width, as connect's
// sign-in and "no devices" pages do) and dongle selected
export function mount(main, {devices, dongle, signedIn}) {
  if (el.parentElement !== main) main.replaceChildren(el);
  drawer(false);
  closeMenu();
  if (dongle) localStorage.setItem(LAST_KEY, dongle);
  const bare = !devices?.length;
  el.classList.toggle("bare", bare);
  header.classList.toggle("bare", bare);
  for (const a of [logoLink, brand]) a.setAttribute("href", `/connect${dongle ? `/${dongle}` : ""}`);
  gift.hidden = !signedIn;
  dot.hidden = !!localStorage.getItem(REFERRALS_SEEN);
  gift.classList.toggle("on", location.pathname === "/connect/referrals");
  if (shown?.devices !== devices || shown?.dongle !== dongle) {
    const drawn = shown = {devices, dongle};
    list.replaceChildren(...[...devices || []].sort(order).map((d) => deviceRow(d, d.dongle_id === dongle)));
    // connect's "add new device", from pages/pair.js
    if (signedIn) optional("/pages/pair.js", "pairButton", "add new device", true).then((b) => {
      if (!b || shown !== drawn) return;
      list.append(h("div", {class: "cn-add"}, b));
    });
  }
  return content;
}

// a device: online dot, name and dongle id; its owner (or a superuser) gets connect's settings gear, which opens its
// settings modal (pages/settings.js; the device page's settings tab when opened in a new tab)
function deviceRow(d, on) {
  const path = `/connect/${d.dongle_id}`, settings = `${path}?tab=settings`;
  const row = h("div", {class: `cn-dev ${on ? "on" : ""}`},
    h("a", {href: path, onclick: link(path), onpointerenter: () => prefetchDevice(d.dongle_id), "aria-current": on ? "page" : null},
      onlineMark(d), h("span", {class: "cn-dev-name"}, h("b", {}, fmt.deviceName(d)), h("span", {}, d.dongle_id))));
  const gear = (ok) => ok && row.append(h("a", {class: "cn-hbtn cn-gear", href: settings, "aria-label": "device settings", onclick: (e) => {
    if (e.metaKey || e.ctrlKey || e.button !== 0) return;
    e.preventDefault();
    optional("/pages/settings.js", "settingsModal", d);
  }}, cnIcon("settings", 24)));
  if (d.is_owner) gear(true); else auth.user().then((me) => gear(!!me?.superuser));
  return row;
}
