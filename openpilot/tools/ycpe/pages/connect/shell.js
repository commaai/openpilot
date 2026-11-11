// user mode's frame, as connect's explorer: its AppHeader (menu or logo, "connect", then the referral gift and the
// account menu, with Developer mode before them) and, beside the page, its DeviceList (a drawer from the menu button up
// to 1080px wide, as connect's). The pages inside are the route browser's own; pairing, prime and clips mount where
// connect shows them.
import {h, css, fmt} from "../../dom.js";
import {USERADMIN, auth} from "../../api.js";
import {commitLink} from "../../daemon.js";
import {icon} from "../../icons.js";
import {REFERRALS_SEEN, go, link, onlineMark, switchMode} from "../../app.js";
import {loginPath} from "../login.js";
import {prefetchDevice} from "../device.js";
import {onDeviceChange} from "../../devices.js";
import {cnIcon} from "./icons.js";
import {cnMenu} from "../cnui.js";

await css(new URL("./connect.css", import.meta.url));

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
export const header = h("header", {class: "cn-header"},
  h("div", {class: "cn-title"}, menu, logoLink, brand),
  h("div", {class: "cn-actions"},
    h("button", {class: "cn-devmode", title: "Developer mode: this place with plots, CAN, files and logs", "aria-label": "Developer mode", onclick: switchMode,
      onpointerenter: () => shown?.dongle && prefetchDevice(shown.dongle, false)},
      icon("code", 18), h("span", {}, "Developer mode")),
    gift, account));

// connect's AccountMenu: who's signed in and the build, then referrals, useradmin and logging out, under the button's
// right end (cnui.js cnMenu). Signed out, the button goes to sign in, as connect's goes to its home page
let accountOpen = null;  // the menu, while it's open
async function accountMenu() {
  if (!auth.isAuthenticated()) return go(loginPath());
  const me = await auth.user();
  if (!me || accountOpen) return;
  const close = () => accountOpen?.close();
  const row = (attrs, text) => h(attrs.href ? "a" : "button", {class: "cn-menu-row", role: "menuitem", ...attrs}, text);
  accountOpen = cnMenu(account, "cn-menu-panel", [
    h("div", {class: "cn-menu-head"},
      h("b", {}, me.email || me.username || ""),
      h("span", {}, me.user_id),
      h("span", {}, "Version: ", commitLink({rel: "noreferrer"}) ?? "dev")),
    h("div", {class: "cn-menu-line"}),
    row({href: "/connect/referrals", onclick: (e) => { close(); link("/connect/referrals")(e); }}, "Referrals"),
    row({href: USERADMIN, target: "_blank", rel: "noreferrer", onclick: close}, "Manage account"),
    row({onclick: () => { close(); auth.signOut(); }}, "Log out")],
  (a, r) => [a.right - r.width, a.bottom + 4, 0, 0], {margin: 0, grow: false, onclose: () => { accountOpen = null; }});
  accountOpen?.paper.setAttribute("role", "menu");
}

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
  accountOpen?.close();
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
    if (signedIn) import("/pages/pair.js").then(({pairButton}) => {
      if (shown === drawn) list.append(h("div", {class: "cn-add"}, pairButton("add new device", true)));
    });
  }
  return content;
}

// a device renamed (in either mode's settings): its row named anew, and its dashboard if it's shown
onDeviceChange((dongle, left) => {
  const d = shown?.devices?.find((d) => d.dongle_id === dongle);
  if (left || !d) return;
  for (const b of list.querySelectorAll(`a[href="/connect/${dongle}"] .cn-dev-name > b`)) b.textContent = fmt.deviceName(d);
  if (location.pathname === `/connect/${dongle}`) go(location.pathname + location.search, true);
});

// a device: online dot, name and dongle id; its owner (or a superuser) gets connect's settings gear, which opens its
// settings modal (pages/settings.js; the device page's settings tab when opened in a new tab)
function deviceRow(d, on) {
  const path = `/connect/${d.dongle_id}`, settings = `${path}?tab=settings`;
  const row = h("div", {class: `cn-dev ${on ? "on" : ""}`},
    h("a", {href: path, onclick: link(path), onpointerenter: () => prefetchDevice(d.dongle_id), "aria-current": on ? "page" : null},
      onlineMark(d), h("span", {class: "cn-dev-name"}, h("b", {}, fmt.deviceName(d)), h("span", {}, d.dongle_id))));
  const gear = (ok) => ok && row.append(h("a", {class: "cn-hbtn cn-gear", href: settings, "aria-label": "device settings",
    onclick: link(settings, () => import("/pages/settings.js").then((m) => m.settingsModal(d)))}, cnIcon("settings", 24)));
  if (d.is_owner) gear(true); else auth.user().then((me) => gear(!!me?.superuser));
  return row;
}
