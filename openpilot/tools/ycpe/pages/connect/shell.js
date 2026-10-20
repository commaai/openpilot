// user mode's frame, as connect's explorer: its header (menu, logo, Developer mode, referrals, account) and, beside the
// page, the device list (a drawer from the menu button on a narrower window). The pages inside are the route browser's
// own; the other tracks' pairing, prime and clips mount where connect shows them, once their modules are there.
import {h, css, fmt} from "../../dom.js";
import {icon, logo} from "../../icons.js";
import {REFERRALS_SEEN, accountButton, link, onlineMark, switchMode} from "../../app.js";
import {prefetchDevice} from "../device.js";
import {canManage} from "../home.js";

await css(new URL("./connect.css", import.meta.url));

// another track's export called with args, or null while its module isn't merged
export const optional = (path, name, ...args) => import(path).then((m) => m[name]?.(...args) ?? null, () => null);

export const wide = matchMedia("(min-width: 761px)");  // not a phone: /connect opens on a device, as connect does
const LAST_KEY = "ycpe_cn_device";  // the device shown last, which /connect opens on
export const lastDevice = (devices) => devices.find((d) => d.dongle_id === localStorage.getItem(LAST_KEY)) || devices[0];

const side = h("nav", {class: "cn-side", "aria-label": "Devices"});
const scrim = h("div", {class: "cn-scrim", onclick: () => drawer(false)});
const content = h("div", {class: "cn-content"});
const el = h("div", {class: "cn-shell"}, side, scrim, content);

const menu = h("button", {class: "cn-hbtn cn-menu", "aria-label": "Devices", "aria-expanded": "false", onclick: () => drawer(!el.classList.contains("open"))}, icon("menu", 24));
const brand = h("a", {class: "cn-brand", href: "/connect"}, logo(34), h("span", {}, "connect"));
brand.addEventListener("click", (e) => link(brand.getAttribute("href"))(e));
const dot = h("span", {class: "cn-gift-dot", "aria-label": "New referrals"});
const gift = h("a", {class: "cn-hbtn", href: "/connect/referrals", title: "Refer a friend", "aria-label": "Referrals",
  onclick: (e) => link(location.pathname === "/connect/referrals" ? brand.getAttribute("href") : "/connect/referrals")(e)}, icon("gift", 28), dot);
export const header = h("header", {class: "cn-header"}, menu, brand, h("span", {class: "grow"}),
  h("button", {class: "cn-devmode", title: "Developer mode: this place with plots, CAN, files and logs", "aria-label": "Developer mode", onclick: switchMode},
    icon("code", 18), h("span", {}, "Developer mode")),
  gift, accountButton("cn-hbtn"));

// the drawer is modal, as connect's: the page under it is inert until a tap beside it or Escape
function drawer(open) {
  el.classList.toggle("open", open);
  menu.setAttribute("aria-expanded", open);
  content.inert = header.inert = open;
  if (open) side.querySelector("a")?.focus();
}
side.addEventListener("keydown", (e) => { if (e.key === "Escape" && el.classList.contains("open")) { drawer(false); menu.focus(); } });

let shown = null;  // the devices and selection the list was drawn for
// the page's container, with the device list for the account's devices (none: the page has the width, as connect's
// sign-in and "no devices" pages do) and dongle selected
export function mount(main, {devices, dongle, signedIn}) {
  if (el.parentElement !== main) main.replaceChildren(el);
  drawer(false);
  if (dongle) localStorage.setItem(LAST_KEY, dongle);
  const bare = !devices?.length;
  el.classList.toggle("bare", bare);
  menu.hidden = bare;
  brand.setAttribute("href", `/connect${dongle ? `/${dongle}` : ""}`);
  gift.hidden = !signedIn;
  dot.hidden = !!localStorage.getItem(REFERRALS_SEEN);
  gift.classList.toggle("on", location.pathname === "/connect/referrals");
  if (shown?.devices !== devices || shown?.dongle !== dongle) {
    const list = shown = {devices, dongle};
    side.replaceChildren(h("a", {class: "cn-brand cn-side-head", href: "/connect", onclick: link("/connect")}, logo(34), h("span", {}, "connect")),
      ...(devices || []).map((d) => deviceRow(d, d.dongle_id === dongle)));
    // connect's "add new device", from pages/pair.js
    if (signedIn) optional("/pages/pair.js", "pairButton").then((b) => b && shown === list && side.append(h("div", {class: "cn-add"}, b)));
  }
  return content;
}

// a device: online dot, name and dongle id; its owner gets connect's settings gear (the device page's settings tab)
function deviceRow(d, on) {
  const path = `/connect/${d.dongle_id}`, settings = `${path}?tab=settings`;
  const row = h("div", {class: `cn-dev ${on ? "on" : ""}`},
    h("a", {href: path, onclick: link(path), onpointerenter: () => prefetchDevice(d.dongle_id), "aria-current": on ? "page" : null},
      onlineMark(d), h("span", {class: "cn-dev-name"}, h("b", {}, fmt.deviceName(d)), h("span", {class: "mono"}, d.dongle_id))));
  canManage(d).then((ok) => ok && row.append(h("a", {class: "cn-hbtn cn-gear", href: settings, onclick: link(settings), title: "Device settings",
    "aria-label": `${fmt.deviceName(d)} settings`}, icon("settings", 22))));
  return row;
}

// the device page as connect's dashboard: prime's banner over it (the shown device first in the list it's given) and
// the device's clips above its tabs, each once per page (a page kept for going back already has them)
export function dashboard(page, device, devices) {
  if (page.dataset.cnExtras) return;
  page.dataset.cnExtras = "1";
  optional("/pages/prime.js", "primeBanner", [device, ...(devices || []).filter((d) => d !== device)]).then((b) => b && page.prepend(b));
  optional("/panes/clips.js", "deviceClips", device).then((c) => c && page.querySelector(".page-tabs")?.before(c));
}
