// home: sign-in when signed out, otherwise the device list. deviceName is the inline rename the device page shares.
import {api, auth} from "../api.js";
import {h, fmt} from "../dom.js";
import {icon} from "../icons.js";
import {link, onlineMark} from "../app.js";
import {loginPage} from "./login.js";
import {recentRoutes, routePath} from "./omnibox.js";

const WHEN = new Intl.DateTimeFormat(undefined, {weekday: "short", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false});

export const canManage = async (device) => device.is_owner || !!(await auth.user())?.superuser;

// the name, plus a pencil for those who may rename; Enter or blur saves, Escape cancels
export function deviceName(device) {
  const name = () => device.alias || device.dongle_id;
  const label = h("span", {}, name());
  const edit = h("button", {class: "icon-btn", title: "Rename", "aria-label": `Rename ${name()}`, hidden: true, onclick: start}, icon("edit", 14));
  const failed = h("span", {class: "small error", role: "alert"});
  const el = h("span", {class: "device-name"}, label, edit);
  canManage(device).then((ok) => { edit.hidden = !ok; });

  function start(e) {
    e.preventDefault();
    e.stopPropagation();
    const input = h("input", {class: "field", value: device.alias || "", placeholder: device.dongle_id, maxlength: 32, "aria-label": "Device name",
      onclick: (e) => { e.preventDefault(); e.stopPropagation(); }});
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      const refocus = document.activeElement === input;  // Enter or Escape, not a click away
      const alias = input.value.trim();
      failed.textContent = "";
      if (save && alias !== (device.alias || "")) {
        input.disabled = true;
        try {
          device.alias = (await api.setDeviceAlias(device.dongle_id, alias)).alias;
        } catch (err) {
          failed.textContent = `Rename failed: ${err.message}`;
        }
      }
      label.textContent = name();
      edit.setAttribute("aria-label", `Rename ${name()}`);
      el.replaceChildren(label, edit, failed);
      if (refocus) edit.focus();
    };
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") finish(true);
      else if (e.key === "Escape") finish(false);
    });
    input.addEventListener("blur", () => finish(true));
    el.replaceChildren(input);
    input.focus();
    input.select();
  }
  return el;
}

// the last routes opened here, newest first
function recentSection(devices) {
  const routes = recentRoutes();
  if (!routes.length) return null;
  const alias = (dongle) => devices?.find((d) => d.dongle_id === dongle)?.alias || dongle;
  return h("div", {class: "page recent"},
    h("h3", {}, "Recent routes"),
    h("table", {class: "list"},
      h("tbody", {}, routes.map((r) => {
        const path = routePath(r.name), dongle = r.name.split("|")[0];
        return h("tr", {class: "clickable", onclick: (e) => { if (!e.target.closest("a")) link(path)(e); }},
          h("td", {class: "nowrap"}, h("a", {href: path, onclick: link(path)}, WHEN.format(r.start))),
          h("td", {}, alias(dongle)),
          h("td", {class: "num"}, fmt.minutes(r.end - r.start)),
          h("td", {class: "num"}, fmt.miles(r.distance ?? 0)),
          h("td", {class: "muted"}, r.platform || ""),
          h("td", {class: "mono muted"}, r.name));
      }))));
}

// files from this machine (pages/local.js); dropping them anywhere works too, but nothing else says so
const localEntry = () => h("p", {class: "small muted local-entry"},
  h("a", {href: "/local", onclick: link("/local")}, "Open local logs…"), " (or drop rlog/qlog files anywhere) · ",
  h("a", {href: "/live", onclick: link("/live")}, "Live data…"));

export function homePage(devices) {
  if (!auth.isAuthenticated()) return h("div", {class: "home-page"}, loginPage(new URLSearchParams()), localEntry(), recentSection(devices));
  if (!devices?.length) return h("div", {class: "home-page"}, h("div", {class: "home"}, h("h2", {}, "No devices"), h("p", {class: "muted"}, "Pair a device from its setup screen, or open a public route by name."), localEntry()), recentSection(devices));
  return h("div", {class: "home-page"}, recentSection(devices), h("div", {class: "page"},
    h("table", {class: "list devices"},
      h("thead", {}, h("tr", {}, h("th"), h("th", {}, "device"), h("th", {}, "dongle id"), h("th", {}, "type"), h("th", {}, "version"), h("th", {}, "last seen"))),
      h("tbody", {}, devices.map((d) => h("tr", {class: "clickable", onclick: (e) => { if (!e.target.closest("a")) link(`/${d.dongle_id}`)(e); }},
        h("td", {}, onlineMark(d)),
        h("td", {}, h("a", {href: `/${d.dongle_id}`, onclick: link(`/${d.dongle_id}`)}, deviceName(d))),
        h("td", {class: "mono muted"}, d.dongle_id),
        h("td", {class: "muted"}, d.device_type),
        h("td", {class: "muted"}, d.openpilot_version || ""),
        h("td", {class: "muted"}, onlineMark(d, "ago")))))),
    localEntry()));
}
