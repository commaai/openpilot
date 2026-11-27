// home: sign-in when signed out, otherwise the device list. deviceName is the inline rename the device page shares.
import {auth} from "../api.js";
import {h, fmt} from "../dom.js";
import {icon} from "../icons.js";
import {link, onlineMark} from "../app.js";
import {renameDevice} from "../devices.js";
import {loginPage} from "./login.js";

export const canManage = async (device) => device.is_owner || !!(await auth.user())?.superuser;

// the name, plus a pencil for those who may rename; Enter or blur saves, Escape cancels
export function deviceName(device) {
  const name = () => fmt.deviceName(device);
  const label = h("span", {}, name());
  const edit = h("button", {class: "icon-btn", title: "Rename", "aria-label": `Rename ${name()}`, hidden: true, onclick: start}, icon("edit", 14));
  const failed = h("span", {class: "small error", role: "alert"});
  const el = h("span", {class: "device-name"}, label, edit);
  canManage(device).then((ok) => { edit.hidden = !ok; });

  function start(e) {
    e.preventDefault();
    e.stopPropagation();
    const input = h("input", {class: "field", value: device.alias || "", placeholder: fmt.deviceType(device.device_type), maxlength: 32, "aria-label": "Device name",
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
          await renameDevice(device, alias);
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

// the API turns this device's uploads away (useradmin's flag); said only when it does, as it's rare and surprising
const uploadsIgnored = (device) => device.ignore_uploads &&
  h("span", {class: "tag bad", title: "The comma API ignores this device's uploads"}, "uploads ignored");

// empty: what an account without devices sees (pair.js noDevices, which app.js loads only then)
export function homePage(devices, empty) {
  if (!auth.isAuthenticated()) return h("div", {class: "home-page"}, loginPage(new URLSearchParams()));
  if (!devices?.length) return h("div", {class: "home-page"}, empty);
  return h("div", {class: "home-page"}, h("div", {class: "page"},
    h("table", {class: "list devices"},
      h("thead", {}, h("tr", {}, h("th"), h("th", {}, "device"), h("th", {}, "dongle id"), h("th", {}, "type"), h("th", {}, "version"), h("th", {}, "last seen"))),
      h("tbody", {}, devices.map((d) => h("tr", {class: "clickable", onclick: (e) => { if (!e.target.closest("a")) link(`/${d.dongle_id}`)(e); }},
        h("td", {}, onlineMark(d)),
        h("td", {}, h("a", {href: `/${d.dongle_id}`, onclick: link(`/${d.dongle_id}`)}, deviceName(d)), uploadsIgnored(d)),
        h("td", {class: "mono muted"}, d.dongle_id),
        h("td", {class: "muted"}, d.device_type && fmt.deviceType(d.device_type)),
        h("td", {class: "muted"}, d.openpilot_version || ""),
        h("td", {class: "muted"}, onlineMark(d, "ago")))))),
    h("button", {class: "button small add-device", onclick: () => import("./pair.js").then((m) => m.openPair())}, icon("add", 14), "Add device")));
}
