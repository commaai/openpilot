// the device list, and the inline rename shared with the device page
import {api} from "../api.js";
import {h, fmt} from "../dom.js";
import {icon} from "../icons.js";
import {isOnline, link} from "../app.js";

export const canManage = async (device) => device.is_owner || !!(await api.me().catch(() => null))?.superuser;

// the name, plus a pencil for those who may rename; Enter or blur saves, Escape cancels
export function deviceName(device) {
  const label = h("span", {}, device.alias || device.dongle_id);
  const edit = h("button", {class: "icon-button small", "aria-label": "Rename", hidden: true, onclick: start}, icon("edit", 16));
  const el = h("span", {class: "device-name"}, label, edit);
  canManage(device).then((ok) => { edit.hidden = !ok; });

  function start(e) {
    e.preventDefault();
    e.stopPropagation();
    const input = h("input", {class: "rename", value: device.alias || "", placeholder: device.dongle_id, maxlength: 32,
      onclick: (e) => { e.preventDefault(); e.stopPropagation(); }});
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      const alias = input.value.trim();
      if (save && alias !== (device.alias || "")) {
        input.disabled = true;
        try {
          device.alias = (await api.setDeviceAlias(device.dongle_id, alias)).alias;
        } catch (err) {
          alert(`Rename failed: ${err.message}`);
        }
      }
      label.textContent = device.alias || device.dongle_id;
      el.replaceChildren(label, edit);
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

export function devicesPage(devices) {
  const rows = devices.map((d) => h("a", {class: "device-row", href: `/${d.dongle_id}`, onclick: link(`/${d.dongle_id}`)},
    h("span", {class: `dot ${isOnline(d) ? "online" : ""}`}),
    h("span", {class: "first"}, deviceName(d)),
    h("span", {class: "mono muted"}, d.dongle_id),
    h("span", {class: "muted"}, d.openpilot_version || ""),
    h("span", {class: "muted"}, isOnline(d) ? "online" : fmt.ago(d.last_athena_ping))));
  return h("div", {class: "page"},
    h("h2", {}, "Devices"),
    h("div", {class: "device-table"},
      h("div", {class: "device-row head muted small"}, h("span"), h("span", {}, "name"), h("span", {}, "dongle id"), h("span", {}, "version"), h("span", {}, "last seen")),
      rows));
}
