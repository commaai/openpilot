// home: sign-in when signed out, otherwise the device list. deviceName is the inline rename the device page shares.
import {api, auth} from "../api.js";
import {h, fmt} from "../dom.js";
import {icon} from "../icons.js";
import {isOnline, link} from "../app.js";

const DEMO = "/5beb9b58bd12b691/0000010a--a51155e496/drive";

export const canManage = async (device) => device.is_owner || !!(await api.me().catch(() => null))?.superuser;

// the name, plus a pencil for those who may rename; Enter or blur saves, Escape cancels
export function deviceName(device) {
  const label = h("span", {}, device.alias || device.dongle_id);
  const edit = h("button", {class: "icon-btn", title: "Rename", hidden: true, onclick: start}, icon("edit", 14));
  const el = h("span", {class: "device-name"}, label, edit);
  canManage(device).then((ok) => { edit.hidden = !ok; });

  function start(e) {
    e.preventDefault();
    e.stopPropagation();
    const input = h("input", {class: "field", value: device.alias || "", placeholder: device.dongle_id, maxlength: 32,
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
          label.title = "";
        } catch (err) {
          label.title = `Rename failed: ${err.message}`;
        }
      }
      label.classList.toggle("error", !!label.title);
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

function signIn() {
  return h("div", {class: "home"},
    h("h2", {}, "ycpe"),
    h("p", {class: "muted"}, "Routes, plots and CAN for openpilot drives."),
    h("div", {class: "sign-in"}, ["google", "github", "apple"].map((p) =>
      h("a", {class: "button", href: auth.loginUrl(p)}, `Sign in with ${p[0].toUpperCase()}${p.slice(1)}`))),
    h("p", {}, h("a", {href: DEMO, onclick: link(DEMO)}, "Open the demo route"), h("span", {class: "muted"}, " or paste a route name above.")));
}

export function homePage(devices) {
  if (!auth.isAuthenticated()) return signIn();
  if (!devices?.length) return h("div", {class: "home"}, h("h2", {}, "No devices"), h("p", {class: "muted"}, "Pair a device from its setup screen, or open a public route by name."));
  return h("div", {class: "page"},
    h("table", {class: "list devices"},
      h("thead", {}, h("tr", {}, h("th"), h("th", {}, "device"), h("th", {}, "dongle id"), h("th", {}, "type"), h("th", {}, "version"), h("th", {}, "last seen"))),
      h("tbody", {}, devices.map((d) => h("tr", {class: "clickable", onclick: (e) => { if (!e.target.closest("a")) link(`/${d.dongle_id}`)(e); }},
        h("td", {}, h("span", {class: `dot ${isOnline(d) ? "online" : ""}`})),
        h("td", {}, h("a", {href: `/${d.dongle_id}`, onclick: link(`/${d.dongle_id}`)}, deviceName(d))),
        h("td", {class: "mono muted"}, d.dongle_id),
        h("td", {class: "muted"}, d.device_type),
        h("td", {class: "muted"}, d.openpilot_version || ""),
        h("td", {class: "muted"}, isOnline(d) ? "online" : fmt.ago(d.last_athena_ping)))))));
}
