import {api} from "/api.js";
import {h, fmt} from "/dom.js";
import {link} from "/app.js";

export function deviceCard(d) {
  const online = d.last_athena_ping && Date.now() / 1000 - d.last_athena_ping < 120;
  return h("a", {class: "card device", href: `/${d.dongle_id}`, onclick: link(`/${d.dongle_id}`)},
    h("div", {class: "title"}, d.alias || d.dongle_id, h("span", {class: `dot ${online ? "online" : ""}`, title: online ? "online" : `last seen ${fmt.ago(d.last_athena_ping)}`})),
    h("div", {class: "muted"}, d.dongle_id, " · ", d.device_type, d.openpilot_version ? ` · ${d.openpilot_version}` : ""));
}

export async function devicesPage() {
  const devices = await api.devices();
  if (devices.length === 0) {
    return h("div", {class: "page"}, h("h2", {}, "No devices"), h("p", {class: "muted"}, "Pair a device from its setup screen."));
  }
  if (devices.length === 1) {
    history.replaceState(null, "", `/${devices[0].dongle_id}`);
    const {devicePage} = await import("/pages/device.js");
    return devicePage(devices[0].dongle_id);
  }
  return h("div", {class: "page"}, h("h2", {}, "Devices"), h("div", {class: "cards"}, devices.map(deviceCard)));
}
