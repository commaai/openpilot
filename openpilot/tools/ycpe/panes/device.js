// the device pane: the route's device as an inspector beside it, with the device page's tabs (its drives, preserved
// drives, uploads, users, boot and crash logs, settings). The navigator's device links open it on a tab (show).
import {deviceRecord} from "/devices.js";
import {h, fmt} from "/dom.js";
import {onlineMark} from "/app.js";
import {deviceName} from "/pages/home.js";
import {DEVICE_TABS, deviceTabs} from "/pages/device.js";

export function create(ctx, state) {
  const dongle = ctx.route.dongle_id || ctx.route.fullname.split("|")[0];
  const el = h("div", {class: "device-pane page"}, h("p", {class: "muted center"}, "Loading the device…"));
  let tab = DEVICE_TABS[state.tab] ? state.tab : "settings", tabs = null;
  deviceRecord(dongle).then((device) => {
    tabs = deviceTabs(device, (key) => { tab = key; });
    tabs.select(tab);
    el.replaceChildren(
      h("div", {class: "device-pane-head"}, onlineMark(device), deviceName(device),
        h("span", {class: "mono muted"}, dongle), h("span", {class: "muted"}, fmt.deviceType(device.device_type)),
        device.openpilot_version && h("span", {class: "muted"}, `openpilot ${device.openpilot_version}`), onlineMark(device, "ago")),
      tabs.bar, tabs.body);
  }, (e) => el.replaceChildren(h("p", {class: "muted center"},
    e.status === 401 || e.status === 403 ? `Device ${dongle} isn't shared with this account.` : `Couldn't load the device: ${e.message}`)));
  return {name: "device", el, render() {}, save: () => ({tab}), show: (s) => s.tab && tabs?.select(s.tab)};
}
