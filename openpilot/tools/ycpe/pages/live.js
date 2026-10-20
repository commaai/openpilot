// /live: what can stream into the page. This computer's webrtcd (through the local daemon), a panda over WebUSB, the
// account's devices (an online one streams through athena), and a demo route's log played back as if live.
import {auth} from "../api.js";
import {h, fmt} from "../dom.js";
import {isOnline, link} from "../app.js";
import {daemon, needsDaemon} from "../daemon.js";
import {loginPath} from "./login.js";

export const DEMO_ROUTE = "5beb9b58bd12b691|0000010a--a51155e496";

// a source to open, or why it can't be (then it's shown but not a link)
function row(path, dot, name, detail, why = null) {
  const go = (e) => { if (!why && !e.target.closest("a")) link(path)(e); };
  return h("tr", {class: why ? "" : "clickable", onclick: go},
    h("td", {}, h("span", {class: `dot ${dot ? "online" : ""}`})),
    h("td", {}, why ? h("span", {}, name) : h("a", {href: path, onclick: link(path)}, name)),
    h("td", {class: "muted"}, why || detail));
}

export function livePage(devices) {
  const demo = `/live/replay?route=${encodeURIComponent(DEMO_ROUTE)}`;
  return h("div", {class: "page live-page"},
    h("h3", {}, "On this computer"),
    h("table", {class: "list"}, h("tbody", {},
      row("/live/local", daemon.present, "openpilot", "a webrtcd here: openpilot, the simulator or tools/replay",
        daemon.present ? null : needsDaemon("Streaming from this computer")),
      row("/live/panda", !!navigator.usb, "Panda over USB", "CAN and panda health, receive only",
        navigator.usb ? null : "needs WebUSB, which this browser doesn't have"),
      row(demo, true, "Demo replay", "a demo route's log played back as if it were live"))),
    h("h3", {}, "Devices"),
    !auth.isAuthenticated() ? h("p", {class: "muted"}, h("a", {href: loginPath(), onclick: link(loginPath())}, "Sign in"), " to stream from your devices.")
      : !devices?.length ? h("p", {class: "muted"}, "No devices on this account.")
      : h("table", {class: "list"}, h("tbody", {}, devices.map((d) => row(`/${d.dongle_id}/live`, isOnline(d), d.alias || d.dongle_id,
        isOnline(d) ? `online · ${d.device_type || ""}` : `offline · last seen ${fmt.ago(d.last_athena_ping)}`)))));
}
