// stand-ins for the pairing, prime and clips tracks' exports, served at their paths by the um-* scenarios: each says it
// was called in window.__mounts, with what
import {h, css} from "/dom.js";
import {cnIcon} from "/pages/cnui.js";

await css(new URL("/pages/pair.css", import.meta.url));

const called = (what) => (window.__mounts ??= []).push(what);

// the sidebar's row, as pair.js draws it
export const pairButton = () => (called("pairButton"), h("button", {class: "cnui-btn white pair-cn row fake-pair"}, "add new device", cnIcon("add", 24)));
export const noDevices = () => (called("noDevices"), h("div", {class: "page-msg fake-no-devices"}, h("h2", {}, "Pair your device")));
export const handlePairLink = (q) => (called(`handlePairLink:${q.get("pair")}`), false);
export const primeBanner = (devices) => (called(`primeBanner:${devices[0]?.dongle_id}`), h("div", {class: "fake-prime-banner"}, "comma prime"));
export const primePage = (device) => (called(`primePage:${device.dongle_id}`), h("div", {class: "page fake-prime"}, h("h2", {}, "comma prime")));
export const clipButton = (ctx) => (called(`clipButton:${ctx.route.fullname}`), h("button", {class: "cn-option fake-clip"}, "Clip"));
export const deviceClips = (device) => (called(`deviceClips:${device.dongle_id}`), h("div", {class: "fake-clips"}, "Clips"));
