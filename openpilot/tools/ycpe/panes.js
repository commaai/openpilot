// every pane type a layout can hold; each module exports create(ctx, state) per the pane contract
import {h} from "/dom.js";

export const PANES = {
  empty: {title: "Empty", hidden: true},
  routes: {title: "Routes", src: "/panes/routes.js"},
  camera: {title: "Camera", src: "/panes/camera.js"},
  map: {title: "Map", src: "/panes/map.js"},
  info: {title: "Route info", src: "/panes/routeinfo.js"},
  connect: {title: "Connect", src: "/panes/connect.js"},
  files: {title: "Files", src: "/panes/files.js"},
  plot: {title: "Plot", src: "/panes/plot/plot.js", data: true},
  xy: {title: "XY plot", src: "/panes/plot/xy.js", data: true},
  browser: {title: "Series", src: "/panes/browser.js", data: true},
  logs: {title: "Logs", src: "/panes/logs.js", data: true},
  inspector: {title: "Inspector", src: "/panes/inspector.js", data: true},
  "can-messages": {title: "CAN messages", src: "/panes/can/messages.js", data: true},
  "can-binary": {title: "CAN binary", src: "/panes/can/binary.js", data: true},
  "can-signals": {title: "CAN signals", src: "/panes/can/signals.js", data: true},
  "can-history": {title: "CAN history", src: "/panes/can/history.js", data: true},
  "can-charts": {title: "CAN charts", src: "/panes/can/charts.js", data: true},
};

// the browser remembers a failed import for good, so a retry asks for the module under a fresh URL
const retries = {};

export async function loadPane(type) {
  if (type === "empty" || !PANES[type]) return emptyPane;
  try {
    return (await import(PANES[type].src + (retries[type] || ""))).create;
  } catch (e) {
    console.error(e);
    return (ctx, state) => failedPane(ctx, type, state, e);
  }
}

// an empty slot offers every pane type
function emptyPane(ctx) {
  const el = h("div", {class: "empty-pane"},
    Object.entries(PANES).filter(([, p]) => !p.hidden).map(([type, p]) =>
      h("button", {class: "chip", onclick: () => ctx.layout.replaceSelf(el, {pane: type, state: {}})}, p.title)));
  return {name: "empty", el, render() {}};
}

function failedPane(ctx, type, state, error) {
  const retry = () => {
    retries[type] = `?retry=${Date.now()}`;
    ctx.layout.replaceSelf(el, {pane: type, state});
  };
  const el = h("div", {class: "pane-missing"}, `${PANES[type].title} didn't load: ${error.message} `, h("button", {class: "chip", onclick: retry}, "Retry"));
  return {name: type, el, render() {}, save: () => state};
}
