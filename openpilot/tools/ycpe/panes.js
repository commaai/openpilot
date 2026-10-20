// every pane type a layout can hold; each module exports create(ctx, state) per the pane contract
import {h} from "/dom.js";

export const PANES = {
  empty: {title: "Empty", hidden: true},
  routes: {title: "Routes", src: "/panes/routes.js"},
  camera: {title: "Camera", src: "/panes/camera.js"},
  map: {title: "Map", src: "/panes/map.js"},
  info: {title: "Route info", src: "/panes/routeinfo.js"},
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

export async function loadPane(type) {
  if (type === "empty" || !PANES[type]) return emptyPane;
  const mod = await import(PANES[type].src).catch(() => null);
  return mod?.create || ((ctx, state) => missingPane(type));
}

// an empty slot offers every pane type
function emptyPane(ctx) {
  const el = h("div", {class: "empty-pane"},
    Object.entries(PANES).filter(([, p]) => !p.hidden).map(([type, p]) =>
      h("button", {class: "chip", onclick: () => ctx.layout.replaceSelf(el, {pane: type, state: {}})}, p.title)));
  return {name: "empty", el, render() {}};
}

function missingPane(type) {
  return {name: type, el: h("div", {class: "pane-missing"}, `${PANES[type].title} isn't built yet`), render() {}};
}
