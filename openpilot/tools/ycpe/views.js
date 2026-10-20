// the built-in views: each is a default layout (users' edits are saved per view)
const pane = (type, state = {}) => ({pane: type, state});
const split = (dir, sizes, ...children) => ({split: dir, sizes, children});
const view = (title, root) => ({title, layout: {active: 0, tabs: [{name: title.toLowerCase(), root}]}});

export const VIEWS = {
  drive: view("Drive", split("row", [0.2, 0.8],
    pane("routes"),
    split("col", [0.68, 0.32], pane("camera", {camera: "qcamera"}), split("row", [0.5, 0.5], pane("map"), pane("info"))))),
  plot: view("Plot", split("row", [0.2, 0.55, 0.25],
    pane("browser"),
    split("col", [1 / 3, 1 / 3, 1 / 3], pane("plot"), pane("plot"), pane("plot")),
    split("col", [0.5, 0.5], pane("camera", {camera: "qcamera"}), pane("map")))),
  can: view("CAN", split("row", [0.32, 0.36, 0.32],
    pane("can-messages"),
    split("col", [0.38, 0.37, 0.25], pane("can-binary"), pane("can-signals"), pane("can-history")),  // as cabana's detail view
    split("col", [0.6, 0.4], pane("can-charts"), pane("camera", {camera: "qcamera"})))),
  files: view("Files", split("row", [0.2, 0.8], pane("routes"), pane("files"))),
  // connect's route page, the default for a route opened without a view (style.css drops its pane's chrome)
  connect: {...view("Connect", pane("connect")), bare: true},
};

// a live session's: plot and CAN as for routes (sharing their saved layouts), drive without the route's own panes,
// and logs. key: where the layout is saved, when that isn't the view's name
export const LIVE_VIEWS = {
  drive: {...view("Drive", split("row", [0.68, 0.32],
    pane("camera", {camera: "fcamera"}), split("col", [0.5, 0.5], pane("map"), pane("logs")))), key: "live-drive"},
  plot: VIEWS.plot,
  can: VIEWS.can,
  logs: {...view("Logs", split("row", [0.62, 0.38], pane("logs"), pane("inspector"))), key: "live-logs"},
};
