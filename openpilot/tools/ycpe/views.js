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
    split("col", [0.4, 0.3, 0.3], pane("camera", {camera: "qcamera"}), pane("map"), pane("logs")))),  // logs list the alerts
  can: view("CAN", split("row", [0.32, 0.36, 0.32],
    pane("can-messages"),
    split("col", [0.38, 0.37, 0.25], pane("can-binary"), pane("can-signals"), pane("can-history")),  // as cabana's detail view
    split("col", [0.6, 0.4], pane("can-charts"), pane("camera", {camera: "qcamera"})))),
  files: view("Files", split("row", [0.2, 0.8], pane("routes"), pane("files"))),
};
