// the presets: layouts a workspace tab starts from, the same for a route and a live session. A route or a live session
// opens as the one workspace whose tabs are these (users' edits are saved with it); the URL's view names the tab's
// preset, and a preset not open yet opens as a new tab
const pane = (type, state = {}) => ({pane: type, state});
const fixed = (type, state = {}) => ({...pane(type, state), fixed: true});
const split = (dir, sizes, ...children) => ({split: dir, sizes, children});
const preset = (title, root) => ({title, root});

export const PRESETS = {
  // the road camera and the map; the route's details are its menu's, its files the timeline's, its logs a tab's
  drive: preset("Drive", split("row", [0.7, 0.3], pane("camera", {camera: "qcamera"}), pane("map"))),
  plot: preset("Plot", split("row", [0.22, 0.53, 0.25],
    pane("browser"),
    split("col", [1 / 3, 1 / 3, 1 / 3], pane("plot"), pane("plot"), pane("plot")),
    split("col", [0.5, 0.5], pane("camera", {camera: "qcamera"}), pane("map")))),
  // cabana: one tool, so its panes stay (more can be split off them); charts are plots split off its plot
  can: preset("CAN", split("row", [0.32, 0.36, 0.32],
    fixed("can-messages"),
    split("col", [0.38, 0.37, 0.25], fixed("can-binary"), fixed("can-signals"), fixed("can-history")),  // as cabana's detail view
    split("col", [0.6, 0.4], pane("plot"), fixed("camera", {camera: "qcamera"})))),
  logs: preset("Logs", split("row", [0.62, 0.38], pane("logs"), pane("inspector"))),
  device: preset("Device", pane("device")),
};

// a preset as a tab, and the tabs the workspace opens with
export const tabOf = (key) => ({name: PRESETS[key].title, preset: key, root: structuredClone(PRESETS[key].root)});
export const WORKSPACE = {active: 0, tabs: ["drive", "plot", "can", "logs"].map(tabOf)};

// user mode's drive: connect's route page, and only it (style.css drops its pane's chrome)
export const CONNECT = {active: 0, tabs: [{name: "connect", root: pane("connect")}]};
