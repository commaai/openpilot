// layout import: jotpluggler JSON → our layout model
//   {active, tabs: [{name, root}]}, node = {split: "row" | "col", sizes, children} | {pane, state}
// Their saved time ranges are dropped: they were on another route, and a layout opens where the user is looking.
// Custom series keep their inputs and name, and jotpluggler's Python code comes along commented out, to be rewritten
// in JS. Imported code is marked untrusted, so it only runs once the user has reviewed it in the console.

const CAMERAS = {road: "fcamera", wide_road: "ecamera", driver: "dcamera"};

// text of a .json layout (ours, as exported, or jotpluggler's) → {layout, warnings}
export function importLayout(text) {
  const warnings = [];
  const j = JSON.parse(text);
  return {layout: j.ycpe ? untrusted(j) : fromJotpluggler(j, warnings), warnings};
}

// our own layouts as exported: {ycpe: 1, active, tabs}; whatever custom code they carry waits for review
export const exportLayout = (layout) => JSON.stringify({ycpe: 1, ...layout}, null, 2);

function untrusted({active = 0, tabs = []}) {
  const walk = (n) => n.children ? {...n, children: n.children.map(walk)}
    : {...n, state: n.state?.curves ? {...n.state, curves: n.state.curves.map((c) => c.custom ? {...c, custom: {...c.custom, untrusted: true}} : c)} : n.state};
  return {active, tabs: tabs.map((t) => ({...t, root: walk(t.root)}))};
}

// ---- custom series names: a custom series computes into the store at /custom/<name>, which other custom series
// read as an input, so a name means one piece of code

// what a custom series computes
export const specKey = ({inputs, globals = "", code}) => JSON.stringify([inputs, globals, code]);

// fn(curves) for every time plot in a layout model
export function plots({tabs}, fn) {
  const walk = (n) => n.children ? n.children.forEach(walk) : n.pane === "plot" && fn(n.state?.curves || []);
  tabs.forEach((t) => walk(t.root));
}

// path → specKey of the custom series in a layout model
export function customsOf(layout) {
  const out = new Map();
  plots(layout, (curves) => { for (const c of curves) if (c.custom) out.set(c.path, specKey(c.custom)); });
  return out;
}

// renames, in place, the custom series of layout whose name another one (earlier in it, or in taken: path → specKey)
// has with different code: x becomes x_2. A curve or input that reads it follows the definition in its own plot, else
// the layout's when there is just one
export function uniqueCustoms(layout, taken = new Map()) {
  const own = new Map(), defs = new Map();  // curves → (name → where its plot's definition went); name → where every definition went
  const follow = (curves, p) => own.get(curves).get(p) ?? (defs.get(p)?.size === 1 ? [...defs.get(p)][0] : p);
  plots(layout, (curves) => {
    own.set(curves, new Map());
    for (const c of curves) {
      if (!c.custom) continue;
      const key = specKey({...c.custom, inputs: c.custom.inputs.map((p) => follow(curves, p))});
      let path = c.path;
      for (let n = 2; taken.has(path) && taken.get(path) !== key; n++) path = `${c.path}_${n}`;
      if (!own.get(curves).has(c.path)) own.get(curves).set(c.path, path);
      defs.set(c.path, (defs.get(c.path) || new Set()).add(path));
      if (path !== c.path) {
        if (c.label === c.path.slice("/custom/".length)) c.label = path.slice("/custom/".length);
        c.path = path;
      }
      taken.set(path, key);
    }
  });
  // then whatever reads them, before or after
  plots(layout, (curves) => {
    for (const c of curves) {
      if (c.custom) c.custom.inputs = c.custom.inputs.map((p) => follow(curves, p));
      else c.path = follow(curves, c.path);
    }
  });
  return layout;
}

// jotpluggler's per-message fields, named as PlotJuggler's: /<service>/__valid is our /<service>/valid (__logMonoTime is the same path)
function field(path) {
  const m = path?.match(/^(\/[^/]+)\/__valid$/);
  return m ? `${m[1]}/valid` : path;
}

// ---- jotpluggler

function fromJotpluggler(j, warnings) {
  const tabs = (j.tabs || []).map((t) => ({name: t.name, root: jpNode(t.root, warnings)}));
  return {active: j.current_tab_index || 0, tabs};
}

function jpNode(n, warnings) {
  if (n.split) return {split: n.split === "vertical" ? "col" : "row", sizes: n.sizes, children: n.children.map((c) => jpNode(c, warnings))};
  if (n.kind === "camera") return {pane: "camera", state: {camera: CAMERAS[n.camera_view] || "fcamera"}};
  if (n.kind === "thumbnail") return {pane: "camera", state: {camera: "qcamera"}};
  if (n.kind === "map") return {pane: "map", state: {}};
  const curves = (n.curves || []).map((c) => {
    const out = {path: field(c.name), color: c.color};
    if (c.transform === "derivative") out.transform = {kind: "derivative", param: c.derivative_dt || 0};
    if (c.transform === "scale") Object.assign(out, {scale: c.scale ?? 1, offset: c.offset ?? 0});
    if (c.custom_python) {
      const p = c.custom_python, name = c.name.replaceAll("/", "_");
      warnings.push(`${name}: its Python code needs rewriting in JS`);
      Object.assign(out, {path: `/custom/${name}`, label: name, custom: {inputs: [p.linked_source, ...p.additional_sources].map(field),
        globals: commented(p.globals_code), code: commented(p.function_code), untrusted: true}});
    }
    return out;
  });
  const y = n.y_limits ? {min: n.y_limits.min, max: n.y_limits.max} : undefined;
  return {pane: "plot", state: {title: n.title && n.title !== "..." ? n.title : "", curves, ...(y && {y})}};
}

// code kept as comments
const commented = (code) => code ? code.split("\n").map((l) => `// ${l}`).join("\n") : "";
