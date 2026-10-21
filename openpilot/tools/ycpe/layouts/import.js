// layout import: jotpluggler JSON → our layout model
//   {active, tabs: [{name, root}]}, node = {split: "row" | "col", sizes, children} | {pane, state}
// Their saved time ranges are dropped: they were on another route, and a layout opens where the user is looking.
// Custom series code is translated to JS from jotpluggler's Python where that is
// mechanical; what isn't is flagged in a comment. Imported code is marked untrusted, so it only runs once
// the user has reviewed it in the console.

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
  const tabs = (j.tabs || []).map((t) => {
    return {name: t.name, root: jpNode(t.root, warnings)};
  });
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
      Object.assign(out, {path: `/custom/${name}`, label: name,
        custom: translate([p.linked_source, ...p.additional_sources].map(field), p.globals_code, pyBody(p.function_code), warnings, name)});
    }
    return out;
  });
  const y = n.y_limits ? {min: n.y_limits.min, max: n.y_limits.max} : undefined;
  return {pane: "plot", state: {title: n.title && n.title !== "..." ? n.title : "", curves, ...(y && {y})}};
}

// the body of jotpluggler's per-sample wrapper: def __jotpluggler_eval_sample(time, value, …): <body>
function pyBody(code) {
  const lines = code.split("\n");
  if (!/^def __jotpluggler_eval_sample\(/.test(lines[0])) return null;
  const body = [];
  for (const line of lines.slice(1)) {
    if (line.trim() && !/^\s/.test(line)) break;
    body.push(line.replace(/^ {4}/, ""));
  }
  return body.join("\n").trim();
}

// ---- Python → JS for per-sample custom series

// {inputs, globals, code, untrusted}; untranslatable lines stay as comments and are listed in warnings
function translate(inputs, globals, body, warnings, name) {
  const params = new Set(["time", "value", ...inputs.slice(1).map((_, i) => `v${i + 1}`)]);
  const problems = [];
  const g = py(globals || "", problems), f = body === null ? (problems.push("not a per-sample function"), "") : py(body, problems);
  const gNames = assigned(g);
  const fNames = assigned(f).filter((n) => !gNames.includes(n) && !params.has(n));
  const head = ["// translated from jotpluggler Python", ...problems.map((p) => `// could not translate: ${p}`)];
  if (problems.length) warnings.push(`${name}: ${problems.join("; ")}`);
  return {
    inputs, untrusted: true,
    globals: (gNames.length ? `let ${gNames.join(", ")};\n` : "") + g,
    code: [...head, ...(fNames.length ? [`let ${fNames.join(", ")};`] : []), f].join("\n"),
  };
}

// names assigned at the start of a statement (a = …, [a, b] = …)
function assigned(js) {
  const names = [];
  for (const m of js.matchAll(/^\s*\[?([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\]?\s*=(?!=)/gm)) {
    for (const n of m[1].split(",").map((s) => s.trim())) if (!names.includes(n)) names.push(n);
  }
  return names;
}

// a, b = x, y → [a, b] = [x, y]
const tuple = (s) => s.replace(/^([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)+)\s*=(?!=)\s*(.+)$/, "[$1] = [$2]");
const count = (s, re) => (s.match(re) || []).length;

// a condition in exactly one pair of parentheses
function cond(s) {
  s = s.trim();
  let depth = 0, wrapped = s.startsWith("(") && s.endsWith(")");
  for (let i = 0; i < s.length && wrapped; i++) {
    depth += s[i] === "(" ? 1 : s[i] === ")" ? -1 : 0;
    if (depth === 0 && i < s.length - 1) wrapped = false;
  }
  return wrapped ? s : `(${s})`;
}

// joins translated lines; a statement gets its ";" where it ends (brackets closed, no trailing operator), since
// a line starting with "[" or "(" would otherwise continue the previous one
function emit(lines) {
  let depth = 0;
  return lines.map(({pad = "", js = "", comment = "", stmt = false}) => {
    if (stmt) {
      depth += count(js, /[([{]/g) - count(js, /[)\]}]/g);
      if (depth <= 0 && !/[-+*/%,=&|<>!]$/.test(js)) {
        js += ";";
        depth = 0;
      }
    }
    return pad + js + comment;
  }).join("\n");
}

const MATH = {pi: "PI", arctan2: "atan2", arcsin: "asin", arccos: "acos", arctan: "atan"};
const mathCall = (pre) => (s) => s.replace(new RegExp(`\\b${pre}\\.(\\w+)`, "g"), (_, f) => `Math.${MATH[f] || f}`);

function pyExpr(s, problems) {
  s = mathCall("(?:np|numpy|math)")(s)
    .replace(/\band\b/g, "&&").replace(/\bor\b/g, "||").replace(/\bnot\b/g, "!")
    .replace(/\bTrue\b/g, "true").replace(/\bFalse\b/g, "false").replace(/\bNone\b/g, "null")
    .replace(/(^|[^.\w])(abs|min|max)\(/g, "$1Math.$2(");
  if (/\bif\b.*\belse\b|\blambda\b|\bin\b/.test(s)) problems.push(s.trim());
  return s;
}

function py(src, problems) {
  const out = [], blocks = [];  // indents of open blocks
  let depth = 0, cont = false;
  for (const raw of src.split("\n")) {
    const hash = raw.indexOf("#"), comment = hash >= 0 ? ` //${raw.slice(hash + 1)}` : "";
    const line = (hash >= 0 ? raw.slice(0, hash) : raw).trimEnd(), t = line.trim(), ind = line.length - t.length;
    if (!t) { out.push({comment: comment.trim()}); continue; }
    if (!cont) {
      while (blocks.length && ind <= blocks.at(-1) && !(ind === blocks.at(-1) && /^(elif|else)\b/.test(t))) out.push({pad: " ".repeat(blocks.pop()), js: "}"});
    }
    const pad = " ".repeat(ind);
    let m;
    if (cont) out.push({pad, js: pyExpr(t, problems), comment, stmt: true});
    else if ((m = t.match(/^(if|elif|while)\s+(.+):$/))) {
      out.push({pad, js: `${m[1] === "elif" ? "} else if" : m[1]} ${cond(pyExpr(m[2], problems))} {`, comment});
      if (m[1] !== "elif") blocks.push(ind);
    } else if (/^else\s*:$/.test(t)) out.push({pad, js: "} else {", comment});
    else if (/^global\b/.test(t)) continue;
    else if (/^(for|def|try|except|with|class|import|from)\b/.test(t)) { problems.push(t); out.push({pad, js: `// ${t}`}); }
    else out.push({pad, js: tuple(pyExpr(t, problems)), comment, stmt: true});
    depth += count(t, /[([{]/g) - count(t, /[)\]}]/g);
    cont = depth > 0 || /[-+*/%,=&|<>\\]$/.test(t);
  }
  while (blocks.length) out.push({pad: " ".repeat(blocks.pop()), js: "}"});
  return emit(out);
}
