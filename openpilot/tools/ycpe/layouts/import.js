// layout import: jotpluggler JSON and PlotJuggler XML → our layout model
//   {active, tabs: [{name, range?, root}]}, node = {split: "row" | "col", sizes, children} | {pane, state}
// Custom series code is translated to JS from jotpluggler's Python or PlotJuggler's Lua where that is
// mechanical; what isn't is flagged in a comment. Imported code is marked untrusted, so it only runs once
// the user has reviewed it in the console.

const CAMERAS = {road: "fcamera", wide_road: "ecamera", driver: "dcamera"};
const STYLES = {Dots: "dots", LinesAndDots: "line+dots", Sticks: "sticks", Steps: "steps", StepsInv: "steps"};

// text of a .json or .xml layout (ours, as exported, jotpluggler's or PlotJuggler's) → {layout, warnings}
export function importLayout(text) {
  const warnings = [];
  if (text.trimStart().startsWith("<")) return {layout: fromPlotJuggler(text, warnings), warnings};
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

// PlotJuggler's per-message fields: /<service>/__valid is our /<service>/valid (__logMonoTime is the same path)
function field(path) {
  const m = path?.match(/^(\/[^/]+)\/__valid$/);
  return m ? `${m[1]}/valid` : path;
}

// ---- jotpluggler

function fromJotpluggler(j, warnings) {
  const tabs = (j.tabs || []).map((t) => {
    const ranges = [];
    const root = jpNode(t.root, ranges, warnings);
    return {name: t.name, ...(ranges.length && {range: ranges[0]}), root};
  });
  return {active: j.current_tab_index || 0, tabs};
}

function jpNode(n, ranges, warnings) {
  if (n.split) return {split: n.split === "vertical" ? "col" : "row", sizes: n.sizes, children: n.children.map((c) => jpNode(c, ranges, warnings))};
  if (n.kind === "camera") return {pane: "camera", state: {camera: CAMERAS[n.camera_view] || "fcamera"}};
  if (n.kind === "thumbnail") return {pane: "camera", state: {camera: "qcamera"}};
  if (n.kind === "map") return {pane: "map", state: {}};
  if (n.range) ranges.push([n.range.left, n.range.right]);
  const curves = (n.curves || []).map((c) => {
    const out = {path: field(c.name), color: c.color};
    if (c.transform === "derivative") out.transform = {kind: "derivative", param: c.derivative_dt || 0};
    if (c.transform === "scale") Object.assign(out, {scale: c.scale ?? 1, offset: c.offset ?? 0});
    if (c.custom_python) {
      const p = c.custom_python, name = c.name.replaceAll("/", "_");
      Object.assign(out, {path: `/custom/${name}`, label: name,
        custom: translate("python", [p.linked_source, ...p.additional_sources].map(field), p.globals_code, pyBody(p.function_code), warnings, name)});
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

// ---- PlotJuggler

function fromPlotJuggler(text, warnings) {
  const doc = new DOMParser().parseFromString(text, "application/xml");
  if (doc.querySelector("parsererror")) throw new Error("not a PlotJuggler layout");
  const snippets = new Map();
  for (const s of doc.querySelectorAll("customMathEquations > snippet")) {
    const name = s.getAttribute("name"), linked = s.querySelector("linked_source")?.textContent;
    const extra = [...(s.querySelector("additional_sources")?.children || [])].map((e) => e.textContent);
    snippets.set(name, translate("lua", [linked, ...extra].map(field), text1(s, "global"), text1(s, "function"), warnings, name));
  }
  const tabs = [...doc.querySelectorAll("tabbed_widget > Tab")].map((tab) => {
    const ranges = [];
    const root = pjNode(tab.querySelector("Container").firstElementChild, ranges, snippets, warnings);
    return {name: tab.getAttribute("tab_name"), ...(ranges.length && {range: ranges[0]}), root};
  });
  return {active: +(doc.querySelector("currentTabIndex")?.getAttribute("index") || 0), tabs};
}

const text1 = (el, tag) => el.querySelector(tag)?.textContent || "";
const num = (el, attr, fallback) => el?.hasAttribute(attr) ? +el.getAttribute(attr) : fallback;

function pjNode(el, ranges, snippets, warnings) {
  if (el.tagName === "DockSplitter") {
    return {split: el.getAttribute("orientation") === "-" ? "col" : "row", sizes: el.getAttribute("sizes").split(";").map(Number),
      children: [...el.children].map((c) => pjNode(c, ranges, snippets, warnings))};
  }
  const plot = el.querySelector("plot"), title = el.getAttribute("name") || "";
  if (!plot) return {pane: "plot", state: {title, curves: []}};
  const range = plot.querySelector("range");
  if (range) ranges.push([num(range, "left"), num(range, "right")]);
  const style = STYLES[plot.getAttribute("style")];
  const curves = [...plot.querySelectorAll("curve")];
  if (plot.getAttribute("mode") === "XYPlot") {
    return {pane: "xy", state: {curves: curves.map((c) => ({x: field(c.getAttribute("curve_x")), y: field(c.getAttribute("curve_y")), color: c.getAttribute("color")}))}};
  }
  const lim = plot.querySelector("limitY");
  const y = lim && (lim.hasAttribute("min") || lim.hasAttribute("max")) ? {min: num(lim, "min"), max: num(lim, "max")} : undefined;
  return {pane: "plot", state: {title, curves: curves.map((c) => pjCurve(c, style, snippets, warnings)), ...(y && {y})}};
}

function pjCurve(el, style, snippets, warnings) {
  const name = el.getAttribute("name"), out = {path: field(name), color: el.getAttribute("color")};
  if (style) out.style = style;
  if (snippets.has(name)) Object.assign(out, {path: `/custom/${name.replaceAll("/", "_")}`, label: name, custom: snippets.get(name)});
  const t = el.querySelector("transform"), o = t?.querySelector("options");
  if (t?.getAttribute("name") === "Derivative") {
    out.transform = {kind: "derivative", param: o?.getAttribute("radioChecked") === "radioCustom" ? num(o, "lineEdit", 0) : 0};
  } else if (t?.getAttribute("name") === "Scale/Offset") {
    Object.assign(out, {scale: num(o, "value_scale", 1), offset: num(o, "value_offset", 0), shift: num(o, "time_offset", 0)});
  } else if (t && t.getAttribute("name") !== "noTransform") {
    warnings.push(`${name}: transform "${t.getAttribute("name")}" is not supported`);
  }
  return out;
}

// ---- Python / Lua → JS for per-sample custom series

// {inputs, globals, code, untrusted}; untranslatable lines stay as comments and are listed in warnings
function translate(lang, inputs, globals, body, warnings, name) {
  const params = new Set(["time", "value", ...inputs.slice(1).map((_, i) => `v${i + 1}`)]);
  const problems = [];
  const toJs = lang === "lua" ? lua : py;
  const g = toJs(globals || "", problems), f = body === null ? (problems.push("not a per-sample function"), "") : toJs(body, problems);
  const gNames = assigned(g);
  const fNames = assigned(f).filter((n) => !gNames.includes(n) && !params.has(n));
  const head = [`// translated from ${lang === "lua" ? "PlotJuggler Lua" : "jotpluggler Python"}`, ...problems.map((p) => `// could not translate: ${p}`)];
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

function luaExpr(s, problems) {
  s = s.replace(/\bmath\.atan\(([^()]*(?:\([^()]*\)[^()]*)*),/g, "Math.atan2($1,");
  s = mathCall("math")(s)
    .replace(/~=/g, "!=").replace(/\band\b/g, "&&").replace(/\bor\b/g, "||").replace(/\bnot\b/g, "!")
    .replace(/\bnil\b/g, "null").replace(/\^/g, "**");
  if (/\.\.|#|\bfunction\b/.test(s)) problems.push(s.trim());
  return s;
}

function lua(src, problems) {
  const out = [];
  for (const raw of src.split("\n")) {
    const dash = raw.indexOf("--"), comment = dash >= 0 ? ` //${raw.slice(dash + 2)}` : "";
    const line = (dash >= 0 ? raw.slice(0, dash) : raw).trimEnd(), t = line.trim(), pad = line.slice(0, line.length - line.trimStart().length);
    if (!t) { out.push({comment: comment.trim()}); continue; }
    let m;
    if ((m = t.match(/^if\s+(.+)\s+then\s+(.+)\s+end$/))) out.push({pad, js: `if ${cond(luaExpr(m[1], problems))} { ${luaExpr(m[2], problems)}; }`, comment});
    else if ((m = t.match(/^if\s+(.+)\s+then$/))) out.push({pad, js: `if ${cond(luaExpr(m[1], problems))} {`, comment});
    else if ((m = t.match(/^elseif\s+(.+)\s+then$/))) out.push({pad, js: `} else if ${cond(luaExpr(m[1], problems))} {`, comment});
    else if (t === "else") out.push({pad, js: "} else {", comment});
    else if (t === "end") out.push({pad, js: "}", comment});
    else if (/^(for|while|repeat|until|function)\b/.test(t)) { problems.push(t); out.push({pad, js: `// ${t}`}); }
    else out.push({pad, js: tuple(luaExpr(t.replace(/^local\s+/, ""), problems)), comment, stmt: true});
  }
  return emit(out);
}
