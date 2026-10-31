// the layout picker: open a reference layout, import a file (ours or jotpluggler's), open one
// from a URL, or export the view's layout. Custom code in an opened layout stays untrusted until the user
// reviews it in the code console, except in the bundled references (commaai/openpilot's own layouts).
import {h, css, download} from "/dom.js";
import {importLayout, exportLayout, uniqueCustoms, customsOf} from "./import.js";

css(new URL("./picker.css", import.meta.url));

// the reference layouts (ref/), from jotpluggler
const REFERENCES = ["cameras-and-map", "camera-timings", "CAN-bus-debug", "can-states", "controls_mismatch_debug", "driver-monitoring-debug",
  "gps", "gps_vs_llk", "locationd_debug", "longitudinal", "max-torque-debug", "new-layout", "system_lag_debug", "thermal_debug",
  "torque-controller", "tuning", "ublox-debug"];
const refUrl = (name) => new URL(`./ref/${name}.json`, import.meta.url).href;
const TRUSTED = new Set(REFERENCES.map(refUrl));

// src: a File, a URL, the layout text itself or a reference's name (as ?layout= may hold any of those, like
// jotpluggler --layout takes a name) → {layout, warnings}
export async function loadLayout(src) {
  if (src instanceof Blob) return named(importLayout(await src.text()), src.name);
  if (/^\s*\{/.test(src)) return importLayout(src);
  if (REFERENCES.includes(src)) src = refUrl(src);
  const r = await fetch(src);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  const opened = named(importLayout(await r.text()), new URL(src, location.href).pathname.split("/").pop());
  if (TRUSTED.has(src)) trust(opened.layout);
  return opened;
}

function trust(layout) {
  const walk = (n) => n.children ? n.children.forEach(walk) : n.state?.curves?.forEach((c) => { if (c.custom) delete c.custom.untrusted; });
  layout.tabs.forEach((t) => walk(t.root));
}

// tabs with generic names ("tab1") are named after the layout's file instead
function named(opened, file) {
  const base = file?.replace(/\.json$/i, "");
  const generic = opened.layout.tabs.filter((t) => !t.name || /^tab\s*\d*$/i.test(t.name));
  if (base) generic.forEach((t, i) => { t.name = generic.length > 1 ? `${base} ${i + 1}` : base; });
  return opened;
}

// the view's model with an opened one: its tabs added after the view's own (the first selected), or instead of them.
// Opened custom series named like different ones are renamed
export function mergeLayout(current, opened, how) {
  if (!opened.tabs.length) throw new Error("the layout has no tabs");
  uniqueCustoms(opened, how === "replace" ? new Map() : customsOf(current));
  if (how === "replace") return opened;
  return {...current, active: current.tabs.length, tabs: [...current.tabs, ...opened.tabs]};
}

// this page (or href) with the layout in ?layout=
export function shareUrl(layout, href = location.href) {
  const url = new URL(href, location.href);
  url.searchParams.set("layout", JSON.stringify({ycpe: 1, ...layout}));
  return url.href;
}

// files dropped on el open as new tabs; open(file) does the rest (loadLayout, mergeLayout, apply)
export function acceptLayoutFiles(el, open) {
  const isFile = (e) => e.dataTransfer?.types.includes("Files");
  el.addEventListener("dragover", (e) => { if (isFile(e)) e.preventDefault(); });
  el.addEventListener("drop", (e) => {
    const files = [...(e.dataTransfer?.files || [])].filter((f) => /\.json$/i.test(f.name));
    if (!files.length) return;
    e.preventDefault();
    for (const f of files) open(f);
  });
}

// the picker's panel. current(): the view's layout model; apply(model): installs a new one; close(): dismisses
export function layoutPicker({current, apply, close, name = "layout"}) {
  let how = "tab";
  const status = h("div", {class: "picker-status"});
  async function open(label, src) {
    status.textContent = `opening ${label}…`;
    try {
      const {layout, warnings} = await loadLayout(src);
      apply(mergeLayout(current(), layout, how));
      if (!warnings.length) return close();
      status.replaceChildren(`opened ${label} with ${warnings.length} warning${warnings.length > 1 ? "s" : ""}:`,
        h("ul", {}, warnings.map((w) => h("li", {}, w))));
    } catch (e) {
      status.textContent = `${label}: ${e.message}`;
    }
  }

  const mode = (value, label) => h("label", {}, h("input", {type: "radio", name: "picker-how", value, checked: value === how,
    onchange: () => { how = value; }}), label);
  const file = h("input", {type: "file", accept: ".json", hidden: true,
    onchange: () => { if (file.files[0]) open(file.files[0].name, file.files[0]); }});
  const url = h("input", {type: "url", placeholder: "layout URL", onkeydown: (e) => { if (e.key === "Enter") go(); }});
  const go = () => url.value.trim() && open(url.value.trim(), url.value.trim());

  return h("div", {class: "layout-picker"},
    h("div", {class: "picker-how"}, "open as", mode("tab", "new tabs"), mode("replace", "this view")),
    h("div", {class: "picker-row"},
      h("button", {onclick: () => file.click()}, "Import file…"),
      h("button", {onclick: () => download(new Blob([exportLayout(current())], {type: "application/json"}), `${name}.json`)}, "Export"),
      h("button", {title: "Copy a link that opens this page with this layout", onclick: async () => {
        await navigator.clipboard.writeText(shareUrl(current()));
        status.textContent = "Link copied: opens this route at the playhead with these tabs. Custom series in it ask before they run.";
      }}, "Copy link with these tabs"),
      file),
    h("div", {class: "picker-row"}, url, h("button", {onclick: go}, "Open")),
    status,
    h("div", {class: "picker-refs"}, h("div", {class: "picker-group"}, "jotpluggler layouts"),
      REFERENCES.map((n) => h("button", {class: "menu-row", onclick: () => open(n, refUrl(n))}, n))));
}
