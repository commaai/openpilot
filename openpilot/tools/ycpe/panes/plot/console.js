// the custom series console: input paths (the first is the time base), a globals block that runs once, and a
// function body that runs per sample in a worker (data/custom.js)
import {h} from "/dom.js";
import {popup, place, close} from "./menu.js";
import {acceptPaths} from "./view.js";
import {TEMPLATES} from "/data/custom.js";

// spec = {name, inputs, globals, code}; run(spec) resolves when the series is computed, or rejects with the error
export function openConsole(x, y, spec, run) {
  const name = h("input", {value: spec.name || "", placeholder: "name", spellcheck: false});
  const inputs = h("textarea", {rows: 3, spellcheck: false, placeholder: "/carState/vEgo\n(one path per line, or drop series here)"}, (spec.inputs || []).join("\n"));
  const globals = h("textarea", {rows: 3, spellcheck: false, placeholder: "// runs once; its variables persist across samples"}, spec.globals || "");
  const code = h("textarea", {rows: 8, spellcheck: false, placeholder: "return value * 3.6;"}, spec.code || "");
  const error = h("div", {class: "error"}), size = h("span"), warning = h("div", {class: "error"});
  // all of the code shows, however it's padded, since it runs with this page's powers (see data/custom.js)
  const fit = () => {
    const n = code.value.split("\n").length;
    globals.rows = Math.max(3, globals.value.split("\n").length);
    code.rows = Math.max(8, n);
    size.textContent = ` · ${n} line${n === 1 ? "" : "s"}`;
    warning.textContent = /\b(fetch|import|importScripts|XMLHttpRequest|WebSocket|EventSource|eval|Function)\b/.test(globals.value + code.value)
      ? "this code can reach the network or load more code: read all of it before running" : "";
  };
  globals.addEventListener("input", fit);
  code.addEventListener("input", fit);
  fit();
  const paths = () => inputs.value.split("\n").map((s) => s.trim()).filter(Boolean);
  acceptPaths(inputs, (ps) => { inputs.value = [...paths(), ...ps].join("\n"); });

  const submit = () => {
    const s = {name: name.value.trim().replaceAll("/", "_"), inputs: paths(), globals: globals.value, code: code.value};
    if (!s.name || !s.inputs.length || !s.code.trim()) return void (error.textContent = "name, at least one input and code are required");
    error.textContent = "running…";
    run(s).then(close, (e) => { error.textContent = e.message; });
  };

  const el = popup(x, y,
    h("div", {class: "title"}, "Custom series"),
    warning,
    h("label", {}, "name", name),
    h("label", {}, "template", h("select", {onchange: (e) => {
      const t = TEMPLATES[e.target.value];
      if (!t) return;
      globals.value = t.globals;
      code.value = t.code;
      fit();
      while (paths().length < (t.inputs || 1)) inputs.value += (inputs.value && !inputs.value.endsWith("\n") ? "\n" : "") + "/";
    }}, h("option", {value: ""}, "…"), Object.keys(TEMPLATES).map((k) => h("option", {value: k}, k)))),
    h("div", {class: "hint"}, "inputs: the first is the time base; the others are resampled onto it as v1 … vN"),
    inputs,
    h("div", {class: "hint"}, "globals"),
    globals,
    h("div", {class: "hint"}, "function (time, value, v1 … vN) {"),
    code,
    h("div", {class: "hint"}, "}", size),
    error,
    h("button", {onclick: submit, title: "ctrl+enter"}, "Run"));
  el.classList.add("plot-console");
  place(el, x, y);  // console-sized now
  el.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit(); });
  (spec.name ? code : name).focus();
}
