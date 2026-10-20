// popups for the plot panes: context menus and the curve editor
import {h} from "/dom.js";
import {STYLES} from "./draw.js";
import {TRANSFORMS} from "/data/transforms.js";

let open = null;

export function close() {
  open?.remove();
  open = null;
}

// a floating panel at client (x, y), kept on screen; closes on an outside press or Escape
export function popup(x, y, ...content) {
  close();
  const el = open = h("div", {class: "popover plot-popup", onpointerdown: (e) => e.stopPropagation()}, ...content);
  document.body.append(el);
  return place(el, x, y);
}

// el at (x, y), moved on screen; again after a change of its size
export function place(el, x, y) {
  const r = el.getBoundingClientRect();
  el.style.left = `${Math.max(4, Math.min(x, innerWidth - r.width - 4))}px`;
  el.style.top = `${Math.max(4, Math.min(y, innerHeight - r.height - 4))}px`;
  return el;
}
addEventListener("pointerdown", close);
addEventListener("blur", close);
addEventListener("keydown", (e) => { if (e.key === "Escape" && open) close(); });  // wherever the focus is (the plot, say)

// items: [{label, fn, checked?, disabled?} | null (separator)]
export function menu(x, y, items) {
  return popup(x, y, items.map((it) => it ? h("div", {class: `menu-row ${it.disabled ? "muted" : ""}`, onclick: () => {
    if (it.disabled) return;
    close();
    it.fn();
  }}, h("span", {class: "tick"}, it.checked ? "✓" : ""), it.label) : h("hr")));
}

const num = (label, value, set, step = "any") => h("label", {}, label,
  h("input", {type: "number", step, value, oninput: (e) => e.target.value !== "" && isFinite(e.target.value) && set(+e.target.value)}));

// edits one curve in place; change() after every edit
// actions: [{label, fn}] run from buttons, after the editor closes
export function curveEditor(x, y, curve, {change, remove, code, actions = []}) {
  const set = (k) => (v) => { curve[k] = v; change(); };
  const param = h("span");
  const showParam = () => {
    const spec = TRANSFORMS[curve.transform?.kind];
    param.replaceChildren(spec?.param ? num(spec.param.label, curve.transform.param ?? spec.param.value, (v) => { curve.transform.param = v; change(); }) : "");
  };
  showParam();
  return popup(x, y,
    h("div", {class: "title"}, curve.label || curve.path),
    h("label", {}, "color", h("input", {type: "color", value: curve.color, oninput: (e) => set("color")(e.target.value)})),
    h("label", {}, "style", h("select", {onchange: (e) => set("style")(e.target.value || undefined)},
      h("option", {value: ""}, "auto"), STYLES.map((s) => h("option", {value: s, selected: curve.style === s}, s)))),
    num("scale", curve.scale ?? 1, set("scale")),
    num("offset", curve.offset ?? 0, set("offset")),
    num("time shift (s)", curve.shift ?? 0, set("shift")),
    h("label", {}, "transform", h("select", {onchange: (e) => {
      const kind = e.target.value;
      curve.transform = kind ? {kind, param: TRANSFORMS[kind].param?.value} : undefined;
      showParam();
      change();
    }}, h("option", {value: ""}, "none"), Object.entries(TRANSFORMS).map(([k, s]) => h("option", {value: k, selected: curve.transform?.kind === k}, s.label)))),
    param,
    code && h("button", {onclick: code}, "Edit code…"),
    actions.map(({label, fn}) => h("button", {onclick: () => { close(); fn(); }}, label)),
    h("button", {class: "danger", onclick: () => { close(); remove(); }}, "Remove curve"));
}
