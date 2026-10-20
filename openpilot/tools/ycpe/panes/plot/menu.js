// popups for the plot panes: context menus and the curve editor
import {h} from "/dom.js";
import {STYLES} from "./draw.js";
import {icon} from "/icons.js";
import {TRANSFORMS} from "/data/transforms.js";

let open = null, opener = null;  // the popup, and what had the focus before it

// focus inside the popup goes back where it came from
export function close() {
  if (open?.contains(document.activeElement)) opener?.focus({preventScroll: true});
  open?.remove();
  open = opener = null;
}

// a floating panel at client (x, y), kept on screen; closes on an outside press or Escape
export function popup(x, y, ...content) {
  close();
  opener = document.activeElement;
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

// items: [{label, fn, checked?, disabled?} | null (separator)]; it takes the focus, and the arrow keys move it
export function menu(x, y, items) {
  const rows = items.map((it) => it ? h("button", {class: `menu-row ${it.disabled ? "muted" : ""}`, role: "menuitem", disabled: !!it.disabled, onclick: () => {
    close();
    it.fn();
  }}, h("span", {class: "tick"}, it.checked && icon("check", 14)), it.label) : h("hr"));
  const list = h("div", {class: "menu-list", role: "menu", onkeydown: (e) => {
    const live = [...list.querySelectorAll("button:enabled")], k = live.indexOf(document.activeElement);
    const to = {ArrowDown: k + 1, ArrowUp: k - 1, Home: 0, End: live.length - 1}[e.key];
    if (to === undefined) return;
    e.preventDefault();
    live[(to + live.length) % live.length]?.focus();
  }}, rows);
  const el = popup(x, y, list);
  list.querySelector("button:enabled")?.focus();
  return el;
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
  const el = popup(x, y,
    h("div", {class: "title"}, curve.label || curve.path),
    h("label", {}, "label", h("input", {value: curve.label || "", placeholder: curve.path.slice(1), spellcheck: false, oninput: (e) => set("label")(e.target.value.trim() || undefined)})),
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
  el.querySelector("input").focus();
  return el;
}
