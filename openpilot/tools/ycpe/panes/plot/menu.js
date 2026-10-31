// popups for the plot panes: context menus and the curve editor
import {h, popover, closePopover, menuList} from "/dom.js";
import {STYLES} from "./draw.js";
import {TRANSFORMS} from "/data/transforms.js";

let open = null;  // the last popup opened here: the pane going closes it
export const close = () => closePopover(open);

// a form of content at client (x, y)
export const popup = (x, y, content, attrs = {}) => open = popover({x, y}, () => content, {...attrs, class: `plot-popup ${attrs.class ?? ""}`});

// a context menu at client (x, y): items as menuList's
export const menu = (x, y, items) => open = popover({x, y}, () => menuList(items));

const PARAM_MS = 300;  // a transform parameter's settling time

const num = (label, value, set, step = "any") => h("label", {}, label,
  h("input", {type: "number", step, value, oninput: (e) => e.target.value !== "" && isFinite(e.target.value) && set(+e.target.value)}));

// edits one curve in place; change() after every edit
export function curveEditor(x, y, curve, {change, remove, code}) {
  const set = (k) => (v) => { curve[k] = v; change(); };
  // each parameter value is a series of its own: one is computed once the value settles
  const param = h("span");
  let paramTimer = 0;
  const setParam = (v) => {
    clearTimeout(paramTimer);
    paramTimer = setTimeout(() => { curve.transform.param = v; change(); }, PARAM_MS);
  };
  const showParam = () => {
    clearTimeout(paramTimer);
    const spec = TRANSFORMS[curve.transform?.kind];
    param.replaceChildren(spec?.param ? num(spec.param.label, curve.transform.param ?? spec.param.value, setParam) : "");
  };
  showParam();
  return popup(x, y, [
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
    h("button", {class: "danger", onclick: () => { close(); remove(); }}, "Remove curve")]);
}
