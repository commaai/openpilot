// connect's material pieces for user mode's pairing, prime, clips and drives (Material UI v1 in connect's dark theme):
// the mode's icons, the circular and linear progress, segmented buttons, menus, tooltips, and a modal on a paper
import {h, css, openModal, popover} from "../dom.js";
import {icon} from "../icons.js";
import {cnIcon} from "./connect/icons.js";
import {userMode} from "./connect/mode.js";

await css(new URL("./cnui.css", import.meta.url));

// replaceChildren that skips what isn't there and flattens lists, as h() does
export const fill = (el, ...children) => el.replaceChildren(...children.flat(Infinity).filter((c) => c != null && c !== false));

const NS = "http://www.w3.org/2000/svg";
const svg = (tag, attrs) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
};

// an icon in the mode's set: connect's (connect/icons.js) in user mode, cnName at cnSize px, else ours, name at size px
export const modeIcon = (name, size, cnName = name, cnSize = size) => userMode() ? cnIcon(cnName, cnSize) : icon(name, size);

// Material UI's indeterminate circular progress: size px, in color (connect's primary blue unless said), its ring
// thickness px of 44
export function progress(size = 40, color = "", thickness = 3.6) {
  const el = svg("svg", {viewBox: "22 22 44 44", width: size, height: size, class: "cnui-progress", role: "progressbar"});
  if (color) el.style.color = color;
  el.append(svg("circle", {cx: 44, cy: 44, r: (44 - thickness) / 2, fill: "none", "stroke-width": thickness}));
  return el;
}

// busy: Material UI's circular progress in user mode (size px, in color), the theme's spinner elsewhere (its big one
// from 32 px)
export const spinner = (size, color) => userMode() ? progress(size, color) : h("span", {class: size >= 32 ? "spinner big" : "spinner"});

// connect's own spinner, where it isn't Material UI's
export const ring = (size, label) => h("span", {class: "cnui-ring", style: `--size: ${size}px`, role: "progressbar", "aria-label": label});

// Material UI's linear progress: indeterminate, or a share (0..1). Its cycle runs on the page's clock, so one drawn
// again (a list rebuilt) carries on where the last left off
export function linear(share = null) {
  const el = h("div", {class: `cnui-linear${share === null ? " busy" : ""}`, role: "progressbar"}, h("i"), h("i"));
  if (share === null) el.style.setProperty("--t", `${-Math.round(performance.now() % 2100)}ms`);
  else el.style.setProperty("--p", share);
  return el;
}

// connect's Notification over the map's top right: a title and its link a, text under them, and a round x that
// dismisses it
export const promo = (label, title, a, text, dismiss) => h("div", {class: "cnui-promo-slot"}, h("aside", {class: "cnui-promo", "aria-label": label},
  h("button", {class: "cnui-promo-x", "aria-label": `Dismiss ${label.toLowerCase()}`, onclick: dismiss}, cnIcon("clear", 12)),
  h("div", {class: "cnui-promo-head"}, h("b", {}, title), a), h("p", {}, text)));

// a modal on connect's paper: Escape or a click beside it closes it unless busy() says not now, then onclose runs
// once. kind is the paper's class (its size and place); the backdrop fades in as Material UI's does
export function cnModal(label, kind, onclose = () => {}, busy = () => false) {
  const paper = h("div", {class: `cnui-paper ${kind}`});
  return {el: openModal({class: "cnui-modal", "aria-label": label, autofocus: true, onclose, busy}, paper), paper};
}

// a modal's title, its caption (a dongle id) across from it, and the rule under them
export const cnTitle = (text, caption) => [h("div", {class: "cnui-head"}, h("h2", {class: "cnui-title"}, text), caption && h("span", {}, caption)), h("hr")];

// connect's menu (Material UI's Menu): its paper of class kind, holding children, over a transparent backdrop a click
// closes it from (that click doing nothing else), as Escape does. place(anchor's rect, the paper's size) → [left, top,
// origin x, origin y] puts it, kept margin px inside the window; with grow, it grows in from the origin, and fades out.
// The paper's first [tabindex="0"] takes the focus, else the paper. onclose runs as it closes. Returns {paper, close}
export function cnMenu(anchor, kind, children, place, {margin = 16, grow = true, onclose} = {}) {
  const paper = h("div", {class: `cnui-menu ${kind}${grow ? " grow" : ""}`, tabindex: -1, autofocus: true}, children);
  let closed = false, hide = null;
  const done = () => {
    if (closed) return;
    closed = true;
    onclose?.();
  };
  const close = () => {
    if (closed) return;
    done();
    if (!grow) return hide();
    paper.classList.remove("open");
    layer.classList.add("closing");
    setTimeout(hide, 300);
  };
  const layer = popover(anchor, (hidden) => { hide = hidden; return [h("div", {class: "cnui-backdrop", onclick: close}), paper]; }, {
    class: "cnui-menu-layer", light: false, onclose: done, place: () => {
      // read once as it opens, a click away from any render; unscaled, as it grows in
      const a = anchor.getBoundingClientRect(), r = {width: paper.offsetWidth, height: paper.offsetHeight}, [x, y, ox, oy] = place(a, r);
      const left = Math.max(margin, Math.min(innerWidth - margin - r.width, x)), top = Math.max(margin, Math.min(innerHeight - margin - r.height, y));
      Object.assign(paper.style, {left: `${left}px`, top: `${top}px`, transformOrigin: `${ox + x - left}px ${oy + y - top}px`});
      paper.classList.add("open");
    }});
  if (!layer) return null;  // it was open
  paper.querySelector("[tabindex='0']")?.focus({preventScroll: true});
  return {paper, close};
}

// connect's tooltips, over el (which it returns) as the pointer is on it: "below", the poppers' rounded grey one, kept
// in the window and toggled by a press too; "above", the info icons' dark bubble with its arrow, as "below"; "pill",
// Material UI's under the dashboard's pills, fading in. The first two are in the top layer, over the menu they're in;
// a pill's is in the pill, as nothing is over the pills (in the top layer its text would lose its subpixel smoothing,
// and in the page's its overflow would widen the page). Again on the same el, its text is replaced
const tips = new WeakMap();  // el → its text
export function cnTip(el, text, kind = "below") {
  const had = tips.has(el);
  tips.set(el, text);
  if (had) return el;
  let pop = null;
  const show = () => {
    if (pop) return;
    if (kind === "pill") {
      pop = h("span", {class: "cnui-tip pill", role: "tooltip"}, tips.get(el));
      el.append(pop);
      requestAnimationFrame(() => pop?.classList.add("open"));
      return;
    }
    pop = h("div", {class: `cnui-tip ${kind}`, role: "tooltip", popover: "manual"}, tips.get(el));
    document.body.append(pop);
    pop.showPopover();
    const r = el.getBoundingClientRect(), middle = r.left + r.width / 2;
    const p = pop.getBoundingClientRect(), x = Math.max(8, Math.min(innerWidth - 8 - p.width, middle - p.width / 2));
    Object.assign(pop.style, {left: `${x}px`, top: `${kind === "above" ? r.top - p.height - 8 : r.bottom + 12}px`});
    pop.style.setProperty("--arrow", `${middle - x}px`);
  };
  const hide = () => {
    const was = pop;
    pop = null;
    if (kind !== "pill") return was?.remove();
    was?.classList.remove("open");
    setTimeout(() => was?.remove(), 150);
  };
  el.addEventListener("pointerenter", show);
  el.addEventListener("pointerleave", hide);
  if (kind !== "pill") el.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (pop) hide();
    else show();
  });
  return el;
}

// connect's segmented buttons, the group of class group and each button of class button: options [value, label,
// detail?], the label in a span (connect's menus open from it) and the detail in a small under it. pick(value, span)
// on a press; set(on, disabled?) marks the one chosen
export function segmented([group, button], options, pick) {
  const buttons = options.map(([value, label, detail]) => h("button", {class: button, "data-key": value, "aria-pressed": "false",
    onclick: (e) => pick(value, e.currentTarget.firstChild)}, h("span", {}, label), detail && h("small", {}, detail)));
  return {el: h("div", {class: group}, buttons), set(on, disabled = () => false) {
    options.forEach(([value], i) => { buttons[i].ariaPressed = String(value === on); buttons[i].disabled = disabled(value); });
  }};
}
