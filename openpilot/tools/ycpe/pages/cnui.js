// connect's material pieces for user mode's pairing, prime, clips and drives (Material UI v1 in connect's dark theme):
// its icons, the circular and linear progress, segmented buttons, and a modal on a paper
import {h, css, openModal} from "../dom.js";
import {cnIcon as connectIcon} from "./connect/icons.js";

await css(new URL("./cnui.css", import.meta.url));

// shown in user mode: connect's look; developer mode keeps its own
export const userMode = () => location.pathname.startsWith("/connect");

// replaceChildren that skips what isn't there and flattens lists, as h() does
export const fill = (el, ...children) => el.replaceChildren(...children.flat(Infinity).filter((c) => c != null && c !== false));

const NS = "http://www.w3.org/2000/svg";
const svg = (tag, attrs) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
};

// a connect icon (connect/icons.js), size px square, that a flex row doesn't squeeze
export function cnIcon(name, size = 24) {
  const el = connectIcon(name, size);
  el.classList.add("cnui-icon");
  return el;
}

// Material UI's indeterminate circular progress: size px, in color (connect's primary blue unless said)
export function progress(size = 40, color = "") {
  const el = svg("svg", {viewBox: "22 22 44 44", width: size, height: size, class: "cnui-progress", role: "progressbar"});
  if (color) el.style.color = color;
  el.append(svg("circle", {cx: 44, cy: 44, r: 20.2, fill: "none", "stroke-width": 3.6}));
  return el;
}

// busy: Material UI's circular progress in user mode (size px, in color), the theme's spinner elsewhere
export const spinner = (size, color) => userMode() ? progress(size, color) : h("span", {class: "spinner"});

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
