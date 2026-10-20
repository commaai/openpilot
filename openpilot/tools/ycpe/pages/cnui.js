// connect's material pieces for user mode's pairing, prime and clips (Material UI v1 in connect's dark theme): its
// icons (connect's own paths, some on a 960 grid), the circular and linear progress, and a modal on a paper
import {h, css} from "../dom.js";

await css(new URL("./cnui.css", import.meta.url));

// shown in user mode: connect's look; developer mode keeps its own
export const userMode = () => location.pathname.startsWith("/connect");

// replaceChildren that skips what isn't there and flattens lists, as h() does
export const fill = (el, ...children) => el.replaceChildren(...children.flat(Infinity).filter((c) => c != null && c !== false));

const ICONS = {
  back: ["0 -960 960 960", "M360-240 120-480l240-240 56 56-144 144h568v80H272l144 144-56 56Z"],
  check: ["0 0 24 24", "M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"],
  clear: ["0 0 24 24", "M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"],
  closeBold: ["0 0 24 24", "m6.225 19.425-1.65-1.65L10.35 12 4.575 6.225l1.65-1.65L12 10.35l5.775-5.775 1.65 1.65L13.65 12l5.775 5.775-1.65 1.65L12 13.65l-5.775 5.775Z"],
  download: ["0 0 24 24", "m12 16.175-4.825-4.825 1.075-1.075 3 3V4h1.5v9.275l3-3 1.075 1.075L12 16.175ZM5.5 20q-.6 0-1.05-.45T4 18.5v-3.575h1.5V18.5h13v-3.575H20V18.5q0 .6-.45 1.05T18.5 20h-13Z"],
  trash: ["0 0 24 24", "M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12Zm2-10h2v8H8V9Zm4 0h2v8h-2V9Zm-7-5v2h14V4h-3.5l-1-1h-5l-1 1H5Z"],
  play: ["0 -960 960 960", "M320-203v-560l440 280-440 280Z"],
  info: ["0 -960 960 960", "M453-280h60v-240h-60v240Zm27-314q14 0 23-9t10-23q0-14-9-24-10-10-24-10t-23 10-10 24q0 14 9 23 10 9 24 9Zm0 514q-82 0-155-31t-128-86q-54-55-86-128-31-73-31-155 0-83 32-156 31-73 86-127t127-85q73-32 156-32 82 0 155 32 73 31 127 85t86 127q31 73 31 156 0 82-31 155t-86 127q-54 55-127 86-73 32-156 32Zm0-60q142 0 241-99t99-241q0-142-99-241t-241-99q-141 0-240 99-100 99-100 241 0 141 100 241t241 99Zm0-340Z"],
  error: ["0 -960 960 960", "M480-280q14 0 24-9 9-10 9-24t-9-23q-10-10-24-10t-23 9q-10 10-10 24t9 23q10 10 24 10Zm-27-153h60v-253h-60v253Zm27 353q-82 0-155-31t-127-86q-55-55-86-128-32-73-32-155 0-83 32-156 31-73 86-127t127-85q73-32 156-32 82 0 155 32 73 31 127 85t86 127q31 73 31 156 0 82-31 155t-86 127q-54 55-127 86-73 32-156 32Zm1-60q141 0 240-99t99-241q0-142-99-241t-241-99q-141 0-240 99-100 99-100 241 0 141 100 241t241 99Zm-1-340Z"],
  priority: ["0 -960 960 960", "M480-120q-33 0-56.5-23.5T400-200q0-33 23.5-56.5T480-280q33 0 56.5 23.5T560-200q0 33-23.5 56.5T480-120Zm-80-240v-480h160v480H400Z"],
  cut: ["0 -960 960 960", "M760-120 480-400l-94 94q8 15 11 32t3 34q0 66-47 113T240-80q-66 0-113-47T80-240q0-66 47-113t113-47q17 0 34 3t32 11l94-94-94-94q-15 8-32 11t-34 3q-66 0-113-47T80-720q0-66 47-113t113-47q66 0 113 47t47 113q0 17-3 34t-11 32l494 494v40H760ZM600-520l-80-80 240-240h120v40L600-520ZM296.5-663.5Q320-687 320-720t-23.5-56.5Q273-800 240-800t-56.5 23.5Q160-753 160-720t23.5 56.5Q207-640 240-640t56.5-23.5ZM494-466q6-6 6-14t-6-14q-6-6-14-6t-14 6q-6 6-6 14t6 14q6 6 14 6t14-6ZM296.5-183.5Q320-207 320-240t-23.5-56.5Q273-320 240-320t-56.5 23.5Q160-273 160-240t23.5 56.5Q207-160 240-160t56.5-23.5Z"],
  add: ["0 -960 960 960", "M440-280h80v-160h160v-80H520v-160h-80v160H280v80h160v160Zm40 200q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm0-80q134 0 227-93t93-227q0-134-93-227t-227-93q-134 0-227 93t-93 227q0 134 93 227t227 93Zm0-320Z"],
};

const NS = "http://www.w3.org/2000/svg";
const svg = (tag, attrs) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
};

// a connect icon, size px square, in currentColor
export function cnIcon(name, size = 24) {
  const [viewBox, d] = ICONS[name];
  const el = svg("svg", {viewBox, width: size, height: size, "aria-hidden": "true", class: "cnui-icon"});
  el.append(svg("path", {d, fill: "currentColor"}));
  return el;
}

// Material UI's indeterminate circular progress: size px, in color (connect's primary blue unless said)
export function progress(size = 40, color = "") {
  const el = svg("svg", {viewBox: "22 22 44 44", width: size, height: size, class: "cnui-progress", role: "progressbar"});
  if (color) el.style.color = color;
  el.append(svg("circle", {cx: 44, cy: 44, r: 20.2, fill: "none", "stroke-width": 3.6}));
  return el;
}

// Material UI's linear progress: indeterminate, or a share (0..1). Its cycle runs on the page's clock, so one drawn
// again (a list rebuilt) carries on where the last left off
export function linear(share = null) {
  const el = h("div", {class: `cnui-linear${share === null ? " busy" : ""}`, role: "progressbar"}, h("i"), h("i"));
  if (share === null) el.style.setProperty("--t", `${-Math.round(performance.now() % 2100)}ms`);
  else el.style.setProperty("--p", share);
  return el;
}

// a modal on connect's paper: Escape or a click beside it closes it unless busy() says not now, then onclose runs
// once. kind is the paper's class (its size and place); the backdrop fades in as Material UI's does
export function cnModal(label, kind, onclose = () => {}, busy = () => false) {
  const paper = h("div", {class: `cnui-paper ${kind}`});
  const el = h("dialog", {class: "cnui-modal", "aria-label": label, autofocus: true, oncancel: (e) => busy() && e.preventDefault(),
    onclose: () => { el.remove(); onclose(); }, onclick: (e) => { if (e.target === el && !busy()) el.close(); }}, paper);
  document.body.append(el);
  el.showModal();
  return {el, paper};
}
