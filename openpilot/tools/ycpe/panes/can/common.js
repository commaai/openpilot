// what the CAN panes share: the route's store.can, following the playhead, dialogs and small formatters
import {css, h} from "/dom.js";
import {addrOf, busOf} from "/data/can.js";
import {MUX, MUXED, NODE, cloneMsg, flip, newSignal} from "/data/dbc.js";

await css(new URL("./can.css", import.meta.url));

// store.can, following the clock once per workspace: live bytes are recomputed (in the worker) as t moves
export function canOf(ctx) {
  const can = ctx.store.can;
  if (!can.clock) {
    can.clock = ctx.clock;
    ctx.clock.on("t", () => can.requestState(ctx.clock.t));
    can.on("messages", () => {
      can.requestState(ctx.clock.t);
      ctx.perf.mark("can frames", `${(can.bytes / 1e6).toFixed(0)} MB`);
    });
    can.requestState(ctx.clock.t);
  }
  return can;
}

// subscriptions that end with the pane
export function subs() {
  const offs = [];
  return {add: (off) => offs.push(off), dispose: () => offs.splice(0).forEach((f) => f())};
}

export const hex2 = (x) => x.toString(16).toUpperCase().padStart(2, "0");
// a signal color dark enough for text and lines on either theme
export const ink = (color) => color.replace(/(\d+(\.\d+)?)%\)$/, "45%)");

// a modal dialog; close() removes it
export function dialog(title, ...body) {
  const el = h("dialog", {class: "can-dialog", onclose: () => el.remove()},
    h("div", {class: "can-dialog-head"}, h("b", {}, title), h("button", {class: "can-x", title: "Close", onclick: () => el.close()}, "×")),
    h("div", {class: "can-dialog-body"}, body));
  document.body.append(el);
  el.showModal();
  return el;
}

// a popover menu under anchor; rows are menuItem()s or any element (a checkbox row keeps it open)
export function popup(anchor, rows) {
  const menu = h("div", {class: "can-menu", popover: "auto", ontoggle: (e) => { if (e.newState === "closed") menu.remove(); }}, rows);
  document.body.append(menu);
  const r = anchor.getBoundingClientRect();
  menu.style.left = `${r.left}px`;
  menu.style.top = `${r.bottom + 2}px`;
  menu.showPopover();
  return menu;
}
export const menuItem = (label, fn) => h("button", {class: "menu-row", onclick: (e) => { e.currentTarget.closest("[popover]").hidePopover(); fn(); }}, label);
export const menuCheck = (label, checked, fn) => h("label", {class: "menu-row"}, h("input", {type: "checkbox", checked, onchange: (e) => fn(e.target.checked)}), label);

// a labelled input row for forms
export const field = (label, input, hint) => h("label", {class: "can-field", title: hint || ""}, h("span", {}, label), input);

export function download(name, text) {
  const a = h("a", {href: URL.createObjectURL(new Blob([text], {type: "text/plain"})), download: name});
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// rgba() of a theme color with alpha, for canvases
export function alpha(color, a) {
  const c = document.createElement("canvas").getContext("2d");
  c.fillStyle = color;
  const v = c.fillStyle;
  if (v.startsWith("#")) {
    const n = parseInt(v.slice(1), 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
  }
  return v.replace(/rgba?\(([^)]+?)(,\s*[\d.]+)?\)/, `rgba($1,${a})`);
}

// edits message k through the undo stack: fn(m) changes a copy (a new message if k has none)
export function editMsg(can, k, text, fn) {
  const address = addrOf(k);
  const m = cloneMsg(can.dbcMsg(k)) || {address, name: `NEW_MSG_${address.toString(16).toUpperCase()}`, size: can.msg(k)?.size || 8,
                                         transmitter: NODE, comment: "", sigs: []};
  if (fn(m) === false) return;
  can.dbcs.edit(busOf(k), address, m, text);
}

export function createSignal(can, k, props) {
  let name;
  editMsg(can, k, "add signal", (m) => {
    let i = 1;
    while (m.sigs.some((s) => s.name === `NEW_SIGNAL_${i}`)) i++;
    name = `NEW_SIGNAL_${i}`;
    m.sigs.push(newSignal({...props, name, max: 2 ** props.size - 1}));
  });
  return name;
}

// removing a multiplexor also removes the signals it selects, as in cabana
export function removeSignal(can, k, name) {
  editMsg(can, k, `remove signal ${name}`, (m) => {
    const s = m.sigs.find((x) => x.name === name);
    m.sigs = m.sigs.filter((x) => x !== s && !(s.type === MUX && x.type === MUXED));
  });
}

export function updateSignal(can, k, name, props) {
  editMsg(can, k, `edit signal ${name}`, (m) => {
    const i = m.sigs.findIndex((x) => x.name === name);
    if (i < 0 || (props.name !== undefined && props.name !== name && m.sigs.some((x) => x.name === props.name))) return false;
    // a new endianness keeps the signal on its bytes, as cabana's saveSignal
    const start = props.le !== undefined && props.le !== m.sigs[i].le && props.start === undefined ? flip(m.sigs[i].start) : m.sigs[i].start;
    m.sigs[i] = {...m.sigs[i], start, ...props};
  });
}
