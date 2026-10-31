// what the CAN panes share: the route's store.can, following the playhead, dialogs and small formatters
import {css, h} from "/dom.js";
import {icon} from "/icons.js";
import {addrOf, busOf} from "/data/can.js";
import {MUX, MUXED, NODE, cloneMsg, flip, newSignal, newSignalName} from "/data/dbc.js";

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

// a phone stacks the view's panes in one scroll (pages.css): this scrolls the pane whose root has class cls into view
export function reveal(el, cls) {
  const layout = el.closest(".layout");
  if (layout && getComputedStyle(layout).display === "block") layout.querySelector(`.${cls}`)?.closest(".pane").scrollIntoView({behavior: "smooth"});
}

export const setText = (el, text) => { if (el.textContent !== text) el.textContent = text; };
export const hex2 = (x) => x.toString(16).toUpperCase().padStart(2, "0");
// a signal color dark enough for text and lines on either theme: its lightness set to l%
export const ink = (color, l = 45) => color.replace(/(\d+(\.\d+)?)%\)$/, `${l}%)`);

// a modal dialog; close() removes it
export function dialog(title, ...body) {
  const el = h("dialog", {class: "can-dialog", onclose: () => el.remove()},
    h("div", {class: "can-dialog-head"}, h("b", {}, title), h("button", {class: "can-icon can-x", title: "Close", onclick: () => el.close()}, icon("close", 14))),
    h("div", {class: "can-dialog-body"}, body));
  document.body.append(el);
  el.showModal();
  return el;
}

// a labelled input row for forms
export const field = (label, input, hint) => h("label", {class: "can-field", title: hint || ""}, h("span", {}, label), input);

// edits message k through the undo stack: fn(m) changes a copy (a new message if k has none), or returns false to
// change nothing. Returns whether it changed
export function editMsg(can, k, text, fn) {
  const m = cloneMsg(can.dbcMsg(k)) || newMsg(can, k);
  if (fn(m) === false) return false;
  can.dbcs.edit(busOf(k), m.address, m, text);
  return true;
}

// the DBC message a received one starts as
export function newMsg(can, k) {
  const address = addrOf(k);
  return {address, name: `NEW_MSG_${address.toString(16).toUpperCase()}`, size: can.msg(k)?.size || 8, transmitter: NODE, comment: "", sigs: []};
}

export function createSignal(can, k, props) {
  let name;
  editMsg(can, k, "add signal", (m) => {
    name = newSignalName(m);
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
  return editMsg(can, k, `edit signal ${name}`, (m) => {
    const i = m.sigs.findIndex((x) => x.name === name);
    if (i < 0 || (props.name !== undefined && props.name !== name && m.sigs.some((x) => x.name === props.name))) return false;
    // a new endianness keeps the signal on its bytes, as cabana's saveSignal
    const start = props.le !== undefined && props.le !== m.sigs[i].le && props.start === undefined ? flip(m.sigs[i].start) : m.sigs[i].start;
    m.sigs[i] = {...m.sigs[i], start, ...props};
  });
}
