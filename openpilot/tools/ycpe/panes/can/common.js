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

export const hex2 = (x) => x.toString(16).toUpperCase().padStart(2, "0");
// a signal color dark enough for text and lines on either theme
export const ink = (color) => color.replace(/(\d+(\.\d+)?)%\)$/, "45%)");

// a modal dialog; close() removes it
export function dialog(title, ...body) {
  const el = h("dialog", {class: "can-dialog", onclose: () => el.remove()},
    h("div", {class: "can-dialog-head"}, h("b", {}, title), h("button", {class: "can-x", title: "Close", onclick: () => el.close()}, icon("close", 14))),
    h("div", {class: "can-dialog-body"}, body));
  document.body.append(el);
  el.showModal();
  return el;
}

// a popover menu under anchor, focused on its first row; the arrow keys move between rows. rows are menuItem()s,
// menuCheck()s (which keep it open) or separators. Focus goes back to anchor when it closes
export function popup(anchor, rows) {
  const items = () => [...menu.querySelectorAll("[role^=menuitem]")];
  const menu = h("div", {class: "can-menu", role: "menu", popover: "auto", ontoggle: (e) => {
    if (e.newState !== "closed") return;
    menu.remove();
    anchor.setAttribute("aria-expanded", "false");
  }, onkeydown: (e) => {
    const all = items(), k = all.indexOf(document.activeElement);
    const to = {ArrowDown: k + 1, ArrowUp: k - 1, Home: 0, End: all.length - 1}[e.key];
    if (to === undefined) return;
    e.preventDefault();
    all[(to + all.length) % all.length]?.focus();
  }}, rows);
  document.body.append(menu);
  const r = anchor.getBoundingClientRect();
  menu.style.left = `${r.left}px`;
  menu.style.top = `${r.bottom + 2}px`;
  menu.showPopover({source: anchor});
  anchor.setAttribute("aria-expanded", "true");
  items()[0]?.focus();
  return menu;
}
export const menuItem = (label, fn) => h("button", {class: "menu-row", role: "menuitem", onclick: (e) => { e.currentTarget.closest("[popover]").hidePopover(); fn(); }}, label);
export function menuCheck(label, checked, fn) {
  const tick = h("span", {class: "tick"});
  const set = (on) => { row.setAttribute("aria-checked", String(checked = on)); tick.replaceChildren(on ? icon("check", 14) : ""); };
  const row = h("button", {class: "menu-row", role: "menuitemcheckbox", onclick: () => { set(!checked); fn(checked); }}, tick, label);
  set(checked);
  return row;
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
