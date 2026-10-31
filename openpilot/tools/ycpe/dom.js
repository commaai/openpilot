import {icon} from "./icons.js";

// attributes that take "true"/"false" rather than being present or absent
const ENUMERATED = new Set(["draggable", "contenteditable", "spellcheck"]);

// the only DOM helper: h(tag, attrs, ...children)
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) el.setAttribute(k, v === true ? (ENUMERATED.has(k) ? "true" : "") : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c != null && c !== false) el.append(c instanceof Node ? c : String(c));
  }
  return el;
}

// subscribers: on(fn) adds one and returns what removes it, emit(...args) calls each
export function listeners() {
  const fns = new Set();
  return {
    on(fn) { fns.add(fn); return () => fns.delete(fn); },
    emit(...args) { for (const fn of fns) fn(...args); },
  };
}

// obj's state and detail, as a live source has them: the returned set(state, detail) changes them and calls
// obj.on("state", fn)'s fns with them; obj has no other events
export function stateful(obj) {
  const subs = listeners();
  obj.on = (event, fn) => event === "state" ? subs.on(fn) : () => {};
  return (state, detail = "") => {
    if (obj.state === state && obj.detail === detail) return;
    Object.assign(obj, {state, detail});
    subs.emit(state, detail);
  };
}

// async fn, one call in flight: calls meanwhile make one more once it's done, with the newest arguments
export function latest(fn) {
  let busy = false, next = null;
  return async (...args) => {
    next = args;
    if (busy) return;
    busy = true;
    try {
      while (next) {
        const now = next;
        next = null;
        await fn(...now);
      }
    } finally {
      busy = false;
    }
  };
}

// confirm(), in the theme (the browser's own dialog ignores it, see style.css): resolves with whether ok was chosen
export function ask(text, ok = "OK", cancel = "Cancel") {
  return new Promise((done) => {
    const el = h("dialog", {class: "ask", onclose: () => { el.remove(); done(el.returnValue === "ok"); }},
      h("form", {method: "dialog"}, h("p", {}, text),
        h("div", {class: "ask-buttons"}, h("button", {class: "button", value: ""}, cancel), h("button", {class: "button primary", value: "ok", autofocus: true}, ok))));
    document.body.append(el);
    el.showModal();
  });
}

// a note over the page: gone after 3 s, or, with an action ([label, fn]), up until the page goes
export function toast(text, action = null) {
  const el = h("div", {class: "toast", role: action ? "alert" : "status"}, text, action && h("button", {class: "button small", onclick: action[1]}, action[0]));
  document.body.append(el);
  if (!action) setTimeout(() => el.remove(), 3000);
}

// a modal <dialog> of attrs and children, shown. Escape or a click beside its content closes it, unless busy() says not
// now; once closed it's removed, then onclose runs
export function openModal({onclose, busy, ...attrs}, ...children) {
  const el = h("dialog", {...attrs, oncancel: (e) => busy?.() && e.preventDefault(), onclose: () => { el.remove(); onclose?.(); },
    onclick: (e) => { if (e.target === el && !busy?.()) el.close(); }}, ...children);
  document.body.append(el);
  el.showModal();
  return el;
}

// a floating panel in the top layer (the native popover API) at `at`, an element (under it) or a point {x, y}, kept on
// screen; build(close) makes what's in it. Escape or close() closes it and, if light, a press or the focus outside it
// or another light one opening; the focus then goes back where it was. Its [autofocus] element, else its first menu
// row or other control, takes the focus. Once closed it's removed, then onclose runs. An anchor says whether it's open
// (aria-expanded), and pressing it again closes it: null then.
// (Its own light dismiss, not the browser's: that would close a context menu opened on a press as the press ends.)
const opened = new WeakMap();  // anchor → its open popover
let lightOne = null;  // the last light popover opened
export function popover(at, build, {light = true, onclose, ...attrs} = {}) {
  const anchor = at instanceof Element ? at : null;
  if (opened.has(anchor)) {
    closePopover(opened.get(anchor));
    return null;
  }
  if (light) closePopover();
  const close = () => closePopover(el);
  const outside = (e) => { if (!el.contains(e.target) && !anchor?.contains(e.target)) close(); };  // a press or the focus
  const listening = new AbortController(), on = (type, fn) => document.addEventListener(type, fn, {signal: listening.signal});
  const opener = document.activeElement;
  const el = h("div", {...attrs, class: `popover ${attrs.class ?? ""}`, popover: "manual", onbeforetoggle: (e) => {
    if (e.newState === "open") return;
    listening.abort();
    if (el.contains(document.activeElement)) opener?.focus({preventScroll: true});
    opened.delete(anchor);
    anchor?.setAttribute("aria-expanded", "false");
  }, ontoggle: (e) => {
    if (e.newState === "open") return;
    el.remove();
    onclose?.();
  }}, build(close));
  on("keydown", (e) => { if (e.key === "Escape") close(); });
  if (light) {
    on("pointerdown", outside);
    on("focusin", outside);
    lightOne = el;
  }
  document.body.append(el);
  el.showPopover(anchor ? {source: anchor} : {});
  const a = anchor?.getBoundingClientRect(), x = a ? a.left : at.x, y = a ? a.bottom + 4 : at.y;
  el.style.left = `${Math.max(8, Math.min(x, innerWidth - el.offsetWidth - 8))}px`;
  el.style.top = `${Math.max(8, Math.min(y, innerHeight - el.offsetHeight - 8))}px`;
  if (!el.contains(document.activeElement)) {
    (el.querySelector(".menu-list > :is(a, button:enabled)") ?? el.querySelector("input, select, textarea, button:enabled, a[href]"))?.focus({preventScroll: true});
  }
  if (anchor) {
    opened.set(anchor, el);
    anchor.setAttribute("aria-expanded", "true");
  }
  return el;
}

// closes popover el if it's open; by default the light one
export function closePopover(el = lightOne) {
  if (el?.matches(":popover-open")) el.hidePopover();
}

// a menu's rows: items are {label, fn, checked?, disabled?} or elements (an <hr>, a row of the caller's); falsy ones
// are left out. Choosing an item closes the menu and calls fn(); a checkable one (checked a boolean) stays and calls
// fn(on). The arrow keys, Home and End move between the rows
export function menuList(items) {
  const ticks = items.some((it) => typeof it?.checked === "boolean");
  const row = ({label, fn, checked, disabled}) => {
    const checkable = typeof checked === "boolean", tick = h("span", {class: "tick"});
    const set = (on) => { el.setAttribute("aria-checked", String(checked = on)); tick.replaceChildren(on ? icon("check", 14) : ""); };
    const el = h("button", {class: "menu-row", role: checkable ? "menuitemcheckbox" : "menuitem", disabled, onclick: () => {
      if (!checkable) {
        closePopover(el.closest(".popover"));
        return fn();
      }
      set(!checked);
      fn(checked);
    }}, ticks && tick, label);
    if (checkable) set(checked);
    return el;
  };
  const list = h("div", {class: "menu-list", role: "menu", onkeydown: (e) => {
    const rows = [...list.querySelectorAll(":scope > :is(a, button:enabled)")], k = rows.indexOf(document.activeElement);
    const to = {ArrowDown: k + 1, ArrowUp: k - 1, Home: 0, End: rows.length - 1}[e.key];
    if (to === undefined) return;
    e.preventDefault();
    e.stopPropagation();  // not the playhead's arrows
    rows[(to + rows.length) % rows.length]?.focus();
  }}, items.map((it) => it && !(it instanceof Node) ? row(it) : it));
  for (const r of list.querySelectorAll(":scope > :is(a, button):not([role])")) r.setAttribute("role", "menuitem");
  return list;
}

// a top-bar button's panel, titled: a click opens it, make(close) → {body, actions, update, closed}: what's in it, more
// buttons for its head, what refreshes it, what it lets go of once closed. Another click closes it, as do its close
// button, Escape and, if light, a press outside
export function anchoredPanel(button, title, make, {light = true, ...attrs} = {}) {
  let open = null;
  button.setAttribute("aria-expanded", "false");
  button.addEventListener("click", () => {
    const el = popover(button, (close) => {
      open = make(close);
      return [h("div", {class: "panel-head"}, h("b", {}, title), open.actions, h("button", {class: "link", autofocus: true, onclick: close}, "close")), open.body];
    }, {...attrs, class: `panel ${attrs.class ?? ""}`, role: "dialog", "aria-label": title, light, onclose: () => { open.closed?.(); open = null; }});
    if (el) open.update?.();
  });
  return {update: () => open?.update?.()};
}

// data (a Blob, or text) saved as file name; its URL is let go after keep ms
export function download(data, name, keep = 1000) {
  const a = h("a", {href: URL.createObjectURL(data instanceof Blob ? data : new Blob([data], {type: "text/plain"})), download: name});
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), keep);
}

// an icon button that copies text() and says whether it could; in a summary, it doesn't toggle it
export function copyButton(text, label = "Copy", size = 16) {
  const btn = h("button", {class: "icon-btn", title: label, "aria-label": label, onclick: async (e) => {
    e.preventDefault();
    const ok = await navigator.clipboard.writeText(text()).then(() => true, () => false);
    btn.title = ok ? "Copied" : "Couldn't copy: the clipboard was denied";
    btn.replaceChildren(icon(ok ? "check" : "warning", size));
    setTimeout(() => {
      btn.title = label;
      btn.replaceChildren(icon("copy", size));
    }, 1500);
  }}, icon("copy", size));
  return btn;
}

export const fmt = {
  ago(unixSeconds) {
    if (!unixSeconds) return "never";
    const s = Date.now() / 1000 - unixSeconds;
    const [n, unit] = s < 90 ? [s, "s"] : s < 5400 ? [s / 60, "m"] : s < 129600 ? [s / 3600, "h"] : [s / 86400, "d"];
    return `${Math.round(n)}${unit} ago`;
  },
  minutes: (ms) => `${Math.max(1, Math.round(ms / 60000))} min`,
  duration(ms) {
    const m = Math.floor(ms / 60000), s = Math.floor((ms % 60000) / 1000);
    return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m ${String(s).padStart(2, "0")}s`;
  },
  // a time on the route: "1:15:03", or "1:15:03.25" withMs
  clock(s, withMs = false) {
    s = withMs ? Math.round(s * 100) / 100 : Math.trunc(s);  // to what's shown first: no "1:60.00" or "-0:00.00"
    const sign = s < 0 ? "-" : "";
    s = Math.abs(s);
    const hr = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, sec = s % 60;
    const ss = withMs ? sec.toFixed(2).padStart(5, "0") : String(sec).padStart(2, "0");
    return `${sign}${hr ? `${hr}:${String(m).padStart(2, "0")}` : m}:${ss}`;
  },
  // the API's miles in the visitor's unit: [value, unit], or "12.3 km"
  distanceIn: (mi) => isMetric() ? [mi * KM_PER_MI, "km"] : [mi, "mi"],
  distance: (mi) => { const [v, unit] = fmt.distanceIn(mi); return `${v.toFixed(1)} ${unit}`; },
  // as connect names devices: the alias, else what it is
  deviceType: (type) => ({threex: "comma 3X", neo: "EON", freon: "freon", unknown: "unknown"})[type] ?? `comma ${type}`,
  deviceName: (device) => device.alias || fmt.deviceType(device.device_type),
};

// only a few countries drive in miles; connect decides by the browser's language alone, and so do we
const KM_PER_MI = 1.60934;
let metric = null;
export const isMetric = () => metric ??= !["en-us", "en-gb"].includes(navigator.language.toLowerCase());

// a typed time on the route, as fmt.clock writes it or in seconds ("3:12", "1:03:12.5", "192") → seconds, else NaN
export function parseClock(s) {
  const m = /^\s*(?:(?:(\d+):)?(\d+):)?(\d+(?:\.\d*)?)\s*$/.exec(s ?? "");
  return m ? (+m[1] || 0) * 3600 + (+m[2] || 0) * 60 + +m[3] : NaN;
}

// a module's own stylesheet, added once; resolves once it's applied, so a pane module can wait for it at the top
// level (await css(new URL("./camera.css", import.meta.url))) and never show an unstyled first frame
const sheets = new Map();
export function css(href) {
  href = String(href);
  if (!sheets.has(href)) sheets.set(href, new Promise((done) => document.head.append(h("link", {rel: "stylesheet", href, onload: done, onerror: done}))));
  return sheets.get(href);
}

// sizes canvas cv's backing store to w×hgt css px at devicePixelRatio, only when that changed (reallocating clears
// it); returns its context, drawing in css px. The css size is the caller's
export function fitCanvas(cv, w, hgt) {
  const r = devicePixelRatio || 1, bw = Math.max(1, Math.round(w * r)), bh = Math.max(1, Math.round(hgt * r));
  if (cv.width !== bw || cv.height !== bh) [cv.width, cv.height] = [bw, bh];
  const g = cv.getContext("2d");
  g.setTransform(r, 0, 0, r, 0, 0);
  return g;
}
