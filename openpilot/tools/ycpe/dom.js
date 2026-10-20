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

export function clear(el) {
  el.replaceChildren();
  return el;
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
  miles: (mi) => `${mi.toFixed(1)} mi`,
};

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
