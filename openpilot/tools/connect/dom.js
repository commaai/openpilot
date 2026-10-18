// the only DOM helper: h(tag, attrs, ...children)
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c != null && c !== false) el.append(c instanceof Node ? c : String(c));
  }
  return el;
}

export function clear(el) {
  el.replaceChildren();
  return el;
}

export const fmt = {
  date: (ms) => new Date(ms).toLocaleString(undefined, {dateStyle: "medium", timeStyle: "short"}),
  time: (ms) => new Date(ms).toLocaleTimeString(undefined, {timeStyle: "short"}),
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
  clock(sec) {
    const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  },
  miles: (mi) => `${mi.toFixed(1)} mi`,
};
