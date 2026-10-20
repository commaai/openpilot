// the omnibox: anything that names a route or a device, typed or pasted (route names in any separator form,
// connect, useradmin and cabana URLs, segment ranges and selectors), with a dropdown of matching recent routes and devices
import {api} from "../api.js";
import {h} from "../dom.js";
import {parseIndirect, parseRange} from "../data/sources.js";

const RECENT_KEY = "ycpe_recent_routes";
const RECENT_MAX = 10;
const ROUTE = /(?<![0-9a-f])([0-9a-f]{16})(?:\||\/|_)([0-9a-f]{8}--[0-9a-f]{10}|\d{4}-\d\d-\d\d--\d\d-\d\d-\d\d)(?:--(\d+))?/i;
const CONNECT = /(?<![0-9a-f])([0-9a-f]{16})\/(\d{13})\/(\d{13})/i;  // connect.comma.ai/<dongle>/<start ms>/<end ms>
const DONGLE = /(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])/i;
const ROUTE_ID = /^[0-9a-f]{8}--[0-9a-f]{10}$/i;

export function recentRoutes() {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY)) || [];
  } catch {
    return [];
  }
}

export function rememberRoute(r) {
  const entry = {name: r.fullname, start: r.start_time_utc_millis, end: r.end_time_utc_millis, distance: r.distance, platform: r.platform};
  const rest = recentRoutes().filter((x) => x.name !== entry.name);
  localStorage.setItem(RECENT_KEY, JSON.stringify([entry, ...rest].slice(0, RECENT_MAX)));
}

export const routePath = (name, t) => `/${name.replace("|", "/")}${t >= 1 ? `?t=${Math.round(t)}` : ""}`;

// the page a text names: "/<dongle>/<route>?t=" or "/<dongle>", else null. A connect time-range URL is matched
// to the drive it starts in, so this may ask the API.
export async function resolve(text) {
  let s = text.trim();
  if (/^https?:\/\/\S+\/(?:rlog|qlog)\.(?:zst|bz2)(?:\?|$)|^https?:\/\/\S+\/[dfeq]camera\.(?:hevc|ts)(?:\?|$)/.test(s)) {
    return `/local/url/plot?url=${encodeURIComponent(s)}`;  // a direct link to a file opens it on its own
  }
  try { s = decodeURIComponent(s); } catch { /* keep it as typed */ }
  // a connect time range ("connect.comma.ai/<dongle>/<log id>/<start>/<end>"), opened as it is
  const span = parseIndirect(s).match(/^[0-9a-f]{16}\/[^/]+\/\d+\/\d+$/i);
  if (span && parseRange(span[0])) return `/${span[0]}`;
  // a segment range or selector, as the Python tools take them ("<route>/2:5/q"); "<route>--3" stays a seek
  const r = parseRange(s);
  if (r && (r.selector || r.slice && !s.endsWith(`--${r.slice}`))) return `/${r.dongle}/${r.logId}/${[r.slice, r.selector].filter(Boolean).join("/")}`;
  const t = Number(s.match(/[?&#]t=([\d.]+)/)?.[1]);
  const m = s.match(ROUTE);
  if (m) return routePath(`${m[1]}|${m[2]}`, t || (m[3] ? m[3] * 60 : 0));
  const c = s.match(CONNECT);
  if (c) {
    const start = +c[2];
    const routes = await api.routes(c[1].toLowerCase(), start, +c[3]).catch(() => []);
    const r = routes.find((r) => r.start_time_utc_millis <= start && start < r.end_time_utc_millis) || routes.at(-1);
    return r ? routePath(r.fullname, (start - r.start_time_utc_millis) / 1000) : `/${c[1].toLowerCase()}`;
  }
  const d = s.match(DONGLE);
  if (d) return `/${d[0].toLowerCase()}`;
  const recent = ROUTE_ID.test(s) && recentRoutes().find((r) => r.name.endsWith(`|${s}`));
  return recent ? routePath(recent.name) : null;
}

const DATE = new Intl.DateTimeFormat(undefined, {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false});

// go(path) navigates; devices() is the signed-in user's list; onMiss(text) is told when nothing matched
export function omnibox({go, devices, onMiss}) {
  const input = h("input", {type: "search", class: "omnibox", placeholder: "route, dongle id or URL  ( / )", spellcheck: false, autocomplete: "off"});
  const list = h("div", {class: "omni-list", hidden: true});
  const el = h("span", {class: "omni"}, input, list);
  let items = [], sel = -1;

  const alias = (dongle) => devices()?.find((d) => d.dongle_id === dongle)?.alias || dongle;
  function suggest() {
    const q = input.value.trim().toLowerCase();
    const recent = recentRoutes().map((r) => ({path: routePath(r.name), label: `${alias(r.name.split("|")[0])} · ${DATE.format(r.start)}`,
      detail: r.name, text: `${r.name} ${alias(r.name.split("|")[0])} ${r.platform || ""}`}));
    const devs = (devices() || []).map((d) => ({path: `/${d.dongle_id}`, label: d.alias || d.dongle_id, detail: d.dongle_id,
      text: `${d.alias || ""} ${d.dongle_id}`}));
    items = [...recent, ...devs].filter((i) => !q || i.text.toLowerCase().includes(q)).slice(0, 12);
    sel = -1;
    paint();
  }
  function paint() {
    list.hidden = !items.length || document.activeElement !== input;
    list.replaceChildren(...items.map((i, k) => h("a", {class: `omni-row ${k === sel ? "on" : ""}`, href: i.path,
      onpointerdown: (e) => e.preventDefault(),  // keep focus in the input
      onclick: (e) => { e.preventDefault(); open(i.path); }},
      h("span", {}, i.label), h("span", {class: "mono muted"}, i.detail))));
  }
  function open(path) {
    input.value = "";
    input.blur();
    go(path);
  }
  async function enter() {
    if (sel >= 0) return open(items[sel].path);
    const text = input.value;
    const path = await resolve(text);
    if (path) open(path);
    else if (text.trim()) onMiss(text);
  }

  input.addEventListener("input", suggest);
  input.addEventListener("focus", suggest);
  input.addEventListener("blur", () => { list.hidden = true; });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!items.length) return;
      const n = items.length + 1;  // -1 is the typed text itself
      sel = (sel + 1 + (e.key === "ArrowDown" ? 1 : -1) + n) % n - 1;
      paint();
      list.children[sel]?.scrollIntoView({block: "nearest"});
    } else if (e.key === "Enter") {
      enter();
    } else if (e.key === "Escape") {
      input.blur();
    }
  });
  return el;
}
