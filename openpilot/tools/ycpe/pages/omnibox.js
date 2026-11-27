// the omnibox: anything that names a route or a device, typed or pasted (route names in any separator form,
// connect, useradmin and cabana URLs, segment ranges and selectors)
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
  // a bare route id: one opened before, else the device in view's
  if (!ROUTE_ID.test(s)) return null;
  const recent = recentRoutes().find((r) => r.name.endsWith(`|${s}`));
  const here = location.pathname.split("/")[1];
  return recent ? routePath(recent.name) : /^[0-9a-f]{16}$/i.test(here) ? routePath(`${here.toLowerCase()}|${s}`) : null;
}

// go(path) navigates; me() is the signed-in user's account; onMiss(text) is told when nothing matched. What's typed
// filters the navigator (nav.js) as it goes; Enter opens what it names
export function omnibox({go, me, onMiss}) {
  // the "/" shortcut is only worth a mention with a keyboard
  const placeholder = `route, dongle id or URL${matchMedia("(pointer: coarse)").matches ? "" : "  ( / )"}`;
  const input = h("input", {type: "search", class: "omnibox", placeholder, spellcheck: false, autocomplete: "off", "aria-label": "Open a route or device"});

  // as in useradmin's search box, your own user id, email or username finds your account
  const mine = (q) => [me()?.user_id, me()?.email, me()?.username].some((s) => s?.toLowerCase() === q);
  function open(path) {
    input.value = "";
    input.blur();
    go(path);
  }
  async function enter() {
    const text = input.value, q = text.trim().toLowerCase();
    if (!q) return open("/");  // as useradmin's empty search, which shows your account
    if (mine(q)) return open("/account");
    const path = await resolve(text);
    if (path) open(path);
    else onMiss(text);
  }

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") enter();
    else if (e.key === "Escape") input.blur();
  });
  return h("span", {class: "omni"}, input);
}
