// the two modes: user mode is connect's site under /connect, developer mode is the workspace at today's URLs. A path
// that names no mode ("/", a device, a drive without a view, account and referrals) opens in the mode in use, else in
// the one last chosen with the switch; a new visitor gets user mode
import {routeFromPath} from "../../data/sources.js";

const KEY = "ycpe_mode";
const DONGLE_RE = /^[0-9a-f]{16}$/i;
let current = null;  // the mode of the page shown, once one has been

export const mode = () => current ?? (localStorage.getItem(KEY) === "dev" ? "dev" : "user");

// chosen: the switch was used, so "/" opens in it from now on
export function setMode(m, chosen = false) {
  current = m;
  if (chosen) localStorage.setItem(KEY, m);
}

// user mode's path for a path that names no mode (its parts, after the host), else null
export function userPath(parts, search = "") {
  const [a] = parts;
  if (!parts.length) return `/connect${search}`;
  if (parts.length === 1 && (DONGLE_RE.test(a) || a === "account" || a === "referrals")) return `/connect/${a}${search}`;
  const target = DONGLE_RE.test(a) && parts.length >= 2 && routeFromPath(parts);
  // a drive, or connect's time range in one; a segment slice or a view is developer mode's alone
  return target && !target.view && !target.name.includes("/") ? `/connect/${parts.join("/")}${search}` : null;
}

// developer mode's path for a user-mode one ("/connect/..."): the same place without the prefix; prime is user mode's
export const devPath = (parts, search = "") => `/${parts.slice(1).filter((p, i) => !(i === 1 && p === "prime")).join("/")}${search}`;
