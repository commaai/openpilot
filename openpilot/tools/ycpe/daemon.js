// the optional local daemon (serve.py): present when index.html carries its session token. Everything works
// without it; with it, logs come from every source and through its disk cache, and live data from this PC.
import {h, fmt, toast} from "./dom.js";

const AUTH_OFF = "ycpe_daemon_auth_off";  // set by signing out, so the daemon's JWT isn't picked up again
const TOKEN_REFUSED = "missing or wrong token";  // daemon/core.py check_local's 403

const token = globalThis.document?.querySelector('meta[name="ycpe-daemon"]')?.content || null;
const headers = token ? {"X-Ycpe-Token": token} : {};

async function get(path) {
  const resp = await fetch(path, {headers});
  if (!resp.ok) throw new Error(`${path}: ${resp.status}`);
  return resp.json();
}

// one localhost round trip before the app starts, and only when served by the daemon
const [caps, jwt] = token ? await Promise.all([
  get("/local/caps").catch(() => null),
  get("/local/auth").then((r) => r.token, () => null),
]) : [null, null];

// the commit serve.py runs from, linked as connect links its own, and how long ago it was made: [link, ", 3d ago"], or
// null from a static server, which can't tell
export function commitLink(attrs = {}) {
  const c = caps?.commit;
  return c && [h("a", {href: `https://github.com/commaai/openpilot/commit/${c.sha}`, target: "_blank", ...attrs}, c.sha.slice(0, 7)), `, ${fmt.ago(c.time)}`];
}

export const daemon = {
  present: !!caps,
  caps,
  hasJwt: !!jwt,
  headers,  // the session token, for requests made elsewhere (the live data's signaling)
  // the comma JWT from ~/.comma/auth.json, unless the user signed out here
  jwt: () => (jwt && !globalThis.localStorage?.getItem(AUTH_OFF) ? jwt : null),
  useJwt(on) { on ? localStorage.removeItem(AUTH_OFF) : localStorage.setItem(AUTH_OFF, "1"); },

  async fetch(path, opts = {}) {
    if (!caps) throw new Error("needs the local daemon (run tools/ycpe/serve.py)");
    const body = opts.body !== undefined && typeof opts.body !== "string" && !(opts.body instanceof Blob) ? JSON.stringify(opts.body) : opts.body;
    const h = {...headers, ...(body !== opts.body ? {"Content-Type": "application/json"} : {}), ...opts.headers};
    const resp = await fetch(path, {...opts, body, headers: h});
    if (resp.ok) return resp;
    const error = (await resp.json().catch(() => null))?.error;
    if (error === TOKEN_REFUSED && !stale) staleToken();
    throw Object.assign(new Error(error || `${path}: ${resp.status}`), {error});
  },
  json: async (path, opts) => (await daemon.fetch(path, opts)).json(),
  // a URL that carries the token, for places that can't set headers (media elements, worker fetches)
  url(path) {
    const u = new URL(path, location.origin);
    u.searchParams.set("tok", token);
    return u.pathname + u.search;
  },
};

// a daemon that refuses this page's token was restarted with a new one (serve.py keeps it in ~/.cache/ycpe/token,
// so that's rare); only a reload picks up the new one. Said once, and it stays up.
let stale = false;
function staleToken() {
  stale = true;
  toast("The local daemon restarted. Reload to reconnect. ", ["Reload", () => location.reload()]);
}

// the short note features show when they need the daemon
export const needsDaemon = (what = "This") => `${what} needs the local daemon: run tools/ycpe/serve.py with openpilot's venv.`;
