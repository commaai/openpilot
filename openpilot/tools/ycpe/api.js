// comma API client: a JWT in localStorage, plain fetch. Settings are read lazily so workers can import this.

const TOKEN_KEY = "comma_jwt";
const setting = (key) => globalThis.localStorage?.getItem(key);
export const apiHost = () => setting("api_host") || "https://api.commadotai.com";
// athena only answers CORS for *.comma.ai, connect previews and localhost:3000
const athenaHost = () => setting("athena_host") || "https://athena.comma.ai";

// sign-in state: the token, the signed-in user (v1/me, cached), and subscribers to changes
const authSubs = new Set();
let user = null;

export const auth = {
  token: () => setting(TOKEN_KEY),
  isAuthenticated: () => !!setting(TOKEN_KEY),
  user: () => (user ??= auth.isAuthenticated() ? get("v1/me/").catch(() => null) : Promise.resolve(null)),
  on(fn) { authSubs.add(fn); return () => authSubs.delete(fn); },
  setToken(token) {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
    user = null;
    for (const cache of caches) cache.clear();  // what one account could read isn't the next one's
    for (const k of Object.keys(sessionStorage)) if (k.startsWith("ycpe_")) sessionStorage.removeItem(k);
    for (const fn of authSubs) fn();
  },
  signOut: () => auth.setToken(null),

  // the provider sends the code to the API, which redirects it back to <host>/auth/?code=..&provider=..
  loginUrl(provider) {
    const params = {
      redirect_uri: `https://api.comma.ai/v2/auth/${{google: "g", apple: "a", github: "h"}[provider]}/redirect/`,
      state: `service,${location.host}`,
    };
    if (provider === "google") {
      Object.assign(params, {
        type: "web_server",
        client_id: "45471411055-ornt4svd2miog6dnopve7qtmh5mnu6id.apps.googleusercontent.com",
        response_type: "code",
        scope: "https://www.googleapis.com/auth/userinfo.email",
        prompt: "select_account",
      });
      return "https://accounts.google.com/o/oauth2/auth?" + new URLSearchParams(params);
    } else if (provider === "github") {
      Object.assign(params, {client_id: "28c4ecb54bb7272cb5a4", scope: "read:user"});
      return "https://github.com/login/oauth/authorize?" + new URLSearchParams(params);
    } else {
      Object.assign(params, {client_id: "ai.comma.login", response_type: "code", response_mode: "form_post", scope: "name email"});
      return "https://appleid.apple.com/auth/authorize?" + new URLSearchParams(params);
    }
  },

  async exchangeCode(code, provider) {
    const resp = await fetch(`${apiHost()}/v2/auth/`, {
      method: "POST",
      headers: {"Content-Type": "application/x-www-form-urlencoded"},
      body: new URLSearchParams({code, provider}),
    });
    if (!resp.ok) throw new ApiError(resp.status, await resp.text());
    const {access_token} = await resp.json();
    if (!access_token) throw new Error("no token in the sign-in response");
    auth.setToken(access_token);
  },

  // a pasted developer JWT is kept only if the API accepts it
  async useToken(token) {
    const resp = await fetch(`${apiHost()}/v1/me/`, {headers: {Authorization: `JWT ${token}`}});
    if (!resp.ok) throw new ApiError(resp.status, resp.status === 401 ? "that token isn't valid" : await resp.text());
    auth.setToken(token);
  },
};

export class ApiError extends Error {
  constructor(status, body) {
    super(`API ${status}: ${body}`);
    this.status = status;
  }
}

export async function request(method, endpoint, {params, body, host = apiHost()} = {}) {
  const url = new URL(endpoint, host + "/");
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== undefined && v !== null) url.searchParams.set(k, v);
  }
  const headers = {};
  if (auth.token()) headers["Authorization"] = "JWT " + auth.token();
  if (body) headers["Content-Type"] = "application/json";
  const resp = await fetch(url, {method, headers, body: body ? JSON.stringify(body) : undefined});
  if (resp.status === 401 && auth.isAuthenticated()) auth.signOut();
  if (!resp.ok) throw new ApiError(resp.status, await resp.text());
  return resp.json();
}
const get = (endpoint, params) => request("GET", endpoint, {params});

// a read kept in sessionStorage until expires(value) (ms), so a reload doesn't wait ~1 s on the API again
async function sessionCached(key, fetch, expires) {
  const hit = JSON.parse(sessionStorage.getItem(key) || "null");
  if (hit && hit.until > Date.now()) return hit.value;
  const value = await fetch();
  sessionStorage.setItem(key, JSON.stringify({value, until: expires(value)}));
  return value;
}

// signed URLs carry their expiry (se=); a file list is good until the first one lapses, less a margin
function signedUntil(files) {
  let until = Date.now() + 10 * 60e3;
  for (const urls of Object.values(files)) for (const u of urls || []) {
    const se = Date.parse(new URL(u).searchParams.get("se") || "");
    if (se) until = Math.min(until, se - 5 * 60e3);
  }
  return until;
}

// reads are cached for the session so revisits never wait on the network; failures are not kept
const caches = new Set();
function memo(fn, key = (...args) => JSON.stringify(args)) {
  const cache = new Map();
  caches.add(cache);
  const wrapped = (...args) => {
    const k = key(...args);
    if (!cache.has(k)) cache.set(k, fn(...args).catch((e) => { cache.delete(k); throw e; }));
    return cache.get(k);
  };
  wrapped.prime = (value, ...args) => cache.set(key(...args), Promise.resolve(value));
  wrapped.forget = (...args) => cache.delete(key(...args));
  return wrapped;
}

export const api = {
  me: () => auth.user(),
  devices: () => get("v1/me/devices/"),
  device: memo((dongleId) => get(`v1.1/devices/${dongleId}/`)),
  deviceStats: (dongleId) => get(`v1.1/devices/${dongleId}/stats`),
  deviceLocation: memo((dongleId) => get(`v1/devices/${dongleId}/location`)),
  setDeviceAlias: (dongleId, alias) => request("PATCH", `v1/devices/${dongleId}/`, {body: {alias}}),
  deviceUsers: (dongleId) => get(`v1/devices/${dongleId}/users`),
  addDeviceUser: (dongleId, email) => request("POST", `v1/devices/${dongleId}/add_user`, {body: {email}}),
  // without an email this removes the signed-in user
  removeDeviceUser: (dongleId, email) => request("POST", `v1/devices/${dongleId}/del_user`, {body: email ? {email} : {}}),
  // signed blob urls, oldest first
  bootlogs: memo((dongleId) => get(`v1/devices/${dongleId}/bootlogs`)),
  crashlogs: memo((dongleId) => get(`v1/devices/${dongleId}/crashlogs`)),

  // routes that overlap [start, end] in unix millis, newest first
  routes: memo((dongleId, start, end, limit = 50) => get(`v1/devices/${dongleId}/routes_segments`, {start, end, limit})),
  preservedRoutes: (dongleId) => get(`v1/devices/${dongleId}/routes/preserved`),
  // the device listing carries the share credentials and utc times; v1/route is the fallback for public routes
  route: memo(async (routeName) => {
    const dongleId = routeName.split("|")[0];
    const [r] = await get(`v1/devices/${dongleId}/routes_segments`, {route_str: routeName}).catch(() => []);
    if (r) return r;
    const route = await get(`v1/route/${routeName}/`);
    route.start_time_utc_millis ??= Date.parse(route.start_time + "Z");
    route.end_time_utc_millis ??= Date.parse(route.end_time + "Z");
    return route;
  }),
  routeFiles: memo((routeName) => sessionCached(`ycpe_files_${routeName}`, () => get(`v1/route/${routeName}/files`), signedUntil)),
  setRoutePublic: (routeName, isPublic) => request("PATCH", `v1/route/${routeName}/`, {body: {is_public: isPublic}}),
  setRoutePreserved: (routeName, preserved) => request(preserved ? "POST" : "DELETE", `v1/route/${routeName}/preserve`),

  // paths are "<route>--<segment>/<file name>" without the dongle id
  uploadUrls: async (dongleId, paths) => (await request("POST", `v1/${dongleId}/upload_urls/`, {body: {paths, expiry_days: 7}})).map((r) => r.url),
  athenaOfflineQueue: (dongleId) => get(`v1/devices/${dongleId}/athena_offline_queue`),
  // JSON-RPC on the device; with an expiry, athena queues the call while the device is offline
  async athena(dongleId, method, params, expiry) {
    const resp = await request("POST", dongleId, {host: athenaHost(), body: {id: 0, jsonrpc: "2.0", method, params, expiry}});
    if (resp.error) throw new Error(resp.error.data?.message || resp.error.message || String(resp.error));
    return resp.result;
  },

  // public routes need no share credentials
  qcameraUrl: (route) => `${apiHost()}/v1/route/${route.fullname}/qcamera.m3u8` +
    (route.share_sig ? "?" + new URLSearchParams({exp: route.share_exp, sig: route.share_sig}) : ""),
  // per-segment derived assets live next to the route on the CDN
  segmentAsset: (route, segment, name) => `${route.url}/${segment}/${name}`,
};

const assets = (route, name) => Promise.all(Array.from({length: (route.maxqlog ?? -1) + 1}, async (_, n) => {
  const resp = await fetch(api.segmentAsset(route, n, name)).catch(() => null);
  return resp?.ok ? resp.json() : [];
}));

// GPS fixes [{t (route seconds), lat, lng, speed, dist}] from the per-segment coords.json
export const routeCoords = memo(async (route) => (await assets(route, "coords.json")).flat(), (route) => route.fullname);

// engaged, alert, override and bookmark spans from the per-segment events, in route milliseconds
export const routeEvents = memo(async (route) => {
  const events = (await assets(route, "events.json")).flat().sort((a, b) => a.route_offset_millis - b.route_offset_millis);
  const spans = [];
  const open = {};
  const track = (kind, active, t) => {
    if (open[kind] && !active) {
      open[kind].end = t;
      open[kind] = null;
    } else if (!open[kind] && active) {
      open[kind] = {kind, start: t, end: null};
      spans.push(open[kind]);
    }
  };
  for (const ev of events) {
    const t = ev.route_offset_millis;
    if (ev.type === "user_flag" || ev.type === "user_bookmark") spans.push({kind: "bookmark", start: t, end: t + 1000});
    if (ev.type !== "state") continue;
    track("engaged", ev.data.enabled, t);
    track("alert", ev.data.alertStatus === "userPrompt" || ev.data.alertStatus === 1, t);
    track("critical", ev.data.alertStatus === "critical" || ev.data.alertStatus === 2, t);
    track("override", ev.data.state === "overriding" || ev.data.state === "preEnabled", t);
  }
  const duration = route.end_time_utc_millis - route.start_time_utc_millis;
  for (const s of spans) if (s.end === null) s.end = duration;
  return spans;
}, (route) => route.fullname);

export function engagedFraction(spans, duration) {
  const ms = spans.reduce((sum, s) => sum + (s.kind === "engaged" ? s.end - s.start : 0), 0);
  return duration > 0 ? Math.min(1, ms / duration) : 0;
}
