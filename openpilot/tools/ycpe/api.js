// comma API client: a JWT in localStorage (or the local daemon's), plain fetch. Settings are read lazily so workers can import this.
import {daemon} from "./daemon.js";
import {listeners} from "./dom.js";

const TOKEN_KEY = "comma_jwt";
const setting = (key) => globalThis.localStorage?.getItem(key);
export const apiHost = () => setting("api_host") || "https://api.commadotai.com";
// athena only answers CORS for *.comma.ai, connect previews and localhost:3000; elsewhere the daemon relays the calls
const athenaHost = () => setting("athena_host") || "https://athena.comma.ai";
// billing only answers CORS for *.comma.ai, localhost:3000 and connect's LAN and preview hosts, and nothing relays it:
// elsewhere prime can't be managed (a billing_host setting says this origin reaches one)
const billingHost = () => setting("billing_host") || "https://billing.comma.ai";
const BILLING_ORIGINS = /^(https:\/\/([\w-]+\.)*comma\.ai|http:\/\/(localhost|192\.168\.\d{1,3}\.\d{1,3}):3000|https:\/\/[\w.-]+\.connect-d5y\.pages\.dev)$/;
export const billingReachable = () => !!setting("billing_host") || BILLING_ORIGINS.test(location.origin);
const billing = (method, endpoint, opts) => request(method, endpoint, {...opts, host: billingHost()});
const ATHENA_ORIGINS = /^(https:\/\/([\w-]+\.)*comma\.ai|http:\/\/localhost:3000)$/;
const athenaRelay = () => !setting("athena_host") && daemon.has("athena") && !ATHENA_ORIGINS.test(location.origin);

// sign-in state: the token, the signed-in user (v1/me, cached), and subscribers to changes
const authSubs = listeners();
let user = null;
let refused = null;  // a token the API turned down (expired, revoked): not sent again until signing in

// fn(reason): reason is "expired" when the API refused the token, rather than the user signing in or out
function authChanged(reason) {
  user = null;
  for (const cache of apiCaches) cache.clear();  // what one account could read isn't the next one's
  for (const k of Object.keys(sessionStorage)) if (k.startsWith("ycpe_")) sessionStorage.removeItem(k);
  authSubs.emit(reason);
}

export const auth = {
  token() {
    const token = setting(TOKEN_KEY) || daemon.jwt();
    return token !== refused ? token : null;
  },
  isAuthenticated: () => !!auth.token(),
  viaDaemon: () => !setting(TOKEN_KEY) && auth.isAuthenticated(),  // ~/.comma/auth.json through serve.py
  user: () => (user ??= auth.isAuthenticated() ? get("v1/me/").catch(() => null) : Promise.resolve(null)),
  on: authSubs.on,
  setToken(token, viaDaemon = false) {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else {
      localStorage.removeItem(TOKEN_KEY);
      daemon.useJwt(viaDaemon);
    }
    refused = null;
    authChanged();
  },
  signOut: () => auth.setToken(null),
  useDaemon: () => auth.setToken(null, true),
  // the API answered 401 to this token; the daemon's stays on for the next page load, in case it's renewed
  expired(token) {
    if (token !== auth.token()) return;  // already handled
    refused = token;
    if (setting(TOKEN_KEY) === token) localStorage.removeItem(TOKEN_KEY);
    authChanged("expired");
  },

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

  // a fresh 90-day token for the code; exchangeCode also signs in with it
  async newToken(code, provider) {
    const resp = await fetch(`${apiHost()}/v2/auth/`, {
      method: "POST",
      headers: {"Content-Type": "application/x-www-form-urlencoded"},
      body: new URLSearchParams({code, provider}),
    });
    if (!resp.ok) throw new ApiError(resp.status, await resp.text());
    const {access_token} = await resp.json();
    if (!access_token) throw new Error("no token in the sign-in response");
    return access_token;
  },
  exchangeCode: async (code, provider) => auth.setToken(await auth.newToken(code, provider)),

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

export async function request(method, endpoint, {params, body, host = apiHost(), headers = {}} = {}) {
  const url = new URL(endpoint, host + "/");
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== undefined && v !== null) url.searchParams.set(k, v);
  }
  const token = auth.token();
  if (token) headers["Authorization"] = "JWT " + token;
  if (body) headers["Content-Type"] = "application/json";
  const resp = await fetch(url, {method, headers, body: body ? JSON.stringify(body) : undefined});
  if (resp.status === 401 && token) auth.expired(token);
  if (!resp.ok) throw new ApiError(resp.status, await resp.text());
  return resp.json();
}
const get = (endpoint, params) => request("GET", endpoint, {params});

// a read kept in sessionStorage until expires(value) (ms), so a reload doesn't wait ~1 s on the API again
async function sessionCached(key, load, expires) {
  const hit = JSON.parse(sessionStorage.getItem(key) || "null");
  if (hit && hit.until > Date.now()) return hit.value;
  const value = await load();
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

// reads are cached for the session (or ttl ms) so revisits never wait on the network; failures are not kept
const apiCaches = new Set();
function memo(fn, key = (...args) => JSON.stringify(args), ttl = Infinity) {
  const cache = new Map(), at = new Map();
  apiCaches.add(cache);
  const wrapped = (...args) => {
    const k = key(...args);
    if (!cache.has(k) || performance.now() - at.get(k) > ttl) {
      at.set(k, performance.now());
      cache.set(k, fn(...args).catch((e) => { cache.delete(k); throw e; }));
    }
    return cache.get(k);
  };
  wrapped.prime = (value, ...args) => { at.set(key(...args), performance.now()); cache.set(key(...args), Promise.resolve(value)); };
  wrapped.forget = (...args) => cache.delete(key(...args));
  return wrapped;
}

// a device record with when it was fetched (s): it says whether the device was online then (app.js isOnline)
const fetched = (device) => ({...device, fetched_at: Date.now() / 1000});

export const api = {
  me: () => auth.user(),
  devices: () => get("v1/me/devices/").then((list) => list.map(fetched)),
  // signed out, a device is never readable: answer the API's 401 without asking
  device: memo(async (dongleId) => {
    if (!auth.isAuthenticated()) throw new ApiError(401, "signed out");
    return fetched(await get(`v1.1/devices/${dongleId}/`));
  }, undefined, 60e3),  // its online state ages
  deviceStats: (dongleId) => get(`v1.1/devices/${dongleId}/stats`),
  deviceLocation: memo((dongleId) => get(`v1/devices/${dongleId}/location`)),
  setDeviceAlias: (dongleId, alias) => request("PATCH", `v1/devices/${dongleId}/`, {body: {alias}}),
  deviceUsers: (dongleId) => get(`v1/devices/${dongleId}/users`),
  addDeviceUser: (dongleId, email) => request("POST", `v1/devices/${dongleId}/add_user`, {body: {email}}),
  // an empty body removes the signed-in user, so that's its own call and never a missing email
  removeDeviceUser: async (dongleId, email) => {
    if (!email) throw new Error("removing a user needs their email");
    return request("POST", `v1/devices/${dongleId}/del_user`, {body: {email}});
  },
  leaveDevice: (dongleId) => request("POST", `v1/devices/${dongleId}/del_user`, {body: {}}),
  unpairDevice: (dongleId) => request("POST", `v1/devices/${dongleId}/unpair`),
  // a form post, as connect sends it; its 401 is the device's pair token failing, not ours, so never auth.expired
  async pilotPair(pairToken) {
    const resp = await fetch(`${apiHost()}/v2/pilotpair/`, {method: "POST", headers: {Authorization: `JWT ${auth.token()}`},
      body: new URLSearchParams({pair_token: pairToken})});
    if (!resp.ok) throw new ApiError(resp.status, await resp.text());
    return resp.json();
  },
  // the owner's comma prime billing; 403 and 404 mean there's none this account can see
  primeSubscription: (dongleId) => billing("GET", "v1/prime/subscription", {params: {dongle_id: dongleId}})
    .catch((e) => { if (e.status === 403 || e.status === 404) return null; throw e; }),
  // prime's other billing calls, as connect makes them. Checkout and the portal answer {url} on Stripe, which comes back
  // to <origin>/<dongle>/prime (from checkout with ?stripe_success=<session id> or ?stripe_cancelled=1)
  prime: {
    subscribeInfo: (dongle_id) => billing("GET", "v1/prime/subscribe_info", {params: {dongle_id}}),
    checkout: (dongle_id, sim_id, plan) => billing("POST", "v1/prime/stripe_checkout", {body: {dongle_id, sim_id, plan}}),
    session: (dongle_id, session_id) => billing("GET", "v1/prime/stripe_session", {params: {dongle_id, session_id}}),
    portal: (dongle_id) => billing("GET", "v1/prime/stripe_portal", {params: {dongle_id}}),
    switchPlan: (dongle_id, plan, sim_id = null) => billing("POST", "v1/prime/switch_plan", {body: {dongle_id, plan, sim_id}}),
    cancel: (dongle_id) => billing("POST", "v1/prime/cancel", {body: {dongle_id}}),
  },
  // the referral code and rewards; 403 for accounts that don't own a device eligible for them
  referrals: () => billing("GET", "v1/referrals"),
  // signed blob urls, oldest first
  bootlogs: memo((dongleId) => get(`v1/devices/${dongleId}/bootlogs`)),
  crashlogs: memo((dongleId) => get(`v1/devices/${dongleId}/crashlogs`)),

  // routes that overlap [start, end] in unix millis, newest first
  routes: memo((dongleId, start, end, limit = 50) => get(`v1/devices/${dongleId}/routes_segments`, {start, end, limit})),
  preservedRoutes: (dongleId) => get(`v1/devices/${dongleId}/routes/preserved`),
  // the device listing carries the share credentials and utc times; v1/route is the fallback for public routes
  route: memo(async (routeName) => {
    const dongleId = routeName.split("|")[0];
    const [r] = auth.isAuthenticated() ? await get(`v1/devices/${dongleId}/routes_segments`, {route_str: routeName}).catch(() => []) : [];
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
  // JSON-RPC on the device; with an expiry, athena queues the call while the device is offline. The relay takes dongle ids
  // lowercase only, as the API gives them (a typed /5BEB… url isn't)
  async athena(dongleId, method, params, expiry) {
    const body = {id: 0, jsonrpc: "2.0", method, params, expiry};
    const resp = await (athenaRelay() ? request("POST", `local/athena/${dongleId.toLowerCase()}`, {host: location.origin, body, headers: {...daemon.headers}})
      : request("POST", dongleId, {host: athenaHost(), body}));
    if (resp.error) throw new Error(resp.error.data?.message || resp.error.message || String(resp.error));
    return resp.result;
  },

  // public routes need no share credentials
  qcameraUrl: (route) => `${apiHost()}/v1/route/${route.fullname}/qcamera.m3u8` +
    (route.share_sig ? "?" + new URLSearchParams({exp: route.share_exp, sig: route.share_sig}) : ""),
  // per-segment derived assets live next to the route on the CDN; null for routes not in the API (no url)
  segmentAsset: (route, segment, name) => route.url ? `${route.url}/${segment}/${name}` : null,
};

// connect's version check: "0.10" is at least "0.9.4"; a device that doesn't say is older than anything
export function versionAtLeast(device, version) {
  if (!device.openpilot_version) return false;
  const have = device.openpilot_version.split(".");
  for (const [i, want] of version.split(".").map(Number).entries()) {
    const v = have[i] ? parseInt(have[i], 10) : 0;
    if (!Number.isInteger(v) || v < want) return false;
    if (v > want) return true;
  }
  return true;
}

export const SEGMENT = 60;  // nominal seconds per segment, until a segment's real span is known

// the segment a timeline slot plays: a route opened with a range (data/sources.js routeRecord) plays its segments
// back to back, so slot k is segment_numbers[k]; otherwise the slot is the segment
export const segmentAt = (route, slot) => route.range ? route.segment_numbers[slot] : slot;
const slots = (route) => route.range ? route.segment_numbers.length : (route.maxqlog ?? -1) + 1;
const memoKey = (route) => `${route.fullname}/${route.range ?? ""}`;

// a per-segment asset for each slot, with the seconds its times move by to land there. A few requests at a time:
// issuing one per segment at once is a long task on long routes. One starts now, and each answer adds another
// worker up to 8, so no single task issues them all
async function assets(route, name) {
  const out = new Array(slots(route));
  const workers = [];
  let next = 0;
  const worker = async () => {
    while (next < out.length) {
      const slot = next++, n = segmentAt(route, slot);
      const url = api.segmentAsset(route, n, name);
      const resp = url && await fetch(url).catch(() => null);
      if (workers.length < 8 && next < out.length) workers.push(worker());
      out[slot] = {shift: (slot - n) * SEGMENT, data: resp?.ok ? await resp.json() : []};
    }
  };
  workers.push(worker());
  for (let i = 0; i < workers.length; i++) await workers[i];
  return out;
}

// GPS fixes [{t (route seconds), lat, lng, speed, dist}] from the per-segment coords.json
export const routeCoords = memo(async (route) => (await assets(route, "coords.json")).flatMap(({shift, data}) =>
  shift ? data.map((c) => ({...c, t: c.t + shift})) : data), memoKey);

// engaged, alert, override and bookmark spans from the per-segment events, in route milliseconds
export const routeEvents = memo(async (route) => {
  const events = (await assets(route, "events.json")).flatMap(({shift, data}) => data.map((ev) => shift ? {...ev, route_offset_millis: ev.route_offset_millis + shift * 1000} : ev))
    .sort((a, b) => a.route_offset_millis - b.route_offset_millis);
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
}, memoKey);

export function engagedFraction(spans, duration) {
  const ms = spans.reduce((sum, s) => sum + (s.kind === "engaged" ? s.end - s.start : 0), 0);
  return duration > 0 ? Math.min(1, ms / duration) : 0;
}
