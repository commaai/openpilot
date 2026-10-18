// comma API client: a JWT in localStorage, plain fetch

export const API_HOST = localStorage.getItem("api_host") || "https://api.commadotai.com";
const TOKEN_KEY = "comma_jwt";

export const auth = {
  token: () => localStorage.getItem(TOKEN_KEY),
  isAuthenticated: () => !!localStorage.getItem(TOKEN_KEY),
  logOut() {
    localStorage.removeItem(TOKEN_KEY);
    location.href = "/";
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

  async exchangeCode(code, provider) {
    const resp = await fetch(`${API_HOST}/v2/auth/`, {
      method: "POST",
      headers: {"Content-Type": "application/x-www-form-urlencoded"},
      body: new URLSearchParams({code, provider}),
    });
    if (!resp.ok) throw new Error(`sign-in failed (${resp.status})`);
    const {access_token} = await resp.json();
    localStorage.setItem(TOKEN_KEY, access_token);
  },
};

export class ApiError extends Error {
  constructor(status, body) {
    super(`API ${status}: ${body}`);
    this.status = status;
  }
}

export async function request(method, endpoint, {params, body} = {}) {
  const url = new URL(endpoint, API_HOST + "/");
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== undefined && v !== null) url.searchParams.set(k, v);
  }
  const headers = {};
  if (auth.token()) headers["Authorization"] = "JWT " + auth.token();
  if (body) headers["Content-Type"] = "application/json";
  const resp = await fetch(url, {method, headers, body: body ? JSON.stringify(body) : undefined});
  if (resp.status === 401) auth.logOut();
  if (!resp.ok) throw new ApiError(resp.status, await resp.text());
  return resp.json();
}
const get = (endpoint, params) => request("GET", endpoint, {params});

export const api = {
  me: () => get("v1/me/"),
  devices: () => get("v1/me/devices/"),
  device: (dongleId) => get(`v1.1/devices/${dongleId}/`),
  deviceStats: (dongleId) => get(`v1.1/devices/${dongleId}/stats`),
  deviceLocation: (dongleId) => get(`v1/devices/${dongleId}/location`),
  setDeviceAlias: (dongleId, alias) => request("PATCH", `v1/devices/${dongleId}/`, {body: {alias}}),
  // routes that overlap [start, end] in unix millis, newest first
  routes: (dongleId, start, end, limit = 50) => get(`v1/devices/${dongleId}/routes_segments`, {start, end, limit}),
  preservedRoutes: (dongleId) => get(`v1/devices/${dongleId}/routes/preserved`),
  // the device listing carries the share credentials and utc times; v1/route is the fallback for public routes
  async route(routeName) {
    const dongleId = routeName.split("|")[0];
    const [r] = await get(`v1/devices/${dongleId}/routes_segments`, {route_str: routeName}).catch(() => []);
    if (r) return r;
    const route = await get(`v1/route/${routeName}/`);
    route.start_time_utc_millis ??= Date.parse(route.start_time + "Z");
    route.end_time_utc_millis ??= Date.parse(route.end_time + "Z");
    return route;
  },
  routeFiles: (routeName) => get(`v1/route/${routeName}/files`),
  setRoutePublic: (routeName, isPublic) => request("PATCH", `v1/route/${routeName}/`, {body: {is_public: isPublic}}),
  setRoutePreserved: (routeName, preserved) => request(preserved ? "POST" : "DELETE", `v1/route/${routeName}/preserve`),

  // public routes need no share credentials
  qcameraUrl: (route) => `${API_HOST}/v1/route/${route.fullname}/qcamera.m3u8` +
    (route.share_sig ? "?" + new URLSearchParams({exp: route.share_exp, sig: route.share_sig}) : ""),
  // per-segment derived assets live next to the route on the CDN
  segmentAsset: (route, segment, name) => `${route.url}/${segment}/${name}`,
};

// engaged, alert and override spans from the per-segment events, in route milliseconds
export async function routeEvents(route) {
  const segments = Array.from({length: (route.maxqlog ?? -1) + 1}, (_, i) => i);
  const lists = await Promise.all(segments.map(async (n) => {
    const resp = await fetch(api.segmentAsset(route, n, "events.json"));
    return resp.ok ? resp.json() : [];
  }));
  const events = lists.flat().sort((a, b) => a.route_offset_millis - b.route_offset_millis);
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
    if (ev.type !== "state") continue;
    const t = ev.route_offset_millis;
    track("engaged", ev.data.enabled, t);
    track("alert", ev.data.alertStatus === "userPrompt" || ev.data.alertStatus === 1, t);
    track("critical", ev.data.alertStatus === "critical" || ev.data.alertStatus === 2, t);
    track("override", ev.data.state === "overriding" || ev.data.state === "preEnabled", t);
  }
  const duration = route.end_time_utc_millis - route.start_time_utc_millis;
  for (const s of spans) if (s.end === null) s.end = duration;
  return spans;
}
