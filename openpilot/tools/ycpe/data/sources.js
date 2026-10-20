// where a route's files come from: the sources, order and selector syntax of the Python tools
// (tools/lib/logreader.py, file_sources.py, route.py), plus local files. A source finds, for some segments of one
// file kind, {url, file, size?}; file is the name the OPFS cache keys on, the same whichever source served it.
import {api, auth} from "../api.js";

const setting = (key) => globalThis.localStorage?.getItem(key);

export const KINDS = {
  qlog: ["qlog.zst", "qlog.bz2"], rlog: ["rlog.zst", "rlog.bz2"], qcamera: ["qcamera.ts"],
  fcamera: ["fcamera.hevc"], ecamera: ["ecamera.hevc"], dcamera: ["dcamera.hevc"],
};
export const API_KIND = {qlog: "qlogs", rlog: "logs", qcamera: "qcameras", fcamera: "cameras", ecamera: "ecameras", dcamera: "dcameras"};
const KIND_OF = Object.fromEntries(Object.entries(KINDS).flatMap(([k, files]) => files.map((f) => [f, k])));

// ---- names: route.py's RE, SegmentRange and logreader.parse_indirect ----

const DONGLE = "[a-f0-9]{16}";
const LOG_ID = "[0-9]{4}-[0-9]{2}-[0-9]{2}--[0-9]{2}-[0-9]{2}-[0-9]{2}|[a-f0-9]{8}--[a-z0-9]{10}";
const ROUTE = `(${DONGLE})[|_/](${LOG_ID})`;
const SLICE = /^(-?[0-9]+)?:?(-?[0-9]+)?:?(-?[0-9]+)?$/;
const RANGE = new RegExp(`^${ROUTE}(?:(?:--|/)((?:-?[0-9]+)?:?(?:-?[0-9]+)?:?(?:-?[0-9]+)?))?(?:/([qra]))?$`);

// useradmin and connect URLs to a segment range (connect's start and end seconds become segments)
export function parseIndirect(text) {
  if (text.includes("useradmin.comma.ai")) return new URL(text).searchParams.get("onebox") || text;
  if (!text.includes("connect.comma.ai")) return text;
  const p = new URL(text).pathname.split("/").filter(Boolean);
  const route = p.slice(0, 2).join("/");
  if (p.length <= 3) return [route, ...p.slice(2)].join("/");
  return `${route}/${Math.floor(p[2] / 60)}:${Math.floor(p[3] / 60) + 1}${p[4] ? `/${p[4]}` : ""}`;
}

// "<dongle>|<log id>[--|/<slice>][/q|r|a]" → {route, dongle, logId, slice, selector}, or null
export function parseRange(text) {
  const m = parseIndirect(text.trim()).match(RANGE);
  return m && {route: `${m[1]}|${m[2]}`, dongle: m[1], logId: m[2], slice: m[3] || "", selector: m[4] || null};
}

// the segment numbers a slice names, as Python slices list(range(n)); maxSeg is needed only for open or negative ends
export function segIdxs({slice}, maxSeg) {
  const m = slice.match(SLICE);
  if (!m) throw new Error(`invalid slice: ${slice}`);
  const [start, end, step] = m.slice(1).map((s) => s === undefined ? null : +s);
  const need = () => {
    if (maxSeg == null) throw new Error("the segment count is unknown: give an explicit range, like /0:10");
    return maxSeg + 1;
  };
  if (start !== null && end === null && !slice.includes(":")) return [start < 0 ? start + need() : start];
  const n = end === null || end < 0 || start < 0 ? need() : end + 1;
  return pySlice(n, start, end, step ?? 1);
}

function pySlice(n, start, stop, step) {
  if (!step) throw new Error("slice step cannot be zero");
  const norm = (i, lo, hi) => Math.min(Math.max(i < 0 ? i + n : i, lo), hi);
  const out = [];
  if (step > 0) {
    for (let i = start === null ? 0 : norm(start, 0, n), e = stop === null ? n : norm(stop, 0, n); i < e; i += step) out.push(i);
  } else {
    for (let i = start === null ? n - 1 : norm(start, -1, n - 1), e = stop === null ? -1 : norm(stop, -1, n - 1); i > e; i += step) out.push(i);
  }
  return out;
}

// ---- the local daemon (serve.py): sources a browser can't reach, resolved and served from localhost ----

// /daemon.js is there only when the daemon serves the page
const local = await import("/daemon.js").then((m) => m.daemon, () => null);

export const daemon = {
  // the sources it resolves, or null without it
  sources: () => local?.present && local.has("sources") ? local.caps.modules.sources.sources : null,
  // {segments: {n: {kind: {url, file, size, source, cached}}}, missing, errors: {source: message}}; the page's sign-in
  // stands in for the daemon's own. sizes: HEAD the API's files too (other sources always know theirs).
  resolve: (sr, segs, kinds, sources, sizes = false) => local.json("/local/sources/resolve", {method: "POST",
    headers: auth.token() ? {Authorization: `JWT ${auth.token()}`} : {}, body: {route: sr.route, segs, kinds, sources, sizes}}),
};

// a source only the daemon reaches (no CORS, or not this machine's network); with it, resolveFiles asks it instead
const daemonOnly = async () => {
  throw new Error("needs the local daemon");
};

// ---- the sources ----

// commaCarSegments (comma_car_segments.py): a route's directory listing in one request, then one LFS batch call for
// the download URLs of the files asked for. Both answer CORS for any origin (raw/ only for some).
const ccsRepo = () => setting("ycpe_ccs_repo") || "https://huggingface.co/datasets/commaai/commaCarSegments";
const ccsBranch = () => setting("ycpe_ccs_branch") || "main";
const ccsLfs = () => setting("ycpe_ccs_lfs") || ccsRepo();

// "<n>/<file>" → {size, lfs: {oid, size} | undefined}; empty when the route isn't in the dataset
const ccsTrees = new Map();
function ccsTree(sr) {
  if (!ccsTrees.has(sr.route)) {
    const url = `${ccsRepo().replace("huggingface.co/", "huggingface.co/api/")}/tree/${ccsBranch()}/segments/${sr.dongle}/${sr.logId}?recursive=true`;
    ccsTrees.set(sr.route, fetch(url).then(async (r) => {
      if (r.status === 404) return new Map();
      if (!r.ok) throw new Error(`huggingface.co ${r.status}`);
      return new Map((await r.json()).filter((e) => e.type === "file").map((e) => [e.path.split("/").slice(-2).join("/"), e]));
    }, () => {
      throw new Error("huggingface.co isn't reachable");
    }));  // a refusal is kept too: asking again won't change it
  }
  return ccsTrees.get(sr.route);
}

async function lfsUrls(objects) {
  const r = await fetch(`${ccsLfs()}.git/info/lfs/objects/batch`, {
    method: "POST",
    headers: {"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"},
    body: JSON.stringify({operation: "download", transfers: ["basic"], hash_algo: "sha256", objects}),
  });
  if (!r.ok) throw new Error(`LFS batch ${r.status}`);
  return new Map((await r.json()).objects.filter((o) => !o.error).map((o) => [o.oid, o.actions.download.href]));
}

export const SOURCES = {
  local: {label: "local disk", daemon: true, find: daemonOnly},
  device: {label: "device", daemon: true, find: daemonOnly},
  internal: {label: "internal", daemon: true, find: daemonOnly},
  api: {
    label: "comma API",
    async find(sr, segs, kind) {
      const files = await api.routeFiles(sr.route);
      const bySeg = new Map((files[API_KIND[kind]] || []).map((url) => {
        const p = new URL(url).pathname.split("/");
        return [+p.at(-2), {url, file: p.at(-1), size: null}];
      }));
      return new Map(segs.filter((s) => bySeg.has(s)).map((s) => [s, bySeg.get(s)]));
    },
  },
  openpilotci: {label: "openpilotci", daemon: true, find: daemonOnly},
  carSegments: {
    label: "commaCarSegments",
    async find(sr, segs, kind) {
      const tree = await ccsTree(sr);
      const found = new Map();
      for (const n of segs) {
        const file = KINDS[kind].find((f) => tree.has(`${n}/${f}`));
        if (!file) continue;
        const {size, lfs} = tree.get(`${n}/${file}`);
        // a file not in LFS yet is served raw
        found.set(n, {url: lfs ? null : `${ccsRepo()}/raw/${ccsBranch()}/segments/${sr.dongle}/${sr.logId}/${n}/${file}`, file, size: lfs?.size ?? size, lfs});
      }
      const pointers = [...found.values()].filter((f) => f.lfs);
      const hrefs = pointers.length ? await lfsUrls(pointers.map(({lfs}) => ({oid: lfs.oid, size: lfs.size}))) : new Map();
      for (const [n, {lfs, ...f}] of found) {
        if (lfs && !hrefs.has(lfs.oid)) found.delete(n);
        else found.set(n, lfs ? {...f, url: hrefs.get(lfs.oid)} : f);
      }
      return found;
    },
  },
};

export const AUTO = ["internal", "api", "openpilotci", "carSegments"];

// the source a route's files come from: "auto" (AUTO's order) or one SOURCES name, kept per route
export const routeSource = {
  get: (route) => setting(`ycpe_source:${route}`) || "auto",
  set: (route, name) => name === "auto" ? localStorage.removeItem(`ycpe_source:${route}`) : localStorage.setItem(`ycpe_source:${route}`, name),
  list: (name) => name === "auto" ? "auto" : [name],
};

// file_sources' walk: for each kind, each source in turn on the segments still without it. Partial results are kept,
// not raised. "auto" is the daemon's order, or without it AUTO's sources a browser reaches.
// → {found: Map "<n>/<kind>" → {url, file, size, source}, errors: {source: message}}
export async function resolveFiles(sr, segs, kinds, sources = "auto", {sizes = false} = {}) {
  const found = new Map(), errors = {};
  if (daemon.sources()) {
    const r = await daemon.resolve(sr, segs, kinds, sources, sizes);
    for (const [n, byKind] of Object.entries(r.segments)) for (const [kind, f] of Object.entries(byKind)) found.set(`${n}/${kind}`, f);
    return {found, errors: r.errors || {}};
  }
  if (sources === "auto") sources = AUTO.filter((name) => !SOURCES[name].daemon);
  await Promise.all(kinds.map(async (kind) => {
    let need = segs;
    for (const name of sources) {
      if (!need.length) break;
      try {
        for (const [n, f] of await SOURCES[name].find(sr, need, kind)) found.set(`${n}/${kind}`, {...f, source: name});
      } catch (e) {
        errors[name] = e.message;
      }
      need = need.filter((n) => !found.has(`${n}/${kind}`));
    }
  }));
  return {found, errors};
}

// logreader.auto_source: rlogs (qlogs for /q) for every segment; /a, the default here, falls back to qlogs for what no
// source has as an rlog. → {logs: Map n → {url, file, size, source, kind}, missing: [n], errors}
export async function resolveLogs(sr, segs, {sources = "auto", mode = sr.selector || "a"} = {}) {
  const logs = new Map(), errors = {};
  let need = segs;
  for (const kind of mode === "q" ? ["qlog"] : mode === "r" ? ["rlog"] : ["rlog", "qlog"]) {
    if (!need.length) break;
    const r = await resolveFiles(sr, need, [kind], sources);
    Object.assign(errors, r.errors);
    for (const n of need) if (r.found.has(`${n}/${kind}`)) logs.set(n, {...r.found.get(`${n}/${kind}`), kind});
    need = need.filter((n) => !logs.has(n));
  }
  return {logs, missing: need, errors};
}

// what the store loads for a route (a name, optionally with a range and selector), from the route's chosen source:
// urls[n] = {url, file, source, kind}. Fails only when no source has any of it.
export async function routeLogs(name) {
  const sr = parseRange(name);
  const route = await api.route(sr.route).catch(() => null);
  const {logs, errors} = await resolveLogs(sr, segIdxs(sr, route?.maxqlog), {sources: routeSource.list(routeSource.get(sr.route))});
  if (!logs.size && Object.keys(errors).length) throw new Error(Object.entries(errors).map(([s, m]) => `${SOURCES[s]?.label ?? s}: ${m}`).join("; "));
  const urls = [];
  for (const [n, f] of logs) urls[n] = f;
  return urls;
}

// ---- routes outside the API: local files and direct URLs ----

const slug = (s) => s.toLowerCase().replace(/[^a-z0-9-]+/g, "-");

// an API-shaped route record for files found outside the API, so a workspace can open them. files: Map "<n>/<kind>"
// → {url, file, size, source}; the record carries them (route.files) and the store gets each segment's log.
function looseRoute(name, files) {
  const urls = [];
  for (const [key, f] of files) {
    const [n, kind] = key.split("/");
    if (kind === "rlog" || kind === "qlog" && urls[+n]?.kind !== "rlog") urls[+n] = {...f, kind};
  }
  const ns = [...new Set([...files.keys()].map((k) => +k.split("/")[0]))].sort((a, b) => a - b);
  const t0 = Date.parse(name.split("|")[1].replace(/^(\d{4}-\d\d-\d\d)--(\d\d)-(\d\d)-(\d\d)$/, "$1T$2:$3:$4")) || 0;  // timestamp log ids only
  return {
    route: {fullname: name, dongle_id: name.split("|")[0], maxqlog: ns.at(-1), segment_numbers: ns, files, loose: true,
      start_time_utc_millis: t0, end_time_utc_millis: t0 + 60000 * (ns.at(-1) + 1)},
    urls,
  };
}

// where a file belongs, from its path or URL: "<route>--<seg>--<file>" (explorer), "<dongle>/<log id>/<seg>/<file>",
// "<route>--<seg>/<file>" or "<log id>--<seg>/<file>" (a device's realdata); a lone file is segment 0
const PLACE = new RegExp(`(?:(${DONGLE})[|_/])?(${LOG_ID})(?:--|/)([0-9]+)(?:--|/)([^/]+)$`);
function place(path) {
  const m = path.match(PLACE), file = m ? m[4] : path.split("/").at(-1);
  if (!KIND_OF[file]) return null;
  return {file, kind: KIND_OF[file], n: m ? +m[3] : 0, route: m?.[1] ? `${m[1]}|${m[2]}` : `local|${slug(m ? m[2] : file)}`};
}

// dropped or picked files (a folder of segments, or single logs) as a route; files of other routes than the one
// with the most are left out. null when none of them is a log or camera file.
export function localRoute(files) {
  const byRoute = new Map();
  for (const f of files) {
    const p = place(f.relativePath || f.webkitRelativePath || f.name);
    if (!p) continue;
    if (!byRoute.has(p.route)) byRoute.set(p.route, new Map());
    byRoute.get(p.route).set(`${p.n}/${p.kind}`, {url: URL.createObjectURL(f), file: p.file, size: f.size, source: "local"});
  }
  const [name, found] = [...byRoute].sort((a, b) => b[1].size - a[1].size)[0] || [];
  return name ? looseRoute(name, found) : null;
}

// a direct link to one log or camera file, as a one-segment route
export function urlRoute(url) {
  const p = URL.canParse(url) && place(decodeURIComponent(new URL(url).pathname));
  return p && looseRoute(p.route, new Map([[`${p.n}/${p.kind}`, {url, file: p.file, size: null, source: "url"}]]));
}

// every file under a drop, folders included. The entries are taken before the first await, while the drop's
// dataTransfer is still readable.
export async function droppedFiles(dataTransfer) {
  const entries = [...dataTransfer.items].map((i) => i.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return [...dataTransfer.files];
  const out = [];
  const walk = async (entry, dir) => {
    if (entry.isFile) {
      const f = await new Promise((resolve, reject) => entry.file(resolve, reject));
      f.relativePath = dir + f.name;
      out.push(f);
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      for (let batch; (batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject))).length;) {
        for (const e of batch) await walk(e, `${dir}${entry.name}/`);
      }
    }
  };
  for (const entry of entries) await walk(entry, "");
  return out;
}
