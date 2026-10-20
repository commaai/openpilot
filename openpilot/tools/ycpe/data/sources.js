// where a route's files come from: the sources, order and selector syntax of the Python tools
// (tools/lib/logreader.py, file_sources.py, route.py), plus local files. A source finds, for some segments of one
// file kind, {url, file, size?}; file is the name the OPFS cache keys on, the same whichever source served it.
import {api} from "../api.js";

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

// ---- requests ----

function once(fn) {
  let p = null;
  return () => p ||= fn();
}

// a few HEADs at a time; each URL is asked once. error: the request itself failed (offline, or refused by CORS)
const heads = new Map();
const queue = [];
let inFlight = 0;
export function head(url) {
  if (!heads.has(url)) {
    heads.set(url, new Promise((resolve) => queue.push({url, resolve})));
    pump();
  }
  return heads.get(url);
}
function pump() {
  while (inFlight < 8 && queue.length) {
    const {url, resolve} = queue.shift();
    inFlight++;
    fetch(url, {method: "HEAD"})
      .then((r) => ({ok: r.ok, size: r.ok ? +r.headers.get("content-length") || null : null, type: r.headers.get("content-type") || ""}),
        () => ({ok: false, size: null, type: "", error: true}))
      .then(resolve)
      .finally(() => { inFlight--; pump(); });
  }
}

// eval_source: the first candidate file that exists, per segment
async function firstExisting(segs, candidates) {
  const found = new Map();
  await Promise.all(segs.map(async (seg) => {
    for (const [file, url] of candidates(seg)) {
      const r = await head(url);
      if (r.ok) return found.set(seg, {url, file, size: r.size});
    }
  }));
  return found;
}

// ---- the local daemon (tools/ycpe/daemon): the sources a browser can't reach, resolved and served from localhost ----

export const daemon = {
  // its capabilities, or null when the page isn't served by it (a static host answers with index.html)
  caps: once(async () => {
    const r = await fetch("/local/caps").catch(() => null);
    return r?.ok && r.headers.get("content-type")?.includes("json") ? r.json() : null;
  }),
  // {segments: {n: {kind: {url, file, size, source}}}, errors: {source: message}}
  async resolve(sr, segs, kinds, sources) {
    const r = await fetch("/local/sources/resolve", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({route: sr.route, segs, kinds, sources})});
    if (!r.ok) throw new Error(`daemon: ${r.status} ${await r.text()}`);
    return r.json();
  },
};

// a source only the daemon reaches (no CORS, or an internal network); with the daemon, resolveFiles asks it instead
const daemonOnly = async () => {
  throw new Error("needs the local daemon");
};

// ---- the sources ----

const ccsRepo = () => setting("ycpe_ccs_repo") || "https://huggingface.co/datasets/commaai/commaCarSegments";
const ccsBranch = () => setting("ycpe_ccs_branch") || "main";
const ccsLfs = () => setting("ycpe_ccs_lfs") || ccsRepo();

export const SOURCES = {
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
    // raw files are LFS pointers; one batch call turns them all into download URLs
    async find(sr, segs, kind) {
      const raw = (seg, f) => `${ccsRepo()}/raw/${ccsBranch()}/segments/${sr.dongle}/${sr.logId}/${seg}/${f}`;
      const found = await firstExisting(segs, (seg) => KINDS[kind].map((f) => [f, raw(seg, f)]));
      if (!found.size && segs.length && (await head(raw(segs[0], KINDS[kind][0]))).error) {
        throw new Error("huggingface.co refused this origin (it answers CORS for localhost:3000)");
      }
      const pointers = new Map();
      await Promise.all([...found].map(async ([seg, f]) => {
        if (!(await head(f.url)).type.startsWith("text/plain")) return;  // not in LFS yet: the raw file is the file
        const m = (await (await fetch(f.url)).text()).match(/^version https:\/\/git-lfs\.github\.com\/spec\/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)/);
        if (m) pointers.set(seg, {oid: m[1], size: +m[2]});
      }));
      if (!pointers.size) return found;
      const r = await fetch(`${ccsLfs()}.git/info/lfs/objects/batch`, {
        method: "POST",
        headers: {"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"},
        body: JSON.stringify({operation: "download", transfers: ["basic"], hash_algo: "sha256", objects: [...pointers.values()]}),
      });
      if (!r.ok) throw new Error(`LFS batch ${r.status}`);
      const hrefs = new Map((await r.json()).objects.filter((o) => !o.error).map((o) => [o.oid, o.actions.download.href]));
      for (const [seg, {oid, size}] of pointers) {
        if (hrefs.has(oid)) found.set(seg, {...found.get(seg), url: hrefs.get(oid), size});
        else found.delete(seg);
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
  list: (name) => name === "auto" ? AUTO : [name],
};

// file_sources' walk: for each kind, each source in turn on the segments still without it. Partial results are kept,
// not raised. → {found: Map "<n>/<kind>" → {url, file, size, source}, errors: {source: message}}
export async function resolveFiles(sr, segs, kinds, sources = AUTO) {
  const found = new Map(), errors = {};
  if (await daemon.caps()) {
    const r = await daemon.resolve(sr, segs, kinds, sources);
    for (const [n, byKind] of Object.entries(r.segments)) for (const [kind, f] of Object.entries(byKind)) found.set(`${n}/${kind}`, f);
    return {found, errors: r.errors || {}};
  }
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
export async function resolveLogs(sr, segs, {sources = AUTO, mode = sr.selector || "a"} = {}) {
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

// ---- local files ----

// the segment and route each dropped or picked file belongs to, from its path: "<route>--<seg>--<file>" (explorer),
// "<dongle>/<log id>/<seg>/<file>", "<route>--<seg>/<file>" or "<log id>--<seg>/<file>" (a device's realdata);
// a lone file is segment 0
const LOCAL = new RegExp(`(?:(${DONGLE})[|_/])?(${LOG_ID})(?:--|/)([0-9]+)(?:--|/)([^/]+)$`);
export function localFiles(files) {
  let route = null;
  const segs = new Map();  // seg → Map kind → {url, file, size, source}
  for (const f of files) {
    const path = f.webkitRelativePath || f.relativePath || f.name;
    const m = path.match(LOCAL);
    const file = m ? m[4] : f.name, kind = KIND_OF[file];
    if (!kind) continue;
    if (m?.[1]) route ||= `${m[1]}|${m[2]}`;
    const seg = m ? +m[3] : 0;
    if (!segs.has(seg)) segs.set(seg, new Map());
    segs.get(seg).set(kind, {url: URL.createObjectURL(f), file, size: f.size, source: "local"});
  }
  return {route, segs};
}

// every file under a drop, folders included
export async function droppedFiles(dataTransfer) {
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
  for (const entry of [...dataTransfer.items].map((i) => i.webkitGetAsEntry?.()).filter(Boolean)) await walk(entry, "");
  return out;
}
