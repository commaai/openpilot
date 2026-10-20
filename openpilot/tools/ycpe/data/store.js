// the route's series store: per-segment columns from the workers behind one path namespace and one clock.
// Columns stay packed as the workers sent them and become views on first use. Over the memory budget, the least
// recently used columns are dropped and re-parsed from the OPFS cache when asked for again.
import {api} from "../api.js";
import {perf} from "../perf.js";
import {Loader} from "./loader.js";
import {Can} from "./can.js";
import {enumRef, service, unpack} from "./pack.js";

const SEGMENT = 60;  // nominal seconds per segment, until a segment's real span is known
const BUDGET = 1.5e9;  // resident column bytes
const IN_USE = 5000;  // ms since last use that a column counts as in use (never evicted)
const BLOCK = 16, FANOUT = 8;  // min/max pyramid: blocks of 16 samples, then 8 blocks per level up

// opens a route and starts loading its logs outward from t; repeat tiles the route for long-route tests
export async function openRoute(name, opts) {
  const files = await api.routeFiles(name);
  const urls = [];
  for (const u of [...files.qlogs, ...files.logs]) urls[+new URL(u).pathname.split("/").at(-2)] = u;  // rlogs win
  return new Store(name, urls, opts);
}

export class Store {
  // urls[n]: the log of segment n; keep: {service: [list path globs]} to split long lists from the start
  constructor(name, urls, {t = 0, repeat = 1, keep = {}, loader, budget = BUDGET} = {}) {
    const count = urls.length * repeat;
    this.route = {name, t0: null, duration: 0, segments: Array.from({length: count}, (_, n) => ({
      n, t0: n * SEGMENT, t1: (n + 1) * SEGMENT, status: urls[n % urls.length] ? "queued" : "missing",
    }))};
    this.loader = loader || new Loader();
    this.budget = budget;
    this.keep = keep;
    this.handlers = {};
    this.can = new Can(this);
    this.entries = new Map();  // path → {path, type, names?, at: [{pack, k} by segment], views: [], version, used}
    this.lists = new Map();  // long lists the workers skipped: path → longest length
    this.missing = new Set();  // migrated paths that couldn't be computed
    this.evicted = new Map();  // service → evicted paths in it, which loads from now on skip
    this.enums = [];
    this.bytes = 0;
    this.progress = {done: 0, failed: 0, total: 0, bytes: 0, raw: 0, cached: 0, ms: {fetch: 0, decompress: 0, read: 0, ingest: 0}, ingestMax: 0, start: performance.now()};
    this.focusSeg = Math.floor(t / SEGMENT);

    for (const seg of this.route.segments) {
      const local = seg.n % urls.length, url = urls[local];
      if (!url) continue;
      // OPFS keys name the route, segment and file; the signed URLs change
      seg.key = `${name.replace("|", "/")}/${local}/${new URL(url).pathname.split("/").at(-1)}`;
      seg.url = url;
      seg.opts = {n: local, shift: (seg.n - local) * SEGMENT};
      this.progress.total++;
      this.load(seg, undefined, {can: true}).then((p) => {  // CAN frames come out of the first parse only
        this.count(p);
        this.ingest(seg, p);
        if (p.can) this.can.ingest(seg.n, p.can, 0);
      }, (e) => this.failed(seg, e));
    }
  }

  // the options are taken when the parse starts, so it skips whatever was evicted by then
  load(seg, only, extra = {}) {
    const opts = () => {
      const skip = {};
      for (const [svc, subs] of this.evicted) if (subs.size) skip[svc] = [...subs];
      return {...seg.opts, keep: this.keep, skip, only, context: this.context, ...extra};
    };
    return this.loader.load(seg.key, seg.url, opts, this.prio(seg.n), seg.n);
  }

  // outward from the focus segment
  prio(n) { return Math.abs(n - this.focusSeg); }

  focus(t) {
    this.focusSeg = Math.max(0, Math.floor(t / SEGMENT));
    this.loader.reprioritize((n) => this.prio(n));
  }

  on(event, fn) {
    (this.handlers[event] ||= new Set()).add(fn);
    return () => this.handlers[event].delete(fn);
  }

  emit(event, arg) {
    for (const fn of this.handlers[event] || []) fn(arg);
  }

  count(p) {
    const pr = this.progress;
    pr.done++;
    pr.bytes += p.bytes;
    pr.raw += p.raw;
    pr.cached += p.cached;
    for (const k in p.ms) pr.ms[k] += p.ms[k];
  }

  failed(seg, e) {
    seg.status = "error";
    seg.error = String(e);
    this.progress.failed++;
    this.emit("progress", this.progress);
  }

  // registers a pack's columns: O(columns), no views and no copies
  ingest(seg, p) {
    const start = performance.now(), route = this.route;
    route.t0 ??= p.start;
    this.context ||= p.context;  // migrations fall back to it when a segment lacks initData or carParams
    for (const m of p.missing || []) this.missing.add(m);
    if (seg.status !== "done") {
      seg.t0 = p.t0;
      seg.t1 = p.t1;
      seg.status = "done";
      // a lower bound that only grows, exact once the last segment is in
      const last = route.segments.at(-1);
      route.duration = last.status === "done" ? last.t1 : Math.max(route.duration, seg.t1, (route.segments.length - 1) * SEGMENT);
    }
    Object.assign(this.enums, p.enums);
    for (const path in p.lists) this.lists.set(path, Math.max(this.lists.get(path) || 0, p.lists[path]));

    const paths = p.paths ? p.paths.split("\n") : [];
    p.types = p.types.split("\n");
    p.live = p.counts.map(() => 0);
    let added = false;
    for (let k = 0; k < paths.length; k++) {
      let e = this.entries.get(paths[k]);
      if (!e) {
        const ref = enumRef(p, k);
        e = {path: paths[k], type: p.types[k], ...(ref >= 0 && {names: this.enums[ref]}), at: [], views: [], version: 0, used: 0};
        this.entries.set(paths[k], e);
        added = true;
      }
      if (e.evicted) continue;  // not resident until asked for again
      if (e.at[seg.n]) this.release(e.at[seg.n]);
      e.at[seg.n] = {pack: p, k};
      e.views[seg.n] = null;
      e.version++;
      p.live[service(p, k)]++;
    }
    p.live.forEach((n, s) => n ? (this.bytes += p.bufs[s].byteLength) : drop(p, s));
    const ms = performance.now() - start;
    this.progress.ms.ingest += ms;
    this.progress.ingestMax = Math.max(this.progress.ingestMax, ms);
    if (added) this.emit("paths");
    this.emit("segment", seg.n);
    this.emit("progress", this.progress);
    if (this.bytes > this.budget) this.evict();
    perf.mark("store", `${(this.bytes / 1e6).toFixed(0)} / ${(this.budget / 1e6).toFixed(0)} MB, ${this.entries.size} paths`);
    perf.mark("ingest", `${ms.toFixed(1)} ms (max ${this.progress.ingestMax.toFixed(1)})`);
  }

  paths() {
    const out = this.can.paths();
    for (const {path, type, names} of this.entries.values()) out.push(names ? {path, type, names} : {path, type});
    return out;
  }

  // the chunks of a path; an unknown index of a skipped long list (/modelV2/position/x/32) starts loading it
  series(path) {
    if (path.startsWith("/dbc/")) return this.can.series(path);
    const e = this.entries.get(path);
    if (!e) return this.request(path), null;
    e.used = performance.now();
    if (e.evicted) this.reload(e);
    if (e.series?.version !== e.version) {
      const chunks = [];
      e.at.forEach((a, n) => a && chunks.push(e.views[n] ||= unpack(a.pack, a.k, n)));
      e.series = {path, type: e.type, names: e.names, chunks, version: e.version};
    }
    return e.series;
  }

  // registers or replaces a derived series (transforms, /custom/*) as one chunk
  put(path, {type, t, v, names}) {
    const old = this.entries.get(path);
    const e = {path, type, ...(names && {names}), at: [], views: [{n: 0, t, v, tt: t}], version: (old?.version || 0) + 1, used: Infinity};
    e.series = {path, type, names, chunks: e.views, version: e.version};
    this.entries.set(path, e);
    if (!old) this.emit("paths");
  }

  // n min/max buckets over [t0, t1] read from the pyramids; the exact samples when there are no more than n
  decimate(path, t0, t1, n) {
    const s = this.series(path);
    if (!s || s.type === "t") return null;
    let total = 0;
    for (const c of s.chunks) total += Math.max(0, upperBound(c.tt, t1) - lowerBound(c.tt, t0));
    if (total <= n) {
      const t = [], v = [];
      for (const c of s.chunks) {
        for (let k = lowerBound(c.tt, t0), end = upperBound(c.tt, t1); k < end; k++) {
          if (c.v[k] === c.v[k]) { t.push(c.tt[k]); v.push(c.v[k]); }
        }
      }
      const vals = Float64Array.from(v);
      return {t: Float64Array.from(t), min: vals, max: vals};
    }
    const w = (t1 - t0) / n;
    const t = new Float64Array(n), min = new Float64Array(n).fill(Infinity), max = new Float64Array(n).fill(-Infinity);
    for (let b = 0; b < n; b++) t[b] = t0 + (b + 0.5) * w;
    const add = (time, lo, hi) => {
      const b = Math.min(n - 1, Math.floor((time - t0) / w));
      if (lo < min[b]) min[b] = lo;
      if (hi > max[b]) max[b] = hi;
    };
    for (const c of s.chunks) {
      const {tt, v} = c, k0 = lowerBound(tt, t0), k1 = upperBound(tt, t1);
      if (k1 <= k0) continue;
      // the coarsest level whose blocks still fit four to a bucket
      const perBucket = (k1 - k0) / Math.max(1, (tt[k1 - 1] - tt[k0]) / w);
      const levels = perBucket >= 4 * BLOCK ? pyramid(c) : [];
      let level = -1, size = 1;
      while (level + 1 < levels.length && (level < 0 ? BLOCK : size * FANOUT) * 4 <= perBucket) {
        size = level < 0 ? BLOCK : size * FANOUT;
        level++;
      }
      const b0 = Math.ceil(k0 / size), b1 = Math.floor(k1 / size);
      if (level < 0 || b1 <= b0) {
        for (let k = k0; k < k1; k++) add(tt[k], v[k], v[k]);
        continue;
      }
      const {lo, hi} = levels[level];
      for (let k = k0; k < b0 * size; k++) add(tt[k], v[k], v[k]);
      for (let b = b0; b < b1; b++) add(tt[b * size], lo[b], hi[b]);
      for (let k = b1 * size; k < k1; k++) add(tt[k], v[k], v[k]);
    }
    for (let b = 0; b < n; b++) if (min[b] > max[b]) min[b] = max[b] = NaN;
    return {t, min, max};
  }

  // the last value at or before t
  valueAt(path, t) {
    const s = this.series(path);
    if (!s) return undefined;
    const {chunks} = s;
    let lo = 0, hi = chunks.length;  // first chunk starting after t
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (chunks[mid].tt[0] <= t) lo = mid + 1; else hi = mid;
    }
    for (let j = lo - 1; j >= 0; j--) {
      const {tt, v} = chunks[j];
      for (let k = upperBound(tt, t) - 1; k >= 0; k--) if (v[k] === v[k] && v[k] !== null) return v[k];
    }
    return undefined;
  }

  async message(service, t) { return null; }

  // a skipped long list is asked for by any of its indexes: keep it from now on and re-parse what's loaded
  request(path) {
    const m = path.match(/^\/([^/]+)(\/.*)\/\d+(\/.*)?$/);
    if (!m || !this.lists.has(`/${m[1]}${m[2]}`)) return;
    const glob = m[2], list = this.keep[m[1]] ||= [];
    if (list.includes(glob)) return;
    list.push(glob);
    this.reparse({[m[1]]: [`${glob}/*`, `${glob}/*/*`]});
  }

  reload(e) {
    e.evicted = false;
    const [svc, sub] = split(e.path);
    this.evicted.get(svc).delete(sub);
    this.reparse({[svc]: [sub]});
  }

  // requests in the same task share one parse per segment
  reparse(only) {
    if (!this.wanted) {
      this.wanted = {};
      queueMicrotask(() => {
        const only = this.wanted;
        this.wanted = null;
        for (const seg of this.route.segments) {
          if (seg.status === "done") this.load(seg, only).then((p) => this.ingest(seg, p), () => {});
        }
      });
    }
    for (const svc in only) (this.wanted[svc] ||= []).push(...only[svc]);
  }

  // drops the least recently used columns not in use until under 90% of the budget
  evict() {
    const now = performance.now(), target = this.budget * 0.9;
    const lru = [...this.entries.values()].filter((e) => e.at.length && now - e.used > IN_USE).sort((a, b) => a.used - b.used);
    for (const e of lru) {
      if (this.bytes <= target) break;
      e.at.forEach((a) => this.release(a));
      e.at = [];
      e.views = [];
      e.series = null;
      e.evicted = true;
      e.version++;
      const [svc, sub] = split(e.path);
      if (!this.evicted.has(svc)) this.evicted.set(svc, new Set());
      this.evicted.get(svc).add(sub);
    }
  }

  // a pack's service buffer goes once none of its columns are referenced
  release({pack, k}) {
    const s = service(pack, k);
    if (--pack.live[s] === 0) {
      this.bytes -= pack.bufs[s].byteLength;
      drop(pack, s);
    }
  }

  dispose() { this.loader.dispose(); this.can.dispose(); }
}

const split = (path) => path.match(/^\/([^/]+)(.*)$/).slice(1);

function drop(pack, s) {
  pack.bufs[s] = null;
  if (pack.t) pack.t[s] = null;
}

// min/max per block of BLOCK samples, then per FANOUT blocks, up to one block per chunk; built on first need
function pyramid(c) {
  if (c.pyramid) return c.pyramid;
  const {v} = c, Arr = ArrayBuffer.isView(v) ? v.constructor : Float64Array;
  const levels = [];
  let lo = v, hi = v, size = BLOCK;
  while (lo.length >= size) {
    const m = Math.ceil(lo.length / size), nlo = new Arr(m), nhi = new Arr(m);
    for (let b = 0; b < m; b++) {
      let a = Infinity, z = -Infinity;
      for (let k = b * size, end = Math.min(lo.length, k + size); k < end; k++) {
        if (lo[k] < a) a = lo[k];
        if (hi[k] > z) z = hi[k];
      }
      nlo[b] = a;
      nhi[b] = z;
    }
    levels.push({lo: nlo, hi: nhi});
    lo = nlo;
    hi = nhi;
    size = FANOUT;
  }
  return (c.pyramid = levels);
}

// first k with a[k] >= x
function lowerBound(a, x) {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// first k with a[k] > x
function upperBound(a, x) {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] <= x) lo = mid + 1; else hi = mid;
  }
  return lo;
}
