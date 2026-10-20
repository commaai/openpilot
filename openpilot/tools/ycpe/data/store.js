// the route's series store: per-segment columns from the workers behind one path namespace and one clock.
// Columns stay packed as the workers sent them and become views on first use. Over the memory budget, the least
// recently used columns are dropped and re-parsed from the OPFS cache when asked for again.
import {api} from "../api.js";
import {perf} from "../perf.js";
import {status} from "../status.js";
import {Loader, POOL} from "./loader.js";
import * as opfs from "./opfs.js";
import {Can} from "./can.js";
import {enumRef, service, times, unpack} from "./pack.js";

const SEGMENT = 60;  // nominal seconds per segment, until a segment's real span is known
const BUDGET = 1.5e9;  // resident column bytes
const IN_USE = 5000;  // ms since last use that a column counts as in use (never evicted)
const BLOCK = 16, FANOUT = 8;  // min/max pyramid: blocks of 16 samples, then 8 blocks per level up
const NEAR = 3;  // segments from the focus on that, once in, let the worker pool shrink
const MONO = "__logMonoTime";  // the message times as a series, as jotpluggler and PlotJuggler name it

// opens a route and starts loading its logs outward from t; repeat tiles the route for long-route tests
export async function openRoute(name, opts) {
  const files = await api.routeFiles(name);
  const urls = [];
  for (const u of [...files.qlogs, ...files.logs]) urls[+new URL(u).pathname.split("/").at(-2)] = u;  // rlogs win
  return new Store(name, urls, opts);
}

export class Store {
  // urls[k]: the log shown in slot k, a URL or {url, file, source, seg} (file names the cache entry when the URL doesn't;
  // seg is the route's segment when only a range of it loads); keep: {service: [list path globs]} to split long lists from the start
  // overlays pass the main store's loader (shared, not disposed here), their own status lane, can: false and migrate: false
  constructor(name, urls, {t = 0, repeat = 1, keep = {}, loader, budget = BUDGET, lane = {name: "logs", order: 0}, can = true, migrate = true} = {}) {
    const count = urls.length * repeat;
    this.route = {name, t0: null, duration: 0, segments: Array.from({length: count}, (_, n) => ({
      n, t0: n * SEGMENT, t1: (n + 1) * SEGMENT, status: urls[n % urls.length] ? "queued" : "missing",
    }))};
    this.loader = loader || new Loader();
    this.shared = !!loader;
    this.withCan = can;
    this.migrate = migrate;
    this.overlays = [];  // [{prefix, name, store}]: other logs on this clock (replays), their paths under prefix
    this.budget = budget;
    this.keep = keep;
    this.handlers = {};
    this.can = new Can(this);
    this.entries = new Map();  // path → {path, type, names?, at: [{pack, k} by segment], views: [], version, used}
    this.monos = new Map();  // /<service>/__logMonoTime → its series
    this.lists = new Map();  // long lists the workers skipped: path → longest length
    this.missing = new Set();  // migrated paths that couldn't be computed
    this.evicted = new Map();  // service → evicted paths in it, which loads from now on skip
    this.enums = [];
    this.bytes = 0;
    this.progress = {done: 0, failed: 0, total: 0, bytes: 0, raw: 0, cached: 0, ms: {fetch: 0, decompress: 0, read: 0, ingest: 0}, ingestMax: 0, start: performance.now()};
    this.t = t;
    this.focusSeg = Math.floor(t / SEGMENT);
    this.received = [];  // [time, bytes] of recent downloads, for the rate
    // OPFS keys name the route, segment and file; the signed URLs change
    this.cacheDir = `logs/${name.replace("|", "/")}/`;
    this.disk = {};  // key → bytes of this route's cached logs
    opfs.list(this.cacheDir).then((d) => {
      Object.assign(this.disk, d);
      for (const seg of this.route.segments) this.publish(seg);
    }, () => {});
    this.status = status.source(lane.name, lane);
    this.status.count(count);

    for (const seg of this.route.segments) {
      const local = seg.n % urls.length, e = typeof urls[local] === "string" ? {url: urls[local]} : urls[local];
      seg.bytes = 0;  // resident
      if (e) this.start(seg, e, local);
      else this.publish(seg);
    }
  }

  // starts loading a segment's log; e = {url, file?, source?, seg?}
  start(seg, e, local = seg.n) {
    seg.real = e.seg ?? local;  // the route's segment, for a range of it (urls[k] then names its seg)
    seg.key = `${this.cacheDir}${seg.real}/${e.file ?? new URL(e.url, location.origin).pathname.split("/").at(-1)}`;
    seg.url = e.url;
    seg.source = e.source;
    seg.opts = {n: seg.real, shift: (seg.n - seg.real) * SEGMENT, ...(!this.migrate && {migrate: false})};
    this.publish(seg);
    this.progress.total++;
    this.load(seg, undefined, {can: this.withCan}).then((p) => {  // CAN frames come out of the first parse only
      if (p.cached || p.stored) this.disk[seg.key] = p.bytes;
      this.count(p);
      this.ingest(seg, p);
      if (p.can) this.can.ingest(seg.n, p.can, 0);
    }, (e) => this.failed(seg, e));
  }

  // the options are taken when the parse starts, so it skips whatever was evicted by then
  load(seg, only, extra = {}) {
    const opts = () => {
      const skip = {};
      for (const [svc, subs] of this.evicted) if (subs.size) skip[svc] = [...subs];
      return {...seg.opts, keep: this.keep, skip, only, context: this.context, ...extra};
    };
    return this.loader.load(seg.key, seg.url, opts, this.prio(seg.n), seg.n, (ev) => this.report(seg, ev), this);
  }

  // outward from the focus segment
  prio(n) { return Math.abs(n - this.focusSeg); }

  // the playhead: loading ranks outward from it, and it buffers while its segment isn't in. Cheap to call every frame
  focus(t) {
    this.t = t;
    for (const o of this.overlays) o.store.focus(t);
    const n = Math.max(0, Math.floor(t / SEGMENT));
    if (n !== this.focusSeg) {
      this.focusSeg = n;
      this.loader.reprioritize((m) => this.prio(m), this);
      this.pool();
    }
    this.buffering();
  }

  // the whole pool while the focus area loads, half of it after
  pool() {
    if (this.shared) return;  // the owner sizes it
    const near = this.route.segments.slice(this.focusSeg, this.focusSeg + NEAR);
    this.loader.resize(near.every((s) => s.status !== "queued") ? Math.max(2, POOL >> 1) : POOL);
  }

  // a load's progress: {loaded, total} downloading, {retry, of, error} before a retry, {paused}, {parsing}
  report(seg, ev) {
    if (ev.retry) Object.assign(seg, {retry: `${ev.retry}/${ev.of}`, error: ev.error});
    else seg.retry = 0;
    if (ev.loaded !== undefined) {
      this.received.push([performance.now(), ev.loaded - (seg.loading?.loaded || 0)]);
      seg.loading = {loaded: ev.loaded, total: ev.total};
    }
    if (ev.paused && seg.loading) seg.loading.paused = true;
    if (ev.parsing) seg.loading = {loaded: 1, total: 1};
    this.publish(seg);
  }

  // a segment's status: on disk when cached but not (or no longer) in memory
  publish(seg) {
    const cached = seg.key in this.disk;
    const state = seg.status === "missing" || seg.status === "error" ? seg.status
      : seg.loading && !seg.loading.paused ? "loading"
      : seg.status === "done" && seg.bytes ? "ready"
      : cached ? "disk" : seg.status === "done" ? "idle" : "queued";
    this.status.seg(seg.n, state, {...seg.loading, source: seg.source, text: seg.retry ? `retrying (${seg.retry})` : null});
    if (seg.n === this.focusSeg) this.buffering();
    this.totals();
  }

  totals() {
    const now = performance.now();
    while (this.received.length && now - this.received[0][0] > 2000) this.received.shift();
    this.status.rate(this.received.reduce((a, [, b]) => a + b, 0) / 2);
    this.status.bytes({memory: this.bytes + this.can.bytes, disk: Object.values(this.disk).reduce((a, b) => a + b, 0)});
  }

  // the segment holding route time t
  segmentAt(t) {
    const segs = this.route.segments;
    let n = Math.min(segs.length - 1, Math.max(0, Math.floor(t / SEGMENT)));
    while (n > 0 && segs[n].t0 > t) n--;
    while (n < segs.length - 1 && segs[n].t1 <= t) n++;
    return segs[n];
  }

  buffering() {
    const seg = this.segmentAt(this.t), n = seg?.n;
    const waiting = seg && seg.status === "queued";
    this.status.buffering(waiting ? {n, text: `segment ${seg.real ?? n}${seg.retry ? ` · retrying (${seg.retry})` : ""}`} : null);
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
    if (this.disposed) return;
    seg.status = "error";
    seg.error = String(e);
    seg.loading = null;
    this.publish(seg);
    this.progress.failed++;
    this.emit("progress", this.progress);
  }

  // registers a pack's columns: O(columns), no views and no copies
  ingest(seg, p) {
    const start = performance.now(), route = this.route;
    route.t0 ??= p.start;
    this.context ||= p.context;  // migrations fall back to it when a segment lacks initData or carParams
    for (const m of p.missing || []) this.missing.add(m);
    seg.start = p.start;  // logMonoTime of the route start, as this segment saw it
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
    p.n = seg.n;
    p.live.forEach((n, s) => {
      if (!n) return drop(p, s);
      this.bytes += p.bufs[s].byteLength;
      seg.bytes += p.bufs[s].byteLength;
    });
    seg.loading = null;
    this.publish(seg);
    this.pool();
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
    for (const {path, type, names} of this.entries.values()) {
      out.push(names ? {path, type, names} : {path, type});
      const mono = path.match(/^(\/[^/]+)\/_?valid$/);  // every service has one
      if (mono) out.push({path: `${mono[1]}/${MONO}`, type: "f64", internal: true});
    }
    for (const o of this.overlays) for (const p of o.store.paths()) out.push({...p, path: o.prefix + p.path});
    return out;
  }

  // the chunks of a path; an unknown index of a skipped long list (/modelV2/position/x/32) starts loading it
  series(path) {
    if (path.startsWith("/dbc/")) return this.can.series(path);
    const o = this.overlays.length ? this.overlayOf(path) : null;
    if (o) return this.overlaySeries(o, path);
    if (path.endsWith(`/${MONO}`)) return this.monoSeries(path);
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

  // /<service>/__logMonoTime: each message's logMonoTime in ns, from the service's times (they sit next to its valid column)
  monoSeries(path) {
    const svc = path.slice(0, -MONO.length - 1), src = this.entries.has(`${svc}/valid`) ? `${svc}/valid` : `${svc}/_valid`;
    if (!this.series(src)) return null;  // keeps the times resident
    const e = this.entries.get(src);
    let m = this.monos.get(path);
    if (m?.version !== e.version) {
      const chunks = [];
      e.at.forEach((a, n) => {
        if (!a) return;
        const seg = this.route.segments[n], t = times(a.pack, service(a.pack, a.k)), off = seg.start - seg.opts.shift;
        chunks.push({n, t, tt: t, v: Float64Array.from(t, (x) => (x + off) * 1e9)});
      });
      this.monos.set(path, m = {path, type: "f64", chunks, version: e.version});
    }
    return m;
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
    const chunks = overlapping(s.chunks, t0, t1);
    let total = 0;
    for (const c of chunks) total += Math.max(0, upperBound(c.tt, t1) - lowerBound(c.tt, t0));
    if (total <= n) {
      const t = [], v = [];
      for (const c of chunks) {
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
    for (const c of chunks) {
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

  // the whole decoded message of a service at or before t, from a worker that keeps the last logs decompressed
  async message(service, t) {
    const seg = this.segmentAt(t);
    if (seg?.start === undefined) return null;
    const mono = (t - seg.opts.shift + seg.start) * 1e9 + 1000;  // 1 µs over, for the round trip through route seconds
    const ev = await this.loader.message(seg.key, seg.url, service, mono);
    return ev?.[service] ?? null;
  }

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
          if (seg.status !== "done") continue;
          this.load(seg, only).then((p) => this.ingest(seg, p), () => {
            seg.loading = null;
            this.publish(seg);
          });
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
      const seg = this.route.segments[pack.n];
      this.bytes -= pack.bufs[s].byteLength;
      seg.bytes -= pack.bufs[s].byteLength;
      drop(pack, s);
      if (!seg.bytes) this.publish(seg);
    }
  }

  // another log on the same clock, such as a replay of this route (its segments carry this route's initData, so
  // times line up), with its paths under prefix. Attaching the same prefix again loads segments added to urls.
  attach(prefix, urls, {name = prefix} = {}) {
    let o = this.overlays.find((x) => x.prefix === prefix);
    if (o) return o.store.add(urls), o;
    const store = new Store(name, urls, {loader: this.loader, t: this.t, lane: {name, label: name, order: 1}, can: false, migrate: false});
    o = {prefix, name, store, cache: new Map(), offs: ["paths", "progress"].map((ev) => store.on(ev, (a) => this.emit(ev, a)))};
    this.overlays.push(o);
    return o;
  }

  detach(prefix) {
    const o = this.overlays.find((x) => x.prefix === prefix);
    if (!o) return;
    this.overlays.splice(this.overlays.indexOf(o), 1);
    for (const off of o.offs) off();
    o.store.dispose();
    this.emit("paths");
  }

  // loads the slots of urls that weren't there before
  add(urls) {
    const segs = this.route.segments, slot = (k) => ({n: k, t0: k * SEGMENT, t1: (k + 1) * SEGMENT, status: "missing", bytes: 0});
    urls.forEach((u, k) => {
      if (!u || segs[k]?.url) return;
      const seg = segs[k] ||= slot(k);
      seg.status = "queued";
      this.start(seg, typeof u === "string" ? {url: u} : u);
    });
    for (let k = 0; k < segs.length; k++) segs[k] ||= slot(k);
    this.status.count(segs.length);
  }

  overlayOf(path) {
    return this.overlays.find((o) => path.startsWith(o.prefix) && path[o.prefix.length] === "/");
  }

  // the overlay's series under the prefixed path, the same object while it's unchanged
  overlaySeries(o, path) {
    const s = o.store.series(path.slice(o.prefix.length));
    if (!s) return null;
    let c = o.cache.get(path);
    if (c?.version !== s.version || c.chunks !== s.chunks) o.cache.set(path, c = {...s, path});
    return c;
  }

  dispose() {
    for (const o of this.overlays) o.store.dispose();
    this.disposed = true;
    if (this.shared) this.loader.cancel(this);
    else this.loader.dispose();
    this.can.dispose();
    this.status.remove();
  }
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

// the chunks overlapping [t0, t1], found by binary search: chunks follow each other in time
function overlapping(chunks, t0, t1) {
  let lo = 0, hi = chunks.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (chunks[mid].tt.at(-1) < t0) lo = mid + 1; else hi = mid;
  }
  let end = lo;
  while (end < chunks.length && !(chunks[end].tt[0] > t1)) end++;
  return chunks.slice(lo, end);
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
