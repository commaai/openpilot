// the route's series store: per-segment columns from the workers behind one path namespace and one clock
import {api} from "../api.js";
import {Loader} from "./loader.js";

const SEGMENT = 60;  // nominal seconds per segment, until a segment's real span is known
const NUMERIC = {f32: "f32", f64: "f64", i8: "f32", u8: "f32", i16: "f32", u16: "f32", i32: "f64", u32: "f64", i64: "f64", u64: "f64"};

// opens a route and starts loading its logs outward from t; repeat tiles the route for long-route tests
export async function openRoute(name, {t = 0, repeat = 1, loader} = {}) {
  const files = await api.routeFiles(name);
  const urls = [];
  for (const u of [...files.qlogs, ...files.logs]) {  // rlogs win over qlogs
    const n = +new URL(u).pathname.split("/").at(-2);
    urls[n] = u;
  }
  return new Store(name, urls, {t, repeat, loader});
}

export class Store {
  constructor(name, urls, {t = 0, repeat = 1, loader} = {}) {
    const count = urls.length * repeat;
    this.route = {name, t0: null, duration: count * SEGMENT, segments: Array.from({length: count}, (_, n) => ({
      n, t0: n * SEGMENT, t1: (n + 1) * SEGMENT, status: urls[n % urls.length] ? "queued" : "missing",
    }))};
    this.loader = loader || new Loader();
    this.handlers = {};
    this.meta = new Map();  // path → {path, type, names}
    this.cols = new Map();  // path → [{n, t, v, i?}] in segment order
    this.cache = new Map();
    this.progress = {done: 0, failed: 0, total: 0, bytes: 0, raw: 0, ms: {fetch: 0, decompress: 0, read: 0}, start: performance.now()};
    this.focusSeg = Math.floor(t / SEGMENT);

    for (const seg of this.route.segments) {
      const url = urls[seg.n % urls.length];
      if (!url) continue;
      seg.shift = Math.floor(seg.n / urls.length) * urls.length * SEGMENT;
      this.progress.total++;
      this.loader.load(url, this.prio(seg.n), seg.n).then((r) => this.loaded(seg, r), (e) => this.failed(seg, e));
    }
  }

  // outward from the focus segment
  prio(n) { return Math.abs(n - this.focusSeg); }

  focus(t) {
    this.focusSeg = Math.floor(t / SEGMENT);
    this.loader.reprioritize((n) => this.prio(n));
  }

  on(event, fn) {
    (this.handlers[event] ||= new Set()).add(fn);
    return () => this.handlers[event].delete(fn);
  }

  emit(event, arg) {
    for (const fn of this.handlers[event] || []) fn(arg);
  }

  loaded(seg, r) {
    const p = this.progress;
    p.done++;
    p.bytes += r.bytes;
    p.raw += r.raw;
    for (const k in r.ms) p.ms[k] += r.ms[k];
    this.ingest(seg, r.services);
    this.emit("progress", p);
  }

  failed(seg, e) {
    seg.status = "error";
    seg.error = String(e);
    this.progress.failed++;
    this.emit("progress", this.progress);
  }

  ingest(seg, services) {
    const {initData, ...rest} = services;
    let lo = Infinity, hi = -Infinity;
    for (const s of Object.values(rest)) if (s.t.length) { lo = Math.min(lo, s.t[0]); hi = Math.max(hi, s.t.at(-1)); }
    // every segment repeats the route's initData, whose time is the route start
    const route = this.route;
    route.t0 ??= initData ? initData.t[0] : lo - (seg.n * SEGMENT - seg.shift);
    const off = seg.shift - route.t0;
    seg.t0 = lo + off;
    seg.t1 = hi + off;
    seg.status = "done";
    route.duration = seg.n === route.segments.length - 1 ? seg.t1 : Math.max(route.duration, seg.t1);

    let newPaths = false;
    for (const [svc, s] of Object.entries(services)) {
      const t = s.t, n = t.length;
      for (let k = 0; k < n; k++) t[k] += off;
      for (const [sub, c] of Object.entries(s.cols)) {
        const path = `/${svc}${sub}`;
        if (!this.meta.has(path)) {
          this.meta.set(path, {path, type: NUMERIC[c.type] || c.type, ...(c.names && {names: c.names})});
          this.cols.set(path, []);
          newPaths = true;
        }
        const chunk = "c" in c ? {n: seg.n, t, v: [c.c, c.c], i: new Uint32Array([0, n - 1])} : {n: seg.n, t, v: c.v};
        const list = this.cols.get(path);
        let at = list.length;
        while (at > 0 && list[at - 1].n > seg.n) at--;
        list.splice(at, 0, chunk);
        this.cache.delete(path);
      }
    }
    if (newPaths) this.emit("paths");
    this.emit("segment", seg.n);
  }

  paths() { return [...this.meta.values()]; }

  series(path) {
    const meta = this.meta.get(path);
    if (!meta) return null;
    let s = this.cache.get(path);
    if (!s) {
      s = {...meta, chunks: this.cols.get(path).map(({t, v, i}) => (i ? {t, v, i} : {t, v})), version: (this.version = (this.version || 0) + 1)};
      this.cache.set(path, s);
    }
    return s;
  }

  // n min/max buckets over [t0, t1]; the exact samples when there are no more than n of them
  decimate(path, t0, t1, n) {
    const s = this.series(path);
    if (!s || s.type === "t") return null;
    const ts = [], vs = [];
    for (const c of s.chunks) {
      if (!c.t.length || c.t[c.i ? c.i.at(-1) : c.t.length - 1] < t0 || c.t[c.i ? c.i[0] : 0] > t1) continue;
      const len = c.i ? c.i.length : c.t.length;
      const at = (k) => c.t[c.i ? c.i[k] : k];
      for (let k = lowerBound(at, len, t0); k < len && at(k) <= t1; k++) {
        const x = c.v[k];
        if (x === x && x !== null) { ts.push(at(k)); vs.push(x); }
      }
    }
    if (ts.length <= n) return {t: Float64Array.from(ts), min: Float64Array.from(vs), max: Float64Array.from(vs)};
    const t = new Float64Array(n), min = new Float64Array(n).fill(Infinity), max = new Float64Array(n).fill(-Infinity);
    const w = (t1 - t0) / n;
    for (let b = 0; b < n; b++) t[b] = t0 + (b + 0.5) * w;
    for (let k = 0; k < ts.length; k++) {
      const b = Math.min(n - 1, Math.floor((ts[k] - t0) / w)), x = vs[k];
      if (x < min[b]) min[b] = x;
      if (x > max[b]) max[b] = x;
    }
    for (let b = 0; b < n; b++) if (min[b] === Infinity) min[b] = max[b] = NaN;
    return {t, min, max};
  }

  // the last value at or before t
  valueAt(path, t) {
    const s = this.series(path);
    if (!s) return undefined;
    for (let j = s.chunks.length - 1; j >= 0; j--) {
      const c = s.chunks[j];
      const len = c.i ? c.i.length : c.t.length;
      const at = (k) => c.t[c.i ? c.i[k] : k];
      if (!len || at(0) > t) continue;
      for (let k = upperBound(at, len, t) - 1; k >= 0; k--) {
        const x = c.v[k];
        if (x === x && x !== null) return x;
      }
    }
    return undefined;
  }

  async message(service, t) { return null; }

  dispose() { this.loader.dispose(); }
}

// first k with at(k) >= x
function lowerBound(at, len, x) {
  let lo = 0, hi = len;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (at(mid) < x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// first k with at(k) > x
function upperBound(at, len, x) {
  let lo = 0, hi = len;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (at(mid) <= x) lo = mid + 1; else hi = mid;
  }
  return lo;
}
