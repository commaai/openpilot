// the route's series store: per-segment columns from the workers behind one path namespace and one clock.
// Columns stay packed as the workers sent them and become views on first use. Over the memory budget, the least
// recently used columns are dropped and re-parsed from the OPFS cache when asked for again.
import {SEGMENT} from "../api.js";
import {perf} from "../perf.js";
import {status} from "../status.js";
import {lowerBound, upperBound} from "./bounds.js";
import {Loader, POOL} from "./loader.js";
import * as opfs from "./opfs.js";
import {Can} from "./can.js";
import {BLOCK, constant, enumRef, service, times, unpack} from "./pack.js";

const BUDGET = 1.5e9;  // resident column bytes
const IN_USE = 5000;  // ms since last use that a column counts as in use (never evicted)
// what a store without CAN (an overlay) has in place of its Can
const NO_CAN = {bytes: 0, paths: () => [], series: () => null, ingest() {}, dispose() {}};
const FANOUT = 8;  // min/max pyramid: blocks of BLOCK samples, then 8 blocks per level up
const NEAR = 3;  // segments from the focus on that, once in, let the worker pool shrink
const MONO = "__logMonoTime";  // the message times as a series, as jotpluggler and PlotJuggler name it

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
    this.kept = [];  // the lists asked for since, in order ({service: [globs]}); a parse notes how many it read with
    this.handlers = {};
    this.can = can ? new Can(this) : NO_CAN;  // an overlay's Can would share and resize the route's CAN lane
    this.entries = new Map();  // path → {path, type, names?, at: [{pack, k} by segment], views: [], version, used}
    this.monos = new Map();  // /<service>/__logMonoTime → its series
    this.lists = new Map();  // long lists the workers skipped: path → longest length
    this.listTypes = new Map();  // and → element type ("s": structs)
    this.missing = new Set();  // migrated paths that couldn't be computed
    this.evicted = new Map();  // service → evicted paths in it, which loads from now on skip
    this.enums = [];
    this.listed = null;  // what paths() has listed (see there)
    this.schema = null;  // the log's own schema ({key, text}, see schemas.js) or null for the bundled one
    this.gen = 0;  // bumped with every schema change; packs carry the generation they were parsed with
    this.unknown = {events: 0, services: new Set()};  // what the reads so far didn't know (capnp.js)
    this.bytes = 0;
    this.puts = 0;  // put() versions: a path's never repeat, even once dropped
    this.progress = {done: 0, failed: 0, total: 0, ingestMax: 0};
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
    this.status.retry(() => this.retry());
    this.online = () => this.retry();
    addEventListener("online", this.online);

    for (const seg of this.route.segments) {
      const local = seg.n % urls.length, e = typeof urls[local] === "string" ? {url: urls[local]} : urls[local];
      seg.bytes = 0;  // resident
      if (e) this.start(seg, e, local);
      else this.publish(seg);
    }
  }

  // starts loading a segment's log; e = {url, file?, source?, seg?}
  start(seg, e, local = seg.n) {
    seg.entry = e;
    seg.real = e.seg ?? local;  // the route's segment, for a range of it (urls[k] then names its seg)
    seg.key = `${this.cacheDir}${seg.real}/${e.file ?? new URL(e.url, location.origin).pathname.split("/").at(-1)}`;
    seg.url = e.url;
    seg.source = e.source;
    seg.opts = {n: seg.real, shift: (seg.n - seg.real) * SEGMENT, ...(!this.migrate && {migrate: false})};
    this.publish(seg);
    this.progress.total++;
    this.load(seg, undefined, {can: this.withCan}).then((p) => {  // CAN frames come out of the first parse only
      if (p.cached || p.stored) this.disk[seg.key] = p.bytes;
      this.progress.done++;
      this.ingest(seg, p);
      if (p.can) this.can.ingest(seg.n, p.can, 0);
    }, (e) => this.failed(seg, e));
  }

  // the options are taken when the parse starts, so it skips whatever was evicted by then
  load(seg, only, extra = {}) {
    let kept;
    const opts = () => {
      kept = this.kept.length;
      const skip = {};
      for (const [svc, subs] of this.evicted) if (subs.size) skip[svc] = [...subs];
      return {...seg.opts, keep: this.keep, skip, only, context: this.context, schema: this.schema, gen: this.gen, ...extra};
    };
    const loaded = this.loader.load(seg.key, seg.url, opts, this.prio(seg.n), seg.n, (ev) => this.report(seg, ev), this);
    return loaded.then((p) => Object.assign(p, {full: !only, kept}));  // every column, not just the ones asked for again
  }

  // reads every loaded segment again with another schema; the columns in memory stay until each segment's new ones
  // replace them. Evicted columns are forgotten: their paths may not exist in the new schema
  setSchema(schema) {
    if (schema?.key === this.schema?.key) return;
    this.schema = schema;
    this.gen++;
    this.enums = [];
    this.listed = null;
    for (const e of this.entries.values()) if (e.evicted) this.entries.delete(e.path);
    this.evicted.clear();
    this.emit("schema", schema);
    for (const seg of this.route.segments) if (seg.status === "done") this.reread(seg);
  }

  // one re-read per segment at a time: a queued one parses with whatever schema is current when it starts, and one
  // that was already parsing with a replaced schema is dropped on arrival and read again (see ingest)
  reread(seg) {
    if (seg.rereading) return;
    seg.rereading = true;
    this.load(seg).then((p) => {
      seg.rereading = false;
      this.ingest(seg, p);
    }, () => {
      seg.rereading = false;
      seg.loading = null;
      this.publish(seg);
    });
  }

  // after a segment's first full parse with the schema in use, its columns of paths that parse didn't have go
  prune(seg, paths) {
    let removed = false;
    for (const e of this.entries.values()) {
      const a = e.at[seg.n];
      if (!a || paths.has(e.path)) continue;
      this.release(a);
      delete e.at[seg.n];
      e.views[seg.n] = null;
      e.version++;
      if (!e.at.some(Boolean)) removed = this.entries.delete(e.path);
    }
    if (removed) {
      this.listed = null;
      this.emit("paths");
    }
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
      if (this.route.segments[n]?.status === "error") this.retry([this.route.segments[n]]);
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
      // the rate falls as the downloads stop: totals again every second until the last two seconds received nothing
      this.ticker ||= setInterval(() => {
        this.totals();
        if (this.received.length) return;
        clearInterval(this.ticker);
        this.ticker = 0;
      }, 1000);
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
    const text = state === "error" ? seg.error : seg.retry ? `retrying (${seg.retry})` : null;
    this.status.seg(seg.n, state, {...seg.loading, source: seg.source, text, label: seg.real ?? seg.n});
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

  // segments are still to load
  get loading() {
    const p = this.progress;
    return p.done + p.failed < p.total;
  }

  // why no segment loaded, once every one has failed; null while any is loaded or loading
  get failure() {
    const segs = this.route.segments.filter((s) => s.status !== "missing");
    return segs.length && segs.every((s) => s.status === "error") ? segs[0].error : null;
  }

  failed(seg, e) {
    if (this.disposed) return;
    seg.status = "error";
    seg.error = String(e?.message ?? e).replace(/^decompress failed \(-\d+\)$/, "truncated or corrupt log");  // zstd's error codes
    seg.retry = 0;
    seg.loading = null;
    this.publish(seg);
    this.progress.failed++;
    this.emit("progress", this.progress);
  }

  // loads failed segments again: back online, at the playhead, or from the status panel's Retry
  retry(segs = this.route.segments) {
    for (const seg of segs) {
      if (seg.status !== "error" || this.disposed) continue;
      seg.status = "queued";
      seg.error = null;
      this.progress.failed--;
      this.progress.total--;
      this.start(seg, seg.entry, seg.real);
    }
  }

  // registers a pack's columns: O(columns), no views and no copies
  ingest(seg, p) {
    if (p.gen !== this.gen && seg.status === "done") {  // superseded: what's in memory stays until the new schema's read
      seg.loading = null;
      return this.reread(seg);
    }
    const start = performance.now(), route = this.route;
    route.t0 ??= p.start;
    this.context ||= p.context;  // migrations fall back to it when a segment lacks initData or carParams
    for (const m of p.missing || []) this.missing.add(m);
    this.unknown.events += p.unknown?.events || 0;
    for (const s of p.unknown?.services || []) this.unknown.services.add(s);
    const fresh = p.full && p.gen === this.gen && seg.gen !== this.gen;  // the first full read with this schema
    if (fresh) seg.gen = this.gen;
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
    let added = false;
    for (const path in p.lists) {
      if (p.lists[path] <= (this.lists.get(path) || 0)) continue;
      this.lists.set(path, p.lists[path]);
      this.listTypes.set(path, p.listTypes[path]);
      added = true;
    }

    const paths = p.paths ? p.paths.split("\n") : [];
    p.types = p.types.split("\n");
    p.live = p.counts.map(() => 0);
    for (let k = 0; k < paths.length; k++) {
      let e = this.entries.get(paths[k]);
      const ref = enumRef(p, k), names = ref >= 0 ? this.enums[ref] : undefined;
      if (!e) {
        e = {path: paths[k], type: p.types[k], ...(names && {names}), at: [], views: [], version: 0, used: 0};
        this.entries.set(paths[k], e);
        added = true;
        if (this.listOf(e.path)) this.listed = null;  // listed as an index of a long list until now
      } else if (fresh && (e.type !== p.types[k] || String(e.names) !== String(names))) {
        e.type = p.types[k];  // another schema can type or name a path differently
        e.names = names;
        this.listed = null;
      }
      if (e.evicted) continue;  // not resident until asked for again
      if (e.at[seg.n]) this.release(e.at[seg.n]);
      e.at[seg.n] = {pack: p, k};
      e.views[seg.n] = null;
      e.version++;
      p.live[service(p, k)]++;
    }
    p.n = seg.n;
    if (fresh) this.prune(seg, new Set(paths));
    if (p.gen !== this.gen) this.reread(seg);  // read with a schema that's since been replaced
    else if (p.full && p.kept < this.kept.length) this.readOnly(seg, this.kept.slice(p.kept).reduce(merge, {}));  // lists asked for mid-parse
    p.live.forEach((n, s) => {
      if (!n) return drop(p, s);
      this.bytes += p.bufs[s].byteLength;
      seg.bytes += p.bufs[s].byteLength;
    });
    seg.loading = null;
    this.publish(seg);
    this.pool();
    const ms = performance.now() - start;
    this.progress.ingestMax = Math.max(this.progress.ingestMax, ms);
    if (added) this.emit("paths");
    this.emit("segment", seg.n);
    this.emit("progress", this.progress);
    this.account();
    perf.mark("ingest", `${ms.toFixed(1)} ms (max ${this.progress.ingestMax.toFixed(1)})`);
  }

  // every path as {path, type, names?, precision?, unit?, states?, internal?, lazy?}: the same array from call to
  // call while paths are only added, with the new ones appended; a new one once any goes or changes (the browser
  // keeps its place in it)
  paths() {
    const can = this.can.paths();  // a new array when the DBCs change, or a live session's messages go
    if (this.listed?.signals !== can) this.listed = null;
    const l = this.listed ||= {out: [], signals: can, can: 0, entries: 0, lists: new Map(), overlays: new Map()};
    const {out} = l;
    for (; l.can < can.length; l.can++) out.push(can[l.can]);
    let k = 0;
    for (const {path, type, names} of this.entries.values()) {
      if (k++ < l.entries) continue;  // entries keep their order
      out.push(names ? {path, type, names} : {path, type});
      const mono = path.match(/^(\/[^/]+)\/valid$/);  // every service has one (and _valid only next to a field named valid)
      if (mono) out.push({path: `${mono[1]}/${MONO}`, type: "f64", internal: true});
    }
    l.entries = k;
    // the indexes of skipped long lists of values, loaded once plotted
    for (const [list, n] of this.lists) {
      const type = this.listTypes.get(list);
      if (!type || type === "s" || type === "l") continue;
      for (let i = l.lists.get(list) || 0; i < n; i++) if (!this.entries.has(`${list}/${i}`)) out.push({path: `${list}/${i}`, type, lazy: true});
      l.lists.set(list, n);
    }
    for (const o of this.overlays) {
      const theirs = o.store.paths(), [seen, from] = l.overlays.get(o.prefix) || [theirs, 0];
      if (seen !== theirs) {  // its paths went or changed
        this.listed = null;
        return this.paths();
      }
      for (let i = from; i < theirs.length; i++) out.push({...theirs[i], path: o.prefix + theirs[i].path});
      l.overlays.set(o.prefix, [theirs, theirs.length]);
    }
    return out;
  }

  // the chunks of a path; an unknown index of a skipped long list (/modelV2/position/x/32) starts loading it
  series(path) {
    if (path.startsWith("/dbc/") && !this.entries.has(path)) return this.can.series(path);  // transforms of signals are put here
    const o = this.overlays.length ? this.overlayOf(path) : null;
    if (o) return this.overlaySeries(o, path);
    if (path.endsWith(`/${MONO}`)) return this.monoSeries(path);
    const e = this.entries.get(path);
    if (!e) return this.request(path), null;
    e.used = performance.now();
    if (e.evicted) this.reload(e);
    this.partial(e);
    if (e.series?.version !== e.version) {
      const chunks = [];
      e.at.forEach((a, n) => a && chunks.push(e.views[n] ||= unpack(a.pack, a.k, n)));
      e.series = {path, type: e.type, names: e.names, chunks, version: e.version};
    }
    return e.series;
  }

  // a path's {type, names?, precision?, unit?, states?} (the last three for DBC signals) without unpacking or decoding
  // it; null when unknown
  meta(path) {
    const o = this.overlays.length ? this.overlayOf(path) : null;
    if (o) return o.store.meta(path.slice(o.prefix.length));
    if (path.startsWith("/dbc/") && !this.entries.has(path)) {
      const p = this.can.parse(path);
      return p && this.can.meta(path, p.sig);
    }
    return this.entries.get(path) || this.series(path);
  }

  // the value of a log column that's the same constant in every loaded segment, else undefined
  constant(path) {
    let c;
    for (const a of this.entries.get(path)?.at || []) {
      if (!a) continue;
      const x = constant(a.pack, a.k);
      if (x === undefined || (c !== undefined && x !== c)) return undefined;
      c = x;
    }
    return c;
  }

  // one segment's chunk of a path, unpacking only that segment's (series() unpacks them all, and text decodes)
  chunk(path, n) {
    const e = this.entries.get(path);
    if (!e?.at.length) return this.series(path)?.chunks.find((c) => c.n === n);
    e.used = performance.now();
    this.partial(e);
    const a = e.at[n];
    return a ? e.views[n] ||= unpack(a.pack, a.k, n) : undefined;
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

  // registers or replaces a derived series (transforms, /custom/*) as one chunk; "put" tells of every new version.
  // Its bytes count against the budget: evicted, it's gone until computed again
  put(path, {type, t, v, names}) {
    const old = this.entries.get(path);
    const e = {path, type, ...(names && {names}), at: [], views: [{n: 0, t, v, tt: t}], version: ++this.puts, used: performance.now(),
      put: t.byteLength + (v.byteLength ?? 0)};
    e.series = {path, type, names, chunks: e.views, version: e.version};
    this.entries.set(path, e);
    this.bytes += e.put - (old?.put || 0);
    if (!old) this.emit("paths");
    this.emit("put", path);
    this.account();
  }

  // forgets a derived series nothing shows any more
  drop(path) {
    const e = this.entries.get(path);
    if (!e?.put) return;
    this.entries.delete(path);
    this.bytes -= e.put;
    this.listed = null;
    this.emit("paths");
    this.account();
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
          const x = "c" in c ? c.c : c.v[k];
          if (x === x) { t.push(c.tt[k]); v.push(x); }
        }
      }
      const vals = Float64Array.from(v);
      return {t: Float64Array.from(t), min: vals, max: vals};
    }
    const w = (t1 - t0) / n;
    const t = new Float64Array(n), min = new Float64Array(n).fill(Infinity), max = new Float64Array(n).fill(-Infinity);
    for (let b = 0; b < n; b++) t[b] = t0 + (b + 0.5) * w;
    for (const c of chunks) {
      const {tt} = c, k0 = lowerBound(tt, t0), k1 = upperBound(tt, t1);
      if (k1 <= k0) continue;
      if ("c" in c) {  // a constant: the value in each bucket with a sample, found by stepping (or bisecting) to bucket ends
        for (let k = k0; k < k1;) {
          const b = Math.min(n - 1, Math.floor((tt[k] - t0) / w)), end = b === n - 1 ? Infinity : t0 + (b + 1) * w;
          if (c.c < min[b]) min[b] = c.c;
          if (c.c > max[b]) max[b] = c.c;
          let j = k + 1;
          while (j < k1 && tt[j] < end && j < k + 8) j++;
          k = j < k1 && tt[j] < end ? lowerBound(tt, end, j, k1) : j;
        }
        continue;
      }
      // the coarsest level whose blocks still fit two to a bucket
      const {v} = c, perBucket = (k1 - k0) / Math.max(1, (tt[k1 - 1] - tt[k0]) / w);
      const levels = perBucket >= 2 * BLOCK ? pyramid(c) : [];
      let level = -1, size = 1;
      while (level + 1 < levels.length && (level < 0 ? BLOCK : size * FANOUT) * 2 <= perBucket) {
        size = level < 0 ? BLOCK : size * FANOUT;
        level++;
      }
      const b0 = Math.ceil(k0 / size), b1 = Math.floor(k1 / size);
      if (level < 0 || b1 <= b0) {
        bin(tt, v, v, k0, k1, 1, t0, w, min, max);
        continue;
      }
      const {lo, hi} = levels[level];
      bin(tt, v, v, k0, b0 * size, 1, t0, w, min, max);
      bin(tt, lo, hi, b0, b1, size, t0, w, min, max);
      bin(tt, v, v, b1 * size, k1, 1, t0, w, min, max);
    }
    for (let b = 0; b < n; b++) if (min[b] > max[b]) min[b] = max[b] = NaN;
    return {t, min, max};
  }

  // the last value at or before t; an index of a skipped long list isn't loaded for it. A log's column unpacks
  // only the chunks from t's segment back to the value
  valueAt(path, t) {
    const e = this.entries.get(path);
    if (e?.at.length) {
      for (let n = Math.min(this.segmentAt(t)?.n, e.at.length - 1); n >= 0; n--) {
        const c = e.at[n] && this.chunk(path, n), v = c && lastValue(c, t);
        if (v !== undefined) return v;
      }
      return undefined;
    }
    if (!e && this.listOf(path)) return undefined;
    const s = this.series(path);
    if (!s) return undefined;
    const {chunks} = s;
    let lo = 0, hi = chunks.length;  // first chunk starting after t
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (chunks[mid].tt[0] <= t) lo = mid + 1; else hi = mid;
    }
    for (let j = lo - 1; j >= 0; j--) {
      const v = lastValue(chunks[j], t);
      if (v !== undefined) return v;
    }
    return undefined;
  }

  // the whole decoded message of a service at or before t, from a worker that keeps the last logs decompressed
  async message(service, t) {
    const seg = this.segmentAt(t);
    if (seg?.start === undefined) return null;
    const mono = (t - seg.opts.shift + seg.start) * 1e9 + 1000;  // 1 µs over, for the round trip through route seconds
    const ev = await this.loader.message(seg.key, seg.url, service, mono, this.schema);
    return ev?.[service] ?? null;
  }

  // a skipped long list is asked for by any of its indexes: keep it from now on and re-parse what's loaded
  request(path) {
    const m = this.listOf(path);
    if (!m) return;
    const glob = m[2], list = this.keep[m[1]] ||= [];
    if (list.includes(glob)) return;
    list.push(glob);
    this.kept.push({[m[1]]: [`${glob}/*`, `${glob}/*/*`]});
    this.reparse(this.kept.at(-1));
  }

  // an index column of a long list that only its short messages filled (/ubloxGnss/measurementReport/measurements/3/svId
  // when some messages have more than MAX_LIST): asks for the whole list once; the columns are replaced as it loads
  partial(e) {
    if (e.lists !== this.lists.size) {
      e.lists = this.lists.size;
      e.list = !!this.listOf(e.path);
    }
    if (!e.list) return;
    e.list = false;
    this.request(e.path);
  }

  // the match of [, service, list] when path is an index into a skipped long list
  listOf(path) {
    const m = path.match(/^\/([^/]+)(\/.*)\/\d+(\/.*)?$/);
    return m && this.lists.has(`/${m[1]}${m[2]}`) ? m : null;
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
        for (const seg of this.route.segments) if (seg.status === "done") this.readOnly(seg, only);
      });
    }
    merge(this.wanted, only);
  }

  // reads just the columns only picks from a loaded segment again
  readOnly(seg, only) {
    this.load(seg, only).then((p) => this.ingest(seg, p), () => {
      seg.loading = null;
      this.publish(seg);
    });
  }

  // evicts over the budget, and shows the total
  account() {
    if (this.bytes > this.budget) this.evict();
    this.totals();
    perf.mark("store", `${(this.bytes / 1e6).toFixed(0)} / ${(this.budget / 1e6).toFixed(0)} MB, ${this.entries.size} paths`);
  }

  // drops the least recently used columns and derived series not in use until under 90% of the budget
  evict() {
    const now = performance.now(), target = this.budget * 0.9;
    const lru = [...this.entries.values()].filter((e) => (e.at.length || e.put) && now - e.used > IN_USE).sort((a, b) => a.used - b.used);
    let dropped = false;
    for (const e of lru) {
      if (this.bytes <= target) break;
      if (e.put) {
        this.entries.delete(e.path);
        this.bytes -= e.put;
        dropped = true;
        continue;
      }
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
    if (dropped) {
      this.listed = null;
      this.emit("paths");
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
    this.listed = null;
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
    clearInterval(this.ticker);
    removeEventListener("online", this.online);
    if (this.shared) this.loader.cancel(this);
    else this.loader.dispose();
    this.can.dispose();
    this.status.remove();
  }
}

// adds {service: [globs]} b to a
const merge = (a, b) => {
  for (const svc in b) (a[svc] ||= []).push(...b[svc]);
  return a;
};

const split = (path) => path.match(/^\/([^/]+)(.*)$/).slice(1);

function drop(pack, s) {
  pack.bufs[s] = null;
  if (pack.t) pack.t[s] = null;
}

// min/max per block of BLOCK samples, then per FANOUT blocks, up to one block per chunk; built on first need from
// the blocks the worker packed, when it did
function pyramid(c) {
  if (c.pyramid) return c.pyramid;
  const {v, blocks} = c, Arr = ArrayBuffer.isView(v) ? v.constructor : Float64Array;
  const levels = blocks ? [blocks] : [];
  let lo = blocks?.lo ?? v, hi = blocks?.hi ?? v, size = blocks ? FANOUT : BLOCK;
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

// the min and max of lo[k] and hi[k], k in [k0, k1), into the buckets of width w from t0 (min.length of them) that
// the times tt[k * size] fall in: the bucket's end is kept, so a sample costs two compares
function bin(tt, lo, hi, k0, k1, size, t0, w, min, max) {
  const last = min.length - 1;
  let b = -1, end = -Infinity, a = 0, z = 0;
  for (let k = k0; k < k1; k++) {
    const time = tt[k * size];
    if (time >= end) {
      if (b >= 0) {
        min[b] = a;
        max[b] = z;
      }
      b = Math.min(last, Math.floor((time - t0) / w));
      end = b === last ? Infinity : t0 + (b + 1) * w;
      a = min[b];
      z = max[b];
    }
    if (lo[k] < a) a = lo[k];
    if (hi[k] > z) z = hi[k];
  }
  if (b >= 0) {
    min[b] = a;
    max[b] = z;
  }
}

// a chunk's last value at or before t
function lastValue(chunk, t) {
  const {tt} = chunk;
  if ("c" in chunk) return upperBound(tt, t) ? chunk.c : undefined;
  const {v} = chunk;
  for (let k = upperBound(tt, t) - 1; k >= 0; k--) if (v[k] === v[k] && v[k] !== null) return v[k];
  return undefined;
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
