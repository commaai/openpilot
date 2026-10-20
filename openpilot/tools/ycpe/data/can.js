// CAN frames by message. The pipeline worker extracts each segment's frames through the reader's can() hook
// (CanFrames) and hands them over by transfer; a dedicated CAN worker (this file loaded as "can.js?worker")
// owns them and runs every scan; the main thread keeps per-message summaries only (Can, i.e. store.can).
import {U8, U32, list} from "./capnp.js";
import {DBC, Dbcs, bits, raw, value} from "./dbc.js";

export const key = (bus, address) => bus * 2 ** 32 + address;
export const busOf = (k) => Math.floor(k / 2 ** 32);
export const addrOf = (k) => k % 2 ** 32;
export const hexId = (k) => `${busOf(k)}:${addrOf(k).toString(16).toUpperCase()}`;

const FADE = 2;  // seconds a byte change stays highlighted, as in cabana

// ---- pipeline worker: one segment's frames, grouped by message ----

export class CanFrames {
  constructor(schema) {
    const f = Object.fromEntries(schema.structs.find((s) => s.name === "CanData").fields.map((x) => [x[0], x[3]]));
    this.addrPos = f.address * 4;
    this.srcPos = f.src;
    this.datIdx = f.dat;
    this.n = this.nbytes = 0;
    this.ids = new Map();  // key -> message id in order of appearance
    this.count = [];
    this.stride = [];
    this.time = new Float64Array(1 << 16);
    this.mid = new Uint16Array(1 << 16);
    this.len = new Uint8Array(1 << 16);
    this.bytes = new Uint8Array(1 << 20);
    this.hook = (name, t, ptr) => { if (name === "can") this.add(t, ptr); };
  }

  add(t, ptr) {
    const l = list(ptr);
    if (!l) return;
    const [off, count, data, ptrs] = l, step = data + ptrs * 8;
    const grow = (a, n) => { const b = new a.constructor(Math.max(n, a.length * 2)); b.set(a); return b; };
    if (this.n + count > this.time.length) {
      this.time = grow(this.time, this.n + count);
      this.mid = grow(this.mid, this.n + count);
      this.len = grow(this.len, this.n + count);
    }
    if (this.nbytes + count * 64 > this.bytes.length) this.bytes = grow(this.bytes, this.nbytes + count * 64);
    for (let j = 0; j < count; j++) {
      const e = off + j * step, d = this.datIdx < ptrs ? list(e + data + this.datIdx * 8) : null;
      const k = key(this.srcPos < data ? U8[e + this.srcPos] : 0, U32[(e + this.addrPos) >> 2]);
      let id = this.ids.get(k);
      if (id === undefined) {
        this.ids.set(k, id = this.count.length);
        this.count.push(0);
        this.stride.push(0);
      }
      const n = this.n++, len = d ? Math.min(d[1], 64) : 0;
      this.time[n] = t;
      this.mid[n] = id;
      this.len[n] = len;
      this.count[id]++;
      if (len > this.stride[id]) this.stride[id] = len;
      for (let b = 0; b < len; b++) this.bytes[this.nbytes + b] = U8[d[0] + b];
      this.nbytes += len;
    }
  }

  // messages in key order, each a run of frames in time order with a fixed data stride (its longest frame)
  finish() {
    const {n} = this, order = [...this.ids].sort((a, b) => a[0] - b[0]), M = order.length;
    const keys = new Float64Array(M), start = new Uint32Array(M + 1), stride = new Uint8Array(M), dstart = new Uint32Array(M + 1);
    const at = new Uint32Array(M), m = new Uint16Array(M);  // by message id: next frame slot, sorted index
    order.forEach(([k, id], j) => {
      keys[j] = k;
      m[id] = j;
      stride[j] = this.stride[id];
      start[j + 1] = start[j] + this.count[id];
      at[id] = start[j];
      dstart[j + 1] = dstart[j] + this.count[id] * stride[j];
    });
    const t0 = n ? this.time[0] : 0, t = new Float32Array(n), len = new Uint8Array(n), data = new Uint8Array(dstart[M]);
    for (let i = 0, src = 0; i < n; i++) {
      const id = this.mid[i], j = m[id], f = at[id]++, l = this.len[i];
      t[f] = this.time[i] - t0;
      len[f] = l;
      for (let b = 0, dst = dstart[j] + (f - start[j]) * stride[j]; b < l; b++) data[dst + b] = this.bytes[src + b];
      src += l;
    }
    const can = {t0, keys, start, stride, dstart, t, len, data};
    return {can, transfer: [keys, start, stride, dstart, t, len, data].map((a) => a.buffer)};
  }
}

// ---- CAN worker: frames of every segment and the scans over them ----

const segs = [];  // by time; each the CanFrames payload plus {n, base, idx, flips, before}

function addSegment(n, c, off) {
  const s = {...c, n, base: c.t0 + off, idx: new Map(), flips: []};
  c.keys.forEach((k, m) => {
    s.idx.set(k, m);
    s.flips.push(flipCounts(s, m, c.start[m], c.start[m + 1]));
  });
  segs.splice(segs.findLastIndex((x) => x.base < s.base) + 1, 0, s);
  // per message frame counts before each segment, for counts at t
  const before = new Map();
  for (const x of segs) {
    x.before = new Map(before);
    x.keys.forEach((k, m) => before.set(k, (before.get(k) || 0) + x.start[m + 1] - x.start[m]));
  }
}

// cabana's HeatmapCounts over frames [i0, i1) of message m: flips between neighbours, never against a byte that just appeared
function flipCounts(s, m, i0, i1, out = new Uint32Array(s.stride[m] * 8)) {
  const w = s.stride[m], d = s.data, base = s.dstart[m] - s.start[m] * w;
  for (let i = i0 + 1; i < i1; i++) {
    const a = base + (i - 1) * w, b = a + w, common = Math.min(s.len[i], s.len[i - 1]);
    for (let j = 0; j < common; j++) {
      const diff = d[a + j] ^ d[b + j];
      if (diff) for (let bit = 0; bit < 8; bit++) if (diff & (128 >> bit)) out[j * 8 + bit]++;
    }
  }
  return out;
}

const time = (s, i) => s.base + s.t[i];
const dataAt = (s, m, i) => s.dstart[m] + (i - s.start[m]) * s.stride[m];

// first frame of message m in s after t
function upper(s, m, t) {
  let lo = s.start[m], hi = s.start[m + 1];
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (time(s, mid) <= t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// the last segment starting at or before t
function segAt(t) {
  let j = segs.length - 1;
  while (j > 0 && segs[j].base > t) j--;
  return j;
}

// the last frame of message k at or before t: [segment, m, frame] or null
function lastFrame(k, t, from = segAt(t)) {
  for (let j = from; j >= 0; j--) {
    const s = segs[j], m = s.idx.get(k);
    if (m === undefined) continue;
    const i = upper(s, m, t) - 1;
    if (i >= s.start[m]) return [s, m, i];
  }
  return null;
}

// every frame of k in [t0, t1]: fn(s, m, i) in time order
function each(k, t0, t1, fn) {
  for (const s of segs) {
    const m = s.idx.get(k);
    if (m === undefined || time(s, s.start[m]) > t1 || time(s, s.start[m + 1] - 1) < t0) continue;
    for (let i = upper(s, m, t0 - 1e-9), end = upper(s, m, t1); i < end; i++) fn(s, m, i);
  }
}

const masks = new Map();  // key -> per-byte bits to ignore when highlighting changes

// every message as of t: count, last data, and per byte the age and kind of its last change (1 up, 2 down, 3 periodic)
function state(t) {
  const keys = [...new Set(segs.flatMap((s) => [...s.keys]))].sort((a, b) => a - b);
  const M = keys.length, from = segAt(t);
  const count = new Uint32Array(M), last = new Float64Array(M).fill(NaN), off = new Uint32Array(M + 1);
  const found = keys.map((k) => lastFrame(k, t, from));
  found.forEach((f, j) => { off[j + 1] = off[j] + (f ? f[0].len[f[2]] : 0); });
  const data = new Uint8Array(off[M]), age = new Float32Array(off[M]).fill(Infinity), kind = new Uint8Array(off[M]);
  found.forEach((f, j) => {
    if (!f) return;
    const [s, m, i] = f, len = s.len[i], d = s.data, cur = dataAt(s, m, i), mask = masks.get(keys[j]);
    count[j] = (s.before.get(keys[j]) || 0) + i - s.start[m] + 1;
    last[j] = time(s, i);
    data.set(d.subarray(cur, cur + len), off[j]);
    // walk back over the fade window, comparing every byte at once
    const n = s.start[m + 1] - s.start[m], freq = n > 1 ? (n - 1) / (time(s, s.start[m + 1] - 1) - time(s, s.start[m])) : 0;
    let open = 0;
    const changed = new Float64Array(len).fill(NaN);
    for (let b = 0; b < len; b++) if (!mask || mask[b] !== 0xff) open++;
    for (let p = i; p > s.start[m] && open && t - time(s, p) < FADE; p--) {
      const a = dataAt(s, m, p - 1), b0 = dataAt(s, m, p);
      for (let b = 0; b < Math.min(len, s.len[p - 1]); b++) {
        const keep = mask ? ~mask[b] & 0xff : 0xff, x = d[a + b] & keep, y = d[b0 + b] & keep;
        if (x === y || changed[b] === -Infinity) continue;
        // the last change sets the direction; an earlier, recent change the other way makes it periodic
        if (Number.isNaN(changed[b])) {
          changed[b] = time(s, p);
          age[off[j] + b] = t - changed[b];
          kind[off[j] + b] = y > x ? 1 : 2;
          continue;
        }
        if ((changed[b] - time(s, p)) * freq <= 10 && (y > x ? 1 : 2) !== kind[off[j] + b]) kind[off[j] + b] = 3;
        changed[b] = -Infinity;
        open--;
      }
    }
  });
  return {keys: Float64Array.from(keys), count, last, off, data, age, kind};
}

// bit flips over [t0, t1]: whole segments come from their summaries
function heatmap(k, t0, t1) {
  let out = new Uint32Array(0);
  for (const s of segs) {
    const m = s.idx.get(k);
    if (m === undefined) continue;
    const a = s.start[m], b = s.start[m + 1];
    if (time(s, b - 1) < t0 || time(s, a) > t1) continue;
    if (out.length < s.stride[m] * 8) { const o = new Uint32Array(s.stride[m] * 8); o.set(out); out = o; }
    if (time(s, a) >= t0 && time(s, b - 1) <= t1) s.flips[m].forEach((x, j) => { out[j] += x; });
    else flipCounts(s, m, upper(s, m, t0 - 1e-9), upper(s, m, t1), out);
  }
  return out;
}

// up to n frames of k at or before t, newest first, optionally only those passing filter(msg, d, o, len)
function frames(k, t, n, msg, filters) {
  const out = {t: [], data: [], more: false};
  const pass = (s, m, i) => !filters?.length || filters.every(({sig, op, v}) => {
    const x = value(msg, msg.sigs.find((q) => q.name === sig), s.data, dataAt(s, m, i), s.len[i]);
    return op === "=" ? x === v : op === "!=" ? x !== v : op === ">" ? x > v : op === ">=" ? x >= v : op === "<" ? x < v : x <= v;
  });
  for (let j = segAt(t); j >= 0; j--) {
    const s = segs[j], m = s.idx.get(k);
    if (m === undefined) continue;
    for (let i = upper(s, m, t) - 1; i >= s.start[m]; i--) {
      if (!pass(s, m, i)) continue;
      if (out.t.length === n) { out.more = true; return out; }
      out.t.push(time(s, i));
      out.data.push(s.data.slice(dataAt(s, m, i), dataAt(s, m, i) + s.len[i]));
    }
  }
  return out;
}

// every signal of msg over every segment (or just `only`), as store-style chunks per signal
function decode(k, msg, only) {
  const out = Object.fromEntries(msg.sigs.map((s) => [s.name, []])), transfer = [];
  for (const s of segs) {
    const m = s.idx.get(k);
    if (m === undefined || (only && !only.includes(s.n))) continue;
    const a = s.start[m], n = s.start[m + 1] - a, t = new Float64Array(n);
    for (let i = 0; i < n; i++) t[i] = time(s, a + i);
    for (const sig of msg.sigs) {
      const v = new Float64Array(n);
      for (let i = 0; i < n; i++) v[i] = value(msg, sig, s.data, dataAt(s, m, a + i), s.len[a + i]);
      out[sig.name].push({n: s.n, t, v});
      transfer.push(v.buffer);
    }
    transfer.push(t.buffer);
  }
  return [out, transfer];
}

// cabana's find similar bits: how often each bit on `bus` disagrees with (or matches) one bit of message k
function similarBits(k, byte, bit, bus, equal, minCount) {
  const rows = [];
  const targets = new Set(segs.flatMap((s) => [...s.keys].filter((x) => busOf(x) === bus)));
  for (const tk of targets) {
    let count = 0, miss = new Uint32Array(0);
    for (const s of segs) {
      const m = s.idx.get(tk);
      if (m === undefined) continue;
      count += s.start[m + 1] - s.start[m];
      for (let i = s.start[m]; i < s.start[m + 1]; i++) {
        const f = lastFrame(k, time(s, i));
        if (!f || f[0].len[f[2]] <= byte) continue;
        const want = (f[0].data[dataAt(f[0], f[1], f[2]) + byte] >> (7 - bit)) & 1, len = s.len[i], o = dataAt(s, m, i);
        if (miss.length < len * 8) { const x = new Uint32Array(len * 8); x.set(miss); miss = x; }
        for (let j = 0; j < len * 8; j++) miss[j] += (((s.data[o + (j >> 3)] >> (7 - (j & 7))) & 1) === want) !== equal;
      }
    }
    if (count <= minCount) continue;
    miss.forEach((x, j) => { if (x / count < 0.5) rows.push({key: tk, byte: j >> 3, bit: j & 7, mismatches: x, total: count, perc: x / count * 100}); });
  }
  return rows.sort((a, b) => a.perc - b.perc);
}

// cabana's find signal: candidates narrow with each search, each remembering where its last match was
const find = {history: []};

function findInit({buses, addresses, t0, t1, minSize, maxSize, le, signed, factor, offset}) {
  const cands = [];
  for (const k of new Set(segs.flatMap((s) => [...s.keys]))) {
    if (buses.length && !buses.includes(busOf(k))) continue;
    if (addresses.length && !addresses.includes(addrOf(k))) continue;
    const f = lastFrame(k, Infinity);
    if (!f || f[0].base + f[0].t[f[2]] < t0) continue;
    const total = f[0].len[f[2]] * 8;
    for (let size = minSize; size <= maxSize; size++) {
      for (let start = 0; start <= total - size; start++) cands.push({key: k, sig: bits({start, size, le, signed, factor, offset}), t: t0, values: []});
    }
  }
  find.t1 = t1;
  find.history = [cands];
}

function findNext(op, v1, v2) {
  const cmp = {"=": (v) => v === v1, ">": (v) => v > v1, ">=": (v) => v >= v1, "!=": (v) => v !== v1, "<": (v) => v < v1, "<=": (v) => v <= v1,
               between: (v) => v >= v1 && v <= v2}[op];
  const next = [];
  for (const c of find.history.at(-1)) {
    let hit = null;
    for (const s of segs) {
      const m = s.idx.get(c.key);
      if (m === undefined || hit) continue;
      for (let i = upper(s, m, c.t), end = upper(s, m, find.t1); i < end; i++) {
        const v = raw(c.sig, s.data, dataAt(s, m, i), s.len[i]) * c.sig.factor + c.sig.offset;
        if (cmp(v)) { hit = {...c, t: time(s, i), values: [...c.values, [time(s, i), v]]}; break; }
      }
    }
    if (hit) next.push(hit);
  }
  find.history.push(next);
  return next;
}

// raw frames of the given messages as CSV, in time order
function csv(keys, t0, t1, msg) {
  const rows = [];
  for (const k of keys) each(k, t0, t1, (s, m, i) => rows.push([time(s, i), k, s.data.subarray(dataAt(s, m, i), dataAt(s, m, i) + s.len[i]), s.len[i]]));
  rows.sort((a, b) => a[0] - b[0]);
  const hex = (d) => [...d].map((x) => x.toString(16).padStart(2, "0")).join("");
  if (msg) {
    return [["time", ...msg.sigs.map((s) => s.name)].join(","),
            ...rows.map(([t, , d, len]) => [t.toFixed(6), ...msg.sigs.map((s) => value(msg, s, d, 0, len))].join(","))].join("\n") + "\n";
  }
  return ["time,addr,bus,data", ...rows.map(([t, k, d]) => `${t.toFixed(6)},0x${addrOf(k).toString(16)},${busOf(k)},${hex(d)}`)].join("\n") + "\n";
}

if (import.meta.url.endsWith("?worker")) {
  const ops = {
    add: ({n, can, off}) => { addSegment(n, can, off); },
    masks: ({entries}) => { masks.clear(); for (const [k, v] of entries) masks.set(k, v); },
    state: ({t}) => state(t),
    heatmap: ({k, t0, t1}) => heatmap(k, t0, t1),
    frames: ({k, t, n, msg, filters}) => frames(k, t, n, msg, filters),
    decode: ({k, msg, only}) => decode(k, msg, only),
    similar: ({k, byte, bit, bus, equal, min}) => similarBits(k, byte, bit, bus, equal, min),
    findInit: (q) => { findInit(q); return find.history[0].length; },
    findNext: ({cmp, v1, v2}) => findNext(cmp, v1, v2).slice(0, 300),
    findUndo: () => { if (find.history.length > 1) find.history.pop(); return find.history.at(-1).slice(0, 300); },
    findCount: () => find.history.at(-1)?.length || 0,
    csv: ({keys, t0, t1, msg}) => csv(keys, t0, t1, msg),
    memory: () => segs.reduce((b, s) => b + s.t.byteLength + s.len.byteLength + s.data.byteLength + s.keys.byteLength * 4, 0),
  };
  globalThis.onmessage = ({data: {id, op, ...args}}) => {
    try {
      let r = ops[op](args), transfer = [];
      if (op === "decode") [r, transfer] = r;
      postMessage({id, r}, transfer);
    } catch (e) {
      postMessage({id, error: String(e.stack || e)});
    }
  };
}

// ---- main thread: store.can ----

export class Can {
  constructor(store) {
    this.store = store;
    this.worker = new Worker(new URL("./can.js?worker", import.meta.url), {type: "module"});
    this.calls = new Map();
    this.nextId = 0;
    this.worker.onmessage = ({data: {id, r, error}}) => {
      const c = this.calls.get(id);
      this.calls.delete(id);
      if (error) c.reject(new Error(error)); else c.resolve(r);
    };
    this.msgs = new Map();  // key -> {key, bus, address, id, count, t0, t1, size, segs}
    this.handlers = {};
    this.dbcs = new Dbcs();
    this.dbcs.on(() => this.dbcChanged());
    this.dbcVersion = 0;
    this.cache = new Map();  // series path -> {series, msgVersion, have: Set(n), pending}
    this.selected = null;
    this.live = null;
    this.bytes = 0;
    this.suppressed = new Map();  // key -> per-byte 0xff for suppressed bytes
    this.suppressDefined = false;
    this.autoDbc = store.on?.("segment", () => this.fingerprint());
  }

  call(op, args = {}, transfer = []) {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.calls.set(id, {resolve, reject});
      this.worker.postMessage({id, op, ...args}, transfer);
    });
  }

  on(event, fn) {
    (this.handlers[event] ||= new Set()).add(fn);
    return () => this.handlers[event].delete(fn);
  }
  emit(event, arg) { for (const fn of this.handlers[event] || []) fn(arg); }

  // a segment's frames from the pipeline worker (route time = mono time + off); summaries stay here
  ingest(n, can, off) {
    const {keys, start, t, len, stride} = can;
    keys.forEach((k, m) => {
      const a = start[m], b = start[m + 1];
      let s = this.msgs.get(k);
      if (!s) this.msgs.set(k, s = {key: k, bus: busOf(k), address: addrOf(k), id: hexId(k), count: 0, t0: Infinity, t1: -Infinity, size: 0, segs: new Set()});
      s.count += b - a;
      s.t0 = Math.min(s.t0, can.t0 + off + t[a]);
      s.t1 = Math.max(s.t1, can.t0 + off + t[b - 1]);
      s.size = Math.max(s.size, len[b - 1], stride[m] && s.size);
      s.freq = s.count > 1 ? (s.count - 1) / (s.t1 - s.t0) : 0;
      s.segs.add(n);
    });
    this.bytes += t.byteLength + len.byteLength + can.data.byteLength + keys.byteLength * 4;
    this.call("add", {n, can, off}, [keys, can.start, stride, can.dstart, t, len, can.data].map((a) => a.buffer));
    this.emit("messages");
    for (const c of this.cache.values()) c.stale = true;
  }

  messages() { return [...this.msgs.values()]; }
  msg(k) { return this.msgs.get(k); }
  dbcMsg(k) { return this.dbcs.msg(busOf(k), addrOf(k)); }
  name(k) { return this.dbcMsg(k)?.name || ""; }

  select(k) {
    if (this.selected === k) return;
    this.selected = k;
    this.emit("select", k);
  }

  // live bytes at t, coalesced to one request in flight; emits "live"
  requestState(t) {
    if (this.stateBusy) { this.stateNext = t; return; }
    this.stateBusy = true;
    this.call("state", {t}).then((r) => {
      this.live = r;
      r.index = new Map(Array.from(r.keys, (k, j) => [k, j]));
      r.t = t;
      this.stateBusy = false;
      this.emit("live", r);
      if (this.stateNext !== undefined) { const n = this.stateNext; this.stateNext = undefined; this.requestState(n); }
    });
  }

  frames(k, t, n = 200, filters = null) { return this.call("frames", {k, t, n, msg: filters?.length ? plain(this.dbcMsg(k)) : null, filters}); }
  heatmap(k, t0, t1) { return this.call("heatmap", {k, t0, t1}); }
  similarBits(q) { return this.call("similar", q); }
  findInit(q) { return this.call("findInit", q); }
  findNext(cmp, v1, v2) { return this.call("findNext", {cmp, v1, v2}); }
  findUndo() { return this.call("findUndo"); }
  csv(keys, t0, t1, decoded = false) { return this.call("csv", {keys, t0, t1, msg: decoded ? plain(this.dbcMsg(keys[0])) : null}); }

  // suppression masks for live highlighting: suppressed bytes plus, optionally, every DBC-defined bit
  suppressHighlighted(on) {
    if (!on) this.suppressed.clear();
    else if (this.live) {
      const {keys, off, age} = this.live;
      keys.forEach((k, j) => {
        for (let b = off[j]; b < off[j + 1]; b++) {
          if (age[b] < FADE) (this.suppressed.get(k) || this.suppressed.set(k, new Uint8Array(64)).get(k))[b - off[j]] = 0xff;
        }
      });
    }
    this.sendMasks();
    return [...this.suppressed.values()].reduce((n, m) => n + m.filter((x) => x).length, 0);
  }
  setSuppressDefined(on) {
    this.suppressDefined = on;
    this.sendMasks();
  }
  sendMasks() {
    const entries = new Map(this.suppressed);
    if (this.suppressDefined) {
      for (const k of this.msgs.keys()) {
        const m = this.dbcMsg(k);
        if (!m) continue;
        const mask = new Uint8Array(64);
        mask.set(m.mask.subarray(0, 64));
        entries.get(k)?.forEach((x, b) => { mask[b] |= x; });
        entries.set(k, mask);
      }
    }
    this.call("masks", {entries: [...entries]});
    if (this.live) this.requestState(this.live.t);
  }

  // ---- DBCs ----

  dbcChanged() {
    this.dbcVersion++;
    for (const c of this.cache.values()) c.stale = true;
    if (this.suppressDefined) this.sendMasks();
    this.emit("dbc");
    this.store.emit?.("paths");
  }

  async opendbc() {
    return this.index ||= fetch(new URL("./opendbc.json", import.meta.url)).then((r) => r.json());
  }

  async loadOpendbc(name, buses = [-1]) {
    const text = await fetch(new URL(`./opendbc/${name}.dbc`, import.meta.url)).then((r) => {
      if (!r.ok) throw new Error(`${name}.dbc: ${r.status}`);
      return r.text();
    });
    this.dbcs.open(buses, new DBC(name, text));
  }

  // the car's DBC from the fingerprint, once, unless a DBC is already open
  async fingerprint() {
    if (this.fingerprinted || this.dbcs.all().length) return;
    const fp = this.store.valueAt?.("/carParams/carFingerprint", Infinity);
    if (!fp) return;
    this.fingerprinted = fp;
    const name = (await this.opendbc()).fingerprints[fp];
    if (name && !this.dbcs.all().length) await this.loadOpendbc(name);
  }

  // ---- decoded signals as series: /dbc/<bus>/<Msg>/<Sig> ----

  paths() {
    const out = [];
    for (const s of this.msgs.values()) {
      const m = this.dbcMsg(s.key);
      if (m) for (const sig of m.sigs) out.push({path: `/dbc/${s.bus}/${m.name}/${sig.name}`, type: "f64"});
    }
    return out;
  }

  parse(path) {
    const [, dbc, bus, msg, sig] = path.split("/");
    if (dbc !== "dbc") return null;
    const m = this.dbcs.file(+bus)?.msgByName(msg);
    return m && m.sigs.find((s) => s.name === sig) ? {k: key(+bus, m.address), m, sig} : null;
  }

  // the store's series shape; decoding happens in the worker and "series" fires when chunks arrive
  series(path) {
    const p = this.parse(path);
    if (!p) return null;
    let c = this.cache.get(path);
    if (!c || c.dbcVersion !== this.dbcVersion) {
      c = {dbcVersion: this.dbcVersion, chunks: new Map(), version: 0};
      this.cache.set(path, c);
      c.stale = true;
    }
    if (c.stale && !c.pending) this.decode(p, path, c);
    return {path, type: "f64", chunks: [...c.chunks.values()], version: c.version};
  }

  async decode({k, m}, path, c) {
    const only = [...(this.msgs.get(k)?.segs || [])].filter((n) => !c.chunks.has(n));
    c.stale = false;
    if (!only.length) return;
    c.pending = true;
    const version = this.dbcVersion;
    const r = await this.call("decode", {k, msg: plain(m), only});
    c.pending = false;
    if (version !== this.dbcVersion) return;
    // every signal of the message came back; fill the caches of the ones in use
    for (const [name, chunks] of Object.entries(r)) {
      const sp = path.slice(0, path.lastIndexOf("/") + 1) + name, sc = sp === path ? c : this.cache.get(sp);
      if (!sc || sc.dbcVersion !== version) continue;
      for (const ch of chunks) sc.chunks.set(ch.n, {t: ch.t, v: ch.v});
      sc.chunks = new Map([...sc.chunks].sort((a, b) => a[0] - b[0]));
      sc.version++;
      this.emit("series", sp);
    }
  }

  dispose() {
    this.autoDbc?.();
    this.worker.terminate();
  }
}

// a message without derived references, for postMessage
function plain(m) {
  return m && {...m, mux: m.mux && {...m.mux}, sigs: m.sigs.map((s) => ({...s}))};
}
