// CAN frames by message. The pipeline worker extracts each segment's frames through the reader's can() hook
// (CanFrames) and hands them over by transfer; a dedicated CAN worker (this file loaded as "can.js?worker")
// owns them and runs every scan; the main thread keeps per-message summaries only (Can, i.e. store.can).
import {U8, U32, list} from "./capnp.js";
import {DBC, Dbcs, bits, raw, value} from "./dbc.js";
import {parseRemote} from "./github.js";
import {upstream as github} from "./remote.js";

export const key = (bus, address) => bus * 2 ** 32 + address;
export const busOf = (k) => Math.floor(k / 2 ** 32);
export const addrOf = (k) => k % 2 ** 32;
export const hexId = (k) => `${busOf(k)}:${addrOf(k).toString(16).toUpperCase()}`;

const FADE = 2;  // seconds a byte change stays highlighted, as in cabana
const BUDGET = 512e6;  // frame bytes the CAN worker holds, ~90 segments

// ---- pipeline worker: one segment's frames, grouped by message ----

export class CanFrames {
  constructor(schema) {
    const f = Object.fromEntries(schema.structs.find((s) => s.name === "CanData").fields.map((x) => [x[0], x[3]]));
    this.addrPos = f.address * 4;
    this.srcPos = f.src;
    this.datIdx = f.dat;
    this.n = this.nwords = 0;
    this.ids = [];  // by bus: message ids in order of appearance, a table for 11-bit addresses and a map for the rest
    this.keys = [];  // by message id
    this.count = [];
    this.stride = [];
    this.time = new Float64Array(1 << 16);
    this.mid = new Uint16Array(1 << 16);
    this.len = new Uint8Array(1 << 16);
    this.words = new Uint32Array(1 << 18);  // frame data, each frame word-aligned
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
    if (this.nwords + count * 16 > this.words.length) this.words = grow(this.words, this.nwords + count * 16);
    for (let j = 0; j < count; j++) {
      const e = off + j * step, bus = this.srcPos < data ? U8[e + this.srcPos] : 0, address = U32[(e + this.addrPos) >> 2];
      const ids = this.ids[bus] ||= {std: new Int32Array(2048).fill(-1), ext: new Map()};
      let id = address < 2048 ? ids.std[address] : ids.ext.get(address) ?? -1;
      if (id < 0) {
        id = this.keys.length;
        if (address < 2048) ids.std[address] = id; else ids.ext.set(address, id);
        this.keys.push(key(bus, address));
        this.count.push(0);
        this.stride.push(0);
      }
      // dat: a near byte-list pointer, read inline (the hot path); anything else through list()
      let src = 0, len = 0;
      if (this.datIdx < ptrs) {
        const p = e + data + this.datIdx * 8, w0 = U32[p >> 2], w1 = U32[(p >> 2) + 1];
        if ((w0 & 3) === 1 && (w1 & 7) === 2) {
          src = p + 8 + (w0 >> 2) * 8;
          len = w1 >>> 3;
        } else if (w0) {
          const l = list(p);
          if (l) [src, len] = l;
        }
      }
      len = Math.min(len, 64);
      const n = this.n++;
      this.time[n] = t;
      this.mid[n] = id;
      this.len[n] = len;
      this.count[id]++;
      if (len > this.stride[id]) this.stride[id] = len;
      // capnp data is word-aligned, so it copies four bytes at a time
      for (let w = 0, end = (len + 3) >> 2; w < end; w++) this.words[this.nwords + w] = U32[(src >> 2) + w];
      this.nwords += (len + 3) >> 2;
    }
  }

  // messages in key order, each a run of frames in time order with a fixed data stride (its longest frame)
  finish() {
    const {n} = this, order = this.keys.map((k, id) => [k, id]).sort((a, b) => a[0] - b[0]), M = order.length;
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
    const bytes = new Uint8Array(this.words.buffer);
    for (let i = 0, src = 0; i < n; i++) {
      const id = this.mid[i], j = m[id], f = at[id]++, l = this.len[i];
      t[f] = this.time[i] - t0;
      len[f] = l;
      for (let b = 0, dst = dstart[j] + (f - start[j]) * stride[j]; b < l; b++) data[dst + b] = bytes[src + b];
      src += (l + 3) & ~3;
    }
    const can = {t0, keys, start, stride, dstart, t, len, data};
    return {can, transfer: [keys, start, stride, dstart, t, len, data].map((a) => a.buffer)};
  }
}

// ---- CAN worker: frames of every segment and the scans over them ----

const segs = [];  // resident, by time; each the CanFrames payload plus {n, base, idx, flips, before}
const counts = new Map();  // n -> {base, frames: Map(key -> count)} of every segment ever added, resident or not

function addSegment(n, c, off) {
  dropSegment(n);
  const s = {...c, n, base: c.t0 + off, idx: new Map(), flips: []};
  c.keys.forEach((k, m) => {
    s.idx.set(k, m);
    s.flips.push(flipCounts(s, m, c.start[m], c.start[m + 1]));
  });
  segs.splice(segs.findLastIndex((x) => x.base < s.base) + 1, 0, s);
  counts.set(n, {base: s.base, frames: new Map(Array.from(c.keys, (k, m) => [k, c.start[m + 1] - c.start[m]]))});
  // per message frame counts before each segment, for counts at t; evicted segments still count
  const before = new Map(), at = new Map();
  for (const [x, {frames}] of [...counts].sort((a, b) => a[1].base - b[1].base)) {
    at.set(x, new Map(before));
    for (const [k, f] of frames) before.set(k, (before.get(k) || 0) + f);
  }
  for (const x of segs) x.before = at.get(x.n);
}

// evicts a segment's frames; its counts stay
function dropSegment(n) {
  const j = segs.findIndex((s) => s.n === n);
  if (j >= 0) segs.splice(j, 1);
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
    const n = s.start[m + 1] - s.start[m], span = time(s, s.start[m + 1] - 1) - time(s, s.start[m]), freq = n > 1 && span > 0 ? (n - 1) / span : 0;
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

// signals of msg over [t0, t1] in n buckets: first, min, max and last value of each (NaN where empty)
function decimate(k, msg, names, t0, t1, n) {
  const sigs = names.map((x) => msg.sigs.find((sg) => sg.name === x));
  const out = sigs.map(() => ({first: new Float64Array(n).fill(NaN), min: new Float64Array(n).fill(NaN), max: new Float64Array(n).fill(NaN), last: new Float64Array(n).fill(NaN)}));
  each(k, t0, t1, (s, m, i) => {
    const b = Math.min(n - 1, Math.floor((time(s, i) - t0) / (t1 - t0) * n)), o = dataAt(s, m, i);
    sigs.forEach((sg, j) => {
      const v = sg ? value(msg, sg, s.data, o, s.len[i]) : NaN, d = out[j];
      if (v !== v) return;
      if (d.first[b] !== d.first[b]) d.first[b] = d.min[b] = d.max[b] = v;
      else if (v < d.min[b]) d.min[b] = v;
      else if (v > d.max[b]) d.max[b] = v;
      d.last[b] = v;
    });
  });
  return out;
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

const HEX = Array.from({length: 256}, (_, i) => i.toString(16).toUpperCase().padStart(2, "0"));

// the given messages' frames in time order as cabana's CSV: raw bytes, or the signals of msg. A Blob, built in parts,
// as a whole route runs to hundreds of MB
function csv(keys, t0, t1, msg) {
  const rows = [];
  for (const k of keys) each(k, t0, t1, (s, m, i) => rows.push([time(s, i), k, s.data, dataAt(s, m, i), s.len[i]]));
  rows.sort((a, b) => a[0] - b[0]);
  const cell = (sg, v) => (Number.isNaN(v) ? "" : v.toFixed(sg.precision));
  const line = ([t, k, d, o, len]) => {
    let x = `${t.toFixed(6)},0x${addrOf(k).toString(16)},${busOf(k)},`;
    if (msg) return x + msg.sigs.map((sg) => cell(sg, value(msg, sg, d, o, len))).join(",") + "\n";
    x += "0x";
    for (let b = o; b < o + len; b++) x += HEX[d[b]];
    return x + "\n";
  };
  const parts = [["time,addr,bus", ...(msg ? msg.sigs.map((sg) => sg.name) : ["data"])].join(",") + "\n"];
  let part = "";
  for (const r of rows) {
    part += line(r);
    if (part.length > 1 << 20) { parts.push(part); part = ""; }
  }
  parts.push(part);
  return new Blob(parts, {type: "text/csv"});
}

if (import.meta.url.endsWith("?worker")) {
  const ops = {
    add: ({n, can, off}) => { addSegment(n, can, off); },
    parse: ({name, text}) => new DBC(name, text).msgs,
    serialize: ({name, text, msgs}) => DBC.revive(name, text, msgs).serialize(),
    drop: ({n}) => { dropSegment(n); },
    masks: ({entries}) => { masks.clear(); for (const [k, v] of entries) masks.set(k, v); },
    state: ({t}) => state(t),
    heatmap: ({k, t0, t1}) => heatmap(k, t0, t1),
    frames: ({k, t, n, msg, filters}) => frames(k, t, n, msg, filters),
    decode: ({k, msg, only}) => decode(k, msg, only),
    decimate: ({k, msg, names, t0, t1, n}) => decimate(k, msg, names, t0, t1, n),
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
      postMessage({id, error: e.message || String(e), stack: e.stack});
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
    this.worker.onmessage = ({data: {id, r, error, stack}}) => {
      const c = this.calls.get(id);
      this.calls.delete(id);
      if (error) c.reject(Object.assign(new Error(error), {stack})); else c.resolve(r);
    };
    // a worker that doesn't load, or a reply that can't be read, would leave calls waiting forever
    this.worker.onerror = this.worker.onmessageerror = (e) => this.fail(e.message || "the CAN worker didn't load");
    this.msgs = new Map();  // key -> {key, bus, address, id, count, t0, t1, size, segs}
    this.handlers = {};
    this.dbcs = new Dbcs();
    this.dbcs.on((what) => what !== "saved" && this.dbcChanged());
    // leaving with unsaved DBC edits asks first
    this.unload = (e) => { if (this.dbcs.dirty) e.preventDefault(); };
    addEventListener("beforeunload", this.unload);
    this.dbcVersion = 0;
    this.dataVersion = 0;  // bumps on new frames or DBC edits
    this.cache = new Map();  // series path -> {series, msgVersion, have: Set(n), pending}
    this.selected = null;
    this.live = null;
    this.failed = null;  // why the CAN worker failed, if it did
    this.bytes = 0;
    this.budget = BUDGET;
    this.resident = new Map();  // segment -> its bytes in the worker
    this.evicted = new Set();
    this.reloading = new Set();
    this.at = 0;  // the playhead's segment
    // status.js touches the DOM, which this file's worker side can't import
    import("../status.js").then(({status}) => {
      if (this.disposed) return;
      this.status = status.source("can", {label: "CAN", order: 3});
      const segs = store.route?.segments || [];
      this.status.count(segs.length);
      for (const s of segs) if (s.status === "missing") this.status.seg(s.n, "missing");
      for (const n of this.resident.keys()) this.status.seg(n, this.failed ? "error" : "ready", {text: this.failed});
      this.status.bytes({memory: this.bytes});
    });
    this.suppressed = new Map();  // key -> per-byte 0xff for suppressed bytes
    this.suppressDefined = false;
    this.autoDbc = store.on?.("segment", () => this.fingerprint());
  }

  call(op, args = {}, transfer = []) {
    if (this.failed) return Promise.reject(new Error(this.failed));
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.calls.set(id, {resolve, reject});
      this.worker.postMessage({id, op, ...args}, transfer);
    });
  }

  // every call fails from here on; the segments show it, and the message list says why
  fail(reason) {
    if (this.failed) return;
    this.failed = reason;
    for (const c of this.calls.values()) c.reject(new Error(reason));
    this.calls.clear();
    for (const n of this.resident.keys()) this.status?.seg(n, "error", {text: reason});
    this.emit("notice");
  }

  on(event, fn) {
    (this.handlers[event] ||= new Set()).add(fn);
    return () => this.handlers[event].delete(fn);
  }
  emit(event, arg) { for (const fn of this.handlers[event] || []) fn(arg); }

  // a segment's frames from the pipeline worker (route time = mono time + off); summaries stay here
  ingest(n, can, off) {
    const {keys, start, t, len, stride} = can;
    // a reloaded segment is already in the summaries
    let named = false;  // a message the DBC knows showed up for the first time: its signals are new paths
    if (!this.evicted.delete(n)) keys.forEach((k, m) => {
      const a = start[m], b = start[m + 1];
      let s = this.msgs.get(k);
      if (!s) {
        this.msgs.set(k, s = {key: k, bus: busOf(k), address: addrOf(k), id: hexId(k), count: 0, t0: Infinity, t1: -Infinity, size: 0, segs: new Set()});
        named ||= !!this.dbcMsg(k);
      }
      s.count += b - a;
      s.t0 = Math.min(s.t0, can.t0 + off + t[a]);
      s.t1 = Math.max(s.t1, can.t0 + off + t[b - 1]);
      s.size = Math.max(s.size, len[b - 1], stride[m] && s.size);
      s.freq = s.count > 1 && s.t1 > s.t0 ? (s.count - 1) / (s.t1 - s.t0) : 0;
      s.segs.add(n);
    });
    const bytes = t.byteLength + len.byteLength + can.data.byteLength + keys.byteLength * 4;
    this.bytes += bytes - (this.resident.get(n) || 0);
    this.resident.set(n, bytes);
    this.call("add", {n, can, off}, [keys, can.start, stride, can.dstart, t, len, can.data].map((a) => a.buffer));
    this.status?.seg(n, this.failed ? "error" : "ready", {text: this.failed});
    this.evict();
    this.status?.bytes({memory: this.bytes});
    this.dataVersion++;
    this.emit("messages");
    if (named) this.store.emit?.("paths");
    for (const c of this.cache.values()) c.stale = true;
  }

  // over budget, the segments farthest from the playhead go; their summaries stay, and they come back from the
  // cached log when the playhead nears them
  evict() {
    const far = (n) => Math.abs(n - this.at);
    for (const n of [...this.resident.keys()].filter((n) => far(n) > 1).sort((a, b) => far(b) - far(a))) {
      if (this.bytes <= this.budget * 0.9) break;
      this.bytes -= this.resident.get(n);
      this.resident.delete(n);
      this.evicted.add(n);
      this.call("drop", {n});
      this.status?.seg(n, "disk");
    }
  }

  reload(n) {
    if (!this.evicted.has(n) || this.reloading.has(n)) return;
    this.reloading.add(n);
    this.status?.seg(n, "loading");
    this.store.load(this.store.route.segments[n], {}, {can: true}).then((p) => p.can && this.ingest(n, p.can, 0),
      () => this.status?.seg(n, "disk")).finally(() => this.reloading.delete(n));
  }

  messages() { return [...this.msgs.values()]; }

  // DBC messages never seen on any bus, listed as cabana does so they can still be edited; keyed on their DBC's bus
  unseen() {
    const seen = new Set(Array.from(this.msgs.values(), (s) => s.address)), out = [];
    for (const [b, f] of this.dbcs.files) {
      const bus = Math.max(b, 0);
      for (const m of f.msgs.values()) {
        if (seen.has(m.address)) continue;
        seen.add(m.address);
        const k = key(bus, m.address);
        out.push({key: k, bus, address: m.address, id: hexId(k), count: 0, freq: 0, size: m.size, t0: Infinity, t1: -Infinity, segs: new Set(), unseen: true});
      }
    }
    return out;
  }
  countAt(k) { const j = this.live?.index.get(k); return j === undefined ? 0 : this.live.count[j]; }
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
    this.at = this.store.route?.segments.find((s) => t >= s.t0 && t < s.t1)?.n ?? this.at;
    for (const n of [this.at - 1, this.at, this.at + 1]) this.reload(n);
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
  decimate(k, names, t0, t1, n) { return this.call("decimate", {k, msg: plain(this.dbcMsg(k)), names, t0, t1, n}); }
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
    this.dataVersion++;
    for (const c of this.cache.values()) c.stale = true;
    if (this.suppressDefined) this.sendMasks();
    this.emit("dbc");
    this.store.emit?.("paths");
  }

  // opendbc's DBC names and the fingerprint → DBC map
  async opendbc() {
    this.index ||= fetch(new URL("./opendbc.json", import.meta.url)).then((r) => r.json());
    const [fingerprints, src, local] = await Promise.all([this.index, this.upstream(), this.local()]);
    return {fingerprints, files: [...new Set([...src?.names || [], ...local?.names || []])].sort(), label: src?.co.label, local: local?.dir};
  }

  // opendbc from GitHub at the log's openpilot commit, so DBCs match the car code that ran; else opendbc master. The
  // fetching and generating happen in remote.js's worker: {names, co: {label}, text(name)}, or null offline
  upstream() {
    const remote = parseRemote(this.store.valueAt?.("/initData/gitRemote", Infinity)), commit = this.store.valueAt?.("/initData/gitCommit", Infinity);
    const at = remote && commit ? `${remote}@${commit}` : "master";
    if (this.src?.at !== at) {
      const args = remote && commit ? {repo: remote, commit} : {};
      this.src = {at, opendbc: github("dbcs", args).then(({label, names}) => ({names, co: {label},
        text: (name) => github("dbc", {...args, name}).then((r) => r.text)}), () => null)};
    }
    return this.src.opendbc;
  }

  // the opendbc checkout on this machine through the local daemon, when there is one: {daemon, id, dir, names}
  local() {
    return this.localDbc ||= import("../daemon.js").then(async ({daemon}) => {
      if (!daemon.has("repos")) return null;
      const [repo] = await daemon.json("/local/repos");
      if (!repo?.opendbc) return null;
      const files = await daemon.json(`/local/repos/${repo.id}/dbc`);
      return {daemon, id: repo.id, dir: repo.opendbc.dbc, names: new Set(files.map((f) => f.name.replace(/\.dbc$/, "")))};
    }).catch(() => null);
  }

  // saves into the local opendbc; a file changed on disk since it was read (or one that exists, for a DBC that
  // didn't come from there) is a conflict unless forced. Returns {conflict} or the daemon's {path, generated}
  async saveLocal(dbc, force = false) {
    const local = await this.local();
    const path = `/local/repos/${local.id}/dbc/${dbc.name}.dbc${force ? "" : `?mtime=${dbc.local?.mtime ?? 0}`}`;
    const r = await fetch(local.daemon.url(path), {method: "PUT", body: await this.serialize(dbc)});
    if (r.status === 409) return {conflict: true};
    if (!r.ok) throw new Error((await r.json().catch(() => null))?.error || `saving ${dbc.name}.dbc: ${r.status}`);
    const out = await r.json();
    dbc.local = {mtime: out.mtime};
    local.names.add(dbc.name);
    this.dbcs.saved(dbc);
    return out;
  }

  // DBC text parsed and serialized in the worker, as a big one takes tens of ms
  async parseDbc(name, text) { return DBC.revive(name, text, await this.call("parse", {name, text})); }
  serialize(dbc) { return this.call("serialize", {name: dbc.name, text: dbc.text, msgs: dbc.msgs}); }

  // a DBC by name: from the local opendbc when the daemon has it (that's the copy to edit), else from GitHub
  async loadOpendbc(name, buses = [-1]) {
    const local = await this.local();
    if (local?.names.has(name)) {
      const {text, mtime} = await local.daemon.json(`/local/repos/${local.id}/dbc/${name}.dbc`);
      const dbc = await this.parseDbc(name, text);
      dbc.local = {mtime};
      return this.dbcs.open(buses, dbc);
    }
    const src = await this.upstream();
    if (!src) {
      this.src = null;  // try again next time
      throw new Error("can't reach opendbc on GitHub");
    }
    this.dbcs.open(buses, await this.parseDbc(name, await src.text(name)));
  }

  // the car's DBC from the fingerprint, once, unless a DBC is already open; a failure is a notice, not an error
  async fingerprint() {
    if (this.fingerprinted || this.dbcs.all().length) return;
    const fp = this.store.valueAt?.("/carParams/carFingerprint", Infinity);
    if (!fp) return;
    this.fingerprinted = fp;
    const name = (await this.opendbc()).fingerprints[fp];
    if (!name || this.dbcs.all().length) return;
    await this.loadOpendbc(name).catch((e) => {
      this.notice = `${name}.dbc didn't load: ${e.message}`;
      this.emit("notice");
    });
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
    const m = this.dbcs.file(+bus)?.msgByName(msg), s = m?.sigs.find((x) => x.name === sig);
    return s ? {k: key(+bus, m.address), m, sig: s} : null;
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
      for (const ch of chunks) sc.chunks.set(ch.n, {t: ch.t, v: ch.v, tt: ch.t});
      sc.chunks = new Map([...sc.chunks].sort((a, b) => a[0] - b[0]));
      sc.version++;
      this.emit("series", sp);
    }
  }

  dispose() {
    this.autoDbc?.();
    removeEventListener("beforeunload", this.unload);
    this.status?.remove();
    this.disposed = true;
    this.worker.terminate();
  }
}

// a message without derived references, for postMessage
function plain(m) {
  return m && {...m, mux: m.mux && {...m.mux}, sigs: m.sigs.map((s) => ({...s}))};
}
