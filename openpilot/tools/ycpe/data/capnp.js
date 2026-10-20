// rlog reader: walks back-to-back capnp Event messages and pulls every scalar into per-service columns,
// driven by schema.json (offsets from the capnp compiler). No per-event allocation: the walk follows
// a lazily built tree of field plans whose leaves point straight at their output column.

// views of the log being read; exported as live bindings for the can() hook
export let U8, I8, U16, I16, U32, I32, F32, F64;
const segBase = new Int32Array(64);
const f32 = new Float32Array(1), f32u = new Uint32Array(f32.buffer);
const f64 = new Float64Array(1), f64u = new Uint32Array(f64.buffer);

// the result of follow(): content offset in bytes, and either struct sizes or list element info
let rOff = 0, rData = 0, rPtrs = 0, rElem = 0, rCount = 0;

function views(buf) {
  const b = buf.buffer, o = buf.byteOffset, n = buf.byteLength;
  U8 = new Uint8Array(b, o, n);
  I8 = new Int8Array(b, o, n);
  U16 = new Uint16Array(b, o, n >> 1);
  I16 = new Int16Array(b, o, n >> 1);
  U32 = new Uint32Array(b, o, n >> 2);
  I32 = new Int32Array(b, o, n >> 2);
  F32 = new Float32Array(b, o, n >> 2);
  F64 = new Float64Array(b, o, n >> 3);
}

// resolves the pointer at byte p (including far pointers); false for null or unsupported
function follow(p) {
  let lo = U32[p >> 2], hi = U32[(p >> 2) + 1];
  if (lo === 0 && hi === 0) return false;
  let kind = lo & 3, target;
  if (kind === 2) {
    const pad = segBase[hi] + (lo >>> 3) * 8;
    if (lo & 4) {  // double far: the pad holds a far pointer to the content and a tag with the sizes
      const flo = U32[pad >> 2], fhi = U32[(pad >> 2) + 1];
      target = segBase[fhi] + (flo >>> 3) * 8;
      lo = U32[(pad >> 2) + 2];
      hi = U32[(pad >> 2) + 3];
      kind = lo & 3;
    } else {
      lo = U32[pad >> 2];
      hi = U32[(pad >> 2) + 1];
      kind = lo & 3;
      target = pad + 8 + (lo >> 2) * 8;
    }
  } else {
    target = p + 8 + (lo >> 2) * 8;
  }
  if (kind === 0) {
    rOff = target;
    rData = (hi & 0xffff) * 8;
    rPtrs = hi >>> 16;
    return true;
  }
  if (kind === 1) {
    rElem = hi & 7;
    rCount = hi >>> 3;
    if (rElem === 7) {  // composite: a tag word with the element count and struct sizes leads the elements
      const tlo = U32[target >> 2], thi = U32[(target >> 2) + 1];
      rCount = tlo >>> 2;
      rData = (thi & 0xffff) * 8;
      rPtrs = thi >>> 16;
      target += 8;
    }
    rOff = target;
    return true;
  }
  return false;
}

// the struct list at pointer p as [byte offset, count, data bytes, pointers] per element (far pointers followed), or null
export function list(p) {
  return follow(p) ? [rOff, rCount, rData, rPtrs] : null;
}

const decoder = new TextDecoder();
function readText(p) {
  if (!follow(p) || rElem !== 2 || rCount === 0) return "";
  const n = rCount - 1;  // trailing NUL
  if (n < 64) {
    let s = "";
    for (let i = 0; i < n; i++) {
      const c = U8[rOff + i];
      if (c > 127) return decoder.decode(U8.subarray(rOff, rOff + n));
      s += String.fromCharCode(c);
    }
    return s;
  }
  return decoder.decode(U8.subarray(rOff, rOff + n));
}

// scalar types: [size in bytes (0 = bit), reader(bytePos)]
const u64 = (p) => U32[p >> 2] + U32[(p >> 2) + 1] * 4294967296;
const i64 = (p) => U32[p >> 2] + I32[(p >> 2) + 1] * 4294967296;
const SCALAR = {
  i8: [1, (p) => I8[p]], u8: [1, (p) => U8[p]], i16: [2, (p) => I16[p >> 1]], u16: [2, (p) => U16[p >> 1]],
  i32: [4, (p) => I32[p >> 2]], u32: [4, (p) => U32[p >> 2]], i64: [8, i64], u64: [8, u64],
  f32: [4, (p) => F32[p >> 2]], f64: [8, (p) => F64[p >> 3]], e: [2, (p) => U16[p >> 1]],
};
// non-zero defaults are stored XORed; only a handful of fields have them
function xorDefault(type, raw, def) {
  if (type === "f32") { f32[0] = raw; f32u[0] ^= def; return f32[0]; }
  if (type === "f64") {
    f64[0] = raw;
    f64u[0] ^= def % 4294967296;
    f64u[1] ^= Math.floor(def / 4294967296);
    return f64[0];
  }
  if (type === "i64" || type === "u64") return Number(BigInt[type === "i64" ? "asIntN" : "asUintN"](64, BigInt(raw) ^ BigInt(def)));
  return type === "u32" ? (raw ^ def) >>> 0 : raw ^ def;
}
const MAX_LIST = 16;  // longer lists are not split into per-index columns unless asked for

// path globs: * matches one path component
export const pattern = (globs) => new RegExp(`^(${globs.map((g) => g.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, "[^/]+")).join("|")})$`);

// a column fills as events arrive; missing values stay NaN
class Column {
  constructor(type, n, names) {
    this.type = type;
    this.names = names;
    this.v = type === "t" ? new Array(n).fill(null) : type === "f64" || type === "i64" || type === "u64" || type === "u32" || type === "i32" ? new Float64Array(n).fill(NaN) : new Float32Array(n).fill(NaN);
  }
}

export class Reader {
  constructor(schema) {
    this.schema = schema;
    this.event = schema.structs[0];
    this.ev = Object.fromEntries(this.event.fields.map((f) => [f[0], f]));
  }

  // indexes and extracts one decompressed log; returns Map(which → {name, t, valid, cols: Map(path → Column), lists})
  // keep: {service: ["/speeds", "/laneLines/*/y"]} splits those long lists per index too; every other long list
  // is skipped and reported in lists (path → longest length) so it can be asked for later
  read(buf, {can = null, keep = {}} = {}) {
    views(buf);
    const {event} = this;
    const services = new Map();
    const byDisc = [];
    for (const f of event.fields) if (f[1] >= 0) byDisc[f[1]] = f;
    const validOff = this.ev.valid[3];
    const msgs = [];
    const counts = new Uint32Array(byDisc.length);

    // pass 1: message boundaries and service counts
    let m = 0;
    const end = buf.byteLength - 8;
    while (m < end) {
      const nseg = U32[m >> 2] + 1;
      if (nseg > segBase.length) break;
      let size = 0;
      for (let s = 0; s < nseg; s++) size += U32[(m >> 2) + 1 + s];
      const head = (4 + 4 * nseg + 7) & ~7;
      const next = m + head + size * 8;
      if (next > buf.byteLength) break;  // truncated tail
      segBase[0] = m + head;
      for (let s = 1; s < nseg; s++) segBase[s] = segBase[s - 1] + U32[(m >> 2) + s] * 8;
      if (follow(segBase[0])) {
        const which = U16[(rOff >> 1) + event.disc];
        if (byDisc[which]) {
          counts[which]++;
          msgs.push(m, which);
        }
      }
      m = next;
    }

    for (let w = 0; w < counts.length; w++) {
      if (!counts[w]) continue;
      const f = byDisc[w];
      services.set(w, {name: f[0], field: f, n: 0, t: new Float64Array(counts[w]), valid: new Uint8Array(counts[w]), plan: null,
        cols: new Map(), keep: keep[f[0]] && pattern(keep[f[0]]), listPlans: [], lists: new Map()});
    }

    // pass 2: extraction
    for (let k = 0; k < msgs.length; k += 2) {
      m = msgs[k];
      const svc = services.get(msgs[k + 1]);
      const nseg = U32[m >> 2] + 1;
      segBase[0] = m + ((4 + 4 * nseg + 7) & ~7);
      for (let s = 1; s < nseg; s++) segBase[s] = segBase[s - 1] + U32[(m >> 2) + s] * 8;
      follow(segBase[0]);
      const evOff = rOff, evData = rData;
      const i = svc.n++;
      svc.t[i] = u64(evOff) / 1e9;
      svc.valid[i] = validOff >> 3 < evData ? ((U8[evOff + (validOff >> 3)] >> (validOff & 7)) & 1) ^ 1 : 1;
      const f = svc.field, ptr = evOff + evData + f[3] * 8;
      if (f[2] === "s") {
        if (!follow(ptr)) continue;
        svc.plan ||= this.plan(f[4], "", svc, counts[msgs[k + 1]]);
        this.walk(svc.plan, rOff, rData, rPtrs, i);
      } else if (f[2] === "t") {
        svc.plan ||= {text: this.col(svc, "", "t", counts[msgs[k + 1]])};
        svc.plan.text.v[i] = readText(ptr);
      } else if (f[2] === "l") {
        if (can && (f[0] === "can" || f[0] === "sendcan")) can(f[0], svc.t[i], ptr);
        else {
          svc.plan ||= this.listPlan(f[4], "", svc, counts[msgs[k + 1]]);
          this.walkList(svc.plan, ptr, i);
        }
      }
    }
    for (const svc of services.values()) {
      for (const lp of svc.listPlans) if (lp.longest) svc.lists.set(lp.path, lp.longest);
      delete svc.plan;
      delete svc.listPlans;
    }
    return services;
  }

  col(svc, path, type, n, names) {
    let c = svc.cols.get(path);
    if (!c) svc.cols.set(path, c = new Column(type, n, names));
    return c;
  }

  // the field plan of one struct at one path: scalars, groups, sub-structs, lists, texts
  plan(structIdx, path, svc, n) {
    const s = this.schema.structs[structIdx];
    const p = {disc: s.disc, scalars: [], bools: [], groups: [], structs: [], lists: [], texts: []};
    for (const [name, disc, type, off, ref, def] of s.fields) {
      const sub = `${path}/${name}`;
      if (type === "b") p.bools.push({disc, byte: off >> 3, bit: off & 7, def: def ? 1 : 0, col: this.col(svc, sub, "b", n)});
      else if (SCALAR[type]) {
        const [size, read] = SCALAR[type];
        p.scalars.push({disc, pos: off * size, size, read, type, def: def || 0, col: this.col(svc, sub, type, n, type === "e" ? this.schema.enums[ref] : undefined)});
      } else if (type === "g") p.groups.push({disc, plan: this.plan(ref, sub, svc, n)});
      else if (type === "s") p.structs.push({disc, idx: off, ref, sub, plan: null});
      else if (type === "t") p.texts.push({disc, idx: off, col: this.col(svc, sub, "t", n)});
      else if (type === "l") p.lists.push({disc, idx: off, plan: this.listPlan(ref, sub, svc, n)});
    }
    p.svc = svc;
    p.n = n;
    return p;
  }

  listPlan(elem, path, svc, n) {
    const lp = {elem, path, svc, n, items: [], keep: !!svc.keep?.test(path), longest: 0};
    svc.listPlans.push(lp);
    return lp;
  }

  walk(p, off, dataBytes, ptrs, i) {
    const which = p.disc >= 0 && p.disc * 2 < dataBytes ? U16[(off >> 1) + p.disc] : 0;
    for (const f of p.scalars) {
      if (f.disc >= 0 && f.disc !== which) continue;
      let v = f.pos + f.size <= dataBytes ? f.read(off + f.pos) : 0;
      if (f.def) v = xorDefault(f.type, v, f.def);
      f.col.v[i] = v;
    }
    for (const f of p.bools) {
      if (f.disc >= 0 && f.disc !== which) continue;
      f.col.v[i] = (f.byte < dataBytes ? (U8[off + f.byte] >> f.bit) & 1 : 0) ^ f.def;
    }
    for (const g of p.groups) if (g.disc < 0 || g.disc === which) this.walk(g.plan, off, dataBytes, ptrs, i);
    const pbase = off + dataBytes;
    for (const f of p.structs) {
      if (f.disc >= 0 && f.disc !== which || f.idx >= ptrs) continue;
      if (!follow(pbase + f.idx * 8)) continue;
      f.plan ||= this.plan(f.ref, f.sub, p.svc, p.n);
      this.walk(f.plan, rOff, rData, rPtrs, i);
    }
    for (const f of p.texts) {
      if (f.disc >= 0 && f.disc !== which) continue;
      f.col.v[i] = f.idx < ptrs ? readText(pbase + f.idx * 8) : "";
    }
    for (const f of p.lists) {
      if (f.disc >= 0 && f.disc !== which || f.idx >= ptrs) continue;
      this.walkList(f.plan, pbase + f.idx * 8, i);
    }
  }

  // small lists become one column (or sub-struct) per index: /path/0, /path/1, ...
  walkList(lp, ptr, i) {
    if (!follow(ptr)) return;
    if (rCount > MAX_LIST && !lp.keep) {
      if (rCount > lp.longest) lp.longest = rCount;
      return;
    }
    const [type, ref] = lp.elem;
    const count = rCount, base = rOff, elem = rElem, data = rData, ptrs = rPtrs;
    for (let j = 0; j < count; j++) {
      let item = lp.items[j];
      if (type === "s") {
        if (elem !== 7) return;
        item ||= lp.items[j] = this.plan(ref, `${lp.path}/${j}`, lp.svc, lp.n);
        this.walk(item, base + j * (data + ptrs * 8), data, ptrs, i);
      } else if (type === "b") {
        if (elem !== 1) return;
        item ||= lp.items[j] = this.col(lp.svc, `${lp.path}/${j}`, "b", lp.n);
        item.v[i] = (U8[base + (j >> 3)] >> (j & 7)) & 1;
      } else if (SCALAR[type]) {
        const [size, read] = SCALAR[type];
        if ([0, 0, 1, 2, 4, 8][elem] !== size) return;
        item ||= lp.items[j] = this.col(lp.svc, `${lp.path}/${j}`, type, lp.n, type === "e" ? this.schema.enums[ref] : undefined);
        item.v[i] = read(base + j * size);
      } else if (type === "t") {
        if (elem !== 6) return;
        item ||= lp.items[j] = this.col(lp.svc, `${lp.path}/${j}`, "t", lp.n);
        item.v[i] = readText(base + j * 8);
      } else {
        return;
      }
    }
  }
}

// columns shrink before transfer: never-set ones are dropped, constant ones collapse to c, ones set in under
// half the events keep an index into the service times (i), and values move to the narrowest exact array
export function compact(services) {
  const out = {};
  const transfer = [];
  for (const svc of services.values()) {
    const cols = {};
    // the event's valid flag, unless the struct has its own (lateralTorqueParameters, vehicleParameters)
    svc.cols.set(svc.cols.has("/valid") ? "/_valid" : "/valid", {type: "b", v: svc.valid});
    for (const [path, c] of svc.cols) {
      const col = c.type === "t" ? compactText(c.v) : compactNumbers(c.v);
      if (!col) continue;
      col.type = c.type;
      if (c.names) col.names = c.names;
      if (ArrayBuffer.isView(col.v)) transfer.push(col.v.buffer);
      if (col.i) transfer.push(col.i.buffer);
      cols[path] = col;
    }
    out[svc.name] = {t: svc.t, cols, lists: svc.lists};
    transfer.push(svc.t.buffer);
  }
  return {services: out, transfer};
}

const indexArray = (n, count) => n < 65536 ? new Uint16Array(count) : new Uint32Array(count);

function intArray(lo, hi) {
  if (lo >= 0) return hi < 256 ? Uint8Array : hi < 65536 ? Uint16Array : hi < 4294967296 ? Uint32Array : Float64Array;
  if (lo >= -128 && hi < 128) return Int8Array;
  return lo >= -32768 && hi < 32768 ? Int16Array : lo >= -2147483648 && hi < 2147483648 ? Int32Array : Float64Array;
}

function compactNumbers(v) {
  const n = v.length;
  const wide = v instanceof Float64Array;
  let count = 0, lo = Infinity, hi = -Infinity, ints = true, f32 = true;
  for (let k = 0; k < n; k++) {
    const x = v[k];
    if (x !== x) continue;
    count++;
    if (x < lo) lo = x;
    if (x > hi) hi = x;
    if (ints && !Number.isInteger(x)) ints = false;
    if (wide && f32 && Math.fround(x) !== x) f32 = false;
  }
  if (!count) return null;
  if (count === n && lo === hi && n > 1) return {c: lo};
  if (count * 2 >= n) {
    if (count < n) return {v: wide && f32 ? new Float32Array(v) : v};  // gaps stay NaN, so floats
    const Arr = ints ? intArray(lo, hi) : f32 ? Float32Array : Float64Array;
    return {v: v instanceof Arr ? v : new Arr(v)};
  }
  const Arr = ints ? intArray(lo, hi) : f32 ? Float32Array : Float64Array;
  const i = indexArray(n, count), out = new Arr(count);
  for (let k = 0, j = 0; k < n; k++) if (v[k] === v[k]) { i[j] = k; out[j++] = v[k]; }
  return {v: out, i};
}

function compactText(v) {
  const n = v.length;
  let count = 0, first = null, constant = true;
  for (let k = 0; k < n; k++) {
    if (v[k] === null) continue;
    if (count++ === 0) first = v[k];
    else if (v[k] !== first) constant = false;
  }
  if (!count) return null;
  if (count === n && constant && n > 1) return {c: first};
  if (count === n) return {v};  // any gap makes it sparse, so text never holds nulls
  const i = indexArray(n, count), out = new Array(count);
  for (let k = 0, j = 0; k < n; k++) if (v[k] !== null) { i[j] = k; out[j++] = v[k]; }
  return {v: out, i};
}
