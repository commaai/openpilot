// the worker → main transfer format: each service's times and columns packed into one ArrayBuffer, described by
// flat records, so the main thread deserializes a few objects per segment and makes views only on demand

const ARRAYS = [Uint8Array, Int8Array, Uint16Array, Int16Array, Uint32Array, Int32Array, Float32Array, Float64Array];
const TEXT = 8, CONST = 9;
// a column record: service, kind (ARRAYS index | TEXT | CONST), offset, length, index kind (-1: dense), index offset,
// constant (the value, or a consts index for text), enum ref (-1: none), text bytes offset, block minimums offset
// (-1: none), block maximums offset (see minmax)
const REC = 11;
export const BLOCK = 16;  // samples per block of the min/max pyramid the store decimates with (see minmax)

const encoder = new TextEncoder(), decoder = new TextDecoder();
const align = (x) => (x + 7) & ~7;

// services from compact(); want(service, path), if given, picks the columns. The enum names a pack uses go with it:
// they're small, and a worker may serve several stores
export function pack(services, enums, want) {
  const out = {services: [], counts: [], bufs: [], paths: [], types: [], consts: [], enums: {}, lists: {}};
  const recs = [];
  for (const [name, s] of Object.entries(services)) {
    for (const [path, n] of s.lists) out.lists[`/${name}${path}`] = n;
    const cols = Object.entries(s.cols).filter(([sub]) => !want || want(name, sub));
    if (!cols.length) continue;
    const svc = out.services.length, parts = [s.t];
    let size = align(s.t.byteLength);
    const put = (a) => {
      parts.push(a);
      const at = size;
      size = align(size + a.byteLength);
      return at;
    };
    for (const [sub, c] of cols) {
      const ref = c.names ? enums.indexOf(c.names) : -1;
      if (ref >= 0) out.enums[ref] = c.names;
      out.paths.push(`/${name}${sub}`);
      out.types.push(c.type);
      const r = [svc, CONST, 0, 0, -1, 0, 0, ref, 0, -1, 0];
      if ("c" in c) {
        r[6] = c.type === "t" ? out.consts.push(c.c) - 1 : c.c;
      } else if (c.type === "t") {
        const bytes = c.v.map((x) => encoder.encode(x)), offs = new Uint32Array(bytes.length + 1);
        for (let k = 0; k < bytes.length; k++) offs[k + 1] = offs[k] + bytes[k].length;
        const flat = new Uint8Array(offs[bytes.length]);
        bytes.forEach((b, k) => flat.set(b, offs[k]));
        r[1] = TEXT;
        r[2] = put(offs);
        r[3] = bytes.length;
        r[8] = put(flat);
      } else {
        r[1] = ARRAYS.indexOf(c.v.constructor);
        r[2] = put(c.v);
        r[3] = c.v.length;
        if (c.v.length >= BLOCK) {
          const {lo, hi} = minmax(c.v, c.v, BLOCK);
          r[9] = put(lo);
          r[10] = put(hi);
        }
      }
      if (c.i) {
        r[4] = ARRAYS.indexOf(c.i.constructor);
        r[5] = put(c.i);
      }
      recs.push(...r);
    }
    const buf = new ArrayBuffer(size), bytes = new Uint8Array(buf);
    for (let k = 0, at = 0; k < parts.length; at = align(at + parts[k].byteLength), k++) {
      bytes.set(new Uint8Array(parts[k].buffer, parts[k].byteOffset, parts[k].byteLength), at);
    }
    out.services.push(name);
    out.counts.push(s.t.length);
    out.bufs.push(buf);
  }
  out.paths = out.paths.join("\n");
  out.types = out.types.join("\n");
  out.recs = new Float64Array(recs);
  return out;
}

// the min and max of each `size` values of lo and hi (one array twice for samples), as arrays of type Arr: a level of
// the min/max pyramid the store decimates with. Its first level is built here while the worker has the column at
// hand, so a series' first decimation on the main thread reads 1/BLOCK of its samples
export function minmax(lo, hi, size, Arr = lo.constructor) {
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
  return {lo: nlo, hi: nhi};
}

// the service and enum ref of column k, and the times of service s as a view
export const service = (p, k) => p.recs[k * REC];
export const enumRef = (p, k) => p.recs[k * REC + 7];
export const times = (p, s) => (p.t ||= [])[s] ||= new Float64Array(p.bufs[s], 0, p.counts[s]);

// column k of a pack as a contract chunk {t, v, i?} plus tt, the time of each value, and the first pyramid level of
// numbers (see minmax). A constant column's chunk also has c, its value: its v (a value at every message time, like
// any other dense column) is only filled in when read, and the store's readers that can read c do
export function unpack(p, k, n) {
  const r = p.recs, o = k * REC, s = r[o], kind = r[o + 1], len = r[o + 3], buf = p.bufs[s];
  const t = times(p, s);
  if (kind === CONST) {
    const c = constant(p, k);
    let v;
    return {n, t, tt: t, c, get v() { return v ||= typeof c === "string" ? new Array(t.length).fill(c) : new Float64Array(t.length).fill(c); }};
  }
  let v, i, blocks;
  if (kind === TEXT) {
    const offs = new Uint32Array(buf, r[o + 2], len + 1), bytes = new Uint8Array(buf, r[o + 8], offs[len]);
    // one decode, cut at the byte offsets when they're also character offsets (ASCII); else one decode each
    const all = decoder.decode(bytes);
    v = all.length === bytes.length ? Array.from({length: len}, (_, j) => all.slice(offs[j], offs[j + 1]))
      : Array.from({length: len}, (_, j) => decoder.decode(bytes.subarray(offs[j], offs[j + 1])));
  } else {
    v = new ARRAYS[kind](buf, r[o + 2], len);
    const m = Math.ceil(len / BLOCK);
    if (r[o + 9] >= 0) blocks = {lo: new ARRAYS[kind](buf, r[o + 9], m), hi: new ARRAYS[kind](buf, r[o + 10], m)};
  }
  if (r[o + 4] >= 0) i = new ARRAYS[r[o + 4]](buf, r[o + 5], len);
  return {n, t, v, i, tt: i ? Float64Array.from(i, (j) => t[j]) : t, blocks};
}

// the value of column k when it's a constant, else undefined
export function constant(p, k) {
  const r = p.recs, o = k * REC;
  if (r[o + 1] !== CONST) return undefined;
  return p.types[k] === "t" ? p.consts[r[o + 6]] : r[o + 6];
}
