// the worker → main transfer format: each service's times and columns packed into one ArrayBuffer, described by
// flat records, so the main thread deserializes a few objects per segment and makes views only on demand

const ARRAYS = [Uint8Array, Int8Array, Uint16Array, Int16Array, Uint32Array, Int32Array, Float32Array, Float64Array];
const TEXT = 8, CONST = 9;
// a column record: service, kind (ARRAYS index | TEXT | CONST), offset, length, index kind (-1: dense), index offset,
// constant (the value, or a consts index for text), enum ref (-1: none), text bytes offset
const REC = 9;

const encoder = new TextEncoder(), decoder = new TextDecoder();
const align = (x) => (x + 7) & ~7;

// services from compact(); only: {service: RegExp} keeps just those columns; sentEnums: refs already sent
export function pack(services, enums, only, sentEnums) {
  const out = {services: [], counts: [], bufs: [], paths: [], types: [], consts: [], enums: {}, lists: {}};
  const recs = [];
  for (const [name, s] of Object.entries(services)) {
    if (only && !only[name]) continue;
    for (const [path, n] of s.lists) out.lists[`/${name}${path}`] = n;
    const svc = out.services.length, parts = [s.t];
    let size = align(s.t.byteLength);
    const put = (a) => {
      parts.push(a);
      const at = size;
      size = align(size + a.byteLength);
      return at;
    };
    for (const [sub, c] of Object.entries(s.cols)) {
      if (only && !only[name].test(sub)) continue;
      const ref = c.names ? enums.indexOf(c.names) : -1;
      if (ref >= 0 && !sentEnums.has(ref)) {
        sentEnums.add(ref);
        out.enums[ref] = c.names;
      }
      out.paths.push(`/${name}${sub}`);
      out.types.push(c.type);
      const r = [svc, CONST, 0, 0, -1, 0, 0, ref, 0];
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

// the service and enum ref of column k, and the times of service s as a view
export const service = (p, k) => p.recs[k * REC];
export const enumRef = (p, k) => p.recs[k * REC + 7];
export const times = (p, s) => (p.t ||= [])[s] ||= new Float64Array(p.bufs[s], 0, p.counts[s]);

// column k of a pack as a contract chunk {t, v, i?} plus tt, the time of each value
export function unpack(p, k, n) {
  const r = p.recs, o = k * REC, s = r[o], kind = r[o + 1], len = r[o + 3], buf = p.bufs[s];
  const t = times(p, s);
  let v, i;
  if (kind === CONST) {
    const c = p.types[k] === "t" ? p.consts[r[o + 6]] : r[o + 6];
    v = [c, c];
    i = Uint32Array.of(0, t.length - 1);
  } else if (kind === TEXT) {
    const offs = new Uint32Array(buf, r[o + 2], len + 1), bytes = new Uint8Array(buf, r[o + 8], offs[len]);
    v = Array.from({length: len}, (_, j) => decoder.decode(bytes.subarray(offs[j], offs[j + 1])));
  } else {
    v = new ARRAYS[kind](buf, r[o + 2], len);
  }
  if (r[o + 4] >= 0) i = new ARRAYS[r[o + 4]](buf, r[o + 5], len);
  return {n, t, v, i, tt: i ? Float64Array.from(i, (j) => t[j]) : t};
}
