// whole messages as plain objects, for the inspector: the reader only keeps columns, so a message is decoded again
// from the log when it's asked for. Fields follow schema.json; a union holds only its set member, enums are names,
// 64-bit integers are Numbers (or exact BigInts, asked for), data is a Uint8Array, and a null struct or list is null.
import {upperBound} from "./bounds.js";

const READ = {
  i8: (d, p) => d.getInt8(p), u8: (d, p) => d.getUint8(p), i16: (d, p) => d.getInt16(p, true), u16: (d, p) => d.getUint16(p, true),
  e: (d, p) => d.getUint16(p, true), i32: (d, p) => d.getInt32(p, true), u32: (d, p) => d.getUint32(p, true),
  i64: (d, p) => d.getUint32(p, true) + d.getInt32(p + 4, true) * 4294967296,
  u64: (d, p) => d.getUint32(p, true) + d.getUint32(p + 4, true) * 4294967296,
  f32: (d, p) => d.getFloat32(p, true), f64: (d, p) => d.getFloat64(p, true),
};
const BIG = {...READ, i64: (d, p) => d.getBigInt64(p, true), u64: (d, p) => d.getBigUint64(p, true)};
const SIZE = {i8: 1, u8: 1, i16: 2, u16: 2, e: 2, i32: 4, u32: 4, f32: 4, i64: 8, u64: 8, f64: 8};
const ELEM_BYTES = [0, 0, 1, 2, 4, 8, 8];  // by list element size code: void, bit, bytes..., pointer
const ZERO = new DataView(new ArrayBuffer(8)), bits = new DataView(new ArrayBuffer(8));
const text = new TextDecoder();

// one message (segment table first) as an Event object; with big, 64-bit integers are BigInts
export function decode(schema, bytes, big = false) {
  const read = big ? BIG : READ, dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const nseg = dv.getUint32(0, true) + 1, segs = [];
  for (let s = 0, at = (4 + 4 * nseg + 7) & ~7; s < nseg; s++) {
    segs.push(at);
    at += dv.getUint32(4 + 4 * s, true) * 8;
  }

  // the pointer at byte p as {struct: true, at, data, ptrs} or {at, elem, count}; null when unset
  function pointer(p) {
    let lo = dv.getUint32(p, true), hi = dv.getUint32(p + 4, true), at;
    if (!lo && !hi) return null;
    if ((lo & 3) === 2) {
      const pad = segs[hi] + (lo >>> 3) * 8;
      if (lo & 4) {  // double far: a far pointer to the content, then a tag with the sizes
        at = segs[dv.getUint32(pad + 4, true)] + (dv.getUint32(pad, true) >>> 3) * 8;
        [lo, hi] = [dv.getUint32(pad + 8, true), dv.getUint32(pad + 12, true)];
      } else {
        [lo, hi] = [dv.getUint32(pad, true), dv.getUint32(pad + 4, true)];
        at = pad + 8 + (lo >> 2) * 8;
      }
    } else {
      at = p + 8 + (lo >> 2) * 8;
    }
    if ((lo & 3) === 0) return {struct: true, at, data: (hi & 0xffff) * 8, ptrs: hi >>> 16};
    return (lo & 3) === 1 ? {struct: false, at, elem: hi & 7, count: hi >>> 3} : null;
  }

  // a scalar at p, zero past the end of its struct's data; non-zero defaults are stored XORed
  function scalar(type, p, end, def) {
    let src = dv;
    if (p + SIZE[type] > end) [src, p] = [ZERO, 0];
    if (!def) return read[type](src, p);
    if (type === "f32" || type === "f64") {
      bits.setUint32(0, src.getUint32(p, true) ^ def % 4294967296, true);
      if (type === "f64") bits.setUint32(4, src.getUint32(p + 4, true) ^ Math.floor(def / 4294967296), true);
      return READ[type](bits, 0);
    }
    const v = read[type](src, p);
    if (type === "i64" || type === "u64") {
      const x = BigInt[type === "i64" ? "asIntN" : "asUintN"](64, BigInt(v) ^ BigInt(def));
      return big ? x : Number(x);
    }
    return type === "u32" ? (v ^ def) >>> 0 : v ^ def;
  }

  const enumName = (ref, v) => schema.enums[ref]?.[v] ?? v;

  function struct(idx, at, data, ptrs) {
    const s = schema.structs[idx], out = {};
    const which = s.disc >= 0 && s.disc * 2 < data ? dv.getUint16(at + s.disc * 2, true) : 0;
    for (const [name, disc, type, off, ref, def] of s.fields) {
      if (disc >= 0 && disc !== which) continue;
      if (type === "b") out[name] = !!((off >> 3 < data ? (dv.getUint8(at + (off >> 3)) >> (off & 7)) & 1 : 0) ^ (def ? 1 : 0));
      else if (type === "e") out[name] = enumName(ref, scalar(type, at + off * 2, at + data, def));
      else if (SIZE[type]) out[name] = scalar(type, at + off * SIZE[type], at + data, def);
      else if (type === "g") out[name] = struct(ref, at, data, ptrs);
      else out[name] = off < ptrs ? value(type, ref, at + data + off * 8) : type === "t" ? "" : null;
    }
    return out;
  }

  // the pointer at p as a field of type s (ref: struct), t, d, l (ref: element [type, ref]) or p (AnyPointer)
  function value(type, ref, p) {
    const ptr = pointer(p);
    // schema.json doesn't resolve generics (Map entries): a byte list is text if its only NUL ends it, else data
    if (type === "p" && ptr?.elem === 2) type = bytes.subarray(ptr.at, ptr.at + ptr.count).indexOf(0) === ptr.count - 1 ? "t" : "d";
    if (type === "t") return ptr && !ptr.struct && ptr.count ? text.decode(bytes.subarray(ptr.at, ptr.at + ptr.count - 1)) : "";
    if (!ptr || type === "p" || ptr.struct !== (type === "s")) return null;
    if (type === "s") return struct(ref, ptr.at, ptr.data, ptr.ptrs);
    return type === "d" ? bytes.slice(ptr.at, ptr.at + ptr.count) : list(ref, ptr);
  }

  function list([type, ref], {at, elem, count}) {
    const each = (fn) => Array.from({length: count}, (_, j) => fn(j));
    if (elem === 7) {  // composite: a tag word with the count and struct sizes leads the elements
      const hi = dv.getUint32(at + 4, true), data = (hi & 0xffff) * 8, ptrs = hi >>> 16, stride = data + ptrs * 8;
      count = dv.getUint32(at, true) >>> 2;
      return type === "s" ? each((j) => struct(ref, at + 8 + j * stride, data, ptrs)) : [];
    }
    const size = ELEM_BYTES[elem];
    if (type === "b") return elem === 1 ? each((j) => !!((dv.getUint8(at + (j >> 3)) >> (j & 7)) & 1)) : [];
    if (type === "s") return each((j) => struct(ref, at + j * size, elem === 6 ? 0 : size, elem === 6 ? 1 : 0));
    if (!SIZE[type]) return elem === 6 ? each((j) => value(type, ref, at + j * 8)) : [];
    if (SIZE[type] !== size) return [];
    return each((j) => type === "e" ? enumName(ref, READ.e(dv, at + j * 2)) : read[type](dv, at + j * size));
  }

  const ev = pointer(segs[0]);
  return ev?.struct ? struct(0, ev.at, ev.data, ev.ptrs) : null;
}

// the messages of a decompressed log as [byte offset, length]
export function* messages(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  for (let m = 0; m + 8 <= buf.byteLength;) {
    const nseg = dv.getUint32(m, true) + 1;
    if (nseg > 64 || m + 4 + 4 * nseg > buf.byteLength) return;
    let size = (4 + 4 * nseg + 7) & ~7;
    for (let s = 0; s < nseg; s++) size += dv.getUint32(m + 4 + 4 * s, true) * 8;
    if (m + size > buf.byteLength) return;  // truncated tail
    yield [m, size];
    m += size;
  }
}

// where each service's messages are in a decompressed log: Map(service → {mono: [logMonoTime (ns)], at: [], size: []})
export function index(schema, buf) {
  const event = schema.structs[0], names = [], out = new Map();
  for (const f of event.fields) if (f[1] >= 0) names[f[1]] = f[0];
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  for (const [m, size] of messages(buf)) {
    // the Event is the first object of segment 0: logMonoTime is its first word, then the union tag
    const root = m + ((4 + 4 * (dv.getUint32(m, true) + 1) + 7) & ~7), lo = dv.getUint32(root, true);
    const ev = root + 8 + (lo >> 2) * 8, data = (dv.getUint32(root + 4, true) & 0xffff) * 8;
    const name = (lo & 3) === 0 && event.disc * 2 < data && names[dv.getUint16(ev + event.disc * 2, true)];
    if (!name) continue;
    if (!out.has(name)) out.set(name, {mono: [], at: [], size: []});
    const s = out.get(name);
    s.mono.push(READ.u64(dv, ev));
    s.at.push(m);
    s.size.push(size);
  }
  return out;
}

// the last message of a service at or before logMonoTime mono (ns), decoded with exact 64-bit integers; null when
// there's none
export function messageAt(schema, buf, idx, service, mono) {
  const s = idx.get(service);
  if (!s) return null;
  const lo = upperBound(s.mono, mono);
  return lo ? decode(schema, buf.subarray(s.at[lo - 1], s.at[lo - 1] + s.size[lo - 1]), true) : null;
}
