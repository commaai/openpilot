// whole messages as plain objects, for the inspector: the reader only keeps columns, so a message is decoded again
// from the log when it's asked for. Fields follow schema.json; a union holds only its set member, enums are names,
// 64-bit integers are Numbers (or exact BigInts, asked for), data is a Uint8Array, and a null struct or list is null.
import {upperBound} from "./bounds.js";
import {pointer as follow, root, views, xorDefault} from "./capnp.js";

const READ = {
  i8: (d, p) => d.getInt8(p), u8: (d, p) => d.getUint8(p), i16: (d, p) => d.getInt16(p, true), u16: (d, p) => d.getUint16(p, true),
  e: (d, p) => d.getUint16(p, true), i32: (d, p) => d.getInt32(p, true), u32: (d, p) => d.getUint32(p, true),
  i64: (d, p) => d.getUint32(p, true) + d.getInt32(p + 4, true) * 4294967296,
  u64: (d, p) => d.getUint32(p, true) + d.getUint32(p + 4, true) * 4294967296,
  f32: (d, p) => d.getFloat32(p, true), f64: (d, p) => d.getFloat64(p, true),
};
const BIG = {...READ, i64: (d, p) => d.getBigInt64(p, true), u64: (d, p) => d.getBigUint64(p, true)};
const SIZE = {i8: 1, u8: 1, i16: 2, u16: 2, e: 2, i32: 4, u32: 4, f32: 4, i64: 8, u64: 8, f64: 8};
const ELEM_BITS = [0, 1, 8, 16, 32, 64, 64];  // by list element size code: void, bit, bytes..., pointer
const ZERO = new DataView(new ArrayBuffer(8));
const text = new TextDecoder();

// one message (segment table first) as an Event object; with big, 64-bit integers are BigInts
export function decode(schema, bytes, big = false) {
  const read = big ? BIG : READ, dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  views(bytes);

  // the pointer at byte p (capnp.js follows it); one whose content leaves the message throws, as a read past it would
  function pointer(p) {
    const ptr = follow(p);
    if (!ptr) return null;
    const stride = ptr.data + ptr.ptrs * 8;
    const size = ptr.struct ? stride : ptr.elem === 7 ? ptr.count * stride : Math.ceil(ptr.count * ELEM_BITS[ptr.elem] / 8);
    if (!(ptr.at >= 0 && ptr.at + size <= bytes.length)) throw new RangeError(`pointer at ${p} out of the message`);
    return ptr;
  }

  // a scalar at p, zero past the end of its struct's data
  function scalar(type, p, end, def) {
    const v = p + SIZE[type] > end ? read[type](ZERO, 0) : read[type](dv, p);
    return def ? xorDefault(type, v, def) : v;
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

  function list([type, ref], {at, elem, count, data, ptrs}) {
    const each = (fn) => Array.from({length: count}, (_, j) => fn(j));
    if (elem === 7) return type === "s" ? each((j) => struct(ref, at + j * (data + ptrs * 8), data, ptrs)) : [];
    const size = ELEM_BITS[elem] >> 3;
    if (type === "b") return elem === 1 ? each((j) => !!((dv.getUint8(at + (j >> 3)) >> (j & 7)) & 1)) : [];
    if (type === "s") return each((j) => struct(ref, at + j * size, elem === 6 ? 0 : size, elem === 6 ? 1 : 0));
    if (!SIZE[type]) return elem === 6 ? each((j) => value(type, ref, at + j * 8)) : [];
    if (SIZE[type] !== size) return [];
    return each((j) => type === "e" ? enumName(ref, READ.e(dv, at + j * 2)) : read[type](dv, at + j * size));
  }

  const ev = pointer(root(0));
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
  views(buf);
  for (const [m, size] of messages(buf)) {
    // logMonoTime is the Event's first word, then the union tag
    const ev = follow(root(m));
    const name = ev?.struct && event.disc * 2 < ev.data && names[dv.getUint16(ev.at + event.disc * 2, true)];
    if (!name) continue;
    if (!out.has(name)) out.set(name, {mono: [], at: [], size: []});
    const s = out.get(name);
    s.mono.push(READ.u64(dv, ev.at));
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
