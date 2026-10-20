// capnp Event writer for live sources that have values rather than log bytes (panda CAN frames, panda health).
// Driven by schema.json, like the reader: event(logMonoTime, valid, "can", [{address, dat, src}]) returns the
// event in rlog framing. Only the fields given are written; nested structs are objects, lists are arrays.
import schema from "../schema.json" with {type: "json"};

const SIZE = {b: 0, i8: 1, u8: 1, i16: 2, u16: 2, e: 2, i32: 4, u32: 4, f32: 4, i64: 8, u64: 8, f64: 8};
const LIST_SIZE = {1: 2, 2: 3, 4: 4, 8: 5};  // capnp list element size codes by byte size (bools are 1)
const SET = {i8: "setInt8", u8: "setUint8", i16: "setInt16", u16: "setUint16", e: "setUint16", i32: "setInt32", u32: "setUint32",
             f32: "setFloat32", f64: "setFloat64"};

const layouts = new Map();
const layout = (i) => layouts.get(i) || layouts.set(i, {...schema.structs[i], byName: new Map(schema.structs[i].fields.map((f) => [f[0], f]))}).get(i);
const encoder = new TextEncoder();

// nanoseconds on the page's monotonic clock, as logMonoTime
export const monoNanos = (ms = performance.now()) => Math.round(ms * 1e6);

// non-zero defaults are stored XORed (see capnp.js); integer ones are all a writer has needed so far
function xorDefault(type, v, def) {
  if (SIZE[type] > 4 || type[0] === "f") throw new Error(`can't write a ${type} with a default`);
  return type === "u32" ? (v ^ def) >>> 0 : v ^ def;
}

class Writer {
  constructor(bytes) {
    this.grow(bytes);
    this.end = 8;  // the rlog framing's segment table: one segment
  }

  grow(bytes) {
    const u8 = new Uint8Array(bytes);
    if (this.u8) u8.set(this.u8.subarray(0, this.end));
    this.u8 = u8;
    this.dv = new DataView(u8.buffer);
  }

  // words of zeroes; returns their byte offset
  alloc(words) {
    const at = this.end;
    this.end += words * 8;
    if (this.end > this.u8.length) this.grow(Math.max(this.end, this.u8.length * 2));
    return at;
  }

  u64(at, v) {
    const hi = Math.floor(v / 4294967296);
    this.dv.setUint32(at, v - hi * 4294967296, true);
    this.dv.setUint32(at + 4, hi, true);
  }

  pointer(at, target, kind, hi) {
    this.dv.setInt32(at, (((target - at - 8) / 8) << 2) | kind, true);
    this.dv.setUint32(at + 4, hi, true);
  }

  struct(at, idx, value) {
    const s = layout(idx), off = this.alloc(s.data + s.ptrs);
    this.pointer(at, off, 0, s.data | s.ptrs << 16);
    this.fields(off, s, value);
  }

  fields(off, s, value) {
    const ptrs = off + s.data * 8;
    for (const name in value) {
      const v = value[name], f = s.byName.get(name);
      if (!f) throw new Error(`${s.name} has no field ${name}`);
      const [, disc, type, pos, ref, def] = f;
      if (disc >= 0) this.dv.setUint16(off + s.disc * 2, disc, true);
      if (type === "b") {
        if (!v !== !def) this.u8[off + (pos >> 3)] |= 1 << (pos & 7);
      } else if (type in SIZE) {
        this.scalar(off + pos * SIZE[type], type, def ? xorDefault(type, v, def) : v);
      } else if (type === "s") {
        this.struct(ptrs + pos * 8, ref, v);
      } else if (type === "g") {
        this.fields(off, layout(ref), v);
      } else if (type === "d" || type === "t") {
        const bytes = type === "t" ? encoder.encode(v + "\0") : v, at = this.alloc((bytes.length + 7) >> 3);
        this.u8.set(bytes, at);
        this.pointer(ptrs + pos * 8, at, 1, 2 | bytes.length << 3);
      } else if (type === "l") {
        this.list(ptrs + pos * 8, ref, v);
      }
    }
  }

  list(at, [type, ref], items) {
    const n = items.length;
    if (type === "s") {
      const s = layout(ref), step = s.data + s.ptrs, off = this.alloc(1 + n * step);
      this.pointer(at, off, 1, 7 | (n * step) << 3);
      this.dv.setUint32(off, n << 2, true);  // the tag: element count, and each element's size
      this.dv.setUint32(off + 4, s.data | s.ptrs << 16, true);
      for (let i = 0; i < n; i++) this.fields(off + 8 + i * step * 8, s, items[i]);
    } else if (type === "b") {
      const off = this.alloc((n + 63) >> 6);
      this.pointer(at, off, 1, 1 | n << 3);
      for (let i = 0; i < n; i++) if (items[i]) this.u8[off + (i >> 3)] |= 1 << (i & 7);
    } else if (type in SIZE) {
      const size = SIZE[type], off = this.alloc((n * size + 7) >> 3);
      this.pointer(at, off, 1, LIST_SIZE[size] | n << 3);
      for (let i = 0; i < n; i++) this.scalar(off + i * size, type, items[i]);
    } else {
      throw new Error(`can't write a list of ${type}`);
    }
  }

  scalar(at, type, v) {
    if (type === "i64" || type === "u64") this.u64(at, v);
    else this.dv[SET[type]](at, v, true);
  }
}

// one Event in rlog framing (segment table, then the message), as a Uint8Array of its own
export function event(logMonoTime, valid, which, value) {
  const w = new Writer(1024);
  const root = w.alloc(1);
  w.struct(root, 0, {logMonoTime, valid, [which]: value});
  w.dv.setUint32(4, (w.end - 8) / 8, true);
  return w.u8.slice(0, w.end);
}
