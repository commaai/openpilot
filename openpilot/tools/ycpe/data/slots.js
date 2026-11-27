// the worker side of a live session: the events received so far, kept by slot (SEGMENT seconds of route time from the
// first event, like a route's segments). The open slot is parsed again whenever more arrives, at most every few times
// its last parse took, so a fast source costs a bounded share of the worker; each parse is the slot's whole pack, as
// for a segment of a log. Closed slots keep their bytes for re-reads (lists asked for later), the inspector and saving.
import {messages} from "./message.js";
import {flatten} from "./flatten.js";

const MIN_MS = 100;  // between parses of the open slot
const SLOWDOWN = 2;  // and at least this many times the last one took

export class Slots {
  // span: seconds per slot; schema: the bundled one, for JSON messages; parse(bytes, opts, more): the worker's segment
  // parse; post(pack, transfer) sends a slot's pack to the store
  constructor({span, schema, parse, post}) {
    this.span = span;
    this.schema = schema;
    this.parse = parse;
    this.post = post;
    this.slots = new Map();  // n → {bytes, len, json: Map(service → [{t, valid, data}]), dirty}
    this.base = null;  // logMonoTime (s) of the first event: route time 0
    this.open = -1;
    this.context = undefined;
    this.keep = {};
    this.timer = 0;
    this.busy = false;
    this.ran = undefined;  // when the last parse started
    this.last = 0;  // ms the last parse took
  }

  slot(t) {
    this.base ??= t;
    const n = Math.max(this.open, Math.floor((t - this.base) / this.span));  // late arrivals join the open slot
    if (n > this.open) {
      const last = this.slots.get(this.open);
      if (last) last.closing = last.dirty = true;  // its final parse says so
      this.open = n;
    }
    if (!this.slots.has(n)) this.slots.set(n, {bytes: new Uint8Array(1 << 16), len: 0, json: new Map(), dirty: false});
    return this.slots.get(n);
  }

  // whole events in rlog framing; each run of them going to one slot is copied in one piece
  events(buf) {
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    let to = null, run = 0, end = 0;
    for (const [m, size] of messages(buf)) {
      // the Event is the first object of segment 0, and logMonoTime its first word
      const root = m + ((4 + 4 * (dv.getUint32(m, true) + 1) + 7) & ~7), ev = root + 8 + (dv.getUint32(root, true) >> 2) * 8;
      const s = this.slot((dv.getUint32(ev, true) + dv.getUint32(ev + 4, true) * 4294967296) / 1e9);
      if (s !== to) {
        if (to) append(to, buf.subarray(run, m));
        [to, run] = [s, m];
        s.dirty = true;
      }
      end = m + size;
    }
    if (to) append(to, buf.subarray(run, end));
    this.schedule();
  }

  // [service, logMonoTime (ns), valid, data] messages as webrtcd sends them
  json(msgs) {
    for (const [service, mono, valid, data] of msgs) {
      const s = this.slot(mono / 1e9), list = s.json.get(service) || s.json.set(service, []).get(service);
      list.push({t: mono / 1e9, valid, data});
      s.dirty = true;
    }
    this.schedule();
  }

  schedule() {
    if (this.timer || this.busy) return;
    const wait = this.ran === undefined ? 0 : Math.max(MIN_MS, SLOWDOWN * this.last) - (performance.now() - this.ran);
    this.timer = setTimeout(() => this.run(), Math.max(0, wait));
  }

  // parses each slot with news, oldest first, so a slot that closed meanwhile gets its final parse
  async run() {
    this.timer = 0;
    this.busy = true;
    this.ran = performance.now();
    for (const [n, s] of this.slots) {
      if (!s.dirty) continue;
      s.dirty = false;
      const closed = !!s.closing;
      if (closed) {
        s.closing = false;
        s.bytes = s.bytes.slice(0, s.len);  // no room to grow anymore
      }
      try {
        const p = await this.read(n);
        this.context ??= p.context && Object.keys(p.context).length ? p.context : undefined;
        this.post({...p, n, closed, held: this.held(), size: s.bytes.byteLength}, [...p.bufs, p.recs.buffer, ...p.canBufs]);
      } catch (e) {
        this.post({n, error: e?.message ?? String(e)}, []);
      }
    }
    this.last = performance.now() - this.ran;
    this.busy = false;
    if ([...this.slots.values()].some((s) => s.dirty)) this.schedule();
  }

  // slot n's pack: every column, or the ones `only` picks when asked for again; can: with its CAN frames
  read(n, only, can = !only) {
    const s = this.slots.get(n);
    if (!s) return Promise.reject(new Error(`slot ${n} is gone`));
    const more = [];
    for (const [name, msgs] of s.json) {
      const svc = flatten(this.schema, name, msgs);
      if (svc) more.push(svc);
    }
    return this.parse(s.bytes.subarray(0, s.len), {n, shift: 0, base: this.base, keep: this.keep, context: this.context, only, can}, more);
  }

  // the bytes held, for the session's memory budget
  held() {
    let b = 0;
    for (const s of this.slots.values()) b += s.bytes.byteLength;
    return b;
  }

  bytes(n) {
    const s = this.slots.get(n);
    return s ? s.bytes.subarray(0, s.len) : new Uint8Array(0);
  }

  drop(n) {
    this.slots.delete(n);
  }

  // every event received, in order, as one rlog
  save() {
    return concat([...this.slots.keys()].sort((a, b) => a - b).map((n) => this.bytes(n)));
  }
}

// adds bytes to a growing buffer {bytes, len} (a slot, or the page's batch of events), doubling it as needed
export function append(s, bytes) {
  if (s.len + bytes.length > s.bytes.length) {
    const grown = new Uint8Array(Math.max(s.len + bytes.length, s.bytes.length * 2));
    grown.set(s.bytes.subarray(0, s.len));
    s.bytes = grown;
  }
  s.bytes.set(bytes, s.len);
  s.len += bytes.length;
}

// the byte arrays one after another in a new one
export function concat(chunks) {
  const out = new Uint8Array(chunks.reduce((a, c) => a + c.length, 0));
  for (let k = 0, at = 0; k < chunks.length; at += chunks[k++].length) out.set(chunks[k], at);
  return out;
}
