// DBC files: a port of cabana's dbcfile.cc parser plus an exact round trip. The source is kept as a list of
// statements; serializing emits each statement's original text unless the model it describes has changed, so
// unknown sections, formatting and order survive, and edits only rewrite what they touch.

export const NODE = "XXX";
export const NORMAL = 0, MUXED = 1, MUX = 2;  // signal types, ordered as in cabana

const BO = /^BO_ (\w+) (\w+) *: (\w+) (\w+)/;
const SG = /^SG_ (\w+)(?: +(\w+))? *: ([0-9]+)\|([0-9]+)@([0-9]+)([+-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] "(.*)" (.*)/;
const VAL = /^VAL_ (\w+) (\w+) (.*)/;
const VAL_ENTRY = /([+-]?[0-9]+(?:\.[0-9]+)?)\s+"([^"]*)"/g;
const CM = /^CM_ (BO_|SG_)\s*(\d+)\s*(\S*)/;

// printf("%.15g"), as cabana writes numbers
export function num(v) {
  if (v === 0 || !Number.isFinite(v)) return String(v);
  const exp = +v.toExponential(14).split("e")[1];
  if (exp < -4 || exp >= 15) {
    const [m, e] = v.toExponential(14).split("e");
    return `${m.replace(/\.?0+$/, "")}e${e[0]}${e.slice(1).padStart(2, "0")}`;
  }
  return v.toFixed(14 - exp).replace(/\.?0+$/, "");
}

// a bit position's mirror within its byte
export const flip = (b) => 8 * (b >> 3) + 7 - (b & 7);
const quote = (s) => s.replaceAll('"', '\\"');

function commentComplete(s) {
  for (let i = 0, esc = false; i < s.length; i++) {
    const c = s[i];
    if (c === "\\" && !esc) { esc = true; continue; }
    if (c === '"' && !esc && /^\s*;/.test(s.slice(i + 1))) return true;
    esc = false;
  }
  return false;
}

export function newSignal(props) {
  return {name: "", type: NORMAL, muxValue: 0, start: 0, size: 1, le: true, signed: false, factor: 1, offset: 0,
          min: 0, max: 1, unit: "", receiver: NODE, comment: "", vals: [], ...props};
}

// a signal's most and least significant bit positions
export function bits(s) {
  s.lsb = s.le ? s.start : flip(flip(s.start) + s.size - 1);
  s.msb = s.le ? s.start + s.size - 1 : s.start;
  return s;
}

// derived fields: msb/lsb, color, display precision, the byte mask, and the multiplexor link
export function updateMsg(m) {
  m.transmitter ||= NODE;
  m.sigs.sort((l, r) => r.type - l.type || l.muxValue - r.muxValue || l.start - r.start || (l.name < r.name ? -1 : l.name > r.name ? 1 : 0));
  m.mask = new Uint8Array(m.size);
  m.mux = m.sigs.find((s) => s.type === MUX) || null;
  for (const s of m.sigs) {
    s.receiver ||= NODE;
    bits(s);
    s.precision = Math.max(decimals(s.factor), decimals(s.offset));
    s.color = color(s);
    for (let i = s.msb >> 3, bits = s.size; i >= 0 && i < m.size && bits > 0; i += s.le ? -1 : 1) {
      const lo = s.lsb >> 3 === i ? s.lsb : i * 8, hi = s.msb >> 3 === i ? s.msb : i * 8 + 7;
      m.mask[i] |= ((1 << (hi - lo + 1)) - 1) << (lo - i * 8);
      bits -= hi - lo + 1;
    }
  }
  for (const s of m.sigs) {
    if (s.type === MUXED && !m.mux) s.type = NORMAL;
    if (s.type !== MUXED) s.muxValue = 0;
  }
  return m;
}

function decimals(v) {
  let d = 0;
  while (d < 6 && Math.abs(v - Math.round(v)) > 1e-9) { v *= 10; d++; }
  return d;
}

function color(s) {
  let hash = 0;
  for (let i = 0; i < s.name.length; i++) hash = Math.imul(hash, 31) + s.name.charCodeAt(i) | 0;
  const h = (19 * s.lsb / 64) % 1, sat = 0.25 + 0.25 * (hash & 0xff) / 255, v = 0.75 + 0.25 * ((hash >>> 8) & 0xff) / 255;
  const l = v * (1 - sat / 2), sl = l === 0 || l === 1 ? 0 : (v - l) / Math.min(l, 1 - l);
  return `hsl(${(h * 360).toFixed(1)} ${(sl * 100).toFixed(1)}% ${(l * 100).toFixed(1)}%)`;
}

// the raw (unscaled) value of a signal in d[o, o + n), as cabana's get_raw_value
export function raw(s, d, o = 0, n = d.length - o) {
  const mb = s.msb >> 3, lb = s.lsb >> 3;
  if (mb >= n) return 0;
  let v;
  if (mb === lb) v = (d[o + mb] >> (s.lsb & 7)) & ((1 << s.size) - 1);
  else {
    v = 0;
    let bits = s.size;
    for (let i = mb; i >= 0 && i < n && bits > 0; i += s.le ? -1 : 1) {
      const lo = i === lb ? s.lsb & 7 : 0, nb = (i === mb ? s.msb & 7 : 7) - lo + 1;
      v = v * (1 << nb) + ((d[o + i] >> lo) & ((1 << nb) - 1));
      bits -= nb;
    }
    // a signal running past the end of the frame keeps its high bits in place, as opendbc's get_raw_value
    v *= 2 ** bits;
  }
  return s.signed && v >= 2 ** (s.size - 1) ? v - 2 ** s.size : v;
}

// the physical value, or NaN when a multiplexed signal is not selected
export function value(m, s, d, o = 0, n = d.length - o) {
  if (s.type === MUXED && m.mux && raw(m.mux, d, o, n) !== s.muxValue) return NaN;
  return raw(s, d, o, n) * s.factor + s.offset;
}

// writes a physical value into d
export function encode(s, d, v) {
  let r = Math.round((v - s.offset) / s.factor);
  if (r < 0) r += 2 ** s.size;
  const lb = s.lsb >> 3;
  for (let i = lb, bits = 0; i >= 0 && i < d.length && bits < s.size; i += s.le ? 1 : -1) {
    const lo = i === lb ? s.lsb & 7 : 0, nb = Math.min(8 - lo, s.size - bits), mask = ((1 << nb) - 1) << lo;
    d[i] = (d[i] & ~mask) | ((Math.floor(r / 2 ** bits) << lo) & mask);
    bits += nb;
  }
}

export function format(s, v, unit = true) {
  const r = Math.round((v - s.offset) / s.factor);
  const desc = s.vals.find(([x]) => Math.abs(r - x) < 1e-6);
  if (desc) return desc[1];
  return v.toFixed(s.precision) + (unit && s.unit ? ` ${s.unit}` : "");
}

export class DBC {
  constructor(name = "untitled", text = "") {
    this.name = name;
    this.text = text;
    this.msgs = new Map();
    this.items = [];  // source statements: {text} or {key, text, canon}
    this.parse(text);
  }

  // a DBC parsed elsewhere (the CAN worker): its messages arrive as a structured clone, shared references intact. The
  // source statements stay behind, as cloning them costs a frame; serialize() parses the text again for them
  static revive(name, text, msgs) { return Object.assign(Object.create(DBC.prototype), {name, text, msgs}); }

  msg(address) { return this.msgs.get(address); }
  msgByName(name) { for (const m of this.msgs.values()) if (m.name === name) return m; }
  get empty() { return this.msgs.size === 0; }

  parse(text) {
    const lines = text.match(/[^\n]*\n|[^\n]+$/g) || [];
    let msg = null, block = null, muxCount = 0;
    const keyed = [];
    for (let n = 0; n < lines.length; n++) {
      let src = lines[n], line = src.trimStart().replace(/\r?\n$/, "");
      if (/^CM_ (BO_|SG_ )/.test(line) && !commentComplete(line)) {
        while (n + 1 < lines.length && !commentComplete(line)) {
          src += lines[++n];
          line += "\n" + lines[n].replace(/\r?\n$/, "");
        }
      }
      try {
        if (line.startsWith("BO_ ")) {
          const [, addr, name, size, tx] = BO.exec(line) || fail("Invalid BO_ line format");
          const address = +addr;
          if (this.msgs.has(address)) fail(`Duplicate message address: ${address}`);
          msg = {address, name, size: +size, transmitter: tx.trim(), comment: "", sigs: []};
          this.msgs.set(address, msg);
          muxCount = 0;
          this.items.push(block = {key: `BO_ ${address}`, text: src});
          keyed.push(block);
          continue;
        }
        if (line.startsWith("SG_ ")) {
          if (!msg) fail("No Message");
          const g = SG.exec(line) || fail("Invalid SG_ line format");
          if (msg.sigs.some((s) => s.name === g[1])) fail("Duplicate signal name");
          let type = NORMAL, muxValue = 0;
          if (g[2] === "M") {
            if (++muxCount >= 2) fail("Multiple multiplexor");
            type = MUX;
          } else if (g[2]) {
            type = MUXED;
            muxValue = g[2].length > 1 ? parseInt(g[2].slice(1)) || 0 : 0;
          }
          msg.sigs.push(newSignal({name: g[1], type, muxValue, start: +g[3], size: +g[4], le: g[5] === "1", signed: g[6] === "-",
                                   factor: parseFloat(g[7]), offset: parseFloat(g[8]), min: parseFloat(g[9]), max: parseFloat(g[10]),
                                   unit: g[11], receiver: g[12].trim()}));
          // the signal joins its message's block, along with any unknown lines between them
          const i = this.items.lastIndexOf(block);
          if (this.items.slice(i + 1).every((it) => !it.key)) block.text += this.items.splice(i + 1).map((it) => it.text).join("") + src;
          else this.items.push({text: src});
          continue;
        }
        if (line.startsWith("VAL_ ")) {
          const [, addr, name, entries] = VAL.exec(line) || fail("invalid VAL_ line format");
          const s = this.msgs.get(+addr)?.sigs.find((x) => x.name === name);
          if (s) {
            for (const [, v, d] of entries.matchAll(VAL_ENTRY)) s.vals.push([parseFloat(v), d.trim()]);
            keyed.push(this.items[this.items.push({key: `VAL_ ${+addr} ${name}`, text: src}) - 1]);
            continue;
          }
        } else if (line.startsWith("CM_ BO_") || line.startsWith("CM_ SG_ ")) {
          const [, kind, addr, word] = CM.exec(line) || [], q0 = line.indexOf('"'), q1 = line.lastIndexOf('"');
          const sig = kind === "SG_" ? word : "";
          if (!kind || (kind === "SG_" && !sig) || q1 <= q0) fail(kind === "BO_" ? "Invalid message comment format" : "Invalid CM_ SG_ line format");
          const m = this.msgs.get(+addr), target = sig ? m?.sigs.find((s) => s.name === sig) : m;
          if (target) {
            target.comment = line.slice(q0 + 1, q1).replaceAll('\\"', '"').trim();
            keyed.push(this.items[this.items.push({key: `CM_ ${kind} ${+addr}${sig && " " + sig}`, text: src}) - 1]);
            continue;
          }
        }
      } catch (e) {
        throw new Error(`[${this.name}:${n + 1}] ${e.message}: ${line}`);
      }
      this.items.push({text: src});
    }
    for (const m of this.msgs.values()) updateMsg(m);
    const gen = this.generate();
    for (const it of keyed) it.canon = gen.get(it.key);
  }

  // canonical text of every statement in the model, keyed and in cabana's order: messages, comments, values
  generate() {
    const bo = new Map(), cm = new Map(), val = new Map();
    for (const [address, m] of [...this.msgs].sort((a, b) => a[0] - b[0])) {
      // a fraction would write a file that doesn't parse again
      if (![m.size, ...m.sigs.flatMap((s) => [s.start, s.size, s.muxValue])].every(Number.isInteger)) fail(`${m.name} has a size or start bit that isn't a whole number`);
      let text = `BO_ ${address} ${m.name}: ${m.size} ${m.transmitter || NODE}\n`;
      if (m.comment) cm.set(`CM_ BO_ ${address}`, `CM_ BO_ ${address} "${quote(m.comment)}";\n`);
      for (const s of m.sigs) {
        const mux = s.type === MUX ? "M " : s.type === MUXED ? `m${s.muxValue} ` : "";
        text += ` SG_ ${s.name} ${mux}: ${s.start}|${s.size}@${s.le ? 1 : 0}${s.signed ? "-" : "+"} (${num(s.factor)},${num(s.offset)}) ` +
                `[${num(s.min)}|${num(s.max)}] "${s.unit}" ${s.receiver || NODE}\n`;
        if (s.comment) cm.set(`CM_ SG_ ${address} ${s.name}`, `CM_ SG_ ${address} ${s.name} "${quote(s.comment)}";\n`);
        if (s.vals.length) val.set(`VAL_ ${address} ${s.name}`, `VAL_ ${address} ${s.name} ${s.vals.map(([v, d]) => `${num(v)} "${d}"`).join(" ")};\n`);
      }
      bo.set(`BO_ ${address}`, text + "\n");
    }
    return new Map([...bo, ...cm, ...val]);
  }

  // original text for unchanged statements; changed ones are regenerated in place, new ones join their kind
  serialize() {
    const items = this.items || new DBC(this.name, this.text).items, gen = this.generate(), used = new Set(), after = new Map();
    const kind = (key) => key.slice(0, 3), addr = (key) => key.split(" ")[kind(key) === "CM_" ? 2 : 1];
    const out = items.map((it) => {
      if (!it.key) return it.text;
      const g = gen.get(it.key);
      if (g === it.canon) {
        used.add(it.key);
        return it.text;
      }
      if (g === undefined || used.has(it.key)) return "";
      used.add(it.key);
      // blocks are generated with their trailing blank line; the source keeps it as a separate statement
      return kind(it.key) === "BO_" ? g.slice(0, -1) : g;
    });
    for (const [key, text] of gen) {
      if (used.has(key)) continue;
      const live = (it) => it.key && kind(it.key) === kind(key) && gen.has(it.key);
      let at = items.findLastIndex((it) => live(it) && addr(it.key) === addr(key));
      if (at < 0) at = items.findLastIndex(live);
      // comments and values must follow their message's definition
      if (kind(key) !== "BO_") {
        const bo = items.findLastIndex((it) => it.key === `BO_ ${addr(key)}`);
        at = Math.max(at, bo >= 0 ? bo : items.findLastIndex((it) => it.key?.startsWith("BO_ ")));
      }
      after.set(at, (after.get(at) || "") + (at >= 0 && kind(key) === "BO_" ? "\n" + text.slice(0, -1) : text));
    }
    // a statement appended after an unterminated last line starts its own line
    const parts = [];
    const add = (t) => t && parts.push(parts.length && !parts.at(-1).endsWith("\n") ? "\n" + t : t);
    out.forEach((t, i) => { add(t); add(after.get(i)); });
    add(after.get(-1));
    return parts.join("");
  }

  // edits replace a whole message (or remove it with null); the undo stack records both sides
  setMsg(address, m) {
    if (m) this.msgs.set(address, updateMsg(m));
    else this.msgs.delete(address);
  }
}

function fail(msg) { throw new Error(msg); }

export const cloneMsg = (m) => m && {...m, mask: undefined, mux: undefined, sigs: m.sigs.map((s) => ({...s, vals: s.vals.map((v) => [...v])}))};

export function newSignalName(m) {
  for (let i = 1; ; i++) if (!m.sigs.some((s) => s.name === `NEW_SIGNAL_${i}`)) return `NEW_SIGNAL_${i}`;
}

// a DBC per bus, falling back to the "all buses" file, with one undo stack for every edit. A file is clean while its
// last applied edit is the one it was saved (or opened) at
export class Dbcs {
  constructor() {
    this.files = new Map();  // bus (-1 = all) -> DBC
    this.undos = [];
    this.index = 0;
    this.clean = new Map();  // DBC -> its last applied edit when saved
    this.listeners = new Set();
  }

  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit(what) { for (const fn of this.listeners) fn(what); }

  // the files that opening a DBC on `buses` unloads
  replaced(buses) { return this.all().filter((f) => this.buses(f).every((b) => buses.includes(b))); }
  open(buses, dbc) {
    const gone = this.replaced(buses);
    for (const b of buses) this.files.set(b, dbc);
    for (const f of gone) this.forget(f);
    this.emit("file");
  }
  close(dbc) {
    for (const [b, f] of this.files) if (f === dbc) this.files.delete(b);
    this.forget(dbc);
    this.emit("file");
  }
  // an unloaded file's edits leave the undo stack
  forget(dbc) {
    this.index -= this.undos.slice(0, this.index).filter((c) => c.dbc === dbc).length;
    this.undos = this.undos.filter((c) => c.dbc !== dbc);
    this.clean.delete(dbc);
  }
  file(bus) { return this.files.get(bus) || this.files.get(-1); }
  buses(dbc) { return [...this.files].filter(([, f]) => f === dbc).map(([b]) => b); }
  all() { return [...new Set(this.files.values())]; }
  msg(bus, address) { return this.file(bus)?.msg(address); }

  // replaces message `address` on `bus` with m (null removes it); undoable
  edit(bus, address, m, text) {
    const dbc = this.file(bus) || this.ensure();
    this.undos.length = this.index;
    const cmd = {dbc, address, before: cloneMsg(dbc.msg(address)), after: cloneMsg(m), text};
    this.undos.push(cmd);
    this.redo();
  }
  ensure() {
    const dbc = new DBC();
    this.files.set(-1, dbc);
    return dbc;
  }
  undo() { if (this.index > 0) this.apply(this.undos[--this.index], "before"); }
  redo() { if (this.index < this.undos.length) this.apply(this.undos[this.index++], "after"); }
  apply(cmd, side) {
    cmd.dbc.setMsg(cmd.address, cloneMsg(cmd[side]));
    this.emit("msg");
  }
  last(dbc) { return this.undos.slice(0, this.index).findLast((c) => c.dbc === dbc) || null; }
  isDirty(dbc) { return this.last(dbc) !== (this.clean.get(dbc) || null); }
  get dirty() { return this.all().some((f) => this.isDirty(f)); }
  saved(dbc) { this.clean.set(dbc, this.last(dbc)); this.emit("saved"); }
}
