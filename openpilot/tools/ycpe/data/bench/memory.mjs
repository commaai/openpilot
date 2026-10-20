// bytes per segment after compact(), by column kind: node data/bench/memory.mjs [rlog]
import fs from "node:fs";
import {logArg, schema} from "./node.mjs";
import {decompress} from "../../codec/codec.js";
import {Reader, compact} from "../capnp.js";

const {services} = compact(new Reader(schema).read(decompress(new Uint8Array(fs.readFileSync(logArg())))));

// strings are counted as their characters plus a 16-byte header, shared strings once
const seen = new Set();
const size = (v) => {
  if (!v) return 0;
  if (ArrayBuffer.isView(v)) return v.byteLength;
  let b = v.length * 8;
  for (const s of v) if (typeof s === "string" && !seen.has(s)) { seen.add(s); b += 16 + s.length; }
  return b;
};
const kinds = {};
const add = (k, b) => { kinds[k] ||= [0, 0]; kinds[k][0]++; kinds[k][1] += b; };
const bySvc = [];
let total = 0;
for (const [name, s] of Object.entries(services)) {
  let b = s.t.byteLength;
  add("t", b);
  for (const c of Object.values(s.cols)) {
    const cb = size(c.v) + size(c.i);
    const kind = "c" in c ? "const" : `${c.type === "t" ? "text" : c.v.constructor.name}${c.i ? " sparse" : ""}`;
    add(kind, cb);
    b += cb;
  }
  bySvc.push([name, +(b / 1e6).toFixed(3)]);
  total += b;
}
console.log(`total ${(total / 1e6).toFixed(2)} MB`);
console.table(Object.fromEntries(Object.entries(kinds).sort((a, b) => b[1][1] - a[1][1]).map(([k, [n, b]]) => [k, {cols: n, MB: +(b / 1e6).toFixed(3)}])));
console.log(bySvc.sort((a, b) => b[1] - a[1]).slice(0, 10));
