// compact() must be lossless: every compacted column expands back to the reader's dense column
// usage: node data/bench/compact.mjs [rlog]
import fs from "node:fs";
import {logArg, schema} from "./node.mjs";
import {decompress} from "../../codec/codec.js";
import {Reader, compact} from "../capnp.js";

const reader = new Reader(schema);
const buf = decompress(new Uint8Array(fs.readFileSync(logArg())));
const raw = reader.read(buf);
const dense = new Map();
for (const s of raw.values()) { for (const [p, c] of s.cols) dense.set(s.name + p, Array.from(c.v)); dense.set(s.name + (s.cols.has("/valid") ? "/_valid" : "/valid"), Array.from(s.valid)); }
const {services} = compact(reader.read(buf));
let bad = 0, n = 0;
for (const [name, s] of Object.entries(services)) for (const [p, c] of Object.entries(s.cols)) {
  n++;
  const want = dense.get(name + p), got = new Array(s.t.length).fill(c.type === "t" ? null : NaN);
  if ("c" in c) got.fill(c.c);
  else if (c.i) c.i.forEach((k, j) => got[k] = c.v[j]);
  else for (let k = 0; k < got.length; k++) got[k] = c.v[k];
  for (let k = 0; k < got.length; k++) if (!Object.is(got[k], want[k]) && !(got[k] === 0 && want[k] === 0)) { bad++; console.log(name + p, k, got[k], want[k]); break; }
}
console.log(`${n} cols, ${bad} mismatched`);
