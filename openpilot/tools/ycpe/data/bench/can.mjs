// node data/bench/can.mjs [rlog.raw]: CAN extraction throughput, memory per segment and worker scan timings
import {readFileSync} from "node:fs";
import {Reader} from "../capnp.js";
import {CanFrames} from "../can.js";

const here = new URL("..", import.meta.url).pathname;
const raw = new Uint8Array(readFileSync(process.argv[2] || "/tmp/aio-probe/rlog.raw"));
const schema = JSON.parse(readFileSync(`${here}schema.json`, "utf8"));
const reader = new Reader(schema);
const ms = (f) => { const t = performance.now(); const r = f(); return [r, performance.now() - t]; };

const [, plain] = ms(() => reader.read(raw));
const frames = new CanFrames(schema);
const [, hooked] = ms(() => reader.read(raw, {can: frames.hook}));
const [{can}, grouped] = ms(() => frames.finish());
const n = can.t.length, bytes = [can.t, can.len, can.data, can.keys, can.start, can.stride, can.dstart].reduce((b, a) => b + a.byteLength, 0);
console.log(`${n} frames in ${can.keys.length} messages`);
console.log(`read without hook ${plain.toFixed(0)} ms, with hook ${hooked.toFixed(0)} ms, grouping ${grouped.toFixed(0)} ms`);
console.log(`extraction ${(n / (hooked - plain + grouped) / 1e3).toFixed(1)} M frames/s, stored ${(bytes / 1e6).toFixed(2)} MB per segment (${(bytes / n).toFixed(1)} B/frame)`);

// the worker side, run in-process
await import("../can.js?worker");
let reply;
globalThis.postMessage = (m) => { reply = m; };
const call = (op, args) => { onmessage({data: {id: 0, op, ...args}}); if (reply.error) throw new Error(reply.error); return reply.r; };
const [, added] = ms(() => call("add", {n: 0, can, off: 0}));
console.log(`worker ingest (flip summaries) ${added.toFixed(0)} ms`);
const t = can.t0 + 30, k = can.keys[can.keys.length >> 1];
for (const [op, args] of [["state", {t}], ["heatmap", {k, t0: can.t0 + 10, t1: can.t0 + 40}], ["frames", {k, t, n: 200}],
                          ["similar", {k, byte: 0, bit: 0, bus: 0, equal: true, min: 0}]]) {
  const [, d] = ms(() => call(op, args));
  console.log(`${op} ${d.toFixed(1)} ms`);
}

