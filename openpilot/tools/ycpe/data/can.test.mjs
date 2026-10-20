// node --test data/can.test.mjs: the CAN worker's scans over hand-made segments
import assert from "node:assert/strict";
import {test} from "node:test";
import {key} from "./can.js";

await import("./can.js?worker");
let reply;
globalThis.postMessage = (m) => { reply = m; };
const call = (op, args = {}) => {
  onmessage({data: {id: 0, op, ...args}});
  if (reply.error) throw new Error(reply.error);
  return reply.r;
};

// a CanFrames payload for one message from [time, bytes] frames
function segment(k, frames) {
  const stride = Math.max(...frames.map(([, d]) => d.length)), n = frames.length;
  const data = new Uint8Array(n * stride);
  frames.forEach(([, d], i) => data.set(d, i * stride));
  return {t0: 0, keys: Float64Array.of(k), start: Uint32Array.of(0, n), stride: Uint8Array.of(stride), dstart: Uint32Array.of(0, n * stride),
          t: Float32Array.from(frames, ([t]) => t), len: Uint8Array.from(frames, ([, d]) => d.length), data};
}

const A = key(0, 0x100), B = key(1, 0x200);
const ramp = Array.from({length: 256}, (_, v) => [v * 0.01, [v, 42, v >> 4]]);
call("add", {n: 0, can: segment(A, ramp), off: 0});

test("heatmap counts as cabana", () => {
  const bits = call("heatmap", {k: A, t0: 0, t1: 10});
  assert.deepEqual([...bits.slice(0, 8)], [1, 3, 7, 15, 31, 63, 127, 255]);
  assert.deepEqual([...bits.slice(8, 16)], Array(8).fill(0));
  // a range starts from a baseline, and a shorter frame only compares what it has
  call("add", {n: 1, can: segment(B, [[0, [255, 42, 15]], [0.1, [0]], [0.2, [255, 42, 15]]]), off: 0});
  assert.deepEqual([...call("heatmap", {k: B, t0: 0, t1: 1}).slice(0, 8)], Array(8).fill(2));
  // partial ranges scan frames
  assert.equal(call("heatmap", {k: A, t0: 0, t1: 0.015})[7], 1);
});

test("state at t", () => {
  const s = call("state", {t: 1.005});
  const j = [...s.keys].indexOf(A);
  assert.equal(s.count[j], 101);
  assert.deepEqual([...s.data.slice(s.off[j], s.off[j + 1])], [100, 42, 6]);
  assert.equal(s.kind[s.off[j]], 1);  // byte 0 counts up
  assert.ok(s.age[s.off[j]] < 0.01);
  assert.equal(s.age[s.off[j] + 1], Infinity);  // byte 1 never changes
});

test("history frames newest first", () => {
  const r = call("frames", {k: A, t: 0.5, n: 3});
  assert.deepEqual(r.t.map((t) => Math.round(t * 100)), [50, 49, 48]);
  assert.ok(r.more);
});

test("decode signals into chunks", () => {
  const msg = {address: 0x100, size: 3, mux: null, sigs: [{name: "b0", type: 0, start: 0, size: 8, le: true, signed: false, factor: 2, offset: 1, lsb: 0, msb: 7}]};
  const r = call("decode", {k: A, msg});
  assert.equal(r.b0[0].v[10], 21);
  assert.equal(r.b0[0].t.length, 256);
});

test("find similar bits", () => {
  const rows = call("similar", {k: A, byte: 0, bit: 7, bus: 0, equal: true, min: 0});
  assert.equal(rows[0].mismatches, 0);  // the bit itself matches
  assert.deepEqual([rows[0].byte, rows[0].bit], [0, 7]);
});

test("find signal narrows", () => {
  assert.ok(call("findInit", {buses: [0], addresses: [], t0: -1, t1: Infinity, minSize: 8, maxSize: 8, le: true, signed: false, factor: 1, offset: 0}) > 0);
  const hits = call("findNext", {cmp: "=", v1: 200});
  assert.ok(hits.some((h) => h.sig.start === 0 && h.values[0][1] === 200));
});
