// node --test data/live/encode.test.mjs: events written here read back through capnp.js, the CAN pipeline, and pycapnp
import assert from "node:assert/strict";
import {spawnSync} from "node:child_process";
import {existsSync, mkdtempSync, writeFileSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {test} from "node:test";
import {fileURLToPath} from "node:url";
import schema from "../schema.json" with {type: "json"};
import {Reader} from "../capnp.js";
import {CanFrames} from "../can.js";
import {event} from "./encode.js";

const frames = [
  {address: 0x1a4, dat: Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8), src: 0},
  {address: 0x18daf100, dat: Uint8Array.of(0x10, 0x14), src: 1},
  {address: 0x7ff, dat: new Uint8Array(0), src: 130},
  {address: 0x300, dat: Uint8Array.from({length: 64}, (_, i) => i * 3), src: 2},
];
const canState = (i) => ({busOff: i === 1, busOffCnt: 450 + i, lastStoredError: 2, totalRxCnt: 29440884 + i, canSpeed: 500, canDataSpeed: 2000, irq0CallRate: 2608});
const state = {voltage: 13882, current: 522, ignitionLine: true, controlsAllowed: true, rxBufferOverflow: 1900, pandaType: 10, safetyModel: 6,
               uptime: 431299, faults: [0, 26], harnessStatus: 1, alternativeExperience: -2, interruptLoad: 0.25, safetyParam: 3,
               sbu1Voltage: 0.5, hardfaultReset: true, canState0: canState(0), canState1: canState(1), canState2: canState(2)};
const T = 2 ** 53 - 1e6;  // logMonoTime beyond 32 bits
const events = [event(1e9, true, "can", frames), event(T, false, "can", []), event(1.5e9, true, "pandaStates", [state])];
const log = new Uint8Array(events.reduce((n, e) => n + e.length, 0));
events.reduce((at, e) => (log.set(e, at), at + e.length), 0);

test("capnp.js reads the events and the can hook reads the frames", () => {
  const cf = new CanFrames(schema), reader = new Reader(schema);
  const services = new Map([...reader.read(log, {can: cf.hook}).values()].map((s) => [s.name, s]));
  assert.deepEqual([...services.get("can").t], [1, T / 1e9]);
  assert.deepEqual([...services.get("can").valid], [1, 0]);
  const {can} = cf.finish();
  assert.equal(can.t.length, frames.length);
  for (const f of frames) {
    const j = [...can.keys].indexOf(f.src * 2 ** 32 + f.address);
    assert.ok(j >= 0, f.address.toString(16));
    assert.deepEqual([...can.data.slice(can.dstart[j], can.dstart[j] + f.dat.length)], [...f.dat]);
    assert.equal(can.len[can.start[j]], f.dat.length);
  }

  const ps = services.get("pandaStates"), col = (p) => ps.cols.get(`/0/${p}`).v[0];
  assert.equal(ps.t[0], 1.5);
  for (const k of ["voltage", "current", "rxBufferOverflow", "uptime", "alternativeExperience", "interruptLoad", "safetyParam", "sbu1Voltage"]) assert.equal(col(k), state[k], k);
  for (const k of ["ignitionLine", "controlsAllowed", "hardfaultReset"]) assert.equal(col(k), 1, k);
  assert.equal(col("ignitionCan"), 0);
  assert.equal(ps.cols.get("/0/pandaType").names[col("pandaType")], "cuatro");
  assert.equal(ps.cols.get("/0/faults/1").names[ps.cols.get("/0/faults/1").v[0]], "heartbeatLoopWatchdog");
  assert.equal(col("canState1/busOff"), 1);
  assert.equal(col("canState2/totalRxCnt"), 29440886);
  assert.equal(ps.valid[0], 1);
});

test("unknown fields throw", () => {
  assert.throws(() => event(0, true, "can", [{address: 1, nope: 2}]), /no field nope/);
});

// pycapnp from openpilot's venv: $PYTHON, else the checkout's .venv; skipped when cereal doesn't import
const root = fileURLToPath(new URL("../../../../../", import.meta.url));
const python = process.env.PYTHON || (existsSync(join(root, ".venv/bin/python")) ? join(root, ".venv/bin/python") : "python3");
const READ = `
import json, sys
from openpilot.cereal import log
evs = [e.to_dict() | {"valid": e.valid} for e in log.Event.read_multiple_bytes(open(sys.argv[1], "rb").read())]
print(json.dumps(evs, default=lambda b: list(b)))
`;

test("pycapnp reads the events", (t) => {
  const path = join(mkdtempSync(join(tmpdir(), "ycpe-encode-")), "rlog");
  writeFileSync(path, log);
  const r = spawnSync(python, ["-c", READ, path], {env: {...process.env, PYTHONPATH: root}, encoding: "utf8"});
  if (r.status !== 0 && /No module named|ENOENT/.test(r.stderr + (r.error?.message || ""))) return t.skip(`no cereal for ${python}`);
  assert.equal(r.status, 0, r.stderr);
  const [can, empty, ps] = JSON.parse(r.stdout);
  assert.equal(can.logMonoTime, 1e9);
  assert.equal(can.valid, true);
  assert.deepEqual(can.can, frames.map((f) => ({address: f.address, dat: [...f.dat], src: f.src, deprecated: {busTime: 0}})));
  assert.equal(empty.logMonoTime, T);
  assert.equal(empty.valid, false);
  assert.deepEqual(empty.can, []);
  const p = ps.pandaStates[0];
  assert.equal(p.pandaType, "cuatro");
  assert.equal(p.safetyModel, "ford");
  assert.deepEqual(p.faults, ["relayMalfunction", "heartbeatLoopWatchdog"]);
  assert.equal(p.alternativeExperience, -2);
  assert.equal(p.harnessStatus, "normal");
  assert.equal(p.ignitionCan, false);
  assert.equal(p.hardfaultReset, true);
  assert.equal(p.canState1.busOff, true);
  assert.equal(p.canState1.lastStoredError, "formError");
  assert.equal(p.canState2.totalRxCnt, 29440886);
  assert.equal(Math.fround(p.sbu1Voltage), Math.fround(0.5));
});
