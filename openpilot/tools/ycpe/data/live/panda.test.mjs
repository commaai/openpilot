// node --test data/live/panda.test.mjs: the panda source against fake-panda.js, which replays bytes panda's Python
// packer made from the demo route (fixtures/gen_panda.py)
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {test} from "node:test";
import schema from "../schema.json" with {type: "json"};
import {Reader} from "../capnp.js";
import {CanFrames} from "../can.js";
import {event} from "./encode.js";
import {fakeUsb, loadFixture} from "./fake-panda.js";
import {CAN_VERSION, HEALTH_VERSION, Unpacker, create, pandaState} from "./panda.js";

const fixture = await loadFixture(readFileSync(new URL("./fixtures/panda.bin.gz", import.meta.url)));
const stream = Uint8Array.from(fixture.chunks.flatMap((c) => [...c.data]));
const hex = (d) => Buffer.from(d).toString("hex");
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

function unpackAll(pieces) {
  const u = new Unpacker(), frames = [];
  for (const p of pieces) {
    const r = u.push(p);
    assert.ok(r.ok);
    // dat views the transfer, which a reader copies before the next one
    frames.push(...r.frames.map((f) => [f.address, f.src, hex(f.dat)]));
  }
  return frames;
}

test("packet versions match panda's", () => {
  assert.deepEqual(fixture.versions, [HEALTH_VERSION, CAN_VERSION]);
});

test("frames unpack as panda packed them, however the transfers split", () => {
  const whole = unpackAll(fixture.chunks.map((c) => c.data));
  assert.equal(whole.length, fixture.frames);
  assert.deepEqual(whole.slice(0, 20), fixture.first);
  assert.deepEqual(whole.slice(-20), fixture.last);
  const bySrc = {};
  for (const [, src] of whole) bySrc[src] = (bySrc[src] || 0) + 1;
  assert.deepEqual(bySrc, fixture.bySrc);
  assert.ok(bySrc[128] && bySrc[130]);  // returned frames

  // odd-sized pieces, down to single bytes
  for (const size of [1, 5, 7, 64, 1000]) {
    const pieces = [];
    for (let at = 0; at < stream.length; at += size) pieces.push(stream.slice(at, at + size));
    assert.deepEqual(unpackAll(pieces), whole, `pieces of ${size}`);
  }
});

test("a bad checksum keeps the frames before it and drops the rest", () => {
  const u = new Unpacker(), bad = stream.slice(0, 200);
  bad[100] ^= 1;
  const r = u.push(bad);
  assert.equal(r.ok, false);
  const good = unpackAll([stream.slice(0, 200)]);
  assert.ok(r.frames.length > 0 && r.frames.length < good.length);
  assert.equal(u.tail.length, 0);
});

test("health becomes the PandaState pandad would log", () => {
  const dv = (b64) => new DataView(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)).buffer);
  const ps = pandaState(dv(fixture.health), fixture.canHealth.map(dv), fixture.hwType);
  const [svc] = new Reader(schema).read(event(1e9, true, "pandaStates", [ps])).values();
  const check = (want, path) => {
    for (const [k, v] of Object.entries(want)) {
      if (v && typeof v === "object" && !Array.isArray(v)) { check(v, `${path}/${k}`); continue; }
      const col = (p) => svc.cols.get(p);
      if (Array.isArray(v)) {
        v.forEach((name, i) => assert.equal(col(`${path}/${k}/${i}`).names[col(`${path}/${k}/${i}`).v[0]], name, k));
        continue;
      }
      const c = col(`${path}/${k}`), got = c.v[0];
      if (typeof v === "string") assert.equal(c.names[got], v, k);
      else if (typeof v === "boolean") assert.equal(got, +v, k);
      else assert.ok(Math.abs(got - v) <= (k === "interruptLoad" ? 1 / 255 : 1e-3), `${k}: ${got} != ${v}`);  // health has it in 255ths
    }
  };
  check(fixture.state, "/0");
});

// a source on the fake, with its events read back through capnp.js
async function live(opts = {}) {
  const usb = await fakeUsb({fixture, ...opts});
  const source = create({usb}), events = [], states = [];
  source.on("state", (s) => states.push(s));
  await source.start({events: (b) => events.push(b)});
  const read = () => {
    const cf = new CanFrames(schema), all = new Uint8Array(events.reduce((n, e) => n + e.length, 0));
    events.reduce((at, e) => (all.set(e, at), at + e.length), 0);
    const services = new Map([...new Reader(schema).read(all, {can: cf.hook}).values()].map((s) => [s.name, s]));
    return {services, can: cf.finish().can};
  };
  return {usb, source, events, states, read};
}

test("streams CAN and health from a granted panda", async () => {
  const {usb, source, states, read} = await live();
  assert.equal(source.state, "live");
  assert.match(source.detail, /^cuatro fake/);
  await sleep(600);
  const {services, can} = read();
  const t = services.get("can").t;
  assert.ok(t.length > 30, `${t.length} can events`);
  const gaps = Array.from(t.subarray(1), (x, i) => (x - t[i]) * 1000);
  assert.ok(Math.min(...gaps) >= 9, `events ${Math.min(...gaps)} ms apart`);
  assert.ok(can.t.length === source.stats.frames && can.t.length > 1000, `${can.t.length} frames`);
  assert.equal(services.get("pandaStates").t.length, 2);
  assert.equal(source.stats.checksumErrors, 0);
  // receive only: control reads, and the one comms reset on connect
  const outs = usb.device.requests.filter(([dir]) => dir === "out");
  assert.deepEqual(outs, [["out", 0xc0, 0]]);
  assert.deepEqual([...new Set(usb.device.requests.map(([, r]) => r))].sort(), [0xc0, 0xc1, 0xc2, 0xd2, 0xdd]);
  await source.stop();
  assert.equal(source.state, "closed");
  assert.equal(usb.device.opened, false);
  assert.deepEqual(states, ["connecting", "live", "closed"]);
});

test("packets split across short transfers", async () => {
  const {source, read} = await live({max: 100});
  await sleep(300);
  await source.stop();
  assert.equal(source.stats.checksumErrors, 0);
  assert.ok(read().can.t.length > 100);
});

test("a corrupt transfer is logged invalid, comms reset, and frames resume", async () => {
  const {usb, source, read} = await live();
  await sleep(100);
  usb.corrupt();
  usb.babble();
  await sleep(300);
  await source.stop();
  assert.equal(source.stats.checksumErrors, 1);
  assert.equal(source.stats.overflows, 1);
  const can = read().services.get("can"), bad = can.valid.indexOf(0);
  assert.ok(bad > 0 && bad < can.t.length - 5, `invalid event ${bad} of ${can.t.length}`);
  assert.equal(usb.device.requests.filter(([dir]) => dir === "out").length, 2);
});

test("unplugging waits, and a replug reconnects", async () => {
  const {usb, source, states} = await live();
  usb.unplug();
  assert.equal(source.state, "connecting");
  assert.match(source.detail, /unplugged/);
  await sleep(50);
  const frames = source.stats.frames;
  usb.plug();
  await sleep(200);
  assert.equal(source.state, "live");
  assert.ok(source.stats.frames > frames);
  await source.stop();
  assert.deepEqual(states, ["connecting", "live", "connecting", "connecting", "live", "closed"]);
});

test("an ungranted panda waits for connect()", async () => {
  const {usb, source} = await live({granted: false});
  assert.equal(source.state, "connecting");
  assert.equal(usb.device.opened, false);
  await source.connect();
  assert.equal(source.state, "live");
  await source.stop();
});
