// node --test data/live/files.test.mjs: the files channel client against a scripted device
import assert from "node:assert/strict";
import {test} from "node:test";
import {FilesChannel} from "./files.js";

// a data channel whose device answers with answer(request, reply)
function channel(answer) {
  const ch = new EventTarget();
  ch.readyState = "open";
  ch.sent = [];
  const deliver = (data) => setTimeout(() => ch.dispatchEvent(Object.assign(new Event("message"), {data})));
  ch.send = (text) => {
    const msg = JSON.parse(text);
    ch.sent.push(msg);
    answer(msg, (m) => deliver(typeof m === "string" || m instanceof ArrayBuffer ? m : JSON.stringify(m)));
  };
  ch.close = () => {
    ch.readyState = "closed";
    ch.dispatchEvent(new Event("close"));
  };
  return ch;
}

const chunk = (id, offset, bytes) => {
  const buf = new Uint8Array(8 + bytes.length), dv = new DataView(buf.buffer);
  dv.setUint32(0, id, true);
  dv.setUint32(4, offset, true);
  buf.set(bytes, 8);
  return buf.buffer;
};

const FILE = Uint8Array.from({length: 200000}, (_, i) => i * 7 % 251);

// a device streaming FILE in 64 KB messages
const device = (msg, reply) => {
  if (msg.op === "list") return reply({id: msg.id, routes: [{name: "0000010a--a51155e496", segments: [{n: 0, files: {"rlog.zst": FILE.length}}]}]});
  if (msg.op !== "read") return;
  if (msg.path !== "0000010a--a51155e496--0/rlog.zst") return reply({id: msg.id, error: "no such file"});
  const end = msg.length != null ? msg.offset + msg.length : FILE.length;
  reply({id: msg.id, size: FILE.length});
  for (let at = msg.offset; at < end; at += 65528) reply(chunk(msg.id, at, FILE.subarray(at, Math.min(end, at + 65528))));
  reply({id: msg.id, done: true});
};

test("lists the log root", async () => {
  const routes = await new FilesChannel(channel(device)).list();
  assert.deepEqual(routes[0].segments[0].files, {"rlog.zst": FILE.length});
});

test("joins a listing sent in parts", async () => {
  const seg = (n) => ({n, files: {"rlog.zst": 1}});
  const routes = await new FilesChannel(channel((msg, reply) => {
    reply({id: msg.id, routes: [{name: "a", segments: [seg(0)]}, {name: "b", segments: [seg(0)]}], more: true});
    reply({id: msg.id, routes: [{name: "b", segments: [seg(1)]}, {name: "c", segments: [seg(0)]}]});
  })).list();
  assert.deepEqual(routes.map((r) => [r.name, r.segments.map((s) => s.n)]), [["a", [0]], ["b", [0, 1]], ["c", [0]]]);
});

test("reads a file into the bytes a download gives", async () => {
  const files = new FilesChannel(channel(device));
  assert.deepEqual(await files.bytes("0000010a--a51155e496--0/rlog.zst"), FILE);
  assert.deepEqual(await files.bytes("0000010a--a51155e496--0/rlog.zst", {offset: 70000}), FILE.subarray(70000));
  assert.deepEqual(await files.bytes("0000010a--a51155e496--0/rlog.zst", {offset: 1000, length: 100000}), FILE.subarray(1000, 101000));
});

test("reads one at a time", async () => {
  const ch = channel(device), files = new FilesChannel(ch);
  const reads = [0, 1, 2].map(() => files.bytes("0000010a--a51155e496--0/rlog.zst"));
  assert.equal(ch.sent.length, 0);  // queued behind a microtask
  for (const r of await Promise.all(reads)) assert.deepEqual(r, FILE);
  assert.deepEqual(ch.sent.map((m) => m.id), [1, 2, 3]);
});

test("device errors and gaps fail the read", async () => {
  const files = new FilesChannel(channel(device));
  await assert.rejects(files.bytes("nope--0/rlog.zst"), (e) => e.message === "no such file" && e.device);
  const gappy = new FilesChannel(channel((msg, reply) => {
    reply({id: msg.id, size: 10});
    reply(chunk(msg.id, 5, FILE.subarray(5, 10)));
  }));
  await assert.rejects(gappy.bytes("x--0/rlog.zst"), /a chunk at 5, expected 0/);
});

test("cancelling tells the device and frees the channel", async () => {
  let held = null;
  const ch = channel((msg, reply) => {
    if (msg.op === "cancel") return;
    if (!held) return held = {msg, reply};  // the first read never finishes on its own
    device(msg, reply);
  });
  const files = new FilesChannel(ch), ctl = new AbortController();
  const first = files.bytes("0000010a--a51155e496--0/rlog.zst", {signal: ctl.signal});
  const second = files.bytes("0000010a--a51155e496--0/rlog.zst");
  await new Promise((r) => setTimeout(r));
  ctl.abort(new Error("cancelled"));
  await assert.rejects(first, /cancelled/);
  held.reply(chunk(held.msg.id, 0, FILE.subarray(0, 10)));  // the cancelled read's tail is dropped
  assert.deepEqual(await second, FILE);
  assert.deepEqual(ch.sent.map((m) => m.op), ["read", "cancel", "read"]);
});

test("a closed channel fails what's pending", async () => {
  const ch = channel(() => {});
  const files = new FilesChannel(ch), listing = files.list();
  ch.close();
  await assert.rejects(listing, /closed/);
});
