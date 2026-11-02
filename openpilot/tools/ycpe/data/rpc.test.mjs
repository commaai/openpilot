// node --test data/rpc.test.mjs: worker calls against a fake worker, and serve() answering them
import assert from "node:assert/strict";
import {test} from "node:test";
import {client, serve} from "./rpc.js";

// a worker that keeps what's posted to it; reply() answers as the worker would
function fake() {
  const w = {posted: [], postMessage: (msg, transfer) => w.posted.push({msg, transfer})};
  w.reply = (data) => w.onmessage({data});
  return w;
}

test("resolves and rejects by id, whatever order replies come in", async () => {
  const w = fake(), call = client(w);
  const a = call("sum", {x: 1}), b = call("fail");
  assert.deepEqual(w.posted.map((p) => p.msg), [{id: 0, op: "sum", x: 1}, {id: 1, op: "fail"}]);
  w.reply({id: 1, error: "nope", stack: "at worker"});
  w.reply({id: 0, result: 2});
  assert.equal(await a, 2);
  await assert.rejects(b, (e) => e.message === "nope" && e.stack === "at worker");
  w.reply({live: {}});  // not a call's: ignored
});

test("transfer lists pass through", () => {
  const w = fake(), buf = new ArrayBuffer(8);
  client(w)("add", {buf}, [buf]);
  assert.deepEqual(w.posted[0].transfer, [buf]);
});

test("a worker error fails the calls waiting and every later one", async () => {
  const w = fake(), seen = [], call = client(w, (e) => { seen.push(e.type); return "it broke"; });
  const a = call("x"), b = call("y");
  w.onerror({type: "error"});
  w.onmessageerror({type: "messageerror"});
  await assert.rejects(a, /it broke/);
  await assert.rejects(b, /it broke/);
  await assert.rejects(call("z"), /it broke/);
  assert.deepEqual(seen, ["error"]);
  assert.equal(w.posted.length, 2);
});

test("serve answers with results, transfers and errors", async () => {
  const out = [];
  globalThis.postMessage = (msg, transfer) => out.push({msg, transfer});
  const buf = new ArrayBuffer(4);
  serve({
    async echo({x}) { return x; },
    give: (args, transfer) => { transfer.push(buf); return buf; },
    fail() { throw new Error("bad"); },
  });
  await onmessage({data: {id: 0, op: "echo", x: 3}});
  await onmessage({data: {id: 1, op: "give"}});
  await onmessage({data: {id: 2, op: "fail"}});
  assert.deepEqual(out[0], {msg: {id: 0, result: 3}, transfer: []});
  assert.deepEqual(out[1], {msg: {id: 1, result: buf}, transfer: [buf]});
  assert.equal(out[2].msg.error, "bad");
  assert.match(out[2].msg.stack, /bad/);
  delete globalThis.onmessage;
});
