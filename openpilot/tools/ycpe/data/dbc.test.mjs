// node --test data/dbc.test.mjs (OPENDBC=<dir of .dbc files>, defaults to the repo's opendbc and data/opendbc/)
import assert from "node:assert/strict";
import {readdirSync, readFileSync, existsSync} from "node:fs";
import {test} from "node:test";
import {DBC, Dbcs, MUX, MUXED, cloneMsg, encode, format, newSignal, num, raw, value} from "./dbc.js";

const here = new URL(".", import.meta.url).pathname;
const dirs = [process.env.OPENDBC || `${here}../../../../opendbc_repo/opendbc/dbc`, `${here}opendbc`].filter((d) => existsSync(d));
const files = dirs.flatMap((d) => readdirSync(d).filter((f) => f.endsWith(".dbc")).map((f) => `${d}/${f}`));
const load = (f) => new DBC(f.split("/").pop().slice(0, -4), readFileSync(f, "utf8"));
// %.15g rounds huge limits, as in cabana
const close = (a, b) => typeof a === "number" ? a === b || Math.abs(a - b) <= 1e-14 * Math.abs(b) : a === b;
const assertSigs = (a, b, msg) => {
  assert.equal(a.length, b.length, msg);
  for (const s of a) {
    const o = b.find((x) => x.name === s.name);
    for (const k in s) if (k !== "vals" && k !== "color") assert.ok(close(s[k], o[k]), `${msg} ${s.name}.${k}`);
  }
};
const regenerate = (dbc) => [...dbc.generate().values()].join("");

test("opendbc dir", () => assert.ok(files.length > 50, `found ${files.length} DBCs in ${dirs}`));

test("generate dbc", () => {
  const origin = load(files.find((f) => f.endsWith("/tesla_can.dbc")));
  const generated = new DBC("", regenerate(origin));
  assert.equal(generated.msgs.size, origin.msgs.size);
  for (const [address, m] of origin.msgs) {
    const n = generated.msg(address);
    assert.equal(n.name, m.name);
    assert.equal(n.size, m.size);
    assert.deepEqual(n.sigs, m.sigs);
  }
});

test("comment order", () => {
  const content = `BO_ 160 message_1: 8 EON
 SG_ signal_1 : 0|12@1+ (1,0) [0|4095] "unit" XXX

BO_ 162 message_2: 8 EON
 SG_ signal_2 : 0|12@1+ (1,0) [0|4095] "unit" XXX

CM_ BO_ 160 "message comment";
CM_ SG_ 160 signal_1 "signal comment";
CM_ BO_ 162 "message comment";
CM_ SG_ 162 signal_2 "signal comment";
`;
  const dbc = new DBC("", content);
  assert.equal(dbc.serialize(), content);
  assert.equal(regenerate(dbc), content);
});

test("preserve original header", () => {
  const header = `VERSION "1.0"

NS_ :
 CM_

BS_:

BU_: EON

`;
  const body = `BO_ 160 message_1: 8 EON
 SG_ signal_1 : 0|12@1+ (1,0) [0|4095] "unit" XXX

CM_ BO_ 160 "message comment";
CM_ SG_ 160 signal_1 "signal comment";
`;
  const dbc = new DBC("", header + body);
  assert.equal(dbc.serialize(), header + body);
  assert.equal(regenerate(dbc), body);
});

test("escaped quotes", () => {
  const content = `BO_ 160 message_1: 8 EON
 SG_ signal_1 : 0|12@1+ (1,0) [0|4095] "unit" XXX

CM_ BO_ 160 "message comment with \\"escaped quotes\\"";
CM_ SG_ 160 signal_1 "signal comment with \\"escaped quotes\\"";
`;
  const dbc = new DBC("", content);
  assert.equal(dbc.serialize(), content);
  assert.equal(regenerate(dbc), content);
  assert.equal(dbc.msg(160).comment, 'message comment with "escaped quotes"');
});

const PARSE = `
BO_ 160 message_1: 8 EON
  SG_ signal_1 : 0|12@1+ (1,0) [0|4095] "unit"  XXX
  SG_ signal_2 : 12|1@1+ (1.0,0.0) [0.0|1] ""  XXX

BO_ 162 message_1: 8 XXX
  SG_ signal_1 M : 0|12@1+ (1,0) [0|4095] "unit" XXX
  SG_ signal_2 M4 : 12|1@1+ (1.0,0.0) [0.0|1] "" XXX

VAL_ 160 signal_1 0 "disabled" 1.2 "initializing" 2 "fault";

CM_ BO_ 160 "message comment" ;
CM_ SG_ 160 signal_1 "signal comment";
CM_ SG_ 160 signal_2 "multiple line comment\x20
1
2
";

CM_ BO_ 162 "message comment with \\"escaped quotes\\"";
CM_ SG_ 162 signal_1 "signal comment with \\"escaped quotes\\"";
`;

test("parse dbc", () => {
  const file = new DBC("", PARSE);
  let msg = file.msg(160);
  assert.equal(msg.name, "message_1");
  assert.equal(msg.size, 8);
  assert.equal(msg.comment, "message comment");
  assert.equal(msg.sigs.length, 2);
  assert.equal(msg.transmitter, "EON");
  assert.ok(file.msgByName("message_1"));

  const [sig1, sig2] = msg.sigs;
  assert.equal(sig1.name, "signal_1");
  assert.equal(sig1.start, 0);
  assert.equal(sig1.size, 12);
  assert.equal(sig1.min, 0);
  assert.equal(sig1.max, 4095);
  assert.equal(sig1.unit, "unit");
  assert.equal(sig1.comment, "signal comment");
  assert.equal(sig1.receiver, "XXX");
  assert.deepEqual(sig1.vals, [[0, "disabled"], [1.2, "initializing"], [2, "fault"]]);
  assert.equal(sig2.comment, "multiple line comment \n1\n2");

  msg = file.msg(162);
  assert.equal(msg.sigs.length, 2);
  assert.equal(msg.sigs[0].type, MUX);
  assert.equal(msg.sigs[1].type, MUXED);
  assert.equal(msg.sigs[1].muxValue, 4);
  assert.equal(msg.sigs[1].start, 12);
  assert.equal(msg.sigs[1].size, 1);
  assert.equal(msg.sigs[1].receiver, "XXX");
  assert.equal(msg.comment, 'message comment with "escaped quotes"');
  assert.equal(msg.sigs[0].comment, 'signal comment with "escaped quotes"');
  assert.equal(file.serialize(), PARSE);
});

test("parse errors", () => {
  assert.throws(() => new DBC("x", "BO_ 1 a: 8 XXX\nBO_ 1 b: 8 XXX\n"), /x:2\] Duplicate message address: 1/);
  assert.throws(() => new DBC("x", ' SG_ a : 0|1@1+ (1,0) [0|1] "" XXX\n'), /No Message/);
  assert.throws(() => new DBC("x", 'BO_ 1 a: 8 XXX\n SG_ a M : 0|1@1+ (1,0) [0|1] "" XXX\n SG_ b M : 1|1@1+ (1,0) [0|1] "" XXX\n'), /Multiple multiplexor/);
  assert.throws(() => new DBC("x", 'BO_ 1 a: 8 XXX\n SG_ a : 0|1@1+ (1,0) [0|1] "" XXX\n SG_ a : 1|1@1+ (1,0) [0|1] "" XXX\n'), /Duplicate signal name/);
});

test("opendbc parses and round-trips exactly", () => {
  for (const f of files) {
    const text = readFileSync(f, "utf8"), dbc = new DBC(f, text);
    assert.equal(dbc.serialize(), text, f);
    // cabana's generator output parses back to the same model
    const again = new DBC(f, regenerate(dbc));
    for (const [address, m] of dbc.msgs) assertSigs(again.msg(address).sigs, m.sigs, `${f} ${m.name}`);
  }
});

test("edits only rewrite what they touch", () => {
  for (const f of files) {
    const dbc = load(f), text = dbc.serialize();
    const [address, m] = [...dbc.msgs].find(([, x]) => x.sigs.length > 1) || [];
    if (!m) continue;
    const edited = cloneMsg(m);
    edited.sigs[0] = {...edited.sigs[0], name: "RENAMED", comment: "a \"new\" comment", vals: [[1, "one"]]};
    edited.sigs.pop();
    dbc.setMsg(address, edited);
    dbc.setMsg(0x7ff00, {address: 0x7ff00, name: "NEW_MSG", size: 8, transmitter: "", comment: "new", sigs: [newSignal({name: "S", size: 8, max: 255})]});
    const out = dbc.serialize();
    const back = new DBC(f, out);
    assert.deepEqual([...back.msgs.keys()].sort(), [...dbc.msgs.keys()].sort(), f);
    for (const [a, x] of dbc.msgs) {
      assert.deepEqual(back.msg(a).sigs, x.sigs, `${f} ${x.name}`);
      assert.equal(back.msg(a).comment, x.comment);
    }
    // everything unrelated is untouched: lines removed are only the edited message's
    const before = new Set(text.split("\n")), after = new Set(out.split("\n"));
    const gone = [...before].filter((l) => !after.has(l));
    assert.ok(gone.every((l) => l.includes(m.name) || m.sigs.some((s) => l.includes(s.name)) || l.includes(String(address))), `${f}: ${gone.slice(0, 3)}`);
  }
});

test("new file matches cabana's layout", () => {
  const dbc = new DBC("new", "");
  dbc.setMsg(160, {address: 160, name: "message_1", size: 8, transmitter: "EON", comment: "message comment",
                   sigs: [newSignal({name: "signal_1", size: 12, max: 4095, unit: "unit", comment: "signal comment", vals: [[0, "off"]]})]});
  assert.equal(dbc.serialize(), `BO_ 160 message_1: 8 EON
 SG_ signal_1 : 0|12@1+ (1,0) [0|4095] "unit" XXX

CM_ BO_ 160 "message comment";
CM_ SG_ 160 signal_1 "signal comment";
VAL_ 160 signal_1 0 "off";
`);
});

test("undo and redo restore the exact text", () => {
  const mgr = new Dbcs(), dbc = new DBC("t", PARSE);
  mgr.open([-1], dbc);
  const m = cloneMsg(dbc.msg(160));
  m.sigs.push(newSignal({name: "speed", start: 16, size: 8}));
  mgr.edit(0, 160, m, "add signal");
  mgr.edit(0, 162, null, "remove message");
  assert.ok(mgr.msg(0, 160).sigs.some((s) => s.name === "speed"));
  assert.equal(mgr.msg(0, 162), undefined);
  assert.ok(mgr.dirty);
  mgr.undo();
  mgr.undo();
  assert.equal(dbc.serialize(), PARSE);
  assert.ok(!mgr.dirty);
  mgr.redo();
  mgr.redo();
  assert.ok(mgr.msg(0, 160).sigs.some((s) => s.name === "speed"));
  assert.ok(!dbc.serialize().includes("signal_2 M4"));
  assert.ok(dbc.serialize().includes(" SG_ speed : 16|8@1+"));
});

test("dbc manager", () => {
  const mgr = new Dbcs();
  let changes = 0;
  mgr.on(() => changes++);
  mgr.open([-1], new DBC("test", "BO_ 160 message: 8 XXX\n"));
  const m = cloneMsg(mgr.msg(0, 160));
  m.sigs.push(newSignal({name: "speed", size: 8}));
  mgr.edit(0, 160, m);
  assert.equal(changes, 2);
  assert.ok(mgr.msg(5, 160).sigs.find((s) => s.name === "speed"));
  mgr.open([1], new DBC("bus1", ""));
  assert.equal(mgr.msg(1, 160), undefined);
  assert.ok(mgr.msg(2, 160));
});

test("number formatting matches %.15g", () => {
  const cases = [[0, "0"], [1, "1"], [0.1, "0.1"], [-0.5, "-0.5"], [1e-5, "1e-05"], [0.0001, "0.0001"], [123456789012345, "123456789012345"],
                 [1e15, "1e+15"], [0.03125, "0.03125"], [1 / 3, "0.333333333333333"], [4095, "4095"], [-1.2e-7, "-1.2e-07"]];
  for (const [v, s] of cases) assert.equal(num(v), s);
});

test("signal decode and encode", () => {
  const dbc = new DBC("", `BO_ 1 m: 8 XXX
 SG_ le : 4|12@1+ (1,0) [0|1] "" XXX
 SG_ be : 7|12@0- (0.5,-1) [0|1] "" XXX
 SG_ mx M : 56|4@1+ (1,0) [0|1] "" XXX
 SG_ a m2 : 48|8@1+ (1,0) [0|1] "" XXX
`);
  const m = dbc.msg(1), sig = (n) => m.sigs.find((s) => s.name === n);
  const d = Uint8Array.from([0xab, 0xcd, 0xef, 0, 0, 0, 7, 2]);
  assert.equal(raw(sig("le"), d), 0xcda);  // bits 4..15, little endian
  assert.equal(raw(sig("be"), d), -0x544);  // 0xabc, signed
  assert.equal(value(m, sig("be"), d), -0x544 * 0.5 - 1);
  assert.equal(value(m, sig("a"), d), 7);
  d[7] = 3;
  assert.ok(Number.isNaN(value(m, sig("a"), d)));
  assert.equal(format(newSignal({factor: 0.1, precision: 1, unit: "m", vals: []}), 2.345), "2.3 m");

  // every layout: encode then decode is the identity, and touches only the signal's bits
  let seed = 1;
  const rand = (n) => (seed = (seed * 16807) % 2147483647) % n;
  for (let i = 0; i < 20000; i++) {
    const size = 1 + rand(48), le = rand(2) === 1, signed = rand(2) === 1;
    const start = le ? rand(64 - size + 1) : pickBE(size, rand);
    const msg = {address: 1, name: "m", size: 8, sigs: [newSignal({name: "s", start, size, le, signed})]};
    new DBC().setMsg(1, msg);
    const s = msg.sigs[0], buf = Uint8Array.from({length: 8}, () => rand(256)), orig = buf.slice();
    const v = signed ? rand(2 ** Math.min(size, 30)) - 2 ** (Math.min(size, 30) - 1) : rand(2 ** Math.min(size, 30));
    encode(s, buf, v);
    assert.equal(raw(s, buf), v, `size ${size} le ${le} start ${start}`);
    for (let b = 0; b < 8; b++) assert.equal((buf[b] ^ orig[b]) & ~msg.mask[b], 0);
  }
});

// a big endian start bit whose signal fits in 8 bytes
function pickBE(size, rand) {
  const ok = [];
  for (let b = 0; b < 64; b++) if (8 * (b >> 3) + 7 - (b & 7) + size - 1 < 64) ok.push(b);
  return ok[rand(ok.length)];
}
