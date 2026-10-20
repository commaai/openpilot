// reads one rlog with the bundled schema and with the schema of the log's own commit (fetched and compiled like the
// app does), and compares every column both have: the values must agree, enums by name.
// usage: node schema-parse.mjs <rlog.zst>
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const base = new URL("../../", import.meta.url);
const netFetch = globalThis.fetch;
globalThis.fetch = async (u, o) => String(u).startsWith("file:") ? new Response(fs.readFileSync(u), {headers: {"content-type": "application/wasm"}}) : netFetch(u, o);
const {initCodec, decompress} = await import(new URL("codec/codec.js", base));
const {Reader} = await import(new URL("data/capnp.js", base));
const {setCache, parseRemote} = await import(new URL("data/github.js", base));
const {schemaAt} = await import(new URL("data/upstream.js", base));

const CACHE = path.join(os.tmpdir(), "ycpe-github");
setCache({
  async get(key) { try { return fs.readFileSync(path.join(CACHE, key), "utf8"); } catch { return null; } },
  async put(key, text) { fs.mkdirSync(path.dirname(path.join(CACHE, key)), {recursive: true}); fs.writeFileSync(path.join(CACHE, key), text); },
});

await initCodec();
const buf = decompress(new Uint8Array(fs.readFileSync(process.argv[2])));
const bundled = fs.readFileSync(new URL("data/schema.json", base), "utf8");
const stock = new Reader(JSON.parse(bundled));

// columns by path; enums keep their values, with the names they have in that schema
function columns(reader) {
  const t0 = performance.now(), out = new Map(), names = new Map();
  for (const s of reader.read(buf).values()) {
    for (const [p, c] of s.cols) {
      out.set(`/${s.name}${p}`, Array.from(c.v));
      if (c.names) names.set(`/${s.name}${p}`, c.names);
    }
  }
  return {cols: out, names, ms: performance.now() - t0, unknown: reader.unknown};
}

const a = columns(stock);
const remote = a.cols.get("/initData/gitRemote")?.[0], commit = a.cols.get("/initData/gitCommit")?.[0];
const repo = parseRemote(remote);
console.log(`log: ${remote} @ ${commit} (${a.cols.get("/initData/gitBranch")?.[0]}), stock read ${a.ms.toFixed(0)} ms, ${a.cols.size} columns, ` +
  `unknown to it: ${a.unknown.events} events, fields in [${[...a.unknown.services]}]`);

const {text, fetchMs, ms} = await schemaAt(repo, commit);
console.log(`log schema ${repo}@${commit.slice(0, 7)}: ${(text.length / 1024).toFixed(0)} KB, fetch ${fetchMs.toFixed(0)} ms, fetch+compile ${ms.toFixed(0)} ms, ${text === bundled ? "identical to" : "differs from"} the bundled schema`);
const b = columns(new Reader(JSON.parse(text)));

// a path's field as [type, offset], walking structs (and skipping list indexes) from Event
function field(schema, p) {
  let s = schema.structs[0], f;
  for (const name of p.split("/").slice(1)) {
    if (/^\d+$/.test(name)) continue;
    f = s?.fields.find((x) => x[0] === name);
    if (!f) return null;
    const ref = f[2] === "l" ? f[4][1] : f[4];
    s = f[2] === "s" || f[2] === "g" || f[2] === "l" && f[4][0] === "s" ? schema.structs[ref] : null;
  }
  return f && `${f[2]}@${f[3]}`;
}
const sa = JSON.parse(bundled), sb = JSON.parse(text);

let same = 0, bad = 0;
const renamed = [], moved = [];
for (const [p, want] of a.cols) {
  const got = b.cols.get(p);
  if (!got) continue;
  const an = a.names.get(p), bn = b.names.get(p);
  if (an && bn && an.some((n, k) => k < bn.length && n !== bn[k])) renamed.push(p);
  if (got.length === want.length && got.every((x, k) => Object.is(x, want[k]))) same++;
  else if (field(sa, p) !== field(sb, p)) moved.push(`${p} (stock ${field(sa, p)}, log ${field(sb, p)})`);
  else {
    bad++;
    const k = got.findIndex((x, k) => !Object.is(x, want[k]));
    if (bad < 10) console.log(`mismatch ${p}[${k}]: stock ${want[k]}, log ${got[k]}`);
  }
}
const onlyStock = [...a.cols.keys()].filter((p) => !b.cols.has(p)), onlyLog = [...b.cols.keys()].filter((p) => !a.cols.has(p));
console.log(`${same + bad + moved.length} shared columns: ${same} agree, ${bad} mismatched, ${moved.length} differ because the field moved upstream; ${onlyStock.length} only in stock, ${onlyLog.length} only in the log schema`);
if (moved.length) console.log(`  moved (stock misreads these in this log): ${moved.join(" ")}`);
if (renamed.length) console.log(`  enums with names changed since the log (values agree): ${renamed.join(" ")}`);
if (onlyStock.length) console.log("  only stock:", onlyStock.slice(0, 8).join(" "), onlyStock.length > 8 ? "…" : "");
if (onlyLog.length) console.log("  only log:", onlyLog.slice(0, 8).join(" "), onlyLog.length > 8 ? "…" : "");
process.exit(bad ? 1 : 0);
