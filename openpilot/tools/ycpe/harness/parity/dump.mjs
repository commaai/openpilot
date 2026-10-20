#!/usr/bin/env node
// dumps a log through data/capnp.js in the same shape as dump.py: {service: {t, valid, cols: {path: [value|null]}}}
// usage: dump.mjs <log> <out.json> [--migrate] [--all]
import fs from "node:fs";

const root = new URL("../../", import.meta.url);
globalThis.fetch = async (u) => new Response(fs.readFileSync(new URL(u)), {headers: {"content-type": "application/wasm"}});
const {initCodec, decompress} = await import(new URL("codec/codec.js", root));
const {Reader, compact} = await import(new URL("data/capnp.js", root));
const {migrate, withKeep} = await import(new URL("data/migrate.js", root));

const json = (x) => x === null || x === undefined || x !== x ? null : x === Infinity ? "inf" : x === -Infinity ? "-inf" : x;

// a compacted column back to one value per event
function expand(col, n) {
  if ("c" in col) return new Array(n).fill(json(col.c));
  if (!col.i) return Array.from(col.v, json);
  const out = new Array(n).fill(null);
  col.i.forEach((i, k) => { out[i] = json(col.v[k]); });
  return out;
}

export function dump(services) {
  const out = {};
  for (const [name, {t, cols}] of Object.entries(services)) {
    const valid = "/_valid" in cols ? "/_valid" : "/valid";
    const c = {};
    for (const [path, col] of Object.entries(cols)) if (path !== valid) c[path] = expand(col, t.length);
    out[name] = {t: Array.from(t), valid: expand(cols[valid], t.length), cols: c};
  }
  return out;
}

const [file, dest] = process.argv.slice(2);
await initCodec();
const schema = JSON.parse(fs.readFileSync(new URL("data/schema.json", root)));
const migrating = process.argv.includes("--migrate"), all = process.argv.includes("--all");
const services = new Reader(schema).read(decompress(new Uint8Array(fs.readFileSync(file))), migrating ? {keep: withKeep()} : {});
if (migrating) {
  const {missing} = migrate(services, {}, {managerStates: all, pandaStates: all, cameraStates: all});
  if (missing.length) console.error(`unavailable: ${missing.join(", ")}`);
}
fs.writeFileSync(dest, JSON.stringify(dump(compact(services).services)));
