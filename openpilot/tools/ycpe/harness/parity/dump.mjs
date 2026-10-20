#!/usr/bin/env node
// dumps a log through data/capnp.js in the same shape as dump.py: {service: {t, valid, cols: {path: [value|null]}}}
// usage: dump.mjs <log> <out.json> [--migrate] [--all]
import fs from "node:fs";

const root = new URL("../../", import.meta.url);
globalThis.fetch = async (u) => new Response(fs.readFileSync(new URL(u)), {headers: {"content-type": "application/wasm"}});
const {initCodec, decompress} = await import(new URL("codec/codec.js", root));
const {Reader, compact} = await import(new URL("data/capnp.js", root));

const json = (x) => x === null || x === undefined || x !== x ? null : x === Infinity ? "inf" : x === -Infinity ? "-inf" : x;

export function dump(services) {
  const out = {};
  for (const [name, {t, valid, cols}] of Object.entries(services)) {
    const c = {};
    for (const [path, col] of Object.entries(cols)) c[path] = Array.from(t, (_, i) => json("c" in col ? col.c : col.v[i]));
    out[name] = {t: Array.from(t), valid: Array.from(valid), cols: c};
  }
  return out;
}

const [file, dest] = process.argv.slice(2);
await initCodec();
const schema = JSON.parse(fs.readFileSync(new URL("data/schema.json", root)));
const {services} = compact(new Reader(schema).read(decompress(new Uint8Array(fs.readFileSync(file)))));
if (process.argv.includes("--migrate")) {
  const all = process.argv.includes("--all");
  const {migrate} = await import(new URL("data/migrate.js", root));
  const {missing} = migrate(services, {}, {managerStates: all, pandaStates: all, cameraStates: all});
  if (missing.length) console.error(`unavailable: ${missing.join(", ")}`);
}
fs.writeFileSync(dest, JSON.stringify(dump(services)));
