#!/usr/bin/env node
// decodes every step-th message of a log through data/message.js, as JSON: [[index, message], ...]
// usage: message.mjs <log> <out.json> [step]
import fs from "node:fs";

const root = new URL("../../", import.meta.url);
globalThis.fetch = async (u) => new Response(fs.readFileSync(new URL(u)), {headers: {"content-type": "application/wasm"}});
const {initCodec, decompress} = await import(new URL("codec/codec.js", root));
const {decode, messages} = await import(new URL("data/message.js", root));
const schema = JSON.parse(fs.readFileSync(new URL("data/schema.json", root)));

const [log, out, step = 1] = process.argv.slice(2);
await initCodec();
const buf = decompress(new Uint8Array(fs.readFileSync(log)));
const json = (k, x) => x instanceof Uint8Array ? Array.from(x) : x !== x ? "nan" : x === Infinity ? "inf" : x === -Infinity ? "-inf" : x;
const picked = [];
let k = 0;
for (const [m, size] of messages(buf)) if (k++ % step === 0) picked.push([k - 1, decode(schema, buf.subarray(m, m + size))]);
fs.writeFileSync(out, JSON.stringify(picked, json));
