#!/usr/bin/env node
// decodes every step-th message of a log through data/message.js, as JSON: [[index, message], ...]
// usage: message.mjs <log> <out.json> [step]
import fs from "node:fs";
import {schema} from "../../data/bench/node.mjs";
import {decompress} from "../../codec/codec.js";
import {decode, messages} from "../../data/message.js";

const [log, out, step = 1] = process.argv.slice(2);
const buf = decompress(new Uint8Array(fs.readFileSync(log)));
const json = (k, x) => x instanceof Uint8Array ? Array.from(x) : x !== x ? "nan" : x === Infinity ? "inf" : x === -Infinity ? "-inf" : x;
const picked = [];
let k = 0;
for (const [m, size] of messages(buf)) if (k++ % step === 0) picked.push([k - 1, decode(schema, buf.subarray(m, m + size))]);
fs.writeFileSync(out, JSON.stringify(picked, json));
