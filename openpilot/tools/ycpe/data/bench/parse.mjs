// worker pipeline timings on one segment, stage by stage as worker.js runs them: node data/bench/parse.mjs [rlog] [iterations]
// the first iteration is what a fresh worker pays (cold JIT); the median is a warm one
import fs from "node:fs";
import {logArg, schema} from "./node.mjs";
import {withDecompressed} from "../../codec/codec.js";
import {Reader, compact} from "../capnp.js";
import {pack} from "../pack.js";
import {migrate, withKeep} from "../migrate.js";
import {CanFrames} from "../can.js";

const reader = new Reader(schema);
const src = new Uint8Array(fs.readFileSync(logArg()));

const STAGES = ["decompress", "read", "read+can", "migrate", "compact", "pack", "can finish"];
const runs = [];
for (let k = 0; k < (+process.argv[3] || 10); k++) {
  const ms = [];
  let t = performance.now();
  const lap = () => { const now = performance.now(); ms.push(now - t); t = now; };
  withDecompressed(src, (bytes) => {
    lap();
    reader.read(bytes, {keep: withKeep({})});  // without the CAN hook, to price it below
    lap();
    const frames = new CanFrames(schema);
    const read = reader.read(bytes, {keep: withKeep({}), can: frames.hook});
    lap();
    migrate(read, {}, {keep: {}});
    lap();
    const services = compact(read);
    lap();
    pack(services, schema.enums);
    lap();
    frames.finish();
    lap();
  });
  runs.push(ms);
}
const median = (j) => runs.map((r) => r[j]).sort((a, b) => a - b)[runs.length >> 1];
const row = (r) => STAGES.map((s, j) => `${s} ${r[j].toFixed(0)}`).join(" · ");
console.log(`first: ${row(runs[0])} ms`);
console.log(`median: ${row(STAGES.map((_, j) => median(j)))} ms`);
