// worker pipeline timings on one segment, stage by stage as worker.js runs them: node data/bench/parse.mjs [rlog] [iterations]
// the first iteration is what a fresh worker pays (cold JIT); the median is a warm one
import fs from "fs";
const base = new URL("../../", import.meta.url);
globalThis.fetch = async (u) => new Response(fs.readFileSync(u), {headers: {"content-type": "application/wasm"}});
const {initCodec, withDecompressed} = await import(new URL("codec/codec.js", base));
const {Reader, compact} = await import(new URL("data/capnp.js", base));
const {pack} = await import(new URL("data/pack.js", base));
const {migrate, withKeep} = await import(new URL("data/migrate.js", base));
const {CanFrames} = await import(new URL("data/can.js", base));
const schema = (await import(new URL("data/schema.json", base), {with: {type: "json"}})).default;
await initCodec();
const reader = new Reader(schema);
const src = new Uint8Array(fs.readFileSync(process.argv[2] || "/tmp/aio-probe/rlog"));

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
    const {services} = compact(read);
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
