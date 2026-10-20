// worker pipeline timings on one segment: node data/bench/parse.mjs [rlog] [iterations]
// the first iteration is what a fresh worker pays (cold JIT); the median is a warm one
import fs from "fs";
const base = new URL("../../", import.meta.url);
globalThis.fetch = async (u) => new Response(fs.readFileSync(u), {headers: {"content-type": "application/wasm"}});
const {initCodec, withDecompressed} = await import(new URL("codec/codec.js", base));
const {Reader, compact} = await import(new URL("data/capnp.js", base));
const {pack} = await import(new URL("data/pack.js", base));
await initCodec();
const schema = JSON.parse(fs.readFileSync(new URL("data/schema.json", base)));
const reader = new Reader(schema);
const src = new Uint8Array(fs.readFileSync(process.argv[2] || "/tmp/aio-probe/rlog"));

const runs = [];
let out;
for (let k = 0; k < (+process.argv[3] || 10); k++) {
  const t = [performance.now()];
  out = withDecompressed(src, (bytes) => {
    t.push(performance.now());
    const svcs = reader.read(bytes);
    t.push(performance.now());
    const {services} = compact(svcs);
    t.push(performance.now());
    return pack(services, schema.enums);
  });
  t.push(performance.now());
  runs.push(t.slice(1).map((x, j) => x - t[j]));
}
const median = (j) => runs.map((r) => r[j]).sort((a, b) => a - b)[runs.length >> 1];
const row = (r) => ["decompress", "read", "compact", "pack"].map((s, j) => `${s} ${r[j].toFixed(0)}`).join(" · ");
console.log(`first: ${row(runs[0])} ms`);
console.log(`median: ${row([0, 1, 2, 3].map(median))} ms`);
console.log(`${out.services.length} services · ${out.paths.split("\n").length} columns · ${(out.bufs.reduce((s, b) => s + b.byteLength, 0) / 1e6).toFixed(2)} MB packed`);
