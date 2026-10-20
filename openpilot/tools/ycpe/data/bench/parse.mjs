import fs from 'fs';
const base = new URL('../../', import.meta.url).pathname;
globalThis.fetch = async (u) => new Response(fs.readFileSync(new URL(u)), {headers: {'content-type': 'application/wasm'}});
const {initCodec, decompress} = await import(base + 'codec/codec.js');
const {Reader, compact} = await import(base + 'data/capnp.js');
await initCodec();
const schema = JSON.parse(fs.readFileSync(base + 'data/schema.json'));
const reader = new Reader(schema);
const src = new Uint8Array(fs.readFileSync('/tmp/aio-probe/' + (process.argv[2] || 'rlog')));
for (let k = 0; k < 4; k++) {
  let t0 = performance.now();
  const buf = decompress(src);
  let t1 = performance.now();
  const svcs = reader.read(buf);
  let t2 = performance.now();
  const {services} = compact(svcs);
  let t3 = performance.now();
  let ncols = 0, nconst = 0, bytes = 0;
  for (const s of Object.values(services)) { bytes += s.t.byteLength; for (const c of Object.values(s.cols)) { ncols++; if ('c' in c) nconst++; else bytes += c.v.byteLength || c.v.length * 8; } }
  console.log(`decompress ${(t1-t0).toFixed(0)}ms read ${(t2-t1).toFixed(0)}ms compact ${(t3-t2).toFixed(0)}ms | ${Object.keys(services).length} services ${ncols} cols ${nconst} const ${(bytes/1e6).toFixed(1)}MB`);
  if (k === 3) {
    const cs = services.carState;
    console.log('carState n', cs.t.length, 'vEgo', cs.cols['/vEgo'].v.slice(0, 3), 'gear', cs.cols['/gearShifter'], );
    console.log('selfdrive/controls', Object.keys(services.controlsState?.cols || {}).slice(0, 12));
    console.log('logMessage', services.logMessage.cols[''].v.slice(0, 2));
    console.log('pandaStates', Object.keys(services.pandaStates?.cols || {}).slice(0, 5), 'deviceState', Object.keys(services.deviceState.cols).filter(p => p.includes('cpuTemp')));
  }
}
{
  const {services} = compact(reader.read(decompress(src)));
  const by = [];
  for (const [n, s] of Object.entries(services)) { let b = s.t.byteLength, nc = 0; for (const c of Object.values(s.cols)) if (c.v) { b += c.v.byteLength || c.v.length * 8; nc++; } by.push([n, (b/1e6).toFixed(2), nc, s.t.length]); }
  by.sort((a, b) => b[1] - a[1]); console.log(by.slice(0, 12));
}
