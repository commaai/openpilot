// store checks against brute force over the loaded route; run in dev/pipeline.html once loaded:
//   (await import("/data/bench/store_test.js")).check(store)
export async function check(store) {
  const fails = [];
  const expect = (ok, msg) => ok || fails.push(msg);
  const dense = (path) => {  // every (t, v) of a series, the slow way
    const t = [], v = [];
    for (const c of store.series(path).chunks) for (let k = 0; k < c.tt.length; k++) if (c.v[k] === c.v[k]) { t.push(c.tt[k]); v.push(c.v[k]); }
    return {t, v};
  };
  const {duration} = store.route;

  for (const path of ["/carState/vEgo", "/carState/steeringAngleDeg", "/modelV2/frameId", "/deviceState/cpuTempC/0", "/carState/gearShifter"]) {
    const s = store.series(path), {t, v} = dense(path);
    expect(s && t.length, `${path}: no samples`);
    expect(["f32", "u32", "e", "f64"].includes(s.type), `${path}: type ${s.type}`);
    for (const [t0, t1, n] of [[0, duration, 800], [100, 400, 300], [61.5, 62.5, 50], [0, duration, 7]]) {
      const d = store.decimate(path, t0, t1, n);
      let lo = Infinity, hi = -Infinity, cnt = 0;
      for (let k = 0; k < t.length; k++) if (t[k] >= t0 && t[k] <= t1) { lo = Math.min(lo, v[k]); hi = Math.max(hi, v[k]); cnt++; }
      let dlo = Infinity, dhi = -Infinity;
      for (let b = 0; b < d.t.length; b++) if (d.min[b] === d.min[b]) { dlo = Math.min(dlo, d.min[b]); dhi = Math.max(dhi, d.max[b]); }
      expect(dlo === lo && dhi === hi, `${path} decimate(${t0}, ${t1}, ${n}): range [${dlo}, ${dhi}] vs [${lo}, ${hi}]`);
      expect(cnt > n || d.t.length === cnt, `${path} decimate(${t0}, ${t1}, ${n}): ${d.t.length} exact samples vs ${cnt}`);
    }
    for (const at of [0, 30.123, 61, 500.5, duration + 10]) {
      let want;
      for (let k = 0; k < t.length && t[k] <= at; k++) want = v[k];
      expect(store.valueAt(path, at) === want, `${path} valueAt(${at}): ${store.valueAt(path, at)} vs ${want}`);
    }
  }

  // derived series
  store.put("/custom/ramp", {type: "f64", t: Float64Array.of(0, 1, 2), v: Float64Array.of(5, 6, 7)});
  expect(store.valueAt("/custom/ramp", 1.5) === 6, "put: valueAt");
  expect(store.paths().some((p) => p.path === "/custom/ramp"), "put: paths");

  // sendcan is columns from the first parse on (only can goes to the CAN hook): demo segment 0 sent 5968 frames
  let frames = 0;
  for (let k = 0; store.series(`/sendcan/${k}/address`); k++) {
    const c = store.series(`/sendcan/${k}/address`).chunks.find((c) => c.n === 0);
    if (c) for (let j = 0; j < c.tt.length; j++) frames += c.v[j] === c.v[j];
  }
  expect(frames === 5968, `sendcan: ${frames} frames in segment 0`);

  // a skipped long list loads on demand
  const list = "/modelV2/position/x/32";
  expect(store.series(list) === null, "long list: not loaded yet");
  const all = () => store.series(list)?.chunks.length === store.series("/carState/vEgo").chunks.length;
  await new Promise((resolve) => { const off = store.on("segment", () => all() && (off(), resolve())); });
  expect(store.valueAt(list, 100) > 50, `long list: x[32] = ${store.valueAt(list, 100)}`);

  // eviction drops columns and a later read brings them back
  const n = dense("/carState/vEgo").t.length, before = store.bytes, budget = store.budget;
  store.budget = before / 2;
  for (const e of store.entries.values()) if (e.at.length) e.used = 0;
  store.evict();
  expect(store.bytes < before / 2, `evict: ${store.bytes} bytes left of ${before}`);
  store.budget = budget;
  expect(store.entries.get("/carState/vEgo").evicted || store.entries.get("/modelV2/frameId").evicted, "evict: something evicted");
  const back = ["/carState/vEgo", "/modelV2/frameId"].find((p) => store.entries.get(p).evicted);
  store.series(back);
  await new Promise((resolve) => { const off = store.on("segment", () => store.series(back).chunks.length === store.route.segments.length && (off(), resolve())); });
  expect(back !== "/carState/vEgo" || dense(back).t.length === n, "evict: reload restores every sample");
  return fails;
}
