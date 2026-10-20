// the browser side of store.py: loads logs through the app's Store (loader, workers, migration, pack, unpack) and
// dumps every series back to one value per event, in dump.py's shape: {service: {t, valid, cols: {path: [value]}}}
import {Store} from "/data/store.js";

const json = (x) => x === null || x === undefined || x !== x ? null : x === Infinity ? "inf" : x === -Infinity ? "-inf" : x;

// a chunk's values, one per event; a constant chunk unpacks as v [c, c] at its first and last events
function expand(c) {
  const n = c.t.length;
  if (!c.i) return Array.from(c.v, json);
  if (c.i.length === 2 && c.v.length === 2 && c.i[0] === 0 && c.i[1] === n - 1 && c.v[0] === c.v[1] && n > 2) return new Array(n).fill(json(c.v[0]));
  const out = new Array(n).fill(null);
  c.i.forEach((i, k) => { out[i] = json(c.v[k]); });
  return out;
}

// urls as the Store takes them; resolves once every segment loaded or failed
export async function dumpStore(name, urls) {
  const store = new Store(name, urls);
  await new Promise((resolve) => {
    const check = () => store.progress.done + store.progress.failed >= store.progress.total && resolve();
    store.on("progress", check);
    check();
  });
  const out = {};
  for (const {path} of store.paths()) {
    const [, name, ...rest] = path.split("/"), s = store.series(path);
    const svc = out[name] ||= {t: s.chunks.flatMap((c) => Array.from(c.t)), valid: null, cols: {}};
    svc.cols[rest.length ? `/${rest.join("/")}` : ""] = s.chunks.flatMap(expand);
  }
  for (const svc of Object.values(out)) {
    const valid = "/_valid" in svc.cols ? "/_valid" : "/valid";
    svc.valid = svc.cols[valid];
    delete svc.cols[valid];
  }
  // CAN frames from the CAN worker: {"bus:address": [[t, hex data], ...]} in time order
  const can = {};
  for (const m of store.can.messages()) {
    const f = await store.can.call("frames", {k: m.key, t: Infinity, n: 1e7});
    can[`${m.bus}:${m.address}`] = f.t.map((t, i) => [t, Array.from(f.data[i], (b) => b.toString(16).padStart(2, "0")).join("")]).reverse();
  }
  store.dispose();
  return JSON.stringify({failed: store.progress.failed, services: out, can});
}
