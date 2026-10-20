// dev/panda.html's worker: reads the panda source's events through capnp.js and sends back the messages each one changed
import schema from "../data/schema.json" with {type: "json"};
import {Reader, U8, U32, list} from "../data/capnp.js";

const reader = new Reader(schema);
const pos = Object.fromEntries(schema.structs.find((s) => s.name === "CanData").fields.map((f) => [f[0], f[3]]));
const seen = new Map();  // bus * 2 ** 32 + address → {count, first}
let changed = new Map(), newest = 0;

function can(name, t, ptr) {
  const l = list(ptr);
  if (!l) return;
  newest = t;
  const [off, n, data, ptrs] = l;
  for (let j = 0; j < n; j++) {
    const e = off + j * (data + ptrs * 8), key = U8[e + pos.src] * 2 ** 32 + U32[(e + pos.address * 4) >> 2];
    const [at, len] = list(e + data + pos.dat * 8) || [0, 0];
    let hex = "";
    for (let i = 0; i < len; i++) hex += U8[at + i].toString(16).padStart(2, "0");
    const m = seen.get(key) || seen.set(key, {count: 0, first: t}).get(key);
    m.count++;
    changed.set(key, [key, m.count, m.count > 1 ? (m.count - 1) / (t - m.first) : 0, hex]);
  }
}

onmessage = ({data}) => {
  reader.read(data, {can});
  postMessage({t: newest, rows: [...changed.values()]});
  changed = new Map();
};
