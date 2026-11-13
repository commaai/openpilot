// cereal messages as JSON (webrtcd's to_dict output) into the reader's columns, so they get the paths, types and
// enum names a log of them would: one service's messages become the {name, t, valid, cols, lists} of Reader.read.
import {Column, MAX_LIST} from "./capnp.js";

// msgs: [{t (mono s), valid, data}] of one service; null when the schema has no such service
export function flatten(schema, name, msgs) {
  const field = schema.structs[0].fields.find((f) => f[0] === name && f[1] >= 0);
  if (!field) return null;
  const n = msgs.length, cols = new Map(), lists = new Map();
  const col = (path, type, names) => {
    let c = cols.get(path);
    if (!c) cols.set(path, c = new Column(type, n, names));
    return c;
  };

  // a value of field type/ref at path into message i's columns
  function put(path, type, ref, x, i) {
    if (x === null || x === undefined) return;
    if (type === "s" || type === "g") {
      if (typeof x !== "object") return;
      for (const [sub, , t, , r] of schema.structs[ref].fields) put(`${path}/${sub}`, t, r, x[sub], i);
    } else if (type === "l") {
      if (!Array.isArray(x)) return;
      if (x.length > MAX_LIST) return lists.set(path, Math.max(lists.get(path) || 0, x.length));
      x.forEach((y, j) => put(`${path}/${j}`, ref[0], ref[1], y, i));
    } else if (type === "t") {
      if (typeof x === "string") col(path, "t").v[i] = x;
    } else if (type === "e") {
      const names = schema.enums[ref], k = typeof x === "number" ? x : names.indexOf(x);
      if (k >= 0) col(path, "e", names).v[i] = k;
    } else if (type === "b") {
      if (typeof x === "boolean") col(path, "b").v[i] = +x;
    } else if (typeof x === "number") {
      col(path, type).v[i] = x;
    }
  }

  const t = new Float64Array(n), valid = new Uint8Array(n), seq = new Uint32Array(n);
  msgs.forEach((m, i) => {
    t[i] = m.t;
    valid[i] = m.valid ? 1 : 0;
    seq[i] = i;
    put("", field[2], field[4], m.data, i);
  });
  return {name, field, n, t, valid, seq, cols, lists};
}
