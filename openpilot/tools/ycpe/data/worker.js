// segment worker for the loader: fetches logs through the OPFS cache, and parses compressed log bytes into packed
// columns. Bytes and columns cross to the main thread by transfer, never by copy.
import {initCodec, withDecompressed} from "../codec/codec.js";
import {Reader, compact, pattern} from "./capnp.js";
import {migrate, withKeep} from "./migrate.js";
import {pack} from "./pack.js";
import {CanFrames} from "./can.js";
import schema from "./schema.json" with {type: "json"};  // the same instance as migrate.js's, so enum names match

const ready = initCodec().then(() => new Reader(schema));

// compressed logs in OPFS keyed route/segment/file (the signed URLs change), so a revisit makes no requests
async function file(key, create) {
  let dir = await navigator.storage.getDirectory();
  const parts = key.split("/");
  for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p, {create});
  return dir.getFileHandle(parts.at(-1), {create});
}

async function cached(key) {
  try {
    const f = await (await file(key, false)).getFile();
    return f.size ? await f.arrayBuffer() : null;
  } catch {
    return null;
  }
}

// written beside the final name and moved into place, so a partial file is never read back
async function store(key, buf) {
  try {
    const part = await file(key + ".part", true), h = await part.createSyncAccessHandle();
    try {
      h.truncate(0);
      h.write(new Uint8Array(buf), {at: 0});
      h.flush();
    } finally {
      h.close();
    }
    await part.move(key.split("/").at(-1));
  } catch {}  // no OPFS, over quota, or another worker is writing it
}

async function load(key, url) {
  let buf = await cached(key);
  if (buf) return {buf, cached: true};
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${resp.status}`);
  buf = await resp.arrayBuffer();
  await store(key, buf);
  return {buf, cached: false};
}

const patterns = (o) => o && Object.fromEntries(Object.entries(o).map(([name, globs]) => [name, pattern(globs)]));

// opts: {n, shift, keep, only, skip, context}; keep, only and skip are {service: [path globs]}
async function parse(buf, opts) {
  const reader = await ready;
  const t0 = performance.now();
  let t1, raw, migrated;
  const frames = opts.can ? new CanFrames(reader.schema) : null;  // CAN frames once, on a segment's first parse
  // the reader walks the decompressed bytes in place, so they never leave wasm memory
  const services = withDecompressed(new Uint8Array(buf), (bytes) => {
    t1 = performance.now();
    raw = bytes.byteLength;
    const read = reader.read(bytes, {keep: withKeep(opts.keep), can: frames?.hook});
    migrated = migrate(read, opts.context, {keep: opts.keep});
    return compact(read).services;
  });

  // times leave in route seconds: every segment repeats the route's initData, whose time is the route start
  let lo = Infinity, hi = -Infinity;
  for (const [name, s] of Object.entries(services)) {
    if (name !== "initData" && s.t.length) {
      lo = Math.min(lo, s.t[0]);
      hi = Math.max(hi, s.t.at(-1));
    }
  }
  const start = services.initData?.t[0] ?? lo - opts.n * 60;
  const off = opts.shift - start;
  for (const s of Object.values(services)) for (let k = 0; k < s.t.length; k++) s.t[k] += off;

  const only = patterns(opts.only), skip = patterns(opts.skip);
  const want = (only || skip) && ((name, sub) => (!only || !!only[name]?.test(sub)) && !skip?.[name]?.test(sub));
  const out = pack(services, reader.schema.enums, want);
  const can = frames?.finish();
  if (can) can.can.t0 += off;
  return {...out, ...migrated, can: can?.can, canBufs: can?.transfer || [], start, t0: lo + off, t1: hi + off, raw, ms: {decompress: t1 - t0, read: performance.now() - t1}};
}

onmessage = async ({data: {id, op, key, url, buf, opts}}) => {
  try {
    if (op === "fetch") {
      const r = await load(key, url);
      postMessage({id, ...r}, [r.buf]);
    } else {
      const r = await parse(buf, opts);
      postMessage({id, ...r}, [...r.bufs, r.recs.buffer, ...r.canBufs]);
    }
  } catch (e) {
    postMessage({id, error: String(e)});
  }
};
