// segment worker for the loader: fetches logs through the OPFS cache, and parses compressed log bytes into packed
// columns. Bytes and columns cross to the main thread by transfer, never by copy.
import {initCodec, withDecompressed} from "../codec/codec.js";
import {Reader, compact, pattern} from "./capnp.js";
import {migrate, withKeep} from "./migrate.js";
import * as opfs from "./opfs.js";
import {pack} from "./pack.js";
import {index, messageAt} from "./message.js";
import {CanFrames} from "./can.js";
import schema from "./schema.json" with {type: "json"};  // the same instance as migrate.js's, so enum names match

const ready = initCodec().then(() => new Reader(schema));

const RETRIES = 6;  // 0.5 s, then doubling: about 30 s of trying before a segment fails
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const concat = (chunks) => {
  const buf = new Uint8Array(chunks.reduce((a, c) => a + c.length, 0));
  for (let k = 0, at = 0; k < chunks.length; at += chunks[k++].length) buf.set(chunks[k], at);
  return buf;
};

// the download, streamed so progress({loaded, total}) can follow it (at most 10 a second). It continues from the
// bytes in head with a Range request, as do retries after network errors and 5xx/429 responses (with backoff,
// telling progress({retry, of, error}) first). Aborting via signal throws {partial} with the bytes so far
async function download(url, progress, signal, head = null) {
  const chunks = head?.length ? [head] : [];
  let loaded = head?.length || 0;
  for (let attempt = 1; ; attempt++) {
    try {
      const resp = await fetch(url, {signal, headers: loaded ? {range: `bytes=${loaded}-`} : {}});
      if (!resp.ok) throw Object.assign(new Error(`HTTP ${resp.status}`), {transient: resp.status >= 500 || resp.status === 429});
      if (resp.status !== 206) chunks.length = loaded = 0;  // the whole file after all
      const total = loaded + (+resp.headers.get("content-length") || 0), reader = resp.body.getReader();
      let last = 0;
      for (let r; !(r = await reader.read()).done;) {
        chunks.push(r.value);
        loaded += r.value.length;
        if (performance.now() - last > 100) {
          last = performance.now();
          progress({loaded, total});
        }
      }
      return {buf: concat(chunks).buffer, hit: resp.headers.get("x-ycpe-cache") === "hit"};
    } catch (e) {
      if (signal?.aborted) throw {partial: concat(chunks).buffer};
      // fetch() rejects with a TypeError on network failures (ERR_NETWORK_CHANGED, resets, ...)
      if (!(e.transient ?? e instanceof TypeError) || attempt > RETRIES) throw e;
      progress({retry: attempt, of: RETRIES, error: e.message});
      await sleep(500 * 2 ** (attempt - 1));
    }
  }
}

// the log from the OPFS cache, else downloaded and cached; key is logs/dongle/route/segment/file. The local daemon's
// URLs come from the disk cache it shares with openpilot's tools, so OPFS would only store them twice
async function load(key, url, progress, signal, head) {
  if (url.startsWith("/local/")) {
    const {buf, hit} = await download(url, progress, signal, head);
    return {buf, cached: hit};
  }
  const f = await opfs.read(key);
  if (f) return {buf: await f.arrayBuffer(), cached: true};
  const {buf} = await download(url, progress, signal, head), stored = await opfs.write(key, new Uint8Array(buf), "logs");
  if (stored) opfs.evict("logs", key.split("/").slice(1, 3).join("/"));
  return {buf, cached: false, stored};
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

// the last logs message() opened, decompressed and indexed: scrubbing within a segment decodes without reloading
const opened = [];

async function message(key, url, service, mono) {
  let log = opened.find((l) => l.key === key);
  if (!log) {
    await ready;
    const bytes = withDecompressed(new Uint8Array((await load(key, url, () => {})).buf), (b) => b.slice());
    opened.unshift(log = {key, bytes, idx: index(schema, bytes)});
    opened.length = Math.min(opened.length, 2);
  }
  return messageAt(schema, log.bytes, log.idx, service, mono);
}

const fetches = new Map();  // id → AbortController of a fetch in flight

onmessage = async ({data: {id, op, key, url, buf, opts}}) => {
  try {
    if (op === "message") {
      postMessage({id, message: await message(key, url, opts.service, opts.mono)});
    } else if (op === "abort") {
      fetches.get(id)?.abort();
    } else if (op === "fetch") {
      const ctl = new AbortController();
      fetches.set(id, ctl);
      const r = await load(key, url, (progress) => postMessage({id, progress}), ctl.signal, buf && new Uint8Array(buf)).finally(() => fetches.delete(id));
      postMessage({id, ...r}, [r.buf]);
    } else {
      const r = await parse(buf, opts);
      postMessage({id, ...r}, [...r.bufs, r.recs.buffer, ...r.canBufs]);
    }
  } catch (e) {
    if (e.partial) postMessage({id, partial: e.partial}, [e.partial]);
    else postMessage({id, error: String(e)});
  }
};
