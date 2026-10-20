// segment worker for the loader: fetches logs through the OPFS cache, and parses compressed log bytes into packed
// columns. Bytes and columns cross to the main thread by transfer, never by copy.
import {initCodec, withDecompressed} from "../codec/codec.js";
import {Reader, compact, pattern} from "./capnp.js";
import {migrate, withKeep} from "./migrate.js";
import * as opfs from "./opfs.js";
import {pack} from "./pack.js";
import {index, messageAt} from "./message.js";
import {CanFrames} from "./can.js";
import {bootLog} from "./boot.js";
import {Slots} from "./slots.js";
import schema from "./schema.json" with {type: "json"};  // the same instance as migrate.js's, so enum names match

const codec = initCodec(), ready = codec.then(() => new Reader(schema));
const readers = new Map();  // key → reader of a log's own schema (data/schemas.js)

// the reader of a log's own schema, or the bundled one for null; either way once the codec is ready
async function readerFor(log) {
  const stock = await ready;
  if (!log) return stock;
  if (!readers.has(log.key)) readers.set(log.key, new Reader(JSON.parse(log.text)));
  return readers.get(log.key);
}

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
      return {buf: concat(chunks).buffer, hit: /^(hit|stale-refetch)$/.test(resp.headers.get("x-ycpe-cache"))};  // the daemon's disk
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
// URLs come from the disk cache it shares with openpilot's tools, and dropped files (blob: URLs) are on disk already,
// so OPFS would only store them twice
async function load(key, url, progress, signal, head) {
  if (url.startsWith("/local/") || url.startsWith("blob:")) {
    const {buf, hit} = await download(url, progress, signal, head);
    return {buf, cached: hit};
  }
  const f = await opfs.read(key);
  if (f) return {buf: await f.arrayBuffer(), cached: true};
  const {buf} = await download(url, progress, signal, head), stored = await opfs.write(key, new Uint8Array(buf), "logs");
  if (stored) opfs.evict("logs", key.split("/").slice(1, 3).join("/"));
  return {buf, cached: false, stored};
}

// the element type of the list at /service/path, from the schema: "s" for structs
function elemType(schema, path) {
  let [type, ref] = [];
  for (const [k, part] of path.split("/").slice(1).entries()) {
    if (type === "l") [type, ref] = ref;  // an index into a list
    else if (!k || type === "s" || type === "g") {
      const f = schema.structs[k ? ref : 0].fields.find((x) => x[0] === part);
      if (!f) return null;
      [type, ref] = [f[2], f[4]];
    } else {
      return null;
    }
  }
  return type === "l" ? ref[0] : null;
}

const patterns = (o) => o && Object.fromEntries(Object.entries(o).map(([name, globs]) => [name, pattern(globs)]));

// opts: {n, shift, base, keep, only, skip, context, schema, gen, migrate}; keep, only and skip are {service: [path globs]};
// schema is the log's own ({key, text}), read as it is with no migrations, or null for the bundled one; migrate: false
// for logs written by current code (replays), where a migration would only invent services from the ones present;
// base: the logMonoTime (s) of route time 0, where a live session started. more: services read elsewhere (JSON)
async function parse(buf, opts, more = []) {
  const reader = await readerFor(opts.schema);
  const t0 = performance.now();
  let t1, raw, migrated;
  const frames = opts.can ? new CanFrames(reader.schema) : null;  // CAN frames once, on a segment's first parse
  // the reader walks the decompressed bytes in place, so they never leave wasm memory
  const services = withDecompressed(new Uint8Array(buf), (bytes) => {
    t1 = performance.now();
    raw = bytes.byteLength;
    const read = reader.read(bytes, {keep: withKeep(opts.keep), can: frames?.hook});
    for (const svc of more) if (![...read.values()].some((s) => s.name === svc.name)) read.set(svc.name, svc);
    if (!read.size) throw new Error(reader.unknown.events ? "none of its messages are in the schema" : "not an openpilot log: no messages in it");
    migrated = opts.schema || opts.migrate === false ? {} : migrate(read, opts.context, {keep: opts.keep});
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
  const start = opts.base ?? services.initData?.t[0] ?? lo - opts.n * 60;
  const off = opts.shift - start;
  // migrated services can share their source's time array: shift each array once
  for (const t of new Set(Object.values(services).map((s) => s.t))) for (let k = 0; k < t.length; k++) t[k] += off;

  const only = patterns(opts.only), skip = patterns(opts.skip);
  const want = (only || skip) && ((name, sub) => (!only || !!only[name]?.test(sub)) && !skip?.[name]?.test(sub));
  const out = pack(services, reader.schema.enums, want);
  const can = frames?.finish();
  if (can) can.can.t0 += off;
  const unknown = {events: reader.unknown.events, services: [...reader.unknown.services]};
  const listTypes = Object.fromEntries(Object.keys(out.lists).map((path) => [path, elemType(reader.schema, path)]));
  return {...out, ...migrated, unknown, listTypes, gen: opts.gen, can: can?.can, canBufs: can?.transfer || [], start, t0: lo + off, t1: hi + off, raw, ms: {decompress: t1 - t0, read: performance.now() - t1}};
}

// the last logs message() opened, decompressed and indexed: scrubbing within a segment decodes without reloading
const opened = [];

async function message(key, url, service, mono, own) {
  const s = (await readerFor(own)).schema;
  let log = opened.find((l) => l.key === key && l.schema === s);
  if (!log) {
    const bytes = withDecompressed(new Uint8Array((await load(key, url, () => {})).buf), (b) => b.slice());
    opened.unshift(log = {key, schema: s, bytes, idx: index(s, bytes)});
    opened.length = Math.min(opened.length, 2);
  }
  return messageAt(s, log.bytes, log.idx, service, mono);
}

// a boot log read as text for the viewer, downloaded each time: they're small, and rarely opened twice
async function boot(url) {
  const {buf} = await download(url, () => {});
  await codec;  // decode() reads by the schema itself, no Reader
  return withDecompressed(new Uint8Array(buf), (bytes) => bootLog(schema, bytes));
}

const fetches = new Map();  // id → AbortController of a fetch in flight
let live = null;  // a live session's slots (slots.js), in a worker of its own

// a live session's ops; its packs go out unasked, as {live: pack}. live-free hands back the store's spent pack buffers,
// so this thread's GC frees them, not the main thread's
async function liveOp(op, {id, span, buf, msgs, n, opts}) {
  if (op === "live-start") {
    live = new Slots({span, schema, parse, post: (pack, transfer) => postMessage({live: pack}, transfer)});
  } else if (op === "live-events") {
    live.events(new Uint8Array(buf));
  } else if (op === "live-json") {
    live.json(msgs);
  } else if (op === "live-drop") {
    live.drop(n);
  } else if (op === "live-keep") {
    live.keep = opts.keep;
  } else if (op === "live-read") {
    const r = await live.read(n, opts.only, opts.can);
    postMessage({id, result: r}, [...r.bufs, r.recs.buffer, ...r.canBufs]);
  } else if (op === "live-message") {
    const bytes = live.bytes(n), {schema: s} = await ready;
    postMessage({id, result: messageAt(s, bytes, index(s, bytes), opts.service, opts.mono)});
  } else if (op === "live-save") {
    const bytes = live.save();
    postMessage({id, result: bytes}, [bytes.buffer]);
  } else if (op === "live-free") {
    // nothing to do: the buffers came with the message, and go with it
  }
}

onmessage = async ({data: {id, op, key, url, buf, opts, ...rest}}) => {
  try {
    if (op.startsWith("live-")) {
      await liveOp(op, {id, buf, opts, ...rest});
    } else if (op === "message") {
      postMessage({id, message: await message(key, url, opts.service, opts.mono, opts.schema)});
    } else if (op === "boot") {
      postMessage({id, boot: await boot(url)});
    } else if (op === "abort") {
      fetches.get(id)?.abort();
    } else if (op === "fetch") {
      const ctl = new AbortController();
      fetches.set(id, ctl);
      const r = await load(key, url, (progress) => postMessage({id, progress}), ctl.signal, buf && new Uint8Array(buf)).finally(() => fetches.delete(id));
      postMessage({id, ...r}, [r.buf]);
    } else {
      // a cached log that doesn't parse is dropped, so a retry downloads it again
      const r = await parse(buf, opts).catch(async (e) => {
        if (key) await opfs.remove(key);
        throw e;
      });
      postMessage({id, ...r}, [...r.bufs, r.recs.buffer, ...r.canBufs]);
    }
  } catch (e) {
    if (e.partial) postMessage({id, partial: e.partial}, [e.partial]);
    else postMessage({id, error: e?.message ?? String(e)});
  }
};
