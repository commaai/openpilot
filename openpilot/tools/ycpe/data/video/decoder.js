// one camera's decoder worker: loads the per-segment Annex B files (from the OPFS cache, else streamed from the
// network and cached), decodes from the keyframe before each request and transfers the frames the page asked for.
// Frame n = segment * SEGMENT_FRAMES + index in the file. It reports every segment's state to the page.
import {AnnexB} from "./annexb.js";
import {openBackend} from "./backends.js";
import {SEGMENT_FRAMES} from "./frames.js";
import * as opfs from "../opfs.js";

const GAP = 40;   // decode forward through at most this many unwanted frames instead of restarting at a keyframe
const MEMORY = 120e6;  // bytes of segment files held in memory, nearest the playhead first
const AHEAD = 2;  // segments downloaded to disk ahead of the playhead while playing

let urls = [], keys = [];  // keys[n] names a segment in the OPFS cache; none for files that shouldn't be cached
const segs = new Map();  // n → {n, data, length, au: AnnexB, done, failed, abort}: the files in memory
const disk = new Map();  // n → bytes of the files cached on disk
const fetching = new Set();
let backend = null;      // null until the first segment's codec is known
let want = null;         // {at, to, back, playing, keyOnly}: send frames [at - back, to], decoding toward at first
let next = -1;           // the next frame to feed, -1 to restart at the keyframe before want.at
let out = 0;             // the next frame the decoder will output
let key = -1;            // the keyframe of the current run, always sent so scrubbing shows something at once
let fresh = true;        // the next chunk is the first of a run and needs the parameter sets
const sent = new Set();  // frames the page holds (or has in flight)
const pool = [];         // plane buffers the page is done with
let waitingOn = null, decoded = 0, received = 0;

onmessage = async ({data}) => {
  for (const n of data.closed || []) sent.delete(n);
  pool.push(...data.buffers || []);
  if (data.urls) {
    ({urls, keys} = data);
    await Promise.all(Array.from(urls, async (u, n) => {  // holes too: segments without a file
      const f = u && keys[n] && await opfs.read(keys[n]);
      if (f) disk.set(n, f.size);
      state(n, !u ? "missing" : f ? "disk" : "idle");
    }));
  }
  if (data.want) {
    want = data.want;
    // restart unless the decoder is already headed to at, or the page has it; scrubbing decodes just the keyframe
    if (want.keyOnly || (want.at < out ? !sent.has(want.at) : want.at > next + GAP)) next = -1;
    pump();
  }
};

// a segment's state for the page, at most once per percent of a download
const states = [];
function state(n, s, loaded = 0, total = 0) {
  const tag = `${s} ${total && Math.floor(100 * loaded / total)}`;
  if (states[n] !== tag) postMessage({seg: n, state: s, loaded, total});
  states[n] = tag;
}

// memory, disk and download rate, twice a second
let lastReceived = 0;
setInterval(() => {
  let memory = 0;
  for (const s of segs.values()) memory += s.data?.byteLength || 0;
  postMessage({memory, disk: [...disk.values()].reduce((a, b) => a + b, 0), rate: (received - lastReceived) * 2, decoded});
  lastReceived = received;
}, 500);

// the playhead's segment isn't here yet: s, or null once it is
function waiting(s) {
  const b = s && {n: s.n, loaded: s.length, total: s.total || 0};
  if (JSON.stringify(b) !== JSON.stringify(waitingOn)) postMessage({buffering: b});
  waitingOn = b;
}

function emit(n, frame) {
  out = n + 1;
  decoded++;
  if (want && (n === key || (n >= want.at - want.back && n <= want.to)) && !sent.has(n)) {
    sent.add(n);
    postMessage({n, frame}, frame.data ? [frame.data.buffer] : [frame]);
  } else {
    frame.close?.();
  }
}

function alloc(size) {
  const i = pool.findIndex((b) => b.byteLength === size);
  return new Uint8Array(i < 0 ? size : pool.splice(i, 1)[0]);
}

function segment(n) {
  let s = segs.get(n);
  if (!s) {
    s = {n, data: null, length: 0, au: new AnnexB(), done: false, failed: !urls[n], abort: new AbortController()};
    segs.set(n, s);
    if (!s.failed) load(s);
  }
  return s;
}

// drops the files farthest from the playhead until the rest fit in MEMORY; a download more than half done
// finishes into the disk cache rather than being thrown away
function trim() {
  const at = Math.floor(want.at / SEGMENT_FRAMES);
  let total = 0;
  for (const s of segs.values()) total += s.data?.byteLength || 0;
  for (const f of [...segs.values()].sort((a, b) => Math.abs(b.n - at) - Math.abs(a.n - at))) {
    if (total <= MEMORY || f.n === at) break;
    total -= f.data?.byteLength || 0;
    segs.delete(f.n);
    if (!f.done && !(f.length > f.total / 2)) f.abort.abort();
    if (urls[f.n]) state(f.n, disk.has(f.n) ? "disk" : "idle");
  }
}

async function load(s) {
  try {
    const cached = keys[s.n] && await opfs.read(keys[s.n]);
    if (cached) {
      s.data = new Uint8Array(await cached.arrayBuffer());
      s.length = s.data.length;
      trim();
      await scanned(s);
    } else {
      fetching.add(s.n);
      const resp = await fetch(urls[s.n], {signal: s.abort.signal});
      if (!resp.ok) throw new Error(`${resp.status}`);
      s.total = +resp.headers.get("content-length") || 0;  // none while the file is still being written
      s.data = new Uint8Array(s.total || 64 << 20);
      trim();
      for await (const chunk of resp.body) {
        if (s.length + chunk.length > s.data.length) {
          const grown = new Uint8Array(2 * (s.length + chunk.length));
          grown.set(s.data.subarray(0, s.length));
          s.data = grown;
        }
        s.data.set(chunk, s.length);
        s.length += chunk.length;
        received += chunk.length;
        state(s.n, "loading", s.length, s.total);
        await scanned(s);
      }
      fetching.delete(s.n);
      save(s.n, s.data.subarray(0, s.length)).then(() => segs.get(s.n) !== s && state(s.n, disk.has(s.n) ? "disk" : "idle"));
    }
    s.au.scan(s.data, s.length, true);
    s.done = true;
    if (segs.get(s.n) === s) state(s.n, "ready");
  } catch (e) {
    fetching.delete(s.n);
    if (s.abort.signal.aborted) return;
    s.failed = true;
    state(s.n, "error");
  }
  pump();
}

async function scanned(s) {
  s.au.scan(s.data, s.length, false);
  if (backend === null && s.au.codec) await start(s.au.codec);
  pump();
}

async function save(n, bytes) {
  if (!keys[n] || !await opfs.write(keys[n], bytes, "video")) return;
  disk.set(n, bytes.length);
  for (const k of await opfs.evict("video", keys[n].split("/").slice(0, 2).join("/"))) {
    const i = keys.indexOf(k);
    if (i >= 0) { disk.delete(i); if (!segs.has(i)) state(i, "idle"); }
  }
}

// download to disk only, one at a time behind every other download, to ride out the network during playback
async function prefetch(n) {
  if (!keys[n] || segs.has(n) || disk.has(n) || fetching.size) return;
  fetching.add(n);
  state(n, "queued");
  try {
    const resp = await fetch(urls[n]);
    if (!resp.ok) throw new Error(`${resp.status}`);
    const total = +resp.headers.get("content-length");
    const parts = [];
    let loaded = 0;
    for await (const chunk of resp.body) {
      parts.push(chunk);
      loaded += chunk.length;
      received += chunk.length;
      state(n, "loading", loaded, total);
    }
    await save(n, await new Blob(parts).bytes());
    if (!segs.has(n)) state(n, disk.has(n) ? "disk" : "idle");
  } catch {
    state(n, "error");
  }
  fetching.delete(n);
  pump();
}

async function start(codec) {
  backend = false;  // opening
  let unbuilt = false;
  backend = await openBackend(codec, {emit, pump, alloc, error: () => { next = -1; pump(); }}).catch((e) => {
    unbuilt = !!e.unbuilt;
    console.error(e);
    return null;
  });
  postMessage(backend ? {backend: backend.name} : {unsupported: codec, unbuilt});
}

function pump() {
  if (!want) return;
  const s = segment(Math.floor(want.at / SEGMENT_FRAMES));  // starts the download, which opens the backend
  if (want.playing) {
    segment(s.n + 1);
    for (let i = 2; i <= AHEAD + 1; i++) prefetch(s.n + i);
  }
  if (!backend) return waiting(s.done || s.failed ? null : s);
  if (next < 0 && !restart()) return;
  while (next <= (want.keyOnly ? key : want.to) && backend.ready()) {
    const s = segment(Math.floor(next / SEGMENT_FRAMES)), i = next % SEGMENT_FRAMES;
    if (s.failed || (s.done && i >= s.au.aus.length)) {  // missing or short segment: the next one starts with a keyframe
      if (s.n >= urls.length) return;
      next = (s.n + 1) * SEGMENT_FRAMES;
      continue;
    }
    const au = s.au.aus[i];
    if (!au) return waiting(next <= want.at ? s : null);  // still downloading; load() pumps again
    if (fresh && !au.key) { next = -1; return; }
    let data = s.data.subarray(au.off, au.off + au.len);
    if (fresh && au.off) data = concat(s.data.subarray(0, s.au.header), data);
    backend.decode(data, au.key, next);
    fresh = false;
    next++;
  }
  waiting(null);
}

// start over at the keyframe at or before want.at, once its segment has that far
function restart() {
  const s = segment(Math.floor(want.at / SEGMENT_FRAMES)), aus = s.au.aus;
  if (s.failed) return false;
  if (!aus.length || (!s.done && aus.length <= want.at % SEGMENT_FRAMES)) return waiting(s), false;
  let k = Math.min(want.at % SEGMENT_FRAMES, aus.length - 1);
  while (k > 0 && !aus[k].key) k--;
  backend.reset();
  next = out = key = s.n * SEGMENT_FRAMES + k;
  fresh = true;
  return true;
}

function concat(a, b) {
  const c = new Uint8Array(a.length + b.length);
  c.set(a);
  c.set(b, a.length);
  return c;
}
