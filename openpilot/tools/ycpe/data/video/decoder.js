// one camera's decoder worker: streams the per-segment Annex B files, decodes from the keyframe before each
// request and transfers the frames the page asked for. Frame n = segment * SEGMENT_FRAMES + index in the file.
import {AnnexB} from "./annexb.js";
import {openBackend} from "./backends.js";
import {SEGMENT_FRAMES} from "./frames.js";

const GAP = 40;   // decode forward through at most this many unwanted frames instead of restarting at a keyframe
const KEEP = 3;   // segment files held in memory

let urls = [];
const segs = new Map();  // n → {n, data, length, au: AnnexB, done, failed, abort}
let backend = null;      // null until the first segment's codec is known
let want = null;         // {at, to, back, playing}: send frames [at - back, to], decoding toward at first
let next = -1;           // the next frame to feed, -1 to restart at the keyframe before want.at
let out = 0;             // the next frame the decoder will output
let key = -1;            // the keyframe of the current run, always sent so scrubbing shows something at once
let fresh = true;        // the next chunk is the first of a run and needs the parameter sets
const sent = new Set();  // frames the page holds (or has in flight)
const pool = [];         // plane buffers the page is done with
let decoded = 0, bytes = 0;

onmessage = ({data}) => {
  if (data.urls) urls = data.urls;
  for (const n of data.closed || []) sent.delete(n);
  pool.push(...data.buffers || []);
  if (data.want) {
    want = data.want;
    // restart unless the decoder is already headed to at, or the page has it
    if (want.at < out ? !sent.has(want.at) : want.at > next + GAP) next = -1;
    pump();
  }
};

function emit(n, frame) {
  out = n + 1;
  decoded++;
  if (want && (n === key || (n >= want.at - want.back && n <= want.to)) && !sent.has(n)) {
    sent.add(n);
    postMessage({n, frame, decoded, bytes}, frame.data ? [frame.data.buffer] : [frame]);
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
    const far = [...segs.values()].sort((a, b) => Math.abs(b.n - n) - Math.abs(a.n - n));
    for (const f of far.slice(0, Math.max(0, segs.size - KEEP))) { f.abort.abort(); segs.delete(f.n); }
  }
  return s;
}

async function load(s) {
  try {
    const resp = await fetch(urls[s.n], {signal: s.abort.signal});
    if (!resp.ok) throw new Error(`${resp.status}`);
    s.data = new Uint8Array(+resp.headers.get("content-length") || 64 << 20);
    for await (const chunk of resp.body) {
      if (s.length + chunk.length > s.data.length) {
        const grown = new Uint8Array(2 * (s.length + chunk.length));
        grown.set(s.data.subarray(0, s.length));
        s.data = grown;
      }
      s.data.set(chunk, s.length);
      s.length += chunk.length;
      bytes += chunk.length;
      s.au.scan(s.data, s.length, false);
      if (backend === null && s.au.codec) await start(s.au.codec);
      pump();
    }
    s.au.scan(s.data, s.length, true);
    s.done = true;
  } catch (e) {
    if (s.abort.signal.aborted) return;
    s.failed = true;
  }
  pump();
}

async function start(codec) {
  backend = false;  // opening
  backend = await openBackend(codec, {emit, pump, alloc, error: () => { next = -1; pump(); }}).catch((e) => console.error(e) ?? null);
  postMessage(backend ? {backend: backend.name} : {unsupported: codec});
}

function pump() {
  if (!want) return;
  segment(Math.floor(want.at / SEGMENT_FRAMES));  // starts the download, which opens the backend
  if (!backend || (next < 0 && !restart())) return;
  while (next <= want.to && backend.ready()) {
    const s = segment(Math.floor(next / SEGMENT_FRAMES)), i = next % SEGMENT_FRAMES;
    if (s.failed || (s.done && i >= s.au.aus.length)) {  // missing or short segment: the next one starts with a keyframe
      if (s.n >= urls.length) return;
      next = (s.n + 1) * SEGMENT_FRAMES;
      continue;
    }
    const au = s.au.aus[i];
    if (!au) return;  // still downloading; load() pumps again
    if (fresh && !au.key) { next = -1; return; }
    let data = s.data.subarray(au.off, au.off + au.len);
    if (fresh && au.off) data = concat(s.data.subarray(0, s.au.header), data);
    backend.decode(data, au.key, next);
    fresh = false;
    next++;
  }
  if (want.playing) segment(Math.floor(want.to / SEGMENT_FRAMES) + 1);  // stream the next segment ahead
}

// start over at the keyframe at or before want.at, once its segment has that far
function restart() {
  const s = segment(Math.floor(want.at / SEGMENT_FRAMES)), aus = s.au.aus;
  if (s.failed || !aus.length || (!s.done && aus.length <= want.at % SEGMENT_FRAMES)) return false;
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
