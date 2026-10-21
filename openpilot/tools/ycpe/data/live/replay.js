// a source for development and the harness: one segment of a route's rlog (fetched the way the store fetches it, so
// from the OPFS cache once seen) played back paced in real time, or at rate, and looping. Each lap moves the events'
// logMonoTimes on by the segment's length, so time only goes forward. The pacing runs in a worker (this file loaded as
// "replay.js?worker"), which sends the events due every TICK ms as one piece.
import {initCodec, decompress} from "../../codec/codec.js";
import {stateful} from "../../dom.js";
import {messages} from "../message.js";

const TICK = 50;  // ms

export function create({route, rate = 1, segment = 0}) {
  let worker = null, stopped = false;
  const source = {
    kind: "replay", label: `replay of ${route.replace("|", "/")}--${segment}`, state: "connecting", detail: "fetching the log",
    async start(sink) {
      try {
        const buf = await fetchLog(route, segment);
        if (stopped) return;
        worker = new Worker(new URL("./replay.js?worker", import.meta.url), {type: "module"});
        worker.onmessage = ({data}) => {
          if (data.error) return set("error", data.error);
          if (source.state !== "live") set("live", `${rate}×`);
          sink.events(new Uint8Array(data.events));
        };
        worker.postMessage({buf, rate}, [buf]);
      } catch (e) {
        set("error", e.message);
      }
    },
    stop() {
      stopped = true;
      worker?.terminate();
      set("closed");
    },
  };
  const set = stateful(source);
  return source;
}

// the segment's rlog (else qlog) bytes, through a pipeline worker's fetch: the OPFS cache, else a download into it
async function fetchLog(route, segment) {
  const [{routeRecord, routeLogs}, {Loader}] = await Promise.all([import("../sources.js"), import("../loader.js")]);
  const urls = await routeLogs(route, routeRecord(route)), e = urls[segment];
  if (!e) throw new Error(`no log for segment ${segment}`);
  const url = typeof e === "string" ? e : e.url, file = e.file ?? new URL(url, location.origin).pathname.split("/").at(-1);
  const loader = new Loader(1);
  try {
    const {buf} = await loader.call(loader.spawn(), {op: "fetch", key: `logs/${route.replace("|", "/")}/${segment}/${file}`, url});
    return buf;
  } finally {
    loader.dispose();
  }
}

if (import.meta.url.endsWith("?worker")) {
  let log = null, rate = 1, timer = 0;
  // where playback is: the next message, the lap, and the log time (s from the first message) at wall time `at`
  const play = {k: 0, lap: 0, pos: 0, at: 0};

  const tick = () => {
    const now = performance.now();
    play.pos += (now - play.at) / 1000 * rate;
    play.at = now;
    const {pace, span} = log;
    // whole laps due at once (a slow tab) are skipped, not replayed
    if (play.pos > span * 2) {
      play.lap += Math.floor(play.pos / span) - 1;
      play.pos = span + play.pos % span;
    }
    while (play.pos >= 0) {
      const k0 = play.k;
      while (play.k < pace.length && (pace[play.k] - pace[0]) / 1e9 <= play.pos) play.k++;
      if (play.k > k0) send(k0, play.k);
      if (play.k < pace.length) break;
      play.k = 0;
      play.lap++;
      play.pos -= span;
    }
  };

  // messages [k0, k1) as one copy, their logMonoTimes moved on by the laps so far
  const send = (k0, k1) => {
    const {bytes, at, size, mono, span} = log, shift = Math.round(play.lap * span * 1e9);
    const out = bytes.slice(at[k0], at[k1 - 1] + size[k1 - 1]), dv = new DataView(out.buffer);
    for (let k = k0; k < k1; k++) {
      const m = at[k] - at[k0], root = m + ((4 + 4 * (dv.getUint32(m, true) + 1) + 7) & ~7);
      const ev = root + 8 + (dv.getUint32(root, true) >> 2) * 8, t = mono[k] + shift;
      dv.setUint32(ev, t % 4294967296, true);
      dv.setUint32(ev + 4, Math.floor(t / 4294967296), true);
    }
    postMessage({events: out.buffer}, [out.buffer]);
  };

  onmessage = async ({data}) => {
    try {
      await initCodec();
      log = index(decompress(new Uint8Array(data.buf)));
      rate = data.rate;
      play.at = performance.now();
      timer = setInterval(tick, TICK);
    } catch (e) {
      clearInterval(timer);
      postMessage({error: e.message});
    }
  };
}

// each message's offset, size and logMonoTime (ns), in log order; pace: the times made to only go forward (a log's do,
// give or take), when each is due. span: the seconds a lap takes
function index(bytes) {
  const at = [], size = [], mono = [];
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (const [m, n] of messages(bytes)) {
    const root = m + ((4 + 4 * (dv.getUint32(m, true) + 1) + 7) & ~7), ev = root + 8 + (dv.getUint32(root, true) >> 2) * 8;
    at.push(m);
    size.push(n);
    mono.push(dv.getUint32(ev, true) + dv.getUint32(ev + 4, true) * 4294967296);
  }
  if (!at.length) throw new Error("no messages in the log");
  const pace = mono.slice();
  for (let k = 1; k < pace.length; k++) pace[k] = Math.max(pace[k], pace[k - 1]);
  return {bytes, at, size, mono, pace, span: (pace.at(-1) - pace[0]) / 1e9 + 0.01};
}
