// live sessions: a source (data/live/<kind>.js) hands a sink what it receives, events in rlog framing or webrtcd's JSON
// messages, and the sink batches them on a timer to a worker of their own (slots.js), which parses them exactly like a
// log. Its packs land in a LiveStore: a Store whose slots arrive while the session runs, the last one growing, and the
// oldest dropped past the window or the budget. Camera tracks go to the camera panes as they are.
import {SEGMENT} from "../api.js";
import {listeners} from "../dom.js";
import {service} from "./pack.js";
import {client} from "./rpc.js";
import {append} from "./slots.js";
import {Store} from "./store.js";

const FLUSH = 100;  // ms between batches to the worker
const WINDOW = 30 * 60;  // seconds of the newest data kept
const BUDGET = 1e9;  // column bytes in memory plus event bytes in the worker

// the session of a source: {name, label, source, store, route, video(camera, fn), save(), stop()}. opts are the
// source module's create() options; key names the session, path is its URL; window: the seconds kept (testing)
export async function openLive(kind, opts, {key = kind, path, window} = {}) {
  const {create} = await import(`./live/${kind}.js`);
  const source = create(opts);
  const name = `live|${key}`, store = new LiveStore(name, {window});
  const tracks = new Map(), videos = listeners();
  const session = {
    name, kind, path, source, store,
    label: source.label || kind,
    // fn(track) now and whenever the camera's track changes (null: it ended); returns an unsubscribe
    video(camera, fn) {
      fn(tracks.get(camera) || null);
      return videos.on((cam, track) => { if (cam === camera) fn(track); });
    },
    save: () => store.save(),
    stop() {
      source.stop();
      store.dispose();
    },
  };
  // an API-shaped route record, so the workspace and its panes take a session like a route
  const now = Date.now();
  session.route = {fullname: name, dongle_id: key.split("/")[0], live: session, start_time_utc_millis: now, end_time_utc_millis: now};
  source.start({
    events: (bytes) => store.events(bytes),
    json: (service, logMonoTime, valid, data) => store.json(service, logMonoTime, valid, data),
    // null: the camera's track went to another one (a one-track stream switched cameras)
    video(camera, track) {
      if (!track) return tracks.delete(camera) && videos.emit(camera, null);
      tracks.set(camera, track);
      track.addEventListener("ended", () => {
        if (tracks.get(camera) !== track) return;
        tracks.delete(camera);
        videos.emit(camera, null);
      });
      videos.emit(camera, track);
    },
  });
  return session;
}

export class LiveStore extends Store {
  constructor(name, {budget = BUDGET, window = WINDOW} = {}) {
    super(name, [], {budget});
    this.window = window;
    this.start = 0;  // route time of the oldest slot kept
    this.newest = 0;  // route time of the newest message in
    this.held = 0;  // event bytes the worker holds
    this.freed = [];  // pack buffers nothing here holds anymore, for the worker to free (release)
    this.pending = {bytes: new Uint8Array(1 << 16), len: 0, json: []};
    this.flusher = 0;
    this.worker = new Worker(new URL("./worker.js", import.meta.url), {type: "module"});
    this.call = client(this.worker, (e) => {
      const why = `the live worker failed: ${e.message || e.type}`;
      this.status.buffering({n: 0, text: why});
      return why;
    });
    // packs come unasked, as {live: pack}
    this.worker.addEventListener("message", ({data}) => data.live && this.arrived(data.live));
    this.worker.postMessage({op: "live-start", span: SEGMENT});
  }

  // whole capnp Events in rlog framing, copied: the source may reuse its buffer
  events(bytes) {
    append(this.pending, bytes);
    this.flushSoon();
  }

  json(service, logMonoTime, valid, data) {
    this.pending.json.push([service, logMonoTime, valid, data]);
    this.flushSoon();
  }

  flushSoon() {
    this.flusher ||= setTimeout(() => this.flush(), FLUSH);
  }

  flush() {
    this.flusher = 0;
    const p = this.pending;
    if (p.len) {
      const buf = p.bytes.slice(0, p.len).buffer;
      this.worker.postMessage({op: "live-events", buf}, [buf]);
      p.len = 0;
    }
    if (p.json.length) {
      this.worker.postMessage({op: "live-json", msgs: p.json});
      p.json = [];
    }
  }

  // slot n, and the empty ones before it
  slot(n) {
    const segs = this.route.segments;
    for (let k = segs.length; k <= n; k++) {
      segs[k] = {n: k, t0: k * SEGMENT, t1: (k + 1) * SEGMENT, status: k < n ? "missing" : "queued", bytes: 0, opts: {n: k, shift: 0}};
      if (k < n) this.publish(segs[k]);
    }
    this.status.count(segs.length);
    return segs[n];
  }

  // a slot's pack from the worker: its every column so far
  arrived(p) {
    if (this.disposed) return;
    if (p.error) return this.status.buffering({n: p.n, text: p.error});
    const seg = this.slot(p.n);
    if (seg.status === "missing") return;  // dropped meanwhile
    Object.assign(p, {full: true, kept: this.kept.length});
    this.held = p.held;
    seg.held = p.size;
    this.ingest(seg, p);
    if (p.t1 >= p.t0) [seg.t0, seg.t1] = [p.t0, p.t1];
    this.newest = Math.max(this.newest, seg.t1);
    this.route.duration = this.newest;
    if (p.can) this.can.ingest(seg.n, p.can, 0);
    this.trim();
    if (this.freed.length) this.worker.postMessage({op: "live-free", bufs: this.freed}, this.freed);
    this.freed = [];
    this.emit("live", this.newest);
  }

  // each parse supersedes the open slot's whole pack, megabytes a few times a second: freed here, they piled up as
  // young garbage until the main thread's GC swept them, 10-25 ms in a frame every few seconds. Their buffers go back
  // to the worker instead (detached, a view still held reads as empty: logs indexing picks up where it stopped)
  release(a) {
    const s = service(a.pack, a.k), buf = a.pack.bufs[s];
    super.release(a);
    if (!a.pack.bufs[s]) this.freed.push(buf);
  }

  // re-reads (lists asked for later) come from the event bytes the worker keeps
  load(seg, only, {can = !only} = {}) {
    return this.call("live-read", {n: seg.n, opts: {only, can}}).then((p) => Object.assign(p, {full: !only, kept: this.kept.length}));
  }

  request(path) {
    super.request(path);
    this.worker.postMessage({op: "live-keep", opts: {keep: this.keep}});
  }

  // the window slides: whole slots go, oldest first, past the window or over the budget (never the newest)
  trim() {
    const segs = this.route.segments;
    for (let n = Math.floor(this.start / SEGMENT); n < segs.length - 1; n++) {
      const over = this.bytes + this.can.bytes + this.held > this.budget;
      if (!over && this.newest - segs[n].t1 <= this.window) break;
      if (segs[n].status === "missing") this.start = (n + 1) * SEGMENT;
      else this.drop(segs[n]);
    }
  }

  drop(seg) {
    let removed = false;
    for (const e of this.entries.values()) {
      const a = e.at[seg.n];
      if (!a) continue;
      this.release(a);
      delete e.at[seg.n];
      e.views[seg.n] = null;
      e.version++;
      if (!e.at.some(Boolean)) removed = this.entries.delete(e.path);
    }
    seg.status = "missing";
    this.publish(seg);
    this.can.drop(seg.n);
    this.worker.postMessage({op: "live-drop", n: seg.n});
    this.held -= seg.held;
    this.start = (seg.n + 1) * SEGMENT;
    if (removed) {
      this.listed = null;
      this.emit("paths");
    }
    this.emit("window", this.start);
  }

  // columns aren't evicted: the window's slots go instead (trim)
  evict() {}

  async message(service, t) {
    const seg = this.segmentAt(t);
    if (seg?.start === undefined) return null;
    const mono = (t + seg.start) * 1e9 + 1000;  // 1 µs over, for the round trip through route seconds
    const message = await this.call("live-message", {n: seg.n, opts: {service, mono}});
    return message?.[service] ?? null;
  }

  // every event received in the window as an rlog
  async save() {
    this.flush();
    return this.call("live-save");
  }

  dispose() {
    clearTimeout(this.flusher);
    this.worker.terminate();
    super.dispose();
  }
}
