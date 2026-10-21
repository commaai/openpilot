// segment scheduler on the main thread: a few downloads (or OPFS cache reads) at a time, highest priority first,
// feeding a small pool of parse workers. The workers do the I/O too; bytes only pass through here by transfer.

const FETCHES = 3;  // downloads in flight, few so the nearest finish first; the focus segment's runs alone
const IDLE = 3000;  // ms before an idle worker exits; each holds ~140 MB at its high-water mark
const WARM = 30000;  // ms the last one stays, so a scrub or route switch parses with JIT-warm code
const byPrio = (a, b) => a.prio - b.prio;
let spare = null;  // a disposed loader's warm worker, for the next one (the route switched)
export const POOL = Math.min(8, Math.max(1, (navigator.hardwareConcurrency || 4) - 2));

export class Loader {
  constructor(size = POOL) {
    this.size = size;
    this.workers = [];
    this.idle = [];
    this.calls = new Map();
    this.nextId = 0;
    this.queue = [];  // waiting to fetch
    this.ready = [];  // fetched, waiting for a worker
    this.fetching = new Set();  // jobs downloading (or reading the cache)
    this.cancelled = new WeakSet();  // owners whose jobs are dropped
    this.downloads = new Map();  // key → fetch in flight, so a repeated segment waits and then reads the cache
    this.next = 0;
  }

  spawn() {
    const w = this.worker();
    this.workers.push(w);
    this.release(w);
    return w;
  }

  worker() {
    const w = spare || new Worker(new URL("./worker.js", import.meta.url), {type: "module"});
    clearTimeout(w.reap);
    spare = null;
    w.calls = 0;
    w.onmessage = ({data}) => {
      w.started = true;
      const call = this.calls.get(data.id);
      if (data.progress) return call.progress?.(data.progress);
      this.calls.delete(data.id);
      w.calls--;
      if (data.error) call.reject(new Error(data.error));
      else call.resolve(data);
    };
    w.onerror = w.onmessageerror = (e) => this.lost(w, w.started ? `the log worker failed: ${e.message || e.type}` : "couldn't start the log worker");
    return w;
  }

  // a worker that failed to load or broke: its calls fail and it leaves the pool
  lost(w, error) {
    w.lost = true;
    for (const [id, call] of this.calls) {
      if (call.w !== w) continue;
      this.calls.delete(id);
      call.reject(new Error(error));
    }
    w.terminate();
    clearTimeout(w.reap);
    for (const list of [this.workers, this.idle]) if (list.includes(w)) list.splice(list.indexOf(w), 1);
    if (this.inspector === w) this.inspector = null;
    if (spare === w) spare = null;
  }

  // a parse worker: an idle one, or a new one while under the pool size
  take() {
    if (!this.idle.length && this.workers.length < this.size) this.spawn();
    return this.idle.pop();
  }

  // idle workers exit after a while, or at once while the pool is over size; the last one stays warm for longer
  release(w) {
    if (w.lost) return;
    this.idle.push(w);
    w.used = performance.now();
    this.reap(w);
  }

  reap(w) {
    clearTimeout(w.reap);
    const left = () => this.workers.length > this.size ? 0 : (this.workers.length > 1 ? IDLE : WARM) - (performance.now() - w.used);
    w.reap = setTimeout(() => {
      if (!this.idle.includes(w)) return;
      if (w.calls || left() > 0) return this.reap(w);  // still fetching, or not idle long enough
      w.terminate();
      this.idle.splice(this.idle.indexOf(w), 1);
      this.workers.splice(this.workers.indexOf(w), 1);
    }, w.calls ? IDLE : Math.max(left(), 0));
  }

  // fewer parse workers once the focus area is in: each holds its high-water memory until it exits
  resize(size) {
    if (size === this.size) return;
    this.size = size;
    for (const w of this.idle) this.reap(w);
    this.pump();
  }

  call(w, msg, transfer = [], progress = null) {
    const id = this.nextId++;
    const p = new Promise((resolve, reject) => this.calls.set(id, {resolve, reject, progress, w}));
    w.calls++;
    w.postMessage({...msg, id}, transfer);
    return Object.assign(p, {id});
  }

  // parses the log cached at key (or downloaded from url) with worker options opts (or opts() when it starts);
  // resolves with the worker's pack plus {bytes, cached, stored, ms}. tag identifies the load to reprioritize(), owner
  // to reprioritize() and cancel() (stores sharing a loader each touch only their own loads), and
  // report(event) hears {loaded, total} while downloading, {retry, of, error} before a retry, {paused: true} when a
  // better job took its place, and {parsing: true}
  load(key, url, opts, prio = 0, tag = key, report = () => {}, owner = null) {
    return new Promise((resolve, reject) => {
      this.queue.push({key, url, opts, prio, tag, report, owner, resolve, reject});
      queueMicrotask(() => this.pump());  // a burst of loads is ranked before any starts
    });
  }

  // re-ranks owner's jobs; fn(tag) gives the new prio
  reprioritize(fn, owner = null) {
    for (const j of [...this.queue, ...this.ready, ...this.fetching]) if (j.owner === owner) j.prio = fn(j.tag);
    this.pump();
  }

  // drops owner's jobs: waiting ones at once, downloads by aborting them, parses by ignoring their result
  cancel(owner) {
    const mine = (j) => j.owner === owner, err = new Error("cancelled");
    for (const j of [...this.queue, ...this.ready].filter(mine)) j.reject(err);
    this.queue = this.queue.filter((j) => !mine(j));
    this.ready = this.ready.filter((j) => !mine(j));
    for (const j of this.fetching) if (mine(j)) j.pause?.();
    this.cancelled.add(owner);
    this.pump();
  }

  pump() {
    this.ready.sort(byPrio);
    let w;
    while (this.ready.length && (w = this.take())) {
      const job = this.ready.shift(), worker = w;
      job.report({parsing: true});
      const opts = typeof job.opts === "function" ? job.opts() : job.opts;
      this.call(worker, {op: "parse", key: job.key, buf: job.buf, opts}, [job.buf])
        .then((r) => this.cancelled.has(job.owner) ? job.reject(new Error("cancelled"))
          : job.resolve({...r, bytes: job.bytes, cached: job.cached, stored: job.stored, ms: {...job.ms, ...r.ms}}), job.reject)
        .finally(() => {
          this.release(worker);
          this.pump();
        });
      job.buf = null;
    }
    // bandwidth goes to the best few: a better job pauses the worst running one, which later resumes where it was
    this.queue.sort(byPrio);
    const running = [...this.fetching].filter((j) => !j.paused).sort(byPrio);
    const limit = running[0]?.prio === 0 || this.queue[0]?.prio === 0 ? 1 : FETCHES;
    while (running.length > limit || (running.length === limit && this.queue[0]?.prio < running.at(-1).prio)) {
      running.pop().pause();
    }
    for (let n = running.length; n < limit && this.ready.length < this.size && this.queue.length; n++) {
      const job = this.queue.shift();
      this.fetching.add(job);
      this.fetch(job).then((done) => this.cancelled.has(job.owner) ? job.reject(new Error("cancelled"))
        : (done ? this.ready : this.queue).push(job), job.reject).finally(() => {
        this.fetching.delete(job);
        this.pump();
      });
    }
  }

  // any worker can fetch while it parses: the download and the OPFS read or write are async there
  async fetch(job) {
    const t0 = performance.now();
    await this.downloads.get(job.key);
    const w = this.workers.length < this.size ? this.spawn() : this.workers[this.next++ % this.workers.length];
    const got = this.call(w, {op: "fetch", key: job.key, url: job.url, buf: job.partial}, job.partial ? [job.partial] : [], job.report);
    const settled = got.then(() => {}, () => {});
    this.downloads.set(job.key, settled);
    job.paused = false;
    job.pause = () => {
      job.paused = true;
      w.postMessage({op: "abort", id: got.id});
    };
    try {
      ({buf: job.buf, cached: job.cached, stored: job.stored, partial: job.partial} = await got);
    } finally {
      if (this.downloads.get(job.key) === settled) this.downloads.delete(job.key);
    }
    if (job.partial) {
      job.report({paused: true});
      return false;
    }
    job.bytes = job.buf.byteLength;
    job.ms = {fetch: performance.now() - t0};
    return true;
  }

  // a whole message, decoded (with the log's own schema when one is in use) by a worker of its own that keeps the last
  // logs it opened (see worker.js)
  message(key, url, service, mono, schema) {
    const w = this.inspector ??= this.worker();
    clearTimeout(w.reap);
    // it exits a while after its last call settles; a call still in flight arms it when it does
    return this.call(w, {op: "message", key, url, opts: {service, mono, schema}}).finally(() => {
      if (w.calls || this.inspector !== w) return;
      w.reap = setTimeout(() => {
        w.terminate();
        this.inspector = null;
      }, 30000);
    }).then((r) => r.message);
  }

  dispose() {
    const warm = this.idle.find((w) => !w.calls);
    if (warm && !spare) {
      spare = warm;
      clearTimeout(warm.reap);
      warm.reap = setTimeout(() => { warm.terminate(); spare = null; }, WARM);
    }
    for (const w of this.workers) if (w !== spare) w.terminate();
    this.inspector?.terminate();
    const err = new Error("disposed");
    for (const c of [...this.queue, ...this.ready, ...this.calls.values()]) c.reject(err);
    this.queue = [];
    this.ready = [];
    this.calls.clear();
  }
}
