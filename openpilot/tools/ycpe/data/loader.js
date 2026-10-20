// segment scheduler on the main thread: a few downloads (or OPFS cache reads) at a time, highest priority first,
// feeding a small pool of parse workers. The workers do the I/O too; bytes only pass through here by transfer.

const FETCHES = 6;  // downloads in flight; the first two run alone so the focus segment gets the bandwidth
const IDLE = 10000;  // ms before an idle worker exits; each holds ~90 MB of wasm memory at its high-water mark
const byPrio = (a, b) => a.prio - b.prio;

export class Loader {
  constructor(size = Math.min(8, Math.max(1, (navigator.hardwareConcurrency || 4) - 2))) {
    this.size = size;
    this.workers = [];
    this.idle = [];
    this.calls = new Map();
    this.nextId = 0;
    this.queue = [];  // waiting to fetch
    this.ready = [];  // fetched, waiting for a worker
    this.fetching = 0;
    this.fetched = 0;
    this.parsing = 0;
    this.downloads = new Map();  // key → fetch in flight, so a repeated segment waits and then reads the cache
    this.next = 0;
  }

  spawn() {
    const w = new Worker(new URL("./worker.js", import.meta.url), {type: "module"});
    w.calls = 0;
    w.onmessage = ({data}) => {
      const call = this.calls.get(data.id);
      this.calls.delete(data.id);
      w.calls--;
      if (data.error) call.reject(new Error(data.error));
      else call.resolve(data);
    };
    this.workers.push(w);
    this.release(w);
    return w;
  }

  // a parse worker: an idle one, or a new one while under the pool size
  take() {
    if (!this.idle.length && this.workers.length < this.size) this.spawn();
    return this.idle.pop();
  }

  release(w) {
    this.idle.push(w);
    this.reap(w);
  }

  reap(w) {
    clearTimeout(w.reap);
    w.reap = setTimeout(() => {
      if (!this.idle.includes(w)) return;
      if (w.calls) return this.reap(w);  // still fetching
      w.terminate();
      this.idle.splice(this.idle.indexOf(w), 1);
      this.workers.splice(this.workers.indexOf(w), 1);
    }, IDLE);
  }

  call(w, msg, transfer = []) {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.calls.set(id, {resolve, reject});
      w.calls++;
      w.postMessage({...msg, id}, transfer);
    });
  }

  // parses the log cached at key (or downloaded from url) with worker options opts; resolves with the worker's pack
  // plus {bytes, cached, ms}. tag identifies the load to reprioritize()
  load(key, url, opts, prio = 0, tag = key) {
    return new Promise((resolve, reject) => {
      this.queue.push({key, url, opts, prio, tag, resolve, reject});
      queueMicrotask(() => this.pump());  // a burst of loads is ranked before any starts
    });
  }

  // re-ranks waiting jobs; fn(tag) gives the new prio
  reprioritize(fn) {
    for (const j of this.queue) j.prio = fn(j.tag);
    for (const j of this.ready) j.prio = fn(j.tag);
    this.pump();
  }

  pump() {
    this.ready.sort(byPrio);
    let w;
    while (this.ready.length && (w = this.take())) {
      const job = this.ready.shift(), worker = w;
      this.parsing++;
      this.call(worker, {op: "parse", buf: job.buf, opts: job.opts}, [job.buf])
        .then((r) => job.resolve({...r, bytes: job.bytes, cached: job.cached, ms: {...job.ms, ...r.ms}}), job.reject)
        .finally(() => {
          this.parsing--;
          this.release(worker);
          this.pump();
        });
      job.buf = null;
    }
    this.queue.sort(byPrio);
    while (this.fetching < Math.min(FETCHES, 2 + this.fetched) && this.ready.length < this.size && this.queue.length) {
      const job = this.queue.shift();
      this.fetching++;
      this.fetch(job).then(() => this.ready.push(job), job.reject).finally(() => {
        this.fetching--;
        this.fetched++;
        this.pump();
      });
    }
  }

  // any worker can fetch while it parses: the download and the OPFS read or write are async there
  async fetch(job) {
    const t0 = performance.now();
    await this.downloads.get(job.key);
    const w = this.workers.length < this.size ? this.spawn() : this.workers[this.next++ % this.workers.length];
    const got = this.call(w, {op: "fetch", key: job.key, url: job.url});
    const settled = got.then(() => {}, () => {});
    this.downloads.set(job.key, settled);
    try {
      ({buf: job.buf, cached: job.cached} = await got);
    } finally {
      if (this.downloads.get(job.key) === settled) this.downloads.delete(job.key);
    }
    job.bytes = job.buf.byteLength;
    job.ms = {fetch: performance.now() - t0};
  }

  get pending() { return this.queue.length + this.fetching + this.ready.length + this.parsing; }

  dispose() {
    for (const w of this.workers) w.terminate();
    const err = new Error("disposed");
    for (const c of [...this.queue, ...this.ready, ...this.calls.values()]) c.reject(err);
    this.queue = [];
    this.ready = [];
    this.calls.clear();
  }
}
