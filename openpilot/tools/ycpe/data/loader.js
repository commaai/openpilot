// segment scheduler on the main thread: a pool of pipeline workers fed from a priority queue

export class Loader {
  constructor(size = Math.max(1, (navigator.hardwareConcurrency || 4) - 2)) {
    this.workers = Array.from({length: size}, () => new Worker(new URL("./worker.js", import.meta.url), {type: "module"}));
    this.idle = [...this.workers];
    this.queue = [];  // jobs, lowest prio first
    this.running = new Map();
    this.nextId = 0;
    for (const w of this.workers) w.onmessage = ({data}) => this.done(w, data);
  }

  // resolves with the worker's {services, bytes, raw, ms}; tag identifies the job to reprioritize()
  load(url, prio = 0, tag = url) {
    return new Promise((resolve, reject) => {
      this.queue.push({id: this.nextId++, url, prio, tag, resolve, reject});
      queueMicrotask(() => this.pump());  // a burst of loads is ranked before any starts
    });
  }

  // re-ranks queued jobs; fn(tag) gives the new prio
  reprioritize(fn) {
    for (const j of this.queue) j.prio = fn(j.tag);
    this.pump();
  }

  pump() {
    this.queue.sort((a, b) => a.prio - b.prio);
    while (this.idle.length && this.queue.length) {
      const job = this.queue.shift(), w = this.idle.pop();
      this.running.set(job.id, job);
      w.postMessage({id: job.id, url: job.url});
    }
  }

  done(w, data) {
    const job = this.running.get(data.id);
    this.running.delete(data.id);
    this.idle.push(w);
    if (data.error) job.reject(new Error(data.error));
    else job.resolve(data);
    this.pump();
  }

  get pending() { return this.queue.length + this.running.size; }

  dispose() {
    for (const w of this.workers) w.terminate();
    for (const j of [...this.queue, ...this.running.values()]) j.reject(new Error("disposed"));
    this.queue = [];
    this.running.clear();
  }
}
