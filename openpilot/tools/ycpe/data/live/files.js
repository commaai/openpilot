// the "files" data channel of a device's webrtcd (SPEC.md Live): a read-only view of its log root. Requests and answers
// are JSON with an id; a read answers {id, size} (the file's size), then binary chunks each led by an 8-byte header
// (u32 id, u32 offset into the file, little-endian), then {id, done} or {id, error}. The device streams one read at a
// time, so reads queue here.
const LIST_MS = 10000, STALL_MS = 15000;

export class FilesChannel {
  constructor(channel) {
    this.channel = channel;
    this.nextId = 1;
    this.pending = new Map();  // id → {resolve, reject, chunk?}
    this.queue = Promise.resolve();
    channel.binaryType = "arraybuffer";
    channel.addEventListener("message", (e) => this.message(e.data));
    channel.addEventListener("close", () => this.fail(new Error("the device closed the files channel")));
  }

  // → [{name, segments: [{n, files: {name: size}}]}]. A long listing comes in parts ({more: true} on all but the last),
  // and a route can go on in the next one
  list() {
    const id = this.nextId++, routes = [];
    const add = (part = []) => {
      for (const r of part) routes.at(-1)?.name === r.name ? routes.at(-1).segments.push(...r.segments) : routes.push(r);
    };
    return this.call(id, {id, op: "list"}, LIST_MS, "the device didn't list its files (an openpilot without the files channel?)", {part: add})
      .then((r) => (add(r.routes), routes));
  }

  // streams [offset, offset + length) of path ("<log id>--<n>/<file>"): onSize(file size), then onChunk(Uint8Array) in
  // file order. Aborting signal cancels it on the device
  read(path, {offset = 0, length, signal, onSize = () => {}, onChunk}) {
    const run = () => {
      if (signal?.aborted) throw signal.reason;
      const id = this.nextId++;
      let at = offset;
      const done = this.call(id, {id, op: "read", path, offset, ...(length != null ? {length} : {})}, STALL_MS, `reading ${path} stalled`, {
        size: onSize,
        chunk: (bytes, from) => {
          if (from !== at) throw new Error(`${path}: a chunk at ${from}, expected ${at}`);
          at += bytes.length;
          onChunk(bytes);
        },
      });
      const abort = () => {
        this.settle(id, null, signal.reason);
        if (this.channel.readyState === "open") this.channel.send(JSON.stringify({id, op: "cancel"}));
      };
      signal?.addEventListener("abort", abort, {once: true});
      return done.finally(() => signal?.removeEventListener("abort", abort));
    };
    const p = this.queue.then(run);
    this.queue = p.catch(() => {});
    return p;
  }

  // the whole file (or a range of it) as bytes, as a download would give them
  async bytes(path, opts = {}) {
    const chunks = [];
    let size = 0;
    await this.read(path, {...opts, onSize: (n) => size = n, onChunk: (c) => chunks.push(c)});
    const out = new Uint8Array(chunks.reduce((a, c) => a + c.length, 0));
    for (let k = 0, at = 0; k < chunks.length; at += chunks[k++].length) out.set(chunks[k], at);
    if (opts.length == null && out.length !== size - (opts.offset || 0)) throw new Error(`${path}: ${out.length} of ${size} bytes`);
    return out;
  }

  // a request whose answer must come within ms of it, and of each chunk
  call(id, msg, ms, stalled, handlers = {}) {
    return new Promise((resolve, reject) => {
      const entry = {resolve, reject, ...handlers, timer: 0};
      entry.alive = () => {
        clearTimeout(entry.timer);
        entry.timer = setTimeout(() => this.settle(id, null, new Error(stalled)), ms);
      };
      this.pending.set(id, entry);
      entry.alive();
      if (this.channel.readyState !== "open") return this.settle(id, null, new Error("the files channel isn't open"));
      this.channel.send(JSON.stringify(msg));
    });
  }

  settle(id, value, error) {
    const entry = this.pending.get(id);
    if (!entry) return;
    this.pending.delete(id);
    clearTimeout(entry.timer);
    error ? entry.reject(error) : entry.resolve(value);
  }

  message(data) {
    if (typeof data !== "string") {
      const head = new DataView(data, 0, 8), entry = this.pending.get(head.getUint32(0, true));
      if (!entry?.chunk) return;  // a cancelled read's tail
      entry.alive();
      try {
        entry.chunk(new Uint8Array(data, 8), head.getUint32(4, true));
      } catch (e) {
        this.settle(head.getUint32(0, true), null, e);
      }
      return;
    }
    let msg;
    try {
      msg = JSON.parse(data);
    } catch {
      return;
    }
    const entry = this.pending.get(msg.id);
    if (!entry) return;
    if (msg.error) return this.settle(msg.id, null, Object.assign(new Error(msg.error), {device: true}));
    if (msg.more && entry.part) {
      entry.alive();
      return entry.part(msg.routes);
    }
    if (!entry.chunk || msg.done) return this.settle(msg.id, msg);
    if (msg.size != null) {
      entry.alive();
      entry.size(msg.size);
    }
  }

  fail(error) {
    for (const id of [...this.pending.keys()]) this.settle(id, null, error);
  }
}
