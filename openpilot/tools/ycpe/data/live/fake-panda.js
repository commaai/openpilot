// a navigator.usb stand-in with one panda, for tests and ?fake-panda: it replays fixtures/panda.bin.gz (two seconds of
// the demo route's CAN, packed by panda's Python library) in real time, in a loop, and answers the control reads
// with the fixture's health packets. It serves transfers as the firmware's comms_can_read does: whole packets up to
// the requested length, the one that doesn't fit split across transfers until a comms reset drops its rest.
// Tests drive it: unplug(), plug(), corrupt() (flips a byte in the next transfer), babble() (fails the next one).
const DLC_TO_LEN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64];
const ALLOWED_OUT = new Set([0xc0]);  // the comms reset; anything else written is a bug in panda.js
const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

export async function loadFixture(gz) {
  gz ??= new Uint8Array(await (await fetch(new URL("./fixtures/panda.bin.gz", import.meta.url))).arrayBuffer());
  const bytes = new Uint8Array(await new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
  const dv = new DataView(bytes.buffer), n = dv.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + n)));
  const chunks = [];
  for (let at = 4 + n; at < bytes.length;) {
    const t = dv.getUint32(at, true) / 1000, len = dv.getUint32(at + 4, true);
    chunks.push({t, data: bytes.subarray(at + 8, at + 8 + len)});
    at += 8 + len;
  }
  return {...header, chunks, duration: chunks.at(-1).t + 10};
}

class FakePanda {
  constructor(fixture, max) {
    Object.assign(this, {fixture, max, vendorId: 0x3801, productId: 0xddcc, serialNumber: fixture.serial, productName: "panda (fake)"});
    this.opened = false;
    this.configuration = null;
    this.connected = true;
    this.requests = [];  // every control request, for tests: [direction, request, value]
  }

  async open() {
    this.check();
    this.opened = true;
    this.start = performance.now();
    Object.assign(this, {next: 0, loop: 0, queue: [], tail: null, flip: false, fail: false});
  }

  async close() { this.opened = false; }
  async selectConfiguration(n) { this.configuration = {configurationValue: n}; }
  async claimInterface() { this.check(); }
  async releaseInterface() {}
  async clearHalt() {}

  check() {
    if (!this.connected) throw new DOMException("The device was disconnected.", "NotFoundError");
  }

  async controlTransferIn(setup, length) {
    this.check();
    this.requests.push(["in", setup.request, setup.value]);
    const {versions, hwType, health, canHealth} = this.fixture;
    const reply = {0xdd: () => new Uint8Array(new Uint32Array(versions).buffer), 0xc1: () => Uint8Array.of(hwType), 0xd2: () => b64(health),
                   0xc2: () => b64(canHealth[setup.value])}[setup.request];
    if (setup.requestType !== "vendor" || !reply) return {status: "stall", data: new DataView(new ArrayBuffer(0))};
    const data = reply().slice(0, length);
    return {status: "ok", data: new DataView(data.buffer)};
  }

  async controlTransferOut(setup) {
    this.check();
    this.requests.push(["out", setup.request, setup.value]);
    if (!ALLOWED_OUT.has(setup.request)) throw new Error(`fake panda: unexpected write 0x${setup.request.toString(16)}`);
    this.tail = null;
    return {status: "ok", bytesWritten: 0};
  }

  async transferOut() {
    throw new Error("fake panda: receive only");
  }

  // what arrived since the last call, up to length; returns at once like the firmware, empty when nothing's queued
  async transferIn(endpoint, length) {
    await new Promise((done) => setTimeout(done, 1));  // a USB round trip
    this.check();
    if (!this.opened) throw new DOMException("The device must be opened first.", "InvalidStateError");
    if (endpoint !== 1) return {status: "stall", data: new DataView(new ArrayBuffer(0))};
    if (this.fail) {
      this.fail = false;
      return {status: "babble", data: new DataView(new ArrayBuffer(0))};
    }
    const {chunks, duration} = this.fixture, now = performance.now() - this.start;
    while (chunks[this.next].t + this.loop * duration <= now) {
      this.queue.push(chunks[this.next].data);
      if (++this.next === chunks.length) [this.next, this.loop] = [0, this.loop + 1];
    }
    const out = new Uint8Array(Math.min(length, this.max)), cap = out.length;
    let pos = 0;
    if (this.tail) {
      const n = Math.min(cap, this.tail.length);
      out.set(this.tail.subarray(0, n));
      this.tail = n < this.tail.length ? this.tail.subarray(n) : null;
      pos = n;
    }
    while (!this.tail && pos < cap && this.queue.length) {
      const chunk = this.queue[0];
      let at = 0;
      while (at < chunk.length && pos < cap) {
        const len = 6 + DLC_TO_LEN[chunk[at] >> 4], n = Math.min(len, cap - pos);
        out.set(chunk.subarray(at, at + n), pos);
        if (n < len) this.tail = chunk.subarray(at + n, at + len);
        pos += n;
        at += len;
      }
      if (at < chunk.length) this.queue[0] = chunk.subarray(at);
      else this.queue.shift();
    }
    if (this.flip && pos) {
      this.flip = false;
      out[pos >> 1] ^= 0x40;
    }
    return {status: "ok", data: new DataView(out.buffer, 0, pos)};
  }
}

// navigator.usb with the fake panda attached; granted: whether it's already permitted (else connect() picks it);
// max: the most one transfer returns, to split packets across transfers
class FakeUsb extends EventTarget {
  constructor(fixture, granted, max) {
    super();
    Object.assign(this, {fixture, permitted: granted, max});
    this.device = new FakePanda(fixture, max);
  }

  async getDevices() { return this.permitted && this.device.connected ? [this.device] : []; }

  async requestDevice({filters}) {
    const d = this.device;
    if (!d.connected || !filters.some((f) => f.vendorId === d.vendorId && f.productId === d.productId)) {
      throw new DOMException("No device selected.", "NotFoundError");
    }
    this.permitted = true;
    return d;
  }

  fire(type) { this.dispatchEvent(Object.assign(new Event(type), {device: this.device})); }

  unplug() {
    Object.assign(this.device, {connected: false, opened: false});
    this.fire("disconnect");
  }

  // a replug is a new USBDevice, as in Chrome
  plug() {
    this.device = new FakePanda(this.fixture, this.max);
    if (this.permitted) this.fire("connect");
  }

  corrupt() { this.device.flip = true; }
  babble() { this.device.fail = true; }
}

export async function fakeUsb({fixture, granted = true, max = 16384} = {}) {
  return new FakeUsb(fixture ?? await loadFixture(), granted, max);
}
