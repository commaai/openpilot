// live source: a panda over WebUSB, receive only (panda/python's USB protocol). CAN comes from the bulk IN endpoint
// and health from control reads. It never sends CAN, changes the safety mode or bus speeds, flashes or resets the
// panda; its one write is the comms reset (0xc0) every host sends on connect, which clears only the firmware's
// half-sent packet. ?fake-panda swaps in the harness's fake device.
import {stateful} from "../../dom.js";
import {event, monoNanos} from "./encode.js";

export const FILTERS = [0xbbaa, 0x3801].flatMap((vendorId) => [0xddcc, 0xddee].map((productId) => ({vendorId, productId})));
// the packet versions panda/python expects (hashes of board/health.h and opendbc's can.h); the tests compare them with the fixture
export const HEALTH_VERSION = 0xffbc12f5, CAN_VERSION = 0x75abf276;
const DLC_TO_LEN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64];
const HEAD = 6;  // CANPACKET_HEAD_SIZE
const RECV_SIZE = 16384;  // what can_recv asks for: a full receive batch plus reserve
const PERIOD = 10;  // ms per can event, as pandad's 100 Hz loop
const HEALTH_PERIOD = 500;
const PANDA_TYPES = ["unknown", "white panda", "grey panda", "black panda", "pedal", "uno", "dos", "red panda", "red panda v2", "tres", "cuatro"];

// board/health.h's health_t and can_health_t, field by field, named as cereal's PandaState and PandaCanState
// (null: not in cereal)
const HEALTH = [["uptime", "u32"], ["voltage", "u16"], ["current", "u16"], ["safetyTxBlocked", "u32"], ["safetyRxInvalid", "u32"],
  ["txBufferOverflow", "u32"], ["rxBufferOverflow", "u32"], ["faults", "u32"], ["flags", "u16"], ["harnessStatus", "u8"], ["safetyModel", "u8"],
  ["safetyParam", "u16"], ["faultStatus", "u8"], ["alternativeExperience", "i16"], ["interruptLoad", "u8"], ["fanPower", "u8"],
  ["spiErrorCount", "u16"], ["sbu1Voltage", "u16"], ["sbu2Voltage", "u16"], ["soundOutputLevel", "u16"], [null, "u8"]];
const CAN_HEALTH = [["busOff", "u8"], ["busOffCnt", "u32"], ["errorWarning", "u8"], ["errorPassive", "u8"], ["lastError", "u8"],
  ["lastStoredError", "u8"], ["lastDataError", "u8"], ["lastDataStoredError", "u8"], ["receiveErrorCnt", "u8"], ["transmitErrorCnt", "u8"],
  ["totalErrorCnt", "u32"], ["totalTxLostCnt", "u32"], ["totalRxLostCnt", "u32"], ["totalTxCnt", "u32"], ["totalRxCnt", "u32"],
  ["totalFwdCnt", "u32"], [null, "u32"], ["canSpeed", "u16"], ["canDataSpeed", "u16"], ["canfdEnabled", "u8"], ["brsEnabled", "u8"],
  ["canfdNonIso", "u8"], ["irq0CallRate", "u32"], ["irq1CallRate", "u32"], ["irq2CallRate", "u32"], ["canCoreResetCnt", "u32"]];
const FLAGS = ["ignitionLine", "ignitionCan", "controlsAllowed", "powerSaveEnabled", "heartbeatLost", "safetyRxChecksInvalid", null, "nmiReset",
               "hardfaultReset"];
const GET = {u8: ["getUint8", 1], u16: ["getUint16", 2], i16: ["getInt16", 2], u32: ["getUint32", 4]};
const size = (fields) => fields.reduce((n, [, t]) => n + GET[t][1], 0);

function parse(fields, dv) {
  const out = {};
  let at = 0;
  for (const [name, type] of fields) {
    if (name) out[name] = dv[GET[type][0]](at, true);
    at += GET[type][1];
  }
  return out;
}

// a PandaState as pandad's fill_panda_state makes it, from the health and three CAN health packets
export function pandaState(health, canHealth, hwType) {
  const {flags, faults, ...h} = parse(HEALTH, health);
  const ps = {...h, pandaType: hwType, interruptLoad: h.interruptLoad / 255, sbu1Voltage: h.sbu1Voltage / 1000, sbu2Voltage: h.sbu2Voltage / 1000,
              faults: Array.from({length: 27}, (_, f) => f).filter((f) => faults & 1 << f)};
  FLAGS.forEach((name, bit) => { if (name) ps[name] = !!(flags & 1 << bit); });
  canHealth.forEach((dv, i) => { ps[`canState${i}`] = parse(CAN_HEALTH, dv); });
  return ps;
}

// CAN packets in, frames out ({address, dat, src}, src + 128 when returned, + 192 when rejected). A packet split
// across transfers waits for the rest. A bad checksum drops everything buffered (pandad's unpack_can_buffer), and
// push() says so, so the caller resets comms and the next transfer starts on a packet boundary.
export class Unpacker {
  tail = new Uint8Array(0);

  push(data) {
    let buf = data;
    if (this.tail.length) {
      buf = new Uint8Array(this.tail.length + data.length);
      buf.set(this.tail);
      buf.set(data, this.tail.length);
    }
    const frames = [];
    let pos = 0;
    while (buf.length - pos >= HEAD) {
      const end = pos + HEAD + DLC_TO_LEN[buf[pos] >> 4];
      if (end > buf.length) break;
      let x = 0;
      for (let i = pos; i < end; i++) x ^= buf[i];
      if (x) {
        this.tail = new Uint8Array(0);
        return {frames, ok: false};
      }
      const flags = buf[pos + 1];
      frames.push({
        address: (flags | buf[pos + 2] << 8 | buf[pos + 3] << 16 | buf[pos + 4] << 24) >>> 3,
        dat: buf.subarray(pos + HEAD, end),
        src: ((buf[pos] >> 1) & 7) + (flags & 2 ? 128 : 0) + (flags & 1 ? 192 : 0),
      });
      pos = end;
    }
    this.tail = buf.slice(pos);
    return {frames, ok: true};
  }
}

const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
const vendor = (request, value = 0) => ({requestType: "vendor", recipient: "device", request, value, index: 0});
const isPanda = (d) => FILTERS.some((f) => f.vendorId === d.vendorId && f.productId === d.productId);
const fakeRequested = () => new URLSearchParams(globalThis.location?.search).get("fake-panda");

// opts.usb: a navigator.usb stand-in (tests)
export function create(opts = {}) {
  const fake = opts.usb ? null : fakeRequested();
  const usbReady = opts.usb ? Promise.resolve(opts.usb)
    : fake !== null ? import("./fake-panda.js").then((m) => m.fakeUsb({granted: fake !== "click"})) : Promise.resolve(globalThis.navigator?.usb);
  let usb = null, sink = null, device = null, opening = null;

  const source = {
    kind: "panda",
    label: "panda",
    state: "closed",
    detail: "",
    stats: {frames: 0, checksumErrors: 0, overflows: 0},

    async start(s) {
      sink = s;
      usb = await usbReady;
      if (sink !== s) return;  // stopped meanwhile
      if (!usb) return set("error", "this browser has no WebUSB (use Chrome on desktop or Android)");
      usb.addEventListener("connect", onConnect);
      usb.addEventListener("disconnect", onDisconnect);
      const granted = (await usb.getDevices()).filter(isPanda);
      if (granted.length) await open(granted[0]);
      else set("connecting", "plug in a panda and connect it");
    },

    // needs a user gesture: the browser's device picker
    async connect() {
      if (!usb) return;
      let d;
      try {
        d = await usb.requestDevice({filters: FILTERS});
      } catch {
        return;  // picker closed
      }
      if (d !== device) await open(d);
    },

    async stop() {
      if (usb) {
        usb.removeEventListener("connect", onConnect);
        usb.removeEventListener("disconnect", onDisconnect);
      }
      sink = null;
      await close();
      set("closed", "");
    },
  };
  const set = stateful(source);

  async function open(d) {
    if (opening || !sink) return;
    await close();
    if ((d.productId & 0xf0) === 0xe0) return set("error", "this panda is in its bootstub; flash it with panda's tools first");
    const name = `panda ${d.serialNumber || ""}`.trim();
    set("connecting", `opening ${name}`);
    opening = d;
    try {
      await d.open();
      if (d.configuration === null) await d.selectConfiguration(1);
      await d.claimInterface(0);
      const versions = await control(d, 0xdd, 8);
      const hwType = (await control(d, 0xc1, 0x40)).getUint8(0);
      await d.controlTransferOut(vendor(0xc0));
      if (!sink) return await d.close();  // stopped meanwhile
      device = d;
      const health = versions.getUint32(0, true) === HEALTH_VERSION;
      const warn = versions.getUint32(4, true) !== CAN_VERSION ? " (its CAN packet version differs from this page's; update panda)"
        : !health ? " (no health: its packet version differs from this page's)" : "";
      set("live", `${PANDA_TYPES[hwType] || "panda"} ${d.serialNumber || ""}`.trim() + warn);
      receive(d);
      if (health) poll(d, hwType);
    } catch (e) {
      await d.close().catch(() => {});
      const hint = {NetworkError: " (is pandad or another tab using it?)", SecurityError: " (on Linux, panda's udev rules give access)"}[e.name] || "";
      set("error", `couldn't open ${name}: ${e.message}${hint}`);
    } finally {
      opening = null;
    }
  }

  async function close() {
    const d = device;
    device = null;
    if (!d?.opened) return;
    await d.releaseInterface(0).catch(() => {});
    await d.close().catch(() => {});
  }

  // a vendor control read, as a DataView
  async function control(d, request, length, value = 0) {
    const r = await d.controlTransferIn(vendor(request, value), length);
    if (r.status !== "ok") throw new Error(`control request 0x${request.toString(16)}: ${r.status}`);
    return r.data;
  }

  // one can Event per PERIOD tick: reads back to back while the panda has a full transfer waiting, else sleeps to the next tick
  // (except in a hidden tab, whose timers would starve the panda's queue)
  async function receive(d) {
    const unpacker = new Unpacker(), {stats} = source;
    let frames = [], next = performance.now() + PERIOD;
    while (device === d) {
      let r;
      try {
        r = await d.transferIn(1, RECV_SIZE);
      } catch (e) {
        if (device !== d) return;
        if (e.name === "NotFoundError") return onDisconnect({device: d});
        r = {status: "babble"};
      }
      if (device !== d) return;
      if (r.status === "stall") {
        await d.clearHalt("in", 1).catch(() => {});
        continue;
      }
      if (r.status !== "ok") {  // overflow or a transfer error: panda/python retries after 100 ms
        stats.overflows++;
        await sleep(100);
        continue;
      }
      const data = new Uint8Array(r.data.buffer, r.data.byteOffset, r.data.byteLength);
      const now = performance.now(), got = unpacker.push(data);
      for (const f of got.frames) frames.push(f);
      if (!got.ok || now >= next) {
        if (frames.length || !got.ok) {
          sink?.events(event(monoNanos(now), got.ok, "can", frames));
          stats.frames += frames.length;
        }
        frames = [];
        next = next + PERIOD < now ? now + PERIOD : next + PERIOD;  // a late tick starts a new schedule, as pandad's RateKeeper
      }
      if (!got.ok) {
        stats.checksumErrors++;
        await d.controlTransferOut(vendor(0xc0)).catch(() => {});
      }
      if (data.length < RECV_SIZE && !globalThis.document?.hidden) await sleep(next - performance.now());
    }
  }

  async function poll(d, hwType) {
    while (device === d) {
      const t = performance.now();
      try {
        const health = await control(d, 0xd2, size(HEALTH));
        const cans = [];
        for (let i = 0; i < 3; i++) cans.push(await control(d, 0xc2, size(CAN_HEALTH), i));
        if (device !== d) return;
        sink?.events(event(monoNanos(), true, "pandaStates", [pandaState(health, cans, hwType)]));
      } catch {
        if (device !== d) return;
      }
      await sleep(t + HEALTH_PERIOD - performance.now());
    }
  }

  function onConnect(e) {
    if (sink && !device && isPanda(e.device)) open(e.device);
  }

  function onDisconnect(e) {
    if (e.device !== device) return;
    device = null;
    set("connecting", "panda unplugged; waiting for it to come back");
  }

  return source;
}
