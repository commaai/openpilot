// boot logs (loggerd's bootlog: an initData, then a boot event) as plain text for the viewer in pages/logs.js. Runs in
// the log worker; the bytes are decompressed already
import {decode, messages} from "./message.js";

const lossy = new TextDecoder(), strict = new TextDecoder("utf-8", {fatal: true});
// pstore files and command outputs can be NUL-padded
const text = (v) => (typeof v === "string" ? v : v ? lossy.decode(v) : "").replaceAll("\0", "");

// a param's value: its text, or a note of its size when it's binary
function param(v) {
  if (typeof v === "string" || !v) return v || "";
  try {
    return strict.decode(v);
  } catch {
    return `(${v.length} bytes of binary)`;
  }
}

// a decoded Map(Text, Data) as [[key, text]]
const entries = (map, fn = text) => (map?.entries || []).map((e) => [text(e.key), fn(e.value)]);

// {wallTimeNanos, launchLog, commands, pstore, lastKmsg, lastPmsg, init: {...} | null}; throws on a file that isn't one
export function bootLog(schema, bytes) {
  let boot = null, init = null, n = 0, end = 0;
  for (const [at, size] of messages(bytes)) {
    let ev;
    try {
      ev = decode(schema, bytes.subarray(at, at + size));
    } catch (e) {
      // a pointer out of the message's bounds
      if (e instanceof RangeError) throw new Error(`corrupt message ${n + 1}: ${e.message}`);
      throw e;
    }
    if (ev?.boot) boot = ev.boot;
    else if (ev?.initData) init = ev.initData;
    n++;
    end = at + size;
  }
  // messages() stops short at a tail that isn't a whole message: one cut off, or bytes that aren't a message at all
  if (end < bytes.length && !boot) {
    const nseg = bytes.length - end >= 4 ? new DataView(bytes.buffer, bytes.byteOffset + end).getUint32(0, true) + 1 : 1;
    if (nseg <= 64) throw new Error(`truncated after ${n} message${n === 1 ? "" : "s"}`);
    throw new Error(n ? `corrupt after message ${n}` : "not a capnp log");
  }
  if (!n) throw new Error("the file is empty");
  if (!boot) throw new Error(`no boot event in its ${n} message${n === 1 ? "" : "s"}: not a boot log, or from an unsupported schema`);
  return {
    wallTimeNanos: boot.wallTimeNanos,
    launchLog: boot.launchLog,
    commands: entries(boot.commands),
    pstore: entries(boot.pstore),
    lastKmsg: text(boot.deprecated?.lastKmsg),
    lastPmsg: text(boot.deprecated?.lastPmsg),
    init: init && {
      version: init.version, gitBranch: init.gitBranch, gitCommit: init.gitCommit, dirty: init.dirty,
      deviceType: init.deviceType, osVersion: init.osVersion, kernelVersion: init.kernelVersion,
      kernelArgs: init.kernelArgs || [], commands: entries(init.commands), params: entries(init.params, param),
    },
  };
}
