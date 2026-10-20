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
  let boot = null, init = null, n = 0;
  for (const [at, size] of messages(bytes)) {
    n++;
    const ev = decode(schema, bytes.subarray(at, at + size));
    if (ev?.boot) boot = ev.boot;
    else if (ev?.initData) init = ev.initData;
  }
  if (!n) throw new Error("not a capnp log: no messages in it");
  if (!boot) throw new Error(`no boot event in its ${n} message${n === 1 ? "" : "s"}: not a boot log, or from an unsupported schema`);
  return {
    wallTimeNanos: boot.wallTimeNanos,
    launchLog: boot.launchLog,
    commands: entries(boot.commands),
    pstore: entries(boot.pstore),
    lastKmsg: text(boot.deprecated?.lastKmsg),
    lastPmsg: text(boot.deprecated?.lastPmsg),
    init: init && {
      version: init.version, gitBranch: init.gitBranch, gitCommit: init.gitCommit, gitCommitDate: init.gitCommitDate, dirty: init.dirty,
      deviceType: init.deviceType, osVersion: init.osVersion, kernelVersion: init.kernelVersion, bootlogId: init.bootlogId,
      kernelArgs: init.kernelArgs || [], commands: entries(init.commands), params: entries(init.params, param),
    },
  };
}
