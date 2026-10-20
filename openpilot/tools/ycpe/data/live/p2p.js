// a route's files from its device, peer to peer over the "files" channel (SPEC.md Live): the "p2p" source in
// data/sources.js. Its URLs are served by /p2p-sw.js, which asks this page for the bytes, so the segment workers fetch,
// resume and cache them in OPFS like any download. The connection is the device's shared one (peer.js), let go a
// while after the last read.
import {api} from "../../api.js";
import {peer} from "./peer.js";

const IDLE_MS = 30000, LISTED_MS = 30000, KEPT_MS = 10 * 60e3, ONLINE_S = 120;
const PAGE = crypto.randomUUID().slice(0, 8);  // URLs name the page that can read them
const devices = new Map();  // dongle → {reads, idle, listed: {at, routes}}: this page's use of a device's connection

// the device's files channel, keeping the connection while it's used
async function files(dongle) {
  if (!devices.has(dongle)) devices.set(dongle, {reads: 0, idle: 0, listed: null});
  const user = devices.get(dongle);
  clearTimeout(user.idle);
  const conn = await peer(dongle).use(user, {files: true});
  return conn.filesChannel();
}

function idle(dongle) {
  const user = devices.get(dongle);
  if (!user || user.reads) return;
  clearTimeout(user.idle);
  user.idle = setTimeout(() => {
    devices.delete(dongle);
    peer(dongle).release(user);
  }, IDLE_MS);
}

async function listing(dongle) {
  const ch = await files(dongle), user = devices.get(dongle);
  if (!user.listed || performance.now() - user.listed.at > LISTED_MS) user.listed = {at: performance.now(), routes: ch.list()};
  try {
    return await user.listed.routes;
  } catch (e) {
    user.listed = null;
    throw e;
  } finally {
    idle(dongle);
  }
}

// the service worker that serves this page's URLs, and this page's answers to it
let registered = null;
const serviceWorker = () => registered ||= (async () => {
  const sw = navigator.serviceWorker;
  if (!sw) throw new Error("peer-to-peer files need service workers");
  sw.addEventListener("message", read);
  sw.startMessages();
  await sw.register("/p2p-sw.js");
  if (!sw.controller) await new Promise((resolve) => sw.addEventListener("controllerchange", resolve, {once: true}));
})().catch((e) => {
  registered = null;
  throw e;
});

// streams one file (or a range) to the service worker's port: {size}, {chunk}s, then {done} or {error}
async function read({data, ports: [port]}) {
  const ask = data?.p2p;
  if (!ask || ask.page !== PAGE || !port) return;
  const ctl = new AbortController();
  port.onmessage = (e) => e.data.cancel && ctl.abort(new Error("cancelled"));
  let user;
  try {
    const ch = await files(ask.dongle);
    user = devices.get(ask.dongle);
    user.reads++;
    await ch.read(ask.path, {offset: ask.offset, length: ask.length, signal: ctl.signal,
      onSize: (size) => port.postMessage({size}),
      onChunk: (bytes) => {
        const chunk = bytes.slice().buffer;
        port.postMessage({chunk}, [chunk]);
      }});
    port.postMessage({done: true});
  } catch (e) {
    // the device's own errors (no such file) are final; a dropped connection is worth the worker's retries
    if (!ctl.signal.aborted) port.postMessage({error: e.message, status: e.device ? 404 : 503});
  } finally {
    if (user) user.reads--;
    idle(ask.dongle);
  }
}

// whether the device reached athena lately, by a device record at most LISTED_MS old; false for devices this account
// can't see
const seen = new Map();  // dongle → {at, online: Promise}
function online(dongle) {
  if (!(performance.now() - seen.get(dongle)?.at < LISTED_MS)) {
    api.device.forget(dongle);
    seen.set(dongle, {at: performance.now(), online: api.device(dongle).then((d) => Date.now() / 1000 - d.last_athena_ping < ONLINE_S, () => false)});
  }
  return seen.get(dongle).online;
}

// a route's segments as its device lists them, kept for the session like the API's file lists: a revisit whose logs are
// in OPFS connects to nothing
async function routeSegments({dongle, logId}) {
  const key = `ycpe_p2p_${dongle}|${logId}`, hit = JSON.parse(sessionStorage.getItem(key) || "null");
  if (hit?.until > Date.now()) return hit.segments;
  const segments = (await listing(dongle)).find((r) => r.name === logId)?.segments || [];
  sessionStorage.setItem(key, JSON.stringify({segments, until: Date.now() + KEPT_MS}));
  return segments;
}

// SOURCES.p2p.find: for each segment, the first of names (a kind's file names) the device has; nothing while it's offline
export async function find(sr, segs, names) {
  if (!await online(sr.dongle)) return new Map();
  const [segments] = await Promise.all([routeSegments(sr), serviceWorker()]), found = new Map();
  for (const n of segs) {
    const seg = segments.find((s) => s.n === n), file = seg && names.find((f) => seg.files[f] != null);
    if (file) found.set(n, {url: `/p2p/${PAGE}/${sr.dongle}/${sr.logId}--${n}/${file}`, file, size: seg.files[file]});
  }
  return found;
}
