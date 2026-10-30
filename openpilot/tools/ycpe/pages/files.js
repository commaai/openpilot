// a route's uploads: request files from the device over athena (upload_urls → uploadFilesToUrls, as connect does),
// and follow or cancel the device's upload queue. Keys are "<segment>/<kind>" with data/sources.js kind names.
import {SEGMENT, api} from "../api.js";
import {h} from "../dom.js";
import {icon} from "../icons.js";
import {isOnline} from "../app.js";
import {canManage} from "./home.js";
import {KINDS, KIND_OF} from "../data/sources.js";

export const UPLOADABLE = ["rlog", "fcamera", "ecamera", "dcamera"];  // qlogs and qcameras upload on their own
export const LABELS = {qlog: "qlog", rlog: "rlog", qcamera: "qcamera", fcamera: "road", ecamera: "wide", dcamera: "driver"};
const WEEK = 7 * 86400;
const POLL_MS = 2000;

// ".../<dongle>/<route>/<segment>/<file>?sig" → {route, n, file, kind}
export function parseUploadUrl(url) {
  const [route, n, file] = new URL(url).pathname.split("/").slice(-3);
  return {route, n: Number(n), file, kind: KIND_OF[file]};
}

// the "<route>--<segment>/<file>"s athena holds for an offline device, with when each request expires: [[fn, expiry]]
async function offlineFiles(dongleId) {
  return (await api.athenaOfflineQueue(dongleId)).filter((q) => q.expiry > Date.now() / 1000).flatMap((q) =>
    q.method === "uploadFilesToUrls" ? q.params.files_data.map((f) => [f.fn, q.expiry]) : q.method === "uploadFileToUrl" ? [[q.params[0], q.expiry]] : []);
}

// a route's segment numbers
export const segmentsOf = (route) => route.segment_numbers || Array.from({length: (route.maxqlog ?? -1) + 1}, (_, i) => i);

// the keys of kinds in the segments that overlap [r0, r1] s of the route (where they play on the timeline)
export const keysIn = (route, [r0, r1], kinds) => segmentsOf(route)
  .filter((n, i) => { const slot = route.range ? i : n; return slot * SEGMENT < r1 && (slot + 1) * SEGMENT > r0; })
  .flatMap((n) => kinds.map((kind) => `${n}/${kind}`));

// the key of an uploaded file's url (".../<segment>/<file>?sig") as kind
export const urlKey = (url, kind) => `${new URL(url).pathname.split("/").at(-2)}/${kind}`;

// a device's upload queue: listUploadQueue every 2 s while the device is online and keep() holds, whether it's on a
// metered connection, and cancel(ids) for those who manage the device. changed(polled) follows every change; polled
// when the queue was just asked for. A failed ask (not registered, no access, athena down) stops the polling until the
// next poll(). Without a device (it couldn't be fetched) it's offline.
export function deviceQueue(device, changed, keep = () => true) {
  const dongleId = device?.dongle_id;
  let timer = 0, asks = 0;
  const q = {
    queue: [], loaded: false, error: null, metered: false, cancelling: new Set(), cancelError: null, disposed: false,
    offline: () => !device || !isOnline(device),
    waiting: () => q.queue.filter((u) => !u.current).map((u) => u.id),  // the ids of those not uploading yet
    cancel: null,  // for those who manage the device, once ready
    ready: (async () => { if (device && await canManage(device)) q.cancel = cancel; })(),
    poll, dispose() { q.disposed = true; clearTimeout(timer); },
  };
  const emit = (polled = false) => !q.disposed && changed(polled);

  async function cancel(ids) {
    for (const id of ids) q.cancelling.add(id);
    q.cancelError = null;
    emit();
    try {
      await api.athena(dongleId, "cancelUpload", {upload_id: ids});
    } catch (err) {
      for (const id of ids) q.cancelling.delete(id);
      q.cancelError = err;
    }
    poll();
  }

  // asks now; of overlapping asks, only the latest counts
  async function poll() {
    clearTimeout(timer);
    await q.ready;
    if (q.disposed) return;
    if (q.offline()) return emit();
    const ask = ++asks;
    try {
      if (!q.loaded) q.metered = await api.athena(dongleId, "getNetworkMetered").catch(() => false);
      const queue = await api.athena(dongleId, "listUploadQueue");
      if (ask !== asks) return;
      Object.assign(q, {queue, loaded: true, error: null});
    } catch (err) {
      if (ask === asks) { q.error = err; emit(); }
      return;
    }
    q.cancelling = new Set([...q.cancelling].filter((id) => q.queue.some((u) => u.id === id)));
    emit(true);
    if (!q.disposed && keep()) timer = setTimeout(poll, POLL_MS);
  }
  return q;
}

// a route's uploads, on its device's queue. has(key): the file is already uploaded. changed({uploaded}) is called on
// every state change; uploaded is true when an upload finished, so the caller refreshes its file list.
export function uploads(route, device, has, changed) {
  const [dongleId, routeId] = route.fullname.split("|");
  const states = new Map();  // key → {requested} | {queued, current, progress, paused, id} | {notFound} | {error}
  // asked for while anything is queued or requested
  const up = Object.assign(deviceQueue(device, (polled) => polled ? sync() : emit(),
    () => up.queue.length || [...states.values()].some((s) => s.requested)), {
    canUpload: false,
    state: (key) => states.get(key),
    // the keys of these that can still be asked for
    missing: (keys) => keys.filter((k) => !has(k) && !states.get(k) && UPLOADABLE.includes(k.split("/")[1])),
    request,  // for those who manage the device
  });
  const emit = (uploaded = false) => !up.disposed && changed({uploaded});

  async function request(keys) {
    keys = up.missing(keys);
    if (!keys.length) return;
    for (const k of keys) states.set(k, {requested: true});
    emit();
    const set = (ks, state) => { for (const k of ks) states.set(k, state); emit(); };
    try {
      const paths = keys.flatMap((k) => { const [n, kind] = k.split("/"); return KINDS[kind].map((name) => `${routeId}--${n}/${name}`); });
      const urls = await api.uploadUrls(dongleId, paths);
      const files_data = paths.map((fn, i) => ({fn, url: urls[i], headers: {"x-ms-blob-type": "BlockBlob"}, allow_cellular: false, priority: 0}));
      const result = await api.athena(dongleId, "uploadFilesToUrls", {files_data}, Math.floor(Date.now() / 1000) + WEEK);
      if (result === "Device offline, message queued") return set(keys, {queued: true, paused: true});
      // a file is missing only when none of its names exist
      const failed = new Set(result?.failed || []);
      set(keys.filter((k) => { const [n, kind] = k.split("/"); return KINDS[kind].every((name) => failed.has(`${routeId}--${n}/${name}`)); }), {notFound: true});
      up.poll();
    } catch (err) {
      set(keys.filter((k) => states.get(k)?.requested), {error: err.message});
    }
  }

  // the route's files in the queue just asked for: on their way, or done once they've left it
  function sync() {
    const before = [...states].filter(([, s]) => s.queued).map(([k]) => k);
    const now = new Set();
    for (const u of up.queue) {
      const {route: r, n, kind} = parseUploadUrl(u.url);
      if (r !== routeId || !kind) continue;
      const key = `${n}/${kind}`;
      now.add(key);
      states.set(key, {queued: true, id: u.id, current: u.current, progress: u.progress, paused: up.metered && !u.allow_cellular});
    }
    const done = before.filter((k) => !now.has(k));
    for (const k of done) states.delete(k);
    emit(done.length > 0);
  }

  // uploads requested while the device was offline wait in athena's queue
  async function offlineQueue() {
    for (const [fn] of await offlineFiles(dongleId).catch(() => [])) {
      const [seg, name] = fn.split("/");
      const [r, n] = [seg.slice(0, seg.lastIndexOf("--")), seg.slice(seg.lastIndexOf("--") + 2)];
      if (r === routeId && KIND_OF[name]) states.set(`${n}/${KIND_OF[name]}`, {queued: true, paused: true});
    }
    emit();
  }

  up.ready.then(() => {
    up.canUpload = !!up.cancel;
    emit();
    if (up.offline() && up.canUpload) offlineQueue();
    up.poll();
  });
  return up;
}

// a cell's upload state, or null when there's nothing to say
export function uploadState(up, key) {
  const s = up.state(key), kind = key.split("/")[1];
  if (!s) return null;
  if (s.queued) return h("span", {class: "small muted"}, s.current ? `${Math.floor(100 * s.progress)}%` : s.paused ? "paused" : "pending");
  if (s.requested) return h("span", {class: "spinner"});
  if (s.notFound) return h("span", {class: "small muted", title: kind === "dcamera" ? "enable \"Record and Upload Driver Camera\" on the device" : ""}, "not on device");
  return h("span", {class: "small error", title: s.error}, "failed");
}

// keys: the route's "<segment>/<kind>"s. Nothing to say when none of them can be asked for or are on their way.
export function uploadNotice(up, keys) {
  if (!up.offline() && !up.metered) return null;
  if (!up.missing(keys).length && !keys.some((k) => up.state(k))) return null;
  return h("div", {class: "notice"}, icon("warning", 16), up.offline()
    ? "Device offline: requested uploads start when it comes back online."
    : "Device on a metered connection: uploads are paused until it's on wifi.");
}

// the device page's whole upload queue, asked every 2 s while it's on the page; resume() asks again as the tab is
// shown. Offline, it's the requests waiting in athena for the device to come back.
export function deviceUploads(device) {
  const el = h("div", {}, h("p", {class: "muted"}, "Loading…"));
  const q = deviceQueue(device, () => show(), () => el.isConnected);
  const show = () => q.offline() ? offline().catch((err) => el.replaceChildren(failed(q, err))) : el.replaceChildren(uploadQueue(q));

  async function offline() {
    const waiting = q.cancel ? await offlineFiles(device.dongle_id) : [];
    el.replaceChildren(h("p", {class: "muted"}, waiting.length ? "Device offline. These uploads start when it's back:" : "Device offline."));
    if (waiting.length) {
      el.append(h("table", {class: "list"},
        h("tr", {}, h("th", {}, "file"), h("th", {}, "requested until")),
        waiting.map(([fn, expiry]) => h("tr", {}, h("td", {class: "mono"}, fn), h("td", {class: "muted"}, new Date(expiry * 1000).toLocaleString())))));
    }
  }
  el.resume = q.poll;
  return el;
}

const failed = (q, err) => h("p", {class: "muted", title: err.message}, "Couldn't load the upload queue. ",
  h("button", {class: "button small", onclick: q.poll}, "Retry"));

// a device queue's table (deviceQueue's, or a route's uploads')
export function uploadQueue(q) {
  if (q.offline()) return h("p", {class: "muted"}, "Device offline.");
  if (q.error) return failed(q, q.error);
  if (!q.queue.length) return h("p", {class: "muted"}, "No uploads.");
  const waiting = q.waiting();
  return h("div", {},
    h("table", {class: "list"},
      h("tr", {}, h("th", {}, "segment"), h("th", {}, "file"), h("th", {}, "progress"), h("th")),
      q.queue.map((u) => {
        const {route: r, n, kind} = parseUploadUrl(u.url);
        return h("tr", {},
          h("td", {class: "mono"}, `${r}--${n}`),
          h("td", {}, LABELS[kind] ?? u.path),
          h("td", {}, u.current ? h("progress", {max: 1, value: u.progress}) : q.metered && !u.allow_cellular ? "paused" : "pending"),
          h("td", {}, u.current || !q.cancel ? "" : q.cancelling.has(u.id) ? h("span", {class: "spinner"})
            : h("button", {class: "icon-btn", "aria-label": "Cancel upload", onclick: () => q.cancel([u.id])}, icon("cancel", 18))));
      })),
    q.cancel && h("p", {}, h("button", {class: "button small", disabled: !waiting.length, onclick: () => q.cancel(waiting)}, "Cancel all")),
    q.cancelError && h("p", {class: "small error"}, `Cancel failed: ${q.cancelError.message}`));
}
