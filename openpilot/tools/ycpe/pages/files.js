// a route's uploads: request files from the device over athena (upload_urls → uploadFilesToUrls, as connect does),
// and follow or cancel the device's upload queue. Keys are "<segment>/<kind>" with data/sources.js kind names.
import {api} from "../api.js";
import {h} from "../dom.js";
import {icon} from "../icons.js";
import {isOnline} from "../app.js";
import {canManage} from "./home.js";
import {KINDS} from "../data/sources.js";

export const UPLOADABLE = ["rlog", "fcamera", "ecamera", "dcamera"];  // qlogs and qcameras upload on their own
export const LABELS = {qlog: "qlog", rlog: "rlog", qcamera: "qcamera", fcamera: "road", ecamera: "wide", dcamera: "driver"};
const KIND_OF = Object.fromEntries(Object.entries(KINDS).flatMap(([kind, names]) => names.map((n) => [n, kind])));
const WEEK = 7 * 86400;
const POLL_MS = 2000;

// ".../<dongle>/<route>/<segment>/<file>?sig" → {route, n, kind}
function parseUploadUrl(url) {
  const [route, n, file] = new URL(url).pathname.split("/").slice(-3);
  return {route, n: Number(n), kind: KIND_OF[file]};
}

// has(key): the file is already uploaded. changed({uploaded}) is called on every state change; uploaded is true
// when an upload finished, so the caller refreshes its file list.
export function uploads(route, device, has, changed) {
  const dongleId = route.fullname.split("|")[0], routeId = route.fullname.split("|")[1];
  const states = new Map();  // key → {requested} | {queued, current, progress, paused, id} | {notFound} | {error}
  let polling = null, alive = true;
  const up = {
    canUpload: false, metered: false, queue: [], cancelling: new Set(),
    offline: () => !device || !isOnline(device),
    state: (key) => states.get(key),
    // the keys of these that can still be asked for
    missing: (keys) => keys.filter((k) => !has(k) && !states.get(k) && UPLOADABLE.includes(k.split("/")[1])),
    request, cancel,
    dispose() { alive = false; clearTimeout(polling); },
  };
  const emit = (uploaded = false) => alive && changed({uploaded});

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
      poll();
    } catch (err) {
      set(keys.filter((k) => states.get(k)?.requested), {error: err.message});
    }
  }

  async function cancel(ids) {
    for (const id of ids) up.cancelling.add(id);
    emit();
    try {
      await api.athena(dongleId, "cancelUpload", {upload_id: ids});
    } finally {
      poll();
    }
  }

  // while anything is queued or requested, every 2 s
  async function poll() {
    clearTimeout(polling);
    if (!alive || up.offline()) return;
    up.queue = await api.athena(dongleId, "listUploadQueue").catch(() => []);
    const before = [...states].filter(([, s]) => s.queued).map(([k]) => k);
    const now = new Set();
    for (const q of up.queue) {
      const {route: r, n, kind} = parseUploadUrl(q.url);
      if (r !== routeId || !kind) continue;
      const key = `${n}/${kind}`;
      now.add(key);
      states.set(key, {queued: true, id: q.id, current: q.current, progress: q.progress, paused: up.metered && !q.allow_cellular});
    }
    const done = before.filter((k) => !now.has(k));
    for (const k of done) states.delete(k);
    up.cancelling = new Set([...up.cancelling].filter((id) => up.queue.some((q) => q.id === id)));
    emit(done.length > 0);
    if (up.queue.length || [...states.values()].some((s) => s.requested)) polling = setTimeout(poll, POLL_MS);
  }

  // uploads requested while the device was offline wait in athena's queue
  async function offlineQueue() {
    for (const q of await api.athenaOfflineQueue(dongleId).catch(() => [])) {
      if (q.expiry < Date.now() / 1000) continue;
      const fns = q.method === "uploadFilesToUrls" ? q.params.files_data.map((f) => f.fn) : q.method === "uploadFileToUrl" ? [q.params[0]] : [];
      for (const fn of fns) {
        const [seg, name] = fn.split("/");
        const [r, n] = [seg.slice(0, seg.lastIndexOf("--")), seg.slice(seg.lastIndexOf("--") + 2)];
        if (r === routeId && KIND_OF[name]) states.set(`${n}/${KIND_OF[name]}`, {queued: true, paused: true});
      }
    }
    emit();
  }

  (async () => {
    up.canUpload = !!device && await canManage(device);
    emit();
    if (up.offline()) {
      if (up.canUpload) offlineQueue();
      return;
    }
    up.metered = await api.athena(dongleId, "getNetworkMetered").catch(() => false);
    poll();
  })();
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

export function uploadNotice(up) {
  if (!up.offline() && !up.metered) return null;
  return h("div", {class: "notice"}, icon("warning", 16), up.offline()
    ? "Device offline: requested uploads start when it comes back online."
    : "Device on a metered connection: uploads are paused until it's on wifi.");
}

export function uploadQueue(up) {
  if (up.offline()) return h("p", {class: "muted"}, "Device offline.");
  if (!up.queue.length) return h("p", {class: "muted"}, "No uploads.");
  const waiting = up.queue.filter((q) => !q.current).map((q) => q.id);
  return h("div", {},
    h("table", {class: "list"},
      h("tr", {}, h("th", {}, "segment"), h("th", {}, "file"), h("th", {}, "progress"), h("th")),
      up.queue.map((q) => {
        const {route: r, n, kind} = parseUploadUrl(q.url);
        return h("tr", {},
          h("td", {class: "mono"}, `${r}--${n}`),
          h("td", {}, LABELS[kind] ?? q.path),
          h("td", {}, q.current ? h("progress", {max: 1, value: q.progress}) : up.metered && !q.allow_cellular ? "paused" : "pending"),
          h("td", {}, q.current ? "" : up.cancelling.has(q.id) ? h("span", {class: "spinner"})
            : h("button", {class: "icon-btn", "aria-label": "Cancel upload", onclick: () => up.cancel([q.id])}, icon("cancel", 18))));
      })),
    h("p", {}, h("button", {class: "button small", disabled: !waiting.length, onclick: () => up.cancel(waiting)}, "Cancel all")));
}
