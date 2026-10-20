// a route's files per segment: download what's uploaded, request the rest from the device over athena
// (upload_urls → uploadFilesToUrls, as connect does), and follow or cancel the device's upload queue
import {api} from "../api.js";
import {h} from "../dom.js";
import {icon} from "../icons.js";
import {isOnline} from "../app.js";
import {canManage} from "./devices.js";

// files key → column label and the names the device may have it under
export const KINDS = {
  logs: ["rlog", ["rlog.zst", "rlog.bz2"]],
  qlogs: ["qlog", ["qlog.zst", "qlog.bz2"]],
  cameras: ["road", ["fcamera.hevc"]],
  ecameras: ["wide", ["ecamera.hevc"]],
  dcameras: ["driver", ["dcamera.hevc"]],
  qcameras: ["qcamera", ["qcamera.ts"]],
};
const UPLOADABLE = ["logs", "cameras", "ecameras", "dcameras"];
const KIND_OF = Object.fromEntries(Object.entries(KINDS).flatMap(([kind, [, names]]) => names.map((n) => [n, kind])));
const WEEK = 7 * 86400;
const POLL_MS = 2000;

// ".../<dongle>/<route>/<segment>/<file>?sig" → {route, n, kind}
function parseUploadUrl(url) {
  const [route, n, file] = new URL(url).pathname.split("/").slice(-3);
  return {route, n: Number(n), kind: KIND_OF[file]};
}

export function filesSection(route, device) {
  const dongleId = route.fullname.split("|")[0], routeId = route.fullname.split("|")[1];
  const segments = route.segment_numbers || Array.from({length: (route.maxqlog ?? -1) + 1}, (_, i) => i);
  const files = new Map();  // "n/kind" → {url} | {requested} | {queued, current, progress, paused, id} | {notFound} | {error}
  const cells = new Map();
  let canUpload = false, metered = false, queue = [], polling = null, cancelling = new Set();

  const notice = h("div", {class: "notice", hidden: true});
  const actions = h("div", {class: "files-actions"});
  const queueEl = h("div", {class: "upload-queue"});
  const table = h("table", {class: "files"});
  const el = h("div", {class: "section"}, notice, actions, table, h("h3", {}, "Upload queue"), queueEl);

  const offline = () => !device || !isOnline(device);
  const missing = (keys) => keys.filter((k) => !files.get(k));
  const keysOf = (ns, kinds) => ns.flatMap((n) => kinds.map((kind) => `${n}/${kind}`));

  function cell(key) {
    const f = files.get(key) || {};
    const [n, kind] = key.split("/");
    if (f.url) return h("a", {class: "dl", href: f.url, download: "", title: `${KINDS[kind][0]} ${n}`}, icon("download", 18));
    if (f.queued) return h("span", {class: "small muted"}, f.current ? `${Math.floor(100 * f.progress)}%` : f.paused ? "paused" : "pending");
    if (f.requested) return h("span", {class: "spinner"});
    if (f.notFound) return h("span", {class: "small muted", title: kind === "dcameras" ? "enable \"Record and Upload Driver Camera\" on the device" : ""}, "not found");
    if (f.error) return h("span", {class: "small error", title: f.error}, "failed");
    if (canUpload && UPLOADABLE.includes(kind)) return h("button", {class: "icon-button small", title: `request ${KINDS[kind][0]} ${n}`, onclick: () => request([key])}, icon("upload", 18));
    return h("span", {class: "muted"}, "–");
  }

  function update(keys = cells.keys()) {
    for (const k of keys) cells.get(k)?.replaceChildren(cell(k));
    const all = keysOf(segments, UPLOADABLE), rlogs = keysOf(segments, ["logs"]);
    const button = (label, keys) => h("button", {class: "button small", disabled: !canUpload || !keys.length, onclick: () => request(keys)}, `${label} (${keys.length})`);
    actions.replaceChildren(button("Upload all rlogs", missing(rlogs)), button("Upload all files", missing(all)));
    notice.hidden = !offline() && !metered;
    notice.replaceChildren(icon("warning", 18), offline()
      ? "Device offline: requested uploads start when it comes back online."
      : "Device on a metered connection: uploads are paused until it's on wifi.");
  }

  function build() {
    const kinds = Object.keys(KINDS);
    table.replaceChildren(
      h("thead", {}, h("tr", {}, h("th", {}, "seg"), kinds.map((k) => h("th", {}, KINDS[k][0])), h("th"))),
      h("tbody", {}, segments.map((n) => h("tr", {},
        h("td", {class: "muted"}, n),
        kinds.map((k) => { const td = h("td"); cells.set(`${n}/${k}`, td); return td; }),
        h("td", {}, canUpload ? h("button", {class: "button tiny", title: `request all files of segment ${n}`,
          onclick: () => request(missing(keysOf([n], UPLOADABLE)))}, "all") : "")))));
    update();
  }

  async function loadFiles(fresh = false) {
    if (fresh) api.routeFiles.forget(route.fullname);
    const urls = await api.routeFiles(route.fullname);
    for (const [kind, list] of Object.entries(urls)) {
      if (!KINDS[kind]) continue;
      for (const url of list) files.set(`${new URL(url).pathname.split("/").at(-2)}/${kind}`, {url});
    }
    update();
  }

  async function request(keys) {
    keys = missing(keys);
    if (!keys.length) return;
    for (const k of keys) files.set(k, {requested: true});
    update(keys);
    const set = (ks, state) => { for (const k of ks) if (!files.get(k)?.url) files.set(k, state); update(ks); };
    try {
      const paths = keys.flatMap((k) => { const [n, kind] = k.split("/"); return KINDS[kind][1].map((name) => `${routeId}--${n}/${name}`); });
      const urls = await api.uploadUrls(dongleId, paths);
      const files_data = paths.map((fn, i) => ({fn, url: urls[i], headers: {"x-ms-blob-type": "BlockBlob"}, allow_cellular: false, priority: 0}));
      const result = await api.athena(dongleId, "uploadFilesToUrls", {files_data}, Math.floor(Date.now() / 1000) + WEEK);
      if (result === "Device offline, message queued") return set(keys, {queued: true, paused: true});
      // a file is missing only when none of its names exist
      const failed = new Set(result?.failed || []);
      set(keys.filter((k) => { const [n, kind] = k.split("/"); return KINDS[kind][1].every((name) => failed.has(`${routeId}--${n}/${name}`)); }), {notFound: true});
      pollQueue();
    } catch (err) {
      set(keys.filter((k) => files.get(k)?.requested), {error: err.message});
    }
  }

  async function cancel(ids) {
    for (const id of ids) cancelling.add(id);
    renderQueue();
    try {
      await api.athena(dongleId, "cancelUpload", {upload_id: ids});
    } finally {
      pollQueue();
    }
  }

  function renderQueue() {
    if (offline()) return queueEl.replaceChildren(h("p", {class: "muted"}, "Device offline."));
    if (!queue.length) return queueEl.replaceChildren(h("p", {class: "muted"}, "No uploads."));
    const waiting = queue.filter((q) => !q.current).map((q) => q.id);
    queueEl.replaceChildren(
      h("table", {class: "list"},
        h("tr", {}, h("th", {}, "segment"), h("th", {}, "file"), h("th", {}, "progress"), h("th")),
        queue.map((q) => {
          const {route: r, n, kind} = parseUploadUrl(q.url);
          return h("tr", {},
            h("td", {class: "mono"}, `${r}--${n}`),
            h("td", {}, KINDS[kind]?.[0] ?? q.path),
            h("td", {}, q.current ? h("progress", {max: 1, value: q.progress}) : metered && !q.allow_cellular ? "paused" : "pending"),
            h("td", {}, q.current ? "" : cancelling.has(q.id) ? h("span", {class: "spinner"})
              : h("button", {class: "icon-button small", "aria-label": "Cancel upload", onclick: () => cancel([q.id])}, icon("cancel", 18))));
        })),
      h("p", {}, h("button", {class: "button small", disabled: !waiting.length, onclick: () => cancel(waiting)}, "Cancel all")));
  }

  // while anything is queued or requested and the page is up, every 2 s; a finished upload refreshes the file list
  async function pollQueue() {
    clearTimeout(polling);
    if (offline() || !el.isConnected && polling !== null) return;
    try {
      queue = await api.athena(dongleId, "listUploadQueue");
    } catch {
      queue = [];
    }
    const before = [...files].filter(([, f]) => f.queued).map(([k]) => k);
    const now = new Set();
    for (const q of queue) {
      const {route: r, n, kind} = parseUploadUrl(q.url);
      if (r !== routeId || !kind) continue;
      const key = `${n}/${kind}`;
      now.add(key);
      if (!files.get(key)?.url) files.set(key, {queued: true, id: q.id, current: q.current, progress: q.progress, paused: metered && !q.allow_cellular});
    }
    const done = before.filter((k) => !now.has(k));
    for (const k of done) files.delete(k);
    cancelling = new Set([...cancelling].filter((id) => queue.some((q) => q.id === id)));
    if (done.length) loadFiles(true).catch(() => {});
    update();
    renderQueue();
    if (queue.length || [...files.values()].some((f) => f.requested)) polling = setTimeout(pollQueue, POLL_MS);
  }

  // uploads requested while the device was offline wait in athena's queue
  async function loadOfflineQueue() {
    for (const q of await api.athenaOfflineQueue(dongleId).catch(() => [])) {
      if (q.expiry < Date.now() / 1000) continue;
      const fns = q.method === "uploadFilesToUrls" ? q.params.files_data.map((f) => f.fn) : q.method === "uploadFileToUrl" ? [q.params[0]] : [];
      for (const fn of fns) {
        const [seg, name] = fn.split("/");
        const [r, n] = [seg.slice(0, seg.lastIndexOf("--")), seg.slice(seg.lastIndexOf("--") + 2)];
        if (r === routeId && KIND_OF[name] && !files.get(`${n}/${KIND_OF[name]}`)) files.set(`${n}/${KIND_OF[name]}`, {queued: true, paused: true});
      }
    }
    update();
  }

  (async () => {
    canUpload = !!device && await canManage(device);
    build();
    await loadFiles().catch((err) => table.replaceWith(h("p", {class: "error"}, `Files not available: ${err.message}`)));
    if (offline()) {
      renderQueue();
      if (canUpload) loadOfflineQueue();
      return;
    }
    metered = await api.athena(dongleId, "getNetworkMetered").catch(() => false);
    pollQueue();
  })();
  return el;
}
