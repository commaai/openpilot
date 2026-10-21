// connect's clips (DriveView/ClipMenu.jsx, api/clips.js). The device cuts and encodes them itself, over athena:
// createClip queues a range of one camera, getClipState lists what's queued, encoding and ready (and, for a route, the
// ranges each camera covers), deleteClip cancels or deletes one, and a ready clip comes back in base64 chunks
// (getClipChunk). Downloaded clips stay in the Cache API, so they play again without asking the device.
// clipButton(ctx) is the drive view's Clip button, deviceClips(device) a device's clips
import {api, versionAtLeast} from "/api.js";
import {h, css, ask, fmt, download} from "/dom.js";
import {icon} from "/icons.js";
import {perf} from "/perf.js";
import {isOnline, link} from "/app.js";
import {userMode, cnIcon, cnModal, progress, linear} from "/pages/cnui.js";

await css(new URL("./clips.css", import.meta.url));

// an icon: connect's own in user mode (its name and size there), the theme's elsewhere
const ic = (name, size, cnName = name, cnSize = size) => userMode() ? cnIcon(cnName, cnSize) : icon(name, size);

const MAX_DURATION = 30 * 60, POLL = 1000;
const CHUNKS = 3;  // chunk requests in flight at once
const RETRIES = [500, 1000, 2000, 4000, 8000, 16000];  // ms before each retry of a chunk the network lost
const ACTIVE = new Set(["queued", "encoding"]);
const CAMERAS = [["fcamera.hevc", "Road"], ["ecamera.hevc", "Wide road"], ["dcamera.hevc", "Driver"]];
const BITRATES = [[5, "Standard", "5 Mbps"], [8, "High", "8 Mbps"], [12, "Extreme", "12 Mbps"]];
const SPEEDUPS = [1, 2, 4, 5, 10];

const dur = (s) => {
  s = Math.max(0, Math.round(s));
  const hr = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, sec = s % 60;
  return [hr && `${hr}h`, m && `${m}m`, (sec || !hr && !m) && `${sec}s`].filter(Boolean).join(" ");
};
const mb = (bytes) => bytes ? `${Math.max(1, Math.round(bytes / 1048576))} MB` : "";
const stem = (name) => name.trim().replace(/\.mp4$/i, "");
const validName = (name) => !name.trim() || /^[a-zA-Z0-9][a-zA-Z0-9._-]*$/.test(stem(name));
const routeName = (route) => route?.fullname?.split(/[|/]/).pop() || null;
const cameraName = (camera) => CAMERAS.find(([v]) => v === camera)?.[1] || camera;
const covers = (ranges, camera, t0, t1) => !!ranges?.[camera]?.available_ranges?.some(([a, b]) => a <= t0 && b >= t1);
const clipTitle = (clip) => stem(clip.filename || "") || "Clip";
const clipLength = (clip) => dur((clip.source_end_time - clip.source_start_time) / (clip.speedup || 1));

// as connect: openpilot 0.11.2 on, with a getVersion that has a commit date (athenad from before clips has none). A yes
// is remembered, so an offline device still shows its button
const SUPPORT_KEY = "ycpe_clips_";
const support = new Map();
function supportsClips(device) {
  const id = device.dongle_id;
  if (!versionAtLeast(device, "0.11.2")) return Promise.resolve(false);
  if (Number(localStorage.getItem(SUPPORT_KEY + id)) > 0) return Promise.resolve(true);
  if (!isOnline(device)) return Promise.resolve(false);
  if (!support.has(id)) {
    const req = api.athena(id, "getVersion").then((v) => {
      const date = Number(v?.commit_date);
      if (!(date > 0)) return support.delete(id) && false;
      localStorage.setItem(SUPPORT_KEY + id, date);
      return true;
    });
    req.catch(() => support.delete(id));
    support.set(id, req);
  }
  return support.get(id);
}

// ---- downloads: a clip's chunks, a few at a time, into the cache; one download per clip however many ask ----

const store = () => caches.open("ycpe-clips");
const keyOf = (dongle, clip) => `/clips/${dongle}/${encodeURIComponent(clip.filename)}/${clip.requested_at}`;
const cached = async (key) => !!await store().then((c) => c.match(key)).catch(() => null);
const downloads = new Map();  // cache key → {promise, listeners, loaded, total, cancelled}

async function chunk(dongle, filename, offset, attempt = 0) {
  try {
    return await api.athena(dongle, "getClipChunk", {filename, offset});
  } catch (e) {
    // the network or athena lost it, not the device refusing: again, a little later each time
    if (!(e instanceof TypeError || e.status >= 500) || attempt === RETRIES.length) throw e;
    await new Promise((done) => setTimeout(done, RETRIES[attempt]));
    return chunk(dongle, filename, offset, attempt + 1);
  }
}

// decoded by the browser off the main thread, not a loop over the string
const bytes = async (b64) => new Uint8Array(await (await fetch(`data:application/octet-stream;base64,${b64}`)).arrayBuffer());

// the first chunk says the size and the chunk length, then CHUNKS at a time
async function fetchClip(dongle, filename, progress, cancelled) {
  const parts = [];
  let size, step = 0, next = 0;
  while (size === undefined || next < size) {
    if (cancelled()) throw new Error("Clip download cancelled");
    const n = step ? Math.min(CHUNKS, Math.ceil((size - next) / step)) : 1;
    const got = await Promise.all(Array.from({length: n}, (_, i) => chunk(dongle, filename, next + i * step)));
    if (cancelled()) throw new Error("Clip download cancelled");
    for (const r of got) {
      size ??= r.size;
      if (r.size !== size || r.offset !== next) throw new Error("Clip changed during download");
      const part = await bytes(r.data);
      if (!part.length && next < size) throw new Error("Clip download returned an empty chunk");
      step ||= part.length;
      parts.push(part);
      next += part.length;
      progress(next, size);
    }
  }
  return new Blob(parts, {type: "video/mp4"});
}

async function clipBlob(dongle, clip, onProgress) {
  const key = keyOf(dongle, clip);
  const hit = await store().then((c) => c.match(key)).catch(() => null);
  if (hit) return hit.blob();
  let d = downloads.get(key);
  if (!d) {
    d = {cancelled: false, listeners: new Set(), loaded: 0, total: 0};
    d.promise = fetchClip(dongle, clip.filename, (loaded, total) => {
      Object.assign(d, {loaded, total});
      for (const fn of d.listeners) fn(loaded, total);
    }, () => d.cancelled).then(async (blob) => {
      if (downloads.get(key) === d) await store().then((c) => c.put(key, new Response(blob))).catch(() => {});
      return blob;
    }).finally(() => { if (downloads.get(key) === d) downloads.delete(key); });
    downloads.set(key, d);
  }
  d.listeners.add(onProgress);
  if (d.total) onProgress(d.loaded, d.total);
  return d.promise.finally(() => d.listeners.delete(onProgress));
}

// a deleted clip's download stops and its cached copies go (a new clip may take its name)
async function forget(dongle, filename) {
  const prefix = `/clips/${dongle}/${encodeURIComponent(filename)}/`;
  for (const [key, d] of downloads) {
    if (!key.startsWith(prefix)) continue;
    d.cancelled = true;
    d.listeners.clear();
    downloads.delete(key);
  }
  const c = await store().catch(() => null);
  for (const req of await c?.keys() ?? []) if (new URL(req.url).pathname.startsWith(prefix)) await c.delete(req);
}

// as connect: a phone shares the file (to its photos, say), a desktop downloads it
async function save(blob, clip) {
  const filename = `${clipTitle(clip)}.mp4`;
  if (/android|iphone|ipad|ipod/i.test(navigator.userAgent) && navigator.share) {
    try {
      const file = new File([blob], filename, {type: "video/mp4"});
      if (!navigator.canShare || navigator.canShare({files: [file]})) return await navigator.share({files: [file]});
    } catch (e) {
      if (e.name === "AbortError") return;
    }
  }
  download(blob, filename, 60e3);
}

// a downloaded clip, playing. Shown once its first frame is decoded: a video that comes up empty holds the compositor
// up for its decoder (~70 ms). Not if wanted() has changed its mind by then
async function viewer(clip, blob, wanted) {
  const url = URL.createObjectURL(blob);
  const video = h("video", {src: url, controls: true, muted: true, playsinline: true, preload: "auto"});
  await new Promise((done) => { video.onloadeddata = video.onerror = done; setTimeout(done, 2000); });
  if (!wanted()) return URL.revokeObjectURL(url);
  const el = h("dialog", {class: `clips-viewer${userMode() ? " um" : ""}`, autofocus: true, onclose: () => { el.remove(); URL.revokeObjectURL(url); }},
    h("div", {class: "clips-viewer-head"},
      h("div", {class: "clips-viewer-title"}, h("b", {}, clipTitle(clip)),
        h("div", {class: "clips-meta"}, [cameraName(clip.camera), clipLength(clip), mb(clip.size)].filter(Boolean).join(" · "))),
      h("button", {class: "clips-icon", title: "Download clip", "aria-label": "Download clip", onclick: () => save(blob, clip)}, ic("download", 24)),
      h("button", {class: "clips-icon", title: "Close", "aria-label": "Close video", onclick: () => el.close()}, ic("close", 24, "closeBold"))),
    video);
  el.addEventListener("click", (e) => { if (e.target === el) el.close(); });
  document.body.append(el);
  el.showModal();
  video.play().catch(() => {});
}

// connect's segmented buttons: options [value, label, detail?]; set() marks the chosen one
function segmented(options, pick) {
  const buttons = options.map(([value, label, detail]) => h("button", {class: "clips-seg-btn", "aria-pressed": "false", onclick: () => pick(value)},
    label, detail && h("span", {class: "clips-detail"}, detail)));
  return {el: h("div", {class: "clips-seg"}, buttons), set(on, disabled = () => false) {
    options.forEach(([value], i) => { buttons[i].ariaPressed = String(value === on); buttons[i].disabled = disabled(value); });
  }};
}

// ---- the menu: create a clip of the range (with a route), then the clips on the device ----

// a modal under its button, the button's top right corner its own, as connect's menu; full screen on a phone when it
// creates. range() is the drive's selected range in route seconds
function clipMenu(anchor, device, route = null, range = null) {
  const dongle = device.dongle_id, online = isOnline(device), here = routeName(route), um = userMode();
  const s = {clips: [], camera: "fcamera.hevc", bitrate: 5, speedup: 1, ranges: null, loading: online, creating: false, error: null,
    autoOpen: null, downloaded: new Set(), previewing: null, progress: 0, starts: new Map()};  // starts: the device's recent drives' start times (user mode)
  let poll = null, request = 0;  // request: the preview being downloaded, so a stale one doesn't open
  const changed = () => perf.invalidate(view);

  const field = (label, ...body) => h("div", {class: "clips-field"}, h("span", {class: "clips-label"}, label), body);
  const cameras = segmented(CAMERAS, (v) => { s.camera = v; changed(); });
  const quality = segmented(BITRATES, (v) => { s.bitrate = v; changed(); });
  const speed = segmented(SPEEDUPS.map((v) => [v, `${v}×`]), (v) => { s.speedup = v; changed(); });
  const rangeText = h("div", {class: "clips-range-value"}), hint = h("p", {class: "clips-hint"}, "No camera footage covers this entire range. Choose a smaller range.");
  const output = h("span", {class: "clips-muted"}), size = h("b", {class: "clips-size"});
  const name = h("input", {class: "clips-input", maxlength: 80, spellcheck: false, autocomplete: "off", "aria-label": "Filename", oninput: changed});
  const nameError = h("p", {class: "clips-error"}, "Use letters, numbers, periods, underscores, and hyphens only.");
  const tooLong = h("p", {class: "clips-error"}, "Choose a range of 30 minutes or less.");
  const createBtn = h("button", {class: "clips-create", onclick: () => create()});
  const error = h("p", {class: "clips-error"}), list = h("div", {class: "clips-list"});
  const close = h("button", {class: "clips-icon clips-close", title: "Close", "aria-label": "Close clip menu", onclick: () => menu.close()}, ic("close", 24, "closeBold"));

  const menu = h("dialog", {class: `clips-menu${route ? " creating" : ""}${um ? " um" : ""}`, autofocus: true, "aria-label": route ? "Create a clip" : "Clips", onclose: () => {
    clearTimeout(poll);
    request++;
    anchor.style.anchorName = "";
    menu.remove();
  }}, h("div", {class: "clips-paper"},
    route && h("div", {class: "clips-body"},
      h("div", {class: "clips-head"}, h("b", {}, "Create a clip"), close),
      h("div", {class: "clips-range"}, h("span", {class: "clips-muted"}, "Selected timeline range"), rangeText),
      field("CAMERA", cameras.el, hint),
      field("QUALITY", quality.el),
      field("SPEED", speed.el, h("div", {class: "clips-estimate"}, output, size)),
      field("FILENAME", name, nameError),
      tooLong, createBtn),
    route && h("hr"),
    h("div", {class: "clips-section"},
      h("div", {class: "clips-section-head"}, "CLIPS ON THIS DEVICE",
        h("span", {class: "clips-info", tabindex: 0, title: "Clips are stored on your device and may be cleared to make room for more recent driving footage.",
          "aria-label": "Clips are stored on your device and may be cleared to make room for more recent driving footage."}, ic("info", 16, "info", 18))),
      error, list)));
  menu.addEventListener("click", (e) => { if (e.target === menu) menu.close(); });  // the backdrop

  const defaultName = (t0, t1) => [dongle, here, Math.round(t0), Math.round(t1), s.camera.replace(/\.hevc$/i, ""), s.speedup > 1 && `${s.speedup}x`]
    .filter((p) => p !== false).join("_");

  async function load(showLoading = true) {
    clearTimeout(poll);
    if (!online) {
      Object.assign(s, {clips: [], ranges: null, loading: false, error: null});
      return changed();
    }
    if (showLoading) Object.assign(s, {loading: true, error: null});
    changed();
    try {
      const state = await api.athena(dongle, "getClipState", here ? {route: route.fullname} : {});
      if (!menu.open) return;
      const ready = state.clips.filter((c) => c.status === "ready");
      const have = await Promise.all(ready.map((c) => cached(keyOf(dongle, c))));
      Object.assign(s, {clips: state.clips, ranges: here ? state.cameras || {} : null, loading: false,
        downloaded: new Set(ready.filter((_, i) => have[i]).map((c) => c.filename))});
      const auto = s.clips.find((c) => c.filename === s.autoOpen && c.status === "ready");
      if (auto) {
        s.autoOpen = null;
        preview(auto);
      }
      if (s.clips.some((c) => ACTIVE.has(c.status))) poll = setTimeout(() => load(false), POLL);
    } catch (e) {
      Object.assign(s, {loading: false, error: e.message || "Could not reach the device"});
    }
    changed();
  }

  async function create() {
    const [t0, t1] = range().map((t) => Math.round(t * 1000) / 1000);
    const filename = `${stem(name.value) || defaultName(t0, t1)}.mp4`;
    Object.assign(s, {creating: true, autoOpen: filename, error: null});
    changed();
    try {
      await api.athena(dongle, "createClip", {route: route.fullname, source_start_time: t0, source_end_time: t1,
        clip: {camera: s.camera, bitrate: s.bitrate, speedup: s.speedup, filename}});
      s.creating = false;
      await load(false);
    } catch (e) {
      Object.assign(s, {creating: false, autoOpen: null, error: e.message || "Could not create clip"});
      changed();
    }
  }

  async function remove(clip) {
    if (s.previewing === clip.filename) {
      request++;
      Object.assign(s, {previewing: null, progress: 0});
    }
    try {
      await forget(dongle, clip.filename);
      await api.athena(dongle, "deleteClip", {filename: clip.filename});
      if (clip.filename === s.autoOpen) s.autoOpen = null;
      await load(false);
      return true;
    } catch (e) {
      s.error = e.message || "Could not remove clip";
      changed();
      return false;
    }
  }

  // connect's delete confirmation: it stays, its Delete spinning, until the device has deleted it
  function confirmDelete(clip) {
    let busy = false;
    const {el, paper} = cnModal("Delete clip?", "clips-delete", () => {}, () => busy);
    const cancel = h("button", {class: "cnui-btn", onclick: () => el.close()}, "Cancel");
    const del = h("button", {class: "cnui-btn clips-delete-btn", onclick: async () => {
      busy = cancel.disabled = del.disabled = true;
      del.replaceChildren(progress(18));
      const ok = await remove(clip);
      busy = cancel.disabled = del.disabled = false;
      if (ok) el.close();
      else del.textContent = "Delete";
    }}, "Delete");
    paper.append(h("h2", {}, "Delete clip?"), h("p", {}, `${stem(clip.filename || "") || "this clip"} will be permanently deleted from your comma device.`),
      h("div", {class: "clips-delete-actions"}, cancel, del));
  }

  // downloaded with its progress under it, then played
  async function preview(clip) {
    const mine = ++request;
    Object.assign(s, {previewing: clip.filename, progress: 0, error: null});
    changed();
    try {
      const blob = await clipBlob(dongle, clip, (loaded, total) => {
        if (mine !== request) return;
        s.progress = loaded / total;
        changed();
      });
      if (mine !== request) return;
      s.downloaded.add(clip.filename);
      await viewer(clip, blob, () => mine === request);
      if (mine === request) Object.assign(s, {previewing: null, progress: 0});
    } catch (e) {
      if (mine === request) Object.assign(s, {previewing: null, progress: 0, error: e.message || "Could not preview clip"});
    }
    changed();
  }

  function row(clip) {
    const ready = clip.status === "ready", active = ACTIVE.has(clip.status), previewing = s.previewing === clip.filename;
    const downloaded = s.downloaded.has(clip.filename), path = `/${dongle}/${clip.route}`;
    // (user mode as connect: when its drive started, if it's one of the device's recent ones, in words)
    const start = um && s.starts.get(clip.route);
    const where = clip.route === here ? "This route" : um ? (start ? new Date(start).toLocaleString([], {dateStyle: "medium", timeStyle: "short"}) : clip.route)
      : h("a", {href: path, onclick: (e) => { menu.close(); link(path)(e); }}, clip.route);
    const play = downloaded ? "Play clip" : "Download clip", drop = active ? "Cancel clip" : "Delete clip";
    return h("div", {class: "clips-clip", "data-clip": clip.filename, "data-status": clip.status},
      h("div", {class: "clips-top"},
        h("div", {class: "clips-details"},
          h("div", {class: "clips-title"}, clipTitle(clip)),
          h("div", {class: "clips-meta"}, where),
          h("div", {class: "clips-meta"}, [cameraName(clip.camera), clipLength(clip), mb(clip.size)].filter(Boolean).join(" · "))),
        h("div", {class: "clips-actions"},
          ready && h("button", {class: "clips-icon", "data-focus": `play ${clip.filename}`, "aria-label": play, title: previewing ? "Downloading" : play,
            disabled: previewing, onclick: () => preview(clip)}, downloaded ? ic("play", 24, "play", 27) : ic("download", 24)),
          active && h("span", {class: "clips-meta"}, clip.status === "encoding" ? "Encoding" : "Queued"),
          h("button", {class: "clips-icon", "data-focus": `drop ${clip.filename}`, "aria-label": drop, title: drop, onclick: async () => {
            if (um && !active) return confirmDelete(clip);
            if (active || await ask([h("b", {}, "Delete clip?"), h("br"), `${clipTitle(clip)} will be permanently deleted from your comma device.`], "Delete")) remove(clip);
          }}, active ? ic("close", 22, "closeBold", 24) : ic("trash", 22, "trash", 24)))),
      clip.status === "encoding" && (um ? linear() : h("div", {class: "clips-bar busy"})),
      previewing && [um ? linear(s.progress) : h("div", {class: "clips-bar", style: `--p: ${s.progress}`}),
        h("div", {class: "clips-transfer"}, `Downloading · ${Math.round(s.progress * 100)}%`)]);
  }

  const view = {name: "clips", el: menu, render() {
    if (route) {
      const [t0, t1] = range(), duration = t1 - t0, out = duration / s.speedup;
      const available = !s.ranges || CAMERAS.some(([v]) => covers(s.ranges, v, t0, t1));
      rangeText.textContent = `${fmt.clock(Math.round(t0))}–${fmt.clock(Math.round(t1))} · ${dur(duration)}`;
      cameras.set(s.camera, (v) => s.ranges && !covers(s.ranges, v, t0, t1));
      quality.set(s.bitrate);
      speed.set(s.speedup);
      hint.hidden = available;
      output.textContent = `Output: ${dur(out)}`;
      size.textContent = `About ${mb(out * s.bitrate * 125000)}`;
      name.placeholder = defaultName(t0, t1);
      nameError.hidden = validName(name.value);
      tooLong.hidden = duration <= MAX_DURATION;
      const busy = s.clips.some((c) => ACTIVE.has(c.status));
      createBtn.disabled = !online || s.loading || s.creating || busy || duration <= 0 || duration > MAX_DURATION || !validName(name.value)
        || (s.ranges && !covers(s.ranges, s.camera, t0, t1));
      createBtn.replaceChildren(s.creating ? (um ? progress(18) : h("span", {class: "spinner"})) : !online ? "Device offline" : busy ? "Clip in progress" : "Create clip");
    }
    error.textContent = s.error || "";
    error.hidden = !s.error;
    // rebuilt, focus kept on the same control
    const focus = list.contains(document.activeElement) && document.activeElement.dataset.focus;
    list.replaceChildren(...s.loading ? [h("div", {class: "clips-empty"}, um ? progress(18) : h("span", {class: "spinner"}))]
      : !s.clips.length ? [h("p", {class: "clips-empty"}, online ? "No clips yet" : "Device offline")] : s.clips.map(row));
    if (focus) [...list.querySelectorAll("[data-focus]")].find((b) => b.dataset.focus === focus)?.focus();
  }};

  // connect names a clip's drive by its start, for the drives its device page has (the same first page device.js asks for)
  if (um) {
    api.routes(dongle, undefined, Math.ceil(Date.now() / 3600e3) * 3600e3, 50).then((routes) => {
      for (const r of routes) s.starts.set(routeName(r), r.start_time_utc_millis);
      changed();
    }, () => {});
  }
  anchor.style.anchorName = "--clips";
  document.body.append(menu);
  view.render();
  menu.showModal();
  load();
  return menu;
}

// the drive view's Clip button (connect's, beside Files): shown once the device says it cuts clips, dimmed while offline
export function clipButton({route, clock}) {
  const b = h("button", {class: "cn-option clips-button", hidden: true, "aria-haspopup": "dialog"}, "Clip");
  if (route.loose || !route.fullname?.includes("|")) return b;
  api.device(route.fullname.split("|")[0]).then(async (device) => {
    if (!await supportsClips(device)) return;
    const online = isOnline(device);
    b.hidden = false;
    b.title = online ? "" : "Device offline";
    b.ariaDisabled = String(!online);
    b.onclick = () => online && clipMenu(b, device, route, () => clock.range);
  }).catch(() => {});
  return b;
}

// a device's clips, connect's scissors in its header
export function deviceClips(device) {
  const online = isOnline(device);
  const b = h("button", {class: "icon-btn clips-device", hidden: true, title: online ? "Clips" : "Device offline", "aria-label": "Clips",
    "aria-haspopup": "dialog", disabled: !online, onclick: () => clipMenu(b, device)}, ic("cut", 16, "cut", 24));
  supportsClips(device).then((ok) => { b.hidden = !ok; }, () => {});
  return b;
}
