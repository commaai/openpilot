// the camera pane: qcamera through hls.js, or full-res road/wide/driver video decoded in a worker (WebCodecs,
// else wasm). It follows the clock: each render shows the frame at clock.t, or the newest one decoded so far.
// Each open camera publishes its segments to status.js; while the playhead waits on one, the pane says so.
// Once logs are loading, the onroad alert at the playhead shows over the picture, as in cabana.
import {SEGMENT, api, segmentAt} from "/api.js";
import {daemon} from "/daemon.js";
import {h, css} from "/dom.js";
import {onJobs} from "/local.js";
import {status} from "/status.js";
import {routeCameraFiles, routeCameras} from "/data/sources.js";
import {HEVC} from "/data/video/annexb.js";
import {FPS, SEGMENT_FRAMES, cameraTime, frameAt} from "/data/video/frames.js";
import {frameRenderer} from "/data/video/gl.js";
import {liveVideo} from "/panes/live.js";

await css(new URL("./camera.css", import.meta.url));

const CAMERAS = {qcamera: "qcam", fcamera: "road", ecamera: "wide", dcamera: "driver"};
const ALERT_STATUS = ["normal", "prompt", "critical"];  // selfdriveState.alertStatus
const LANES = {fcamera: ["fcam", 1], ecamera: ["ecam", 2], dcamera: ["dcam", 3], qcamera: ["qcam", 4]};
const AHEAD = 8;          // frames decoded ahead of the playhead while playing
const BACK = 19;          // frames kept behind the playhead for stepping backwards while paused (one GOP)
const SCRUB_MS = 100;     // seeks closer together than this are a scrub: keyframes only until it settles
const MIN_RATE = 0.0625;  // the slowest <video> playbackRate; slower playback seeks frame by frame
const QCAMERA_BYTES_PER_S = 37e3;
const HLS = "https://cdn.jsdelivr.net/npm/hls.js@1.6.7/dist/hls.light.min.js";  // 348 KB, too big to vendor
const HLS_SRI = "sha384-voUqi4Z0V5OBtD0Fe5jaAgsk8ZD/RDweFf6rHo9CrwrIyfJh304PNzV/VxKIWfMG";
const SOFTWARE = "This browser has no hardware HEVC decoder (Chrome on Linux needs VA-API), so video decodes in software.";
const TRANSCODED = "This browser has no hardware HEVC decoder, so the local daemon transcodes the video to H.264.";
const UNBUILT = "full-res needs hardware HEVC, the local daemon's transcode, or a built hevc.wasm (tools/ycpe/data/video/build.sh)";

export function create(ctx, state) {
  let camera = CAMERAS[state.camera] ? state.camera : "qcamera";
  let source = null;
  const note = h("div", {class: "camera-note", hidden: true});
  const bar = h("i");
  const waitText = h("span");
  const progress = h("div", {class: "camera-progress"}, bar);
  const overlay = h("div", {class: "camera-buffering", hidden: true}, waitText, progress);
  let overlaid = "";
  const picker = h("select", {class: "camera-picker", title: "Camera", onchange: (e) => use(e.target.value)},
    Object.entries(CAMERAS).map(([k, label]) => h("option", {value: k}, label)));
  const alertEl = h("div", {class: "camera-alert", hidden: true});
  let alerted = "";
  const el = h("div", {class: "camera-pane"});
  const pane = {name: "camera", el, render() { source.render(); showAlert(); }, resize() {}, save: () => ({camera}), dispose,
    stats: () => source.stats};

  // selfdriveState's alert at the playhead; the DOM changes only with it
  function showAlert() {
    const store = ctx.peekStore(), at = (f) => store.valueAt(`/selfdriveState/${f}`, ctx.clock.t);
    const text = store && at("alertSize") ? [at("alertText1"), at("alertText2")].filter(Boolean).join("\n") : "";
    const alertStatus = text ? ALERT_STATUS[at("alertStatus")] || "normal" : "";
    if (`${alertStatus} ${text}` === alerted) return;
    alerted = `${alertStatus} ${text}`;
    alertEl.hidden = !text;
    alertEl.textContent = text;
    alertEl.dataset.status = alertStatus;
  }

  // the overlay over the last frame, dimmed: text with a progress bar at f (0..1), text alone (f null), or nothing
  const cover = (text, f) => {
    const key = text === null ? "" : `${text} ${f}`;
    if (key === overlaid) return;  // renders run every frame while playing
    overlaid = key;
    overlay.hidden = text === null;
    el.classList.toggle("buffering", text !== null);
    if (text === null) return;
    waitText.textContent = text;
    progress.hidden = f === null;
    bar.style.width = `${(100 * (f || 0)).toFixed(1)}%`;
  };
  const ui = {
    say(text = "", title = "") {
      note.textContent = text;
      note.title = title || text;  // the note is cut to one line
      note.hidden = !text;
    },
    // the playhead waits on a segment: {text, f} with f the fraction loaded, or null
    wait: (b) => cover(b ? `buffering ${b.text}` : null, b?.f || 0),
    // the playhead is where the camera has no file: why, or null
    absent: (why) => cover(why, null),
  };

  function show(src, why) {
    source?.dispose();
    source = src;
    ui.say(why);
    ui.absent(null);
    el.replaceChildren(src.el, alertEl, overlay, picker, note);
    ctx.perf.invalidate(pane);
  }

  // the full-res canvas and its renderer outlive a switch of camera: a new WebGL context costs ~13 ms
  let screen = null;
  function fullresScreen() {
    if (!screen) {
      const canvas = h("canvas");
      screen = {canvas, restored: () => {}, draw: frameRenderer(canvas, () => screen.restored())};
    }
    screen.canvas.hidden = true;  // a canvas is black until its first frame, or shows the last camera's
    return screen;
  }

  function use(cam) {
    camera = picker.value = cam;
    if (ctx.route.live) show(liveVideo(ctx, pane, cam, CAMERAS[cam], {...ui, use}));
    else if (cam === "qcamera") show(qcamera(ctx, pane, ui));
    else show(fullres(ctx, pane, cam, fullresScreen(), {...ui, fail: (why) => show(qcamera(ctx, pane, ui), why)}));
  }

  const offs = ["t", "play"].map((ev) => ctx.clock.on(ev, () => ctx.perf.invalidate(pane)));
  let disposed = false;
  offs.push(ctx.onStore((store) => !disposed && offs.push(store.on("segment", () => ctx.perf.invalidate(pane)))));  // its alerts
  function dispose() {
    disposed = true;
    for (const off of offs) off?.();
    source.dispose();
  }

  use(camera);
  // a camera the route has no file of can't be picked; a live session's cameras are whatever the source sends
  if (!ctx.route.live) {
    routeCameras(ctx.route).then((have) => {
      if (disposed) return;
      for (const o of picker.options) {
        o.disabled = !have.has(o.value);
        o.textContent = `${CAMERAS[o.value]}${o.disabled ? ctx.route.loose ? " (no files)" : " (not uploaded)" : ""}`;
      }
    }, () => {});
  }
  return pane;
}

function lane(camera) {
  const [label, order] = LANES[camera];
  return status.source(camera, {label, order});
}

// hls.js (the light build: no subtitles, alternate audio or DRM) loads with the first qcamera pane, not with the page.
// It runs in the origin that holds the auth tokens, so the CDN's copy must match the pinned hash. It downloads right
// away but runs in idle time: its ~20 ms evaluation would otherwise pile onto the busy frames of a cold load. (Chrome's
// native HLS would skip it, but its player stalls the compositor ~65 ms on every start, and hangs on a seek past the
// video's last frame.)
let hlsReady = null;
const loadHls = () => hlsReady ||= new Promise((resolve, reject) => {
  const attrs = {integrity: HLS_SRI, crossorigin: "anonymous"};
  document.head.append(h("link", {rel: "preload", as: "script", href: HLS, ...attrs}));
  requestIdleCallback(() => document.head.append(h("script", {src: HLS, ...attrs, onload: () => resolve(window.Hls), onerror: reject})), {timeout: 1000});
});

// the HLS player plays along with the clock and corrects drift by nudging its rate; paused, it seeks to each t.
// The segment's thumbnail stands in until the first frame arrives. Where the route has no qcamera file there's
// nothing to wait for: the pane says so, and with none at all it doesn't load a player. A segment that fails to
// load stops the player until the next seek, play or reconnect.
function qcamera(ctx, pane, {wait, absent}) {
  const poster = ctx.route.loose ? "" : api.segmentAsset(ctx.route, segmentAt(ctx.route, Math.floor(ctx.clock.t / SEGMENT)), "sprite.jpg");
  const video = h("video", {playsinline: true, preload: "auto", poster});
  video.muted = true;  // the property: the attribute set after creation doesn't mute it, and only a muted video plays unasked
  const lanes = lane("qcamera");
  const count = Math.ceil(ctx.clock.duration / SEGMENT);
  lanes.count(count);
  const loading = new Set();
  let hls = null, gone = false, files = null, list = null, src = null;  // files[k]: slot k's file, once known; list: our playlist's blob URL
  let failed = null, lastT = 0;  // failed: {n, t}, the segment that didn't load and the playhead then

  // segments the <video> has buffered, from its time ranges
  const update = () => {
    const r = video.buffered;
    let seconds = 0;
    for (let n = 0; n < count; n++) {
      const a = n * SEGMENT, b = Math.min(ctx.clock.duration, a + SEGMENT);
      let have = 0;
      for (let i = 0; i < r.length; i++) have += Math.max(0, Math.min(b, r.end(i)) - Math.max(a, r.start(i)));
      seconds += have;
      lanes.seg(n, files && !files[n] ? "missing" : have >= 0.98 * (b - a) ? "ready" : failed?.n === n ? "error" : loading.has(n) ? "loading" : "idle");
    }
    lanes.bytes({memory: seconds * QCAMERA_BYTES_PER_S});
  };
  const retry = () => {
    if (!failed || gone) return;
    failed = null;
    hls.loadSource(src);  // hls.js skips a failed fragment as a gap for good: start over, and render() seeks back
    update();
  };
  Promise.all([loadHls(), routeCameraFiles(ctx.route, "qcamera").catch(() => null)]).then(([Hls, found]) => {
    if (gone) return;
    files = found;
    update();
    ctx.perf.invalidate(pane);
    if (files && !files.some(Boolean)) return;
    src = ctx.route.loose || ctx.route.range ? list = playlist(ctx.route, files) : api.qcameraUrl(ctx.route);
    hls = new Hls({maxBufferLength: 20});
    hls.on(Hls.Events.FRAG_LOADING, (e, {frag}) => { loading.add(frag.sn); update(); });
    hls.on(Hls.Events.FRAG_BUFFERED, (e, {frag}) => { loading.delete(frag.sn); update(); });
    hls.on(Hls.Events.ERROR, (e, data) => {
      // a fragment out of retries isn't fatal when there's no other level to switch to, but it never loads
      const {retryCount, retryConfig} = data.errorAction || {};
      if (!data.fatal && !(data.frag && retryConfig && retryCount >= retryConfig.maxNumRetry)) return;  // hls.js retries
      if (data.type === Hls.ErrorTypes.MEDIA_ERROR) return hls.recoverMediaError();
      failed = {n: data.frag?.sn ?? Math.floor(video.currentTime / SEGMENT), t: ctx.clock.t};
      loading.clear();
      update();
      ctx.perf.invalidate(pane);
    });
    hls.loadSource(src);
    hls.attachMedia(video);
  }, () => absent("hls.js failed to load"));
  for (const ev of ["loadedmetadata", "seeked", "canplay", "waiting"]) video.addEventListener(ev, () => ctx.perf.invalidate(pane));
  // a paused <video> keeps its poster up over the decoded frame until a seek or play
  video.addEventListener("loadeddata", () => video.removeAttribute("poster"), {once: true});
  video.addEventListener("progress", update);
  const offPlay = ctx.clock.on("play", retry);
  addEventListener("online", retry);

  return {
    el: video,
    render() {
      const {playing, rate} = ctx.clock, t = cameraTime(ctx.peekStore(), ctx.route, "qcamera", ctx.clock.t);
      const n = Math.floor(t / SEGMENT);
      // a seek retries: any move while paused, a jump while playing
      if (failed && (playing ? Math.abs(ctx.clock.t - lastT) > 1 : ctx.clock.t !== failed.t)) retry();
      lastT = ctx.clock.t;
      if (files && !files[n]) {
        lanes.buffering(null);
        absent(files.some(Boolean) ? `no qcamera for segment ${segmentAt(ctx.route, n)}` : "no qcamera for this route");
        if (!video.paused) video.pause();
        return;
      }
      const stalled = video.readyState < 3 && (playing || video.seeking || video.readyState < 2);
      if (failed && stalled) {
        lanes.buffering(null);
        return absent(`segment ${segmentAt(ctx.route, failed.n)} failed to load · seek or play to retry`);
      }
      const b = stalled && {n, text: `segment ${segmentAt(ctx.route, n)}`};
      lanes.buffering(b || null);
      wait(b);
      if (video.readyState < 1 || video.seeking) return;  // "seeked" renders again with the latest t
      const drift = video.currentTime - t;
      if (playing && rate >= MIN_RATE) {
        if (Math.abs(drift) > Math.max(1, rate)) video.currentTime = t;
        video.playbackRate = Math.abs(drift) < 1 / FPS ? rate : Math.max(MIN_RATE, rate * Math.min(2, Math.max(0.5, 1 - drift)));
        if (video.paused) video.play().catch(() => {});
      } else {
        if (!video.paused) video.pause();
        if (Math.abs(drift) > 0.5 / FPS) video.currentTime = t;
      }
    },
    dispose() {
      gone = true;
      offPlay();
      removeEventListener("online", retry);
      lanes.remove();
      hls?.destroy();
      if (list) URL.revokeObjectURL(list);
      video.removeAttribute("src");
      video.load();
    },
  };
}

// a window of decoded frames around the playhead, filled by the decoder worker
function fullres(ctx, pane, camera, screen, {say, wait, absent, fail}) {
  const {canvas, draw} = screen;
  screen.restored = () => { shown = -1; ctx.perf.invalidate(pane); };
  const worker = new Worker("/data/video/decoder.js", {type: "module"});
  const lanes = lane(camera);
  const frames = new Map();  // n → VideoFrame, or I420 planes from the software decoder
  const stats = {shown: -1, exact: false, drawn: 0, lag: 0, fps: 0, held: 0, seekMs: null};
  let ready = false, transcoded = false, software = false, shown = -1, last = -1, asked = "", missingSince = 0, decoded = 0;
  let jumped = 0, settle = 0, scrubbing = false;  // when the playhead last jumped; a render once a scrub settles
  let gone = false;

  const jobs = new Map();  // segment → {done, total} frames of the daemon's transcode
  const segState = [];     // the worker's last word on each segment
  const badge = () => say([transcoded && "daemon H.264", software && "software decode", software && stats.fps && `${stats.fps} fps`].filter(Boolean).join(" · "),
    transcoded ? TRANSCODED : SOFTWARE);
  let stopJobs = null, waitingOn = null;
  const waitText = (b) => {
    const job = jobs.get(b.n), f = job ? job.done / job.total : b.total ? b.loaded / b.total : 0;
    return {n: b.n, text: `segment ${segmentAt(ctx.route, b.n)}${job ? ` · transcoding${job.done ? "" : " · starting"}` : ""}${f ? ` · ${Math.round(100 * f)}%` : ""}`, f};
  };
  const buffering = (b) => {
    waitingOn = b;
    const w = b && waitText(b);
    lanes.buffering(w);
    wait(w);
  };

  let real = [], have = [];  // each slot's segment, and whether it has a file
  sources(ctx.route, camera).then(({urls, keys, segs, ...src}) => {
    if (gone) return;
    real = segs;
    have = segs.map((_, k) => !!urls[k]);
    if (!have.some(Boolean)) return fail(`no ${CAMERAS[camera]} camera files, showing qcamera`);
    lanes.count(segs.length);
    worker.postMessage({urls, keys});
    ready = true;
    ctx.perf.invalidate(pane);
    if (!(transcoded = src.transcoded)) return;
    badge();
    stopJobs = onJobs((job) => {
      const {route, camera: cam} = job.meta || {}, seg = real.indexOf(job.meta?.seg);  // its timeline slot
      if (job.kind !== "transcode" || route !== ctx.route.fullname || cam !== camera || seg < 0) return;
      if (job.state === "running") {
        jobs.set(seg, job);
        lanes.seg(seg, "loading", {loaded: job.done, total: job.total});
      } else {
        jobs.delete(seg);  // the daemon holds the finished file
        lanes.seg(seg, job.state === "done" && segState[seg] !== "ready" ? "disk" : segState[seg] || "error");
      }
      if (waitingOn?.n === seg) buffering(waitingOn);
    });
  }).catch(() => gone || fail("camera files unavailable, showing qcamera"));

  worker.onmessage = ({data}) => {
    if (data.unbuilt) return fail(UNBUILT);
    if (data.unsupported) return fail(`this browser can't decode ${data.unsupported.startsWith("hev") ? "HEVC" : data.unsupported}, showing qcamera`);
    if (data.backend) {
      software = (stats.backend = data.backend) !== "hardware";
      badge();
    } else if (data.seg !== undefined) {
      segState[data.seg] = data.state;
      if (!(data.state === "loading" && jobs.has(data.seg))) lanes.seg(data.seg, data.state, data);  // transcode progress wins
    } else if (data.buffering !== undefined) {
      buffering(data.buffering);
    } else if (data.memory !== undefined) {
      let held = 0;
      for (const f of frames.values()) held += f.data?.byteLength ?? f.allocationSize();
      lanes.bytes({memory: data.memory + held, disk: data.disk});
      lanes.rate(data.rate);
      stats.fps = (data.decoded - decoded) * 2;
      decoded = data.decoded;
      ctx.perf.mark(`${camera} decode`, `${stats.fps} fps · ${frames.size} held`);
      badge();
    } else {
      frames.set(data.n, data.frame);
      ctx.perf.invalidate(pane);
    }
  };

  return {
    el: canvas,
    stats,
    render() {
      if (!ready) return;
      const n = frameAt(ctx.peekStore(), ctx.route, camera, ctx.clock.t), slot = Math.floor(n / SEGMENT_FRAMES);
      if (!have[slot]) absent(`no ${CAMERAS[camera]} camera for segment ${segmentAt(ctx.route, slot)}`);
      else wait(waitingOn && waitText(waitingOn));
      const playing = ctx.clock.playing, back = n < last ? BACK : 0, now = performance.now();
      if (!playing && Math.abs(n - last) > 1) {
        scrubbing = now - jumped < SCRUB_MS;
        jumped = now;
        clearTimeout(settle);
        settle = setTimeout(() => ctx.perf.invalidate(pane), SCRUB_MS);
      } else if (now - jumped >= SCRUB_MS) {
        scrubbing = false;
      }
      last = n;
      // hardware decoders have few output buffers, so playback keeps only what's ahead
      const lo = n - (playing ? 1 : BACK), closed = [], buffers = [];
      for (const [k, f] of frames) {
        if (k >= lo && k <= n + AHEAD) continue;
        if (f.data) buffers.push(f.data.buffer);  // planes go back to the worker for reuse
        else f.close();
        frames.delete(k);
        closed.push(k);
      }
      stats.held = frames.size;

      let best = -1;
      for (const k of frames.keys()) if (k <= n && k > best) best = k;
      if (best !== shown && best >= 0) {
        if (draw(frames.get(best))) {
          canvas.hidden = false;
          shown = stats.shown = best;
          stats.drawn++;
        } else {
          ctx.perf.invalidate(pane);  // the shaders are still linking
        }
      }
      stats.exact = best === n;  // else a keyframe (or older frame) stands in until the exact one decodes
      if (playing && best >= 0) stats.lag = Math.max(stats.lag, n - best);
      if (best === n && missingSince) {
        stats.seekMs = Math.round(performance.now() - missingSince);
        ctx.perf.mark(`${camera} seek`, `${stats.seekMs} ms`);
        missingSince = 0;
      } else if (best !== n && !missingSince) {
        missingSince = performance.now();
      }

      const want = {at: n, to: n + (playing ? AHEAD : 2), back, playing, keyOnly: scrubbing};
      const key = `${want.at} ${want.to} ${scrubbing}`;
      if (key !== asked || closed.length) worker.postMessage({want, closed, buffers}, buffers);
      asked = key;
    },
    dispose() {
      gone = true;
      stopJobs?.();
      lanes.remove();
      worker.terminate();
      for (const f of frames.values()) f.close?.();
      frames.clear();
    },
  };
}

// the camera's per-segment files from the route's source: the device's HEVC (cached in OPFS under the logs' key
// scheme), or, when this browser can't decode HEVC in hardware, the local daemon's H.264 transcode, which the daemon
// caches itself. Loose routes (dropped files, direct links) have just the files they came with. All of it is by timeline
// slot (a range plays its segments back to back); segs[k] is slot k's segment.
async function sources(route, camera) {
  const files = await routeCameraFiles(route, camera), urls = [], keys = [];
  const segs = Array.from({length: slots(route)}, (_, k) => segmentAt(route, k));
  urls.length = keys.length = segs.length;
  files.forEach(({url, file}, k) => {
    urls[k] = url;
    keys[k] = `${route.fullname.replace("|", "/")}/${segs[k]}/${file}`;
  });
  const hardware = (await VideoDecoder.isConfigSupported({codec: HEVC, hardwareAcceleration: "prefer-hardware"})).supported;
  if (hardware || route.loose || !daemon.has("transcode")) return {urls, keys, segs, transcoded: false};
  const [dongle, log] = route.fullname.split("|");
  const transcode = (k) => daemon.url(`/local/video/${dongle}/${log}/${segs[k]}/${camera}.h264?preset=medium&ahead=2`);
  return {urls: urls.map((u, k) => u && transcode(k)), keys: [], segs, transcoded: true};
}

const slots = (route) => route.range ? route.segment_numbers.length : route.maxqlog + 1;

// an HLS playlist of qcamera files by timeline slot, for a loose route (which the API doesn't have) or a range of
// a route; segments it lacks are gaps, and a jump to a segment not next to the last restarts the timestamps
function playlist(route, files) {
  const lines = ["#EXTM3U", "#EXT-X-VERSION:3", `#EXT-X-TARGETDURATION:${SEGMENT}`, "#EXT-X-PLAYLIST-TYPE:VOD"];
  for (let k = 0, last = null; k < slots(route); k++) {
    const n = segmentAt(route, k);
    if (last !== null && n !== last + 1) lines.push("#EXT-X-DISCONTINUITY");
    const f = files?.[k];
    lines.push(`#EXTINF:${SEGMENT},`, ...f ? [] : ["#EXT-X-GAP"], new URL(f?.url ?? `gap${n}.ts`, location.href).href);
    last = f ? n : null;
  }
  lines.push("#EXT-X-ENDLIST");
  return URL.createObjectURL(new Blob([lines.join("\n")], {type: "application/vnd.apple.mpegurl"}));
}
