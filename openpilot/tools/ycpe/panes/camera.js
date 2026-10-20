// the camera pane: qcamera through hls.js, or full-res road/wide/driver video decoded in a worker (WebCodecs,
// else wasm). It follows the clock: each render shows the frame at clock.t, or the newest one decoded so far.
// Each open camera publishes its segments to status.js; while the playhead waits on one, the pane says so.
import {api} from "/api.js";
import {h, css} from "/dom.js";
import {status} from "/status.js";
import {FPS, frameAt} from "/data/video/frames.js";
import {frameRenderer} from "/data/video/gl.js";

const CAMERAS = {qcamera: "qcam", fcamera: "road", ecamera: "wide", dcamera: "driver"};
const LANES = {fcamera: ["fcam", 1], ecamera: ["ecam", 2], dcamera: ["dcam", 3], qcamera: ["qcam", 4]};
const FILES = {fcamera: "cameras", ecamera: "ecameras", dcamera: "dcameras"};
const SEGMENT = 60;       // seconds
const AHEAD = 8;          // frames decoded ahead of the playhead while playing
const BACK = 19;          // frames kept behind the playhead for stepping backwards while paused (one GOP)
const MIN_RATE = 0.0625;  // the slowest <video> playbackRate; slower playback seeks frame by frame
const QCAMERA_BYTES_PER_S = 37e3;
const HLS = "https://cdn.jsdelivr.net/npm/hls.js@1.6.7/dist/hls.min.js";
const SOFTWARE = "This browser has no hardware HEVC decoder (Chrome on Linux needs VA-API), so video decodes in software.";

export function create(ctx, state) {
  let camera = CAMERAS[state.camera] ? state.camera : "qcamera";
  let source = null;
  css(new URL("./camera.css", import.meta.url));
  const note = h("div", {class: "camera-note", hidden: true});
  const bar = h("i");
  const waitText = h("span");
  const overlay = h("div", {class: "camera-buffering", hidden: true}, waitText, h("div", {class: "camera-progress"}, bar));
  const picker = h("select", {class: "camera-picker", title: "Camera", onchange: (e) => use(e.target.value)},
    Object.entries(CAMERAS).map(([k, label]) => h("option", {value: k}, label)));
  const el = h("div", {class: "camera-pane"});
  const pane = {name: "camera", el, render: () => source.render(), resize() {}, save: () => ({camera}), dispose,
    stats: () => source.stats};

  const ui = {
    say(text = "", title = "") {
      note.textContent = text;
      note.title = title;
      note.hidden = !text;
    },
    // the playhead waits on a segment: {text, f} with f the fraction loaded, or null; the last frame stays, dimmed
    wait(b) {
      overlay.hidden = !b;
      el.classList.toggle("buffering", !!b);
      if (!b) return;
      waitText.textContent = `buffering ${b.text}`;
      bar.style.width = `${(100 * (b.f || 0)).toFixed(1)}%`;
    },
  };

  function show(src, why) {
    source?.dispose();
    source = src;
    ui.say(why);
    ui.wait(null);
    el.replaceChildren(src.el, overlay, picker, note);
    ctx.perf.invalidate(pane);
  }

  function use(cam) {
    camera = picker.value = cam;
    if (cam === "qcamera") show(qcamera(ctx, pane, ui));
    else show(fullres(ctx, pane, cam, {...ui, fail: (why) => show(qcamera(ctx, pane, ui), `${why}, showing qcamera`)}));
  }

  const offs = ["t", "play"].map((ev) => ctx.clock.on(ev, () => ctx.perf.invalidate(pane)));
  function dispose() {
    for (const off of offs) off();
    source.dispose();
  }

  use(camera);
  return pane;
}

function lane(camera) {
  const [label, order] = LANES[camera];
  return status.source(camera, {label, order});
}

// hls.js loads with the first qcamera pane, not with the page
let hlsReady = null;
const loadHls = () => hlsReady ||= new Promise((resolve, reject) => {
  document.head.append(h("script", {src: HLS, onload: () => resolve(window.Hls), onerror: reject}));
});

// the HLS player plays along with the clock and corrects drift by nudging its rate; paused, it seeks to each t.
// The segment's thumbnail stands in until the first frame arrives.
function qcamera(ctx, pane, {say, wait}) {
  const poster = api.segmentAsset(ctx.route, Math.floor(ctx.clock.t / SEGMENT), "sprite.jpg");
  const video = h("video", {muted: true, playsinline: true, preload: "auto", poster});
  const lanes = lane("qcamera");
  const count = Math.ceil(ctx.clock.duration / SEGMENT);
  lanes.count(count);
  const loading = new Set();
  let hls = null, gone = false;

  // segments the <video> has buffered, from its time ranges
  const update = () => {
    const r = video.buffered;
    let seconds = 0;
    for (let n = 0; n < count; n++) {
      const a = n * SEGMENT, b = Math.min(ctx.clock.duration, a + SEGMENT);
      let have = 0;
      for (let i = 0; i < r.length; i++) have += Math.max(0, Math.min(b, r.end(i)) - Math.max(a, r.start(i)));
      seconds += have;
      lanes.seg(n, have >= 0.98 * (b - a) ? "ready" : loading.has(n) ? "loading" : "idle");
    }
    lanes.bytes({memory: seconds * QCAMERA_BYTES_PER_S});
  };
  loadHls().then((Hls) => {
    if (gone) return;
    hls = new Hls({maxBufferLength: 20});
    hls.on(Hls.Events.FRAG_LOADING, (e, {frag}) => { loading.add(frag.sn); update(); });
    hls.on(Hls.Events.FRAG_BUFFERED, (e, {frag}) => { loading.delete(frag.sn); update(); });
    hls.loadSource(api.qcameraUrl(ctx.route));
    hls.attachMedia(video);
  }, () => say("hls.js failed to load"));
  for (const ev of ["loadedmetadata", "seeked", "canplay", "waiting"]) video.addEventListener(ev, () => ctx.perf.invalidate(pane));
  video.addEventListener("progress", update);

  return {
    el: video,
    render() {
      const {t, playing, rate} = ctx.clock;
      const n = Math.floor(t / SEGMENT), stalled = video.readyState < 3 && (playing || video.seeking || video.readyState < 2);
      const b = stalled && {n, text: `segment ${n}`};
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
      lanes.remove();
      hls?.destroy();
      video.removeAttribute("src");
      video.load();
    },
  };
}

// a window of decoded frames around the playhead, filled by the decoder worker
function fullres(ctx, pane, camera, {say, wait, fail}) {
  const canvas = h("canvas");
  const draw = frameRenderer(canvas);
  const worker = new Worker("/data/video/decoder.js", {type: "module"});
  const lanes = lane(camera);
  const frames = new Map();  // n → VideoFrame, or I420 planes from the software decoder
  const stats = {shown: -1, drawn: 0, lag: 0, fps: 0, held: 0, seekMs: null};
  let ready = false, software = false, shown = -1, last = -1, asked = "", missingSince = 0, decoded = 0;

  api.routeFiles(ctx.route.fullname).then((files) => {
    const urls = [], keys = [];
    for (const u of files[FILES[camera]] || []) {
      const path = new URL(u).pathname.split("/"), n = +path.at(-2);
      urls[n] = u;
      keys[n] = `${ctx.route.fullname.replace("|", "/")}/${n}/${path.at(-1)}`;  // the logs' OPFS key scheme
    }
    if (!urls.length) return fail(`no ${CAMERAS[camera]} camera files`);
    lanes.count(urls.length);
    worker.postMessage({urls, keys});
    ready = true;
    ctx.perf.invalidate(pane);
  }).catch(() => fail("camera files unavailable"));

  worker.onmessage = ({data}) => {
    if (data.unsupported) return fail(`this browser can't decode ${data.unsupported.startsWith("hev") ? "HEVC" : data.unsupported}`);
    if (data.backend) {
      software = (stats.backend = data.backend) !== "hardware";
      if (software) say("software decode", SOFTWARE);
    } else if (data.seg !== undefined) {
      lanes.seg(data.seg, data.state, data);
    } else if (data.buffering !== undefined) {
      const b = data.buffering, f = b?.total ? b.loaded / b.total : 0;
      const w = b && {n: b.n, text: `segment ${b.n}${f ? ` · ${Math.round(100 * f)}%` : ""}`, f};
      lanes.buffering(w);
      wait(w);
    } else if (data.memory !== undefined) {
      let held = 0;
      for (const f of frames.values()) held += f.data?.byteLength ?? f.allocationSize();
      lanes.bytes({memory: data.memory + held, disk: data.disk});
      lanes.rate(data.rate);
      stats.fps = (data.decoded - decoded) * 2;
      decoded = data.decoded;
      ctx.perf.mark(`${camera} decode`, `${stats.fps} fps · ${frames.size} held`);
      if (software) say(`software decode${stats.fps ? ` · ${stats.fps} fps` : ""}`, SOFTWARE);
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
      const n = frameAt(ctx.peekStore(), camera, ctx.clock.t);
      const playing = ctx.clock.playing, back = n < last ? BACK : 0;
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
        draw(frames.get(best));
        shown = stats.shown = best;
        stats.drawn++;
      }
      if (playing && best >= 0) stats.lag = Math.max(stats.lag, n - best);
      if (best === n && missingSince) {
        stats.seekMs = Math.round(performance.now() - missingSince);
        ctx.perf.mark(`${camera} seek`, `${stats.seekMs} ms`);
        missingSince = 0;
      } else if (best !== n && !missingSince) {
        missingSince = performance.now();
      }

      const want = {at: n, to: n + (playing ? AHEAD : 2), back, playing};
      const key = `${want.at} ${want.to}`;
      if (key !== asked || closed.length) worker.postMessage({want, closed, buffers}, buffers);
      asked = key;
    },
    dispose() {
      lanes.remove();
      worker.terminate();
      for (const f of frames.values()) f.close?.();
      frames.clear();
    },
  };
}
