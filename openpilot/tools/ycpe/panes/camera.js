// the camera pane: qcamera through hls.js, or full-res road/wide/driver video decoded in a worker (WebCodecs,
// else wasm). It follows the clock: each render shows the frame at clock.t, or the newest one decoded so far.
import {api} from "/api.js";
import {h, css} from "/dom.js";
import {FPS, frameAt} from "/data/video/frames.js";
import {frameRenderer} from "/data/video/gl.js";

const CAMERAS = {qcamera: "qcam", fcamera: "road", ecamera: "wide", dcamera: "driver"};
const FILES = {fcamera: "cameras", ecamera: "ecameras", dcamera: "dcameras"};
const AHEAD = 8;          // frames decoded ahead of the playhead while playing
const BACK = 19;          // frames kept behind the playhead for stepping backwards while paused (one GOP)
const MIN_RATE = 0.0625;  // the slowest <video> playbackRate; slower playback seeks frame by frame
const SOFTWARE = "This browser has no hardware HEVC decoder (Chrome on Linux needs VA-API), so video decodes in software.";

export function create(ctx, state) {
  let camera = CAMERAS[state.camera] ? state.camera : "qcamera";
  let source = null;
  css(new URL("./camera.css", import.meta.url));
  const note = h("div", {class: "camera-note", hidden: true});
  const picker = h("select", {class: "camera-picker", title: "Camera", onchange: (e) => use(e.target.value)},
    Object.entries(CAMERAS).map(([k, label]) => h("option", {value: k}, label)));
  const el = h("div", {class: "camera-pane"});
  const pane = {name: "camera", el, render: () => source.render(), resize() {}, save: () => ({camera}), dispose,
    stats: () => source.stats};

  const say = (text = "", title = "") => {
    note.textContent = text;
    note.title = title;
    note.hidden = !text;
  };

  function show(src, why) {
    source?.dispose();
    source = src;
    say(why);
    el.replaceChildren(src.el, picker, note);
    ctx.perf.invalidate(pane);
  }

  function use(cam) {
    camera = picker.value = cam;
    if (cam === "qcamera") show(qcamera(ctx, pane));
    else show(fullres(ctx, pane, cam, {say, fail: (why) => show(qcamera(ctx, pane), `${why}, showing qcamera`)}));
  }

  const offs = ["t", "play"].map((ev) => ctx.clock.on(ev, () => ctx.perf.invalidate(pane)));
  function dispose() {
    for (const off of offs) off();
    source.dispose();
  }

  use(camera);
  return pane;
}

// the HLS player plays along with the clock and corrects drift by nudging its rate; paused, it seeks to each t.
// The segment's thumbnail stands in until the first frame arrives.
function qcamera(ctx, pane) {
  const poster = api.segmentAsset(ctx.route, Math.floor(ctx.clock.t / 60), "sprite.jpg");
  const video = h("video", {muted: true, playsinline: true, preload: "auto", poster});
  const hls = new Hls({maxBufferLength: 20});
  hls.loadSource(api.qcameraUrl(ctx.route));
  hls.attachMedia(video);
  for (const ev of ["loadedmetadata", "seeked", "canplay"]) video.addEventListener(ev, () => ctx.perf.invalidate(pane));
  return {
    el: video,
    render() {
      const {t, playing, rate} = ctx.clock;
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
      hls.destroy();
      video.removeAttribute("src");
      video.load();
    },
  };
}

// a window of decoded frames around the playhead, filled by the decoder worker
function fullres(ctx, pane, camera, {say, fail}) {
  const canvas = h("canvas");
  const draw = frameRenderer(canvas);
  const worker = new Worker("/data/video/decoder.js", {type: "module"});
  const frames = new Map();  // n → VideoFrame, or I420 planes from the software decoder
  const stats = {shown: -1, drawn: 0, lag: 0, fps: 0, held: 0, mb: 0, seekMs: null};
  let ready = false, software = false, shown = -1, last = -1, asked = "", missingSince = 0, statTime = performance.now(), statFrames = 0;

  api.routeFiles(ctx.route.fullname).then((files) => {
    const urls = [];
    for (const u of files[FILES[camera]] || []) urls[+new URL(u).pathname.split("/").at(-2)] = u;
    if (!urls.length) return fail(`no ${CAMERAS[camera]} camera files`);
    worker.postMessage({urls});
    ready = true;
    ctx.perf.invalidate(pane);
  }).catch(() => fail("camera files unavailable"));

  worker.onmessage = ({data}) => {
    if (data.unsupported) return fail(`this browser can't decode ${data.unsupported.startsWith("hev") ? "HEVC" : data.unsupported}`);
    if (data.backend) {
      software = (stats.backend = data.backend) !== "hardware";
      return software && say("software decode", SOFTWARE);
    }
    frames.set(data.n, data.frame);
    const now = performance.now();
    if (now - statTime > 1000) {
      Object.assign(stats, {fps: Math.round((data.decoded - statFrames) * 1000 / (now - statTime)), mb: Math.round(data.bytes / 1e6)});
      statTime = now;
      statFrames = data.decoded;
      ctx.perf.mark(`${camera} decode`, `${stats.fps} fps · ${stats.held} held · ${stats.mb} MB`);
      if (software) say(`software decode${stats.fps ? ` · ${stats.fps} fps` : ""}`, SOFTWARE);
    }
    ctx.perf.invalidate(pane);
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
      worker.terminate();
      for (const f of frames.values()) f.close?.();
      frames.clear();
    },
  };
}
