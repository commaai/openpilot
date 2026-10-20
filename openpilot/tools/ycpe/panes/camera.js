// the camera pane: qcamera through hls.js, or full-res road/wide/driver video decoded in a worker with WebCodecs.
// It follows the clock: each render shows the frame at clock.t, or the newest one decoded so far.
import {api} from "/api.js";
import {h, css} from "/dom.js";
import {FPS, frameAt} from "/data/video/frames.js";

const CAMERAS = {qcamera: "qcam", fcamera: "road", ecamera: "wide", dcamera: "driver"};
const FILES = {fcamera: "cameras", ecamera: "ecameras", dcamera: "dcameras"};
const AHEAD = 8;          // frames decoded ahead of the playhead while playing
const BACK = 19;          // frames kept behind the playhead for stepping backwards (one GOP)
const MIN_RATE = 0.0625;  // the slowest <video> playbackRate; slower playback seeks frame by frame

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

  function show(src, why = "") {
    source?.dispose();
    source = src;
    note.textContent = why;
    note.hidden = !why;
    el.replaceChildren(src.el, picker, note);
    ctx.perf.invalidate(pane);
  }

  function use(cam) {
    camera = picker.value = cam;
    if (cam === "qcamera") show(qcamera(ctx, pane));
    else show(fullres(ctx, pane, cam, (why) => show(qcamera(ctx, pane), `${why}, showing qcamera`)));
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
function fullres(ctx, pane, camera, fail) {
  const canvas = h("canvas");
  const g = canvas.getContext("2d", {alpha: false});
  const worker = new Worker("/data/video/decoder.js", {type: "module"});
  const frames = new Map();  // n → VideoFrame
  const stats = {shown: -1, fps: 0, held: 0, mb: 0, seekMs: null};
  let shown = -1, last = -1, asked = "", missingSince = 0, statTime = performance.now(), statFrames = 0;

  api.routeFiles(ctx.route.fullname).then((files) => {
    const urls = [];
    for (const u of files[FILES[camera]] || []) urls[+new URL(u).pathname.split("/").at(-2)] = u;
    if (!urls.length) return fail(`no ${CAMERAS[camera]} camera files`);
    worker.postMessage({urls});
    ctx.perf.invalidate(pane);
  }).catch(() => fail("camera files unavailable"));

  worker.onmessage = ({data}) => {
    if (data.unsupported) return fail(`this browser can't decode ${data.unsupported.startsWith("hev") ? "HEVC" : data.unsupported}`);
    frames.set(data.n, data.frame);
    const now = performance.now();
    if (now - statTime > 1000) {
      Object.assign(stats, {fps: Math.round((data.decoded - statFrames) * 1000 / (now - statTime)), mb: Math.round(data.bytes / 1e6)});
      statTime = now;
      statFrames = data.decoded;
      ctx.perf.mark(`${camera} decode`, `${stats.fps} fps · ${stats.held} held · ${stats.mb} MB`);
    }
    ctx.perf.invalidate(pane);
  };

  return {
    el: canvas,
    stats,
    render() {
      const n = frameAt(ctx.peekStore?.(), camera, ctx.clock.t);
      const playing = ctx.clock.playing, back = n < last ? BACK : 0;
      last = n;
      const closed = [];
      for (const [k, f] of frames) {
        if (k >= n - BACK && k <= n + AHEAD) continue;
        f.close();
        frames.delete(k);
        closed.push(k);
      }
      stats.held = frames.size;

      let best = -1;
      for (const k of frames.keys()) if (k <= n && k > best) best = k;
      if (best !== shown && best >= 0) {
        const f = frames.get(best);
        if (canvas.width !== f.displayWidth || canvas.height !== f.displayHeight) [canvas.width, canvas.height] = [f.displayWidth, f.displayHeight];
        g.drawImage(f, 0, 0);
        shown = stats.shown = best;
      }
      if (best === n && missingSince) {
        stats.seekMs = Math.round(performance.now() - missingSince);
        ctx.perf.mark(`${camera} seek`, `${stats.seekMs} ms`);
        missingSince = 0;
      } else if (best !== n && !missingSince) {
        missingSince = performance.now();
      }

      const want = {at: n, to: n + (playing ? AHEAD : 2), back};
      const key = `${want.at} ${want.to}`;
      if (key !== asked || closed.length) worker.postMessage({want, closed});
      asked = key;
    },
    dispose() {
      worker.terminate();
      for (const f of frames.values()) f.close();
      frames.clear();
    },
  };
}
