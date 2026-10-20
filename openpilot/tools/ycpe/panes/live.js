// a live session's picture in the camera pane (data/live.js), its track as the source hands it over whatever the
// playhead, with the controls of connect's stream view (BodyTeleop, minus the joystick): the connection's state and
// stats, the stream's quality (bitrate) and a screenshot. 1 and 2 switch road and cabin while the pane has the focus
// (elsewhere they switch views).
import {h, css} from "/dom.js";
import {icon} from "/icons.js";
import {format, sample} from "/data/live/rtcstats.js";

const QUALITY = {auto: "", high: "5 Mbps", med: "1.5 Mbps", low: "500 kbps"};  // webrtcd's bitrates
const NAMES = {fcamera: "road", ecamera: "wideRoad", dcamera: "driver"};  // the cameras as webrtcd names them
const KEYS = {1: "fcamera", 2: "dcamera"};
const STATS = [["fps", "FPS"], ["bitrate", "Bitrate"], ["rtt", "RTT"], ["loss", "Loss"]];
const STATS_MS = 1000;

// camera: the pane's (fcamera...), label: its name to people; ui: the camera pane's (absent), use(camera) switches
export function liveVideo(ctx, pane, camera, label, {absent, use}) {
  const {source} = ctx.route.live;
  css(new URL("./live.css", import.meta.url));  // its controls show once there's a picture
  const video = h("video", {autoplay: true, playsinline: true});
  video.muted = true;
  let track = null, last = null, open = false, drawn = "";
  const off = ctx.route.live.video(camera, (t) => {
    track = t;
    last = null;
    video.srcObject = t && new MediaStream([t]);
    ctx.perf.invalidate(pane);
  });
  if (NAMES[camera]) source.setCamera?.(NAMES[camera]);
  if (!source.stats) return {el: video, render: () => absent(track ? null : `no ${label} stream`), dispose() { off(); video.srcObject = null; }};

  // getStats about once a second, off the frame; the next render shows it
  const poll = async () => {
    if (!track || !pane.el.isConnected) return;
    const id = track.id, report = await source.stats();
    if (!report || id !== track?.id) return;
    last = sample(report, last, id);
    ctx.perf.invalidate(pane);
  };
  const polling = setInterval(poll, STATS_MS);
  poll();

  const dot = h("i"), rtt = h("span");
  const values = Object.fromEntries(STATS.map(([k]) => [k, h("dd")]));
  const panel = h("dl", {class: "live-stats", hidden: true}, STATS.map(([k, name]) => [h("dt", {}, name), values[k]]));
  const status = h("button", {class: "live-status", title: "Connection: show the stats", "aria-expanded": "false", onclick: () => {
    open = !open;
    ctx.perf.invalidate(pane);
  }}, dot, rtt);
  const quality = h("select", {class: "live-quality", title: "Quality (bitrate)", onchange: (e) => source.setQuality(e.target.value)},
    Object.entries(QUALITY).map(([q, rate]) => h("option", {value: q}, q, rate && h("span", {class: "muted"}, ` ${rate}`))));
  quality.value = source.quality;
  const shot = h("button", {class: "live-shot", title: "Save screenshot", "aria-label": "Save screenshot", onclick: () => track && screenshot(track, video, label)},
    icon("photo_camera", 18));
  const controls = h("div", {class: "live-controls"}, quality, shot);
  const el = h("div", {class: "live-video"}, video, h("div", {class: "live-bar"}, status, panel), controls);

  // 1 and 2 as in connect, where the global keys would switch views
  const key = (e) => {
    if (!KEYS[e.key] || e.metaKey || e.ctrlKey || e.altKey || e.target.closest("select")) return;
    e.preventDefault();
    if (KEYS[e.key] !== camera) use(KEYS[e.key]);
  };
  pane.el.tabIndex = 0;
  pane.el.addEventListener("keydown", key);

  return {
    el,
    render() {
      absent(track ? null : `no ${label} stream`);
      const s = format(last), poor = !!last?.poor;
      const now = `${!!track} ${open} ${poor} ${Object.values(s)}`;
      if (now === drawn) return;  // the stats change once a second
      drawn = now;
      el.classList.toggle("on", !!track);
      dot.className = poor ? "poor" : "";
      status.title = `${poor ? "Poor connection" : "Connected"}: ${open ? "hide" : "show"} the stats`;
      status.setAttribute("aria-expanded", String(open));
      rtt.textContent = s.rtt;
      panel.hidden = !open;
      for (const [k] of STATS) values[k].textContent = s[k];
    },
    dispose() {
      off();
      clearInterval(polling);
      pane.el.removeEventListener("keydown", key);
      video.srcObject = null;
    },
  };
}

// the camera's next frame as a PNG, shared where there's a share sheet (phones) or else downloaded, as connect saves it
async function screenshot(track, video, label) {
  // the track's frame, handed over rather than drawn: a canvas's drawImage of the <video>, or createImageBitmap of it,
  // stalls the main thread ~50 ms the first time. Safari has no ImageCapture
  const frame = await (window.ImageCapture ? new ImageCapture(track).grabFrame() : createImageBitmap(video));
  const canvas = new OffscreenCanvas(frame.width, frame.height);
  canvas.getContext("bitmaprenderer").transferFromImageBitmap(frame);
  const blob = await canvas.convertToBlob({type: "image/png"});
  const name = `screenshot_${label}_${Date.now()}.png`, file = new File([blob], name, {type: "image/png"});
  if (navigator.canShare?.({files: [file]})) {
    try {
      return await navigator.share({files: [file]});
    } catch (e) {
      if (e.name === "AbortError") return;
    }
  }
  const a = h("a", {href: URL.createObjectURL(blob), download: name});
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
