// the global playback bar: transport, rate, time, and the route timeline (engagement, a lane per loader showing
// what's loaded, visible range, loop, playhead, hover preview). Everything on the timeline is one canvas; time with
// nothing loaded at all is faded.
import {api, routeEvents} from "/api.js";
import {h} from "/dom.js";
import {perf} from "/perf.js";
import {theme} from "/theme.js";
import {RATES} from "/clock.js";
import {status, drawLane, describe} from "/status.js";

const RANGE_H = 7, SPAN_H = 14, LANE_H = 3, PAD = 3;
const SEGMENT = 60;  // nominal segment length; lanes index segments
const SPAN_COLORS = {engaged: "--engaged", override: "--override", alert: "--alert", critical: "--critical"};

export const clockText = (s, withMs = true) => {
  const sign = s < 0 ? "-" : "";
  s = Math.abs(s);
  const hr = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, sec = s % 60;
  const ss = withMs ? sec.toFixed(2).padStart(5, "0") : String(Math.floor(sec)).padStart(2, "0");
  return `${sign}${hr ? `${hr}:${String(m).padStart(2, "0")}` : m}:${ss}`;
};

export function playbar(ctx) {
  const {clock, route} = ctx;
  let spans = [];
  routeEvents(route).then((s) => { spans = s; perf.invalidate(pane); }).catch(() => {});

  const playBtn = h("button", {class: "tbtn play", title: "Play/pause (space)", onclick: () => clock.toggle()});
  const rate = h("select", {class: "rate", title: "Playback rate ([ and ])", onchange: (e) => clock.setRate(+e.target.value)},
    RATES.map((r) => h("option", {value: r, selected: r === clock.rate}, `${r}×`)));
  const time = h("span", {class: "pb-time"});
  const wall = h("span", {class: "pb-wall muted"});
  const loopBtn = h("button", {class: "tbtn", title: "Loop the visible range (L)", onclick: () => clock.setLoop(clock.loop ? null : [...clock.range])}, "⟲");
  const fitBtn = h("button", {class: "tbtn", title: "Show the whole route (F)", onclick: () => clock.setRange(0, clock.duration)}, "⤢");
  const canvas = h("canvas", {class: "pb-canvas"});
  const tip = h("div", {class: "pb-tip", hidden: true}, h("div", {class: "pb-thumb"}), h("span"));
  const track = h("div", {class: "pb-track"}, canvas, tip);
  const el = h("footer", {class: "playbar"},
    h("button", {class: "tbtn", title: "Back 1 s (←, shift: 10 s)", onclick: () => clock.seek(clock.t - 1)}, "⏮"),
    playBtn,
    h("button", {class: "tbtn", title: "Forward 1 s (→, shift: 10 s)", onclick: () => clock.seek(clock.t + 1)}, "⏭"),
    rate, h("span", {class: "pb-clock"}, time, wall), track, loopBtn, fitBtn);

  let width = 0, hover = null;
  const x2t = (x) => x / width * clock.duration;
  const xOf = (e) => e.clientX - canvas.getBoundingClientRect().left;
  const yOf = (e) => e.clientY - canvas.getBoundingClientRect().top;
  const t2x = (t) => t / clock.duration * width;

  const pane = {name: "playbar", render() {
    playBtn.textContent = clock.playing ? "❚❚" : "▶";
    rate.value = clock.rate;
    loopBtn.classList.toggle("on", !!clock.loop);
    time.textContent = `${clockText(clock.t)} / ${clockText(clock.duration, false)}`;
    wall.textContent = route.start_time_utc_millis ? new Date(route.start_time_utc_millis + clock.t * 1000).toLocaleTimeString(undefined, {hour12: false}) : "";

    const kinds = status.kinds();
    const HEIGHT = PAD + RANGE_H + 2 + SPAN_H + 2 + kinds.length * (LANE_H + 1) + PAD;
    const dpr = devicePixelRatio;
    width = track.clientWidth;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== HEIGHT * dpr) {
      canvas.width = Math.round(width * dpr);
      canvas.height = HEIGHT * dpr;
      canvas.style.height = `${HEIGHT}px`;
    }
    const c = canvas.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, width, HEIGHT);
    const col = (v) => theme.color(v);
    const spanY = PAD + RANGE_H + 2, segY = spanY + SPAN_H + 2;

    // engagement
    c.fillStyle = col("--driving");
    c.fillRect(0, spanY, width, SPAN_H);
    for (const s of spans) {
      if (!SPAN_COLORS[s.kind]) continue;
      c.fillStyle = col(SPAN_COLORS[s.kind]);
      c.fillRect(t2x(s.start / 1000), spanY, Math.max(1, t2x((s.end - s.start) / 1000)), SPAN_H);
    }
    // bookmarks: a flag above the spans
    c.fillStyle = col("--bookmark");
    for (const s of spans) {
      if (s.kind !== "bookmark") continue;
      const x = Math.round(t2x(s.start / 1000));
      c.fillRect(x, spanY - 3, 2, SPAN_H + 3);
      c.fillRect(x, spanY - 3, 5, 3);
    }
    // what's loaded, a lane per loader; fade the time nothing has started on
    const segX = (i) => [t2x(i * SEGMENT), t2x(Math.min(clock.duration, (i + 1) * SEGMENT))];
    kinds.forEach((k, j) => drawLane(c, k, 0, segY + j * (LANE_H + 1), width, LANE_H, segX));
    c.fillStyle = col("--panel");
    c.globalAlpha = 0.6;
    for (let i = 0; i * SEGMENT < clock.duration; i++) {
      if (kinds.some((k) => ["ready", "disk", "loading"].includes(k.segs[i]?.state))) continue;
      const [a, b] = segX(i);
      c.fillRect(a, spanY, b - a, SPAN_H);
    }
    c.globalAlpha = 1;
    // loop, visible range, hover, playhead
    if (clock.loop) {
      c.fillStyle = col("--loop");
      c.fillRect(t2x(clock.loop[0]), 0, t2x(clock.loop[1] - clock.loop[0]), HEIGHT);
    }
    const [r0, r1] = clock.range;
    c.fillStyle = col("--range");
    c.fillRect(t2x(r0), PAD, Math.max(3, t2x(r1 - r0)), RANGE_H);
    if (hover !== null) {
      c.fillStyle = col("--muted");
      c.fillRect(Math.round(t2x(hover)), spanY, 1, HEIGHT - spanY - PAD);
    }
    c.fillStyle = col("--fg");
    const px = Math.round(t2x(clock.t));
    c.fillRect(px - 1, 0, 2, HEIGHT);
  }};

  for (const ev of ["t", "range", "play"]) clock.on(ev, () => perf.invalidate(pane));
  status.on(() => perf.invalidate(pane));
  theme.on(() => perf.invalidate(pane));
  new ResizeObserver(() => perf.invalidate(pane)).observe(track);

  // dragging in the range strip pans/resizes the visible range; anywhere else scrubs the playhead
  canvas.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    const x0 = xOf(e), t0 = x2t(x0);
    const [r0, r1] = clock.range;
    let mode = "scrub";
    if (yOf(e) < PAD + RANGE_H + 1) {
      const edge = 5;
      mode = Math.abs(x0 - t2x(r0)) < edge ? "left" : Math.abs(x0 - t2x(r1)) < edge ? "right" : t0 > r0 && t0 < r1 ? "pan" : "new";
    }
    const move = (ev) => {
      const t = Math.max(0, Math.min(clock.duration, x2t(xOf(ev))));
      if (mode === "scrub") clock.seek(t);
      else if (mode === "pan") clock.setRange(r0 + t - t0, r1 + t - t0);
      else if (mode === "left") clock.setRange(Math.min(t, r1 - 0.1), r1);
      else if (mode === "right") clock.setRange(r0, Math.max(t, r0 + 0.1));
      else clock.setRange(Math.min(t0, t), Math.max(t0, t) + 0.1);
    };
    move(e);
    canvas.onpointermove = move;
    canvas.onpointerup = () => { canvas.onpointermove = null; canvas.onpointerup = null; };
  });
  canvas.addEventListener("pointermove", (e) => {
    const x = xOf(e);
    hover = x2t(x);
    perf.invalidate(pane);
    tip.hidden = false;
    tip.style.left = `${Math.max(0, Math.min(width - 136, x - 68))}px`;
    const seg = Math.max(0, Math.floor(hover / 60));
    tip.firstChild.style.backgroundImage = `url(${api.segmentAsset(route, Math.min(seg, route.maxqlog ?? seg), "sprite.jpg")})`;
    tip.lastChild.replaceChildren(`${clockText(hover, false)} · segment ${seg}`,
      ...status.kinds().filter((k) => k.segs[seg]).map((k) => h("div", {class: "muted"}, `${k.label}: ${describe(k.segs[seg])}`)));
  });
  canvas.addEventListener("pointerleave", () => { hover = null; tip.hidden = true; perf.invalidate(pane); });
  // wheel zooms the visible range around the cursor
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const [r0, r1] = clock.range, t = x2t(xOf(e));
    const k = Math.exp(e.deltaY * 0.002);
    clock.setRange(t - (t - r0) * k, t + (r1 - t) * k);
  }, {passive: false});

  perf.invalidate(pane);
  return el;
}
