// the global playback bar: transport, rate, time, and the route timeline (engagement, a lane per loader showing
// what's loaded, visible range, loop, playhead, hover preview). Everything on the timeline is one canvas; time with
// nothing loaded at all is faded. A live session's timeline is its window, ending at the head of the data, with the
// source's state, "Go live" (or that it's following) and saving the session as an rlog.
import {SEGMENT, api, routeEvents, segmentAt} from "/api.js";
import {h, fmt, css} from "/dom.js";
import {icon} from "/icons.js";
import {perf} from "/perf.js";
import {theme} from "/theme.js";
import {RATES} from "/clock.js";
import {status, drawLane, describe} from "/status.js";

const RANGE_H = 7, SPAN_H = 14, LANE_H = 3, PAD = 3;
const SPAN_COLORS = {engaged: "--engaged", override: "--override", alert: "--alert", critical: "--critical"};
const SPAN_LABELS = {engaged: "engaged", override: "override", alert: "alert", critical: "critical alert"};
// built once: toLocaleTimeString with options builds a formatter each call, and the first costs ~7 ms of ICU setup
const WALL = new Intl.DateTimeFormat(undefined, {hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false});

export function playbar(ctx) {
  const {clock, route} = ctx, live = route.live;
  if (live) css(new URL("./live.css", import.meta.url));
  let spans = [];
  routeEvents(route).then((s) => { spans = s; perf.invalidate(pane); }).catch(() => {});

  const playBtn = h("button", {class: "tbtn play", title: "Play/pause (space)", onclick: () => clock.toggle()});
  const rate = h("select", {class: "rate", title: "Playback rate ([ and ])", onchange: (e) => clock.setRate(+e.target.value)},
    RATES.map((r) => h("option", {value: r, selected: r === clock.rate}, `${r}×`)));
  const time = h("span", {class: "pb-time"});
  const wall = h("span", {class: "pb-wall muted"});
  const loopBtn = h("button", {class: "tbtn", title: "Loop the visible range (L)", "aria-pressed": "false", onclick: () => clock.setLoop(clock.loop ? null : [...clock.range])}, icon("loop", 16));
  const fitBtn = h("button", {class: "tbtn", title: "Show the whole route (F)", onclick: () => clock.setRange(clock.start, clock.duration)}, icon("fit", 16));
  // the timeline is the playhead's slider for keyboards and screen readers
  const canvas = h("canvas", {class: "pb-canvas", role: "slider", tabindex: 0, "aria-label": "Playhead", "aria-valuemin": 0,
    "aria-valuemax": Math.floor(clock.duration), onkeydown: (e) => {
      const step = e.shiftKey || e.key.startsWith("Page") ? 10 : 1;
      const to = {ArrowLeft: -step, ArrowDown: -step, PageDown: -step, ArrowRight: step, ArrowUp: step, PageUp: step}[e.key];
      if (to === undefined && e.key !== "Home" && e.key !== "End") return;
      e.preventDefault();
      clock.seek(e.key === "Home" ? clock.start : e.key === "End" ? clock.duration : clock.t + to);
    }});
  const tip = h("div", {class: "pb-tip", hidden: true}, h("div", {class: "pb-thumb"}), h("span"));
  const track = h("div", {class: "pb-track"}, canvas, tip);
  const liveBtn = live && h("button", {class: "pb-live", onclick: () => clock.goLive()});
  const state = live && h("span", {class: "pb-state"});
  const saveBtn = live && h("button", {class: "tbtn", title: "Save the session as an rlog", onclick: save}, icon("download", 16));
  const el = h("footer", {class: "playbar"},
    h("button", {class: "tbtn", title: "Back 1 s (←, shift: 10 s)", onclick: () => clock.seek(clock.t - 1)}, icon("rewind", 16)),
    playBtn,
    h("button", {class: "tbtn", title: "Forward 1 s (→, shift: 10 s)", onclick: () => clock.seek(clock.t + 1)}, icon("forward", 16)),
    rate, h("span", {class: "pb-clock"}, time, wall), liveBtn, state, track, loopBtn, fitBtn, saveBtn);

  // the timeline spans the route, or a live session's window
  let width = 0, hover = null;
  const x2t = (x) => clock.start + x / width * (clock.duration - clock.start);
  const xOf = (e) => e.clientX - canvas.getBoundingClientRect().left;
  const yOf = (e) => e.clientY - canvas.getBoundingClientRect().top;
  const t2x = (t) => (t - clock.start) / (clock.duration - clock.start) * width;

  // the session's events as rlog bytes, downloaded (raw: the codec here only decompresses)
  async function save() {
    saveBtn.disabled = true;
    try {
      const bytes = await live.save(), a = h("a", {href: URL.createObjectURL(new Blob([bytes])), download: `${live.name.replace(/\W+/g, "_")}--rlog`});
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    } finally {
      saveBtn.disabled = false;
    }
  }

  let shownLive = null, shownState = null;
  function renderLive() {
    if (shownLive !== clock.follow) {
      shownLive = clock.follow;
      liveBtn.classList.toggle("on", clock.follow);
      liveBtn.textContent = clock.follow ? "live" : "go live";
      liveBtn.title = clock.follow ? "Following the newest data" : "Back to the newest data (End)";
    }
    // the source's state when it isn't simply streaming; a source that needs a click to start (WebUSB) gets a button
    const {source} = live, key = `${source.state} ${source.detail}`;
    if (shownState === key) return;
    shownState = key;
    state.hidden = source.state === "live";
    state.dataset.state = source.state;
    state.replaceChildren(source.state === "error" ? `error: ${source.detail}` : source.detail || source.state,
      source.connect && source.state !== "live" && h("button", {class: "chip", onclick: () => source.connect()}, "Connect"));
  }

  let playing = null, second = null, wallSecond = null;  // what the play button, the slider and the wall clock show
  const pane = {name: "playbar", render() {
    if (playing !== clock.playing) playBtn.replaceChildren(icon((playing = clock.playing) ? "pause" : "play", 16));
    rate.value = clock.rate;
    loopBtn.classList.toggle("on", !!clock.loop);
    if (loopBtn.ariaPressed !== String(!!clock.loop)) loopBtn.ariaPressed = String(!!clock.loop);
    if (second !== Math.trunc(clock.t)) {
      second = Math.trunc(clock.t);
      canvas.setAttribute("aria-valuenow", second);
      canvas.setAttribute("aria-valuetext", fmt.clock(second));
    }
    time.textContent = `${fmt.clock(clock.t, true)} / ${fmt.clock(clock.duration)}`;
    if (live) renderLive();
    const w = route.start_time_utc_millis ? Math.floor(route.start_time_utc_millis / 1000 + clock.t) : null;
    if (wallSecond !== w) wall.textContent = (wallSecond = w) === null ? "" : WALL.format(w * 1000);

    const kinds = status.kinds();
    const HEIGHT = PAD + RANGE_H + 2 + SPAN_H + 2 + kinds.length * (LANE_H + 1) + PAD;
    const dpr = devicePixelRatio;
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
      c.fillRect(t2x(s.start / 1000), spanY, Math.max(1, t2x(s.end / 1000) - t2x(s.start / 1000)), SPAN_H);
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
    for (let i = Math.floor(clock.start / SEGMENT); i * SEGMENT < clock.duration; i++) {
      if (kinds.some((k) => ["ready", "disk", "loading"].includes(k.segs[i]?.state))) continue;
      const [a, b] = segX(i);
      c.fillRect(a, spanY, b - a, SPAN_H);
    }
    c.globalAlpha = 1;
    // loop, visible range, hover, playhead
    if (clock.loop) {
      c.fillStyle = col("--loop");
      c.fillRect(t2x(clock.loop[0]), 0, t2x(clock.loop[1]) - t2x(clock.loop[0]), HEIGHT);
    }
    const [r0, r1] = clock.range;
    c.fillStyle = col("--range");
    c.fillRect(t2x(r0), PAD, Math.max(3, t2x(r1) - t2x(r0)), RANGE_H);
    if (hover !== null) {
      c.fillStyle = col("--muted");
      c.fillRect(Math.round(t2x(hover)), spanY, 1, HEIGHT - spanY - PAD);
    }
    c.fillStyle = col("--fg");
    const px = Math.round(t2x(clock.t));
    c.fillRect(px - 1, 0, 2, HEIGHT);
  }};

  const offs = ["t", "range", "play", "duration"].map((ev) => clock.on(ev, () => perf.invalidate(pane)));
  offs.push(status.on(() => perf.invalidate(pane)), theme.on(() => perf.invalidate(pane)));
  if (live) offs.push(live.source.on("state", () => perf.invalidate(pane)));
  const resizer = new ResizeObserver(([e]) => { width = e.contentRect.width; perf.invalidate(pane); });
  resizer.observe(track);

  // dragging in the range strip pans/resizes the visible range; anywhere else scrubs the playhead
  canvas.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    const x0 = xOf(e), t0 = x2t(x0);
    const [r0, r1] = clock.range;
    let mode = "scrub";
    const touch = e.pointerType === "touch";  // a finger gets a taller strip and wider edges
    if (yOf(e) < (touch ? 14 : PAD + RANGE_H + 1)) {
      const edge = touch ? 16 : 5;
      mode = Math.abs(x0 - t2x(r0)) < edge ? "left" : Math.abs(x0 - t2x(r1)) < edge ? "right" : t0 > r0 && t0 < r1 ? "pan" : "new";
    }
    const move = (ev) => {
      const t = Math.max(clock.start, Math.min(clock.duration, x2t(xOf(ev))));
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
    const slot = Math.max(0, Math.min(Math.floor(hover / SEGMENT), Math.ceil(clock.duration / SEGMENT) - 1));
    const n = segmentAt(route, slot);  // a range's slots map to its segments
    const sprite = route.url && api.segmentAsset(route, n, "sprite.jpg");  // local files have none
    tip.firstChild.hidden = !sprite;
    if (sprite) tip.firstChild.style.backgroundImage = `url(${sprite})`;
    // the spans under the cursor, with the alert's text once logs are loaded
    const ms = hover * 1000, alert = ctx.peekStore()?.valueAt("/selfdriveState/alertText1", hover);
    const under = spans.filter((s) => SPAN_LABELS[s.kind] && s.start <= ms && ms < s.end)
      .map((s) => h("div", {}, SPAN_LABELS[s.kind] + ((s.kind === "alert" || s.kind === "critical") && alert ? `: ${alert}` : "")));
    tip.lastChild.replaceChildren(...under, `${fmt.clock(hover)} · segment ${n}`,
      ...status.kinds().filter((k) => k.segs[slot]).map((k) => h("div", {class: "muted"}, `${k.label}: ${describe(k.segs[slot])}`)));
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
  return {el, dispose() { for (const off of offs) off(); resizer.disconnect(); }};
}
