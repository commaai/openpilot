// the global playback bar: transport, rate, time, and the route timeline (engagement, a lane per loader showing
// what's loaded, visible range, loop, playhead, hover preview). Everything on the timeline is one canvas; time with
// nothing loaded at all is faded.
import {SEGMENT, api, routeEvents, segmentAt} from "/api.js";
import {h, fmt, parseClock} from "/dom.js";
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
  const {clock, route} = ctx;
  let spans = [];
  // alerts last, so they're drawn over the engagement around them
  const isAlert = (s) => s.kind === "alert" || s.kind === "critical";
  routeEvents(route).then((s) => { spans = [...s].sort((a, b) => isAlert(a) - isAlert(b)); perf.invalidate(pane); }).catch(() => {});

  const playBtn = h("button", {class: "tbtn play", title: "Play/pause (space)", onclick: () => clock.toggle()});
  const rate = h("select", {class: "rate", title: "Playback rate ([ and ])", onchange: (e) => clock.setRate(+e.target.value)},
    RATES.map((r) => h("option", {value: r, selected: r === clock.rate}, `${r}×`)));
  // click the time to type one to go to
  const time = h("button", {class: "pb-time", title: "Go to a time (3:12, 1:03:12 or seconds)", onclick: () => {
    goto.value = fmt.clock(clock.t, true);
    time.hidden = true;
    goto.hidden = false;
    goto.focus();
    goto.select();
  }});
  const goto = h("input", {class: "pb-time", hidden: true, "aria-label": "Go to time", spellcheck: false});
  const shown = () => {
    const focused = document.activeElement === goto;
    goto.hidden = true;
    goto.removeAttribute("aria-invalid");
    time.hidden = false;
    if (focused) time.focus();
  };
  // Enter goes there (a time it can't read stays, to be fixed); Escape or leaving doesn't
  goto.addEventListener("keydown", (e) => {
    const t = parseClock(goto.value);
    if (e.key === "Enter") e.preventDefault();  // not a click on the time button, focused again
    if (e.key === "Enter" && Number.isNaN(t)) goto.setAttribute("aria-invalid", "true");
    else if (e.key === "Enter") { clock.seek(t); shown(); }
    else if (e.key === "Escape") shown();
  });
  goto.addEventListener("blur", () => { if (!goto.hidden) shown(); });
  const wall = h("span", {class: "pb-wall muted"});
  const loopBtn = h("button", {class: "tbtn", title: "Loop the visible range (L)", "aria-pressed": "false", onclick: () => clock.setLoop(clock.loop ? null : [...clock.range])}, icon("loop", 16));
  const fitBtn = h("button", {class: "tbtn", title: "Show the whole route (F)", onclick: () => clock.setRange(0, clock.duration)}, icon("fit", 16));
  // the timeline is the playhead's slider for keyboards and screen readers
  const canvas = h("canvas", {class: "pb-canvas", role: "slider", tabindex: 0, "aria-label": "Playhead", "aria-valuemin": 0,
    "aria-valuemax": Math.floor(clock.duration), onkeydown: (e) => {
      const step = e.shiftKey || e.key.startsWith("Page") ? 10 : 1;
      const to = {ArrowLeft: -step, ArrowDown: -step, PageDown: -step, ArrowRight: step, ArrowUp: step, PageUp: step}[e.key];
      if (to === undefined && e.key !== "Home" && e.key !== "End") return;
      e.preventDefault();
      clock.seek(e.key === "Home" ? 0 : e.key === "End" ? clock.duration : clock.t + to);
    }});
  const tip = h("div", {class: "pb-tip", hidden: true}, h("div", {class: "pb-thumb"}), h("span"));
  const track = h("div", {class: "pb-track"}, canvas, tip);
  const el = h("footer", {class: "playbar"},
    h("button", {class: "tbtn", title: "Back 1 s (←, shift: 10 s)", onclick: () => clock.seek(clock.t - 1)}, icon("rewind", 16)),
    playBtn,
    h("button", {class: "tbtn", title: "Forward 1 s (→, shift: 10 s)", onclick: () => clock.seek(clock.t + 1)}, icon("forward", 16)),
    rate, h("span", {class: "pb-clock"}, time, goto, wall), track, loopBtn, fitBtn);

  let width = 0, hover = null;
  // where the canvas is, read once as the pointer comes onto it (not on every move, after the tooltip's writes)
  let rect = null;
  const x2t = (x) => x / width * clock.duration;
  const xOf = (e) => e.clientX - (rect ||= canvas.getBoundingClientRect()).left;
  const yOf = (e) => e.clientY - (rect ||= canvas.getBoundingClientRect()).top;
  const t2x = (t) => t / clock.duration * width;

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
      c.fillRect(t2x(s.start / 1000), spanY, Math.max(isAlert(s) ? 3 : 1, t2x((s.end - s.start) / 1000)), SPAN_H);  // a short alert still shows
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

  const offs = ["t", "range", "play"].map((ev) => clock.on(ev, () => perf.invalidate(pane)));
  offs.push(status.on(() => perf.invalidate(pane)), theme.on(() => perf.invalidate(pane)));
  const resizer = new ResizeObserver(([e]) => { width = e.contentRect.width; rect = null; perf.invalidate(pane); });
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
  // the tooltip: its time follows the pointer, the rest is rebuilt only when it changes
  const hoverText = document.createTextNode("");
  let tipText = "";
  canvas.addEventListener("pointermove", (e) => {
    const x = xOf(e);
    hover = Math.max(0, Math.min(clock.duration, x2t(x)));  // a captured pointer goes past the ends
    perf.invalidate(pane);
    tip.hidden = false;
    tip.style.left = `${Math.max(0, Math.min(width - 136, x - 68))}px`;
    const slot = Math.max(0, Math.min(Math.floor(hover / SEGMENT), Math.ceil(clock.duration / SEGMENT) - 1));
    const n = segmentAt(route, slot);  // a range's slots map to its segments
    // the spans under the cursor, with the alert's text once logs are loaded
    const ms = hover * 1000, alert = ctx.peekStore()?.valueAt("/selfdriveState/alertText1", hover);
    const under = spans.filter((s) => SPAN_LABELS[s.kind] && s.start <= ms && ms < s.end)
      .map((s) => SPAN_LABELS[s.kind] + (isAlert(s) && alert ? `: ${alert}` : ""));
    const loads = status.kinds().filter((k) => k.segs[slot]).map((k) => `${k.label}: ${describe(k.segs[slot])}`);
    hoverText.data = `${fmt.clock(hover)} · segment ${n}`;
    const text = [n, ...under, ...loads].join("\n");
    if (text === tipText) return;
    tipText = text;
    const sprite = route.url && api.segmentAsset(route, n, "sprite.jpg");  // local files have none
    tip.firstChild.hidden = !sprite;
    if (sprite) tip.firstChild.style.backgroundImage = `url(${sprite})`;
    tip.lastChild.replaceChildren(...under.map((u) => h("div", {}, u)), hoverText, ...loads.map((l) => h("div", {class: "muted"}, l)));
  });
  canvas.addEventListener("pointerleave", () => { hover = rect = null; tip.hidden = true; perf.invalidate(pane); });
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
