// the connect view: connect's route page built from our panes. A card with the drive's times, its timeline (the strip
// of panes/strip.js over a ruler: tap to seek, drag to zoom into a range), Video | Map and
// Files | More info toggles, the camera or map pane, and connect's control pill. The camera, map, files and info panes
// are mounted here as the layout mounts them. Everything that moves while scrubbing or playing is a transform or a
// changed text, set in one render through the frame scheduler.
import {SEGMENT, segmentAt} from "/api.js";
import {h, css} from "/dom.js";
import {icon} from "/icons.js";
import {RATES} from "/clock.js";
import {loadPane} from "/panes.js";
import {link} from "/app.js";
import {strip} from "./strip.js";
import {sound, muteButton} from "./camera.js";

await css(new URL("./connect.css", import.meta.url));

const MOVE_PX = 6;  // a press that moves this far selects a range to zoom into
const started = new WeakSet();  // the clocks of routes this view opened playing, as connect does, the first time only
const zooms = new WeakMap();  // each clock's ranges zoomed in from, which back steps out to one at a time as on connect
const WIDE = matchMedia("(min-width: 1536px)");  // as connect, the map beside the video from here on
// built once: a formatter per call costs ~7 ms of ICU setup the first time
const HM = new Intl.DateTimeFormat(undefined, {hour: "2-digit", minute: "2-digit", hour12: false});
const HMS = new Intl.DateTimeFormat(undefined, {hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false});
const DAY = new Intl.DateTimeFormat(undefined, {month: "short", day: "numeric"});
const YEAR_DAY = new Intl.DateTimeFormat(undefined, {month: "short", day: "numeric", year: "numeric"});
const WEEKDAY = new Intl.DateTimeFormat(undefined, {weekday: "long"});

// connect's segmented buttons; set(key) shows which is on (null: none)
function segmented(options, pick) {
  const buttons = Object.entries(options).map(([key, label]) => h("button", {class: "cn-option", "data-key": key, "aria-pressed": "false", onclick: () => pick(key)}, label));
  return {el: h("div", {class: "cn-options"}, buttons), set(on) { for (const b of buttons) b.ariaPressed = String(b.dataset.key === on); }};
}

export function create(ctx, state) {
  const {clock, route, perf} = ctx;
  const dongle = route.fullname.split("|")[0], start = route.start_time_utc_millis;
  const offs = [];
  if (!zooms.has(clock)) zooms.set(clock, []);
  const levels = zooms.get(clock);

  // the card's head: back out of a zoom, the times shown, close to the device
  const back = h("button", {class: "cn-icon", title: "Zoom out", "aria-label": "Zoom out",
    onclick: () => zoom(...(levels.pop() ?? [0, clock.duration]), false)}, icon("back", 24));
  const weekday = h("span", {class: "cn-weekday"}), times = h("span");
  const close = h("a", {class: "cn-icon", href: `/${dongle}`, onclick: link(`/${dongle}`), title: "Back to the device", "aria-label": "Close"}, icon("close", 24));

  // the timeline: the strip, then the ruler with what's left of the shown range dimmed after the playhead
  const s = strip(route, clock.duration);
  s.fill().catch(() => {});
  const left = h("div", {class: "cn-left"});
  const head = h("div", {class: "cn-head"});
  const selection = h("div", {class: "cn-selection", hidden: true});
  const bead = h("div", {class: "cn-bead", hidden: true});
  const timeline = h("div", {class: "cn-timeline", role: "slider", tabindex: 0, "aria-label": "Drive timeline", "aria-valuemin": 0,
    "aria-valuemax": Math.floor(clock.duration), oncontextmenu: (e) => e.preventDefault()},
    s.el, h("div", {class: "cn-ruler"}, left), head, selection, bead);

  // the media and the sheets under it: panes, each mounted the first time it's shown
  const boxes = {video: h("div", {class: "cn-box cn-video"}), map: h("div", {class: "cn-box cn-map"}),
    files: h("div", {class: "cn-sheet cn-files"}), info: h("div", {class: "cn-sheet cn-info"})};
  const kinds = {video: ["camera", {camera: "qcamera"}], map: ["map", {}], files: ["files", {}], info: ["info", {}]};
  const panes = {};
  const sized = new ResizeObserver((entries) => {
    for (const e of entries) {
      const pane = e.target._pane;
      pane?.resize?.(e.contentRect.width, e.contentRect.height);
      if (pane) perf.invalidate(pane);
    }
  });
  let disposed = false;
  function mount(key) {
    if (panes[key] !== undefined) return;
    panes[key] = null;
    const [type, paneState] = kinds[key];
    loadPane(type).then((make) => {
      if (disposed) return;
      const pane = panes[key] = make(ctx, paneState);
      pane.name ||= type;
      boxes[key].append(pane.el);
      boxes[key]._pane = pane;
      sized.observe(boxes[key]);
      perf.invalidate(pane);
    });
  }

  let media = state.media === "map" ? "map" : "video", sheet = null;
  const mediaOptions = segmented({video: "Video", map: "Map"}, (key) => { media = key; showMedia(); });
  const sheetOptions = segmented({files: "Files", info: "More info"}, (key) => {
    sheet = sheet === key ? null : key;
    sheetOptions.set(sheet);
    for (const k of ["files", "info"]) boxes[k].hidden = k !== sheet;
    if (sheet) mount(sheet);
  });
  boxes.files.hidden = boxes.info.hidden = true;
  function showMedia() {
    const both = WIDE.matches;
    mediaOptions.el.hidden = both;
    mediaOptions.set(media);
    boxes.video.hidden = !both && media !== "video";
    boxes.map.hidden = !both && media !== "map";
    for (const k of ["video", "map"]) if (!boxes[k].hidden) mount(k);
  }
  WIDE.addEventListener("change", showMedia);
  offs.push(() => WIDE.removeEventListener("change", showMedia));
  showMedia();

  // connect's control pill
  const playBtn = h("button", {class: "cn-play", title: "Play/pause (space)", onclick: () => clock.toggle()});
  const rate = h("select", {class: "cn-rate", title: "Playback rate ([ and ])", onchange: (e) => clock.setRate(+e.target.value)},
    RATES.map((r) => h("option", {value: r}, `${r}×`)));
  const now = h("span", {class: "cn-now"});
  const mute = muteButton("cn-mute", 24);  // the camera pane's sound (its own button is hidden here)
  const pill = h("div", {class: "cn-pill"},
    h("button", {class: "cn-jump", title: "Back 10 s", "aria-label": "Back 10 seconds", onclick: () => clock.seek(clock.t - 10)}, icon("replay10", 30)),
    h("button", {class: "cn-jump", title: "Forward 10 s", "aria-label": "Forward 10 seconds", onclick: () => clock.seek(clock.t + 10)}, icon("forward10", 30)),
    now, h("label", {class: "cn-rate-box"}, icon("up", 16), rate, icon("down", 16)), mute.el, playBtn);

  const el = h("div", {class: "connect-pane"},
    h("div", {class: "cn-card"},
      h("div", {class: "cn-top"}, back, h("div", {class: "cn-title"}, weekday, times), close),
      timeline,
      h("div", {class: "cn-body"},
        h("div", {class: "cn-bar"}, mediaOptions.el, sheetOptions.el),
        h("div", {class: "cn-media"}, boxes.video, boxes.map),
        pill, boxes.files, boxes.info)));

  // zooming in shows and loops just that range, as connect does; the whole drive undoes it. A selected range is a new
  // level to step back out of; the wheel and back move within the one shown
  function zoom(t0, t1, level = true) {
    if (level) levels.push(clock.range);
    clock.setRange(t0, t1);
    [t0, t1] = clock.range;  // as clamped
    const whole = t0 <= 0 && t1 >= clock.duration;
    if (whole) levels.length = 0;
    clock.setLoop(whole ? null : [t0, t1]);
    if (clock.t < t0 || clock.t > t1) clock.seek(t0);
  }

  // a press: a tap seeks, a drag selects a range to zoom into, as on connect
  // the timeline's left edge is read as the pointer arrives, not on every move (a read after a render forces a layout)
  let width = 0, edge = 0, hover = null, press = null;
  const xOf = (e) => Math.max(0, Math.min(width, e.clientX - edge));
  const tOf = (x) => clock.range[0] + x / (width || 1) * (clock.range[1] - clock.range[0]);
  timeline.addEventListener("pointerenter", () => { edge = timeline.getBoundingClientRect().left; });
  timeline.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    edge = timeline.getBoundingClientRect().left;
    timeline.setPointerCapture(e.pointerId);
    const x = xOf(e);
    press = {id: e.pointerId, x0: x, x, mode: "press"};
    perf.invalidate(pane);
  });
  timeline.addEventListener("pointermove", (e) => {
    const x = xOf(e);
    if (e.pointerType === "mouse") hover = x;
    if (press?.id === e.pointerId) {
      press.x = x;
      if (press.mode === "press" && Math.abs(x - press.x0) > MOVE_PX) press.mode = "select";
    }
    perf.invalidate(pane);
  });
  const release = (e) => {
    if (press?.id !== e.pointerId) return;
    const {mode, x0, x} = press;
    press = null;
    if (e.type === "pointerup" && mode === "select" && Math.abs(x - x0) > 3) zoom(tOf(Math.min(x0, x)), tOf(Math.max(x0, x)));
    else if (e.type === "pointerup" && mode !== "select") clock.seek(tOf(x));
    perf.invalidate(pane);
  };
  timeline.addEventListener("pointerup", release);
  timeline.addEventListener("pointercancel", release);
  timeline.addEventListener("pointerleave", () => { hover = null; perf.invalidate(pane); });
  // wheel zooms the shown range around the cursor
  timeline.addEventListener("wheel", (e) => {
    e.preventDefault();
    const [r0, r1] = clock.range, t = tOf(xOf(e)), k = Math.exp(e.deltaY * 0.002);
    zoom(Math.max(0, t - (t - r0) * k), Math.min(clock.duration, t + (r1 - t) * k));
  }, {passive: false});
  timeline.addEventListener("keydown", (e) => {
    const step = e.shiftKey || e.key.startsWith("Page") ? 10 : 1;
    const by = {ArrowLeft: -step, ArrowDown: -step, PageDown: -step, ArrowRight: step, ArrowUp: step, PageUp: step}[e.key];
    if (by === undefined) return;
    e.preventDefault();
    clock.seek(clock.t + by);
  });
  const timelineSize = new ResizeObserver(([e]) => { width = e.contentRect.width; perf.invalidate(pane); });
  timelineSize.observe(timeline);

  const segmentOf = (t) => segmentAt(route, Math.min(Math.floor(t / SEGMENT), Math.ceil(clock.duration / SEGMENT) - 1));
  let shown = "", second = null, playing = null;  // the range the strip and title show, the second the clock shows
  const pane = {name: "connect", el, resize() {}, save: () => ({media}), dispose,
    render() {
      const [r0, r1] = clock.range, span = r1 - r0;
      if (shown !== `${r0} ${r1}`) {
        shown = `${r0} ${r1}`;
        s.show(r0, r1);
        const a = new Date(start + r0 * 1000);
        weekday.textContent = `${WEEKDAY.format(a)} `;
        times.textContent = `${(a.getFullYear() === new Date().getFullYear() ? DAY : YEAR_DAY).format(a)} @ ${HM.format(a)} - ${HM.format(start + r1 * 1000)}`;
        back.disabled = r0 <= 0 && r1 >= clock.duration;
      }
      const x = Math.max(0, Math.min(width, (clock.t - r0) / span * width));
      head.style.transform = left.style.transform = `translateX(${x}px)`;
      selection.hidden = press?.mode !== "select";
      if (!selection.hidden) {
        selection.style.transform = `translateX(${Math.min(press.x0, press.x)}px)`;
        selection.style.width = `${Math.abs(press.x - press.x0)}px`;
      }
      // what's under the cursor, or under the finger while it selects
      const at = press && press.mode !== "press" ? press.x : hover;
      bead.hidden = at === null;
      if (at !== null) {
        const t = tOf(at);
        bead.textContent = `${segmentOf(t)}, ${HMS.format(start + t * 1000)}`;
        bead.style.transform = `translateX(${Math.max(0, Math.min(width - 80, at - 40))}px)`;
      }
      if (second !== Math.trunc(clock.t)) {
        second = Math.trunc(clock.t);
        now.textContent = `${HMS.format(start + second * 1000)} – ${segmentOf(clock.t)}`;
        timeline.setAttribute("aria-valuenow", second);
      }
      if (playing !== clock.playing) playBtn.replaceChildren(icon((playing = clock.playing) ? "pause" : "play", 40));
      mute.show(!!panes.video?.audio);
      if (rate.value !== String(clock.rate)) rate.value = clock.rate;
    }};

  offs.push(...["t", "range", "play"].map((ev) => clock.on(ev, () => perf.invalidate(pane))), sound.on(() => perf.invalidate(pane)));
  if (!started.has(clock)) {
    started.add(clock);
    clock.play();
  }
  function dispose() {
    disposed = true;
    for (const off of offs) off();
    sized.disconnect();
    timelineSize.disconnect();
    for (const p of Object.values(panes)) p?.dispose?.();
  }
  return pane;
}
