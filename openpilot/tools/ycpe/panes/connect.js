// the connect view, user mode's drive: connect's DriveView as connect draws it, built from our panes. A card with the
// drive's times, its timeline (the strip of panes/strip.js and connect's thumbnails over a ruler: tap to seek, drag to
// zoom into a range), Video | Map toggles, Files and More info menus (connect-menus.js), the camera or map pane, and
// connect's control pill. The camera and map panes are mounted here as the layout mounts them. Everything that moves
// while scrubbing or playing is a transform or a changed text, set in one render through the frame scheduler
import {SEGMENT, api, segmentAt, segmentAtTime} from "/api.js";
import {h, css} from "/dom.js";
import {cnIcon} from "/pages/connect/icons.js";
import {loadPane} from "/panes.js";
import {link} from "/app.js";
import {strip} from "./strip.js";
import {sound, muteButton} from "./camera.js";

await css(new URL("./connect.css", import.meta.url));

const MOVE_PX = 6;  // a press that moves this far selects a range to zoom into
const started = new WeakSet();  // the clocks of routes this view opened playing, as connect does, the first time only
const zooms = new WeakMap();  // each clock's ranges zoomed in from, which back steps out to one at a time as on connect
const WIDE = matchMedia("(min-width: 1536px)");  // as connect, the map beside the video from here on
const STEPS = [0.1, 0.25, 0.5, 1, 2, 4, 8];  // connect's play speeds, a step at a time
const THUMB = 32;  // px per timeline thumbnail on connect: its 20 px row at the sprites' 128:80
// built once: a formatter per call costs ~7 ms of ICU setup the first time
const HM = new Intl.DateTimeFormat(undefined, {hour: "2-digit", minute: "2-digit", hour12: false});
const HMS = new Intl.DateTimeFormat(undefined, {hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false});
const DAY = new Intl.DateTimeFormat(undefined, {month: "short", day: "numeric"});
const YEAR_DAY = new Intl.DateTimeFormat(undefined, {month: "short", day: "numeric", year: "numeric"});
const WEEKDAY = new Intl.DateTimeFormat(undefined, {weekday: "long"});

// connect's segmented buttons; set(key) shows which is on (null: none). The label's span is what connect's menus
// open from
function segmented(options, pick) {
  const buttons = Object.entries(options).map(([key, label]) => h("button", {class: "cn-option", "data-key": key, "aria-pressed": "false",
    onclick: (e) => pick(key, e.currentTarget.firstChild)}, h("span", {}, label)));
  return {el: h("div", {class: "cn-options"}, buttons), set(on) { for (const b of buttons) b.ariaPressed = String(b.dataset.key === on); }};
}

export function create(ctx, state) {
  const {clock, route, perf} = ctx;
  const dongle = route.fullname.split("|")[0], start = route.start_time_utc_millis;
  const offs = [];
  if (!zooms.has(clock)) zooms.set(clock, []);
  const levels = zooms.get(clock);

  // the card's head: back out of a zoom, the times shown, close to the device (connect's buttons have no tooltips)
  const back = h("button", {class: "cn-icon", "aria-label": "Go Back", onclick: () => zoom(...(levels.pop() ?? [0, clock.duration]), false)},
    cnIcon("backBold", 24));
  const weekday = h("span", {class: "cn-weekday"}), times = h("span");
  const close = h("a", {class: "cn-icon", href: `/connect/${dongle}`, onclick: (e) => link(close.getAttribute("href"))(e), "aria-label": "Close"},
    cnIcon("close", 24));

  // the timeline: the strip, then the ruler with what's left of the shown range dimmed after the playhead
  const s = strip(route, clock.duration);
  s.fill().catch(() => {});
  const left = h("div", {class: "cn-left"});
  const ruler = h("div", {class: "cn-ruler"}, left);
  const selection = h("div", {class: "cn-selection", hidden: true});
  const bead = h("div", {class: "cn-bead", hidden: true});
  const thumbs = h("div", {class: "cn-thumbs"});  // connect's tiles, a 5 s frame of the sprite each
  const timeline = h("div", {class: "cn-timeline", role: "slider", tabindex: 0, "aria-label": "Drive timeline", "aria-valuemin": 0,
    "aria-valuemax": Math.floor(clock.duration), oncontextmenu: (e) => e.preventDefault()},
    s.el, thumbs, ruler, selection, bead);

  // the media: panes, each mounted the first time it's shown
  const boxes = {video: h("div", {class: "cn-box cn-video"}), map: h("div", {class: "cn-box cn-map"})};
  const kinds = {video: ["camera", {camera: "qcamera"}], map: ["map", {}]};
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

  let media = state.media === "map" ? "map" : "video";
  const mediaOptions = segmented({video: "Video", map: "Map"}, (key) => { media = key; showMedia(); });
  // connect's menus, over the button
  const menuOptions = segmented({files: "Files", info: "More info"}, (key, label) =>
    import("./connect-menus.js").then((m) => !disposed && m[key === "files" ? "filesMenu" : "infoMenu"](label, ctx)));
  // connect's Clip before them, once the device says it cuts clips (loaded after the view)
  import("./clips.js").then(({clipButton}) => disposed || menuOptions.el.prepend(clipButton(ctx)));
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

  // connect's control pill, its speeds a step at a time
  const playBtn = h("button", {class: "cn-play", onclick: () => clock.toggle()});
  const step = (by) => {
    const i = STEPS.indexOf(clock.rate);
    clock.setRate(STEPS[Math.max(0, Math.min(STEPS.length - 1, (i < 0 ? STEPS.indexOf(1) : i) + by))]);
    clock.play();  // as connect: a step plays
  };
  const faster = h("button", {class: "cn-step", "aria-label": "Increase play speed by 1 step", onclick: () => step(1)}, cnIcon("up", 12));
  const slower = h("button", {class: "cn-step", "aria-label": "Decrease play speed by 1 step", onclick: () => step(-1)}, cnIcon("down", 12));
  const speedText = h("span", {class: "cn-speed-text"});
  const speed = h("div", {class: "cn-speed"}, faster, speedText, slower);
  const now = h("span", {class: "cn-now"});
  const mute = muteButton("cn-mute", 24);  // the camera pane's sound (its own button is hidden here)
  const jumpBack = h("button", {class: "cn-jump", "aria-label": "Jump back 10 seconds", onclick: () => clock.seek(clock.t - 10)}, cnIcon("replay10", 32));
  const jumpOn = h("button", {class: "cn-jump", "aria-label": "Jump forward 10 seconds", onclick: () => clock.seek(clock.t + 10)}, cnIcon("forward10", 32));
  const pill = h("div", {class: "cn-controls"}, jumpBack, jumpOn, now, speed, mute.el, playBtn);

  const el = h("div", {class: "connect-pane"},
    h("div", {class: "cn-card"},
      h("div", {class: "cn-top"}, back, h("div", {class: "cn-when"}, weekday, times), close),
      timeline,
      h("div", {class: "cn-body"},
        h("div", {class: "cn-bar"}, mediaOptions.el, menuOptions.el),
        h("div", {class: "cn-media"}, boxes.video, boxes.map),
        pill)));

  // zooming in shows and loops just that range, as connect does; the whole drive undoes it. A selected range is a new
  // level for back to step out of, one at a time
  function zoom(t0, t1, level = true) {
    if (level) levels.push(clock.range);
    clock.setRange(t0, t1);
    [t0, t1] = clock.range;  // as clamped
    const whole = t0 <= 0 && t1 >= clock.duration;
    if (whole) levels.length = 0;
    clock.setLoop(whole ? null : [t0, t1]);
    if (clock.t < t0 || clock.t > t1) clock.seek(t0);
  }

  // a press: a tap seeks, a drag selects a range to zoom into, as on connect (whose ruler alone takes them)
  // the timeline's left edge is read as the pointer arrives, not on every move (a read after a render forces a layout)
  let width = 0, edge = 0, hover = null, press = null;
  const xOf = (e) => Math.max(0, Math.min(width, e.clientX - edge));
  const tOf = (x) => clock.range[0] + x / (width || 1) * (clock.range[1] - clock.range[0]);
  const off = (e) => !ruler.contains(e.target);
  timeline.addEventListener("pointerenter", () => { edge = timeline.getBoundingClientRect().left; });
  timeline.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || off(e)) return;
    edge = timeline.getBoundingClientRect().left;
    timeline.setPointerCapture(e.pointerId);
    const x = xOf(e);
    press = {id: e.pointerId, x0: x, x, mode: "press"};
    perf.invalidate(pane);
  });
  timeline.addEventListener("pointermove", (e) => {
    const x = xOf(e);
    if (e.pointerType === "mouse") hover = press?.id !== e.pointerId && off(e) ? null : x;
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
  timeline.addEventListener("keydown", (e) => {
    const step = e.shiftKey || e.key.startsWith("Page") ? 10 : 1;
    const by = {ArrowLeft: -step, ArrowDown: -step, PageDown: -step, ArrowRight: step, ArrowUp: step, PageUp: step}[e.key];
    if (by === undefined) return;
    e.preventDefault();
    clock.seek(clock.t + by);
  });
  const timelineSize = new ResizeObserver(([e]) => { width = e.contentRect.width; perf.invalidate(pane); });
  timelineSize.observe(timeline);

  // connect's thumbnails for [r0, r1]: a 32 px tile per slot, each the 5 s frame of its segment's sprite at the
  // slot's middle (connect counts the frame from the drive's start, so past the first minute it's the last one),
  // runs of following frames of one segment in one element
  function drawThumbs(r0, r1) {
    const count = Math.ceil(width / THUMB), runs = [];
    if (!route.url || !Number.isFinite(count)) return thumbs.replaceChildren();
    const slots = Math.max(1, Math.ceil(clock.duration / SEGMENT));
    for (let i = 0; i < count; i++) {
      const t = r0 + (i + 0.5) / count * (r1 - r0), image = Math.max(0, Math.min(Math.floor(Math.floor(t) / 5), 11));
      const n = segmentAt(route, Math.max(0, Math.min(slots - 1, Math.floor(t / SEGMENT))));
      const run = runs.at(-1);
      if (run && run.n === n && image === run.end + 1) { run.end = image; run.length++; } else runs.push({n, start: image, end: image, length: 1});
    }
    thumbs.replaceChildren(...runs.map((r) => h("div", {style: `width:${THUMB * r.length}px;background-image:url(${api.segmentAsset(route, r.n, "sprite.jpg")});background-position-x:-${r.start * THUMB}px`})));
  }

  const segmentOf = (t) => segmentAtTime(route, t, clock.duration);
  let shown = "", second = null, playing = null, rated = null;  // the range the strip and title show, the second the clock shows
  const pane = {name: "connect", el, resize() {}, save: () => ({media}), dispose,
    render() {
      const [r0, r1] = clock.range, span = r1 - r0;
      if (shown !== `${r0} ${r1} ${width}`) {
        shown = `${r0} ${r1} ${width}`;
        s.show(r0, r1);
        drawThumbs(r0, r1);
        const a = new Date(start + r0 * 1000);
        weekday.textContent = `${WEEKDAY.format(a)} `;
        times.textContent = `${(a.getFullYear() === new Date().getFullYear() ? DAY : YEAR_DAY).format(a)} @ ${HM.format(a)} - ${HM.format(start + r1 * 1000)}`;
        back.disabled = r0 <= 0 && r1 >= clock.duration;
      }
      const x = Math.max(0, Math.min(width, (clock.t - r0) / span * width));
      left.style.transform = `translateX(${x}px)`;
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
        bead.style.transform = `translateX(${Math.max(-10, Math.min(width - 70, at - 40))}px)`;
      }
      if (second !== Math.trunc(clock.t)) {
        second = Math.trunc(clock.t);
        now.textContent = `${HMS.format(start + second * 1000)} – ${segmentOf(clock.t)}`;
        timeline.setAttribute("aria-valuenow", second);
      }
      if (playing !== clock.playing) {
        playing = clock.playing;
        playBtn.replaceChildren(cnIcon(playing ? "pause" : "play", 47));
        playBtn.ariaLabel = playing ? "Pause" : "Unpause";
      }
      mute.show(!!panes.video?.audio);
      if (rated !== clock.rate) {
        rated = clock.rate;
        const i = STEPS.indexOf(rated) < 0 ? STEPS.indexOf(1) : STEPS.indexOf(rated);
        speedText.textContent = `${rated}×`;
        faster.disabled = i >= STEPS.length - 1;
        slower.disabled = i <= 0;
      }
    }};

  offs.push(...["t", "range", "play"].map((ev) => clock.on(ev, () => perf.invalidate(pane))), sound.on(() => perf.invalidate(pane)));
  // its menus, and what More info shows of the drive, once the view is up (connect asks as the drive opens)
  requestIdleCallback(() => import("./connect-menus.js").then((m) => disposed || m.prefetch(ctx)), {timeout: 2000});
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
