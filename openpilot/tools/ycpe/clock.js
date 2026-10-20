// the shared clock: playhead t and visible x-range, in route seconds. Playback advances on the frame tick.
// A live clock's duration is the head of the data, which glides on between arrivals (setEdge). While it follows, it
// plays at the head and the x-range keeps its width ending there; a seek or a range into the past stops following,
// and a seek to the end, goLive() or catching up while playing starts it again.
import {listeners} from "/dom.js";
import {perf} from "/perf.js";

export const RATES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5];
const LIVE_RANGE = 30;  // s a live clock shows to begin with
const SETTLE = 0.5;  // s the live head takes to close most of a gap to where the data says it is
const SNAP = 2;  // s of data time beyond which the head jumps instead

export function createClock(duration, {live = false} = {}) {
  const subs = {t: listeners(), range: listeners(), play: listeners(), duration: listeners()};
  const emit = (ev) => subs[ev].emit(clock);
  const clamp = (t) => Math.max(clock.start, Math.min(clock.duration, t));
  let stopPlay = null, stopHead = null;
  // the newest data: its time, when it came (ms), and smoothed, the data seconds per second and the ms between arrivals
  const edge = {t: 0, at: 0, rate: 1, gap: 0};

  const clock = {
    t: 0, playing: live, rate: 1, loop: null, duration, range: live ? [-LIVE_RANGE, 0] : [0, duration],
    start: 0,  // the earliest time there is: a live session keeps a window
    live, follow: live,
    on: (ev, fn) => subs[ev].on(fn),  // "t" | "range" | "play" (also following or not) | "duration"
    seek(t) {
      if (clock.live && t >= clock.duration) return clock.goLive();
      follow(false);
      move(t);
    },
    setRange(t0, t1) {
      if (clock.follow && t1 < clock.duration - 1e-3 * (t1 - t0)) follow(false);
      if (clock.follow) [t0, t1] = [clock.duration - (t1 - t0), clock.duration];  // a zoom at the head keeps following
      span(t0, t1);
    },
    setDuration(d) {
      const full = clock.range[0] === clock.start && clock.range[1] === clock.duration;
      clock.duration = d;
      if (full) span(clock.start, d);
      emit("duration");
    },
    // a live session's oldest time, as its window slides
    setStart(s) {
      clock.start = s;
      if (clock.t < s) move(s);
      if (clock.range[1] < s) span(s, s + clock.range[1] - clock.range[0]);
    },
    // a live session's newest data time, as each arrives
    setEdge(t) {
      const now = performance.now();
      if (edge.at && t > edge.t) {
        const dt = now - edge.at;
        edge.rate += ((t - edge.t) / (dt / 1000) - edge.rate) * 0.2;
        edge.gap += (dt - edge.gap) * 0.2;
      }
      [edge.t, edge.at] = [t, now];
    },
    goLive() {
      follow(true);
      clock.play();
      move(clock.duration);
      span(clock.duration - (clock.range[1] - clock.range[0]), clock.duration);
    },
    play() {
      if (clock.playing) return;
      if (clock.t >= clock.duration && !clock.live) clock.seek(clock.loop ? clock.loop[0] : clock.start);
      clock.playing = true;
      if (!clock.live) stopPlay = perf.tick(advance);
      emit("play");
    },
    pause() {
      if (!clock.playing) return;
      clock.playing = clock.follow = false;
      stopPlay?.();
      emit("play");
    },
    toggle() { clock.playing ? clock.pause() : clock.play(); },
    setRate(r) { clock.rate = r; emit("play"); },
    setLoop(loop) { clock.loop = loop; emit("play"); },
    dispose() { stopPlay?.(); stopHead?.(); },
  };

  function follow(on) {
    if (clock.follow === on || !clock.live) return;
    clock.follow = on;
    emit("play");
  }

  function move(t) {
    t = clamp(t);
    if (t === clock.t) return;
    clock.t = t;
    emit("t");
  }

  // a live range may reach back past the data, as the data starts out shorter than it
  function span(t0, t1) {
    const w = Math.max(0.01, clock.live ? t1 - t0 : Math.min(clock.duration, t1 - t0));
    t0 = clock.live ? Math.min(clock.duration - w, t0) : Math.max(0, Math.min(clock.duration - w, t0));
    if (t0 === clock.range[0] && t0 + w === clock.range[1]) return;
    clock.range = [t0, t0 + w];
    emit("range");
  }

  // playback at rate; a live one follows once it catches up with the head
  function advance(f) {
    let t = clock.t + Math.min(f.interval, 100) / 1000 * clock.rate;
    if (clock.loop && t >= clock.loop[1]) t = clock.loop[0];
    if (t >= clock.duration && clock.live) return clock.goLive();
    if (t >= clock.duration) { t = clock.duration; clock.pause(); }
    move(t);
  }

  // live: the head moves at the data's rate toward one arrival behind where the newest data extrapolates to (never past
  // the data), so it glides between arrivals
  function head(f) {
    if (!edge.at) return;
    const dt = Math.min(f.interval, 100) / 1000;
    const target = Math.min(edge.t, edge.t + ((performance.now() - edge.at) - edge.gap) / 1000 * edge.rate);
    let h = clock.duration + dt * edge.rate + (target - clock.duration) * Math.min(1, dt / SETTLE);
    if (Math.abs(target - h) > SNAP * Math.max(1, edge.rate)) h = target;
    h = Math.max(clock.duration, Math.min(edge.t, h));
    if (h !== clock.duration) {
      const w = clock.range[1] - clock.range[0];
      clock.duration = h;
      if (clock.follow) {
        move(h);
        span(h - w, h);
      }
      emit("duration");
    }
    if (clock.playing && !clock.follow) advance(f);
  }
  if (live) stopHead = perf.tick(head);
  return clock;
}
