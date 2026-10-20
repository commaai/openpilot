// the shared clock: playhead t and visible x-range, in route seconds. Playback advances on the frame tick.
import {listeners} from "/dom.js";
import {perf} from "/perf.js";

export const RATES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5];

export function createClock(duration) {
  const subs = {t: listeners(), range: listeners(), play: listeners()};
  const emit = (ev) => subs[ev].emit(clock);
  const clamp = (t) => Math.max(0, Math.min(clock.duration, t));
  let stopTick = null;

  const clock = {
    t: 0, playing: false, rate: 1, loop: null, duration, range: [0, duration],
    on: (ev, fn) => subs[ev].on(fn),
    seek(t) {
      t = clamp(t);
      if (t === clock.t) return;
      clock.t = t;
      emit("t");
    },
    setRange(t0, t1) {
      const span = Math.max(0.01, Math.min(clock.duration, t1 - t0));
      t0 = Math.max(0, Math.min(clock.duration - span, t0));
      if (t0 === clock.range[0] && t0 + span === clock.range[1]) return;
      clock.range = [t0, t0 + span];
      emit("range");
    },
    setDuration(d) {
      const full = clock.range[0] === 0 && clock.range[1] === clock.duration;
      clock.duration = d;
      if (full) clock.setRange(0, d);
    },
    play() {
      if (clock.playing) return;
      if (clock.t >= clock.duration) clock.seek(clock.loop ? clock.loop[0] : 0);
      clock.playing = true;
      stopTick = perf.tick((f) => {
        let t = clock.t + Math.min(f.interval, 100) / 1000 * clock.rate;
        if (clock.loop && t >= clock.loop[1]) t = clock.loop[0];
        if (t >= clock.duration) { t = clock.duration; clock.pause(); }
        clock.seek(t);
      });
      emit("play");
    },
    pause() {
      if (!clock.playing) return;
      clock.playing = false;
      stopTick?.();
      emit("play");
    },
    toggle() { clock.playing ? clock.pause() : clock.play(); },
    setRate(r) { clock.rate = r; emit("play"); },
    setLoop(loop) { clock.loop = loop; emit("play"); },
  };
  return clock;
}
