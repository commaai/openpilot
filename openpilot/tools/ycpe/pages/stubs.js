// stand-ins for clock.js and perf.js until they land: a clock that follows (and seeks) a video, and a bare rAF scheduler

export function videoClock(video, duration) {
  const handlers = {t: new Set(), range: new Set(), play: new Set()};
  const emit = (event) => { for (const fn of handlers[event]) fn(clock); };
  const clock = {
    t: 0, playing: false, rate: 1, loop: null, range: [0, duration],
    seek(t) { clock.t = t; video.currentTime = t; emit("t"); },
    play: () => video.play(),
    pause: () => video.pause(),
    setRange(t0, t1) { clock.range = [t0, t1]; emit("range"); },
    on(event, fn) { handlers[event].add(fn); return () => handlers[event].delete(fn); },
  };
  const follow = () => {
    clock.t = video.currentTime;
    emit("t");
    if (!video.paused) video.requestVideoFrameCallback(follow);
  };
  video.addEventListener("play", () => { clock.playing = true; emit("play"); video.requestVideoFrameCallback(follow); });
  video.addEventListener("pause", () => { clock.playing = false; emit("play"); });
  video.addEventListener("seeked", follow);
  return clock;
}

const queued = new Set();
export const perf = {
  invalidate(pane) {
    if (!queued.size) requestAnimationFrame(() => {
      const panes = [...queued];
      queued.clear();
      for (const p of panes) p.render();
    });
    queued.add(pane);
  },
};

// stands in for the layout engine: panes learn their size from a ResizeObserver
export function mount(pane) {
  new ResizeObserver(([e]) => { pane.resize(e.contentRect.width, e.contentRect.height); perf.invalidate(pane); }).observe(pane.el);
  return pane.el;
}
