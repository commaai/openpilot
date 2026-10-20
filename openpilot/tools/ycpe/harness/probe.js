// injected before any page script: rAF frame trace, long animation frames and layout shifts, read back by run.py
(() => {
  const probe = window.__probe = {raf: [], loaf: [], shifts: []};

  const tick = (t) => { probe.raf.push(t); requestAnimationFrame(tick); };
  requestAnimationFrame(tick);

  const observe = (type, fn) => {
    try { new PerformanceObserver((list) => list.getEntries().forEach(fn)).observe({type, buffered: true}); } catch {}
  };
  observe("long-animation-frame", (e) => probe.loaf.push({
    start: e.startTime, duration: e.duration, blocking: e.blockingDuration,
    scripts: e.scripts.map((s) => ({invoker: s.invoker, url: s.sourceURL.replace(location.origin, ""), pos: s.sourceCharPosition, fn: s.sourceFunctionName, duration: s.duration})),
  }));
  observe("layout-shift", (e) => {
    if (e.hadRecentInput) return;
    probe.shifts.push({t: e.startTime, value: e.value, sources: e.sources.map((s) => ({node: s.node, rect: s.currentRect}))});
  });
})();
