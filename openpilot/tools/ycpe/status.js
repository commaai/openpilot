// what's loaded where. Every loader (logs, each camera, CAN) publishes the state of each segment, whether the
// playhead is waiting on it, and the bytes it holds; the playbar, the load pill and the HUD draw from here.
//
//   state      missing  no such file          queued   requested
//              idle     available, not asked  loading  downloading or decoding (with loaded/total bytes)
//              disk     cached on disk only   ready    in memory
//              error    failed
// A segment may also say which source its bytes came from (api, internal, openpilotci, car segments, local, …).
import {h} from "/dom.js";
import {icon} from "/icons.js";
import {daemon} from "/daemon.js";
import {perf} from "/perf.js";
import {theme} from "/theme.js";

export const STATES = ["missing", "idle", "queued", "loading", "disk", "ready", "error"];

const kinds = new Map();  // name → source
const notes = new Map();  // key → text: standing facts about what's loaded (the log's schema), shown in the pill
const subs = new Set();

// listeners run at most once a frame, however many segments change
const notify = {name: "status", render() { for (const fn of subs) fn(); }};
const changed = () => perf.invalidate(notify);

export const status = {
  // a loader's handle; order sorts the playbar lanes. Loaders of the same data (two panes on one camera) share a
  // source, which goes away when the last of them removes it
  source(name, {label = name, order = 10} = {}) {
    let src = kinds.get(name);
    if (!src) kinds.set(name, src = {name, label, order, segs: [], buffering: null, memory: 0, disk: 0, rate: 0, refs: 0});
    src.refs++;
    changed();
    let removed = false;
    return {
      // text: a note on the state, e.g. "retrying (2/6)"
      seg(n, state, {loaded = 0, total = 0, source = null, text = null} = {}) {
        const s = src.segs[n] ||= {state: "idle", loaded: 0, total: 0, source: null, text: null};
        if (s.state === state && s.loaded === loaded && s.total === total && s.source === source && s.text === text) return;
        s.state = state;
        s.loaded = loaded;
        s.total = total;
        s.source = source;
        s.text = text;
        changed();
      },
      count(n) {
        if (src.segs.length === n) return;
        for (let i = 0; i < n; i++) src.segs[i] ||= {state: "idle", loaded: 0, total: 0};
        src.segs.length = n;
        changed();
      },
      // the playhead is waiting: {n, text} (e.g. "segment 12"), or null once it has what it needs
      buffering(b) {
        if (src.buffering?.text === b?.text) return;
        src.buffering = b;
        changed();
      },
      // fn() loads the failed segments again (the panel's Retry)
      retry(fn) { src.retry = fn; },
      bytes({memory = src.memory, disk = src.disk}) { src.memory = memory; src.disk = disk; changed(); },
      rate(bytesPerSec) { src.rate = bytesPerSec; changed(); },
      remove() {
        if (removed) return;
        removed = true;
        if (--src.refs === 0) kinds.delete(name);
        changed();
      },
    };
  },
  kinds: () => [...kinds.values()].sort((a, b) => a.order - b.order),
  // a standing note under key; empty text removes it
  note(key, text) {
    if ((notes.get(key) || "") === (text || "")) return;
    if (text) notes.set(key, text); else notes.delete(key);
    changed();
  },
  on(fn) { subs.add(fn); return () => subs.delete(fn); },
};

// fraction of a segment that's usable or on its way, for drawing
export const fill = (s) => s.state === "ready" ? 1 : s.state === "loading" ? (s.total ? s.loaded / s.total : 0.05) : s.state === "disk" ? 1 : 0;

export const fmtBytes = (b) => b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(0)} MB` : `${Math.round(b / 1e3)} kB`;

const COLORS = {ready: "--accent", disk: "--accent", loading: "--accent", error: "--critical"};

// one kind's segments as a strip of cells: solid when in memory, faint when only on disk, filling while loading
export function drawLane(c, src, x0, y, w, hgt, segX = null) {
  const col = (v) => theme.color(v);
  const n = src.segs.length;
  src.segs.forEach((s, i) => {
    const [a, b] = segX ? segX(i) : [x0 + i * w / n, x0 + (i + 1) * w / n];
    const cw = Math.max(1, b - a - (b - a > 4 ? 1 : 0));
    if (s.state === "missing") return;
    c.globalAlpha = s.state === "idle" ? 0.35 : 1;
    c.fillStyle = col("--line");
    c.fillRect(a, y, cw, hgt);
    const f = fill(s);
    if (!f && s.state !== "error") return;
    c.globalAlpha = s.state === "disk" ? 0.45 : 1;
    c.fillStyle = col(COLORS[s.state]);
    c.fillRect(a, y, s.state === "error" ? cw : cw * f, hgt);
  });
  c.globalAlpha = 1;
}

function counts(src) {
  const n = {ready: 0, disk: 0, loading: 0, queued: 0, error: 0, total: 0};
  for (const s of src.segs) {
    if (s.state === "missing") continue;
    n.total++;
    if (s.state in n) n[s.state]++;
  }
  return n;
}

const STATE_TEXT = {missing: "not available", idle: "not loaded", queued: "queued", loading: "loading", disk: "on disk", ready: "in memory", error: "failed"};
export const describe = (s) => (s.state === "loading" && s.total ? `${STATE_TEXT.loading} ${Math.round(100 * s.loaded / s.total)}% of ${fmtBytes(s.total)}` : STATE_TEXT[s.state])
  + (s.text ? ` · ${s.text}` : "") + (s.source ? ` · from ${s.source}` : "");

// the top-bar pill: what's loading or buffering right now, and a panel with every kind's segments
export function loadPill() {
  const text = h("span");
  const bar = h("span", {class: "pill-bar"}, h("i"));
  const el = h("button", {class: "load-pill", hidden: true, "aria-expanded": "false"}, text, bar);
  const live = document.body.appendChild(h("span", {class: "sr-only", role: "status"}));
  let panel = null, spoken = "";
  const update = () => {
    const all = status.kinds();
    const extra = [...notes.values()];
    el.hidden = !all.length && !extra.length;
    const waiting = all.find((k) => k.buffering);
    const active = all.filter((k) => k.segs.some((s) => s.state === "loading" || s.state === "queued"));
    el.classList.toggle("buffering", !!waiting);
    el.classList.toggle("done", !waiting && !active.length);
    const each = all.map((k) => { const n = counts(k); return {n, text: `${k.label} ${n.ready + n.disk}/${n.total}${n.error ? ` · ${n.error} failed` : ""}`}; });
    const bad = each.some((e) => e.n.error) && !waiting;
    el.style.color = bad ? "var(--critical-fg)" : "";
    el.style.borderColor = bad ? "var(--critical)" : "";
    el.title = `${[...each.map((e) => e.text), ...extra].join(" · ")} (click for details)`;
    if (waiting) text.textContent = `buffering ${waiting.label} · ${waiting.buffering.text}`;
    else if (active.length) {
      const k = active[0], n = counts(k);
      text.textContent = `${k.label} ${n.ready + n.disk}/${n.total}${n.error ? ` · ${n.error} failed` : ""}${k.rate ? ` · ${fmtBytes(k.rate)}/s` : ""}${active.length > 1 ? ` · +${active.length - 1}` : ""}`;
    } else {  // idle: only what isn't all there (a camera loads near the playhead); the tooltip has everything
      const missing = [...each.filter(({n}) => n.ready + n.disk < n.total).map((e) => e.text), ...extra].join(" · ");
      if (missing) text.textContent = missing;
      else if (!text.querySelector("svg")) text.replaceChildren(icon("check", 14));
    }
    el.classList.toggle("loaded", !text.textContent);
    // read out: the playhead waiting, failures, and "ready" once neither is the case anymore
    const failed = each.reduce((a, e) => a + e.n.error, 0);
    const say = waiting ? `buffering ${waiting.label}, ${waiting.buffering.text}` : failed ? `${failed} segment${failed > 1 ? "s" : ""} failed to load`
      : spoken && "ready";
    if (say !== spoken) live.textContent = spoken = say;
    const segs = active.flatMap((k) => k.segs.filter((s) => s.state !== "missing" && s.state !== "idle"));
    const f = segs.length ? segs.reduce((a, s) => a + fill(s), 0) / segs.length : 1;
    bar.hidden = !active.length && !waiting;
    bar.firstChild.style.width = `${(100 * f).toFixed(1)}%`;
    panel?.update();
  };
  status.on(update);
  el.addEventListener("click", () => {
    if (panel) return panel.close();
    panel = statusPanel(el, () => { panel = null; el.setAttribute("aria-expanded", "false"); });
    el.setAttribute("aria-expanded", "true");
  });
  return el;
}

function statusPanel(anchor, onClose) {
  const body = h("div", {class: "status-rows"});
  const closeBtn = h("button", {class: "link", onclick: () => close()}, "close");
  const el = h("div", {class: "status-panel", role: "dialog", "aria-label": "Loading"}, h("div", {class: "status-head"}, h("b", {}, "Loading"), closeBtn), body);
  const r = anchor.getBoundingClientRect();
  el.style.top = `${r.bottom + 6}px`;
  el.style.width = "min(420px, calc(100vw - 16px))";  // under the anchor, but on screen
  document.body.append(el);
  el.style.right = `${Math.max(8, Math.min(innerWidth - r.right, innerWidth - el.offsetWidth - 8))}px`;
  const rows = new Map();
  function row(k) {
    let rw = rows.get(k.name);
    if (!rw) {
      const canvas = h("canvas", {class: "status-lane"});
      const info = h("span", {class: "muted"});
      const tip = h("div", {class: "status-tip muted"});
      const errors = h("div", {class: "status-tip status-errors"});
      const retry = h("button", {class: "link", hidden: true, onclick: () => k.retry?.()}, "Retry");
      rw = {canvas, info, tip, errors, retry, el: h("div", {class: "status-row"}, h("div", {class: "status-title"}, h("b", {}, k.label), info, " ", retry), canvas, tip, errors)};
      canvas.addEventListener("pointermove", (e) => {
        const i = Math.floor((e.offsetX / canvas.clientWidth) * k.segs.length), s = k.segs[i];
        rw.tip.textContent = s ? `segment ${i}: ${describe(s)}` : "";
      });
      canvas.addEventListener("pointerleave", () => { rw.tip.textContent = ""; });
      rows.set(k.name, rw);
    }
    return rw;
  }
  function update() {
    const kinds = status.kinds();
    body.replaceChildren(...kinds.map((k) => row(k).el));
    for (const k of kinds) {
      const {canvas, info, errors, retry} = row(k), n = counts(k);
      retry.hidden = !n.error || !k.retry;
      const bits = [`${n.ready} in memory`, n.disk && `${n.disk} on disk`, n.loading && `${n.loading} loading`, n.queued && `${n.queued} queued`,
        n.error && `${n.error} failed`, k.rate && `${fmtBytes(k.rate)}/s`, k.memory && `mem ${fmtBytes(k.memory)}`, k.disk && `disk ${fmtBytes(k.disk)}`,
        k.buffering && `waiting on ${k.buffering.text}`];
      info.textContent = ` ${n.total} segments · ${bits.filter(Boolean).join(" · ")}`;
      const failed = k.segs.flatMap((s, i) => s.state === "error" ? [`segment ${i}: ${s.text || "failed"}`] : []).join("\n");
      if (errors.textContent !== failed) errors.textContent = failed;
      const dpr = devicePixelRatio, w = canvas.clientWidth, hgt = 10;
      if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = hgt * dpr; }
      const c = canvas.getContext("2d");
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      c.clearRect(0, 0, w, hgt);
      drawLane(c, k, 0, 0, w, hgt);
    }
  }
  const escape = (e) => { if (e.key === "Escape") { close(); anchor.focus(); } };
  const close = () => {
    el.remove();
    document.removeEventListener("keydown", escape);
    onClose();
  };
  setTimeout(() => document.addEventListener("pointerdown", function off(e) {
    if (!el.contains(e.target) && !anchor.contains(e.target)) { document.removeEventListener("pointerdown", off); if (el.isConnected) close(); }
  }));
  document.addEventListener("keydown", escape);
  closeBtn.focus();
  update();
  return {update, close};
}

// memory and disk in the HUD: what the loaders hold plus the JS heap, and the origin's storage (OPFS caches)
export function footprint() {
  const el = h("span", {class: "hud-mem", title: "memory: loaded data + JS heap · disk: cached logs and video"});
  let disk = 0;
  // with the local daemon, its shared download cache counts too (that's where logs live then)
  const local = () => daemon.has("cache") ? daemon.json("/local/cache").then((s) => s.bytes, () => 0) : 0;
  const poll = () => Promise.all([navigator.storage?.estimate().then((e) => e.usage || 0, () => 0), local()]).then(([a, b]) => {
    disk = a + b;
    perf.mark("disk: browser (OPFS)", fmtBytes(a));
    if (daemon.has("cache")) perf.mark("disk: daemon cache", fmtBytes(b));
  });
  poll();
  setInterval(poll, 3000);
  setInterval(() => {
    const data = status.kinds().reduce((a, k) => a + k.memory, 0), heap = performance.memory?.usedJSHeapSize || 0;
    const text = `${fmtBytes(data + heap)} · ${fmtBytes(disk)} disk`;
    if (el.textContent !== text) el.textContent = text;
  }, 500);
  return el;
}
