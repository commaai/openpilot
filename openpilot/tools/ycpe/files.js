// a route's files, kept where its timeline is. routeFiles is the one model a workspace has of them: each segment's
// files of every kind from the chosen source (data/sources.js) with their sizes, what's uploaded to comma's servers,
// and the uploads asked of the device (pages/files.js). The playbar's Files panel draws it, and so does user mode's
// Files menu (panes/connect-menus.js).
import {api, segmentAtTime} from "/api.js";
import {deviceRecord} from "/devices.js";
import {h, css, listeners, popover, fitCanvas} from "/dom.js";
import {icon} from "/icons.js";
import {perf} from "/perf.js";
import {theme} from "/theme.js";
import {fmtBytes} from "/status.js";
import {KINDS, API_KIND, SOURCES, AUTO, daemon, parseRange, resolveFiles, routeSource} from "/data/sources.js";
import {uploads, uploadState, uploadNotice, queueTable, segmentsOf, segmentsIn, urlKey, kindName, UPLOADABLE} from "/pages/files.js";

await css(new URL("./files.css", import.meta.url));

const COLS = Object.keys(KINDS);
const BATCH = 60;  // segments resolved per ask
const label = (source) => SOURCES[source]?.label ?? source;
const keysOf = (ns, kinds) => ns.flatMap((n) => kinds.map((kind) => `${n}/${kind}`));

export function routeFiles(route) {
  const sr = parseRange(route.fullname), segs = segmentsOf(route);
  const subs = listeners();
  // listeners run at most once a frame, however many answers come in
  const notify = {name: "files", render() { subs.emit(); }};
  const changed = () => perf.invalidate(notify);
  const stop = new AbortController();  // download-alls end with the workspace
  const asking = new Set();  // segments being resolved
  let gen = 0;  // drops answers for a source no longer picked
  const fileName = (n, f) => `${route.fullname.replace("|", "_")}--${n}--${f.file}`;  // as saved
  const m = {
    segs, source: routeSource.get(route.fullname),
    found: new Map(),     // "<n>/<kind>" → {url, file, size, source, cached}
    errors: {},           // source → why it failed
    asked: new Set(),     // segments resolved
    uploaded: new Map(),  // "<n>/<kind>" → its url on comma's servers
    saving: {},           // kind → "done/total" while a download-all runs
    up: null, device: null,  // once the device is fetched (ready)
    on: subs.on, pick, resolve, refresh, downloadAll, fileName,
    dispose() { gen++; stop.abort(); m.up?.dispose(); },
  };

  function pick(name) {
    m.source = name;
    routeSource.set(route.fullname, name);
    gen++;
    m.found.clear();
    m.errors = {};
    m.asked.clear();
    asking.clear();
    changed();
  }

  // the files of these segments not resolved yet, a batch at a time
  async function resolve(ns = segs) {
    ns = ns.filter((n) => !m.asked.has(n) && !asking.has(n));
    for (const n of ns) asking.add(n);
    const g = gen;
    for (let i = 0; i < ns.length; i += BATCH) {
      const batch = ns.slice(i, i + BATCH);
      const r = await resolveFiles(sr, batch, COLS, routeSource.list(m.source), {sizes: true})
        .catch((e) => ({found: new Map(), errors: {[m.source]: e.message}}));
      if (g !== gen) return;
      for (const [k, f] of r.found) m.found.set(k, f);
      for (const n of batch) { asking.delete(n); m.asked.add(n); }
      Object.assign(m.errors, r.errors);
      changed();
    }
  }

  // what's uploaded, which decides what can be requested whichever source is shown; fresh: asked again
  async function refresh(fresh = true) {
    if (fresh) api.routeFiles.forget?.(route.fullname);
    const files = await api.routeFiles(route.fullname).catch(() => ({}));
    m.uploaded.clear();
    for (const kind of COLS) for (const u of files[API_KIND[kind]] || []) m.uploaded.set(urlKey(u, kind), u);
    changed();
  }

  // every file of a kind in these segments, one after another; cross-origin files go through a blob so the download
  // attribute holds
  async function downloadAll(kind, ns) {
    await resolve(ns);
    const files = ns.map((n) => [n, m.found.get(`${n}/${kind}`)]).filter(([, f]) => f);
    for (const [i, [n, f]] of files.entries()) {
      if (stop.signal.aborted) return;
      m.saving[kind] = `${i + 1}/${files.length}`;
      changed();
      const sameOrigin = f.url.startsWith("blob:") || new URL(f.url, location.href).origin === location.origin;
      const href = sameOrigin ? f.url : await fetch(f.url, {signal: stop.signal}).then((r) => r.ok ? r.blob() : null).then((b) => b && URL.createObjectURL(b), () => null);
      if (!href) continue;
      h("a", {href, download: fileName(n, f)}).click();
      if (!sameOrigin) setTimeout(() => URL.revokeObjectURL(href), 10000);
    }
    delete m.saving[kind];
    changed();
  }

  if (route.loose) {  // dropped files or a direct link: nothing to resolve or upload
    for (const [k, f] of route.files) m.found.set(k, f);
    for (const n of segs) m.asked.add(n);
    m.ready = Promise.resolve();
  } else {
    m.ready = Promise.all([refresh(false), deviceRecord(sr.dongle).catch(() => null)]).then(([, device]) => {
      if (stop.signal.aborted) return;
      m.device = device;
      m.up = uploads(route, device, (k) => m.uploaded.has(k), ({uploaded}) => {
        if (uploaded) {
          refresh();
          if (m.source === "auto" || m.source === "api") pick(m.source);
        }
        changed();
      });
    });
  }
  return m;
}

// ---- the Files panel: over the playbar, opened from its button or by right-clicking a segment on the timeline ----

let open = null;  // the panel shown: {show(n)}

// n: the segment to show, else the playhead's. Opened again it closes, unless it's to show a segment
export async function openFiles(anchor, ctx, n = null) {
  const fm = await ctx.routeFiles();
  if (open && n !== null) return open.show(n);
  let p = null;
  const el = popover(anchor, (close) => (p = panel(fm, ctx, n, close)).el,
    {class: "panel files-panel", role: "dialog", "aria-label": "Files", above: true, onclose: () => { p.dispose(); open = null; }});
  if (el) open = p;
}

// a lane per kind across the route, with its totals and its download- and request-all of the segments in view; one
// segment's files with sizes (pinned: picked on a lane or the timeline, else the playhead's); the source, notices,
// and the upload queue
function panel(fm, ctx, pinned, close) {
  const {route, clock} = ctx, segs = fm.segs;
  const inView = () => segmentsIn(route, clock.range);
  const playing = () => segmentAtTime(route, clock.t, clock.duration);
  const shownSeg = () => pinned ?? playing();
  let width = 0;

  const picker = h("select", {class: "fl-source", title: "Where this route's files come from", onchange: (e) => fm.pick(e.target.value)});
  const served = daemon.sources() || [];
  picker.append(h("option", {value: "auto"}, served.length ? "auto" : `auto (${AUTO.filter((s) => !SOURCES[s].daemon).map(label).join(" → ")})`),
    ...Object.entries(SOURCES).map(([name, s]) => {
      const off = s.daemon && !served.includes(name);
      return h("option", {value: name, disabled: off}, off ? `${s.label} (needs the local daemon)` : s.label);
    }));
  picker.value = fm.source;
  picker.hidden = !!route.loose;
  const summary = h("span", {class: "muted"});
  const notes = h("div", {class: "fl-notes"});
  const scope = h("span", {class: "muted"});
  const tip = h("div", {class: "status-tip"});
  const lanes = COLS.map(lane);
  const askAll = h("button", {class: "button small", hidden: true, onclick: () => fm.up.request(keysOf(inView(), UPLOADABLE))});
  const segName = h("b");
  const segNote = h("span", {class: "muted"});
  const follow = h("button", {class: "link", onclick: () => show(null)}, "back to the playhead");
  const step = (d) => { const i = segs.indexOf(shownSeg()) + d; if (segs[i] !== undefined) show(segs[i]); };
  const cells = COLS.map(() => ({el: h("span"), what: undefined}));
  const queueTitle = h("summary", {}, "Upload queue");
  const queueEl = h("div");

  const el = h("div", {class: "fl"},
    h("div", {class: "panel-head"}, h("b", {}, "Files"), picker, summary, h("button", {class: "link", autofocus: true, onclick: close}, "close")),
    notes,
    h("div", {class: "fl-scope"}, scope, askAll),
    h("div", {class: "fl-lanes"}, lanes.flatMap((l) => l.els)),
    tip,
    h("div", {class: "fl-seg"},
      h("div", {class: "fl-seg-head"},
        h("button", {class: "icon-btn", title: "Previous segment", "aria-label": "Previous segment", onclick: () => step(-1)}, icon("rewind", 12)),
        segName,
        h("button", {class: "icon-btn", title: "Next segment", "aria-label": "Next segment", onclick: () => step(1)}, icon("forward", 12)),
        segNote, follow),
      h("div", {class: "fl-cells"}, COLS.map((kind, j) => h("span", {}, h("span", {class: "fl-kind muted"}, kindName(kind)), cells[j].el)))),
    h("details", {class: "fl-queue"}, queueTitle, queueEl));

  // a kind's lane: a cell per segment, clicked to show it
  function lane(kind) {
    const canvas = h("canvas", {class: "fl-lane"});
    const total = h("span", {class: "fl-total muted"});
    const all = h("button", {class: "icon-btn", onclick: () => fm.downloadAll(kind, inView())}, icon("download", 14));
    const ask = h("button", {class: "icon-btn", hidden: true, onclick: () => fm.up.request(keysOf(inView(), [kind]))}, icon("upload", 14));
    const at = (e) => segs[Math.min(segs.length - 1, Math.floor(e.offsetX / canvas.clientWidth * segs.length))];
    canvas.addEventListener("pointermove", (e) => { const n = at(e); tip.textContent = `segment ${n} · ${kindName(kind)}: ${describe(n, kind)}`; });
    canvas.addEventListener("pointerleave", () => { tip.textContent = ""; });
    canvas.addEventListener("click", (e) => show(at(e)));
    return {kind, canvas, total, all, ask, els: [h("span", {class: "fl-kind"}, kindName(kind)), canvas, total, h("span", {class: "fl-acts"}, all, ask)]};
  }

  // a segment's file of a kind, said
  function describe(n, kind) {
    const key = `${n}/${kind}`, f = fm.found.get(key), s = fm.up?.state(key);
    if (f) return `${f.size ? `${fmtBytes(f.size)} ` : ""}from ${label(f.source)}${f.cached ? " (on disk)" : ""}`;
    if (s?.queued) return s.current ? `uploading ${Math.floor(100 * s.progress)}%` : s.paused ? "upload paused" : "upload pending";
    if (s?.requested) return "requested";
    if (s?.notFound) return "not on the device";
    if (s?.error) return `request failed: ${s.error}`;
    if (requestable(key)) return "on the device: click, then request it";
    return fm.asked.has(n) ? "not available" : "…";
  }
  const requestable = (key) => fm.up?.canUpload && fm.up.missing([key]).length > 0;

  // a cell per segment: blue when it's there, orange on its way from the device, grey when it could be asked for
  function paint(l, view) {
    const c = fitCanvas(l.canvas, width, 10), col = (v) => theme.color(v), n = segs.length, shown = shownSeg();
    c.clearRect(0, 0, width, 10);
    segs.forEach((seg, i) => {
      const a = i * width / n, w = Math.max(1, width / n - (width / n > 4 ? 1 : 0)), key = `${seg}/${l.kind}`, s = fm.up?.state(key);
      const fill = (color, alpha, f = 1) => { c.globalAlpha = alpha; c.fillStyle = col(color); c.fillRect(a, 0, w * f, 10); };
      if (fm.found.has(key)) fill("--accent", 1);
      else if (s?.queued || s?.requested) { fill("--line", 1); fill("--alert", s.current ? 1 : 0.45, s.current ? s.progress : 1); }
      else if (s?.error) fill("--critical", 1);
      else if (requestable(key)) fill("--line", 1);
      else if (!fm.asked.has(seg)) fill("--line", 0.3);
      if (!view.has(seg)) fill("--panel", 0.6);  // out of view: what the buttons leave out
      if (seg === shown) { c.globalAlpha = 1; c.strokeStyle = col("--fg"); c.strokeRect(a + 0.5, 0.5, w - 1, 9); }
    });
    c.globalAlpha = 1;
  }

  // what a segment's cell shows: the file, its upload state (objects, replaced on change), "up" when it can be
  // requested, or text
  function shown(n, kind) {
    const key = `${n}/${kind}`;
    return fm.found.get(key) || fm.up?.state(key) || (requestable(key) ? "up" : fm.asked.has(n) ? "–" : "…");
  }

  function cell(n, kind, what) {
    const key = `${n}/${kind}`;
    if (what === "up") {
      return h("button", {class: "button small", title: `Request ${kindName(kind)} ${n} from the device`, onclick: () => fm.up.request([key])}, icon("upload", 12), "request");
    }
    if (typeof what === "string") return h("span", {class: "muted"}, what);
    if (!what.url) return uploadState(fm.up, key) || h("span");
    return h("a", {class: "fl-dl", href: what.url, download: fm.fileName(n, what), title: `${what.file} from ${label(what.source)}${what.cached ? " (on disk)" : ""}`},
      icon("download", 14), what.size ? fmtBytes(what.size) : "download");
  }

  function show(n) {
    pinned = n;
    perf.invalidate(render);
  }

  const render = {name: "files panel", render() {
    fm.resolve();  // what isn't yet: on opening, and after another source is picked
    const ns = inView(), view = new Set(ns), whole = ns.length === segs.length, n = shownSeg();
    scope.textContent = whole ? `all ${segs.length} segments` : `segments ${ns[0]}–${ns.at(-1)} in view (${ns.length} of ${segs.length})`;
    let uploaded = 0;
    for (const l of lanes) {
      let count = 0, bytes = 0;
      for (const seg of ns) {
        const f = fm.found.get(`${seg}/${l.kind}`);
        if (f) { count++; bytes += f.size || 0; }
      }
      for (const seg of segs) uploaded += fm.uploaded.has(`${seg}/${l.kind}`);
      const ask = fm.up?.canUpload ? fm.up.missing(keysOf(ns, [l.kind])).length : 0;
      l.total.textContent = fm.saving[l.kind] ? `saving ${fm.saving[l.kind]}`
        : `${count}/${ns.length}${ns.some((s) => !fm.asked.has(s)) ? "+" : ""}${bytes ? ` · ${fmtBytes(bytes)}` : ""}`;
      l.all.disabled = !!fm.saving[l.kind] || !count;
      l.all.title = `Download every ${kindName(l.kind)} file ${whole ? "" : "in view "}(${count})`;
      l.ask.hidden = !ask;
      l.ask.title = `Request every missing ${kindName(l.kind)} file ${whole ? "" : "in view "}from the device (${ask})`;
      paint(l, view);
    }
    const ask = fm.up?.canUpload ? fm.up.missing(keysOf(ns, UPLOADABLE)).length : 0;
    askAll.hidden = !ask;
    askAll.textContent = `Request ${ask} files`;
    askAll.title = `Request every missing ${UPLOADABLE.map(kindName).join(", ")} file ${whole ? "" : "in view "}from the device`;
    summary.textContent = route.loose ? `${segs.length} segments` : `${segs.length} segments · ${Math.round(100 * uploaded / (segs.length * COLS.length || 1))}% uploaded`;

    segName.textContent = `segment ${n}`;
    const sources = [...new Set(COLS.map((k) => fm.found.get(`${n}/${k}`)?.source).filter(Boolean))].map(label).join(", ");
    segNote.textContent = [pinned === null && "at the playhead", sources && `from ${sources}`].filter(Boolean).join(" · ");
    follow.hidden = pinned === null;
    COLS.forEach((kind, j) => {
      const what = shown(n, kind), c = cells[j];
      if (what === c.what) return;
      c.what = what;
      const el = cell(n, kind, what);
      c.el.replaceWith(el);
      c.el = el;
    });

    const failed = Object.entries(fm.errors);
    notes.replaceChildren(...[fm.up && uploadNotice(fm.up, keysOf(segs, COLS)),
      failed.length && h("div", {class: "small muted"}, failed.map(([s, msg]) => `${label(s)}: ${msg}`).join(" · ")),
      ctx.peekStore() && fm.source !== "api" && h("div", {class: "small muted"}, "Loaded logs keep their source until the route is reopened.")].filter(Boolean));
    if (fm.up) {
      queueTitle.textContent = `Upload queue${fm.up.queue.length ? ` (${fm.up.queue.length})` : ""}`;
      queueEl.replaceChildren(queueTable(fm.up));
    }
  }};

  const update = () => perf.invalidate(render);
  let playingSeg = playing();
  const offs = [fm.on(update), theme.on(update), clock.on("range", update),
    clock.on("t", () => { if (pinned === null && playingSeg !== (playingSeg = playing())) update(); })];
  const sized = new ResizeObserver(([e]) => { width = e.contentRect.width; update(); });
  sized.observe(lanes[0].canvas);
  fm.ready.then(update);
  return {el, show, dispose() { for (const off of offs) off(); sized.disconnect(); }};
}
