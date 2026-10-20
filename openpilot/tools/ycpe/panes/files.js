// files pane: a route's files per segment from the chosen source (data/sources.js), with sizes, downloads, upload
// requests for what's still on the device, per kind a download-all and a request-all, and connect's request of all logs
// or all files in the zoomed range. Rows are virtualized and resolved as they scroll into view, so a 300+ segment route
// costs what's on screen.
import {SEGMENT, api} from "/api.js";
import {h, css} from "/dom.js";
import {icon} from "/icons.js";
import {fmtBytes} from "/status.js";
import {KINDS, API_KIND, SOURCES, AUTO, daemon, parseRange, resolveFiles, routeSource} from "/data/sources.js";
import {uploads, uploadState, uploadNotice, uploadQueue, LABELS, UPLOADABLE} from "/pages/files.js";

await css(new URL("./files.css", import.meta.url));

const ROW = matchMedia("(pointer: coarse)").matches ? 36 : 26;  // px per segment row; files.css agrees
const OVERSCAN = 8;  // rows drawn and resolved beyond the viewport
const COLS = Object.keys(KINDS);
const label = (source) => SOURCES[source]?.label ?? source;

export function create(ctx) {
  const {route, perf, clock} = ctx;
  const sr = parseRange(route.fullname);
  const segs = route.segment_numbers || Array.from({length: (route.maxqlog ?? -1) + 1}, (_, i) => i);
  let source = routeSource.get(route.fullname);
  let found = new Map(), errors = {}, asked = new Set(), gen = 0;  // gen drops answers for a source no longer picked
  let up = null, apiFiles = new Set(), height = 0, top = 0, alive = true;
  const stop = new AbortController();  // download-alls end with the pane
  let dirty = true, stale = new Set();  // every row and the header redraw, or just these segments' rows
  const rows = new Map();  // row index → {el, src, cells, shown} on screen
  const saving = {};  // kind → "done/total" while a download-all runs

  const picker = h("select", {class: "fp-source", title: "Where this route's files come from", onchange: (e) => pick(e.target.value)});
  const summary = h("span", {class: "fp-summary"});
  // connect's "All logs" and "All files": what's missing of those kinds in the segments of the zoomed range
  const inRange = (kinds) => {
    const [r0, r1] = clock.range;
    return segs.filter((n, i) => {
      const slot = route.range ? i : n;  // where it plays on the timeline
      return slot * SEGMENT < r1 && (slot + 1) * SEGMENT > r0;
    }).flatMap((n) => kinds.map((kind) => `${n}/${kind}`));
  };
  const allLogs = h("button", {class: "button small", hidden: true, onclick: () => up.request(inRange(["rlog"]))});
  const allFiles = h("button", {class: "button small", hidden: true, onclick: () => up.request(inRange(UPLOADABLE))});
  let ranged = "";  // the range the buttons count
  const notes = h("div", {class: "fp-notes"});
  const head = h("div", {class: "fp-row fp-head"});
  const body = h("div", {class: "fp-body", style: `height: ${segs.length * ROW}px`});
  const scroller = h("div", {class: "fp-scroll", onscroll: () => { top = scroller.scrollTop; perf.invalidate(pane); }}, head, body);
  const queueTitle = h("summary", {}, "Upload queue");
  const queueEl = h("div", {class: "fp-queue-list"});
  const el = h("div", {class: "files-pane"}, h("div", {class: "fp-bar"}, picker, summary, allLogs, allFiles), notes, scroller,
    h("details", {class: "fp-queue"}, queueTitle, queueEl));
  // the scroller's own size: the notes and the upload queue above and below it change it, not just the pane's. Its
  // scrollTop is kept from here and its scroll events (a reattached scroller starts at 0), as reading it in render
  // would force a layout
  const sized = new ResizeObserver(([e]) => { height = e.contentRect.height; top = scroller.scrollTop; perf.invalidate(pane); });
  sized.observe(scroller);

  const changed = (ns) => {
    if (ns) for (const n of ns) stale.add(n);
    else dirty = true;
    perf.invalidate(pane);
  };

  const served = daemon.sources() || [];
  picker.append(h("option", {value: "auto"}, served.length ? "auto" : `auto (${AUTO.filter((n) => !SOURCES[n].daemon).map(label).join(" → ")})`),
    ...Object.entries(SOURCES).map(([name, s]) => {
      const off = s.daemon && !served.includes(name);
      return h("option", {value: name, disabled: off}, off ? `${s.label} (needs the local daemon)` : s.label);
    }));
  picker.value = source;
  if (route.loose) {  // dropped files or a direct link: nothing to resolve or upload
    picker.hidden = true;
    found = new Map(route.files);
    asked = new Set(segs);
  }

  function pick(name) {
    source = name;
    routeSource.set(route.fullname, name);
    gen++;
    found = new Map();
    errors = {};
    asked = new Set();
    changed();
  }

  async function resolve(batch, kinds = COLS) {
    const g = gen;
    if (kinds === COLS) for (const n of batch) asked.add(n);
    const r = await resolveFiles(sr, batch, kinds, routeSource.list(source), {sizes: true}).catch((e) => ({found: new Map(), errors: {[source]: e.message}}));
    if (g !== gen) return;
    for (const [k, f] of r.found) found.set(k, f);
    Object.assign(errors, r.errors);
    changed(batch);
  }

  // what's uploaded decides what can be requested, whichever source is shown
  async function loadApiFiles(fresh) {
    if (fresh) api.routeFiles.forget(route.fullname);
    const files = await api.routeFiles(route.fullname).catch(() => ({}));
    apiFiles = new Set(COLS.flatMap((kind) => (files[API_KIND[kind]] || []).map((u) => `${new URL(u).pathname.split("/").at(-2)}/${kind}`)));
    changed();
  }
  if (!route.loose) loadApiFiles(false);
  if (!route.loose) api.device(sr.dongle).catch(() => null).then((device) => {
    if (!alive) return;
    up = uploads(route, device, (k) => apiFiles.has(k), ({uploaded}) => {
      if (uploaded) {
        loadApiFiles(true);
        if (routeSource.list(source).includes("api")) pick(source);
      }
      changed();
    });
  });

  // every file of a kind, one after another; cross-origin files go through a blob so the download attribute holds
  async function downloadAll(kind) {
    const missing = segs.filter((n) => !asked.has(n));
    if (missing.length) await resolve(missing, [kind]);
    const files = segs.map((n) => [n, found.get(`${n}/${kind}`)]).filter(([, f]) => f);
    for (const [i, [n, f]] of files.entries()) {
      if (!alive) return;
      saving[kind] = `${i + 1}/${files.length}`;
      changed();
      const sameOrigin = f.url.startsWith("blob:") || new URL(f.url, location.href).origin === location.origin;
      const href = sameOrigin ? f.url : await fetch(f.url, {signal: stop.signal}).then((r) => r.ok ? r.blob() : null).then((b) => b && URL.createObjectURL(b), () => null);
      if (!href) continue;
      h("a", {href, download: `${route.fullname.replace("|", "_")}--${n}--${f.file}`}).click();
      if (!sameOrigin) setTimeout(() => URL.revokeObjectURL(href), 10000);
    }
    delete saving[kind];
    changed();
  }

  // what a cell shows: the file, its upload state (objects, replaced on change), "up" when it can be requested, or text
  function shown(n, kind) {
    const key = `${n}/${kind}`;
    return found.get(key) || up?.state(key) || (up?.canUpload && up.missing([key]).length ? "up" : asked.has(n) ? "–" : "");
  }

  function cell(n, kind, what) {
    const key = `${n}/${kind}`, f = found.get(key);
    if (f) {
      return h("a", {class: "fp-cell dl", href: f.url, download: `${route.fullname.replace("|", "_")}--${n}--${f.file}`, title: `${f.file} from ${label(f.source)}${f.cached ? " (on disk)" : ""}`},
        icon("download", 14), f.size ? fmtBytes(f.size) : "");
    }
    if (typeof what === "object") return h("span", {class: "fp-cell"}, uploadState(up, key));
    if (what === "up") {
      return h("span", {class: "fp-cell"}, h("button", {class: "icon-btn", title: `Request ${LABELS[kind]} ${n} from the device`, onclick: () => up.request([key])}, icon("upload", 14)));
    }
    return h("span", {class: "fp-cell muted"}, what);
  }

  // a row keeps its elements; only the cells whose content changed are rebuilt
  function fill(row, n) {
    const sources = new Set(COLS.map((k) => found.get(`${n}/${k}`)?.source).filter(Boolean));
    const src = asked.has(n) ? [...sources].map(label).join(", ") || "–" : "…";
    if (row.src.textContent !== src) row.src.textContent = src;
    COLS.forEach((kind, j) => {
      const what = shown(n, kind);
      if (what === row.shown[j]) return;
      row.shown[j] = what;
      const c = cell(n, kind, what);
      row.cells[j].replaceWith(c);
      row.cells[j] = c;
    });
  }

  // the header is built once; each change patches its totals and buttons
  const heads = COLS.map((kind) => {
    const all = h("button", {class: "icon-btn", title: `Download every ${LABELS[kind]} file`, onclick: () => downloadAll(kind)}, icon("download", 14));
    const ask = h("button", {class: "icon-btn", hidden: true, onclick: () => up.request(segs.map((n) => `${n}/${kind}`))}, icon("upload", 14));
    const total = h("span", {class: "fp-total muted"});
    return {all, ask, total, el: h("span", {class: "fp-cell"}, h("span", {class: "fp-kind"}, LABELS[kind], all, ask), total)};
  });
  head.append(h("span", {class: "fp-seg"}, "seg"), h("span", {class: "fp-src"}, "source"), ...heads.map((c) => c.el));

  function updateHead() {
    let uploaded = 0;
    COLS.forEach((kind, j) => {
      let count = 0, bytes = 0;
      for (const n of segs) {
        const f = found.get(`${n}/${kind}`);
        if (f) { count++; bytes += f.size || 0; }
        if (apiFiles.has(`${n}/${kind}`)) uploaded++;
      }
      const requestable = up?.canUpload && UPLOADABLE.includes(kind) ? up.missing(segs.map((n) => `${n}/${kind}`)).length : 0;
      const total = saving[kind] ? `saving ${saving[kind]}` : `${count}/${segs.length}${asked.size < segs.length ? "+" : ""}${bytes ? ` · ${fmtBytes(bytes)}` : ""}`;
      const c = heads[j];
      if (c.total.textContent !== total) c.total.textContent = c.total.title = total;
      c.all.disabled = !!saving[kind];
      c.ask.hidden = !requestable;  // nothing to ask for, no button
      c.ask.title = `Request every missing ${LABELS[kind]} file (${requestable})`;
    });
    summary.textContent = `${segs.length} segments · ${Math.round(100 * uploaded / (segs.length * COLS.length || 1))}% uploaded`;
    const failed = Object.entries(errors);
    notes.replaceChildren(...[up && uploadNotice(up, segs.flatMap((n) => COLS.map((kind) => `${n}/${kind}`))),
      failed.length && h("div", {class: "small muted"}, failed.map(([s, m]) => `${label(s)}: ${m}`).join(" · ")),
      ctx.peekStore() && source !== "api" && h("div", {class: "small muted"}, "Loaded logs keep their source until the route is reopened.")].filter(Boolean));
    if (up) {
      queueTitle.textContent = `Upload queue${up.queue.length ? ` (${up.queue.length})` : ""}`;
      queueEl.replaceChildren(uploadQueue(up));
    }
  }

  // the request-alls count what's missing in the range as it is
  function updateAll() {
    ranged = String(clock.range);
    for (const [button, kinds, what] of [[allLogs, ["rlog"], "logs"], [allFiles, UPLOADABLE, "files"]]) {
      const n = up?.canUpload ? up.missing(inRange(kinds)).length : 0, text = `Upload ${n} ${what}`;
      button.hidden = !n;
      if (button.textContent !== text) {
        button.textContent = text;
        button.title = `Request every missing ${kinds.map((k) => LABELS[k]).join(", ")} file of the segments in view from the device`;
      }
    }
  }

  function render() {
    const first = Math.max(0, Math.floor(top / ROW) - OVERSCAN), last = Math.min(segs.length, Math.ceil((top + height) / ROW) + OVERSCAN);
    for (const [i, row] of rows) if (i < first || i >= last) { row.el.remove(); rows.delete(i); }
    const want = [];
    for (let i = first; i < last; i++) {
      let row = rows.get(i);
      if (!row) {
        const cells = COLS.map(() => h("span"));
        row = {src: h("span", {class: "fp-src muted"}), cells, shown: []};
        row.el = body.appendChild(h("div", {class: "fp-row", style: `transform: translateY(${i * ROW}px)`}, h("span", {class: "fp-seg"}, segs[i]), row.src, ...cells));
        rows.set(i, row);
        fill(row, segs[i]);
      } else if (dirty || stale.has(segs[i])) {
        fill(row, segs[i]);
      }
      if (!asked.has(segs[i])) want.push(segs[i]);
    }
    if (dirty || stale.size) updateHead();
    if (dirty || stale.size || ranged !== String(clock.range)) updateAll();
    dirty = false;
    stale.clear();
    if (want.length && height) resolve(want);
  }

  const offRange = clock.on("range", () => perf.invalidate(pane));
  const pane = {name: "files", el, render, resize() {}, save: () => ({}),
    dispose() { alive = false; gen++; stop.abort(); sized.disconnect(); up?.dispose(); offRange(); }};
  return pane;
}
