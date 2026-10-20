// files pane: a route's files per segment from the chosen source (data/sources.js), with sizes, downloads, upload
// requests for what's still on the device, and per kind a download-all and a request-all. Rows are virtualized and
// resolved as they scroll into view, so a 300+ segment route costs what's on screen.
import {api} from "../api.js";
import {h, css} from "../dom.js";
import {icon} from "../icons.js";
import {fmtBytes} from "../status.js";
import {KINDS, API_KIND, SOURCES, AUTO, daemon, parseRange, resolveFiles, routeSource} from "../data/sources.js";
import {uploads, uploadState, uploadNotice, uploadQueue, LABELS, UPLOADABLE} from "../pages/files.js";

await css(new URL("./files.css", import.meta.url));

const ROW = matchMedia("(pointer: coarse)").matches ? 36 : 26;  // px per segment row; files.css agrees
const OVERSCAN = 8;  // rows drawn and resolved beyond the viewport
const COLS = Object.keys(KINDS);
const label = (source) => SOURCES[source]?.label ?? source;

export function create(ctx) {
  const {route, perf} = ctx;
  const sr = parseRange(route.fullname);
  const segs = route.segment_numbers || Array.from({length: (route.maxqlog ?? -1) + 1}, (_, i) => i);
  let source = routeSource.get(route.fullname);
  let found = new Map(), errors = {}, asked = new Set(), gen = 0;  // gen drops answers for a source no longer picked
  let up = null, apiFiles = new Set(), height = 0, alive = true;
  let dirty = true, stale = new Set();  // every row and the header redraw, or just these segments' rows
  const rows = new Map();  // row index → element on screen
  const saving = {};  // kind → "done/total" while a download-all runs

  const picker = h("select", {class: "fp-source", title: "Where this route's files come from", onchange: (e) => pick(e.target.value)});
  const summary = h("span", {class: "fp-summary"});
  const notes = h("div", {class: "fp-notes"});
  const head = h("div", {class: "fp-row fp-head"});
  const body = h("div", {class: "fp-body", style: `height: ${segs.length * ROW}px`});
  const scroller = h("div", {class: "fp-scroll", onscroll: () => perf.invalidate(pane)}, head, body);
  const queueTitle = h("summary", {}, "Upload queue");
  const queueEl = h("div", {class: "fp-queue-list"});
  const el = h("div", {class: "files-pane"}, h("div", {class: "fp-bar"}, picker, summary), notes, scroller,
    h("details", {class: "fp-queue"}, queueTitle, queueEl));
  const sized = new ResizeObserver(([e]) => { height = e.contentRect.height; perf.invalidate(pane); });
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
      saving[kind] = `${i + 1}/${files.length}`;
      changed();
      const sameOrigin = f.url.startsWith("blob:") || new URL(f.url, location.href).origin === location.origin;
      const href = sameOrigin ? f.url : await fetch(f.url).then((r) => r.ok ? r.blob() : null).then((b) => b && URL.createObjectURL(b), () => null);
      if (!href) continue;
      h("a", {href, download: `${route.fullname.replace("|", "_")}--${n}--${f.file}`}).click();
      if (!sameOrigin) setTimeout(() => URL.revokeObjectURL(href), 10000);
    }
    delete saving[kind];
    changed();
  }

  function cell(n, kind) {
    const key = `${n}/${kind}`, f = found.get(key);
    if (f) {
      return h("a", {class: "fp-cell dl", href: f.url, download: `${route.fullname.replace("|", "_")}--${n}--${f.file}`, title: `${f.file} from ${label(f.source)}${f.cached ? " (on disk)" : ""}`},
        icon("download", 14), f.size ? fmtBytes(f.size) : "");
    }
    const state = up && uploadState(up, key);
    if (state) return h("span", {class: "fp-cell"}, state);
    if (up?.canUpload && up.missing([key]).length) {
      return h("span", {class: "fp-cell"}, h("button", {class: "icon-btn", title: `Request ${LABELS[kind]} ${n} from the device`, onclick: () => up.request([key])}, icon("upload", 14)));
    }
    return h("span", {class: "fp-cell muted"}, asked.has(n) ? "–" : "");
  }

  function fill(row, n) {
    const sources = new Set(COLS.map((k) => found.get(`${n}/${k}`)?.source).filter(Boolean));
    row.replaceChildren(h("span", {class: "fp-seg"}, n),
      h("span", {class: "fp-src muted"}, asked.has(n) ? [...sources].map(label).join(", ") || "–" : "…"),
      ...COLS.map((k) => cell(n, k)));
  }

  function updateHead() {
    let uploaded = 0;
    head.replaceChildren(h("span", {class: "fp-seg"}, "seg"), h("span", {class: "fp-src"}, "source"), ...COLS.map((kind) => {
      let count = 0, bytes = 0;
      for (const n of segs) {
        const f = found.get(`${n}/${kind}`);
        if (f) { count++; bytes += f.size || 0; }
        if (apiFiles.has(`${n}/${kind}`)) uploaded++;
      }
      const requestable = up?.canUpload ? up.missing(segs.map((n) => `${n}/${kind}`)) : [];
      return h("span", {class: "fp-cell"},
        h("span", {class: "fp-kind"}, LABELS[kind],
          h("button", {class: "icon-btn", title: `Download every ${LABELS[kind]} file`, disabled: !!saving[kind], onclick: () => downloadAll(kind)}, icon("download", 14)),
          up?.canUpload && UPLOADABLE.includes(kind) && h("button", {class: "icon-btn", title: `Request every missing ${LABELS[kind]} file (${requestable.length})`,
            disabled: !requestable.length, onclick: () => up.request(requestable)}, icon("upload", 14))),
        h("span", {class: "fp-total muted"}, saving[kind] ? `saving ${saving[kind]}` : `${count}/${segs.length}${asked.size < segs.length ? "+" : ""}${bytes ? ` · ${fmtBytes(bytes)}` : ""}`));
    }));
    summary.textContent = `${segs.length} segments · ${Math.round(100 * uploaded / (segs.length * COLS.length || 1))}% uploaded`;
    const failed = Object.entries(errors);
    notes.replaceChildren(...[up && uploadNotice(up),
      failed.length && h("div", {class: "small muted"}, failed.map(([s, m]) => `${label(s)}: ${m}`).join(" · ")),
      ctx.peekStore() && source !== "api" && h("div", {class: "small muted"}, "Loaded logs keep their source until the route is reopened.")].filter(Boolean));
    if (up) {
      queueTitle.textContent = `Upload queue${up.queue.length ? ` (${up.queue.length})` : ""}`;
      queueEl.replaceChildren(uploadQueue(up));
    }
  }

  function render() {
    const top = scroller.scrollTop;
    const first = Math.max(0, Math.floor(top / ROW) - OVERSCAN), last = Math.min(segs.length, Math.ceil((top + height) / ROW) + OVERSCAN);
    for (const [i, row] of rows) if (i < first || i >= last) { row.remove(); rows.delete(i); }
    const want = [];
    for (let i = first; i < last; i++) {
      let row = rows.get(i);
      if (!row) {
        row = body.appendChild(h("div", {class: "fp-row", style: `transform: translateY(${i * ROW}px)`}));
        rows.set(i, row);
        fill(row, segs[i]);
      } else if (dirty || stale.has(segs[i])) {
        fill(row, segs[i]);
      }
      if (!asked.has(segs[i])) want.push(segs[i]);
    }
    if (dirty || stale.size) updateHead();
    dirty = false;
    stale.clear();
    if (want.length && height) resolve(want);
  }

  const pane = {name: "files", el, render, resize() {}, save: () => ({}),
    dispose() { alive = false; gen++; sized.disconnect(); up?.dispose(); }};
  return pane;
}
