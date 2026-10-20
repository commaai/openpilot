// the logs pane: swaglog (logMessage, errorLogMessage), alert and onroad event onsets, and the OS journal as one
// list, filtered by level, source and text. Segments are indexed a few ms per frame as they arrive; the list is
// virtualized with fixed rows, and following the playhead only patches the visible rows.
import {h, css} from "/dom.js";

await css(new URL("./logs.css", import.meta.url));

const ROW = 18;      // px
const BUDGET = 3;    // ms of indexing or filtering per frame
const LEVELS = ["debug", "info", "warning", "error", "critical"];
const SYSLOG = [4, 4, 4, 3, 2, 1, 1, 0];  // journal priority → level
const ALERT = [1, 2, 4];                   // alertStatus normal, userPrompt, critical → level
const ANSI = /\x1b\[[0-9;]*m/g;

export function create(ctx, state) {
  let minLevel = state.level ?? 1, source = state.source ?? "", text = state.text ?? "", follow = state.follow ?? true;
  let rows = [], view = [];  // every row and the rows that pass the filter, by time
  const sources = new Set(), indexed = new Set(), jobs = [];
  let job = null, untick = null, store = null, timer = 0, disposed = false;
  let height = 0, top = 0, expected = -1, current = -1, pool = [], shown = -1;

  const levelPick = h("select", {title: "Minimum level", onchange: (e) => filter({level: +e.target.value})},
    LEVELS.map((l, k) => h("option", {value: k, selected: k === minLevel}, l)));
  const sourcePick = h("select", {title: "Source", onchange: (e) => filter({source: e.target.value})});
  const search = h("input", {type: "search", placeholder: "Filter", value: text, oninput: (e) => {
    clearTimeout(timer);
    timer = setTimeout(() => filter({text: e.target.value.toLowerCase()}), 150);
  }});
  const followBtn = h("button", {class: "chip", title: "Follow the playhead", onclick: () => { follow = true; ctx.perf.invalidate(pane); }}, "Follow");
  const count = h("span", {class: "logs-count"});
  const note = h("div", {class: "logs-note"});
  const win = h("div", {class: "logs-window"});
  const spacer = h("div", {class: "logs-spacer"}, win);
  const list = h("div", {class: "logs-list", onscroll, onclick}, spacer, note);
  const el = h("div", {class: "logs-pane"}, h("div", {class: "logs-bar"}, levelPick, sourcePick, search, followBtn, count), list);

  const pane = {
    name: "logs", el, render, dispose,
    resize(w, ht) {
      height = Math.max(0, ht - el.firstChild.offsetHeight);
      const n = Math.ceil(height / ROW) + 1;
      while (pool.length < n) {
        const cells = [h("span", {class: "logs-t"}), h("span", {class: "logs-lv"}), h("span", {class: "logs-src"}), h("span", {class: "logs-msg"})];
        const row = h("div", {class: "logs-row", "data-slot": pool.length}, cells);
        win.append(row);
        pool.push({el: row, cells, row: null, current: false});
      }
      for (let s = 0; s < pool.length; s++) pool[s].el.hidden = s >= n;
      shown = -1;
      ctx.perf.invalidate(pane);
    },
    save: () => ({level: minLevel, source, text, follow}),
  };

  const work = () => ctx.perf.measure("logs indexing", slice);
  const offs = [ctx.clock.on("t", () => follow && ctx.perf.invalidate(pane))];
  ctx.onStore((s) => {
    if (disposed) return;
    store = s;
    offs.push(s.on("segment", enqueue));
    for (const seg of s.route.segments) if (seg.status === "done") enqueue(seg.n);
  });
  void ctx.store;  // start loading

  function enqueue(n) {
    if (indexed.has(n)) return;
    indexed.add(n);
    jobs.push(index(n));
    untick ||= ctx.perf.tick(work);
  }

  // runs queued indexing and filtering in slices of BUDGET ms
  function slice() {
    const end = performance.now() + BUDGET;
    while (performance.now() < end) {
      job ||= jobs.shift();
      if (!job) {
        untick();
        untick = null;
        break;
      }
      if (job.next().done) job = null;
    }
    ctx.perf.invalidate(pane);
  }

  function* index(n) {
    const block = [];
    for (const row of segmentRows(store, n)) {
      if (row) {
        block.push(row);
        sources.add(row.source);
      }
      yield;
    }
    if (!block.length) return;
    block.sort((a, b) => a.t - b.t);
    rows = splice(rows, block);
    view = splice(view, block.filter(pass));
  }

  function* refilter() {
    const next = [];
    for (const r of rows) {
      if (pass(r)) next.push(r);
      yield;
    }
    view = next;
  }

  function filter(f) {
    ({level: minLevel = minLevel, source = source, text = text} = f);
    if (job?.refilter) job = null;
    const r = refilter();
    r.refilter = true;
    jobs.splice(0, jobs.length, r, ...jobs.filter((j) => !j.refilter));
    untick ||= ctx.perf.tick(work);
  }

  function pass(r) {
    if (r.level < minLevel || source && r.source !== source) return false;
    return !text || (r.lower ??= `${r.source} ${r.text}`.toLowerCase()).includes(text);
  }

  function render() {
    spacer.style.height = `${view.length * ROW}px`;
    count.textContent = `${view.length.toLocaleString()} / ${rows.length.toLocaleString()}`;
    note.textContent = view.length ? "" : jobs.length || job ? "indexing…" : rows.length ? "no matches" : "no log messages";
    followBtn.classList.toggle("on", follow);
    if (sourcePick.options.length !== sources.size + 1) {
      sourcePick.replaceChildren(h("option", {value: ""}, "all sources"),
        [...sources].sort().map((s) => h("option", {value: s, selected: s === source}, s || "(none)")));
    }

    current = lastAtOrBefore(view, ctx.clock.t);
    if (follow) {
      const want = Math.max(0, Math.min(current * ROW - height / 2, view.length * ROW - height));
      if (want !== top) list.scrollTop = expected = top = want;
    }
    const first = Math.floor(top / ROW);
    if (first !== shown) win.style.transform = `translateY(${first * ROW}px)`;
    shown = first;
    for (let s = 0; s < pool.length; s++) {
      const slot = pool[s], r = view[first + s] ?? null;
      if (slot.row !== r) {
        slot.row = r;
        const [t, lv, src, msg] = slot.cells;
        t.textContent = r ? clock(r.t) : "";
        lv.textContent = r ? LEVELS[r.level] : "";
        src.textContent = r?.source ?? "";
        msg.textContent = r?.text ?? "";
        slot.el.className = r ? `logs-row lv${r.level}` : "logs-row";
        slot.current = false;
      }
      const cur = first + s === current;
      if (slot.current !== cur) slot.el.classList.toggle("current", slot.current = cur);
    }
  }

  // a scroll we didn't make stops following
  function onscroll() {
    const y = list.scrollTop;
    if (Math.abs(y - expected) > 1) follow = false;
    expected = -1;
    top = y;
    ctx.perf.invalidate(pane);
  }

  function onclick(e) {
    const r = pool[e.target.closest(".logs-row")?.dataset.slot]?.row;
    if (r) ctx.clock.seek(r.t);
  }

  function dispose() {
    disposed = true;
    clearTimeout(timer);
    untick?.();
    for (const off of offs) off();
  }

  return pane;
}

// one segment's rows, unsorted, and nulls between messages without one so that indexing can pause there
function* segmentRows(store, n) {
  // a step of its own each: decoding a segment's text takes a few ms
  function* chunk(path) {
    yield null;
    return store.chunk(path, n);
  }
  for (const path of ["/logMessage", "/errorLogMessage"]) {
    const c = yield* chunk(path);
    if (c) for (const [t, s] of pairs(c)) yield swaglog(t, s);
  }

  const os = yield* chunk("/operatingSystemLog/message");
  if (os) {
    const tag = values(yield* chunk("/operatingSystemLog/tag")), priority = values(yield* chunk("/operatingSystemLog/priority"));
    for (let [t, text] of pairs(os)) {
      try { text = String(JSON.parse(text).MESSAGE ?? text); } catch {}  // journal entries are JSON, android logs text
      yield {t, level: SYSLOG[priority(t)] ?? 1, source: tag(t) || "os", text};
    }
  }

  const alert = yield* chunk("/selfdriveState/alertType");
  if (alert) {
    const text1 = values(yield* chunk("/selfdriveState/alertText1")), text2 = values(yield* chunk("/selfdriveState/alertText2"));
    const status = values(yield* chunk("/selfdriveState/alertStatus"));
    let prev = "";
    for (const [t, type] of pairs(alert)) {
      yield type && type !== prev ? {t, level: ALERT[status(t)] ?? 1, source: "alert", text: `${text1(t) || ""} ${text2(t) || ""}`.trim() || type} : null;
      prev = type;
    }
  }

  // onroad events where they start
  const events = yield* chunk("/onroadEvents/valid");
  if (events) {
    const items = [];
    for (let k = 0; store.series(`/onroadEvents/${k}/name`); k++) {
      const flag = (f) => [f, values(store.chunk(`/onroadEvents/${k}/${f}`, n))];
      items.push({name: values(yield* chunk(`/onroadEvents/${k}/name`)), names: store.series(`/onroadEvents/${k}/name`).names,
        flags: ["immediateDisable", "softDisable", "noEntry", "userDisable", "warning"].map(flag)});
    }
    let prev = new Set();
    for (const t of events.t) {
      const now = new Set();
      for (const it of items) {
        const name = it.names?.[it.name(t)];
        if (name === undefined) continue;
        now.add(name);
        if (prev.has(name)) continue;
        const flags = it.flags.filter(([, v]) => v(t)).map(([f]) => f);
        const level = flags.some((f) => f.endsWith("Disable") && f !== "userDisable") ? 3 : flags.length ? 2 : 1;
        yield {t, level, source: "event", text: flags.length ? `${name} (${flags.join(", ")})` : name};
      }
      prev = now;
      yield null;
    }
  }
}

function swaglog(t, s) {
  let m;
  try { m = JSON.parse(s); } catch { return {t, level: 1, source: "", text: s}; }
  const msg = typeof m.msg === "string" ? m.msg : JSON.stringify(m.msg);
  const level = Math.max(0, Math.min(4, Math.round((m.levelnum ?? 20) / 10) - 1));
  return {t, level, source: m.ctx?.daemon || m.filename?.split("/").pop() || "", text: msg.replace(ANSI, "")};
}

// a chunk's [time, value] pairs
function* pairs(c) {
  for (let k = 0; k < c.v.length; k++) yield [c.tt[k], c.v[k]];
}

// a chunk's value at each of a rising sequence of message times; undefined where that message doesn't have it
function values(c) {
  if (!c) return () => undefined;
  const {v, tt} = c;
  let p = 0;
  return (t) => {
    while (p < tt.length && tt[p] < t) p++;
    return tt[p] === t ? v[p] : undefined;
  };
}

// inserts a time-sorted block into time-sorted rows (blocks are whole segments, so they don't interleave)
function splice(rows, block) {
  if (!block.length) return rows;
  const at = lastAtOrBefore(rows, block[0].t) + 1;
  return rows.slice(0, at).concat(block, rows.slice(at));
}

function lastAtOrBefore(rows, t) {
  let lo = 0, hi = rows.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (rows[mid].t <= t) lo = mid + 1; else hi = mid;
  }
  return lo - 1;
}

function clock(t) {
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${s.toFixed(3).padStart(6, "0")}`;
}
