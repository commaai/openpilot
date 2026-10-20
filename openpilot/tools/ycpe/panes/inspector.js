// the inspector: one service's message at the playhead as a tree, whole from store.message (decoded again from the
// log), shown from the loaded columns until it arrives. The tree expands lazily, the rows are virtualized at a fixed
// height, and playback only patches the value cells of the visible rows that changed. From the keyboard: up and down
// move, right and left (or Enter) open and close a branch.
import {h, css} from "/dom.js";

await css(new URL("./inspector.css", import.meta.url));

const ROW = 18;  // px
let inspectors = 0;  // ids for the rows' aria-activedescendant

export function create(ctx, state) {
  let service = state.service || "carState";
  const expanded = new Set(state.expanded || [""]);
  let store = null, tree = null, msg = null, asking = false, lines = [], pool = [], height = 0, top = 0, shown = -1, msgT = NaN;
  let cursor = null;  // the path the keyboard is at
  const id = `inspector-${++inspectors}`;
  const offs = [];

  const picker = h("select", {title: "Service", onchange: (e) => { service = e.target.value; msg = null; build(); ask(); }});
  const when = h("span", {class: "inspector-when"});
  const note = h("div", {class: "inspector-note"});
  const win = h("div", {class: "inspector-window"});
  const spacer = h("div", {class: "inspector-spacer"}, win);
  const list = h("div", {class: "inspector-list", tabindex: 0, role: "tree", "aria-label": "Message", onkeydown: key,
    onscroll: () => { top = list.scrollTop; ctx.perf.invalidate(pane); }, onclick,
    onfocus: () => { if (lines.length && !lines.some((l) => l.path === cursor)) move(0); }}, spacer);
  const el = h("div", {class: "inspector-pane"}, h("div", {class: "inspector-bar"}, picker, when), note, list);

  const pane = {
    name: "inspector", el, render, dispose,
    resize(w, ht) {
      height = Math.max(0, ht - el.firstChild.offsetHeight - note.offsetHeight);
      const n = Math.ceil(height / ROW) + 1;
      while (pool.length < n) {
        const key = h("span", {class: "inspector-key"}), value = h("span", {class: "inspector-value"});
        win.append(h("div", {class: "inspector-row", id: `${id}-${pool.length}`, role: "treeitem", "data-slot": pool.length}, key, value));
        pool.push({key, value, line: null, text: ""});
      }
      pool.forEach((s, k) => { s.key.parentNode.hidden = k >= n; });
      shown = -1;
      ctx.perf.invalidate(pane);
    },
    save: () => ({service, expanded: [...expanded]}),
  };

  offs.push(ctx.clock.on("t", () => ctx.perf.invalidate(pane)));
  offs.push(ctx.onStore((s) => {
    if (!offs.length) return;  // disposed
    store = s;
    offs.push(s.on("paths", build), s.on("progress", () => ctx.perf.invalidate(pane)));
    build();
  }));
  void ctx.store;

  // the column tree of the service: {kids: Map(key → node), series?, list?: length of a long list not loaded}
  function build() {
    if (!store) return;
    const all = store.paths(), names = new Set(all.map((p) => p.path.split("/")[1]));
    if (picker.options.length !== names.size) {
      picker.replaceChildren(...[...names].sort().map((n) => h("option", {value: n, selected: n === service}, n)));
    }
    const prefix = `/${service}`, node = () => ({kids: new Map()});
    tree = node();
    const at = (rest) => rest.split("/").filter(Boolean).reduce((n, k) => n.kids.get(k) || n.kids.set(k, node()).get(k), tree);
    for (const p of all) if (p.path === prefix || p.path.startsWith(prefix + "/")) at(p.path.slice(prefix.length)).series = p.path;
    for (const [path, n] of store.lists) if (path.startsWith(prefix + "/")) at(path.slice(prefix.length)).list ??= n;
    flatten();
  }

  // the rows: every child of an expanded node, depth first; kept when only values changed
  function flatten() {
    const out = [];
    const walk = (path, depth) => {
      for (const key of keys(path)) {
        const sub = `${path}/${key}`, open = expanded.has(sub) && keys(sub).length > 0;
        out.push({path: sub, key, depth, open, container: !!keys(sub).length || !msg && !!node(sub)?.list});
        if (open) walk(sub, depth + 1);
      }
    };
    walk("", 0);
    const same = (a, b) => a.path === b.path && a.open === b.open && a.container === b.container;
    if (out.length === lines.length && out.every((l, k) => same(l, lines[k]))) return;
    lines = out;
    shown = -1;
    ctx.perf.invalidate(pane);
  }

  function node(path) { return path.split("/").slice(1).reduce((n, k) => n?.kids.get(k), tree); }
  function field(path) { return path.split("/").slice(1).reduce((o, k) => o?.[k], msg); }

  function keys(path) {
    if (msg) {
      const o = field(path);
      return o && typeof o === "object" && !ArrayBuffer.isView(o) ? Object.keys(o) : [];
    }
    const n = node(path);
    if (!n) return [];
    const ks = [...n.kids.keys()];
    return ks.every((k) => /^\d+$/.test(k)) ? ks.sort((a, b) => a - b) : ks;
  }

  function text(line) {
    if (msg) {
      const v = field(line.path);
      if (Array.isArray(v)) return `[${v.length}]`;
      if (ArrayBuffer.isView(v)) return `${v.length} bytes ${hex(v)}`;
      return v && typeof v === "object" ? "" : show(v);
    }
    const n = node(line.path);
    if (n.list && !n.kids.size) return `[${n.list}] not loaded, open to load`;
    if (!n.series) return n.kids.size && /^\d+$/.test(n.kids.keys().next().value) ? `[${n.kids.size}]` : "";
    const s = store.series(n.series), v = s && exact(s, msgT);
    return v === undefined ? "" : show(s.names && v === v ? s.names[v] ?? v : s.type === "b" ? !!v : v);
  }

  function render() {
    if (!store) return;
    const t = ctx.clock.t, times = store.series(`/${service}/_valid`) || store.series(`/${service}/valid`);
    const prev = msgT;
    msgT = times ? lastAtOrBefore(times, t) : NaN;
    when.textContent = msgT === msgT ? `${msgT.toFixed(3)} s (${(msgT - t).toFixed(3)})` : "no message yet";
    const missing = [...store.missing || []].filter((m) => m === service || m.startsWith(service + "/"));
    note.textContent = missing.length ? `not migrated (a long list wasn't loaded): ${missing.join(", ")}` : "";
    if (msgT !== prev) ask();

    spacer.style.height = `${lines.length * ROW}px`;
    const first = Math.floor(top / ROW);
    if (first !== shown) win.style.transform = `translateY(${first * ROW}px)`;
    shown = first;
    let current = "";
    for (let s = 0; s < pool.length; s++) {
      const slot = pool[s], line = lines[first + s] ?? null, row = slot.key.parentNode;
      if (slot.line !== line) {
        slot.line = line;
        slot.key.textContent = line ? `${line.container ? line.open ? "▾ " : "▸ " : "  "}${line.key}` : "";
        slot.key.style.paddingLeft = `${(line?.depth ?? 0) * 12 + 4}px`;
        row.ariaLevel = line ? line.depth + 1 : null;
        row.ariaExpanded = line?.container ? String(line.open) : null;
      }
      const cur = !!line && line.path === cursor;
      if (cur) current = row.id;
      if (slot.cur !== cur) row.classList.toggle("cur", slot.cur = cur);
      const value = line ? text(line) : "";
      if (slot.text !== value) slot.value.textContent = slot.text = value;
    }
    if (list.getAttribute("aria-activedescendant") !== current) list.setAttribute("aria-activedescendant", current);
  }

  // the whole message at the playhead's message time; one request at a time
  async function ask() {
    if (asking || msgT !== msgT) return;
    asking = true;
    const svc = service, t = msgT, m = await store.message(svc, t).catch(() => null);
    asking = false;
    if (svc !== service) return;
    msg = m !== null && typeof m !== "object" ? {[svc]: m} : m;  // a text service is one field
    flatten();
    ctx.perf.invalidate(pane);
    if (msgT !== t) ask();  // the playhead moved on meanwhile
  }

  function onclick(e) {
    const line = pool[e.target.closest(".inspector-row")?.dataset.slot]?.line;
    if (!line) return;
    cursor = line.path;
    toggle(line);
  }

  function key(e) {
    const k = lines.findIndex((l) => l.path === cursor), line = lines[k];
    if (e.key === "ArrowDown" || e.key === "ArrowUp") move(Math.max(0, Math.min(lines.length - 1, k + (e.key === "ArrowDown" ? 1 : -1))));
    else if (e.key === "Home" || e.key === "End") move(e.key === "Home" ? 0 : lines.length - 1);
    else if (!line) return;
    // right opens a branch, then goes into it; left closes it, then goes to its parent
    else if (e.key === "ArrowRight" && line.container) line.open ? move(k + 1) : toggle(line, true);
    else if (e.key === "ArrowLeft") line.open ? toggle(line, false) : move(lines.findIndex((l) => l.path === line.path.slice(0, line.path.lastIndexOf("/"))));
    else if (e.key === "Enter" && line.container) toggle(line);
    else return;
    e.preventDefault();
  }

  // the keyboard to line k, scrolled into view
  function move(k) {
    if (!lines[k]) return;
    cursor = lines[k].path;
    const y = k * ROW;
    if (y < top) list.scrollTop = top = y;
    else if (y + ROW > top + height) list.scrollTop = top = y + ROW - height;
    ctx.perf.invalidate(pane);
  }

  // opens or closes a branch (open: which; else the other way)
  function toggle(line, open = !expanded.has(line.path)) {
    ctx.perf.invalidate(pane);
    if (!line.container) return;
    if (!open) expanded.delete(line.path);
    else {
      expanded.add(line.path);
      const n = !msg && node(line.path);
      if (n?.list && !n.kids.size) store.series(`/${service}${line.path}/0`);  // starts loading the long list
    }
    flatten();
  }

  function dispose() {
    for (const off of offs.splice(0)) off?.();
  }

  return pane;
}

// the value of a series in the message at time t exactly; undefined when that message doesn't have it
function exact(s, t) {
  const c = s.chunks.find((c) => c.t[0] <= t && t <= c.t[c.t.length - 1]);
  if (!c) return undefined;
  if (c.i?.length === 2 && c.v.length === 2 && c.i[1] === c.t.length - 1 && c.v[0] === c.v[1]) return c.v[0];  // constant
  let lo = 0, hi = c.tt.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (c.tt[mid] < t) lo = mid + 1; else hi = mid;
  }
  return c.tt[lo] === t ? c.v[lo] : undefined;
}

// the time of the series' last message at or before t
function lastAtOrBefore(s, t) {
  let best = NaN;
  for (const c of s.chunks) {
    if (c.t[0] > t) break;
    let lo = 0, hi = c.t.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (c.t[mid] <= t) lo = mid + 1; else hi = mid;
    }
    if (lo) best = c.t[lo - 1];
  }
  return best;
}

const show = (v) => typeof v === "number" && !Number.isInteger(v) ? String(+v.toPrecision(7)) : typeof v === "string" ? JSON.stringify(v) : String(v);
const hex = (b) => Array.from(b.subarray(0, 32), (x) => x.toString(16).padStart(2, "0")).join(" ") + (b.length > 32 ? " …" : "");
