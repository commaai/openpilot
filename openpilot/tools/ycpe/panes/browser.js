// series browser: every store path as a tree with a filter, values at the playhead, multi-select and
// drag onto plots (or double-click into the last used one). Rows are virtualized: only the visible ones
// exist in the DOM.
import {h} from "/dom.js";
import {fmtValue} from "/panes/plot/draw.js";
import {PATHS_MIME, addToPlot} from "/panes/plot/view.js";

const ROW = 18, BUILD_MS = 2;
const deprecated = (p) => p.includes("DEPRECATED") || p.includes("/deprecated/");
// list indices sort numerically, names by code point (Intl.Collator is ~35x slower on 2k paths)
const order = (a, b) => {
  const x = +a, y = +b;
  return x === x && y === y ? x - y : a < b ? -1 : a > b ? 1 : 0;
};

export function create(ctx, state = {}) {
  const {store, clock, perf} = ctx;
  const st = {filter: state.filter || "", deprecated: !!state.deprecated, expanded: new Set(state.expanded || [])};
  const selected = new Set();
  let tree = null, stale = true, rows = [], anchor = -1, H = 0, total = 0;
  let nodes = new Map();  // path → tree node, for every node in the tree

  const filter = h("input", {type: "search", class: "filter", placeholder: "filter, or /regex/", value: st.filter, spellcheck: false,
    oninput: () => { st.filter = filter.value; refresh(); },
    onkeydown: (e) => { if (e.key === "Escape") { filter.value = st.filter = ""; refresh(); } }});
  const count = h("span", {class: "muted count"});
  const depr = h("label", {class: "muted", title: "show deprecated fields and internal ones (__logMonoTime)"},
    h("input", {type: "checkbox", checked: st.deprecated, onchange: (e) => { st.deprecated = e.target.checked; build(true); }}), "deprecated");
  const spacer = h("div", {class: "spacer"});
  const win = h("div", {class: "window"});
  const scroller = h("div", {class: "scroller", onscroll: () => perf.invalidate(pane)}, spacer, win);
  const el = h("div", {class: "browser"}, h("div", {class: "bar"}, filter, depr, count), scroller);
  const pane = {name: "browser", el, render, resize, dispose, save};

  // new paths arrive with segments and are added at most once a frame; a new DBC replaces the /dbc branch
  let all = null, done = 0;  // store.paths(), and how many of them the tree has been through
  const changed = () => { all = null; stale = true; perf.invalidate(pane); };
  const offs = [store.on("paths", changed), clock.on("t", () => perf.invalidate(pane)),
    store.can?.on("dbc", () => { drop("/dbc"); changed(); })].filter(Boolean);

  // the path tree; a node is {name, path, depth, kids? (sorted), meta?}. New paths are added to the tree as they
  // arrive, a few ms worth a frame (a DBC brings thousands), and only the branches they land in are re-sorted;
  // fewer paths than before rebuilds it.
  function build(reset = false) {
    const end = performance.now() + BUILD_MS;
    if (!all || reset) [all, done] = [store.paths(), 0];
    if (reset || !tree || all.length < total) [tree, nodes, done] = [{depth: -1}, new Map(), 0];
    const touched = new Set();
    for (; done < all.length && (done % 256 || performance.now() < end); done++) {
      const meta = all[done], known = nodes.get(meta.path);
      if (known) known.meta = meta;
      else if (st.deprecated || !(meta.internal || deprecated(meta.path))) node(meta.path, touched).meta = meta;
    }
    for (const n of touched) n.kids.sort((a, b) => order(a.name, b.name));
    total = all.length;
    stale = done < all.length;
    refresh();
  }

  // removes a top level branch
  function drop(path) {
    if (!tree?.kids) return;
    tree.kids = tree.kids.filter((k) => k.path !== path);
    for (const p of nodes.keys()) if (p === path || p.startsWith(`${path}/`)) nodes.delete(p);
  }

  // the node at path, added with its missing parents; the parents that got kids go in touched
  function node(path, touched) {
    let n = nodes.get(path);
    if (n) return n;
    const i = path.lastIndexOf("/"), parent = i ? node(path.slice(0, i), touched) : tree;
    n = {name: path.slice(i + 1), path, depth: parent.depth + 1};
    (parent.kids ||= []).push(n);
    touched.add(parent);
    nodes.set(path, n);
    return n;
  }

  function matcher() {
    const f = st.filter.trim();
    if (!f) return null;
    const re = f.length > 2 && f.startsWith("/") && f.endsWith("/") ? (() => { try { return new RegExp(f.slice(1, -1), "i"); } catch { return null; } })() : null;
    if (re) return (p) => re.test(p);
    const terms = f.toLowerCase().split(/\s+/);
    return (p) => { const q = p.toLowerCase(); return terms.every((t) => q.includes(t)); };
  }

  // the visible rows: expanded branches, or with a filter every match and its ancestors
  function refresh() {
    rows = [];
    let hits = 0;
    const match = matcher();
    const walk = (n) => {
      for (const k of n.kids || []) {
        if (match) {
          const start = rows.length;
          rows.push(k);
          const hit = k.meta && match(k.path);
          if (hit) hits++;
          if (k.kids) walk(k);
          if (!hit && rows.length === start + 1) rows.pop();
        } else {
          rows.push(k);
          if (k.kids && st.expanded.has(k.path)) walk(k);
        }
      }
    };
    if (tree) walk(tree);
    count.textContent = match ? `${hits}` : `${total}`;
    spacer.style.height = `${rows.length * ROW}px`;
    perf.invalidate(pane);
  }

  const leavesOf = (n, out = []) => {
    if (n.meta) out.push(n.path);
    for (const k of n.kids || []) leavesOf(k, out);
    return out;
  };

  function click(e, k) {
    const n = rows[k];
    if (n.kids && !e.shiftKey && !e.ctrlKey && !e.metaKey && !st.filter) {
      st.expanded.has(n.path) ? st.expanded.delete(n.path) : st.expanded.add(n.path);
      return refresh();
    }
    if (!n.meta) return;
    if (e.shiftKey && anchor >= 0) {
      for (let i = Math.min(anchor, k); i <= Math.max(anchor, k); i++) if (rows[i].meta) selected.add(rows[i].path);
    } else if (e.ctrlKey || e.metaKey) {
      selected.has(n.path) ? selected.delete(n.path) : selected.add(n.path);
      anchor = k;
    } else {
      selected.clear();
      selected.add(n.path);
      anchor = k;
    }
    perf.invalidate(pane);
  }

  function dragstart(e, k) {
    const n = rows[k];
    const paths = n.meta && selected.has(n.path) ? [...selected] : leavesOf(n);
    e.dataTransfer.setData(PATHS_MIME, JSON.stringify(paths));
    e.dataTransfer.setData("text/plain", paths.join("\n"));
    e.dataTransfer.effectAllowed = "copy";
  }

  // a pool of row elements, refilled from the scroll position
  const pool = [];
  function rowEl(i) {
    if (!pool[i]) {
      const r = pool[i] = h("div", {class: "row", draggable: true,
        onclick: (e) => click(e, r.k), ondragstart: (e) => dragstart(e, r.k),
        ondblclick: () => { if (rows[r.k].meta) addToPlot([rows[r.k].path]); }},
      h("span", {class: "twisty"}), h("span", {class: "name"}), h("span", {class: "val"}));
      win.append(r);
    }
    return pool[i];
  }

  function render() {
    if (stale) build();
    const first = Math.floor(scroller.scrollTop / ROW), n = Math.min(rows.length - first, Math.ceil(H / ROW) + 1);
    win.style.transform = `translateY(${first * ROW}px)`;
    for (let i = 0; i < Math.max(n, pool.length); i++) {
      const r = rowEl(i), k = first + i, node = rows[k];
      r.hidden = !node;
      if (!node) continue;
      r.k = k;
      const [twisty, name, val] = r.children;
      r.style.paddingLeft = `${4 + node.depth * 12}px`;
      r.classList.toggle("sel", selected.has(node.path));
      twisty.textContent = node.kids ? (st.filter || st.expanded.has(node.path) ? "▾" : "▸") : "";
      if (name.textContent !== node.name) name.textContent = node.name;
      const m = node.meta;
      r.title = m ? `${m.path}\n${m.type}${m.names ? `\n${m.names.map((s, i) => `${i} ${s}`).join("\n")}` : ""}` : node.path;
      const text = m ? fmtValue(store.valueAt(m.path, clock.t), m.names).slice(0, 200) : "";  // long text only needs what shows
      if (val.textContent !== text) val.textContent = text;
    }
  }

  function resize(w, hgt = el.clientHeight) {
    H = hgt;
    perf.invalidate(pane);
  }

  function save() {
    return {filter: st.filter, deprecated: st.deprecated, expanded: [...st.expanded]};
  }

  function dispose() {
    for (const off of offs) off();
  }

  build();
  return pane;
}
