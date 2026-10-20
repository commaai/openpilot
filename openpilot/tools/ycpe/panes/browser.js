// series browser: every store path as a tree with a filter, values at the playhead, multi-select and
// drag onto plots (or double-click into the last used one). From the keyboard: arrows move (shift extends the
// selection, right and left open and close a branch), Enter adds to the last used plot. Rows are virtualized:
// only the visible ones exist in the DOM.
import {h} from "/dom.js";
import {icon} from "/icons.js";
import {fmtValue} from "/panes/plot/draw.js";
import {PATHS_MIME, addToPlot} from "/panes/plot/view.js";

const COARSE = matchMedia("(pointer: coarse)").matches;
const ROW = COARSE ? 28 : 18, BUILD_MS = 2;  // px per row; plot.css agrees
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
  let tree = null, stale = true, rows = [], anchor = null, H = 0, total = 0;  // anchor: the path a shift-click ranges from
  let cursor = null;  // the path the keyboard is at
  let top = 0;  // the scroller's scrollTop, kept from its scroll events: reading it in render would force a layout
  let nodes = new Map();  // path → tree node, for every node in the tree

  const filter = h("input", {type: "search", class: "filter", placeholder: "filter, or /regex/", value: st.filter, spellcheck: false,
    oninput: () => { st.filter = filter.value; refresh(); },
    onkeydown: (e) => {
      if (e.key === "Escape") { filter.value = st.filter = ""; refresh(); }
      if (e.key === "ArrowDown" && rows.length) { e.preventDefault(); scroller.focus(); move(0, false); }
    }});
  const count = h("span", {class: "muted count"});
  const depr = h("label", {class: "muted", title: "show deprecated fields and internal ones (__logMonoTime)"},
    h("input", {type: "checkbox", checked: st.deprecated, onchange: (e) => { st.deprecated = e.target.checked; build(true); }}), "deprecated");
  const spacer = h("div", {class: "spacer"});
  const win = h("div", {class: "window"});
  // why the list is empty, inside it: showing or hiding it moves nothing
  const note = h("div", {class: "note", hidden: true});
  const scroller = h("div", {class: "scroller", tabindex: 0, onkeydown: key}, note, spacer, win);
  const el = h("div", {class: "browser"}, h("div", {class: "bar"}, filter, depr, count), scroller);
  const pane = {name: "browser", el, render, resize, dispose, save};
  scroller.addEventListener("scroll", () => { top = scroller.scrollTop; perf.invalidate(pane); }, {passive: true});

  // new paths arrive with segments and are added at most once a frame; a new DBC replaces the /dbc branch
  let all = null, done = 0;  // store.paths(), and how many of them the tree has been through
  const changed = () => { stale = true; perf.invalidate(pane); };
  const offs = [store.on("paths", changed), clock.on("t", () => perf.invalidate(pane)), store.on("progress", () => perf.invalidate(pane)),
    store.can?.on("dbc", () => { drop("/dbc"); changed(); })].filter(Boolean);

  // the path tree; a node is {name, path, depth, kids? (sorted), meta?}. New paths are added to the tree as they
  // arrive, a few ms worth a frame (a DBC brings thousands), and only the branches they land in are re-sorted;
  // fewer paths than before rebuilds it.
  function build(reset = false) {
    const end = performance.now() + BUILD_MS;
    const now = store.paths();  // appended to as paths arrive, so the tree goes on from where it was
    if (now !== all || reset) [all, done] = [now, 0];
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
    cursor = n.path;
    if (n.kids && !e.shiftKey && !e.ctrlKey && !e.metaKey && !st.filter) {
      st.expanded.has(n.path) ? st.expanded.delete(n.path) : st.expanded.add(n.path);
      return refresh();
    }
    select(k, e);
  }

  // selects row k like a click with e's modifier keys
  function select(k, e) {
    const n = rows[k];
    if (!n.meta) return perf.invalidate(pane);
    const a = e.shiftKey ? rows.findIndex((r) => r.path === anchor) : -1;  // rows change as branches open and close
    if (a >= 0) {
      for (let i = Math.min(a, k); i <= Math.max(a, k); i++) if (rows[i].meta) selected.add(rows[i].path);
    } else if (e.ctrlKey || e.metaKey) {
      selected.has(n.path) ? selected.delete(n.path) : selected.add(n.path);
      anchor = n.path;
    } else {
      selected.clear();
      selected.add(n.path);
      anchor = n.path;
    }
    perf.invalidate(pane);
  }

  function key(e) {
    const k = rows.findIndex((r) => r.path === cursor), n = rows[k];
    if (e.key === "ArrowDown" || e.key === "ArrowUp") move(Math.max(0, Math.min(rows.length - 1, k + (e.key === "ArrowDown" ? 1 : -1))), e.shiftKey);
    else if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && n?.kids && !st.filter) {
      st.expanded[e.key === "ArrowRight" ? "add" : "delete"](n.path);
      refresh();
    } else if (e.key === "Enter" && n?.meta) addToPlot(selected.has(n.path) ? [...selected] : [n.path]);
    else if (e.key === "Enter" && n?.kids) click({}, k);
    else return;
    e.preventDefault();
  }

  // the keyboard to row k, scrolled into view; a series there is selected (added to the selection with extend)
  function move(k, extend) {
    if (!rows[k]) return;
    cursor = rows[k].path;
    select(k, {shiftKey: extend});
    const y = k * ROW, height = scroller.clientHeight;
    if (y < top) scroller.scrollTop = top = y;
    else if (y + ROW > top + height) scroller.scrollTop = top = y + ROW - height;
  }

  function dragstart(e, k) {
    const n = rows[k];
    const paths = n.meta && selected.has(n.path) ? [...selected] : leavesOf(n);
    e.dataTransfer.setData(PATHS_MIME, JSON.stringify(paths));
    e.dataTransfer.setData("text/plain", paths.join("\n"));
    e.dataTransfer.effectAllowed = "copy";
  }

  // a pool of row elements, refilled from the scroll position; each remembers what it shows, so a frame writes only
  // what changed
  const pool = [];
  const flag = (r, cls, on) => { if (r.flags[cls] !== on) r.classList.toggle(cls, r.flags[cls] = on); };
  function rowEl(i) {
    if (!pool[i]) {
      const r = pool[i] = h("div", {class: "row", draggable: true,
        onclick: (e) => click(e, r.k), ondragstart: (e) => dragstart(e, r.k),
        ondblclick: () => {
          if (!rows[r.k].meta) return;
          addToPlot([rows[r.k].path]);
          if (COARSE) scroller.blur();  // a phone's series drawer (pages.css) closes over the plot that got it
        }},
      h("span", {class: "twisty"}, icon("chevron", 12)), h("span", {class: "name"}), h("span", {class: "val"}));
      r.flags = {};
      win.append(r);
    }
    return pool[i];
  }

  function render() {
    if (stale) build();
    // why the list is empty
    const failure = !total && store.failure, f = st.filter.trim();
    const text = rows.length ? "" : failure ? `Couldn't load the logs: ${failure}` : f ? `No series match "${f}"` : store.loading ? "Loading logs…" : "";
    if (note.textContent !== text) {
      [note.textContent, note.hidden] = [text, !text];
      note.classList.toggle("error", !!failure);
    }
    const first = Math.floor(top / ROW), n = Math.min(rows.length - first, Math.ceil(H / ROW) + 1);
    const shift = `translateY(${first * ROW}px)`;
    if (win.style.transform !== shift) win.style.transform = shift;
    for (let i = 0; i < Math.max(n, pool.length); i++) {
      const r = rowEl(i), k = first + i, node = rows[k];
      if (r.hidden !== !node) r.hidden = !node;
      if (!node) continue;
      r.k = k;
      const [, name, val] = r.children, m = node.meta;
      if (r.node !== node || r.meta !== m) {
        [r.node, r.meta] = [node, m];
        r.style.paddingLeft = `${4 + node.depth * 12}px`;
        name.textContent = node.name;
        // enum names and DBC value descriptions by value
        r.title = m ? [m.path, m.unit ? `${m.type}, ${m.unit}` : m.type, ...Object.entries(m.names || {}).map(([v, s]) => `${v} ${s}`)].join("\n") : node.path;
      }
      flag(r, "sel", selected.has(node.path));
      flag(r, "cur", node.path === cursor);
      flag(r, "branch", !!node.kids);
      flag(r, "open", !!node.kids && !!(st.filter || st.expanded.has(node.path)));
      const text = m ? fmtValue(store.valueAt(m.path, clock.t), m).slice(0, 200) : "";  // long text only needs what shows
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
