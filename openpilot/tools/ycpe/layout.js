// tabs of recursive splits of panes (jotpluggler style). The model is plain JSON:
//   layout = {tabs: [{name, root}], active}
//   node   = {split: "row" | "col", sizes: [fractions], children: [node]} | {pane: type, state}
// Panes are rearranged by dragging their header onto another pane's edge (dock beside) or center (swap).
// Structural edits (tabs, splits, docking, divider sizes) go in the workspace's undo history (history.js); a pane's
// own state is the pane's business.
import {h} from "/dom.js";
import {undoHistory} from "/history.js";
import {icon} from "/icons.js";
import {perf} from "/perf.js";
import {PANES, loadPane} from "/panes.js";

const MIN = 0.05;   // smallest split fraction
const MIN_PX = {row: 120, col: 60};  // smallest pane, across and down, so a layout saved on a big screen stays usable
const DIVIDER = 4;  // px, as in style.css
const EDGE = 0.25;  // fraction of a pane that docks beside it rather than swapping

// a saved model that isn't one (hand edited, an older format) throws. Sizes that aren't positive numbers take an even
// share, and pane types this version doesn't have are empty.
export function checkLayout(model) {
  const node = (n) => n && typeof n === "object" && (n.children
    ? Array.isArray(n.children) && n.children.length && Array.isArray(n.sizes) && n.sizes.length === n.children.length && n.children.every(node)
    : typeof n.pane === "string");
  if (!Array.isArray(model?.tabs) || !model.tabs.length || !model.tabs.every((t) => node(t?.root))) throw new Error("not a layout");
  const fix = (n) => {
    if (!n.children) {
      if (!PANES[n.pane]) Object.assign(n, {pane: "empty", state: {}});
      return;
    }
    if (!n.sizes.every((s) => Number.isFinite(s) && s > 0)) {
      const sizes = n.sizes.map((s) => Number.isFinite(s) && s > 0 ? s : 1 / n.sizes.length), sum = sizes.reduce((a, b) => a + b);
      n.sizes = sizes.map((s) => s / sum);
    }
    n.children.forEach(fix);
  };
  for (const t of model.tabs) fix(t.root);
  return model;
}

export class Layout {
  constructor(ctx, model, onChange) {
    this.ctx = Object.create(ctx);  // panes see the shared ctx plus their layout
    this.ctx.layout = this;
    this.model = structuredClone(checkLayout(model));
    this.model.active = Math.max(0, Math.min(this.model.tabs.length - 1, this.model.active | 0));
    this.onChange = onChange;
    this.el = h("div", {class: "layout"});
    this.live = new Map();  // rendered node -> {pane, box}
    this.maxPath = null;    // path of the maximized node, if any
    this.resizer = new ResizeObserver((entries) => {
      for (const e of entries) {
        const live = e.target._live;
        if (!live?.pane) continue;
        live.pane.resize?.(e.contentRect.width, e.contentRect.height);
        perf.invalidate(live.pane);
      }
    });
    this.render();
  }

  get tab() { return this.model.tabs[this.model.active]; }

  changed() {
    this.onChange?.(this.save());
  }

  // call before an edit (label: what it does): remembers the layout as it is, or as it was for edits already made live
  // (a drag)
  checkpoint(label, before = structuredClone(this.save())) {
    undoHistory(this.ctx.clock).push(this.entry(label, before));
  }

  // the history entry that puts model back
  entry(label, model) {
    return {label, layout: this, restore: () => {
      const now = this.entry(label, structuredClone(this.save()));
      this.model = model;
      this.maxPath = null;
      this.render();
      this.changed();
      return now;
    }};
  }

  // the current layout with each pane's own state folded in; map (if given) collects rendered -> saved nodes
  save(map) {
    const walk = (n) => {
      const copy = n.children ? {...n, sizes: [...n.sizes], children: n.children.map(walk)}
        : {pane: n.pane, state: this.live.get(n)?.pane?.save?.() ?? n.state};
      map?.set(n, copy);
      return copy;
    };
    return {...this.model, tabs: this.model.tabs.map((t) => t === this.tab ? {...t, root: walk(t.root)} : t)};
  }

  selectTab(i) {
    if (i === this.model.active) return;
    this.model = this.save();
    this.maxPath = null;
    this.model.active = i;
    this.render();
    this.changed();
  }

  addTab(name = `tab ${this.model.tabs.length + 1}`, root = {pane: "empty", state: {}}) {
    this.checkpoint("new tab");
    this.model = this.save();
    this.model.tabs.push({name, root});
    this.model.active = this.model.tabs.length - 1;
    this.maxPath = null;
    this.render();
    this.changed();
  }

  closeTab(i) {
    if (this.model.tabs.length === 1) return;
    this.checkpoint("close tab");
    this.model = this.save();
    this.model.tabs.splice(i, 1);
    if (i <= this.model.active) this.model.active = Math.max(0, this.model.active - 1);
    this.maxPath = null;
    this.render();
    this.changed();
  }

  renameTab(i, name) {
    if (name === this.model.tabs[i].name) return;
    this.checkpoint("rename tab");
    this.model.tabs[i].name = name;
    this.changed();
  }

  dispose() {
    for (const {pane} of this.live.values()) pane?.dispose?.();
    this.live.clear();
    this.resizer.disconnect();
  }

  render() {
    this.dispose();
    const root = this.maxPath ? at(this.tab.root, this.maxPath) : this.tab.root;
    this.el.replaceChildren(this.node(root));
  }

  // every pane redraws (after a view switch, when panes may have missed updates while hidden)
  invalidateAll() {
    for (const {pane} of this.live.values()) if (pane) perf.invalidate(pane);
  }

  node(n) {
    if (!n.children) return this.paneBox(n);
    const box = h("div", {class: `split ${n.split}`});
    // each child's minimum, scaled down together when they don't all fit
    const mins = n.children.map((c) => minPx(c, n.split)), sum = mins.reduce((a, b) => a + b);
    const gaps = (n.children.length - 1) * DIVIDER;
    n.children.forEach((c, i) => {
      if (i) box.append(this.divider(n, i, box));
      const child = this.node(c);
      child.style.flex = `${n.sizes[i]} 1 0`;
      child.style[n.split === "row" ? "minWidth" : "minHeight"] = `min(${mins[i]}px, (100% - ${gaps}px) * ${mins[i] / sum})`;
      box.append(child);
    });
    return box;
  }

  // a divider between children i - 1 and i: dragged, or focused and moved with the arrows (shift: further), Home and
  // End; a double-click or Enter evens out the split. All in place: rendering again would rebuild the panes
  divider(n, i, box) {
    const row = n.split === "row";
    const el = h("div", {class: "divider", tabindex: 0, role: "separator", "aria-orientation": row ? "vertical" : "horizontal",
      "aria-label": "Resize the panes beside it", "aria-valuemin": 0, "aria-valuemax": 100});
    // where it sits in the split, in percent
    el._value = () => el.setAttribute("aria-valuenow", Math.round(100 * n.sizes.slice(0, i).reduce((x, y) => x + y)));
    el._value();
    const kids = () => [...box.children].filter((c) => !c.classList.contains("divider"));
    // the pair's sizes now and the least each can take, over the split's length in px
    const pair = () => {
      const rect = box.getBoundingClientRect(), total = row ? rect.width : rect.height;
      const [a, b] = [n.sizes[i - 1], n.sizes[i]];
      const [minA, minB] = [n.children[i - 1], n.children[i]].map((c, j) => Math.min([a, b][j], Math.max(MIN, minPx(c, n.split) / total)));
      return {total, a, b, minA, minB, kids: kids()};
    };
    const move = ({a, b, minA, minB, kids}, d) => {
      const da = Math.max(minA - a, Math.min(b - minB, d));
      n.sizes[i - 1] = a + da;
      n.sizes[i] = b - da;
      kids[i - 1].style.flex = `${n.sizes[i - 1]} 1 0`;
      kids[i].style.flex = `${n.sizes[i]} 1 0`;
      el._value();
    };
    el.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      const at = pair(), start = row ? e.clientX : e.clientY;
      const before = structuredClone(this.save());
      el.onpointermove = (ev) => move(at, ((row ? ev.clientX : ev.clientY) - start) / at.total);
      el.onpointerup = el.onpointercancel = () => {
        el.onpointermove = el.onpointerup = el.onpointercancel = null;
        if (n.sizes[i - 1] === at.a) return;
        this.checkpoint("resize", before);
        this.changed();
      };
    });
    const even = () => {
      this.checkpoint("resize");
      n.sizes = n.sizes.map(() => 1 / n.sizes.length);
      for (const c of kids()) c.style.flex = `${1 / n.sizes.length} 1 0`;
      for (const c of box.children) c._value?.();
      this.changed();
    };
    el.addEventListener("dblclick", even);
    el.addEventListener("keydown", (e) => {
      const back = row ? "ArrowLeft" : "ArrowUp", forward = row ? "ArrowRight" : "ArrowDown";
      if (e.key === "Enter") { e.preventDefault(); return even(); }
      if (![back, forward, "Home", "End"].includes(e.key)) return;
      e.preventDefault();  // not the playhead's arrows
      const at = pair(), step = (e.shiftKey ? 50 : 10) / at.total, before = structuredClone(this.save());
      move(at, {[back]: -step, [forward]: step, Home: -Infinity, End: Infinity}[e.key]);
      if (n.sizes[i - 1] === at.a) return;
      this.checkpoint("resize", before);
      this.changed();
    });
    return el;
  }

  paneBox(n) {
    const info = PANES[n.pane] || PANES.empty;
    const title = h("span", {class: "pane-title", hidden: true});
    // a closed picker would swap the pane on arrow keys or typing; only picking from the open list does
    const types = h("select", {class: "pane-type", title: "Pane type", "aria-label": `Pane type: ${info.title}`, "data-key": "type", onchange: (e) => this.replace(n, {pane: e.target.value, state: {}}),
      onkeydown: (e) => {
        if (e.key === "Escape") e.target.blur();
        else if (!e.altKey && !e.ctrlKey && !e.metaKey && !["Enter", " ", "Tab", "F4"].includes(e.key)) e.preventDefault();
      }},
      n.pane === "empty" && h("option", {value: "empty", selected: true, disabled: true}, "Choose…"),
      Object.entries(PANES).filter(([, p]) => !p.hidden).map(([k, p]) => h("option", {value: k, selected: k === n.pane}, p.title)));
    // named for their pane, as a screen reader lists every pane's buttons together
    const btn = (name, tip, key, fn) => h("button", {class: "pane-btn", title: tip, "aria-label": `${tip}: ${info.title}`, "data-key": key, onclick: fn}, icon(name, 14));
    const head = h("div", {class: "pane-head", title: "Drag to move this pane"}, types, icon("down", 14), title,
      h("span", {class: "pane-tools"},
        btn("splitRight", "Split right", "row", () => this.split(n, "row")),
        btn("splitDown", "Split down", "col", () => this.split(n, "col")),
        btn(this.maxPath ? "restore" : "maximize", this.maxPath ? "Restore" : "Maximize", "max", () => this.maximize(n)),
        btn("close", "Close pane", "close", () => this.close(n))));
    head.addEventListener("pointerdown", (e) => this.drag(n, e));
    head.addEventListener("dblclick", (e) => { if (e.target === head || e.target === title) this.maximize(n); });
    const body = h("div", {class: "pane-body"});
    const box = h("section", {class: "pane", "aria-label": info.title}, head, body);
    const live = {pane: null, box};
    this.live.set(n, live);
    box._node = n;
    body._live = live;
    loadPane(n.pane).then((create) => {
      if (this.live.get(n) !== live) return;  // re-rendered meanwhile
      const pane = create(this.ctx, n.state || {});
      pane.name ||= n.pane;
      live.pane = pane;
      body.append(pane.el);
      if (pane.title) {
        title.textContent = pane.title;
        title.hidden = false;
        box.setAttribute("aria-label", `${info.title}: ${pane.title}`);
      }
      this.resizer.observe(body);
      perf.invalidate(pane);
    }).catch((e) => body.append(h("div", {class: "pane-missing"}, `${info.title}: ${e.message}`)));
    return box;
  }

  // dragging a pane header: a hint shows where it lands; edges dock beside, the center swaps
  // (not in the phone's stacked drive view: nothing docks there, and its headers scroll the page)
  drag(n, e) {
    if (e.button !== 0 || e.target.closest("button, select, input") || getComputedStyle(this.el).display === "block") return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const x0 = e.clientX, y0 = e.clientY;
    let hint = null, drop = null;
    const move = (ev) => {
      if (ev.pointerId !== e.pointerId || (!hint && Math.hypot(ev.clientX - x0, ev.clientY - y0) < 5)) return;
      hint ||= document.body.appendChild(h("div", {class: "dock-hint"}));
      document.body.classList.add("docking");
      const box = document.elementFromPoint(ev.clientX, ev.clientY)?.closest(".pane");
      drop = box && box._node !== n && this.live.has(box._node) ? {node: box._node, zone: zoneOf(box.getBoundingClientRect(), ev)} : null;
      hint.hidden = !drop;
      if (drop) {
        const r = box.getBoundingClientRect(), z = drop.zone;
        const w = z === "left" || z === "right" ? r.width / 2 : r.width, ht = z === "top" || z === "bottom" ? r.height / 2 : r.height;
        Object.assign(hint.style, {left: `${z === "right" ? r.right - w : r.left}px`, top: `${z === "bottom" ? r.bottom - ht : r.top}px`, width: `${w}px`, height: `${ht}px`});
      }
    };
    // only a release docks: a cancelled gesture (a touch the browser took over) or Escape doesn't
    const end = (ev) => {
      if (ev && ev.pointerId !== e.pointerId) return;
      removeEventListener("pointermove", move);
      removeEventListener("pointerup", end);
      removeEventListener("pointercancel", end);
      removeEventListener("keydown", cancel);
      hint?.remove();
      document.body.classList.remove("docking");
      if (ev?.type === "pointerup" && drop) this.dock(n, drop.node, drop.zone);
    };
    const cancel = (ev) => { if (ev.key === "Escape") end(); };
    addEventListener("pointermove", move);
    addEventListener("pointerup", end);
    addEventListener("pointercancel", end);
    addEventListener("keydown", cancel);
  }

  // structural edits (label: what it does) run on a freshly saved model; get(n) maps a rendered node to its saved copy
  // fn may return a node to focus instead (its pane type)
  edit(label, fn) {
    this.checkpoint(label);
    const map = new Map();
    const maxed = this.maxPath && at(this.tab.root, this.maxPath);
    const focus = this.focused();
    this.model = this.save(map);
    const moved = fn(this.tab, (n) => map.get(n));
    // a maximized node stays maximized, along with anything split off it (which takes its place)
    const path = maxed && pathOf(this.tab.root, map.get(maxed));
    this.maxPath = path && (path.slice(0, this.maxPath.length).join() === this.maxPath.join() ? this.maxPath : path);
    this.render();
    if (moved) this.focus(moved, "type"); else if (focus) this.focus(map.get(focus.node), focus.key);
    this.changed();
  }

  // the pane control with focus, as {node, key}: rendering replaces it
  focused() {
    const el = document.activeElement, box = el?.closest?.(".pane");
    return box && this.el.contains(box) && el.dataset.key ? {node: box._node, key: el.dataset.key} : null;
  }

  // a narrow pane hides its split buttons: then its pane type
  focus(node, key) {
    while (node?.children) node = node.children[0];
    const head = this.live.get(node)?.box.firstChild, el = head?.querySelector(`[data-key="${key}"]`);
    (el?.checkVisibility() ? el : head?.querySelector("[data-key=type]"))?.focus();
  }

  split(n, dir) {
    // the maximized pane splits on its own, rather than into a parent that isn't shown
    const wrap = this.maxPath && at(this.tab.root, this.maxPath) === n;
    this.edit("split pane", (tab, get) => {
      const fresh = {pane: "empty", state: {}};
      if (wrap) replaceNode(tab, get(n), {split: dir, sizes: [0.5, 0.5], children: [get(n), fresh]});
      else insertBeside(tab, get(n), fresh, dir, false);
    });
  }

  close(n) {
    this.edit(`close ${(PANES[n.pane]?.title || n.pane).toLowerCase()}`, (tab, get) => {
      if (tab.root === get(n)) {
        tab.root = {pane: "empty", state: {}};
        return tab.root;
      }
      const [p, i] = parentOf(tab.root, get(n));
      const sibling = p.children[i ? i - 1 : 1];
      detach(tab, get(n));
      return sibling;
    });
  }

  replace(n, fresh) {
    this.edit("change pane", (tab, get) => { Object.assign(get(n), fresh); });  // in place: a maximized pane stays so
  }

  // a pane swapping itself for another (the empty pane's picker)
  replaceSelf(el, fresh) {
    for (const [n, live] of this.live) if (live.pane?.el === el) return this.replace(n, fresh);
  }

  dock(src, target, zone) {
    this.edit("move pane", (tab, get) => {
      const a = get(src), b = get(target);
      if (zone === "center") return swapNodes(tab, a, b);
      detach(tab, a);
      insertBeside(tab, b, a, zone === "left" || zone === "right" ? "row" : "col", zone === "left" || zone === "top");
    });
  }

  maximize(n) {
    const path = pathOf(this.tab.root, n), focus = this.focused(), map = new Map();
    this.model = this.save(map);
    this.maxPath = this.maxPath ? null : path;
    this.render();
    if (focus) this.focus(map.get(focus.node), focus.key);
  }
}

// the smallest a node can be along dir: panes side by side add up
function minPx(n, dir) {
  if (!n.children) return MIN_PX[dir];
  const mins = n.children.map((c) => minPx(c, dir));
  return n.split === dir ? mins.reduce((x, y) => x + y) + (mins.length - 1) * DIVIDER : Math.max(...mins);
}

function zoneOf(r, e) {
  const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  const edges = [["left", x], ["right", 1 - x], ["top", y], ["bottom", 1 - y]].sort((p, q) => p[1] - q[1]);
  return edges[0][1] < EDGE ? edges[0][0] : "center";
}

// tree helpers, all by node identity
function parentOf(n, target) {
  if (!n.children) return null;
  const i = n.children.indexOf(target);
  if (i >= 0) return [n, i];
  for (const c of n.children) {
    const found = parentOf(c, target);
    if (found) return found;
  }
  return null;
}

function pathOf(root, target) {
  if (root === target) return [];
  for (let i = 0; i < (root.children?.length ?? 0); i++) {
    const rest = pathOf(root.children[i], target);
    if (rest) return [i, ...rest];
  }
  return null;
}

function at(root, path) {
  return path.reduce((n, i) => n.children[i], root);
}

function replaceNode(tab, old, fresh) {
  if (tab.root === old) { tab.root = fresh; return; }
  const [p, i] = parentOf(tab.root, old);
  p.children[i] = fresh;
}

// removes a node; its share goes to its siblings, and a split left with one child collapses into it
function detach(tab, node) {
  const [p, i] = parentOf(tab.root, node);
  p.children.splice(i, 1);
  const [freed] = p.sizes.splice(i, 1);
  p.sizes = p.sizes.map((s) => s + freed / p.sizes.length);
  if (p.children.length === 1) replaceNode(tab, p, p.children[0]);
}

// puts node next to target; joins target's split when it already runs in that direction
function insertBeside(tab, target, node, dir, before) {
  const found = parentOf(tab.root, target);
  if (found && found[0].split === dir) {
    const [p, i] = found;
    const half = p.sizes[i] / 2;
    p.sizes.splice(i, 1, half, half);
    p.children.splice(before ? i : i + 1, 0, node);
  } else {
    replaceNode(tab, target, {split: dir, sizes: [0.5, 0.5], children: before ? [node, target] : [target, node]});
  }
}

function swapNodes(tab, a, b) {
  const [pa, ia] = parentOf(tab.root, a), [pb, ib] = parentOf(tab.root, b);
  pa.children[ia] = b;
  pb.children[ib] = a;
}
