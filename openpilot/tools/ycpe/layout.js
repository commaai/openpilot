// tabs of recursive splits of panes (jotpluggler style). The model is plain JSON:
//   layout = {tabs: [{name, root}], active}
//   node   = {split: "row" | "col", sizes: [fractions], children: [node]} | {pane: type, state}
// Panes are rearranged by dragging their header onto another pane's edge (dock beside) or center (swap).
// Structural edits (tabs, splits, docking, divider sizes) can be undone; a pane's own state is the pane's business.
import {h} from "/dom.js";
import {perf} from "/perf.js";
import {PANES, loadPane} from "/panes.js";

const MIN = 0.05;   // smallest split fraction
const EDGE = 0.25;  // fraction of a pane that docks beside it rather than swapping
const UNDO = 50;    // layout edits remembered

// a saved model that isn't one (hand edited, an older format) throws
export function checkLayout(model) {
  const node = (n) => n && typeof n === "object" && (n.children
    ? Array.isArray(n.children) && n.children.length && Array.isArray(n.sizes) && n.sizes.length === n.children.length && n.children.every(node)
    : typeof n.pane === "string");
  if (!Array.isArray(model?.tabs) || !model.tabs.length || !model.tabs.every((t) => node(t?.root))) throw new Error("not a layout");
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
    this.undos = [];        // saved models before each edit, newest last
    this.redos = [];
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

  // call before an edit: remembers the layout as it is (or as it was, for edits already made live, like a drag)
  checkpoint(before = structuredClone(this.save())) {
    this.undos.push(before);
    if (this.undos.length > UNDO) this.undos.shift();
    this.redos.length = 0;
  }

  undo() { this.step(this.undos, this.redos); }
  redo() { this.step(this.redos, this.undos); }

  step(from, to) {
    if (!from.length) return;
    to.push(structuredClone(this.save()));
    this.model = from.pop();
    this.maxPath = null;
    this.render();
    this.changed();
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
    this.checkpoint();
    this.model = this.save();
    this.model.tabs.push({name, root});
    this.model.active = this.model.tabs.length - 1;
    this.maxPath = null;
    this.render();
    this.changed();
  }

  closeTab(i) {
    if (this.model.tabs.length === 1) return;
    this.checkpoint();
    this.model = this.save();
    this.model.tabs.splice(i, 1);
    if (i <= this.model.active) this.model.active = Math.max(0, this.model.active - 1);
    this.maxPath = null;
    this.render();
    this.changed();
  }

  renameTab(i, name) {
    if (name === this.model.tabs[i].name) return;
    this.checkpoint();
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
    n.children.forEach((c, i) => {
      if (i) box.append(this.divider(n, i, box));
      const child = this.node(c);
      child.style.flex = `${n.sizes[i]} 1 0`;
      box.append(child);
    });
    return box;
  }

  divider(n, i, box) {
    const el = h("div", {class: "divider"});
    el.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      const rect = box.getBoundingClientRect();
      const total = n.split === "row" ? rect.width : rect.height;
      const start = n.split === "row" ? e.clientX : e.clientY;
      const [a, b] = [n.sizes[i - 1], n.sizes[i]];
      const kids = [...box.children].filter((c) => !c.classList.contains("divider"));
      const before = structuredClone(this.save());
      el.onpointermove = (ev) => {
        const d = ((n.split === "row" ? ev.clientX : ev.clientY) - start) / total;
        const da = Math.max(MIN - a, Math.min(b - MIN, d));
        n.sizes[i - 1] = a + da;
        n.sizes[i] = b - da;
        kids[i - 1].style.flex = `${n.sizes[i - 1]} 1 0`;
        kids[i].style.flex = `${n.sizes[i]} 1 0`;
      };
      el.onpointerup = () => {
        el.onpointermove = el.onpointerup = null;
        if (n.sizes[i - 1] === a) return;
        this.checkpoint(before);
        this.changed();
      };
    });
    // in place, like the drag: rendering again would rebuild the panes from the model
    el.addEventListener("dblclick", () => {
      this.checkpoint();
      n.sizes = n.sizes.map(() => 1 / n.sizes.length);
      for (const c of box.children) if (!c.classList.contains("divider")) c.style.flex = `${1 / n.sizes.length} 1 0`;
      this.changed();
    });
    return el;
  }

  paneBox(n) {
    const info = PANES[n.pane] || PANES.empty;
    const title = h("span", {class: "pane-title", hidden: true});
    const types = h("select", {class: "pane-type", title: "Pane type", onchange: (e) => this.replace(n, {pane: e.target.value, state: {}})},
      n.pane === "empty" && h("option", {value: "empty", selected: true, disabled: true}, "Choose…"),
      Object.entries(PANES).filter(([, p]) => !p.hidden).map(([k, p]) => h("option", {value: k, selected: k === n.pane}, p.title)));
    const btn = (label, tip, fn) => h("button", {class: "pane-btn", title: tip, onclick: fn}, label);
    const head = h("div", {class: "pane-head", title: "Drag to move this pane"}, types, title,
      h("span", {class: "pane-tools"},
        btn("⇥", "Split right", () => this.split(n, "row")),
        btn("⤓", "Split down", () => this.split(n, "col")),
        btn(this.maxPath ? "⤡" : "⤢", this.maxPath ? "Restore" : "Maximize", () => this.maximize(n)),
        btn("×", "Close pane", () => this.close(n))));
    head.addEventListener("pointerdown", (e) => this.drag(n, e));
    head.addEventListener("dblclick", (e) => { if (e.target === head || e.target === title) this.maximize(n); });
    const body = h("div", {class: "pane-body"});
    const box = h("section", {class: "pane"}, head, body);
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
      if (pane.title) { title.textContent = pane.title; title.hidden = false; }
      this.resizer.observe(body);
      perf.invalidate(pane);
    }).catch((e) => body.append(h("div", {class: "pane-missing"}, `${info.title}: ${e.message}`)));
    return box;
  }

  // dragging a pane header: a hint shows where it lands; edges dock beside, the center swaps
  drag(n, e) {
    if (e.button !== 0 || e.target.closest("button, select, input")) return;
    const x0 = e.clientX, y0 = e.clientY;
    let hint = null, drop = null;
    const move = (ev) => {
      if (!hint && Math.hypot(ev.clientX - x0, ev.clientY - y0) < 5) return;
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
    const end = (ev) => {
      removeEventListener("pointermove", move);
      removeEventListener("pointerup", end);
      removeEventListener("keydown", cancel);
      hint?.remove();
      document.body.classList.remove("docking");
      if (ev && drop) this.dock(n, drop.node, drop.zone);
    };
    const cancel = (ev) => { if (ev.key === "Escape") { drop = null; end(); } };
    addEventListener("pointermove", move);
    addEventListener("pointerup", end);
    addEventListener("keydown", cancel);
  }

  // structural edits run on a freshly saved model; get(n) maps a rendered node to its saved copy
  edit(fn) {
    this.checkpoint();
    const map = new Map();
    this.model = this.save(map);
    fn(this.tab, (n) => map.get(n));
    this.maxPath = null;
    this.render();
    this.changed();
  }

  split(n, dir) {
    this.edit((tab, get) => insertBeside(tab, get(n), {pane: "empty", state: {}}, dir, false));
  }

  close(n) {
    this.edit((tab, get) => {
      if (tab.root === get(n)) tab.root = {pane: "empty", state: {}};
      else detach(tab, get(n));
    });
  }

  replace(n, fresh) {
    this.edit((tab, get) => replaceNode(tab, get(n), fresh));
  }

  // a pane swapping itself for another (the empty pane's picker)
  replaceSelf(el, fresh) {
    for (const [n, live] of this.live) if (live.pane?.el === el) return this.replace(n, fresh);
  }

  dock(src, target, zone) {
    this.edit((tab, get) => {
      const a = get(src), b = get(target);
      if (zone === "center") return swapNodes(tab, a, b);
      detach(tab, a);
      insertBeside(tab, b, a, zone === "left" || zone === "right" ? "row" : "col", zone === "left" || zone === "top");
    });
  }

  maximize(n) {
    const path = pathOf(this.tab.root, n);
    this.model = this.save();
    this.maxPath = this.maxPath ? null : path;
    this.render();
  }
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
