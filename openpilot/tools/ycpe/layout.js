// tabs of recursive splits of panes (jotpluggler style). The model is plain JSON:
//   layout = {tabs: [{name, root}], active}
//   node   = {split: "row" | "col", sizes: [fractions], children: [node]} | {pane: type, state}
import {h} from "/dom.js";
import {perf} from "/perf.js";
import {PANES, loadPane} from "/panes.js";

const MIN = 0.05;  // smallest split fraction

export class Layout {
  constructor(ctx, model, onChange) {
    this.ctx = Object.create(ctx);  // panes see the shared ctx plus their layout
    this.ctx.layout = this;
    this.model = structuredClone(model);
    this.onChange = onChange;
    this.el = h("div", {class: "layout"});
    this.live = new Map();  // node -> {pane, box}
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

  // the current layout with each pane's own state folded in
  save() {
    const walk = (n) => n.children ? {...n, children: n.children.map(walk)} : {pane: n.pane, state: this.live.get(n)?.pane?.save?.() ?? n.state};
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
    this.model = this.save();
    this.model.tabs.push({name, root});
    this.model.active = this.model.tabs.length - 1;
    this.render();
    this.changed();
  }

  closeTab(i) {
    if (this.model.tabs.length === 1) return;
    this.model = this.save();
    this.model.tabs.splice(i, 1);
    this.model.active = Math.min(this.model.active, this.model.tabs.length - 1);
    this.render();
    this.changed();
  }

  renameTab(i, name) {
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
    const root = this.maxPath ? this.at(this.tab.root, this.maxPath).node : this.tab.root;
    this.el.replaceChildren(this.node(root, null, 0));
  }

  // every pane redraws (after a view switch, when panes may have missed updates while hidden)
  invalidateAll() {
    for (const {pane} of this.live.values()) if (pane) perf.invalidate(pane);
  }

  node(n, parent, index) {
    if (!n.children) return this.paneBox(n, parent, index);
    const box = h("div", {class: `split ${n.split}`});
    n.children.forEach((c, i) => {
      if (i) box.append(this.divider(n, i, box));
      const child = this.node(c, n, i);
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
      const move = (ev) => {
        const d = ((n.split === "row" ? ev.clientX : ev.clientY) - start) / total;
        const da = Math.max(MIN - a, Math.min(b - MIN, d));
        n.sizes[i - 1] = a + da;
        n.sizes[i] = b - da;
        kids[i - 1].style.flex = `${n.sizes[i - 1]} 1 0`;
        kids[i].style.flex = `${n.sizes[i]} 1 0`;
      };
      const up = () => {
        el.removeEventListener("pointermove", move);
        el.removeEventListener("pointerup", up);
        this.changed();
      };
      el.addEventListener("pointermove", move);
      el.addEventListener("pointerup", up);
    });
    el.addEventListener("dblclick", () => {
      n.sizes = n.sizes.map(() => 1 / n.sizes.length);
      this.render();
      this.changed();
    });
    return el;
  }

  paneBox(n, parent, index) {
    const info = PANES[n.pane] || PANES.empty;
    const title = h("span", {class: "pane-title"}, info.title);
    const types = h("select", {class: "pane-type", title: "Pane type", onchange: (e) => this.replace(n, {pane: e.target.value, state: {}})},
      Object.entries(PANES).filter(([, p]) => !p.hidden).map(([k, p]) => h("option", {value: k, selected: k === n.pane}, p.title)));
    const btn = (label, tip, fn) => h("button", {class: "pane-btn", title: tip, onclick: fn}, label);
    const head = h("div", {class: "pane-head"}, types, title,
      h("span", {class: "pane-tools"},
        btn("⇥", "Split right", () => this.split(n, "row")),
        btn("⤓", "Split down", () => this.split(n, "col")),
        btn(this.maxPath ? "⤡" : "⤢", this.maxPath ? "Restore" : "Maximize", () => this.maximize(n)),
        btn("×", "Close pane", () => this.close(n))));
    const body = h("div", {class: "pane-body"});
    const box = h("section", {class: "pane"}, head, body);
    const live = {pane: null, box};
    this.live.set(n, live);
    body._live = live;
    title.hidden = true;
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

  // structural edits all go through the saved model, then re-render
  edit(fn) {
    this.model = this.save();
    fn(this.tab);
    this.maxPath = null;
    this.render();
    this.changed();
  }

  find(target) {
    // map a live node to its counterpart in a freshly saved model by path
    const path = [];
    const search = (n, trail) => {
      if (n === target) { path.push(...trail); return true; }
      return n.children?.some((c, i) => search(c, [...trail, i])) ?? false;
    };
    search(this.tab.root, []);
    return path;
  }

  at(root, path) {
    let parent = null, node = root, index = 0;
    for (const i of path) { parent = node; index = i; node = node.children[i]; }
    return {parent, node, index};
  }

  split(n, dir) {
    const path = this.find(n);
    this.edit((tab) => {
      const {parent, node, index} = this.at(tab.root, path);
      const fresh = {pane: "empty", state: {}};
      if (parent && parent.split === dir) {
        const s = parent.sizes[index] / 2;
        parent.sizes.splice(index, 1, s, s);
        parent.children.splice(index + 1, 0, fresh);
      } else {
        const split = {split: dir, sizes: [0.5, 0.5], children: [node, fresh]};
        if (parent) parent.children[index] = split; else tab.root = split;
      }
    });
  }

  close(n) {
    const path = this.find(n);
    this.edit((tab) => {
      const {parent, index} = this.at(tab.root, path);
      if (!parent) { tab.root = {pane: "empty", state: {}}; return; }
      const freed = parent.sizes[index];
      parent.children.splice(index, 1);
      parent.sizes.splice(index, 1);
      parent.sizes = parent.sizes.map((s) => s + freed / parent.sizes.length);
      if (parent.children.length === 1) {
        // collapse a split with one child into that child
        const only = parent.children[0];
        for (const k of Object.keys(parent)) delete parent[k];
        Object.assign(parent, only);
      }
    });
  }

  // a pane swapping itself for another (the empty pane's picker)
  replaceSelf(el, fresh) {
    for (const [n, live] of this.live) if (live.pane?.el === el) return this.replace(n, fresh);
  }

  replace(n, fresh) {
    const path = this.find(n);
    this.edit((tab) => {
      const {parent, index} = this.at(tab.root, path);
      if (parent) parent.children[index] = fresh; else tab.root = fresh;
    });
  }

  maximize(n) {
    const path = this.find(n);
    this.model = this.save();
    this.maxPath = this.maxPath ? null : path;
    this.render();
  }
}
