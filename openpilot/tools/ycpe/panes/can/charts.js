// cabana's charts: tabs of plot-pane grids, one plot per chart. The plot button or the signal picker opens a chart,
// dropping signals on a chart adds them, and dragging a chart's handle onto another merges the two (or between charts,
// moves it). Only the open tab has live plots. Charts show the shared view range, or, as cabana, a window that pages
// along with the playhead.
import {h} from "/dom.js";
import {icon} from "/icons.js";
import {create as plot} from "/panes/plot/plot.js";
import {PATHS_MIME} from "/panes/plot/view.js";
import {canOf, dialog, ink, subs} from "./common.js";

export function create(ctx, state) {
  const can = canOf(ctx), s = subs();
  let cols = state.cols || 1, style = state.style || "", span = state.span || 0, charts = [], dragging = null, shown = null;
  const tabs = state.tabs || [{charts: state.charts || []}];  // the open tab's charts live in `charts`
  let active = Math.min(state.active || 0, tabs.length - 1);
  // the tabs, then the new tab button outside the tablist
  const tabBar = h("div", {class: "can-tablist", role: "tablist"});
  const newTab = h("button", {class: "can-icon", title: "New tab", onclick: () => { tabs.push({charts: []}); open(tabs.length - 1); }}, icon("add", 14));
  const empty = h("div", {class: "can-empty"}, "Plot a signal from the signal list, or drop signals here");
  const grid = h("div", {class: "can-charts-grid", ondragover: (e) => e.preventDefault(), ondrop: (e) => {
    const data = e.dataTransfer.getData(PATHS_MIME);
    // drops on a chart are the plot's; a chart's handle dropped anywhere else moves the chart there
    if (!data || e.defaultPrevented) return;
    e.preventDefault();
    if (dragging) move(dragging, e); else add({curves: JSON.parse(data).map(curve)});
  }}, empty);
  const select = (title, options, value, fn) => h("select", {class: "can-select", title, "aria-label": title.split(":")[0], onchange: (e) => fn(e.target.value)},
    options.map(([v, n]) => h("option", {value: v, selected: v === value}, n)));
  const el = h("div", {class: "can-pane can-charts"},
    h("div", {class: "can-toolbar"},
      h("button", {class: "can-btn", title: "New chart from a list of signals", onclick: () => picker((paths) => add({curves: paths.map(curve)}))}, icon("add", 14), h("span", {}, "Chart")),
      h("span", {class: "grow"}),
      select("Time range: the view's, or a window following the playhead", [["0", "view"], ...[1, 5, 15, 30].map((x) => [String(x), `${x} s`])], String(span),
        (v) => { span = +v; shown = null; follow(); }),
      select("Series type of every chart", [["", "auto"], ["line", "line"], ["steps", "step"], ["dots", "scatter"]], style, restyle),
      select("Columns", [1, 2, 3].map((n) => [String(n), `${n} col`]), String(cols), (v) => { cols = +v; layout(); }),
      h("button", {class: "can-btn", title: "Remove every chart in this tab", onclick: () => { for (const c of [...charts]) remove(c); }}, "Clear")),
    h("div", {class: "can-tabs"}, tabBar, newTab), grid);
  const resizer = new ResizeObserver((entries) => {
    for (const e of entries) charts.find((c) => c.box === e.target)?.pane.resize(e.contentRect.width, e.contentRect.height);
  });
  // a plot rebuilds its legend whenever its curves change (a drop, the legend's x, its editor or undo)
  const legends = new MutationObserver(charted);

  // decoded signals keep their DBC color, darkened on the light theme's white
  const color = (path) => {
    const c = can.parse(path)?.sig.color;
    return c && ctx.theme.name === "light" ? ink(c) : c;
  };
  const curve = (path) => ({path, color: color(path), ...(style && {style})});

  function add(st) {
    const c = {pane: plot(ctx, st)};
    c.box = h("div", {class: "can-chart-plot"}, c.pane.el);
    const paths = () => c.pane.save().curves.map((x) => x.path);
    c.el = h("div", {class: "can-chart"},
      h("div", {class: "can-chart-bar", draggable: true, title: "Drag onto another chart to merge, or between charts to move",
        ondragstart: (e) => { dragging = c; e.dataTransfer.setData(PATHS_MIME, JSON.stringify(paths())); },
        // the curves landed on another chart: this one is merged away
        ondragend: (e) => { if (dragging === c && e.dataTransfer.dropEffect !== "none") remove(c); dragging = null; }},
      icon("drag_indicator", 14), h("span", {class: "grow"}),
      h("button", {class: "can-icon", title: "Add or remove signals", onclick: () => picker((p) => replace(c, {...c.pane.save(), curves: p.map(curve)}), paths())}, icon("edit", 12)),
      h("button", {class: "can-icon", title: "Split into one chart per signal", onclick: () => split(c)}, icon("splitRight", 12)),
      h("button", {class: "can-icon", title: "Remove chart", onclick: () => remove(c)}, icon("close", 12))), c.box);
    // dropped back onto itself: nothing to merge
    c.el.addEventListener("drop", (e) => { if (dragging === c) { e.stopPropagation(); dragging = null; } }, true);
    charts.push(c);
    grid.append(c.el);
    resizer.observe(c.box);
    legends.observe(c.pane.el.querySelector(".legend"), {childList: true});
    if (shown) c.pane.setView({...c.pane.view, range: shown});
    layout();
    return c;
  }

  function remove(c) {
    charts = charts.filter((x) => x !== c);
    resizer.unobserve(c.box);
    c.pane.dispose();
    c.el.remove();
    layout();
  }

  // a chart rebuilt from a new state, in its place
  function replace(c, st) {
    const at = charts.indexOf(c), next = c.el.nextSibling;
    remove(c);
    if (!st.curves.length) return;
    const n = add(st);
    charts.splice(charts.indexOf(n), 1);
    charts.splice(at, 0, n);
    grid.insertBefore(n.el, next);
  }

  // before the first chart whose middle is past the drop, in reading order; the chart stays as it is, so dragend
  // doesn't remove it
  function move(c, e) {
    const before = charts.find((x) => {
      const r = x.el.getBoundingClientRect();
      return x !== c && (e.clientY < r.top || (e.clientY < r.bottom && (cols > 1 ? e.clientX < r.left + r.width / 2 : e.clientY < r.top + r.height / 2)));
    });
    charts.splice(charts.indexOf(c), 1);
    charts.splice(before ? charts.indexOf(before) : charts.length, 0, c);
    grid.insertBefore(c.el, before?.el || null);
    dragging = null;
    layout();
  }

  function split(c) {
    const st = c.pane.save();
    if (st.curves.length < 2) return;
    replace(c, {...st, curves: st.curves.slice(0, 1)});
    for (const x of st.curves.slice(1)) add({curves: [x]});
  }

  // the open tab's charts into its saved state
  const stash = () => { tabs[active].charts = charts.map((c) => c.pane.save()); };
  // every saved curve of every tab
  const curves = () => tabs.flatMap((t) => t.charts.flatMap((st) => st.curves));

  // the open tab's charts saved, changed by fn, and rebuilt
  function rebuild(fn = () => {}) {
    stash();
    for (const c of [...charts]) remove(c);
    fn();
    for (const st of tabs[active].charts) add(st);
    layout();  // for an empty tab too
  }

  const open = (i) => rebuild(() => { active = i; });

  const closeTab = (i) => rebuild(() => {
    tabs.splice(i, 1);
    if (i < active || active === tabs.length) active--;
  });

  function tabsChanged() {
    const count = (t, i) => (i === active ? charts : t.charts).length;
    tabBar.replaceChildren(...tabs.map((t, i) => h("span", {class: `can-tab${i === active ? " active" : ""}`},
      h("button", {class: "can-tab-name", role: "tab", "aria-selected": String(i === active), onclick: () => i !== active && open(i)},
        `Tab ${i + 1}`, count(t, i) > 0 && h("span", {class: "muted"}, ` · ${count(t, i)}`)),
      tabs.length > 1 && h("button", {class: "can-icon", title: "Close tab", onclick: () => closeTab(i)}, icon("close", 12)))));
  }

  // cabana's chart type: one series style for every chart, set on their curves
  function restyle(v) {
    style = v;
    rebuild(() => { for (const c of curves()) c.style = style || undefined; });
  }

  // cabana's chart range: a window of `span` seconds that pages forward once the playhead passes 80% of it
  function follow() {
    let range = null;
    if (span) {
      const t = ctx.clock.t, [lo, hi] = [0, ctx.clock.duration || Infinity];
      let t0 = shown ? shown[0] : t - span * 0.1;
      if (t < t0 || t > t0 + span * 0.8) t0 = t - span * 0.1;
      t0 = Math.max(lo, Math.min(t0, hi - span));
      range = [t0, t0 + span];
    }
    if (range?.[0] === shown?.[0] && range?.[1] === shown?.[1]) return;
    shown = range;
    for (const c of charts) c.pane.setView({...c.pane.view, range});
  }

  function layout() {
    grid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
    empty.hidden = charts.length > 0;
    tabsChanged();
    charted();
  }

  // the plot buttons show what the open tab charts
  function charted() {
    can.charted = new Set(charts.flatMap((c) => c.pane.save().curves.map((x) => x.path)));
    can.emit("charted");
  }

  // cabana's signal selector: every decoded signal, filtered, picked by checkbox
  function picker(done, chosen = []) {
    const picked = new Set(chosen), all = can.paths().map((p) => p.path);
    const filter = h("input", {class: "can-filter", placeholder: "Filter signals", spellcheck: false, oninput: () => list()});
    const out = h("div", {class: "can-list can-picker"});
    const count = h("span", {class: "muted"});
    function list() {
      const q = filter.value.toLowerCase(), hits = all.filter((p) => p.toLowerCase().includes(q));
      out.replaceChildren(...hits.slice(0, 500).map((p) => h("label", {class: "can-list-row"},
        h("input", {type: "checkbox", checked: picked.has(p), onchange: (e) => { if (e.target.checked) picked.add(p); else picked.delete(p); count.textContent = `${picked.size} selected`; }}),
        ` ${p.slice(5)}`)), hits.length > 500 ? h("div", {class: "muted"}, `${hits.length - 500} more, filter to see them`) : "");
      count.textContent = `${picked.size} selected`;
    }
    const d = dialog(chosen.length ? "Chart signals" : "New chart", filter, out,
      h("div", {class: "can-row"}, count, h("span", {class: "grow"}),
        h("button", {class: "can-btn primary", onclick: () => { d.close(); if (picked.size || chosen.length) done([...picked]); }}, chosen.length ? "Update" : "Add chart")));
    list();
    filter.focus();
  }

  const pane = {
    name: "can-charts", el,
    render() {},
    dispose() {
      s.dispose();
      resizer.disconnect();
      legends.disconnect();
      for (const c of charts) c.pane.dispose();
      can.charted = new Set();
      can.emit("charted");
    },
    save: () => {
      stash();
      return {cols, style, span, tabs, active};
    },
  };
  // a signal's plot button charts it (shift: on the first chart, as cabana), or takes it off its chart
  s.add(can.on("chart", (path, {merge = false} = {}) => {
    const c = charts.find((x) => x.pane.save().curves.some((y) => y.path === path));
    if (!c && merge && charts.length) return charts[0].pane.setCurves([...charts[0].pane.curves, curve(path)]);
    if (!c) return add({curves: [curve(path)]});
    const st = c.pane.save();
    replace(c, {...st, curves: st.curves.filter((x) => x.path !== path)});
  }));
  // curves follow a renamed signal or message, and go with a removed one
  s.add(can.on("moved", (move) => {
    stash();
    if (curves().every((x) => move(x.path) === x.path)) return;
    const moved = (st) => ({...st, curves: st.curves.flatMap((x) => (move(x.path) ? [{...x, path: move(x.path)}] : []))});
    rebuild(() => { for (const t of tabs) t.charts = t.charts.map(moved).filter((st) => st.curves.length); });
  }));
  s.add(ctx.theme.on(() => rebuild(() => { for (const c of curves()) c.color = color(c.path) || c.color; })));
  s.add(ctx.clock.on("t", follow));
  follow();
  for (const st of tabs[active].charts) add(st);
  layout();
  return pane;
}
