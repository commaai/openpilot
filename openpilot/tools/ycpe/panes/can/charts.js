// cabana's charts: tabs of plot-pane grids, one plot per chart. The plot button opens a new chart, dropping signals
// on a chart adds them, and dragging a chart's handle onto another merges the two. Only the open tab has live plots.
import {h} from "/dom.js";
import {create as plot} from "/panes/plot/plot.js";
import {PATHS_MIME} from "/panes/plot/view.js";
import {canOf, subs} from "./common.js";

export function create(ctx, state) {
  const can = canOf(ctx), s = subs();
  let cols = state.cols || 1, charts = [], dragging = null;
  const tabs = state.tabs || [{charts: state.charts || []}];  // the open tab's charts live in `charts`
  let active = Math.min(state.active || 0, tabs.length - 1);
  const tabBar = h("div", {class: "can-tabs"});
  const grid = h("div", {class: "can-charts-grid", ondragover: (e) => e.preventDefault(), ondrop: (e) => {
    const data = e.dataTransfer.getData(PATHS_MIME);
    // drops on a chart are the plot's
    if (data && !e.defaultPrevented) { e.preventDefault(); add({curves: JSON.parse(data).map(curve)}); }
  }});
  const el = h("div", {class: "can-pane can-charts"},
    h("div", {class: "can-toolbar"}, h("span", {class: "can-muted"}, "Drop signals here; drag a chart's handle onto another to merge"),
      h("span", {class: "can-spacer"}),
      h("select", {class: "can-select", title: "Columns", onchange: (e) => { cols = +e.target.value; layout(); }},
        [1, 2, 3].map((n) => h("option", {value: n, selected: n === cols}, `${n} col`))),
      h("button", {class: "can-btn", onclick: () => { for (const c of [...charts]) remove(c); }}, "Clear")), tabBar, grid);
  const resizer = new ResizeObserver((entries) => {
    for (const e of entries) charts.find((c) => c.box === e.target)?.pane.resize(e.contentRect.width, e.contentRect.height);
  });

  // decoded signals keep their DBC color
  const curve = (path) => ({path, color: can.parse(path)?.sig.color});

  function add(st) {
    const c = {pane: plot(ctx, st)};
    c.box = h("div", {class: "can-chart-plot"}, c.pane.el);
    c.el = h("div", {class: "can-chart"},
      h("div", {class: "can-chart-bar", draggable: true, title: "Drag onto another chart to merge",
        ondragstart: (e) => { dragging = c; e.dataTransfer.setData(PATHS_MIME, JSON.stringify(c.pane.save().curves.map((x) => x.path))); },
        // the curves landed on another chart: this one is merged away
        ondragend: (e) => { if (dragging === c && e.dataTransfer.dropEffect !== "none") remove(c); dragging = null; }},
      "⠿", h("span", {class: "can-spacer"}), h("button", {class: "can-icon", title: "Close", onclick: () => remove(c)}, "✕")), c.box);
    // dropped back onto itself: nothing to merge
    c.el.addEventListener("drop", (e) => { if (dragging === c) { e.stopPropagation(); dragging = null; } }, true);
    charts.push(c);
    grid.append(c.el);
    resizer.observe(c.box);
    layout();
  }

  function remove(c) {
    charts = charts.filter((x) => x !== c);
    resizer.unobserve(c.box);
    c.pane.dispose();
    c.el.remove();
  }

  function open(i) {
    tabs[active].charts = charts.map((c) => c.pane.save());
    for (const c of [...charts]) remove(c);
    active = i;
    for (const st of tabs[active].charts) add(st);
    tabsChanged();
  }

  function tabsChanged() {
    tabBar.replaceChildren(...tabs.map((_, i) => h("span", {class: `can-tab${i === active ? " active" : ""}`, onclick: () => i !== active && open(i)}, `Tab ${i + 1}`,
      tabs.length > 1 && h("button", {class: "can-icon", title: "Close tab", onclick: (e) => { e.stopPropagation(); closeTab(i); }}, "✕"))),
      h("button", {class: "can-icon", title: "New tab", onclick: () => { tabs.push({charts: []}); open(tabs.length - 1); }}, "+"));
  }

  function closeTab(i) {
    if (i === active) for (const c of [...charts]) remove(c);
    tabs.splice(i, 1);
    if (i < active) active--;
    else if (i === active) {
      active = Math.min(active, tabs.length - 1);
      for (const st of tabs[active].charts) add(st);
    }
    tabsChanged();
  }

  function layout() {
    grid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
  }

  const pane = {
    name: "can-charts", el,
    render() {},
    dispose() { s.dispose(); resizer.disconnect(); for (const c of charts) c.pane.dispose(); },
    save: () => {
      tabs[active].charts = charts.map((c) => c.pane.save());
      return {cols, tabs, active};
    },
  };
  s.add(can.on("chart", (path) => { if (!charts.some((c) => c.pane.save().curves.some((x) => x.path === path))) add({curves: [curve(path)]}); }));
  for (const st of tabs[active].charts) add(st);
  tabsChanged();
  layout();
  return pane;
}
