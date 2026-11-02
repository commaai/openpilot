// what one pane asks of the others, without knowing which are open:
//   plot(ctx, paths, from)       the series onto the plot pane last used, else the layout's first
//   chart(ctx, curve, merge, from)  as cabana's plot button: curve on a plot of its own, split off below the last on
//                                screen (merge: onto the plot last used, as plot()), or off the plot that has it
//   charted(ctx)                 the paths the plots on screen show
//   reveal(ctx, path, t, from)   the clock to t (the camera and map follow it), and where path comes from shown: a DBC
//                                signal's message in the CAN panes, a log field in its message in the inspector
// A layout without the pane to do it opens one beside the pane asking (from: an element in it); the CAN panes are a
// view of their own. offer() is the context menu of the verbs that apply, and items() the same for a menu of a pane's own
import {fmt, menuList, popover} from "/dom.js";
import * as view from "/panes/plot/view.js";

export function plot(ctx, paths, from) {
  const plots = onScreen(ctx, (type) => type === "plot");
  const p = plots.includes(view.lastTouched()) ? view.lastTouched() : plots[0];
  if (p) p.addCurves(paths);
  else openBeside(ctx, from, {pane: "plot", state: {curves: paths.map((path) => ({path}))}});
}

export function chart(ctx, curve, merge, from) {
  const plots = shown(ctx, (type) => type === "plot"), panes = plots.map(([, p]) => p);
  const i = panes.findIndex((p) => p.curves.some((c) => c.path === curve.path));
  if (i >= 0) {
    const rest = panes[i].curves.filter((c) => c.path !== curve.path);
    // an emptied plot goes, unless it's the last
    if (!rest.length && panes.length > 1) return ctx.layout.close(plots[i][0]);
    view.rememberCurves(ctx.clock, panes[i], "remove series");
    return panes[i].setCurves(rest);
  }
  const fresh = {pane: "plot", state: {curves: [curve]}};
  const into = merge ? (panes.includes(view.lastTouched()) ? view.lastTouched() : panes.at(-1)) : panes.find((p) => !p.curves.length);
  if (into) {
    view.rememberCurves(ctx.clock, into, "add series");
    into.setCurves([...into.curves, curve]);
  } else if (plots.length) ctx.layout.split(plots.at(-1)[0], "col", fresh);
  else openBeside(ctx, from, fresh);
}

export const charted = (ctx) => new Set(onScreen(ctx, (type) => type === "plot").flatMap((p) => p.curves.map((c) => c.path)));

export function reveal(ctx, path, t, from) {
  if (t !== undefined) ctx.clock.seek(t);
  if (path.startsWith("/dbc/")) return showInCan(ctx, path);
  const [inspector] = onScreen(ctx, (type) => type === "inspector");
  if (inspector) inspector.reveal(path);
  else openBeside(ctx, from, {pane: "inspector", state: {path}});
}

// the signal's message selected and the signal open in the CAN panes, in the CAN view when the layout has none
function showInCan(ctx, path) {
  const can = ctx.store.can, sig = can?.parse(path);
  if (!sig) return;
  can.select(sig.k, sig.sig.name);
  can.emit("edit", sig.sig.name);  // for a signals pane that had the message already
  if (!onScreen(ctx, (type) => type.startsWith("can-")).length) ctx.setView("can");
}

// a log field is in the inspector, a DBC signal in the CAN panes; computed series are nowhere
const revealable = (path) => !/^\/custom\/|\[/.test(path);

// a series as a menu names it: a DBC signal as MESSAGE.SIGNAL
const named = (path) => path.startsWith("/dbc/") ? path.split("/").slice(3).join(".") : path.slice(1);

// menu items: paths to plot, path at time t to reveal
export function items(ctx, {paths = [], path, t}, from) {
  const at = t === undefined ? "" : ` at ${fmt.time(t, 0.001, true)}`;
  return [
    paths.length && {label: paths.length > 1 ? `Plot ${paths.length} series` : `Plot ${named(paths[0])}`, fn: () => plot(ctx, paths, from)},
    path && revealable(path) && {label: (path.startsWith("/dbc/") ? `Show ${named(path)} in CAN` : `Inspect ${named(path)}`) + at,
      fn: () => reveal(ctx, path, t, from)},
  ].filter(Boolean);
}

// for a contextmenu event
export function offer(ctx, e, what) {
  const list = items(ctx, what, e.currentTarget);
  if (!list.length) return;
  e.preventDefault();
  popover({x: e.clientX, y: e.clientY}, () => menuList(list));
}

// the layout's panes whose type passes is, on screen, in reading order, as [node, pane]
function shown(ctx, is) {
  return [...ctx.layout?.live ?? []].filter(([n, l]) => l.pane && l.box.isConnected && is(n.pane)).map(([n, l]) => [n, l.pane]);
}

const onScreen = (ctx, is) => shown(ctx, is).map(([, p]) => p);

// a pane split off the one from is in, along its longer side
function openBeside(ctx, from, fresh) {
  const [node, live] = [...ctx.layout?.live ?? []].find(([, l]) => l.box.contains(from)) ?? [];
  if (node) ctx.layout.split(node, live.box.clientWidth >= live.box.clientHeight ? "row" : "col", fresh);
}
