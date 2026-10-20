// cabana's history log: the selected message's frames up to the playhead, newest first, decoded or hex, filterable
import {h} from "/dom.js";
import {format, value} from "/data/dbc.js";
import {canOf, hexBytes, subs} from "./common.js";

const PAGE = 200;

export function create(ctx, state) {
  const can = canOf(ctx), s = subs();
  let hex = state.hex || false, frames = {t: [], data: [], more: false}, busy = false, again = false;
  const filter = {sig: h("select", {class: "can-select"}), op: h("select", {class: "can-select"}, ["=", "!=", ">", ">=", "<", "<="].map((o) => h("option", {}, o))),
                  v: h("input", {class: "can-filter", placeholder: "value", type: "number", step: "any"})};
  for (const x of Object.values(filter)) x.addEventListener("change", () => load());
  const hexBox = h("input", {type: "checkbox", checked: hex, onchange: (e) => { hex = e.target.checked; ctx.perf.invalidate(pane); }});
  const table = h("div", {class: "can-table can-history-table"});
  const more = h("button", {class: "can-btn", onclick: () => load(true)}, "Load more");
  const scroll = h("div", {class: "can-history-scroll"}, table, more);
  const el = h("div", {class: "can-pane can-history"},
    h("div", {class: "can-toolbar"}, h("label", {class: "can-check"}, hexBox, "Hex"), h("span", {class: "can-spacer"}), "filter ", filter.sig, filter.op, filter.v), scroll);

  const filters = () => filter.sig.value && filter.v.value !== "" ? [{sig: filter.sig.value, op: filter.op.value, v: +filter.v.value}] : [];

  async function load(append = false) {
    const k = can.selected;
    if (k === null) return;
    if (busy) { again = true; return; }
    busy = true;
    const t = append && frames.t.length ? frames.t.at(-1) - 1e-7 : ctx.clock.t;
    const r = await can.frames(k, t, PAGE, filters());
    busy = false;
    if (k !== can.selected) return load();
    frames = append ? {t: [...frames.t, ...r.t], data: [...frames.data, ...r.data], more: r.more} : r;
    ctx.perf.invalidate(pane);
    if (again) { again = false; load(); }
  }

  function signals() {
    const m = can.dbcMsg(can.selected);
    const opts = [h("option", {value: ""}, "signal"), ...(m?.sigs || []).map((sg) => h("option", {value: sg.name, selected: sg.name === filter.sig.value}, sg.name))];
    filter.sig.replaceChildren(...opts);
  }

  function draw() {
    const m = can.dbcMsg(can.selected), decoded = m && !hex;
    const cols = ["Time", ...(decoded ? m.sigs.map((sg) => sg.name) : ["Data"])];
    table.replaceChildren(h("div", {class: "can-table-head"}, cols.map((c) => h("span", {}, c))),
      ...frames.t.map((t, i) => h("div", {class: "can-table-row", onclick: () => ctx.clock.seek(t)},
        h("span", {class: "can-mono"}, t.toFixed(3)),
        decoded ? m.sigs.map((sg) => {
          const v = value(m, sg, frames.data[i]);
          return h("span", {}, Number.isNaN(v) ? "–" : format(sg, v, false));
        }) : h("span", {class: "can-mono"}, hexBytes(frames.data[i])))));
    more.hidden = !frames.more;
  }

  const pane = {
    name: "can-history", el,
    render: draw,
    dispose() { s.dispose(); },
    save: () => ({hex}),
  };
  s.add(can.on("select", () => { frames = {t: [], data: [], more: false}; signals(); load(); }));
  s.add(can.on("dbc", () => { signals(); ctx.perf.invalidate(pane); }));
  s.add(ctx.clock.on("t", () => load()));
  signals();
  load();
  return pane;
}
