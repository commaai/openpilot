// cabana's history log: the selected message's frames up to the playhead, newest first, decoded or hex, filterable.
// Fixed-height rows from a pool; a render only patches the text of visible cells.
import {h} from "/dom.js";
import {format, value} from "/data/dbc.js";
import {fmtTime} from "/panes/plot/draw.js";
import {canOf, hex2, ink, subs} from "./common.js";

const PAGE = 500, ROW = 20, AHEAD = 1;  // frames a request brings; px per row; s past the playhead a request reads

export function create(ctx, state) {
  const can = canOf(ctx), s = subs();
  let hex = state.hex || false, frames = {t: [], data: [], more: false}, busy = false, again = false;
  let cols = [], pool = [], scrollTop = 0, height = 300, shownH = 0;
  const filter = {sig: h("select", {class: "can-select", "aria-label": "Signal"}), op: h("select", {class: "can-select", "aria-label": "Comparison"}, ["=", "!=", ">", ">=", "<", "<="].map((o) => h("option", {}, o))),
                  v: h("input", {class: "can-filter", placeholder: "value", "aria-label": "Value", type: "number", step: "any"})};
  // the value applies as it's typed
  for (const x of [filter.sig, filter.op]) x.addEventListener("change", () => load());
  filter.v.addEventListener("input", () => load());
  const hexBox = h("input", {type: "checkbox", checked: hex, onchange: (e) => { hex = e.target.checked; columns(); }});
  const head = h("div", {class: "can-hist-row can-hist-head"});
  const spacer = h("div", {class: "can-msg-spacer"});
  const empty = h("div", {class: "can-empty"});
  const body = h("div", {class: "can-msg-body", onscroll: () => { scrollTop = body.scrollTop; need(); }}, spacer, empty);
  const el = h("div", {class: "can-pane can-history"},
    h("div", {class: "can-toolbar"}, h("label", {class: "can-check"}, hexBox, "Hex"), h("span", {class: "can-spacer"}), "filter ", filter.sig, filter.op, filter.v), head, body);

  const filters = () => filter.sig.value && filter.v.value !== "" ? [{sig: filter.sig.value, op: filter.op.value, v: +filter.v.value}] : [];

  async function load(append = false) {
    const k = can.selected;
    if (k === null) return;
    if (busy) { again = true; return; }
    busy = true;
    const version = can.dataVersion, t = append && frames.t.length ? frames.t.at(-1) - 1e-7 : ctx.clock.t + AHEAD;
    const r = await can.frames(k, t, PAGE, filters());
    busy = false;
    if (k !== can.selected) return load();
    frames = append ? {...frames, t: [...frames.t, ...r.t], data: [...frames.data, ...r.data], more: r.more} : {...r, hi: t, version};
    ctx.perf.invalidate(pane);
    if (again) { again = false; load(); }
  }

  // the worker is asked only when the rows shown need frames it hasn't sent: the playhead left the span fetched (which
  // reads AHEAD past it, so playback isn't a request a frame), the frames changed, or the list scrolled near its end
  function need() {
    const t = ctx.clock.t;
    if (frames.version !== can.dataVersion || t > frames.hi || frames.more && t < frames.t.at(-1)) return load();
    if (frames.more && !busy && scrollTop + height > (frames.t.length - shown()) * ROW - ROW * 20) load(true);
    ctx.perf.invalidate(pane);
  }

  // the frames after the playhead (read ahead) aren't listed: the index of the newest at or before it
  function shown() {
    const t = ctx.clock.t, ts = frames.t;
    let lo = 0, hi = ts.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (ts[mid] > t) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  // the columns (time, then signals or the hex bytes) change only with the message, DBC or hex toggle
  function columns() {
    const m = can.dbcMsg(can.selected);
    cols = m && !hex ? m.sigs : [];
    head.replaceChildren(h("span", {}, "Time"), ...(cols.length ? cols.map((sg) => {
      const label = sg.unit ? `${sg.name} (${sg.unit})` : sg.name;
      return h("span", {title: label, style: `color: ${ink(sg.color)}`}, label);
    }) : [h("span", {}, "Data")]));
    filter.sig.replaceChildren(h("option", {value: ""}, "signal"), ...(m?.sigs || []).map((sg) => h("option", {value: sg.name, selected: sg.name === filter.sig.value}, sg.name)));
    const tmpl = `80px repeat(${Math.max(1, cols.length)}, minmax(70px, 1fr))`;
    head.style.gridTemplateColumns = tmpl;
    for (const r of pool) r.remove();
    pool = [];
    ctx.perf.invalidate(pane);
  }

  function row() {
    const cells = Array.from({length: 1 + Math.max(1, cols.length)}, (_, i) => h("span", {class: i === 0 || !cols.length ? "can-mono" : ""}));
    const r = h("div", {class: "can-hist-row", onclick: () => ctx.clock.seek(frames.t[r.i])}, cells);
    r.style.gridTemplateColumns = head.style.gridTemplateColumns;
    r.cells = cells;
    spacer.append(r);
    return r;
  }

  const set = (el, text) => { if (el.textContent !== text) el.textContent = text; };

  // hex bytes, colored where they went up or down from the frame before (the next row down)
  function hexCells(r, d, prev) {
    const cell = r.cells[1];
    r.bytes ||= [];
    while (r.bytes.length < d.length) cell.append(r.bytes[r.bytes.push(h("span", {style: "--a: 1"})) - 1]);
    r.bytes.forEach((b, x) => {
      if (b.hidden !== x >= d.length) b.hidden = x >= d.length;
      if (x >= d.length) return;
      set(b, hex2(d[x]));
      const cls = prev && x < prev.length && d[x] !== prev[x] ? (d[x] > prev[x] ? "can-chg1" : "can-chg2") : "";
      if (b.className !== cls) b.className = cls;
    });
  }

  const pane = {
    name: "can-history", el,
    render() {
      const b = shown(), count = frames.t.length - b;
      set(empty, can.selected === null ? "Select a message to see its history" : count || busy ? "" : filters().length ? "No frames match the filter" : "No frames before the playhead");
      if (empty.hidden !== !empty.textContent) empty.hidden = !empty.textContent;
      if (shownH !== count * ROW) spacer.style.height = `${shownH = count * ROW}px`;
      const m = can.dbcMsg(can.selected), first = Math.floor(scrollTop / ROW), n = Math.ceil(height / ROW) + 1;
      for (let i = 0; i < Math.max(n, pool.length); i++) {
        const j = first + i, f = b + j, r = pool[i] || (pool[i] = row()), on = i < n && j < count;
        if (r.hidden !== !on) r.hidden = !on;
        if (!on) continue;
        r.i = f;
        if (r.top !== j) r.style.top = `${(r.top = j) * ROW}px`;
        set(r.cells[0], fmtTime(frames.t[f], 0.001, true));
        if (!cols.length) hexCells(r, frames.data[f], frames.data[f + 1]);
        else cols.forEach((sg, c) => {
          const v = value(m, sg, frames.data[f]);
          set(r.cells[c + 1], Number.isNaN(v) ? "–" : format(sg, v, false));
        });
      }
    },
    dispose() { s.dispose(); },
    save: () => ({hex}),
  };
  // the list's own height: the toolbar above it wraps on a narrow pane, so the pane's size doesn't tell it
  const sizer = new ResizeObserver(([e]) => { height = e.contentRect.height; ctx.perf.invalidate(pane); });
  sizer.observe(body);
  s.add(() => sizer.disconnect());
  s.add(can.on("select", () => { frames = {t: [], data: [], more: false}; body.scrollTop = scrollTop = 0; columns(); load(); }));
  s.add(can.on("dbc", columns));
  s.add(ctx.clock.on("t", need));
  columns();
  load();
  return pane;
}
