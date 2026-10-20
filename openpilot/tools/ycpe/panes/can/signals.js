// the selected message's signals: values at the playhead, sparklines, and the signal and message editors
import {h} from "/dom.js";
import {icon} from "/icons.js";
import {MUX, MUXED, NORMAL, format, value} from "/data/dbc.js";
import {busOf, hexId} from "/data/can.js";
import {canOf, dialog, editMsg, field, removeSignal, subs, updateSignal} from "./common.js";

const SPARK_W = 80;  // sparkline width

export function create(ctx, state) {
  const can = canOf(ctx), s = subs();
  let open = state.open || null, filter = "", spark = state.spark || 15;  // seconds of history, cabana's default
  // the toolbar is built once and updated, so the filter box keeps focus while the list rebuilds
  const title = h("span", {class: "can-binary-title"}), meta = h("span", {class: "can-muted"}), count = h("span", {class: "can-muted"});
  const sparkSel = h("select", {class: "can-select", title: "Sparkline range", onchange: (e) => { spark = +e.target.value; requestSparks(); }},
    [1, 5, 15, 30].map((x) => h("option", {value: x, selected: x === spark}, `${x} s`)));
  const filterBox = h("input", {class: "can-filter", placeholder: "Filter signals", oninput: (e) => { filter = e.target.value; build(); }});
  const editBtn = h("button", {class: "can-btn", title: "Edit message", onclick: () => messageDialog(can, k())});
  const removeBtn = h("button", {class: "can-btn", title: "Remove message", onclick: () => {
    const m = msg();
    can.dbcs.edit(busOf(k()), m.address, null, `remove message ${m.name}`);
  }}, "Remove message");
  const comment = h("div", {class: "can-comment"});
  const head = h("div", {class: "can-toolbar"}, title, meta, h("span", {class: "can-spacer"}), count, sparkSel, filterBox, editBtn, removeBtn, comment);
  const list = h("div", {class: "can-sig-list"});
  const el = h("div", {class: "can-pane can-signals"}, head, list);
  const k = () => can.selected;
  const msg = () => can.dbcMsg(k());
  const path = (sg) => `/dbc/${busOf(k())}/${msg().name}/${sg.name}`;
  let rows = new Map(), mux;  // signal name -> {row, v, spark, swatch, name, ed, sg}; the multiplexor they were built with
  let sparks = null, sparkBusy = false;

  // the sparklines' buckets for the window before the playhead, one request in flight
  async function requestSparks() {
    const key = k(), m = msg();
    if (sparkBusy || !m) return;
    sparkBusy = true;
    const t = ctx.clock.t, names = m.sigs.map((sg) => sg.name);
    const r = await can.decimate(key, names, t - spark, t, SPARK_W);
    sparkBusy = false;
    if (key !== k()) return;
    sparks = Object.fromEntries(names.map((n, i) => [n, r[i]]));
    ctx.perf.invalidate(pane);
  }

  function header() {
    const key = k(), m = msg();
    head.hidden = key === null;  // the empty list says to select one
    title.textContent = key === null ? "" : `${hexId(key)} ${m?.name || "(no DBC message)"}`;
    meta.textContent = m ? ` ${m.size} bytes · ${m.transmitter}` : "";
    count.textContent = m ? `${m.sigs.length} signals` : "";
    for (const x of [meta, count, sparkSel, filterBox, removeBtn]) x.hidden = !m;
    editBtn.textContent = m ? "Edit message" : "Add to DBC";
    comment.textContent = m?.comment || "";
    comment.hidden = !m?.comment;
  }

  function build() {
    rows = new Map();
    const key = k(), m = msg();
    header();
    if (!m) {
      list.replaceChildren(h("div", {class: "can-empty"}, key === null ? "Select a message to edit its signals" : "Drag across bits in the binary view to create a signal."));
      return;
    }
    list.replaceChildren(...shown(m).map((sg) => {
      const v = h("span", {class: "can-sig-value"}), spark = h("canvas", {class: "can-spark"});
      const swatch = h("span", {class: "can-swatch"}), name = h("span", {class: "can-sig-name"}), ed = sg.name === open && editor(can, k(), m, sg);
      const plot = h("button", {class: "can-icon", title: "Plot", onclick: (e) => { e.stopPropagation(); can.emit("chart", path(sg)); }}, icon("show_chart", 14));
      const row = h("div", {class: `can-sig ${sg.name === open ? "open" : ""}`, onmouseenter: () => can.emit("hover", sg.name), onmouseleave: () => can.emit("hover", null)},
        h("div", {class: "can-sig-row", role: "button", tabindex: 0, "aria-expanded": String(sg.name === open), onclick: () => toggle(sg.name),
          onkeydown: (e) => { if (e.target === e.currentTarget && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); toggle(sg.name, true); } }},
          swatch, name, spark, v,
          plot,
          h("button", {class: "can-icon", title: "Remove signal", onclick: (e) => { e.stopPropagation(); removeSignal(can, k(), sg.name); }}, icon("close", 12))),
        ed);
      row.draggable = true;
      row.addEventListener("dragstart", (e) => e.dataTransfer.setData("application/x-ycpe-paths", JSON.stringify([path(sg)])));
      rows.set(sg.name, label({row, v, spark, swatch, name, ed, plot}, sg));
      return row;
    }));
    mux = m.mux?.name;
    charted();
    ctx.perf.invalidate(pane);
  }

  // opens or closes a signal's editor; from the keyboard, its row keeps focus
  function toggle(name, focus = false) {
    open = open === name ? null : name;
    build();
    if (focus) rows.get(name)?.row.firstChild.focus();
  }

  function charted() {
    for (const r of rows.values()) {
      const on = !!can.charted?.has(path(r.sg));
      r.plot.classList.toggle("on", on);
      r.plot.setAttribute("aria-pressed", on);
      r.plot.title = on ? "Remove from the charts" : "Plot";
    }
  }

  const shown = (m) => m.sigs.filter((sg) => sg.name.toLowerCase().includes(filter.toLowerCase()));

  function label(r, sg) {
    r.sg = sg;
    r.swatch.style.background = sg.color;
    r.name.title = sg.comment || sg.name;
    r.name.textContent = sg.name + (sg.type === MUX ? " (M)" : sg.type === MUXED ? ` (m${sg.muxValue})` : "");
    return r;
  }

  // an edit that leaves the same signals listed updates them in place, so the editor keeps focus as Tab or a click
  // moves to its next field
  function patch() {
    const m = msg(), sigs = m ? shown(m) : [], names = [...rows.keys()], ed = rows.get(open)?.ed;
    if (!m || sigs.length !== names.length || sigs.some((sg, i) => sg.name !== names[i])) return false;
    const sg = sigs.find((x) => x.name === open);
    if (ed && (sg.type !== rows.get(open).sg.type || m.mux?.name !== mux)) return false;
    header();
    for (const x of sigs) label(rows.get(x.name), x);
    charted();
    if (ed) ed.update(sg);
    ctx.perf.invalidate(pane);
    return true;
  }

  const pane = {
    name: "can-signals", el,
    render() {
      const m = msg(), live = can.live, j = live?.index.get(k());
      const d = j === undefined ? null : live.data.subarray(live.off[j], live.off[j + 1]);
      for (const [name, {v, spark, sg}] of rows) {
        const x = d ? value(m, sg, d) : NaN, text = Number.isNaN(x) ? "–" : format(sg, x);
        if (v.textContent !== text) v.textContent = text;
        sparkline(spark, sparks?.[name], sg.color);
      }
    },
    resize() { ctx.perf.invalidate(pane); },
    dispose() { s.dispose(); },
    save: () => ({open, spark}),
  };
  s.add(can.on("select", () => { open = null; sparks = null; build(); requestSparks(); }));
  s.add(can.on("dbc", () => { if (!patch()) build(); requestSparks(); }));
  s.add(can.on("edit", (name) => { open = name; build(); rows.get(name)?.row.scrollIntoView({block: "nearest"}); }));
  s.add(can.on("charted", charted));
  s.add(can.on("live", () => { requestSparks(); ctx.perf.invalidate(pane); }));
  s.add(can.on("hover", (name) => { for (const [n, r] of rows) r.row.classList.toggle("hover", n === name); }));
  build();
  return pane;
}

// the last few seconds of a signal, from worker buckets: a fixed-size canvas, so no layout reads
function sparkline(canvas, d, color) {
  const dpr = devicePixelRatio, w = SPARK_W, hgt = 18;
  if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(hgt * dpr); }
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, hgt);
  if (!d) return;
  let min = Infinity, max = -Infinity;
  for (let x = 0; x < w; x++) { if (d.min[x] < min) min = d.min[x]; if (d.max[x] > max) max = d.max[x]; }
  if (min === Infinity) return;
  const y = (v) => max === min ? hgt / 2 : hgt - 2 - (v - min) / (max - min) * (hgt - 4);
  g.strokeStyle = color.replace(/(\d+(\.\d+)?)%\)$/, "45%)");
  g.beginPath();
  let started = false;
  for (let x = 0; x < w; x++) {
    if (d.first[x] !== d.first[x]) continue;
    if (started) g.lineTo(x, y(d.first[x])); else { g.moveTo(x, y(d.first[x])); started = true; }
    g.lineTo(x, y(d.min[x]));
    g.lineTo(x, y(d.max[x]));
    g.lineTo(x, y(d.last[x]));
  }
  g.stroke();
}

// every signal property; changes commit on change/enter through the undo stack. update(sg) shows a new version of
// the signal in place, but for the field being edited.
function editor(can, k, m, sg) {
  const commit = (props) => updateSignal(can, k, sg.name, props);
  // size, start bit and mux value are whole numbers in range, or the field reverts
  const ints = {size: () => [1, 64], start: () => [0, can.dbcMsg(k).size * 8 - 1], muxValue: () => [0, Infinity]};
  const num = (prop, attrs = {}) => h("input", {type: "number", "data-prop": prop, value: sg[prop], step: "any", ...attrs, onchange: (e) => {
    const v = parseFloat(e.target.value), [lo, hi] = ints[prop]?.() || [-Infinity, Infinity];
    if (Number.isFinite(v) && v >= lo && v <= hi && (!ints[prop] || Number.isInteger(v))) commit({[prop]: v});
    else e.target.value = sg[prop];
  }});
  const text = (prop) => h("input", {"data-prop": prop, value: sg[prop], spellcheck: false, onchange: (e) => commit({[prop]: e.target.value.trim()})});
  const check = (prop) => h("input", {type: "checkbox", "data-prop": prop, checked: sg[prop], onchange: (e) => commit({[prop]: e.target.checked})});
  const type = h("select", {"data-prop": "type", onchange: (e) => commit({type: +e.target.value})},
    [[NORMAL, "normal"], [MUX, "multiplexor"], [MUXED, "multiplexed"]].map(([v, n]) =>
      h("option", {value: v, selected: sg.type === v, disabled: (v === MUXED && !m.mux) || (v === MUX && m.mux && m.mux !== sg)}, n)));
  const valsText = (x) => x.vals.map(([v, d]) => `${v} "${d}"`).join(" ");
  const vals = h("input", {value: valsText(sg), spellcheck: false, placeholder: '0 "off" 1 "on"',
    onchange: (e) => commit({vals: [...e.target.value.matchAll(/(-?[\d.]+)\s+"([^"]*)"/g)].map(([, v, d]) => [+v, d])})});
  const error = h("div", {class: "can-error"});
  const el = h("div", {class: "can-sig-editor"},
    field("Name", h("input", {value: sg.name, spellcheck: false, onchange: (e) => {
      const name = e.target.value.trim();
      error.textContent = !/^\w+$/.test(name) ? "Names are letters, digits and underscores"
        : name !== sg.name && !commit({name}) ? "There is already a signal with the same name" : "";
      if (error.textContent) e.target.value = sg.name;
      else if (name !== sg.name) can.emit("edit", name);
    }})),
    error,
    field("Size", num("size", {min: 1, max: 64, step: 1})),
    field("Start bit", num("start", {min: 0, max: 511, step: 1})),
    h("div", {class: "can-field"}, h("span"), h("span", {class: "can-row"}, h("label", {class: "can-check"}, check("le"), "Little endian"), h("label", {class: "can-check"}, check("signed"), "Signed"))),
    field("Factor", num("factor")), field("Offset", num("offset")),
    field("Min", num("min")), field("Max", num("max")),
    field("Unit", text("unit")), field("Receiver", text("receiver")),
    field("Type", type),
    sg.type === MUXED && field("Mux value", num("muxValue", {min: 0, step: 1})),
    field("Comment", h("textarea", {"data-prop": "comment", rows: 2, onchange: (e) => commit({comment: e.target.value.trim()})}, sg.comment)),
    field("Values", vals, 'value descriptions: 0 "off" 1 "on"'));
  el.update = (next) => {
    sg = next;
    for (const x of el.querySelectorAll("[data-prop]")) {
      if (x === document.activeElement) continue;
      if (x.type === "checkbox") x.checked = sg[x.dataset.prop]; else x.value = sg[x.dataset.prop];
    }
    if (vals !== document.activeElement) vals.value = valsText(sg);
  };
  return el;
}

function messageDialog(can, k) {
  const m = can.dbcMsg(k);
  const name = h("input", {value: m?.name || `NEW_MSG_${(k % 2 ** 32).toString(16).toUpperCase()}`, spellcheck: false});
  const size = h("input", {type: "number", min: 1, max: 64, value: m?.size || can.msg(k)?.size || 8});
  const node = h("input", {value: m?.transmitter || "XXX", spellcheck: false});
  const comment = h("textarea", {rows: 3}, m?.comment || "");
  const error = h("div", {class: "can-error"});
  const d = dialog(m ? `Edit ${m.name}` : "New message", h("div", {class: "can-form"},
    field("Name", name), field("Size", size), field("Node", node), field("Comment", comment), error,
    h("button", {class: "can-btn primary", onclick: () => {
      if (!/^\w+$/.test(name.value)) { error.textContent = "Names are letters, digits and underscores"; return; }
      if (!Number.isInteger(+size.value) || +size.value < 1 || +size.value > 64) { error.textContent = "Size is a whole number of bytes, 1 to 64"; return; }
      const other = can.dbcs.file(busOf(k))?.msgByName(name.value);
      if (other && other !== m) { error.textContent = `${name.value} already exists`; return; }
      editMsg(can, k, `edit message ${name.value}`, (x) => Object.assign(x, {name: name.value, size: +size.value, transmitter: node.value.trim(), comment: comment.value.trim()}));
      d.close();
    }}, "Save")));
}
