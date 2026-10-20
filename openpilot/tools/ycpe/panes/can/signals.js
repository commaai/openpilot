// the selected message's signals: values at the playhead, sparklines, and the signal and message editors
import {h} from "/dom.js";
import {MUX, MUXED, NORMAL, format, value} from "/data/dbc.js";
import {busOf, hexId} from "/data/can.js";
import {canOf, dialog, editMsg, field, removeSignal, subs, updateSignal} from "./common.js";

const SPARK_W = 80;  // sparkline width

export function create(ctx, state) {
  const can = canOf(ctx), s = subs();
  let open = state.open || null, filter = "", spark = state.spark || 15;  // seconds of history, cabana's default
  const head = h("div", {class: "can-toolbar"});
  const list = h("div", {class: "can-sig-list"});
  const el = h("div", {class: "can-pane can-signals"}, head, list);
  const k = () => can.selected;
  const msg = () => can.dbcMsg(k());
  const path = (sg) => `/dbc/${busOf(k())}/${msg().name}/${sg.name}`;
  let rows = new Map();  // signal name -> {row, value, spark}
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

  function build() {
    rows = new Map();
    const key = k(), m = msg();
    head.replaceChildren(...[
      h("span", {class: "can-binary-title"}, key === null ? "Select a message" : `${hexId(key)} ${m?.name || "(no DBC message)"}`),
      key !== null && m && h("span", {class: "can-muted"}, ` ${m.size} bytes · ${m.transmitter}`),
      h("span", {class: "can-spacer"}),
      key !== null && m && h("span", {class: "can-muted"}, `${m.sigs.length} signals`),
      key !== null && m && h("select", {class: "can-select", title: "Sparkline range", onchange: (e) => { spark = +e.target.value; requestSparks(); }},
        [1, 5, 15, 30].map((x) => h("option", {value: x, selected: x === spark}, `${x} s`))),
      key !== null && m && h("input", {class: "can-filter", placeholder: "Filter signals", value: filter, oninput: (e) => { filter = e.target.value; build(); }}),
      key !== null && h("button", {class: "can-btn", title: "Edit message", onclick: () => messageDialog(can, key)}, m ? "Edit message" : "Add to DBC"),
      key !== null && m && h("button", {class: "can-btn", title: "Remove message", onclick: () => can.dbcs.edit(busOf(key), m.address, null, `remove message ${m.name}`)}, "Remove")].filter(Boolean));
    if (!m) {
      list.replaceChildren(h("div", {class: "can-empty"}, key === null ? "" : "Drag across bits in the binary view to create a signal."));
      return;
    }
    if (m.comment) head.append(h("div", {class: "can-comment"}, m.comment));
    list.replaceChildren(...m.sigs.filter((sg) => sg.name.toLowerCase().includes(filter.toLowerCase())).map((sg) => {
      const v = h("span", {class: "can-sig-value"}), spark = h("canvas", {class: "can-spark"});
      const row = h("div", {class: `can-sig ${sg.name === open ? "open" : ""}`, onmouseenter: () => can.emit("hover", sg.name), onmouseleave: () => can.emit("hover", null)},
        h("div", {class: "can-sig-row", onclick: () => { open = open === sg.name ? null : sg.name; build(); }},
          h("span", {class: "can-swatch", style: `background:${sg.color}`}),
          h("span", {class: "can-sig-name", title: sg.comment || sg.name}, sg.name, sg.type === MUX ? " (M)" : sg.type === MUXED ? ` (m${sg.muxValue})` : ""),
          spark, v,
          h("button", {class: "can-icon", title: "Plot", onclick: (e) => { e.stopPropagation(); can.emit("chart", path(sg)); }}, "📈"),
          h("button", {class: "can-icon", title: "Remove signal", onclick: (e) => { e.stopPropagation(); removeSignal(can, k(), sg.name); }}, "✕")),
        sg.name === open && editor(can, k(), m, sg));
      row.draggable = true;
      row.addEventListener("dragstart", (e) => e.dataTransfer.setData("application/x-ycpe-paths", JSON.stringify([path(sg)])));
      rows.set(sg.name, {row, v, spark, sg});
      return row;
    }));
    ctx.perf.invalidate(pane);
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
  s.add(can.on("dbc", () => { build(); requestSparks(); }));
  s.add(can.on("edit", (name) => { open = name; build(); rows.get(name)?.row.scrollIntoView({block: "nearest"}); }));
  s.add(can.on("live", () => { requestSparks(); ctx.perf.invalidate(pane); }));
  s.add(can.on("hover", (name) => { for (const [n, r] of rows) r.row.classList.toggle("hover", n === name); }));
  build();
  return pane;
}

// the last few seconds of a signal, from worker buckets: a fixed-size canvas, so no layout reads
function sparkline(canvas, d, color) {
  const dpr = devicePixelRatio, w = SPARK_W, hgt = 18;
  if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = hgt * dpr; }
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

// every signal property; changes commit on change/enter through the undo stack
function editor(can, k, m, sg) {
  const commit = (props) => updateSignal(can, k, sg.name, props);
  const num = (prop, attrs = {}) => h("input", {type: "number", value: sg[prop], step: "any", ...attrs, onchange: (e) => {
    const v = parseFloat(e.target.value);
    if (Number.isFinite(v)) commit({[prop]: v});
  }});
  const text = (prop) => h("input", {value: sg[prop], spellcheck: false, onchange: (e) => commit({[prop]: e.target.value.trim()})});
  const check = (prop) => h("input", {type: "checkbox", checked: sg[prop], onchange: (e) => commit({[prop]: e.target.checked})});
  const type = h("select", {onchange: (e) => commit({type: +e.target.value})},
    [[NORMAL, "normal"], [MUX, "multiplexor"], [MUXED, "multiplexed"]].map(([v, n]) =>
      h("option", {value: v, selected: sg.type === v, disabled: (v === MUXED && !m.mux) || (v === MUX && m.mux && m.mux !== sg)}, n)));
  const vals = h("input", {value: sg.vals.map(([v, d]) => `${v} "${d}"`).join(" "), spellcheck: false, placeholder: '0 "off" 1 "on"',
    onchange: (e) => commit({vals: [...e.target.value.matchAll(/(-?[\d.]+)\s+"([^"]*)"/g)].map(([, v, d]) => [+v, d])})});
  return h("div", {class: "can-sig-editor"},
    field("Name", h("input", {value: sg.name, spellcheck: false, onchange: (e) => {
      const name = e.target.value.trim();
      if (/^\w+$/.test(name)) { commit({name}); can.emit("edit", name); } else e.target.value = sg.name;
    }})),
    field("Size", num("size", {min: 1, max: 64, step: 1})),
    field("Start bit", num("start", {min: 0, max: 511, step: 1})),
    h("div", {class: "can-field"}, h("span"), h("span", {class: "can-row"}, h("label", {class: "can-check"}, check("le"), "Little endian"), h("label", {class: "can-check"}, check("signed"), "Signed"))),
    field("Factor", num("factor")), field("Offset", num("offset")),
    field("Min", num("min")), field("Max", num("max")),
    field("Unit", text("unit")), field("Receiver", text("receiver")),
    field("Type", type),
    sg.type === MUXED && field("Mux value", num("muxValue", {min: 0, step: 1})),
    field("Comment", h("textarea", {rows: 2, onchange: (e) => commit({comment: e.target.value.trim()})}, sg.comment)),
    field("Values", vals, 'value descriptions: 0 "off" 1 "on"'));
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
      const other = can.dbcs.file(busOf(k))?.msgByName(name.value);
      if (other && other !== m) { error.textContent = `${name.value} already exists`; return; }
      editMsg(can, k, `edit message ${name.value}`, (x) => Object.assign(x, {name: name.value, size: +size.value, transmitter: node.value.trim(), comment: comment.value.trim()}));
      d.close();
    }}, "Save")));
}
