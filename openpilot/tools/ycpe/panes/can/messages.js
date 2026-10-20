// cabana's message list: every (bus, address) with its name, rate, count and live bytes; filters, sorting and suppression
import {h} from "/dom.js";
import {canOf, hex2, subs} from "./common.js";
import {dbcDialog, toolsMenu} from "./tools.js";

const ROW = 20;
const COLS = [
  {id: "name", title: "Name", get: (m, can) => can.name(m.key), text: (m, can) => can.name(m.key) || "untitled"},
  {id: "bus", title: "Bus", get: (m) => m.bus},
  {id: "id", title: "ID", get: (m) => m.address, text: (m) => `0x${m.address.toString(16).toUpperCase()}`},
  {id: "node", title: "Node", get: (m, can) => can.dbcMsg(m.key)?.transmitter || ""},
  {id: "freq", title: "Freq", get: (m) => m.freq, text: (m) => (m.unseen ? "N/A" : m.freq >= 0.95 ? Math.round(m.freq) : m.freq.toFixed(2))},
  {id: "count", title: "Count", get: (m, can) => can.countAt(m.key)},
  {id: "bytes", title: "Bytes", get: (m) => m.size},
];

export function create(ctx, state) {
  const can = canOf(ctx), s = subs();
  let sort = state.sort || {col: "name", dir: 1}, rows = [], scrollTop = 0, height = 400;
  const filters = {...state.filters};

  const head = h("div", {class: "can-msg-head"}, COLS.map((c) => h("div", {class: `can-col can-col-${c.id}`, onclick: () => {
    sort = {col: c.id, dir: sort.col === c.id ? -sort.dir : 1};
    refresh();
  }}, c.title, h("span", {class: "can-sort"}))));
  const filterRow = h("div", {class: "can-msg-filters"}, COLS.map((c) => h("input", {class: `can-col can-col-${c.id}`, placeholder: c.title,
    value: filters[c.id] || "", spellcheck: false, oninput: (e) => { filters[c.id] = e.target.value; refresh(); }})));
  const spacer = h("div", {class: "can-msg-spacer"});
  const body = h("div", {class: "can-msg-body", onscroll: () => { scrollTop = body.scrollTop; ctx.perf.invalidate(pane); }}, spacer);
  const pool = [];

  const undo = h("button", {class: "can-btn", title: "Undo (ctrl+z)", onclick: () => can.dbcs.undo()}, "↶");
  const redo = h("button", {class: "can-btn", title: "Redo (ctrl+shift+z)", onclick: () => can.dbcs.redo()}, "↷");
  const dbcBtn = h("button", {class: "can-btn can-dbc-btn", title: "DBC files", onclick: () => dbcDialog(can)});
  const clearBtn = h("button", {class: "can-btn", title: "Clear suppressed bytes", onclick: () => { can.suppressHighlighted(false); toolbar(); }});
  const suppressDefined = h("input", {type: "checkbox", onchange: (e) => can.setSuppressDefined(e.target.checked)});
  const tools = h("div", {class: "can-toolbar"}, dbcBtn, undo, redo,
    h("button", {class: "can-btn", title: "Stop highlighting bytes that just changed", onclick: () => { can.suppressHighlighted(true); toolbar(); }}, "Suppress highlighted"),
    clearBtn, h("label", {class: "can-check", title: "Don't highlight bits of defined signals"}, suppressDefined, "Suppress signals"),
    h("button", {class: "can-btn", title: "Tools", onclick: (e) => toolsMenu(can, ctx, e.currentTarget)}, "Tools ▾"));
  const notice = h("div", {class: "can-notice can-muted", hidden: true});
  const el = h("div", {class: "can-pane can-messages", tabindex: 0, onkeydown: key}, tools, notice, h("div", {class: "can-msg-table"}, head, filterRow, body));

  function toolbar() {
    const files = can.dbcs.all();
    dbcBtn.textContent = files.length ? files.map((f) => `${f.name}${can.dbcs.dirty ? "*" : ""}`).join(", ") : "No DBC";
    const {undos, index} = can.dbcs;
    undo.disabled = index === 0;
    redo.disabled = index === undos.length;
    undo.title = index ? `Undo ${undos[index - 1].text} (ctrl+z)` : "Undo (ctrl+z)";
    redo.title = index < undos.length ? `Redo ${undos[index].text} (ctrl+shift+z)` : "Redo (ctrl+shift+z)";
    const n = [...can.suppressed.values()].reduce((a, m) => a + m.filter((x) => x).length, 0);
    clearBtn.textContent = n ? `Clear (${n})` : "Clear";
    clearBtn.disabled = !n;
    notice.textContent = files.length ? "" : can.notice || "";
    notice.hidden = !notice.textContent;
  }

  // filtering as cabana: substrings, "a-b" and "a-" ranges for numbers (hex for IDs), and names match signal names too
  function match(m) {
    for (const c of COLS) {
      const f = (filters[c.id] || "").trim().toLowerCase();
      if (!f) continue;
      const v = c.get(m, can), text = String(c.text ? c.text(m, can) : v).toLowerCase();
      if (c.id === "bytes") {
        const j = can.live?.index.get(m.key);
        if (j === undefined || !Array.from(can.live.data.subarray(can.live.off[j], can.live.off[j + 1]), hex2).join("").toLowerCase().includes(f.replace(/\s/g, ""))) return false;
        continue;
      }
      if (c.id === "name" && can.dbcMsg(m.key)?.sigs.some((sg) => sg.name.toLowerCase().includes(f))) continue;
      const hex = c.id === "id", range = (hex ? /^(?:0x)?([\da-f]+)\s*-\s*(?:0x)?([\da-f]*)$/ : /^(-?[\d.]+)\s*-\s*(-?[\d.]*)$/).exec(f);
      if (range && typeof v === "number") {
        const num = (x) => (hex ? parseInt(x, 16) : +x);
        if (v < num(range[1]) || (range[2] && v > num(range[2]))) return false;
      } else if (!(hex ? text.slice(2).includes(f.replace(/^0x/, "")) : text.includes(f))) return false;
    }
    return true;
  }

  function refresh() {
    const col = COLS.find((c) => c.id === sort.col);
    rows = [...can.messages(), ...can.unseen()].filter(match).map((m) => ({m, v: col.get(m, can)}));
    // unnamed messages stay below named ones in either direction, as do ties by bus and address
    rows.sort((a, b) => (sort.col === "name" && !a.v !== !b.v ? (a.v ? -1 : 1) : (a.v < b.v ? -1 : a.v > b.v ? 1 : 0) * sort.dir) || a.m.key - b.m.key);
    rows = rows.map((r) => r.m);
    head.querySelectorAll(".can-sort").forEach((x, i) => { x.textContent = COLS[i].id === sort.col ? (sort.dir > 0 ? " ▲" : " ▼") : ""; });
    spacer.style.height = `${rows.length * ROW}px`;
    toolbar();
    ctx.perf.invalidate(pane);
  }

  function key(e) {
    if (e.target.tagName === "INPUT") return;
    const i = rows.findIndex((m) => m.key === can.selected);
    const page = Math.max(1, Math.floor(height / ROW) - 1);
    const step = {ArrowDown: 1, ArrowUp: -1, PageDown: page, PageUp: -page, Home: -Infinity, End: Infinity}[e.key];
    if (step) {
      const n = Math.max(0, Math.min(rows.length - 1, i + step));
      if (rows[n]) can.select(rows[n].key);
      const top = n * ROW;
      if (top < body.scrollTop) body.scrollTop = top;
      else if (top + ROW > body.scrollTop + body.clientHeight) body.scrollTop = top + ROW - body.clientHeight;
      e.preventDefault();
    }
  }

  function row() {
    const cells = COLS.map((c) => h("div", {class: `can-col can-col-${c.id}`}));
    const r = h("div", {class: "can-msg-row", onclick: () => can.select(r.key)}, cells);
    r.cells = cells;
    r.bytes = [];
    spacer.append(r);
    return r;
  }

  const set = (el, text) => { if (el.textContent !== text) el.textContent = text; };

  const pane = {
    name: "can-messages", el,
    render() {
      const first = Math.floor(scrollTop / ROW), n = Math.ceil(height / ROW) + 1;
      const live = can.live, t = ctx.clock.t;
      for (let i = 0; i < n; i++) {
        const m = rows[first + i], r = pool[i] || (pool[i] = row());
        if (r.hidden !== !m) r.hidden = !m;
        if (!m) continue;
        r.key = m.key;
        if (r.top !== first + i) r.style.top = `${(r.top = first + i) * ROW}px`;
        if (r.sel !== (m.key === can.selected)) r.classList.toggle("selected", r.sel = m.key === can.selected);
        const j = live?.index.get(m.key), active = j !== undefined && t - live.last[j] < 2;
        if (r.active !== active) r.classList.toggle("inactive", !(r.active = active));
        COLS.forEach((c, k) => { if (c.id !== "bytes") set(r.cells[k], String(c.text ? c.text(m, can) : c.get(m, can))); });
        const dm = can.dbcMsg(m.key), named = !!dm;
        if (r.named !== named) r.cells[0].classList.toggle("can-muted", !(r.named = named));
        const tip = dm ? (dm.comment ? `${dm.name}\n${dm.comment}` : dm.name) : "";
        if (r.cells[0].title !== tip) r.cells[0].title = tip;
        // live bytes with change highlighting: up, down or periodic, fading over 2 s as in cabana
        const cell = r.cells.at(-1), len = j === undefined ? 0 : live.off[j + 1] - live.off[j];
        while (r.bytes.length < len) cell.append(r.bytes[r.bytes.push(h("span")) - 1]);
        r.bytes.forEach((b, x) => {
          if (b.hidden !== x >= len) b.hidden = x >= len;
          if (x >= len) return;
          const o = live.off[j] + x, a = Math.max(0, 1 - live.age[o] / 2);
          set(b, hex2(live.data[o]));
          const cls = a > 0 ? `can-chg${live.kind[o]}` : "", av = a.toFixed(2);
          if (b.className !== cls) b.className = cls;
          if (a > 0 && b.av !== av) b.style.setProperty("--a", b.av = av);
        });
      }
    },
    resize() { ctx.perf.invalidate(pane); },
    dispose() { s.dispose(); },
    save: () => ({sort, filters}),
  };
  s.add(can.on("messages", refresh));
  s.add(can.on("dbc", refresh));
  s.add(can.on("live", () => { if (filters.bytes || sort.col === "count") refresh(); else ctx.perf.invalidate(pane); }));
  s.add(can.on("select", () => ctx.perf.invalidate(pane)));
  s.add(can.dbcs.on(toolbar));
  // DBC undo/redo anywhere but in a text box, as cabana
  const undoKeys = (e) => {
    if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "z" || e.target.closest?.("input, textarea, select")) return;
    e.preventDefault();
    if (e.shiftKey) can.dbcs.redo(); else can.dbcs.undo();
  };
  document.addEventListener("keydown", undoKeys);
  s.add(() => document.removeEventListener("keydown", undoKeys));
  s.add(can.on("notice", toolbar));
  const sizer = new ResizeObserver(([e]) => { height = e.contentRect.height; ctx.perf.invalidate(pane); });
  sizer.observe(body);
  s.add(() => sizer.disconnect());
  refresh();
  return pane;
}
