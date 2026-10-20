// cabana's message list: every (bus, address) with its name, rate, count and live bytes; filters, sorting and suppression
import {h} from "/dom.js";
import {icon} from "/icons.js";
import {canOf, hex2, menuCheck, popup, reveal, subs} from "./common.js";
import {dbcDialog, save, toolsMenu} from "./tools.js";

const ROW = matchMedia("(pointer: coarse)").matches ? 28 : 20;  // px per row; can.css agrees
// bus, freq and count are null for DBC messages never seen: "N/A", sorted last, and never matching a number
const COLS = [
  {id: "name", title: "Name", get: (m, can) => can.name(m.key), text: (m, can) => can.name(m.key) || "untitled"},
  {id: "bus", title: "Bus", get: (m) => m.bus, num: true},
  {id: "id", title: "ID", get: (m) => m.address, text: (m) => `0x${m.address.toString(16).toUpperCase()}`},
  {id: "node", title: "Node", get: (m, can) => can.dbcMsg(m.key)?.transmitter || ""},
  {id: "freq", title: "Freq", get: (m) => (m.unseen ? null : m.freq), text: (m) => (m.unseen ? "N/A" : m.freq >= 0.95 ? Math.round(m.freq) : m.freq.toFixed(2)), num: true},
  {id: "count", title: "Count", get: (m, can) => (m.unseen ? null : can.countAt(m.key)), num: true},
  {id: "bytes", title: "Bytes", get: (m) => m.size},
];
const text = (c, m, can) => String(c.text ? c.text(m, can) : c.get(m, can) ?? "N/A");

export function create(ctx, state) {
  const can = canOf(ctx), s = subs();
  let sort = state.sort || {col: "name", dir: 1}, rows = [], scrollTop = 0, height = 400;
  const filters = {...state.filters}, hidden = new Set(state.hidden), shown = new Set(state.shown);
  let unseen = state.unseen ?? true;  // DBC messages never seen on the bus, as cabana's "Show inactive messages"

  const head = h("div", {class: "can-msg-head"}, COLS.map((c) => h("div", {class: `can-col can-col-${c.id}`, onclick: () => {
    sort = {col: c.id, dir: sort.col === c.id ? -sort.dir : 1};
    refresh();
  }}, c.title, h("span", {class: "can-sort"}))));
  const filterRow = h("div", {class: "can-msg-filters"}, COLS.map((c) => h("input", {class: `can-col can-col-${c.id}`, placeholder: c.title,
    value: filters[c.id] || "", spellcheck: false, oninput: (e) => { filters[c.id] = e.target.value; refresh(); }})));
  const spacer = h("div", {class: "can-msg-spacer"});
  const empty = h("div", {class: "can-empty", hidden: true});
  const body = h("div", {class: "can-msg-body", onscroll: () => { scrollTop = body.scrollTop; ctx.perf.invalidate(pane); }}, spacer, empty);
  const pool = [];

  const undo = h("button", {class: "can-btn", title: "Undo (ctrl+z)", onclick: () => can.dbcs.undo()}, icon("undo", 14));
  const redo = h("button", {class: "can-btn", title: "Redo (ctrl+shift+z)", onclick: () => can.dbcs.redo()}, icon("redo", 14));
  const dbcBtn = h("button", {class: "can-btn can-dbc-btn", onclick: () => dbcDialog(can)});
  const clearBtn = h("button", {class: "can-btn", title: "Clear suppressed bytes", onclick: () => { can.suppressHighlighted(false); toolbar(); }});
  // settings live in the ⋯ menu, so the toolbar stays one row
  const tools = h("div", {class: "can-toolbar"}, dbcBtn, undo, redo,
    h("button", {class: "can-btn", title: "Stop highlighting bytes that just changed", onclick: () => { can.suppressHighlighted(true); toolbar(); }}, "Suppress highlighted"),
    clearBtn, h("button", {class: "can-btn", title: "Tools", onclick: (e) => toolsMenu(can, ctx, e.currentTarget)}, h("span", {}, "Tools"), icon("down", 14)),
    h("span", {class: "can-spacer"}),
    h("button", {class: "can-btn", title: "Options", onclick: (e) => popup(e.currentTarget, [
      menuCheck("Show unseen DBC messages", unseen, (on) => { unseen = on; refresh(); }),
      menuCheck("Suppress signals", can.suppressDefined, (on) => can.setSuppressDefined(on)),
      h("hr"),
      ...COLS.slice(1).map((c) => menuCheck(c.title, getComputedStyle(head.querySelector(`.can-col-${c.id}`)).display !== "none", (on) => {
        (on ? hidden : shown).delete(c.id);
        (on ? shown : hidden).add(c.id);
        columns();
      }))])}, icon("more_horiz", 14)));
  const notice = h("div", {class: "can-notice can-muted", hidden: true});
  const el = h("div", {class: "can-pane can-messages", tabindex: 0, onkeydown: key}, tools, notice, h("div", {class: "can-msg-table"}, head, filterRow, body));

  // hidden columns are a class per column on the pane, so rows never change for it. Narrow panes hide some columns
  // unless they were turned on: the menu shows which are showing
  function columns() {
    for (const c of COLS) {
      el.classList.toggle(`can-hide-${c.id}`, hidden.has(c.id));
      el.classList.toggle(`can-show-${c.id}`, shown.has(c.id));
    }
  }

  function toolbar() {
    const files = can.dbcs.all();
    dbcBtn.textContent = files.length ? files.map((f) => `${f.name}${can.dbcs.isDirty(f) ? "*" : ""}`).join(", ") : "No DBC";
    dbcBtn.title = files.length ? `DBC: ${dbcBtn.textContent} (click to open, change or save)` : "No DBC loaded: click to open one";
    const {undos, index} = can.dbcs;
    undo.disabled = index === 0;
    redo.disabled = index === undos.length;
    undo.title = index ? `Undo ${undos[index - 1].text} (ctrl+z)` : "Undo (ctrl+z)";
    redo.title = index < undos.length ? `Redo ${undos[index].text} (ctrl+shift+z)` : "Redo (ctrl+shift+z)";
    const n = [...can.suppressed.values()].reduce((a, m) => a + m.filter((x) => x).length, 0);
    clearBtn.textContent = n ? `Clear (${n})` : "Clear";
    clearBtn.disabled = !n;
    notice.textContent = can.failed ? `CAN failed: ${can.failed}` : files.length ? "" : can.notice || "";
    notice.hidden = !notice.textContent;
  }

  // filtering as cabana's parseRange: bus, freq and count are a number, or an "a-b" or "a-" range, of the value shown;
  // the rest are substrings, IDs also hex ranges, and names match signal names too
  function match(m) {
    for (const c of COLS) {
      const f = (filters[c.id] || "").trim().toLowerCase();
      if (!f) continue;
      const v = c.get(m, can), t = text(c, m, can).toLowerCase();
      if (c.num) {
        const r = /^([\d.]+)(?:\s*-\s*([\d.]*))?$/.exec(f), x = +t;
        if (v === null || !r || x < +r[1] || x > (r[2] === undefined ? +r[1] : r[2] ? +r[2] : Infinity)) return false;
        continue;
      }
      if (c.id === "bytes") {
        const j = can.live?.index.get(m.key);
        if (j === undefined || !Array.from(can.live.data.subarray(can.live.off[j], can.live.off[j + 1]), hex2).join("").toLowerCase().includes(f.replace(/\s/g, ""))) return false;
        continue;
      }
      if (c.id === "name" && can.dbcMsg(m.key)?.sigs.some((sg) => sg.name.toLowerCase().includes(f))) continue;
      const hex = c.id === "id", range = hex && /^(?:0x)?([\da-f]+)\s*-\s*(?:0x)?([\da-f]*)$/.exec(f);
      if (range) {
        if (v < parseInt(range[1], 16) || (range[2] && v > parseInt(range[2], 16))) return false;
      } else if (!(hex ? t.slice(2).includes(f.replace(/^0x/, "")) : t.includes(f))) return false;
    }
    return true;
  }

  // Sorting happens only on an explicit change (column, filter, DBC). Frames arriving keep every row where it is and
  // append new messages below, so nothing moves under the user while a route loads; a re-sort keeps the selected row
  // at its place on screen.
  function refresh(resort = true) {
    const all = [...can.messages(), ...(unseen ? can.unseen() : [])].filter(match);
    if (resort) {
      const col = COLS.find((c) => c.id === sort.col), before = rows.findIndex((m) => m.key === can.selected);
      // unnamed and unseen messages stay below the others in either direction, as do ties by bus and address
      const none = (v) => v === "" || v === null;
      rows = all.map((m) => ({m, v: col.get(m, can)}))
        .sort((a, b) => (none(a.v) !== none(b.v) ? (none(a.v) ? 1 : -1) : (a.v < b.v ? -1 : a.v > b.v ? 1 : 0) * sort.dir) || a.m.key - b.m.key)
        .map((r) => r.m);
      const after = rows.findIndex((m) => m.key === can.selected);
      if (before >= 0 && after >= 0 && before * ROW >= scrollTop && before * ROW < scrollTop + height) {
        spacer.style.height = `${rows.length * ROW}px`;
        body.scrollTop = scrollTop = Math.max(0, scrollTop + (after - before) * ROW);
      }
    } else {
      const at = new Map(rows.map((m, i) => [m.key, i])), fresh = all.filter((m) => !at.has(m.key));
      rows = all.filter((m) => at.has(m.key)).sort((a, b) => at.get(a.key) - at.get(b.key)).concat(fresh.sort((a, b) => a.key - b.key));
    }
    head.querySelectorAll(".can-sort").forEach((x, i) => { x.textContent = COLS[i].id === sort.col ? (sort.dir > 0 ? " ▲" : " ▼") : ""; });
    spacer.style.height = `${rows.length * ROW}px`;
    empty.textContent = rows.length ? "" : can.messages().length ? "No messages match the filters" : can.failed ? `CAN failed: ${can.failed}` : "Waiting for CAN frames…";
    empty.hidden = rows.length > 0;
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
    const r = h("div", {class: "can-msg-row", onclick: () => { can.select(r.key); reveal(el, "can-binary"); }}, cells);
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
      // rows past n are left from a taller pane
      for (let i = 0; i < Math.max(n, pool.length); i++) {
        const m = i < n && rows[first + i];
        if (!m) { if (pool[i] && !pool[i].hidden) pool[i].hidden = true; continue; }
        const r = pool[i] || (pool[i] = row());
        if (r.hidden) r.hidden = false;
        r.key = m.key;
        if (r.top !== first + i) r.style.top = `${(r.top = first + i) * ROW}px`;
        if (r.sel !== (m.key === can.selected)) r.classList.toggle("selected", r.sel = m.key === can.selected);
        const j = live?.index.get(m.key), active = j !== undefined && t - live.last[j] < 2;
        if (r.active !== active) r.classList.toggle("inactive", !(r.active = active));
        COLS.forEach((c, k) => { if (c.id !== "bytes") set(r.cells[k], text(c, m, can)); });
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
    save: () => ({sort, filters, hidden: [...hidden], shown: [...shown], unseen}),
  };
  s.add(can.on("messages", () => refresh(false)));
  s.add(can.on("dbc", () => refresh()));
  s.add(can.on("live", () => { if (filters.bytes) refresh(false); else ctx.perf.invalidate(pane); }));
  s.add(can.on("select", () => ctx.perf.invalidate(pane)));
  s.add(can.dbcs.on(toolbar));
  // DBC undo/redo anywhere but in a text box, as cabana, while this view shows and it has an edit to undo; otherwise
  // the keys go on to a focused plot or the layout
  const undoKeys = (e) => {
    if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "z" || e.defaultPrevented || !el.isConnected) return;
    if (e.target.closest?.("input, textarea, select")) return;
    const d = can.dbcs;
    if (e.shiftKey ? d.index >= d.undos.length : d.index === 0) return;
    e.preventDefault();
    if (e.shiftKey) d.redo(); else d.undo();
  };
  document.addEventListener("keydown", undoKeys);
  s.add(() => document.removeEventListener("keydown", undoKeys));
  // cabana's file keys, rather than the browser's: ctrl+s saves the DBC with edits (or the only one), ctrl+shift+s
  // saves it as, and ctrl+o or more than one to pick from opens the DBC dialog
  const fileKeys = (e) => {
    const k = e.key.toLowerCase();
    if (!(e.ctrlKey || e.metaKey) || e.altKey || (k !== "s" && k !== "o") || e.defaultPrevented || !el.isConnected || e.target.closest?.("dialog")) return;
    e.preventDefault();
    const files = can.dbcs.all(), dirty = files.filter((f) => can.dbcs.isDirty(f));
    const f = k === "s" && (dirty.length === 1 ? dirty[0] : files.length === 1 ? files[0] : null);
    if (f) save(can, f, e.shiftKey).catch((err) => dbcDialog(can, err.message));
    else dbcDialog(can);
  };
  document.addEventListener("keydown", fileKeys);
  s.add(() => document.removeEventListener("keydown", fileKeys));
  s.add(can.on("notice", toolbar));
  columns();
  const sizer = new ResizeObserver(([e]) => { height = e.contentRect.height; ctx.perf.invalidate(pane); });
  sizer.observe(body);
  s.add(() => sizer.disconnect());
  refresh();
  return pane;
}
