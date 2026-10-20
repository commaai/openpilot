// cabana's binary view: the selected message's bits with a flip heatmap, signal overlays, and drag to create or resize
import {h} from "/dom.js";
import {MUX, MUXED, flip, format, raw, value} from "/data/dbc.js";
import {busOf, hexId} from "/data/can.js";
import {canOf, createSignal, hex2, removeSignal, subs, updateSignal} from "./common.js";

const ROW = 22;
// bit position <-> grid cell: row = byte, column 0 = bit 7
const posOf = (row, col) => row * 8 + 7 - col;
const cellOf = (pos) => [pos >> 3, 7 - (pos & 7)];
const withAlpha = (hsl, a) => hsl.replace(")", ` / ${a})`);

const DIR_KEY = "ycpe_can_drag";
let dragDir = localStorage.getItem(DIR_KEY) || "msb";

export function create(ctx, state) {
  const can = canOf(ctx), s = subs();
  let mode = state.mode || "live", heat = null, heatKey = null, hover = null, drag = null, dpr = 1, width = 300;
  const canvas = h("canvas", {class: "can-binary-canvas"});
  const info = h("div", {class: "can-binary-info"});
  const title = h("span", {class: "can-binary-title"});
  const modeBtn = h("select", {class: "can-select", title: "Heatmap: flips up to the playhead, or over the visible range",
    onchange: (e) => { mode = e.target.value; heat = null; requestHeat(); }},
    ["live", "range", "off"].map((v) => h("option", {value: v, selected: mode === v}, v)));
  // cabana's drag direction setting, shared by every binary view
  const dirBtn = h("select", {class: "can-select", title: "Endianness of a dragged signal",
    onchange: (e) => { dragDir = e.target.value; localStorage.setItem(DIR_KEY, dragDir); }},
    [["msb", "MSB first"], ["lsb", "LSB first"], ["le", "always LE"], ["be", "always BE"]].map(([v, n]) => h("option", {value: v, selected: v === dragDir}, n)));
  const empty = h("div", {class: "can-empty"}, "Select a message to see its bits");
  const scroller = h("div", {class: "can-binary-scroll"}, canvas, empty);
  // no toolbar until a message is selected: the empty hint says it
  const bar = h("div", {class: "can-toolbar", hidden: true}, title, h("span", {class: "can-spacer"}), "heatmap", modeBtn, dirBtn);
  const el = h("div", {class: "can-pane can-binary", tabindex: 0, onkeydown: key}, bar, scroller, info);

  const k = () => can.selected;
  const msg = () => can.dbcMsg(k());
  const bytes = () => {
    const live = can.live, j = live?.index.get(k());
    return j === undefined ? new Uint8Array(0) : live.data.subarray(live.off[j], live.off[j + 1]);
  };
  const rows = () => Math.max(msg()?.size || 0, bytes().length, can.msg(k())?.size || 0);

  // the signals shown for the current data: multiplexed ones only when their mux value is selected
  function visible() {
    const m = msg(), d = bytes();
    if (!m) return [];
    return m.sigs.filter((sg) => sg.type !== MUXED || (m.mux && raw(m.mux, d) === sg.muxValue));
  }

  // per bit position, the signals covering it, largest first
  function coverage(sigs, n) {
    const cells = Array.from({length: n * 8}, () => []);
    for (const sg of sigs) {
      for (let j = 0; j < sg.size; j++) {
        const pos = sg.le ? sg.start + j : flip(flip(sg.start) + j);
        if (pos >= 0 && pos < n * 8) cells[pos].push(sg);
      }
    }
    for (const c of cells) c.sort((a, b) => b.size - a.size);
    return cells;
  }

  let heatBusy = false, heatAgain = false;
  async function requestHeat() {
    if (k() === null || mode === "off") return ctx.perf.invalidate(pane);
    if (heatBusy) { heatAgain = true; return; }
    heatBusy = true;
    const want = k(), [t0, t1] = mode === "live" ? [-Infinity, ctx.clock.t] : ctx.clock.range;
    heat = await can.heatmap(want, t0, t1);
    heatKey = want;
    heatBusy = false;
    ctx.perf.invalidate(pane);
    if (heatAgain) { heatAgain = false; requestHeat(); }
  }

  function geometry() {
    const w = width, col = Math.max(24, Math.floor(w / 10));
    return {col, x0: Math.floor((w - col * 10) / 2) + col};
  }

  function at(e) {
    const r = canvas.getBoundingClientRect(), {col, x0} = geometry();
    const c = Math.floor((e.clientX - r.left - x0) / col), row = Math.floor((e.clientY - r.top) / ROW);
    if (row < 0 || row >= rows() || c < 0) return null;
    return {row, col: Math.min(c, 7), hex: c >= 8};
  }

  // cabana's getSelection: endianness from the drag direction setting (by default, dragging toward lower bit
  // positions makes a little endian signal), or the resized signal's
  function selection({anchor, cur, resize}) {
    const a = posOf(anchor.row, anchor.col), b = posOf(cur.row, cur.col);
    const back = cur.row * 8 + cur.col < anchor.row * 8 + anchor.col;
    const le = resize ? resize.le : {msb: back, lsb: !back, le: true, be: false}[dragDir];
    const start = le ? Math.min(a, b) : back ? b : a;
    const size = le ? Math.abs(a - b) + 1 : Math.abs(flip(a) - flip(b)) + 1;
    return {start, size, le};
  }

  canvas.addEventListener("pointerdown", (e) => {
    const c = at(e);
    if (!c || c.hex || e.button) return;
    canvas.setPointerCapture(e.pointerId);
    drag = {anchor: c, cur: c, moved: false, resize: null};
    // grabbing a signal's first or last bit resizes it from the other end
    const pos = posOf(c.row, c.col);
    for (const sg of coverage(visible(), rows())[pos]) {
      if (pos === sg.lsb || pos === sg.msb) {
        const [row, col] = cellOf(pos === sg.lsb ? sg.msb : sg.lsb);
        drag = {anchor: {row, col}, cur: c, moved: false, resize: sg};
        break;
      }
    }
  });
  canvas.addEventListener("pointermove", (e) => {
    const c = at(e);
    if (drag) {
      if (c && (c.row !== drag.cur.row || c.col !== drag.cur.col)) drag.moved = true;
      if (c) drag.cur = c;
    }
    const sg = c && !c.hex ? coverage(visible(), rows())[posOf(c.row, c.col)]?.at(-1) || null : null;
    if (sg !== hover) { hover = sg; can.emit("hover", sg?.name); }
    ctx.perf.invalidate(pane);
  });
  canvas.addEventListener("pointerleave", () => { if (hover) { hover = null; can.emit("hover", null); ctx.perf.invalidate(pane); } });
  canvas.addEventListener("pointerup", () => {
    const d = drag;
    drag = null;
    if (!d) return;
    // a click, even on an M/L edge, opens the editor
    if (d.moved) {
      const sel = selection(d);
      if (!d.resize) can.emit("edit", createSignal(can, k(), sel));
      else if (sel.start !== d.resize.start || sel.size !== d.resize.size) updateSignal(can, k(), d.resize.name, {start: sel.start, size: sel.size, max: d.resize.max});
    } else {
      const sg = d.resize || coverage(visible(), rows())[posOf(d.anchor.row, d.anchor.col)]?.at(-1);
      if (sg) can.emit("edit", sg.name);
    }
    ctx.perf.invalidate(pane);
  });

  function key(e) {
    if (!hover || e.ctrlKey || e.metaKey) return;
    if (["x", "Delete", "Backspace"].includes(e.key)) removeSignal(can, k(), hover.name);
    else if (e.key === "e") updateSignal(can, k(), hover.name, {le: !hover.le});
    else if (e.key === "s") updateSignal(can, k(), hover.name, {signed: !hover.signed});
    else if ("pgc".includes(e.key)) can.emit("chart", `/dbc/${busOf(k())}/${msg().name}/${hover.name}`);
    else return;
    hover = null;
    e.preventDefault();
  }

  const pane = {
    name: "can-binary", el,
    render() {
      const key = k(), n = key === null ? 0 : rows(), {col, x0} = geometry();
      if (bar.hidden !== (key === null)) bar.hidden = key === null;
      if (key !== null) title.textContent = `${hexId(key)} ${msg()?.name || ""}`;
      dpr = devicePixelRatio;
      const w = width, H = Math.max(n * ROW, 1);
      if (canvas.width !== w * dpr || canvas.height !== H * dpr) {
        canvas.width = w * dpr;
        canvas.height = H * dpr;
        canvas.style.height = `${H}px`;
      }
      const g = canvas.getContext("2d");
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, H);
      if (empty.hidden !== n > 0) empty.hidden = n > 0;
      if (!n) return;
      const color = (x) => ctx.theme.color(x), fg = color("--fg"), muted = color("--muted"), line = color("--line"), accent = color("--accent"), mono = color("--mono");
      // light text on dark needs dimmer signal fills
      const fill = ctx.theme.name === "dark" ? [0.25, 0.4] : [0.38, 0.62];
      const d = bytes(), sigs = visible(), cells = coverage(sigs, n);
      const flips = heatKey === key && heat ? heat : new Uint32Array(0);
      const max = Math.max(1, ...flips);
      const sel = drag && (drag.moved || drag.resize) ? selection(drag) : null;
      const selected = new Set();
      if (sel) for (let j = 0; j < sel.size; j++) selected.add(sel.le ? sel.start + j : flip(flip(sel.start) + j));
      g.font = `12px ${mono}`;
      g.textAlign = "center";
      g.textBaseline = "middle";
      const live = can.live, j = live?.index.get(key);
      for (let r = 0; r < n; r++) {
        const y = r * ROW;
        g.fillStyle = muted;
        g.fillText(r, x0 - col / 2, y + ROW / 2);
        for (let c = 0; c < 8; c++) {
          const pos = posOf(r, c), x = x0 + c * col, cover = cells[pos], f = flips[r * 8 + c] || 0;
          // heatmap: log scale as in cabana, stronger under signals
          const heatA = f ? Math.min(1, Math.max(cover.length ? 0.1 : 0.04, Math.log2(1 + f * 1.2) / Math.log2(1 + max * 1.2))) : cover.length ? 0.1 : 0;
          if (selected.has(pos)) {
            g.fillStyle = withAlpha(drag.resize?.color || "hsl(220 70% 55%)", 0.85);
            g.fillRect(x, y, col, ROW);
          } else if (cover.length) {
            for (const sg of cover) {
              g.fillStyle = withAlpha(sg.color, fill[0] + heatA * fill[1]);
              g.fillRect(x, y, col, ROW);
            }
            if (hover && cover.includes(hover)) { g.fillStyle = withAlpha(hover.color, 0.9); g.fillRect(x, y, col, ROW); }
            if (cover.length > 1) hatch(g, x, y, col, ROW, muted);
          } else if (heatA) {
            g.globalAlpha = heatA;
            g.fillStyle = accent;
            g.fillRect(x, y, col, ROW);
            g.globalAlpha = 1;
          }
          if (r < d.length) {
            g.fillStyle = fg;
            g.fillText((d[r] >> (7 - c)) & 1, x + col / 2, y + ROW / 2);
          } else hatch(g, x, y, col, ROW, line);
          // M/L markers need room beside the digit
          if (col >= 30) for (const sg of cover) {
            if (pos === sg.msb || pos === sg.lsb) {
              g.save();
              g.font = `9px ${color("--font")}`;
              g.fillStyle = fg;
              g.fillText(pos === sg.msb ? "M" : "L", x + col - 6, y + ROW - 6);
              g.restore();
            }
          }
        }
        // the byte, colored by its last change (live) or by how often it changed (range)
        const x = x0 + 8 * col;
        if (r < d.length) {
          const o = live.off[j] + r, a = mode === "live" ? Math.max(0, 1 - live.age[o] / 2) : Math.max(...flips.slice(r * 8, r * 8 + 8), 0) / max;
          if (a > 0) {
            g.globalAlpha = a * 0.8;
            g.fillStyle = mode === "live" ? color(["", "--accent", "--critical", "--muted"][live.kind[o]] || "--muted") : accent;
            g.fillRect(x, y, col, ROW);
            g.globalAlpha = 1;
          }
          g.fillStyle = fg;
          g.font = `bold 12px ${mono}`;
          g.fillText(hex2(d[r]), x + col / 2, y + ROW / 2);
          g.font = `12px ${mono}`;
        }
      }
      // grid, then signal outlines on top
      g.strokeStyle = line;
      g.lineWidth = 1;
      g.beginPath();
      for (let c = 0; c <= 9; c++) { g.moveTo(x0 + c * col + 0.5, 0); g.lineTo(x0 + c * col + 0.5, n * ROW); }
      for (let r = 0; r <= n; r++) { g.moveTo(x0, r * ROW + 0.5); g.lineTo(x0 + 9 * col, r * ROW + 0.5); }
      g.stroke();
      for (const sg of sigs) outline(g, sg, cells, n, x0, col, hover === sg);
      info.textContent = hover ? describe(hover) : warnings(d, cells) || (msg() ? `${sigs.length} signals · drag to add, drag an M/L edge to resize · x delete, e endianness, s sign, c chart` : "drag across bits to create a signal");
    },
    resize() { ctx.perf.invalidate(pane); },
    dispose() { s.dispose(); },
    save: () => ({mode}),
  };

  // cabana's detail warnings
  function warnings(d, cells) {
    const m = msg(), out = [];
    if (!d.length) out.push("no messages received yet");
    else if (m && m.size !== d.length) out.push(`the DBC says ${m.size} bytes, received ${d.length}`);
    if (cells.some((c) => c.length > 1)) out.push("signals overlap");
    return out.length ? `⚠ ${out.join(" · ")}` : "";
  }

  function describe(sg) {
    const d = bytes(), v = d.length ? value(msg(), sg, d) : NaN;
    return `${sg.name}: ${Number.isNaN(v) ? "–" : format(sg, v)} · start ${sg.start} size ${sg.size} · msb ${sg.msb} lsb ${sg.lsb} · ${sg.le ? "little" : "big"} endian${sg.signed ? " · signed" : ""}${sg.type === MUX ? " · multiplexor" : sg.type === MUXED ? ` · mux ${sg.muxValue}` : ""}`;
  }

  s.add(can.on("select", () => { heat = null; hover = null; requestHeat(); ctx.perf.invalidate(pane); }));
  s.add(can.on("live", () => { if (mode === "live") requestHeat(); ctx.perf.invalidate(pane); }));
  s.add(can.on("dbc", () => ctx.perf.invalidate(pane)));
  s.add(ctx.clock.on("range", () => { if (mode === "range") requestHeat(); }));
  s.add(ctx.theme.on(() => ctx.perf.invalidate(pane)));
  const sizer = new ResizeObserver(([e]) => { width = e.contentRect.width; ctx.perf.invalidate(pane); });
  sizer.observe(scroller);
  s.add(() => sizer.disconnect());
  requestHeat();
  return pane;
}

function hatch(g, x, y, w, hgt, color) {
  g.save();
  g.beginPath();
  g.rect(x, y, w, hgt);
  g.clip();
  g.strokeStyle = color;
  g.globalAlpha = 0.6;
  g.beginPath();
  for (let i = -hgt; i < w; i += 5) { g.moveTo(x + i, y + hgt); g.lineTo(x + i + hgt, y); }
  g.stroke();
  g.restore();
}

// the signal's border: an edge wherever the neighbouring cell isn't part of it
function outline(g, sg, cells, n, x0, col, hot) {
  const has = (r, c) => r >= 0 && r < n && c >= 0 && c < 8 && cells[posOf(r, c)].includes(sg);
  g.strokeStyle = hot ? sg.color.replace(/(\d+(\.\d+)?)%\)$/, "35%)") : sg.color.replace(/(\d+(\.\d+)?)%\)$/, "45%)");
  g.lineWidth = hot ? 2 : 1.5;
  g.beginPath();
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < 8; c++) {
      if (!has(r, c)) continue;
      const x = x0 + c * col, y = r * ROW;
      if (!has(r, c - 1)) { g.moveTo(x + 1, y); g.lineTo(x + 1, y + ROW); }
      if (!has(r, c + 1)) { g.moveTo(x + col - 1, y); g.lineTo(x + col - 1, y + ROW); }
      if (!has(r - 1, c)) { g.moveTo(x, y + 1); g.lineTo(x + col, y + 1); }
      if (!has(r + 1, c)) { g.moveTo(x, y + ROW - 1); g.lineTo(x + col, y + ROW - 1); }
    }
  }
  g.stroke();
}
