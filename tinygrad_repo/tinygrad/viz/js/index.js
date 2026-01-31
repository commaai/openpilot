// ** graph helpers

const displaySelection = (sel) => {
  for (const e of document.getElementsByClassName("view")) e.style.display = e.matches(sel) ? "flex" : "none";
}
const metadata = document.querySelector(".metadata");

const darkenHex = (h, p = 0) =>
  `#${(
    c = parseInt(h.slice(1), 16),
    f = 1 - p / 100,
    ((c >> 16 & 255) * f | 0) << 16 |
    ((c >>  8 & 255) * f | 0) <<  8 |
    ((c       & 255) * f | 0)
  ).toString(16).padStart(6, '0')}`;

const ANSI_COLORS = ["#b3b3b3", "#ff6666", "#66b366", "#ffff66", "#6666ff", "#ff66ff", "#66ffff", "#ffffff"];
const ANSI_COLORS_LIGHT = ["#d9d9d9","#ff9999","#99cc99","#ffff99","#9999ff","#ff99ff","#ccffff","#ffffff"];
const colorsCache = new Map();
const parseColors = (name, defaultColor="#ffffff") => Array.from(name.matchAll(/(?:\u001b\[(\d+)m([\s\S]*?)\u001b\[0m)|([^\u001b]+)/g),
  ([_, code, colored_st, st]) => ({ st: colored_st ?? st, color: code != null ? (code>=90 ? ANSI_COLORS_LIGHT : ANSI_COLORS)[(parseInt(code)-30+60)%60] : defaultColor }));

const colored = n => d3.create("span").call(s => s.selectAll("span").data(typeof n === "string" ? parseColors(n) : n).join("span")
                       .style("color", d => d.color).text(d => d.st)).node();

const rect = (s) => (typeof s === "string" ? document.querySelector(s) : s).getBoundingClientRect();

let timeout = null;
const Status = {STARTED:0, COMPLETE:1, ERR:2}
const updateProgress = (st, msg) => {
  clearTimeout(timeout);
  const msgEl = d3.select("#progress-message").style("display", "none");
  const customEl = d3.select("#custom").style("display", "none");
  if (st === Status.STARTED) {
    msgEl.text(msg);
    timeout = setTimeout(() => msgEl.style("display", "block"), 2000);
  } else if (st === Status.ERR) {
    displaySelection("#custom");
    customEl.html("").append("div").classed("raw-text", true).append(() => codeBlock(msg));
  }
}

function intersectRect(r1, r2) {
  const dx = r2.x-r1.x;
  const dy = r2.y-r1.y;
  if (dx === 0 && dy === 0) throw new Error("Invalid node coordinates, rects must not overlap");
  const scaleX = dx !== 0 ? (r1.width/2)/Math.abs(dx) : Infinity;
  const scaleY = dy !== 0 ? (r1.height/2)/Math.abs(dy) : Infinity;
  const scale = Math.min(scaleX, scaleY);
  return {x:r1.x+dx*scale, y:r1.y+dy*scale};
}

function addTags(root) {
  root.selectAll("circle").data(d => [d]).join("circle").attr("r", 5);
  root.selectAll("text").data(d => [d]).join("text").text(d => d).attr("dy", "0.35em");
}

const drawGraph = (data) => {
  const g = dagre.graphlib.json.read(data);
  // draw nodes
  d3.select("#graph-svg").on("click", () => d3.selectAll(".highlight").classed("highlight", false));
  const nodes = d3.select("#nodes").selectAll("g").data(g.nodes().map(id => g.node(id)), d => d).join("g").attr("class", d => d.className ?? "node")
    .attr("transform", d => `translate(${d.x},${d.y})`).classed("clickable", d => d.ref != null).on("click", (e,d) => {
      if (d.ref != null) return switchCtx(d.ref);
      const parents = g.predecessors(d.id);
      const children = g.successors(d.id);
      if (parents == null && children == null) return;
      const src = [...parents, ...children, d.id];
      nodes.classed("highlight", n => src.includes(n.id)).classed("child", n => children.includes(n.id));
      if (!e.target.classList.contains("token")) labels.selectAll("rect.bg").classed("highlight", false);
      const matchEdge = (v, w) => (v===d.id && children.includes(w)) ? "highlight child " : (parents.includes(v) && w===d.id) ? "highlight " : "";
      d3.select("#edges").selectAll("path.edgePath").attr("class", e => matchEdge(e.v, e.w)+"edgePath");
      d3.select("#edge-labels").selectAll("g.port").attr("class",  (_, i, n) => matchEdge(...n[i].id.split("-"))+"port");
      e.stopPropagation();
    });
  nodes.selectAll("rect").data(d => [d]).join("rect").attr("width", d => d.width).attr("height", d => d.height).attr("fill", d => d.color)
    .attr("x", d => -d.width/2).attr("y", d => -d.height/2).classed("node", true);
  const STROKE_WIDTH = 1.4, textSpace = g.graph().textSpace;
  const labels = nodes.selectAll("g.label").data(d => [d]).join("g").attr("class", "label");
  labels.attr("transform", d => `translate(-${d.labelWidth/2}, -${d.labelHeight/2+STROKE_WIDTH*2})`);
  const rectGroup = labels.selectAll("g.rect-group").data(d => [d]).join("g").attr("class", "rect-group");
  const tokens = labels.selectAll("g.text-group").data(d => [d]).join("g").attr("class", "text-group").selectAll("text").data(d => {
    if (Array.isArray(d.label)) return [d.label];
    const ret = [[]];
    for (const s of parseColors(d.label, defaultColor="initial")) {
      const color = darkenHex(s.color, 25);
      const lines = s.st.split("\n");
      ret.at(-1).push({ st:lines[0], color });
      for (let i=1; i<lines.length; i++) ret.push([{ st:lines[i], color }]);
    }
    return [ret];
  }).join("text").style("font-family", g.graph().font).selectAll("tspan").data(d => d).join("tspan").attr("x", "0").attr("dy", g.graph().lh)
    .selectAll("tspan").data(d => d).join("tspan").attr("dx", (d, i) => i > 0 && d.st !== "," ? textSpace: 0).text(d => d.st).classed("token", true)
    .attr("xml:space", "preserve").attr("fill", d => d.color);
  const tokensBg = rectGroup.selectAll("rect.bg").data((d, i, nodes) => {
    const ret = [];
    d3.select(nodes[i].parentElement).select("g.text-group").selectAll("tspan.token").each((d, i, nodes) => {
      if (!d.keys?.length) return;
      const b = nodes[i].getBBox(); ret.push({ keys:d.keys, x:b.x, y:b.y, width:b.width, height:b.height });
    });
    return ret;
  }).join("rect").attr("class", "bg").attr("x", d => d.x).attr("y", d => d.y).attr("width", d => d.width).attr("height", d => d.height);
  tokens.on("click", (e, { keys }) => {
    tokensBg.classed("highlight", (d, i, nodes) => !nodes[i].classList.contains("highlight") && d.keys.some(k => keys?.includes(k)));
  });
  addTags(nodes.selectAll("g.tag").data(d => d.tag != null ? [d] : []).join("g").attr("class", "tag")
    .attr("transform", d => `translate(${-d.width/2+8}, ${-d.height/2+8})`).datum(e => e.tag));
  // draw edges
  const line = d3.line().x(d => d.x).y(d => d.y).curve(d3.curveBasis), edges = g.edges();
  d3.select("#edges").selectAll("path.edgePath").data(edges).join("path").attr("class", "edgePath").attr("d", (e) => {
    const edge = g.edge(e);
    const points = edge.points.slice(1, edge.points.length-1);
    points.unshift(intersectRect(g.node(e.v), points[0]));
    points.push(intersectRect(g.node(e.w), points[points.length-1]));
    return line(points);
  }).attr("marker-end", "url(#arrowhead)").attr("stroke", e => g.edge(e).color || "#4a4b57");
}

// ** UOp graph

let workerUrl = null, worker = null;
async function initWorker() {
  const resp = await Promise.all(["/assets/dagrejs.github.io/project/dagre/latest/dagre.min.js","/js/worker.js"].map(u => fetch(u)));
  workerUrl = URL.createObjectURL(new Blob([(await Promise.all(resp.map((r) => r.text()))).join("\n")], { type: "application/javascript" }));
}

function renderDag(layoutSpec, { recenter }) {
  // start calculating the new layout (non-blocking)
  updateProgress(Status.STARTED, "Rendering new graph...");
  if (worker != null) worker.terminate();
  worker = new Worker(workerUrl);
  worker.postMessage(layoutSpec);
  worker.onmessage = (e) => {
    displaySelection("#graph");
    updateProgress(Status.COMPLETE);
    drawGraph(e.data);
    addTags(d3.select("#edge-labels").selectAll("g").data(e.data.edges).join("g").attr("transform", (e) => {
      // get a point near the end
      const [p1, p2] = e.value.points.slice(-2);
      const dx = p2.x-p1.x;
      const dy = p2.y-p1.y;
      // normalize to the unit vector
      const len = Math.sqrt(dx*dx + dy*dy);
      const ux = dx / len;
      const uy = dy / len;
      // avoid overlap with the arrowhead
      const offset = 17;
      const x = p2.x - ux * offset;
      const y = p2.y - uy * offset;
      return `translate(${x}, ${y})`
    }).attr("class", e => e.value.label.type).attr("id", e => `${e.v}-${e.w}`).datum(e => e.value.label.text));
    if (recenter) document.getElementById("zoom-to-fit-btn").click();
  };
  worker.onerror = (e) => {
    e.preventDefault();
    updateProgress(Status.ERR, "Error in graph layout:\n"+e.message);
  }
}

// ** profiler graph

function formatMicroseconds(ts, showUs=true) {
  const s = Math.floor(ts / 1e6), ms = Math.floor((ts % 1e6) / 1e3), us = Math.round(ts % 1e3);
  const parts = [];
  if (s) parts.push(`${s}s`);
  if (ms || (!showUs && !s)) parts.push(`${ms}ms`);
  if (showUs && (us || (!ms && !s))) parts.push(`${us}us`);
  return parts.join(' ');
}

function formatCycles(cycles) {
  const M = Math.floor(cycles / 1e6), K = Math.floor((cycles % 1e6) / 1e3), s = Math.round(cycles % 1e3);
  const parts = [];
  if (M) parts.push(`${M}M`);
  if (K || (!M && s)) parts.push(`${K}K`);
  if (s || (!M && !K)) parts.push(`${s}`);
  return parts.join(" ");
}

const formatUnit = (d, unit="") => d3.format(".3~s")(d)+unit;

const WAVE_COLORS = {VALU:"#ffffc0", SALU:"#cef263", LOAD:"#ffc0c0", STORE:"#4fa3cc", IMMEDIATE:"#f3b44a", BARRIER:"#d00000", JUMP:"#ffb703",
  JUMP_NO:"#fb8500", MESSAGE:"#90dbf4", VMEM:"#b2b7c9", LDS:"#9fb4a6"};
const waveColor = (op) => {
  const cat = op.includes("VALU") || op === "VINTERP" ? "VALU" : op.includes("SALU") ? "SALU" : op.includes("VMEM") ? "VMEM"
            : op.includes("LOAD") || op === "SMEM" ? "LOAD" : op.includes("STORE") ? "STORE" : op;
  ret = WAVE_COLORS[cat] ?? "#ffffff";
  if (op.includes("OTHER_") || op.includes("_ALT")) { ret = darkenHex(ret, 75) }
  return ret
};
const colorScheme = {TINY:new Map([["Schedule","#1b5745"],["get_program","#1d2e62"],["compile","#63b0cd"],["DEFAULT","#354f52"]]),
  DEFAULT:["#2b2e39", "#2c2f3a", "#31343f", "#323544", "#2d303a", "#2e313c", "#343746", "#353847", "#3c4050", "#404459", "#444862", "#4a4e65"],
  BUFFER:["#342483", "#3E2E94", "#4938A4", "#5442B4", "#5E4CC2", "#674FCA"], SIMD:new Map([["OCC", "#101725"], ["INST", "#0A2042"]]),
  WAVE:waveColor, VMEMEXEC:waveColor, ALUEXEC:waveColor}
const cycleColors = (lst, i) => lst[i%lst.length];

const rescaleTrack = (source, tid, k) => {
  for (const shapes of source.views)
    for (const e of shapes) {
      for (let i=0; i<e.y0.length; i++) {
        e.y0[i] = e.y0[i]*k;
        e.y1[i] = e.y1[i]*k;
      }
    }
  const change = (source.height*k)-source.height;
  const div = document.getElementById(tid);
  div.style.height = rect(div).height+change+"px";
  source.height = source.height*k;
  return change;
}

const drawLine = (ctx, x, y, opts) => {
  ctx.beginPath();
  ctx.moveTo(x[0], y[0]);
  ctx.lineTo(x[1], y[1]);
  ctx.fillStyle = ctx.strokeStyle = opts?.color || "#f0f0f5";
  ctx.stroke();
}

function tabulate(rows) {
  const root = d3.create("div").style("display", "grid").style("grid-template-columns", `${Math.max(...rows.map(x => x[0].length), 0)}ch 1fr`).style("gap", "0.2em");
  for (const [k,v] of rows) { root.append("div").text(k); root.append("div").node().append(v); }
  return root;
}

var data, focusedDevice, focusedShape, formatTime, canvasZoom, zoomLevel = d3.zoomIdentity;

function selectShape(key) {
  if (key == null) return {};
  const [t, idx] = key.split("-");
  const track = data.tracks.get(t);
  return { eventType:track?.eventType, e:track?.shapes[idx] };
}

const Modes = {0:'read', 1:'write', 2:'write+read'};

function getMetadata(key) {
  const { eventType, e } = selectShape(key);
  const html = d3.create("div").classed("info", true);
  if (eventType === EventTypes.EXEC) {
    const [n, _, ...rest] = e.arg.tooltipText.split("\n");
    html.append(() => tabulate([["Name", d3.create("p").html(n).node()], ["Duration", formatTime(e.width)], ["Start Time", formatTime(e.x)]]).node());
    let group = html.append("div").classed("args", true);
    for (const r of rest) group.append("p").text(r);
    group = html.append("div").classed("args", true);
    for (const b of e.arg.bufs.sort((a, b) => a.num - b.num)) {
      group.append("p").text(`${Modes[b.mode]}@data${b.num} ${formatUnit(b.nbytes, 'B')}`).style("cursor", "pointer").on("click", () => {
        const row = document.getElementById(b.k); if (!isExpanded(row)) { row.click(); }
        focusShape(b.key);
      });
    }
    if (e.arg.ctx != null) {
      const i = e.arg.ctx; s = e.arg.step;
      html.append("a").text(ctxs[i+1].steps[s].name).on("click", () => switchCtx(i, s));
      const prgSrc = ctxs[i+1].steps.findIndex(s => s.name === "View Program");
      if (prgSrc !== -1) html.append("a").text("View program").on("click", () => switchCtx(i, prgSrc));
    }
  }
  if (eventType === EventTypes.BUF) {
    const [dtype, sz, nbytes, dur] = e.arg.tooltipText.split("\n");
    const rows = [["DType", dtype], ["Len", sz], ["Size", nbytes], ["Lifetime", dur]];
    if (e.arg.users != null) rows.push(["Users", e.arg.users.length]);
    html.append(() => tabulate(rows).node());
    const kernels = html.append("div").classed("args", true);
    for (let u=0; u<e.arg.users?.length; u++) {
      const { repr, num, mode, shape } = e.arg.users[u];
      const p = kernels.append("p").append(() => colored(`[${u}] ${repr} ${Modes[mode]}@data${num}`));
      const shapeInfo = selectShape(shape).e?.arg?.tooltipText?.split("\n");
      if (shapeInfo?.length > 5) p.append("span").text(" "+shapeInfo[5]);
      if (shape != null) p.style("cursor", "pointer").on("click", () => focusShape(shape));
    }
  }
  return html.node();
}

function focusShape(shape) {
  saveToHistory({ shape:focusedShape });
  focusedShape = shape; d3.select("#timeline").call(canvasZoom.transform, zoomLevel);
  return metadata.replaceChildren(getMetadata(focusedShape));
}

const EventTypes = { EXEC:0, BUF:1 };

async function renderProfiler(path, unit, opts) {
  displaySelection("#profiler");
  // support non realtime x axis units
  formatTime = unit === "realtime" ? formatMicroseconds : formatCycles;
  if (data?.path !== path) { data = {tracks:new Map(), axes:{}, path, first:null}; focusedDevice = null; focusedShape = null; }
  metadata.replaceChildren(getMetadata(focusedShape));
  // layout once!
  if (data.tracks.size !== 0) return updateProgress(Status.COMPLETE);
  const profiler = d3.select("#profiler").html("");
  const buf = cache[path] ?? await fetchValue(path);
  const view = new DataView(buf);
  let offset = 0;
  const u8 = () => { const ret = view.getUint8(offset); offset += 1; return ret; }
  const u32 = () => { const ret = view.getUint32(offset, true); offset += 4; return ret; }
  const u64 = () => { const ret = new Number(view.getBigUint64(offset, true)); offset += 8; return ret; }
  const f32 = () => { const ret = view.getFloat32(offset, true); offset += 4; return ret; }
  const optional = (i) => i === 0 ? null : i-1;
  const dur = u32(), tracePeak = u64(), indexLen = u32(), layoutsLen = u32();
  const textDecoder = new TextDecoder("utf-8");
  const { strings, dtypeSize, markers }  = JSON.parse(textDecoder.decode(new Uint8Array(buf, offset, indexLen))); offset += indexLen;
  // place devices on the y axis and set vertical positions
  const [tickSize, padding, baseOffset] = [10, 8, markers.length ? 14 : 0];
  const deviceList = profiler.append("div").attr("id", "device-list").style("padding-top", tickSize+padding+baseOffset+"px");
  const canvas = profiler.append("canvas").attr("id", "timeline").node();
  // NOTE: scrolling via mouse can only zoom the graph
  canvas.addEventListener("wheel", e => (e.stopPropagation(), e.preventDefault()), { passive:false });
  const ctx = canvas.getContext("2d");
  const canvasTop = rect(canvas).top;
  // color by key (name/device)
  const colorMap = new Map();
  // map shapes by event key
  const shapeMap = new Map();
  const heightScale = d3.scaleLinear().domain([0, tracePeak]).range([4,maxheight=100]);
  for (let i=0; i<layoutsLen; i++) {
    const nameLen = view.getUint8(offset, true); offset += 1;
    const k = textDecoder.decode(new Uint8Array(buf, offset, nameLen)); offset += nameLen;
    const div = deviceList.append("div").attr("id", k).text(k).style("padding", padding+"px").style("width", opts.width);
    const { y:baseY, height:baseHeight } = rect(div.node());
    const colors = colorScheme[k.split(":")[0]] ?? colorScheme.DEFAULT;
    const offsetY = baseY-canvasTop+padding/2;
    const shapes = [], visible = [];
    const eventType = u8(), eventsLen = u32();
    const [pcolor, scolor] = path.includes("pkts") ? ["#00c72f", "#858b9d"] : ["#9ea2ad", null];
    // last row doesn't get a border
    const rowBorderColor = i<layoutsLen-1 ? "#22232a" : null;
    if (rowBorderColor != null) div.style("border-bottom", `1px solid ${rowBorderColor}`);
    if (eventType === EventTypes.EXEC) {
      const levelHeight = (baseHeight-padding)*(opts.heightScale ?? 1);
      const levels = [];
      data.tracks.set(k, { shapes, eventType, visible, offsetY, scolor, pcolor, rowBorderColor });
      let colorKey, ref;
      for (let j=0; j<eventsLen; j++) {
        const e = {name:strings[u32()], ref:optional(u32()), key:optional(u32()), st:u32(), dur:f32(), info:strings[u32()] || null};
        // find a free level to put the event
        let depth = 0;
        if (opts.levelKey != null) { depth = opts.levelKey(e); levels[depth] = 0; }
        else {
          depth = levels.findIndex(levelEt => e.st >= levelEt);
          const et = e.st+Math.trunc(e.dur);
          if (depth === -1) {
            depth = levels.length;
            levels.push(et);
          } else levels[depth] = et;
        }
        if (depth === 0 || opts.colorByName) colorKey = e.name.split(" ")[0];
        if (!colorMap.has(colorKey)) {
          const color = typeof colors === "function" ? colors(colorKey)
                      : colors instanceof Map ? (colors.get(colorKey) || colors.get("DEFAULT")) : cycleColors(colors, colorMap.size);
          colorMap.set(colorKey, d3.rgb(color));
        }
        const fillColor = colorMap.get(colorKey).brighter(0.3*depth).toString();
        const label = parseColors(e.name).map(({ color, st }) => ({ color, st, width:ctx.measureText(st).width }));
        let shapeRef = e.ref;
        if (shapeRef != null) { ref = {ctx:e.ref, step:0}; shapeRef = ref; }
        else if (ref != null) {
          const start = ref.step>0 ? ref.step+1 : 0;
          const stepIdx = ctxs[ref.ctx+1].steps.findIndex((s, i) => i >= start && s.name == e.name);
          if (stepIdx !== -1) { ref.step = stepIdx; shapeRef = ref; }
        } else {
          const steps = ctxs[state.currentCtx].steps;
          for (let i=state.currentStep+1; i<steps.length; i++) {
            const loc = steps[i].loc;
            if (loc == null) break;
            if (loc === e.name) { shapeRef = {ctx:state.currentCtx-1, step:i}; break; }
          }
        }
        // tiny device events go straight to the rewrite rule
        const key = k.startsWith("TINY") ? null : `${k}-${j}`;
        const labelHTML = label.map(l=>`<span style="color:${l.color}">${l.st}</span>`).join("");
        const arg = { tooltipText:labelHTML+"\n"+formatTime(e.dur)+(e.info != null ? "\n"+e.info : ""), bufs:[], key,
                      ctx:shapeRef?.ctx, step:shapeRef?.step };
        if (e.key != null) shapeMap.set(e.key, key);
        // offset y by depth
        shapes.push({x:e.st, y:levelHeight*depth, width:e.dur, height:levelHeight, arg, label:opts.hideLabels ? null : label, fillColor });
        if (j === 0) data.first = data.first == null ? e.st : Math.min(data.first, e.st);
      }
      div.style("height", levelHeight*levels.length+padding+"px").style("pointerEvents", "none");
    } else {
      const peak = u64();
      let x = 0, y = 0;
      const buf_shapes = new Map(), temp = new Map();
      const timestamps = [], valueMap = new Map();
      for (let j=0; j<eventsLen; j++) {
        const alloc = u8(), ts = u32(), key = u32();
        if (alloc) {
          const dtype = strings[u32()], sz = u64(), nbytes = dtypeSize[dtype]*sz;
          const shape = {x:[x], y:[y], dtype, sz, nbytes, key};
          buf_shapes.set(key, shape); temp.set(key, shape);
          timestamps.push(ts);
          x += 1; y += nbytes; valueMap.set(ts, y);
        } else {
          const free = buf_shapes.get(key);
          free.users = Array.from({ length: u32() }, () => ({shape:shapeMap.get(u32()), repr:strings[u32()], num:u32(), mode:u8()}));
          timestamps.push(ts); valueMap.set(ts, y);
          x += 1; y -= free.nbytes;
          free.x.push(x);
          free.y.push(free.y.at(-1));
          temp.delete(key);
          for (const [k, v] of temp) {
            if (k <= key) continue;
            v.x.push(x, x);
            v.y.push(v.y.at(-1), v.y.at(-1)-free.nbytes);
          }
        }
      }
      timestamps.push(dur);
      const height = heightScale(peak);
      const yscale = d3.scaleLinear().domain([0, peak]).range([height, 0]);
      for (const [num, {dtype, sz, nbytes, y, x:steps, users}] of buf_shapes) {
        const x = steps.map(s => timestamps[s]);
        const dur = x.at(-1)-x[0];
        const arg = { tooltipText:`${dtype}\n${formatUnit(sz)}\n${formatUnit(nbytes, 'B')}\n${formatTime(dur)}`, users, key:`${k}-${shapes.length}` };
        shapes.push({ x, y0:y.map(yscale), y1:y.map(y0 => yscale(y0+nbytes)), arg, fillColor:cycleColors(colorScheme.BUFFER, shapes.length) });
        users?.forEach((u) => selectShape(u.shape).e?.arg.bufs.push({ key:arg.key, nbytes, num:u.num, mode:u.mode, k }));
      }
      // generic polygon merger
      const base0 = yscale(0);
      const allX = Array.from(new Set(shapes.flatMap(s => s.x))).sort((a,b)=>a-b);
      const idxs = new Map(allX.map((x,i) => [x, i]));
      const maxY = new Map(allX.map(x => [x, base0]));
      // for every [a,b) update the max y at x
      for (const sh of shapes) {
        for (let i=0; i<sh.x.length-1; i++) {
          const startIdx = idxs.get(sh.x[i]), endIdx = idxs.get(sh.x[i+1]);
          const shapeY = sh.y1[i];
          for (let k=startIdx; k<endIdx; k++) {
            const x = allX[k]; maxY.set(x, Math.min(maxY.get(x), shapeY));
          }
        }
      }
      const sum = {x:[], y0:[], y1:[], fillColor:"#2B1B72"};
      for (let i=0; i<allX.length-1; i++) {
        sum.x.push(allX[i], allX[i+1]);
        const y = maxY.get(allX[i]); sum.y1.push(y, y); sum.y0.push(base0, base0);
      }
      if (timestamps.length > 0) data.first = data.first == null ? timestamps[0] : Math.min(data.first, timestamps[0]);
      data.tracks.set(k, { shapes:[sum], eventType, visible, offsetY, pcolor:"#c9a8ff", height, peak, scaleFactor:maxheight*4/height,
                           views:[[sum], shapes], valueMap, rowBorderColor });
      div.style("height", height+padding+"px").style("cursor", "pointer").on("click", (e) => {
        const newFocus = e.currentTarget.id === focusedDevice ? null : e.currentTarget.id;
        let offset = 0;
        for (const [tid, track] of data.tracks) {
          track.offsetY += offset;
          if (tid === newFocus) { track.shapes = track.views[1]; offset += rescaleTrack(track, tid, track.scaleFactor); }
          else if (tid === focusedDevice) { track.shapes = track.views[0]; offset += rescaleTrack(track, tid, 1/track.scaleFactor); }
        }
        data.axes.y = newFocus != null ? { domain:[0, (t=data.tracks.get(newFocus)).peak], range:[t.offsetY+t.height, t.offsetY], fmt:"B" } : null;
        toggleCls(document.getElementById(focusedDevice), document.getElementById(newFocus), "expanded");
        focusedDevice = newFocus;
        return resize();
      });
    }
  }
  for (const m of markers) m.label = m.name.split(/(\s+)/).map(st => ({ st, color:m.color, width:ctx.measureText(st).width }));
  updateProgress(Status.COMPLETE);
  // draw events on a timeline
  const dpr = window.devicePixelRatio || 1;
  const ellipsisWidth = ctx.measureText("...").width;
  const drawText = (ctx, label, lx, ly, maxWidth) => {
    let lw = 0;
    for (let li=0; li<label?.length; li++) {
      if (lw+label[li].width+(li===label.length-1 ? 0 : ellipsisWidth)+2 > maxWidth) {
        if (lw>0) ctx.fillText("...", lx+lw, ly);
        break;
      }
      ctx.fillStyle = label[li].color;
      ctx.fillText(label[li].st, lx+lw, ly);
      lw += label[li].width;
    }
  }
  function render(transform) {
    zoomLevel = transform;
    const canvasWidth = canvas.clientWidth;
    ctx.clearRect(0, 0, canvasWidth, canvas.clientHeight);
    // rescale to match current zoom
    const xscale = d3.scaleLinear().domain([data.first, dur]).range([0, canvasWidth]);
    const visibleX = xscale.range().map(zoomLevel.invertX, zoomLevel).map(xscale.invert, xscale);
    const st = visibleX[0], et = visibleX[1];
    xscale.domain([st, et]);
    ctx.textBaseline = "middle";
    // draw shapes
    for (const [k, { shapes, eventType, visible, offsetY, valueMap, pcolor, scolor, rowBorderColor }] of data.tracks) {
      visible.length = 0;
      const addBorder = scolor != null ? (w) => { if (w > 10) { ctx.strokeStyle = scolor; ctx.stroke(); } } : null;
      for (const e of shapes) {
        if (eventType === EventTypes.BUF) { // generic polygon
          if (e.x[0]>et || e.x.at(-1)<st) continue;
          ctx.beginPath();
          const x = e.x.map(xscale);
          ctx.moveTo(x[0], offsetY+e.y0[0]);
          for (let i=1; i<x.length; i++) {
            ctx.lineTo(x[i], offsetY+e.y0[i]);
            let arg = e.arg;
            if (arg == null && valueMap != null) arg = {tooltipText: `Total: ${formatUnit(valueMap.get(e.x[i-1]), 'B')}`}
            visible.push({ x0:x[i-1], x1:x[i], y0:offsetY+e.y1[i-1], y1:offsetY+e.y0[i], arg });
          }
          for (let i=x.length-1; i>=0; i--) ctx.lineTo(x[i], offsetY+e.y1[i]);
          ctx.closePath();
          ctx.fillStyle = e.fillColor; ctx.fill();
        } else { // contiguous rect
          if (e.x>et || e.x+e.width<st) continue;
          const x = xscale(e.x);
          const y = offsetY+e.y;
          const width = xscale(e.x+e.width)-x;
          ctx.beginPath();
          ctx.rect(x, y, width, e.height);
          visible.push({ y0:y, y1:y+e.height, x0:x, x1:x+width, arg:e.arg });
          ctx.fillStyle = e.fillColor; ctx.fill();
          addBorder?.(width);
          // add label
          drawText(ctx, e.label, x+2, y+e.height/2, width);
        }
        if (focusedShape != null && e.arg?.key === focusedShape) { ctx.strokeStyle = pcolor; ctx.stroke(); }
      }
      // draw row line
      if (rowBorderColor != null) {
        const y = offsetY+rect(document.getElementById(k)).height-padding/2 - 0.5;
        drawLine(ctx, [0, canvasWidth], [y, y], { color:rowBorderColor });
      }
    }
    // draw axes
    ctx.translate(0, baseOffset);
    drawLine(ctx, xscale.range(), [0, 0]);
    let lastLabelEnd = -Infinity;
    for (const tick of xscale.ticks()) {
      const x = xscale(tick);
      drawLine(ctx, [x, x], [0, tickSize]);
      const labelX = x+ctx.lineWidth+2;
      if (labelX <= lastLabelEnd) continue;

      const label = formatTime(tick, et-st <= 1e3 ? true : false);
      ctx.textBaseline = "top";
      ctx.fillText(label, labelX, tickSize);
      lastLabelEnd = labelX + ctx.measureText(label).width + 4;
    }
    if (data.axes.y != null) {
      drawLine(ctx, [0, 0], data.axes.y.range);
      const yscale = d3.scaleLinear().domain(data.axes.y.domain).range(data.axes.y.range);
      for (const tick of yscale.ticks()) {
        const y = yscale(tick);
        drawLine(ctx, [0, tickSize], [y, y]);
        ctx.textBaseline = "middle";
        ctx.fillText(formatUnit(tick, data.axes.y.fmt), tickSize+2, y);
      }
    }
    // draw markers
    ctx.translate(0, -baseOffset);
    ctx.textBaseline = "top";
    for (let i=0; i<markers.length; i++) {
      const m = markers[i];
      const x = xscale(m.ts), tx = x+2;
      drawLine(ctx, [x, x], [0, canvas.clientHeight], { color:m.color });
      let maxWidth = canvasWidth-(tx);
      const nextMark = markers[i+1]?.ts;
      if (nextMark != null) maxWidth = Math.min(maxWidth, xscale(nextMark)-tx);
      if (maxWidth <= 0) continue;
      drawText(ctx, m.label, tx, 1, maxWidth);
    }
  }

  function resize() {
    const profiler = document.querySelector("#profiler");
    const sideRect = rect("#device-list");
    const width = profiler.clientWidth-(sideRect.width+padding), height = Math.round(sideRect.height);
    if (canvas.width === width*dpr && canvas.height === height*dpr) return;
    canvas.width = width*dpr;
    canvas.height = height*dpr;
    canvas.style.height = `${height}px`;
    canvas.style.width = `${width}px`;
    ctx.scale(dpr, dpr);
    d3.select(canvas).call(canvasZoom.transform, zoomLevel);
  }

  zoomLevel = d3.zoomIdentity;
  canvasZoom = d3.zoom().filter(vizZoomFilter).on("zoom", e => render(e.transform));
  d3.select(canvas).call(canvasZoom);
  document.addEventListener("contextmenu", e => e.ctrlKey && e.preventDefault());

  new ResizeObserver(([e]) => e.contentRect.width > 0 && resize()).observe(profiler.node());

  function findRectAtPosition(x, y) {
    let track = null;
    for (const k of data.tracks.keys()) {
      const r = rect(document.getElementById(k));
      if (y >= r.y && y <= r.y+r.height) { track = data.tracks.get(k); break; }
    }
    if (track == null) return;
    const R = rect(canvas);
    const X = ((x-R.left) * (canvas.width/R.width))/dpr;
    const Y = ((y-R.top) * (canvas.height/R.height))/dpr;
    for (const r of track.visible) {
      if (Y>=r.y0 && Y<=r.y1 && X>=r.x0 && X<=r.x1) return r.arg;
    }
  }

  const clickShape = (e) => {
    e.preventDefault();
    const foundRect = findRectAtPosition(e.clientX, e.clientY);
    if (foundRect?.step != null && (foundRect?.key == null || e.type == "dblclick")) { return switchCtx(foundRect.ctx, foundRect.step); }
    if (foundRect?.key != focusedShape) { focusShape(foundRect?.key); }
  }
  canvas.addEventListener("click", clickShape);
  canvas.addEventListener("dblclick", clickShape);

  canvas.addEventListener("mousemove", e => {
    const foundRect = findRectAtPosition(e.clientX, e.clientY);
    if (foundRect?.tooltipText != null) {
      const tooltip = document.getElementById("tooltip");
      tooltip.style.display = "block";
      tooltip.style.left = (e.pageX+10)+"px";
      tooltip.style.top = (e.pageY)+"px";
      tooltip.innerHTML = foundRect.tooltipText;
    } else tooltip.style.display = "none";
  });
  canvas.addEventListener("mouseleave", () => document.getElementById("tooltip").style.display = "none");
}

// ** zoom and recentering

const vizZoomFilter = e => (!e.ctrlKey || e.type === 'wheel' || e.type === 'mousedown') && !e.button && e.type !== 'dblclick';
const svgZoom = d3.zoom().filter(vizZoomFilter).on("zoom", (e) => d3.select("#render").attr("transform", e.transform));
d3.select("#graph-svg").call(svgZoom);

// zoom to fit into view
document.getElementById("zoom-to-fit-btn").addEventListener("click", () => {
  const canvas = d3.select("#timeline");
  if (!canvas.empty() && rect(canvas.node()).width !== 0) {
    return canvas.call(canvasZoom.transform, d3.zoomIdentity);
  }
  const svg = d3.select("#graph-svg");
  svg.call(svgZoom.transform, d3.zoomIdentity);
  const mainRect = rect(".main-container");
  const x0 = rect(".ctx-list-parent").right;
  const x1 = rect(".metadata-parent").left;
  const pad = 16;
  const R = { x: x0+pad, y: mainRect.top+pad, width: (x1>0 ? x1-x0 : mainRect.width)-2*pad, height: mainRect.height-2*pad };
  const r = rect("#render");
  if (r.width === 0) return;
  const scale = Math.min(R.width/r.width, R.height/r.height);
  const [tx, ty] = [R.x+(R.width-r.width*scale)/2-r.left*scale, R.y+(R.height-r.height*scale)/2];
  svg.call(svgZoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
});

// **** main VIZ interfacae

const pathLink = (fp, lineno) => d3.create("a").attr("href", "vscode://file/"+fp+":"+lineno).text(`${fp.split("/").at(-1)}:${lineno}`);
function codeBlock(st, language, { loc, wrap }={}) {
  const code = document.createElement("code");
  // plaintext renders like a terminal print, otherwise render with syntax highlighting
  if (!language || language === "txt") code.appendChild(colored(st));
  else code.innerHTML = hljs.highlight(st, { language }).value;
  code.className = "hljs";
  const ret = document.createElement("pre");
  if (wrap) ret.className = "wrap";
  if (loc != null) ret.appendChild(pathLink(loc[0], loc[1]).style("margin-bottom", "4px").node());
  ret.appendChild(code);
  return ret;
}

function toggleCls(prev, next, cls, value) {
  prev?.classList.remove(cls);
  next?.classList.toggle(cls, value ?? true);
  requestAnimationFrame(() => next?.scrollIntoView({ behavior: "auto", block: "nearest" }));
}

// ** hljs extra definitions for UOps and float4
hljs.registerLanguage("python", (hljs) => ({
  ...hljs.getLanguage("python"),
  case_insensitive: false,
  contains: [
    { begin: 'dtypes\\.[a-zA-Z_][a-zA-Z0-9_-]*(\\.[a-zA-Z_][a-zA-Z0-9_-]*)*' + '(?=[.\\s\\n[:,(])', className: "type" },
    { begin: 'dtypes\\.[a-zA-Z_][a-zA-Z0-9_-].vec*' + '(?=[.\\s\\n[:,(])', className: "type" },
    { begin: '[a-zA-Z_][a-zA-Z0-9_-]*\\.[a-zA-Z_][a-zA-Z0-9_-]*' + '(?=[.\\s\\n[:,()])',  className: "operator" },
    { begin: '[A-Z][a-zA-Z0-9_]*(?=\\()', className: "section", ignoreEnd: true },
    ...hljs.getLanguage("python").contains,
  ]
}));
hljs.registerLanguage("cpp", (hljs) => ({
  ...hljs.getLanguage('cpp'),
  contains: [{ begin: '\\b(?:float|half)[0-9]+\\b', className: 'type' }, ...hljs.getLanguage('cpp').contains]
}));

async function fetchValue(path) {
  const res = await fetch(path);
  return (await (res.headers.get("content-type") === "application/json" ? res.json() : res.arrayBuffer()));
}

var ret = [];
var cache = {};
var ctxs = null;
const evtSources = [];
// VIZ displays graph rewrites in 3 levels, from bottom-up:
// rewrite: a single UOp transformation
// step: collection of rewrites
// context: collection of steps
const state = {currentCtx:-1, currentStep:0, currentRewrite:0, expandSteps:false};
function setState(ns) {
  saveToHistory(state);
  const { ctx:prevCtx, step:prevStep } = select(state.currentCtx, state.currentStep);
  const prevRewrite = state.currentRewrite;
  Object.assign(state, ns);
  // update element styles if needed
  const { ctx, step } = select(state.currentCtx, state.currentStep);
  toggleCls(prevCtx, ctx, "expanded", state.expandSteps);
  if (ctx?.id !== prevCtx?.id) {
    toggleCls(prevCtx, ctx, "active");
  }
  if (ctx?.id !== prevCtx?.id || step?.id !== prevStep?.id) {
    toggleCls(prevStep, step, "active");
    // walk the tree back until all parents expanded so that the child is visible
    let e = step;
    while (e?.parentElement?.id.startsWith("step")) {
      e.parentElement.classList.add("expanded");
      e = e.parentElement;
    }
  }
  // re-render
  main();
}

const getSubrewrites = (ul) => ul.querySelectorAll(":scope > ul");

function saveToHistory(ns) {
  // NOTE: browser does a structured clone, passing a mutable object is safe.
  history.replaceState(ns, "");
  history.pushState(ns, "");
}

// switch to the start of a new graph and expand all the steps
const switchCtx = (newCtx, step) => setState({ expandSteps:true, currentCtx:newCtx+1, currentStep:step ?? 0, currentRewrite:0 });

window.addEventListener("popstate", (e) => {
  if (e.state?.shape != null) return focusShape(e.state?.shape);
  if (e.state != null) setState(e.state);
});

const createToggle = (id, text) => {
  const label = d3.create("label").text(text).node();
  const toggle = d3.create("input").attr("type", "checkbox").attr("id", id).property("checked", true).node();
  label.prepend(toggle);
  return { toggle, label };
}
const { toggle, label:toggleLabel } = createToggle("show-indexing", "Show indexing (r)");
const showGraph = createToggle("show-graph", "Show graph (g)");
showGraph.toggle.onchange = () => displaySelection(rect("#graph").width > 0 ? "#custom" : "#graph");

function appendSteps(root, idx, steps) {
  const stack = [];
  for (const [j,u] of steps.entries()) {
    while (stack.length && stack.at(-1).depth >= u.depth) stack.pop();
    const list = stack.length > 0 ? stack.at(-1).li : root;
    u.li = list.appendChild(document.createElement("ul"));
    u.li.id = `step-${idx}-${j}`
    const p = u.li.appendChild(document.createElement("p"));
    p.appendChild(colored(`${u.name}`+(u.match_count ? ` - ${u.match_count}` : '')));
    p.onclick = (e) => {
      e.stopPropagation();
      const subrewrites = getSubrewrites(e.currentTarget.parentElement);
      if (subrewrites.length) { e.currentTarget.parentElement.classList.toggle("expanded"); }
      setState({ currentStep:j, currentCtx:idx, currentRewrite:0 });
    }
    stack.push(u);
  }
  for (const l of root.querySelectorAll("ul > ul > p")) {
    const subrewrites = getSubrewrites(l.parentElement);
    if (subrewrites.length > 0) { l.appendChild(d3.create("span").text(` (${subrewrites.length})`).node()); l.parentElement.classList.add("has-children"); }
  }
}

async function main() {
  // ** left sidebar context list
  if (ctxs == null) {
    ctxs = [{ name:"Profiler", steps:[] }];
    for (const r of await fetchValue("/ctxs")) ctxs.push(r);
    const ctxList = document.querySelector(".ctx-list");
    for (const [i,{name, steps}] of ctxs.entries()) {
      const ul = ctxList.appendChild(document.createElement("ul"));
      ul.id = `ctx-${i}`;
      const p = ul.appendChild(document.createElement("p"));
      p.appendChild(colored(name));
      p.onclick = () => {
        setState(i === state.currentCtx ? { expandSteps:!state.expandSteps } : { expandSteps:true, currentCtx:i, currentStep:0, currentRewrite:0 });
      }
      appendSteps(ul, i, steps);
    }
    return setState({ currentCtx:-1 });
  }
  // ** center graph
  const { currentCtx, currentStep, currentRewrite, expandSteps } = state;
  if (currentCtx == -1) return;
  const ctx = ctxs[currentCtx];
  const step = ctx.steps[currentStep];
  const ckey = step?.query;
  // close any pending event sources
  let activeSrc = null;
  for (const e of evtSources) {
    const url = new URL(e.url);
    if (url.pathname+url.search !== ckey) e.close();
    else if (e.readyState === EventSource.OPEN) activeSrc = e;
  }
  if (ctx.name === "Profiler") return renderProfiler("/get_profile", "realtime", { width:"132px" });
  if (workerUrl == null) await initWorker();
  if (ckey in cache) {
    ret = cache[ckey];
  }
  if (!ckey.startsWith("/graph")) {
    if (!(ckey in cache)) cache[ckey] = ret = await fetchValue(ckey);
    if (ret.steps?.length > 0) {
      const el = select(state.currentCtx, state.currentStep);
      if (el.step.querySelectorAll("ul").length === ret.steps.length) return;
      // re render the list with new items
      ctx.steps.push(...ret.steps);
      while (el.ctx.children.length > 1) el.ctx.children[1].remove();
      appendSteps(el.ctx, state.currentCtx, ctx.steps);
      return setState({ currentStep:state.currentStep+1, expandSteps:true });
    }
    // timeline with cycles on the x axis
    if (ret instanceof ArrayBuffer) {
      opts = {heightScale:0.5, hideLabels:true, levelKey:(e) => parseInt(e.name.split(" ")[1].split(":")[1]), colorByName:step.name.includes("PKTS")};
      return renderProfiler(ckey, "clk", opts);
    }
    metadata.innerHTML = "";
    ret.metadata?.forEach(m => {
      if (Array.isArray(m)) return metadata.appendChild(tabulate(m.map(({ label, value }) => {
        return [label.trim(), typeof value === "string" ? value : formatUnit(value)];
      })).node());
      metadata.appendChild(codeBlock(m.src)).classList.add("full-height")
    });
    // graph render
    if (ret.data != null) {
      metadata.prepend(showGraph.label);
      renderDag(ret, { recenter:true });
    } else displaySelection("#custom");
    // table / plaintext render
    const root = d3.create("div").classed("raw-text", true);
    function renderTable(root, ret) {
      const table = root.append("table");
      const thead = table.append("thead");
      for (const c of ret.cols) thead.append("th").text(c.title ?? c);
      for (const r of ret.rows) {
        const tr = table.append("tr").classed("main-row", true);
        for (const [i,value] of r.entries()) {
          // nested table
          if (value.cols != null) {
            tr.classed("has-children", true);
            tr.on("click", () => {
              const el = tr.node().nextElementSibling;
              if (el?.classList.contains("nested-row")) { tr.classed("expanded", false); return el.remove(); }
              tr.classed("expanded", true);
              const td = table.insert("tr", () => tr.node().nextSibling).classed("nested-row", true).append("td");
              td.attr("colSpan", ret.cols.length);
              renderTable(td, value);
            });
            continue;
          }
          const td = tr.append("td").classed(ret.cols[i], true);
          // string format scalar values
          td.append(() => typeof value === "string" ? colored(value) : d3.create("p").text(ret.cols[i] === "Duration" ? formatMicroseconds(value) : formatUnit(value)).node());
        }
      }
      return table;
    }
    if (ret.ref != null) {
      const disasmIdx = ctxs[ret.ref+1].steps.findIndex(s => s.name === "View Disassembly")
      metadata.appendChild(d3.create("a").text("View Program Graph").on("click", () => switchCtx(ret.ref, disasmIdx)).node());
    }
    if (ret.cols != null) renderTable(root, ret);
    else if (ret.src != null) root.append(() => codeBlock(ret.src, ret.lang));
    return document.querySelector("#custom").replaceChildren(root.node());
  }
  // ** Graph view
  // if we don't have a complete cache yet we start streaming graphs in this step
  if (!(ckey in cache) || (cache[ckey].length !== step.match_count+1 && activeSrc == null)) {
    ret = [];
    cache[ckey] = ret;
    const eventSource = new EventSource(ckey);
    evtSources.push(eventSource);
    eventSource.onmessage = (e) => {
      if (e.data === "[DONE]") return eventSource.close();
      const chunk = JSON.parse(e.data);
      ret.push(chunk);
      // if it's the first one render this new rgaph
      if (ret.length === 1) return main();
      // otherwise just enable the graph selector
      const ul = document.getElementById(`rewrite-${ret.length-1}`);
      if (ul != null) ul.classList.remove("disabled");
    };
  }
  if (ret.length === 0) return;
  // ** center graph
  const data = ret[currentRewrite];
  const render = (opts) => renderDag({ data, opts }, { recenter:currentRewrite === 0 });
  render({ showIndexing:toggle.checked });
  toggle.onchange = (e) => render({ showIndexing:e.target.checked });
  // ** right sidebar metadata
  metadata.innerHTML = "";
  if (ckey.includes("rewrites")) metadata.appendChild(toggleLabel);
  if (step.code_line != null) metadata.appendChild(codeBlock(step.code_line, "python", { loc:step.loc, wrap:true }));
  if (step.trace) {
    const trace = d3.create("pre").append("code").classed("hljs", true);
    for (let i=step.trace.length-1; i>=0; i--) {
      const [fp, lineno, fn, code] = step.trace[i];
      trace.append("div").style("margin-bottom", "2px").style("display","flex").text(fn+" ").append(() => pathLink(fp, lineno).node());
      trace.append("div").html(hljs.highlight(code, { language: "python" }).value).style("margin-bottom", "1ex");
      metadata.appendChild(trace.node().parentNode);
    }
  }
  if (data.uop != null) metadata.appendChild(codeBlock(data.uop, "python", { wrap:false })).classList.toggle("full-height", step.match_count === 0);
  // ** multi graph in one page
  if (!step.match_count) return;
  const rewriteList = metadata.appendChild(document.createElement("div"));
  rewriteList.className = "rewrite-list";
  for (let s=0; s<=step.match_count; s++) {
    const ul = rewriteList.appendChild(document.createElement("ul"));
    ul.id = `rewrite-${s}`;
    const p = ul.appendChild(document.createElement("p"));
    p.innerText = s;
    ul.onclick = () => setState({ currentRewrite:s });
    ul.className = s > ret.length-1 ? "disabled" : s === currentRewrite ? "active" : "";
    if (s > 0 && s === currentRewrite) {
      const { upat, diff } = ret[s];
      metadata.appendChild(codeBlock(upat[1], "python", { loc:upat[0], wrap:true }));
      const diffCode = metadata.appendChild(document.createElement("pre")).appendChild(document.createElement("code"));
      for (const line of diff) {
        diffCode.appendChild(colored([{st:line, color:line.startsWith("+") ? "#3aa56d" : line.startsWith("-") ? "#d14b4b" : "#f0f0f5"}]));
        diffCode.appendChild(document.createElement("br"));
      }
      diffCode.className = "wrap";
    }
  }
}

// **** collapse/expand

let isCollapsed = false;
document.querySelector(".collapse-btn").addEventListener("click", (e) => {
  isCollapsed = !isCollapsed;
  document.querySelector(".main-container").classList.toggle("collapsed", isCollapsed);
  e.currentTarget.blur();
  e.currentTarget.style.transform = isCollapsed ? "rotate(180deg)" : "rotate(0deg)";
  window.dispatchEvent(new Event("resize"));
});

// **** resizer

function appendResizer(element, { minWidth, maxWidth }, left=false) {
  const handle = Object.assign(document.createElement("div"), { className: "resize-handle", style: left ? "right: 0" : "left: 0; margin-top: 0" });
  element.appendChild(handle);
  const resize = (e) => {
    const change = e.clientX - element.dataset.startX;
    let newWidth = ((Number(element.dataset.startWidth)+(left ? change : -change))/Number(element.dataset.containerWidth))*100;
    element.style.width = `${Math.max(minWidth, Math.min(maxWidth, newWidth))}%`;
  };
  handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    element.dataset.startX = e.clientX;
    element.dataset.containerWidth = rect(".main-container").width;
    element.dataset.startWidth = element.getBoundingClientRect().width;
    document.documentElement.addEventListener("mousemove", resize, false);
    document.documentElement.addEventListener("mouseup", () => {
      document.documentElement.removeEventListener("mousemove", resize, false);
      element.style.userSelect = "initial";
    }, { once: true });
  });
}
appendResizer(document.querySelector(".ctx-list-parent"), { minWidth: 15, maxWidth: 50 }, left=true);
appendResizer(document.querySelector(".metadata-parent"), { minWidth: 20, maxWidth: 50 });

// **** keyboard shortcuts

const select = (ctx, step) => ({ ctx:document.getElementById(`ctx-${ctx}`), step:document.getElementById(`step-${ctx}-${step}`) });
const deselect = (element) => {
  const parts = element?.id.split("-").map(Number);
  return element?.id.startsWith("ctx") ? { ctx:parts[1], step:null } : element?.id.startsWith("step") ? {ctx:parts[1], step:parts[2]} : {};
}
const isExpanded = (el) => el?.classList.contains("expanded");

document.addEventListener("keydown", (event) => {
  const { currentCtx, currentStep, currentRewrite, expandSteps } = state;
  // up and down change the step or context from the list
  const changeStep = expandSteps && ctxs[currentCtx].steps?.length;
  const { step, ctx } = select(currentCtx, currentStep);
  if (event.key == "ArrowUp") {
    event.preventDefault();
    if (changeStep) {
      let prev = deselect(step.previousElementSibling);
      if (prev.step == null && isExpanded(step.parentElement)) prev = deselect(step.parentElement);
      return prev.step != null && !isExpanded(step) && setState({ currentRewrite:0, currentStep:prev.step });
    }
    return setState({ currentStep:0, currentRewrite:0, currentCtx:Math.max(0, currentCtx-1), expandSteps:false });
  }
  if (event.key == "ArrowDown") {
    event.preventDefault();
    if (changeStep) {
      const next = deselect(isExpanded(step) ? step.children[1] : step.nextElementSibling);
      return next.step != null && setState({ currentRewrite:0, currentStep:next.step });
    }
    return setState({ currentStep:0, currentRewrite:0, currentCtx:Math.min(ctxs.length-1, currentCtx+1), expandSteps:false });
  }
  // enter toggles focus on a single rewrite stage
  if (event.key == "Enter") {
    event.preventDefault()
    if (currentCtx === -1) {
      return setState({ currentCtx:0, expandSteps:true });
    }
    if (expandSteps && getSubrewrites(step).length) return step.children[0].click();
    return setState({ expandSteps:!expandSteps });
  }
  // left and right go through rewrites in a single UOp
  if (event.key == "ArrowLeft") {
    event.preventDefault()
    return setState({ currentRewrite:Math.max(0, currentRewrite-1) });
  }
  if (event.key == "ArrowRight") {
    event.preventDefault()
    const totalRewrites = ret.length-1;
    return setState({ currentRewrite:Math.min(totalRewrites, currentRewrite+1) });
  }
  // space recenters the graph
  if (event.key == " ") {
    event.preventDefault()
    document.getElementById("zoom-to-fit-btn").click();
  }
  // r key toggles indexing
  if (event.key === "r") toggle.click();
  // g key toggles graph
  if (event.key === "g") showGraph.toggle.click();
});

main()
