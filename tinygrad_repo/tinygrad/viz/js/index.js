// ** graph helpers

const displayGraph = (cls) => {
  for (const e of document.getElementsByClassName("view")) e.style.display = e.classList.contains(cls) ? "flex" : "none";
}

const darkenHex = (h, p = 0) =>
  `#${(
    c = parseInt(h.slice(1), 16),
    f = 1 - p / 100,
    ((c >> 16 & 255) * f | 0) << 16 |
    ((c >>  8 & 255) * f | 0) <<  8 |
    ((c       & 255) * f | 0)
  ).toString(16).padStart(6, '0')}`;

const ANSI_COLORS = ["#b3b3b3", "#ff6666", "#66b366", "#ffff66", "#6666ff", "#ff66ff", "#66ffff", "#ffffff"];
const parseColors = (name, defaultColor="#ffffff") => Array.from(name.matchAll(/(?:\u001b\[(\d+)m([\s\S]*?)\u001b\[0m)|([^\u001b]+)/g),
  ([_, code, colored_st, st]) => ({ st: colored_st ?? st, color: code != null ? ANSI_COLORS[(parseInt(code)-30+60)%60] : defaultColor }));

const rect = (s) => (typeof s === "string" ? document.querySelector(s) : s).getBoundingClientRect();

let timeout = null;
const updateProgress = ({ start }) => {
  clearTimeout(timeout);
  const msg = document.getElementById("progress-message");
  msg.style.display = "none";
  if (start) {
    msg.innerText = "Rendering new graph...";
    timeout = setTimeout(() => { msg.style.display = "block"; }, 2000);
  }
}

// ** UOp graph

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

let [workerUrl, worker] = [null, null];
async function initWorker() {
  const resp = await Promise.all(["/assets/dagrejs.github.io/project/dagre/latest/dagre.min.js","/js/worker.js"].map(u => fetch(u)));
  workerUrl = URL.createObjectURL(new Blob([(await Promise.all(resp.map((r) => r.text()))).join("\n")], { type: "application/javascript" }));
}

function renderDag(graph, additions, recenter) {
  // start calculating the new layout (non-blocking)
  updateProgress({ start:true });
  if (worker != null) worker.terminate();
  worker = new Worker(workerUrl);
  worker.postMessage({graph, additions});
  worker.onmessage = (e) => {
    displayGraph("graph");
    updateProgress({ start:false });
    const g = dagre.graphlib.json.read(e.data);
    // draw nodes
    const STROKE_WIDTH = 1.4;
    d3.select("#graph-svg").on("click", () => d3.selectAll(".highlight").classed("highlight", false));
    const nodes = d3.select("#nodes").selectAll("g").data(g.nodes().map(id => g.node(id)), d => d).join("g").attr("class", d => d.className ?? "node")
      .attr("transform", d => `translate(${d.x},${d.y})`).classed("clickable", d => d.ref != null).on("click", (e,d) => {
        if (d.ref != null) return setCtxWithHistory(d.ref);
        const parents = g.predecessors(d.id);
        const children = g.successors(d.id);
        if (parents == null && children == null) return;
        const src = [...parents, ...children, d.id];
        nodes.classed("highlight", n => src.includes(n.id)).classed("child", n => children.includes(n.id));
        const matchEdge = (v, w) => (v===d.id && children.includes(w)) ? "highlight child "  : (parents.includes(v) && w===d.id) ? "highlight " : "";
        d3.select("#edges").selectAll("path.edgePath").attr("class", e => matchEdge(e.v, e.w)+"edgePath");
        d3.select("#edge-labels").selectAll("g.port").attr("class",  (_, i, n) => matchEdge(...n[i].id.split("-"))+"port");
        e.stopPropagation();
      });
    nodes.selectAll("rect").data(d => [d]).join("rect").attr("width", d => d.width).attr("height", d => d.height).attr("fill", d => d.color)
      .attr("x", d => -d.width/2).attr("y", d => -d.height/2);
    nodes.selectAll("g.label").data(d => [d]).join("g").attr("class", "label").attr("transform", d => {
      const x = (d.width-d.padding*2)/2;
      const y = (d.height-d.padding*2)/2+STROKE_WIDTH;
      return `translate(-${x}, -${y})`;
    }).selectAll("text").data(d => {
      const ret = [[]];
      for (const { st, color } of parseColors(d.label, defaultColor="initial")) {
        for (const [i, l] of st.split("\n").entries()) {
          if (i > 0) ret.push([]);
          ret.at(-1).push({ st:l, color });
        }
      }
      return [ret];
    }).join("text").selectAll("tspan").data(d => d).join("tspan").attr("x", "0").attr("dy", 14).selectAll("tspan").data(d => d).join("tspan")
      .attr("fill", d => darkenHex(d.color, 25)).text(d => d.st).attr("xml:space", "preserve");
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
    }).attr("marker-end", "url(#arrowhead)");
    addTags(d3.select("#edge-labels").selectAll("g").data(edges).join("g").attr("transform", (e) => {
      // get a point near the end
      const [p1, p2] = g.edge(e).points.slice(-2);
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
    }).attr("class", e => g.edge(e).label.type).attr("id", e => `${e.v}-${e.w}`).datum(e => g.edge(e).label.text));
    if (recenter) document.getElementById("zoom-to-fit-btn").click();
  };
}

// ** profiler graph

function formatTime(ts, dur=ts) {
  if (dur<=1e3) return `${ts.toFixed(2)}us`;
  if (dur<=1e6) return `${(ts*1e-3).toFixed(2)}ms`;
  return `${(ts*1e-6).toFixed(2)}s`;
}
const formatUnit = (d, unit="") => d3.format(".3~s")(d)+unit;

const colorScheme = {TINY:["#1b5745", "#354f52", "#354f52", "#1d2e62", "#63b0cd"],
  DEFAULT:["#2b2e39", "#2c2f3a", "#31343f", "#323544", "#2d303a", "#2e313c", "#343746", "#353847", "#3c4050", "#404459", "#444862", "#4a4e65"],
  BUFFER:["#3A57B7","#5066C1","#6277CD","#7488D8","#8A9BE3","#A3B4F2"],
  CATEGORICAL:["#ff8080", "#F4A261", "#C8F9D4", "#8D99AE", "#F4A261", "#ffffa2", "#ffffc0", "#87CEEB"],}
const cycleColors = (lst, i) => lst[i%lst.length];

const rescaleTrack = (source, tid, k) => {
  for (const e of source.shapes) {
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

var data, focusedDevice, canvasZoom, zoomLevel = d3.zoomIdentity;
async function renderProfiler() {
  displayGraph("profiler");
  d3.select(".metadata").html("");
  // layout once!
  if (data != null) return updateProgress({ start:false });
  const profiler = d3.select(".profiler").html("");
  const buf = await (await fetch("/get_profile")).arrayBuffer();
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
  const [tickSize, padding] = [10, 8];
  const deviceList = profiler.append("div").attr("id", "device-list").style("padding-top", tickSize+padding+"px");
  const canvas = profiler.append("canvas").attr("id", "timeline").node();
  // NOTE: scrolling via mouse can only zoom the graph
  canvas.addEventListener("wheel", e => (e.stopPropagation(), e.preventDefault()), { passive:false });
  const ctx = canvas.getContext("2d");
  const canvasTop = rect(canvas).top;
  // color by key (name/device)
  const colorMap = new Map();
  data = {tracks:new Map(), axes:{}};
  const heightScale = d3.scaleLinear().domain([0, tracePeak]).range([4,maxheight=100]);
  for (let i=0; i<layoutsLen; i++) {
    const nameLen = view.getUint8(offset, true); offset += 1;
    const k = textDecoder.decode(new Uint8Array(buf, offset, nameLen)); offset += nameLen;
    const div = deviceList.append("div").attr("id", k).text(k).style("padding", padding+"px");
    const { y:baseY, height:baseHeight } = rect(div.node());
    const offsetY = baseY-canvasTop+padding/2;
    const shapes = [], visible = [];
    const EventTypes = {TIMELINE:0, MEMORY:1};
    const eventType = u8(), eventsLen = u32();
    if (eventType === EventTypes.TIMELINE) {
      const levelHeight = baseHeight-padding;
      const levels = [];
      data.tracks.set(k, { shapes, visible, offsetY });
      let colorKey, ref;
      for (let j=0; j<eventsLen; j++) {
        const e = {name:strings[u32()], ref:optional(u32()), st:u32(), dur:f32(), info:strings[u32()] || null};
        // find a free level to put the event
        let depth = levels.findIndex(levelEt => e.st >= levelEt);
        const et = e.st+Math.trunc(e.dur);
        if (depth === -1) {
          depth = levels.length;
          levels.push(et);
        } else levels[depth] = et;
        if (depth === 0) colorKey = e.name.split(" ")[0];
        if (!colorMap.has(colorKey)) colorMap.set(colorKey, d3.rgb(cycleColors(colorScheme[k.split(":")[0]] ?? colorScheme.DEFAULT, colorMap.size)));
        const base = colorMap.get(colorKey), s = Math.min(Math.pow(1/0.7, depth), 240 / Math.max(base.r, base.g, base.b));
        const fillColor = d3.rgb(base.r*s, base.g*s, base.b*s).toString();
        const label = parseColors(e.name).map(({ color, st }) => ({ color, st, width:ctx.measureText(st).width }));
        if (e.ref != null) ref = {ctx:e.ref, step:0};
        else if (ref != null) {
          const start = ref.step>0 ? ref.step+1 : 0;
          const stepIdx = ctxs[ref.ctx+1].steps.findIndex((s, i) => i >= start && s.name == e.name);
          ref = stepIdx === -1 ? null : {ctx:ref.ctx, step:stepIdx};
        }
        const htmlLabel = label.map(({color, st}) => `<span style="color:${color}">${st}</span>`).join('');
        const arg = { tooltipText:htmlLabel+"\n"+formatTime(e.dur)+(e.info != null ? "\n"+e.info : ""), ...ref };
        // offset y by depth
        shapes.push({x:e.st, y:levelHeight*depth, width:e.dur, height:levelHeight, arg, label, fillColor });
      }
      div.style("height", levelHeight*levels.length+padding+"px").style("pointerEvents", "none");
    } else {
      const peak = u64();
      let x = 0, y = 0;
      const buf_shapes = new Map(), temp = new Map();
      const timestamps = [];
      for (let j=0; j<eventsLen; j++) {
        const alloc = u8(), ts = u32(), key = u32();
        if (alloc) {
          const dtype = strings[u32()], sz = u64(), nbytes = dtypeSize[dtype]*sz;
          const shape = {x:[x], y:[y], dtype, sz, nbytes, key};
          buf_shapes.set(key, shape); temp.set(key, shape);
          timestamps.push(ts);
          x += 1; y += nbytes;
        } else {
          const free = buf_shapes.get(key);
          timestamps.push(ts);
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
      for (const [_, v] of temp) {
        v.x.push(x);
        v.y.push(v.y.at(-1));
      }
      timestamps.push(dur);
      const height = heightScale(peak);
      const yscale = d3.scaleLinear().domain([0, peak]).range([height, 0]);
      for (const [num, {dtype, sz, nbytes, y, x:steps}] of buf_shapes) {
        const x = steps.map(s => timestamps[s]);
        const dur = x.at(-1)-x[0];
        const arg = {tooltipText:`${dtype} len:${formatUnit(sz)}\n${formatUnit(nbytes, "B")}\nnum:${num}\nalive for ${formatTime(dur)}`};
        shapes.push({ x, y0:y.map(yscale), y1:y.map(y0 => yscale(y0+nbytes)), arg, fillColor:cycleColors(colorScheme.BUFFER, shapes.length) });
      }
      data.tracks.set(k, { shapes, visible, offsetY, height, peak, scaleFactor:maxheight*4/height });
      div.style("height", height+padding+"px").style("cursor", "pointer").on("click", (e) => {
        const newFocus = e.currentTarget.id === focusedDevice ? null : e.currentTarget.id;
        let offset = 0;
        for (const [tid, track] of data.tracks) {
          track.offsetY += offset;
          if (tid === newFocus) offset += rescaleTrack(track, tid, track.scaleFactor);
          else if (tid === focusedDevice) offset += rescaleTrack(track, tid, 1/track.scaleFactor);
        }
        data.axes.y = newFocus != null ? { domain:[0, (t=data.tracks.get(newFocus)).peak], range:[t.offsetY+t.height, t.offsetY], fmt:"B" } : null;
        focusedDevice = newFocus;
        return resize();
      });
    }
  }
  updateProgress({ start:false });
  // draw events on a timeline
  const dpr = window.devicePixelRatio || 1;
  const ellipsisWidth = ctx.measureText("...").width;
  function render(transform) {
    zoomLevel = transform;
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    // rescale to match current zoom
    const xscale = d3.scaleLinear().domain([0, dur]).range([0, canvas.clientWidth]);
    const visibleX = xscale.range().map(zoomLevel.invertX, zoomLevel).map(xscale.invert, xscale);
    const st = visibleX[0], et = visibleX[1];
    xscale.domain(visibleX);
    // draw shapes
    for (const [_, { offsetY, shapes, visible }] of data.tracks) {
      visible.length = 0;
      for (const e of shapes) {
        // generic polygon
        if (e.width == null) {
          if (e.x[0]>et || e.x.at(-1)<st) continue;
          const x = e.x.map(xscale);
          ctx.beginPath();
          ctx.moveTo(x[0], offsetY+e.y0[0]);
          for (let i=1; i<x.length; i++) {
            ctx.lineTo(x[i], offsetY+e.y0[i]);
            visible.push({ x0:x[i-1], x1:x[i], y0:offsetY+e.y1[i-1], y1:offsetY+e.y0[i], arg:e.arg });
          }
          for (let i=x.length-1; i>=0; i--) ctx.lineTo(x[i], offsetY+e.y1[i]);
          ctx.closePath();
          ctx.fillStyle = e.fillColor; ctx.fill();
          continue;
        }
        // contiguous rect
        if (e.x>et || e.x+e.width<st) continue;
        const x = xscale(e.x);
        const y = offsetY+e.y;
        const width = xscale(e.x+e.width)-x;
        ctx.fillStyle = e.fillColor; ctx.fillRect(x, y, width, e.height);
        visible.push({ y0:y, y1:y+e.height, x0:x, x1:x+width, arg:e.arg });
        // add label
        if (e.label == null) continue;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        let labelX = x+2, labelWidth = 0;
        const labelY = y+e.height/2;
        for (const [i,l] of e.label.entries()) {
          if (labelWidth+l.width+(i===e.label.length-1 ? 0 : ellipsisWidth)+2 > width) {
            if (labelWidth !== 0) ctx.fillText("...", labelX, labelY);
            break;
          }
          ctx.fillStyle = l.color;
          ctx.fillText(l.st, labelX, labelY);
          labelWidth += l.width;
          labelX += l.width;
        }
      }
    }
    // draw axes
    drawLine(ctx, xscale.range(), [0, 0]);
    for (const tick of xscale.ticks()) {
      // tick line
      const x = xscale(tick);
      drawLine(ctx, [x, x], [0, tickSize])
      // tick label
      ctx.textBaseline = "top";
      ctx.fillText(formatTime(tick, dur), x+ctx.lineWidth+2, tickSize);
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
    for (const m of markers) {
      const x = xscale(m.ts);
      drawLine(ctx, [x, x], [0, canvas.clientHeight], { color:m.color });
      ctx.fillText(m.name, x+2, 1);
    }
  }

  function resize() {
    const profiler = document.querySelector(".profiler");
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

  canvasZoom = d3.zoom().filter(vizZoomFilter).scaleExtent([1, Infinity]).translateExtent([[0,0], [Infinity,0]]).on("zoom", e => render(e.transform));
  d3.select(canvas).call(canvasZoom);
  document.addEventListener("contextmenu", e => e.ctrlKey && e.preventDefault());

  new ResizeObserver(([e]) => e.contentRect.width > 0 && resize()).observe(profiler.node());

  function findRectAtPosition(x, y) {
    let tid = null;
    for (const k of data.tracks.keys()) {
      const r = rect(document.getElementById(k));
      if (y >= r.y && y <= r.y+r.height) { tid = k; break; }
    }
    if (tid == null) return;
    const { top, left, width, height } = rect(canvas);
    const X = ((x-left) * (canvas.width/width))/dpr;
    const Y = ((y-top) * (canvas.height/height))/dpr;
    for (const r of data.tracks.get(tid).visible) {
      if (Y>=r.y0 && Y<=r.y1 && X>=r.x0 && X<=r.x1) return r.arg;
    }
  }

  canvas.addEventListener("click", e => {
    e.preventDefault();
    const foundRect = findRectAtPosition(e.clientX, e.clientY);
    if (foundRect?.step != null) return setCtxWithHistory(foundRect.ctx, foundRect.step);
  });

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

function codeBlock(st, language, { loc, wrap }={}) {
  const code = document.createElement("code");
  code.innerHTML = hljs.highlight(st, { language }).value;
  code.className = "hljs";
  const ret = document.createElement("pre");
  if (wrap) ret.className = "wrap";
  if (loc != null) {
    const link = ret.appendChild(document.createElement("a"));
    link.href = "vscode://file/"+loc.join(":");
    link.textContent = `${loc[0].split("/").at(-1)}:${loc[1]}`+"\n\n";
  }
  ret.appendChild(code);
  return ret;
}

function appendTd(tr, value, unit=null) {
  const fmt = (typeof value === "number" && !Number.isInteger(value)) ? value.toFixed(2) : value;
  tr.appendChild(document.createElement("td")).innerText = unit == "us" ? formatTime(value) : fmt+(unit ?? "");
}

function appendRow(table, name, value, unit=null, cls="main-row") {
  const tr = table.appendChild(document.createElement("tr"));
  tr.className = cls;
  tr.appendChild(document.createElement("td")).innerText = name;
  appendTd(tr, value, unit);
  return tr;
}

function setActive(e) {
  if (e == null) return;
  e.classList.add("active");
  requestAnimationFrame(() => e.scrollIntoView({ behavior: "auto", block: "nearest" }));
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
  const { currentCtx:prevCtx, currentStep:prevStep } = state;
  Object.assign(state, ns);
  // update element styles if needed
  document.getElementById(`ctx-${state.currentCtx}`)?.classList.toggle("expanded", state.expandSteps);
  if (state.currentCtx !== prevCtx) {
    document.getElementById(`ctx-${prevCtx}`)?.classList.remove("active", "expanded");
    setActive(document.getElementById(`ctx-${state.currentCtx}`));
  }
  if (state.currentCtx !== prevCtx || state.currentStep !== prevStep) {
    document.getElementById(`step-${prevCtx}-${prevStep}`)?.classList.remove("active");
    setActive(document.getElementById(`step-${state.currentCtx}-${state.currentStep}`));
  }
  // re-render
  main();
}

// set a new context and keep the old one in browser history
function setCtxWithHistory(newCtx, step=0) {
  // NOTE: browser does a structured clone, passing a mutable object is safe.
  history.replaceState(state, "");
  history.pushState(state, "");
  setState({ expandSteps:true, currentCtx:newCtx+1, currentStep:step, currentRewrite:0 });
}

window.addEventListener("popstate", (e) => {
  if (e.state != null) setState(e.state);
});

async function main() {
  // ** left sidebar context list
  if (ctxs == null) {
    ctxs = [{ name:"Profiler", steps:[] }];
    for (const r of (await (await fetch("/ctxs")).json())) ctxs.push(r);
    const ctxList = document.querySelector(".ctx-list");
    for (const [i,{name, steps}] of ctxs.entries()) {
      const ul = ctxList.appendChild(document.createElement("ul"));
      ul.id = `ctx-${i}`;
      const p = ul.appendChild(document.createElement("p"));
      p.innerHTML = parseColors(name).map(c => `<span style="color: ${c.color}">${c.st}</span>`).join("");
      p.onclick = () => {
        setState(i === state.currentCtx ? { expandSteps:!state.expandSteps } : { expandSteps:true, currentCtx:i, currentStep:0, currentRewrite:0 });
      }
      for (const [j,u] of steps.entries()) {
        const inner = ul.appendChild(document.createElement("ul"));
        inner.id = `step-${i}-${j}`;
        inner.innerText = `${u.name ?? u.loc[0].replaceAll("\\", "/").split("/").pop()+':'+u.loc[1]}`+(u.match_count ? ` - ${u.match_count}` : '');
        inner.style.marginLeft = `${8*u.depth}px`;
        inner.onclick = (e) => {
          e.stopPropagation();
          setState({ currentStep:j, currentCtx:i, currentRewrite:0 });
        }
      }
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
  if (ctx.name === "Profiler") return renderProfiler();
  if (workerUrl == null) await initWorker();
  if (ckey in cache) {
    ret = cache[ckey];
  }
  // ** Disassembly view
  if (ckey.startsWith("/disasm")) {
    if (!(ckey in cache)) cache[ckey] = ret = await (await fetch(ckey)).json();
    displayGraph("disasm");
    const root = document.createElement("div");
    root.className = "raw-text";
    const metadata = document.querySelector(".metadata");
    metadata.innerHTML = "";
    // detailed assembly view
    if (ret.cols != null) {
      const asm = root.appendChild(document.createElement("table"));
      const thead = asm.appendChild(document.createElement("thead"));
      for (const c of ret.cols) thead.appendChild(document.createElement("th")).innerText = c.title ?? c;
      for (const r of ret.rows) {
        const tr = asm.appendChild(document.createElement("tr"));
        tr.className = "main-row code-row";
        for (const [i,value] of r.entries()) {
          // string format scalar values
          if (!Array.isArray(value)) appendTd(tr, value);
          // display arrays in a bar graph
          else {
            const segmentsTd = tr.appendChild(document.createElement("td"));
            segmentsTd.className = "pct-row";
            const usageBar = segmentsTd.appendChild(document.createElement("div"));
            for (const [k, v, width] of value) {
              const seg = usageBar.appendChild(document.createElement("div"));
              seg.style.width = width+"%";
              seg.title = `${ret.cols[i].labels[k]} ${v}`;
              seg.style.background = cycleColors(colorScheme.CATEGORICAL, parseInt(k));
            }
          }
        }
      }
      const summary = metadata.appendChild(document.createElement("table"));
      for (const s of ret.summary) {
        const tr = summary.appendChild(document.createElement("tr"));
        tr.className = "main-row";
        const td = tr.appendChild(document.createElement("td"));
        const div = td.appendChild(document.createElement("div"));
        div.className = "legend";
        div.appendChild(document.createElement("div")).style.background = cycleColors(colorScheme.CATEGORICAL, s.idx);
        div.appendChild(document.createElement("p")).textContent = s.label;
        appendTd(tr, s.value);
      }
    } else root.appendChild(codeBlock(ret.src, "x86asm"));
    return document.querySelector(".disasm").replaceChildren(root);
  }
  // ** UOp view (default)
  // if we don't have a complete cache yet we start streaming rewrites in this step
  if (!(ckey in cache) || (cache[ckey].length !== step.match_count+1 && activeSrc == null)) {
    ret = [];
    cache[ckey] = ret;
    const eventSource = new EventSource(ckey);
    evtSources.push(eventSource);
    eventSource.onmessage = (e) => {
      if (e.data === "END") return eventSource.close();
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
  renderDag(ret[currentRewrite].graph, ret[currentRewrite].changed_nodes ?? [], currentRewrite === 0);
  // ** right sidebar code blocks
  const metadata = document.querySelector(".metadata");
  const [code, lang] = ctx.fmt != null ? [ctx.fmt, "cpp"] : [ret[currentRewrite].uop, "python"];
  metadata.replaceChildren(codeBlock(step.code_line, "python", { loc:step.loc, wrap:true }), codeBlock(code, lang, { wrap:false }));
  // ** rewrite steps
  if (step.match_count >= 1) {
    const rewriteList = metadata.appendChild(document.createElement("div"));
    rewriteList.className = "rewrite-list";
    for (let s=0; s<=step.match_count; s++) {
      const ul = rewriteList.appendChild(document.createElement("ul"));
      ul.innerText = s;
      ul.id = `rewrite-${s}`;
      ul.onclick = () => setState({ currentRewrite:s });
      ul.className = s > ret.length-1 ? "disabled" : s === currentRewrite ? "active" : "";
      if (s > 0 && s === currentRewrite) {
        const { upat, diff } = ret[s];
        metadata.appendChild(codeBlock(upat[1], "python", { loc:upat[0], wrap:true }));
        const diffCode = metadata.appendChild(document.createElement("pre")).appendChild(document.createElement("code"));
        for (const line of diff) {
          const span = diffCode.appendChild(document.createElement("span"));
          span.style.color = line.startsWith("+") ? "#3aa56d" : line.startsWith("-") ? "#d14b4b" : "#f0f0f5";
          span.innerText = line;
          diffCode.appendChild(document.createElement("br"));
        }
        diffCode.className = "wrap";
      }
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

document.addEventListener("keydown", (event) => {
  const { currentCtx, currentStep, currentRewrite, expandSteps } = state;
  // up and down change the step or context from the list
  const changeStep = expandSteps && ctxs[currentCtx].steps?.length;
  if (event.key == "ArrowUp") {
    event.preventDefault();
    if (changeStep) {
      return setState({ currentRewrite:0, currentStep:Math.max(0, currentStep-1) });
    }
    return setState({ currentStep:0, currentRewrite:0, currentCtx:Math.max(0, currentCtx-1), expandSteps:false });
  }
  if (event.key == "ArrowDown") {
    event.preventDefault();
    if (changeStep) {
      const totalUOps = ctxs[currentCtx].steps.length-1;
      return setState({ currentRewrite:0, currentStep:Math.min(totalUOps, currentStep+1) });
    }
    return setState({ currentStep:0, currentRewrite:0, currentCtx:Math.min(ctxs.length-1, currentCtx+1), expandSteps:false });
  }
  // enter toggles focus on a single rewrite stage
  if (event.key == "Enter") {
    event.preventDefault()
    if (currentCtx === -1) {
      return setState({ currentCtx:0, expandSteps:true });
    }
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
});

main()
