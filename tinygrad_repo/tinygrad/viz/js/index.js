// **** graph renderers

const displayGraph = (cls) => {
  for (const e of document.getElementsByClassName("view")) e.style.display = e.classList.contains(cls) ? "flex" : "none";
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

const rect = (s) => (typeof s === "string" ? document.querySelector(s) : s).getBoundingClientRect();

let [workerUrl, worker, timeout] = [null, null, null];
async function renderDag(graph, additions, recenter=false) {
  // start calculating the new layout (non-blocking)
  if (worker == null) {
    const resp = await Promise.all(["/assets/dagrejs.github.io/project/dagre/latest/dagre.min.js","/js/worker.js"].map(u => fetch(u)));
    workerUrl = URL.createObjectURL(new Blob([(await Promise.all(resp.map((r) => r.text()))).join("\n")], { type: "application/javascript" }));
    worker = new Worker(workerUrl);
  } else {
    worker.terminate();
    worker = new Worker(workerUrl);
  }
  if (timeout != null) clearTimeout(timeout);
  const progressMessage = document.querySelector(".progress-message");
  timeout = setTimeout(() => {progressMessage.style.display = "block"}, 2000);
  worker.postMessage({graph, additions, ctxs});
  worker.onmessage = (e) => {
    displayGraph("graph");
    progressMessage.style.display = "none";
    clearTimeout(timeout);
    d3.select("#bars").html("");
    const g = dagre.graphlib.json.read(e.data);
    // draw nodes
    const STROKE_WIDTH = 1.4;
    const nodes = d3.select("#nodes").selectAll("g").data(g.nodes().map(id => g.node(id)), d => d).join("g")
      .attr("transform", d => `translate(${d.x},${d.y})`).classed("clickable", d => d.ref != null)
      .on("click", (_,d) => setCtxWithHistory(d.ref));
    nodes.selectAll("rect").data(d => [d]).join("rect").attr("width", d => d.width).attr("height", d => d.height).attr("fill", d => d.color)
      .attr("x", d => -d.width/2).attr("y", d => -d.height/2).attr("style", d => d.style ?? `stroke:#4a4b57; stroke-width:${STROKE_WIDTH}px;`);
    nodes.selectAll("g.label").data(d => [d]).join("g").attr("class", "label").attr("transform", d => {
      const x = (d.width-d.padding*2)/2;
      const y = (d.height-d.padding*2)/2+STROKE_WIDTH;
      return `translate(-${x}, -${y})`;
    }).selectAll("text").data(d => [d.label.split("\n")]).join("text").selectAll("tspan").data(d => d).join("tspan").text(d => d).attr("x", "0")
      .attr("dy", 14).attr("xml:space", "preserve");
    const tags = nodes.selectAll("g.tag").data(d => d.tag != null ? [d] : []).join("g").attr("class", "tag")
      .attr("transform", d => `translate(${-d.width/2+8}, ${-d.height/2+8})`);
    tags.selectAll("circle").data(d => [d]).join("circle");
    tags.selectAll("text").data(d => [d.tag]).join("text").text(d => d).attr("dy", "0.35em");
    // draw edges
    const line = d3.line().x(d => d.x).y(d => d.y).curve(d3.curveBasis);
    d3.select("#edges").selectAll("path.edgePath").data(g.edges()).join("path").attr("class", "edgePath").attr("d", (e) => {
      const edge = g.edge(e);
      const points = edge.points.slice(1, edge.points.length-1);
      points.unshift(intersectRect(g.node(e.v), points[0]));
      points.push(intersectRect(g.node(e.w), points[points.length-1]));
      return line(points);
    }).attr("marker-end", "url(#arrowhead)");
    const edgeLabels = d3.select("#edge-labels").selectAll("g").data(g.edges().filter(e => g.edge(e).label != null)).join("g").attr("transform", (e) => {
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
    }).attr("class", "tag");
    edgeLabels.selectAll("circle").data(e => [g.edge(e).label]).join("circle");
    edgeLabels.selectAll("text").data(e => [g.edge(e).label]).join("text").text(d => d).attr("dy", "0.35em");
    if (recenter) document.getElementById("zoom-to-fit-btn").click();
  };

}

// ** Memory graph (WIP)

DTYPE_SIZE = {"bool": 1, "char": 1, "uchar": 1, "short": 2, "ushort": 2, "int": 4, "uint": 4,
              "long": 8, "ulong": 8, "half": 2, "bfloat": 2, "float": 4, "double": 8}
function getBuffer(e) {
  const [_, size, dtype, num, device] = e.label.split("\n");
  return {nbytes:size*DTYPE_SIZE[dtype.split("dtypes.")[1]], dtype, device:device.split(" ")[1], num:parseInt(num.split(" ")[1])};
}

function pluralize(num, name, alt=null) {
  return num === 1 ? `${num} ${name}` : `${num} ${alt ?? name+'s'}`
}

function renderMemoryGraph(graph) {
  displayGraph("graph");
  // ** construct alloc/free traces
  // we can map reads/writes from the kernel graph
  const actions = [];
  const children = new Map(); // {buffer: [...assign]}
  for (const [k,v] of Object.entries(graph)) {
    if (!v.label.startsWith("ASSIGN")) continue;
    actions.push({ op: "write", buffer: v.src[0] });
    for (const ks of graph[v.src[1]].src) {
      const node = graph[ks];
      const s = node.label.startsWith("ASSIGN") ? node.src[0] : ks;
      if (!children.has(s)) children.set(s, []);
      children.get(s).push(v);
      if (s !== v.src[0]) actions.push({ op: "read", buffer: s });
    }
  }
  const prealloc = new Set();
  const traces = [];
  for (const a of actions) {
    // a buffer is allocated immediately before the first write
    // TODO: we don't know the buffer is preallocated if there's only an assign in the graph
    if (a.op === "write") {
      traces.push({ type: "alloc", buffer: a.buffer });
    }
    else {
      if (traces.find(t => t.buffer === a.buffer && t.type === "alloc") == null) {
        prealloc.add(a.buffer);
      }
      else if (a === actions.findLast(({ buffer }) => buffer === a.buffer)) {
        traces.push({type: "free", buffer: a.buffer });
      }
    }
  }
  // ** get coordinates and layout for each buffer
  const ret = {};
  let timestep = 0; // x
  let memUsed = 0; // y
  for (const id of prealloc) {
    const buf = getBuffer(graph[id]);
    ret[id] = { x: [timestep], y: [memUsed], buf, id };
    memUsed += buf.nbytes;
  }
  let peak = memUsed;
  const liveBufs = [...prealloc];
  for (const t of traces) {
    const buf = getBuffer(graph[t.buffer]);
    const idx = liveBufs.findLastIndex(b => t.buffer === b);
    // alloc
    if (idx === -1) {
      liveBufs.push(t.buffer);
      ret[t.buffer] = { x: [timestep], y: [memUsed], buf, id: t.buffer };
      memUsed += buf.nbytes;
      peak = Math.max(memUsed, peak);
      timestep += 1;
    } // free
    else {
      memUsed -= buf.nbytes;
      timestep += 1;
      const removed = ret[liveBufs.splice(idx, 1)[0]];
      removed.x.push(timestep);
      removed.y.push(removed.y.at(-1));
      if (idx < liveBufs.length) {
        for (let j=idx; j<liveBufs.length; j++) {
          const b = ret[liveBufs[j]];
          b.x.push(timestep, timestep);
          b.y.push(b.y.at(-1), b.y.at(-1)-buf.nbytes);
        }
      }
    }
  }
  for (const id of liveBufs) {
    const b = ret[id];
    b.x.push(timestep);
    b.y.push(b.y.at(-1));
  }
  // ** render traces
  // clear existing groups
  document.querySelector(".progress-message").style.display = "none";
  for (c of document.getElementById("render").children) c.innerHTML = "";
  const render = d3.select("#bars");
  const yscale = d3.scaleLinear().domain([0, peak]).range([576, 0]);
  const xscale = d3.scaleLinear().domain([0, timestep]).range([0, 1024]);
  const axesGroup = render.append("g").attr("id", "axes");
  const nbytes_format = (d) => d3.format(".3~s")(d)+"B";
  axesGroup.append("g").call(d3.axisLeft(yscale).tickFormat(nbytes_format));
  axesGroup.append("g").attr("transform", `translate(0, ${yscale.range()[0]})`).call(d3.axisBottom(xscale).tickFormat(() => ""));
  const polygonGroup = render.append("g").attr("id", "polygons");
  const colors = ["7aa2f7", "ff9e64", "f7768e", "2ac3de", "7dcfff", "1abc9c", "9ece6a", "e0af68", "bb9af7", "9d7cd8", "ff007c"];
  const polygons = polygonGroup.selectAll("polygon").data(Object.values(ret)).join("polygon").attr("points", (d) => {
    const xs = d.x.map(t => xscale(t));
    const y1 = d.y.map(t => yscale(t));
    const y2 = d.y.map(t => yscale(t+d.buf.nbytes));
    const p0 = xs.map((x, i) => `${x},${y1[i]}`);
    const p1 = xs.map((x, i) => `${x},${y2[i]}`).reverse();
    return `${p0.join(' ')} ${p1.join(' ')}`;
  }).attr("fill", d => `#${colors[d.buf.num % colors.length]}`).on("mouseover", (e, { id, buf, x }) => {
    d3.select(e.currentTarget).attr("stroke", "rgba(26, 27, 38, 0.8)").attr("stroke-width", 0.8);
    const metadata = document.querySelector(".metadata");
    document.getElementById("current-buf")?.remove();
    const { num, dtype, nbytes, ...rest } = buf;
    let label = `<BUFFER n${num} ${dtype} ${nbytes_format(nbytes)}>\nalive for ${pluralize(x[x.length-1]-x[0], 'timestep')}`;
    label += '\n'+Object.entries(rest).map(([k, v]) => `${k}=${v}`).join('\n');
    const buf_children = children.get(id);
    if (buf_children) {
      label += `\n${pluralize(buf_children.length, 'child', 'children')}\n`;
      label += buf_children.map((c,i) => `[${i+1}] `+graph[c.src[1]].label.split("\n")[1]).join("\n");
    }
    metadata.appendChild(Object.assign(document.createElement("pre"), { innerText: label, id: "current-buf", className: "wrap" }));
  }).on("mouseout", (e, _) => {
    d3.select(e.currentTarget).attr("stroke", null).attr("stroke-width", null);
    document.getElementById("current-buf")?.remove()
  });
  // TODO: add the kernel line here
  document.getElementById("zoom-to-fit-btn").click();
}

const ANSI_COLORS = ["#b3b3b3", "#ff6666", "#66b366", "#ffff66", "#6666ff", "#ff66ff", "#66ffff", "#ffffff"];
const parseColors = (name) => [...name.matchAll(/(?:\u001b\[(\d+)m([\s\S]*?)\u001b\[0m)|([^\u001b]+)/g)].map(([_, code, colored_st, st]) =>
  ({ st: colored_st ?? st, color: code != null ? ANSI_COLORS[(parseInt(code)-30+60)%60] : "#ffffff" }));

// ** profiler graph

function formatTime(ts, dur) {
  if (dur<=1e3) return `${ts}us`;
  if (dur<=1e6) return `${(ts*1e-3).toFixed(2)}ms`;
  return `${(ts*1e-6).toFixed(2)}s`;
}

const colors = ["#1D1F2A", "#2A2D3D", "#373B4F", "#444862", "#12131A", "#2F3244", "#3B3F54", "#4A4E65", "#181A23", "#232532", "#313548", "#404459"];

var data, canvasZoom, zoomLevel = d3.zoomIdentity;
async function renderProfiler() {
  displayGraph("profiler");
  d3.select(".metadata").html("");
  if (data != null) return;
  // fetch and process data
  const { traceEvents } = await (await fetch("/get_profile")).json();
  let st, et;
  const events = new Map();
  for (const e of traceEvents) {
    if (e.name === "process_name") events.set(e.pid, { name:e.args.name, events:[] });
    if (e.ph === "X") {
      if (st == null) [st, et] = [e.ts, e.ts+e.dur];
      else {
        st = Math.min(st, e.ts);
        et = Math.max(et, e.ts+e.dur);
      }
      events.get(e.pid).events.push(e);
    }
  }
  const kernelMap = new Map();
  for (const [i, c] of ctxs.entries()) kernelMap.set(c.name.replace(/\x1b\[\d+m(.*?)\x1b\[0m/g, "$1"), { name:c.name, i });
  // place devices on the y axis and set vertical positions
  const [tickSize, padding] = [10, 8];
  const deviceList = document.getElementById("device-list");
  deviceList.style.paddingTop = `${tickSize+padding}px`;
  const canvas = document.getElementById("timeline");
  const ctx = canvas.getContext("2d");
  const canvasTop = rect(canvas).top;
  // color by name
  const nameMap = new Map();
  data = [];
  for (const [k, v] of events) {
    if (v.events.length === 0) continue;
    const div = deviceList.appendChild(document.createElement("div"));
    div.id = `pid-${k}`;
    div.innerText = v.name;
    div.style.padding = `${padding}px`;
    const { y:baseY, height:baseHeight } = rect(`#pid-${k}`);
    // position events on the y axis, stack ones that overlap
    const levels = [];
    v.events.sort((a,b) => (a.ts-st) - (b.ts-st));
    for (const [i,e] of v.events.entries()) {
      // assign to the first free depth
      const start = e.ts-st;
      const end = start+e.dur;
      let depth = levels.findIndex(l => start >= l);
      if (depth === -1) {
        depth = levels.length;
        levels.push(end);
      } else {
        levels[depth] = end;
      }
      // offset y by depth
      const height = baseHeight-padding;
      const y = (baseY-canvasTop+padding/2)+height*depth;
      if (!nameMap.has(e.name)) {
        const labelParts = parseColors(kernelMap.get(e.name)?.name ?? e.name).map(({ color, st }) => ({ color, st, width:ctx.measureText(st).width }));
        nameMap.set(e.name, { bgColor:colors[i%colors.length], labelParts });
      }
      data.push({ x:start, dur:e.dur, name:e.name, height, y, ...nameMap.get(e.name) });
    }
    // lastly, adjust device rect by number of levels
    div.style.height = `${baseHeight*levels.length}px`;
  }
  // draw events on a timeline
  const dpr = window.devicePixelRatio || 1;
  const ellipsisWidth = ctx.measureText("...").width;
  const rectLst = [];
  function render(transform=null) {
    if (transform != null) zoomLevel = transform;
    rectLst.length = 0;
    ctx.save();
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    // time axis
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(canvas.clientWidth, 0);
    ctx.fillStyle = ctx.strokeStyle = "#f0f0f5";
    ctx.lineWidth = 1;
    ctx.stroke();
    // xticks
    const scale = d3.scaleLinear().domain([0, et-st]).range([0, canvas.clientWidth]);
    scale.domain(scale.range().map(zoomLevel.invertX, zoomLevel).map(scale.invert, scale));
    const ticks = scale.ticks();
    for (const [i, tick] of ticks.entries()) {
      ctx.beginPath();
      const x = (i/(ticks.length-1))*canvas.clientWidth;
      ctx.moveTo(x, ctx.lineWidth);
      ctx.lineTo(x, tickSize+ctx.lineWidth);
      ctx.stroke();
      ctx.fontSize = "10px";
      ctx.textBaseline = "top";
      ctx.textAlign = i === ticks.length-1 ? "right" : "left";
      const padding = i === ticks.length-1 ? -1 : 1;
      ctx.fillText(formatTime(tick, et-st), x+(ctx.lineWidth+2)*padding, tickSize);
    }
    // programs
    for (const e of data) {
      // zoom only changes x and width
      const x = scale(e.x);
      const width = scale(e.x+e.dur)-x;
      ctx.fillStyle = e.bgColor;
      ctx.fillRect(x, e.y, width, e.height);
      rectLst.push({ y0:e.y, y1:e.y+e.height, x0:x, x1:x+width, name:e.name })
      // add labels
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      let [labelX, labelWidth] = [x+2, 0];
      const labelY = e.y+e.height/2;
      for (const [i,l] of e.labelParts.entries()) {
        if (labelWidth+l.width+(i===e.labelParts.length-1 ? 0 : ellipsisWidth)+2 > width) {
          if (labelWidth !== 0) ctx.fillText("...", labelX, labelY);
          break;
        }
        ctx.fillStyle = l.color;
        ctx.fillText(l.st, labelX, labelY);
        labelWidth += l.width;
        labelX += l.width;
      }
    }
    ctx.restore();
  }

  function resize() {
    let { width, height } = rect(".profiler");
    width -= rect("#device-list").width+padding;
    canvas.width = width*dpr;
    canvas.height = height*dpr;
    canvas.style.height = `${height}px`;
    canvas.style.width = `${width}px`;
    ctx.scale(dpr, dpr);
    render();
  }

  resize();
  window.addEventListener("resize", resize);
  canvasZoom = d3.zoom().filter(e => (!e.ctrlKey || e.type === 'wheel' || e.type === 'mousedown') && !e.button)
    .scaleExtent([1, Infinity]).translateExtent([[0,0], [Infinity,0]]).on("zoom", e => render(e.transform));
  d3.select(canvas).call(canvasZoom);
  document.addEventListener("contextmenu", e => e.ctrlKey && e.preventDefault());

  canvas.addEventListener("click", e => {
    e.preventDefault();
    const { top, left, width, height } = rect(canvas);
    const clickX = ((e.clientX-left) * (canvas.width/width))/dpr;
    const clickY = ((e.clientY-top) * (canvas.height/height))/dpr;
    for (const r of rectLst) {
      if (clickY>=r.y0 && clickY<=r.y1 && clickX>=r.x0 && clickX<=r.x1) {
        return setCtxWithHistory(kernelMap.get(r.name)?.i);
      }
    }
  });
}

// ** zoom and recentering

const svgZoom = d3.zoom().on("zoom", (e) => d3.select("#render").attr("transform", e.transform));
d3.select("#graph-svg").call(svgZoom);

// zoom to fit into view
document.getElementById("zoom-to-fit-btn").addEventListener("click", () => {
  const canvas = d3.select("#timeline");
  if (rect(canvas.node()).width !== 0) {
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

function codeBlock(st, language, { loc, wrap }) {
  const code = document.createElement("code");
  code.innerHTML = hljs.highlight(st, { language }).value;
  code.className = "hljs";
  const ret = document.createElement("pre");
  if (wrap) ret.className = "wrap";
  if (loc != null) {
    const link = ret.appendChild(document.createElement("a"));
    link.href = "vscode://file"+loc.join(":");
    link.textContent = `${loc[0].split("/").at(-1)}:${loc[1]}`+"\n\n";
  }
  ret.appendChild(code);
  return ret;
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
function setCtxWithHistory(newCtx) {
  if (newCtx == null) return;
  // NOTE: browser does a structured clone, passing a mutable object is safe.
  history.replaceState(state, "");
  history.pushState(state, "");
  setState({ expandSteps:true, currentCtx:newCtx, currentStep:0, currentRewrite:0 });
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
        inner.innerText = `${u.name ?? u.loc[0].replaceAll("\\", "/").split("/").pop()+':'+u.loc[1]} - ${u.match_count}`;
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
  if (ctx.name === "Profiler") return renderProfiler();
  const step = ctx.steps[currentStep];
  const ckey = `ctx=${currentCtx-1}&idx=${currentStep}`;
  // close any pending event sources
  let activeSrc = null;
  for (const e of evtSources) {
    if (e.url.split("?")[1] !== ckey) e.close();
    else if (e.readyState === EventSource.OPEN) activeSrc = e;
  }
  if (ckey in cache) {
    ret = cache[ckey];
  }
  // if we don't have a complete cache yet we start streaming rewrites in this step
  if (!(ckey in cache) || (cache[ckey].length !== step.match_count+1 && activeSrc == null)) {
    ret = [];
    cache[ckey] = ret;
    const eventSource = new EventSource(`/ctxs?${ckey}`);
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
  if (step.name == "View Memory Graph") {
    renderMemoryGraph(ret[currentRewrite].graph);
  } else {
    renderDag(ret[currentRewrite].graph, ret[currentRewrite].changed_nodes || [], recenter=currentRewrite === 0);
  }
  // ** right sidebar code blocks
  const metadata = document.querySelector(".metadata");
  const [code, lang] = ctx.kernel_code != null ? [ctx.kernel_code, "cpp"] : [ret[currentRewrite].uop, "python"];
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

document.addEventListener("keydown", async function(event) {
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
