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
      .attr("fill", d => d.color).text(d => d.st).attr("xml:space", "preserve");
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

const ANSI_COLORS = ["#b3b3b3", "#ff6666", "#66b366", "#ffff66", "#6666ff", "#ff66ff", "#66ffff", "#ffffff"];
const parseColors = (name, defaultColor="#ffffff") => [...name.matchAll(/(?:\u001b\[(\d+)m([\s\S]*?)\u001b\[0m)|([^\u001b]+)/g)]
  .map(([_, code, colored_st, st]) => ({ st: colored_st ?? st, color: code != null ? ANSI_COLORS[(parseInt(code)-30+60)%60] : defaultColor }));

// ** profiler graph

function formatTime(ts, dur=ts) {
  if (dur<=1e3) return `${ts.toFixed(2)}us`;
  if (dur<=1e6) return `${(ts*1e-3).toFixed(2)}ms`;
  return `${(ts*1e-6).toFixed(2)}s`;
}
const formatUnit = (d, unit="") => d3.format(".3~s")(d)+unit;

const devColors = {"TINY":["#1B5745", "#1D2E62"],
                   "DEFAULT":["#1D1F2A", "#2A2D3D", "#373B4F", "#444862", "#12131A", "#2F3244", "#3B3F54", "#4A4E65", "#181A23", "#232532", "#313548", "#404459"],}
const bufColors = ["#3A57B7","#5066C1","#6277CD","#7488D8","#8A9BE3","#A3B4F2"];

var profileRet, focusedDevice, canvasZoom, zoomLevel = d3.zoomIdentity;
async function renderProfiler() {
  displayGraph("profiler");
  d3.select(".metadata").html("");
  const profiler = d3.select(".profiler").html("");
  const deviceList = profiler.append("div").attr("id", "device-list").node();
  const canvas = profiler.append("canvas").attr("id", "timeline").node();
  if (profileRet == null) profileRet = await (await fetch("/get_profile")).json()
  const { layout, st, et } = profileRet;
  // place devices on the y axis and set vertical positions
  const [tickSize, padding] = [10, 8];
  deviceList.style.paddingTop = `${tickSize+padding}px`;
  const ctx = canvas.getContext("2d");
  const { top:canvasTop, height:canvasHeight } = rect(canvas);
  // color by name
  const nameMap = new Map();
  const data = {shapes:[], axes:{}};
  for (const [k, { timeline, mem }] of Object.entries(layout)) {
    if (timeline.shapes.length === 0 && mem.shapes.length == 0) continue;
    const div = deviceList.appendChild(document.createElement("div"));
    div.innerText = k;
    div.style.padding = `${padding}px`;
    div.onclick = () => { // TODO: make this feature more visible
      focusedDevice = k === focusedDevice ? null : k;
      renderProfiler();
    }
    const { y:baseY, height:baseHeight } = rect(div);
    const levelHeight = baseHeight-padding;
    const offsetY = baseY-canvasTop+padding/2;
    for (const [i,e] of timeline.shapes.entries()) {
      const label = parseColors(e.name).map(({ color, st }) => ({ color, st, width:ctx.measureText(st).width }));
      const colorKey = e.cat ?? e.name;
      if (!nameMap.has(colorKey)) {
        const colors = devColors[k] ?? devColors.DEFAULT;
        nameMap.set(colorKey, { fillColor:colors[i%colors.length] });
      }
      // offset y by depth
      data.shapes.push({ x:e.st-st, dur:e.dur, name:e.name, height:levelHeight, y:offsetY+levelHeight*e.depth, ref:e.ref, label, ...nameMap.get(colorKey) });
    }
    // position shapes on the canvas and scale to fit fixed area
    const startY = offsetY+(levelHeight*timeline.maxDepth)+padding/2;
    let area = 40;
    if (k === focusedDevice) {
      // expand memory graph for the focused device
      area = canvasHeight-baseY;
      data.axes.y = { domain:[0, mem.peak], range:[startY+area, startY], fmt:"B" };
    }
    const yscale = d3.scaleLinear().domain([0, mem.peak]).range([startY+area, startY]);
    for (const [i,e] of mem.shapes.entries()) {
      const x = e.x.map((i,_) => (mem.timestamps[i] ?? et)-st);
      const y1 = e.y.map(yscale);
      const y2 = e.y.map(y => yscale(y+e.arg.nbytes));
      data.shapes.push({ x, y1, y2, arg:e.arg, color:bufColors[i%bufColors.length] });
    }
    // lastly, adjust device rect by number of levels
    div.style.height = `${Math.max(levelHeight*timeline.maxDepth, baseHeight)+area+padding}px`;
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
    // rescale to match current zoom
    const xscale = d3.scaleLinear().domain([0, et-st]).range([0, canvas.clientWidth]);
    xscale.domain(xscale.range().map(zoomLevel.invertX, zoomLevel).map(xscale.invert, xscale));
    let yscale = null;
    if (data.axes.y != null) {
      yscale = d3.scaleLinear().domain(data.axes.y.domain).range(data.axes.y.range);
    }
    // draw shapes
    for (const e of data.shapes) {
      if (Array.isArray(e.x)) {
        const x = e.x.map(xscale);
        ctx.beginPath();
        ctx.moveTo(x[0], e.y1[0]);
        for (let i=1; i<x.length; i++) ctx.lineTo(x[i], e.y1[i]);
        for (let i=x.length-1; i>=0; i--) ctx.lineTo(x[i], e.y2[i]);
        ctx.closePath();
        ctx.fillStyle = e.color;
        ctx.fill();
        const tooltipText = `${e.arg.dtype} len:${formatUnit(e.arg.sz)}\n${formatUnit(e.arg.nbytes, "B")} `;
        for (let i = 0; i < x.length - 1; i++) rectLst.push({ x0:x[i], x1:x[i+1], y0:e.y2[i], y1:e.y1[i], tooltipText });
        continue;
      }
      // zoom only changes x and width
      const x = xscale(e.x);
      const width = xscale(e.x+e.dur)-x;
      ctx.fillStyle = e.fillColor;
      ctx.fillRect(x, e.y, width, e.height);
      rectLst.push({ y0:e.y, y1:e.y+e.height, x0:x, x1:x+width, ref:e.ref, tooltipText:formatTime(e.dur) });
      // add label
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      let [labelX, labelWidth] = [x+2, 0];
      const labelY = e.y+e.height/2;
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
    // draw axes
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(canvas.clientWidth, 0);
    ctx.fillStyle = ctx.strokeStyle = "#f0f0f5";
    ctx.lineWidth = 1;
    ctx.stroke();
    const ticks = xscale.ticks();
    for (const [i, tick] of ticks.entries()) {
      ctx.beginPath();
      // tick line
      const x = xscale(tick);
      ctx.moveTo(x, 0);
      ctx.lineTo(x, tickSize);
      ctx.stroke();
      // tick label
      ctx.fontSize = "10px";
      ctx.textBaseline = "top";
      ctx.textAlign = i === ticks.length-1 ? "right" : "left";
      const padding = i === ticks.length-1 ? -1 : 1;
      ctx.fillText(formatTime(tick, et-st), x+(ctx.lineWidth+2)*padding, tickSize);
    }
    if (yscale != null) {
      ctx.beginPath();
      ctx.moveTo(0, yscale.range()[1]);
      ctx.lineTo(0, yscale.range()[0]);
      ctx.stroke();
      for (const tick of yscale.ticks()) {
        const y = yscale(tick);
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(tickSize, y);
        ctx.stroke();
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(formatUnit(tick, data.axes.y.fmt), tickSize+2, y);
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
  d3.select(canvas).call(canvasZoom).call(canvasZoom.transform, zoomLevel);
  document.addEventListener("contextmenu", e => e.ctrlKey && e.preventDefault());

  function findRectAtPosition(x, y) {
    const { top, left, width, height } = rect(canvas);
    const X = ((x-left) * (canvas.width/width))/dpr;
    const Y = ((y-top) * (canvas.height/height))/dpr;
    for (const r of rectLst) {
      if (Y>=r.y0 && Y<=r.y1 && X>=r.x0 && X<=r.x1) return r;
    }
  }

  canvas.addEventListener("click", e => {
    e.preventDefault();
    const foundRect = findRectAtPosition(e.clientX, e.clientY);
    if (foundRect?.ref != null) return setCtxWithHistory(foundRect.ref);
  });

  const tooltip = document.body.appendChild(document.createElement("div"));
  tooltip.id = "tooltip";
  canvas.addEventListener("mousemove", e => {
    const foundRect = findRectAtPosition(e.clientX, e.clientY);
    if (foundRect?.tooltipText != null) {
      tooltip.style.display = "block";
      tooltip.style.left = (e.pageX+10)+"px";
      tooltip.style.top = (e.pageY)+"px";
      tooltip.innerText = foundRect.tooltipText;
    } else tooltip.style.display = "none";
  });
  canvas.addEventListener("mouseleave", () => tooltip.style.display = "none");
}

// ** zoom and recentering

const svgZoom = d3.zoom().on("zoom", (e) => d3.select("#render").attr("transform", e.transform));
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

function codeBlock(st, language, { loc, wrap }) {
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
  setState({ expandSteps:true, currentCtx:newCtx+1, currentStep:0, currentRewrite:0 });
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
  renderDag(ret[currentRewrite].graph, ret[currentRewrite].changed_nodes || [], recenter=currentRewrite === 0);
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
