// **** graph renderers

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

const rect = (s) => document.querySelector(s).getBoundingClientRect();

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
  worker.postMessage({graph, additions});
  worker.onmessage = (e) => {
    progressMessage.style.display = "none";
    clearTimeout(timeout);
    d3.select("#bars").html("");
    const g = dagre.graphlib.json.read(e.data);
    // draw nodes
    const STROKE_WIDTH = 1.4;
    const nodes = d3.select("#nodes").selectAll("g").data(g.nodes().map(id => g.node(id)), d => d).join("g")
      .attr("transform", d => `translate(${d.x},${d.y})`);
    nodes.selectAll("rect").data(d => [d]).join("rect").attr("width", d => d.width).attr("height", d => d.height).attr("fill", d => d.color)
      .attr("x", d => -d.width/2).attr("y", d => -d.height/2).attr("style", d => `stroke:#4a4b57; stroke-width:${STROKE_WIDTH}px; ${d.style}`);
    nodes.selectAll("g.label").data(d => [d]).join("g").attr("class", "label").attr("transform", d => {
      const x = (d.width-d.padding*2)/2;
      const y = (d.height-d.padding*2)/2+STROKE_WIDTH;
      return `translate(-${x}, -${y})`;
    }).selectAll("text").data(d => [d.label.split("\n")]).join("text").selectAll("tspan").data(d => d).join("tspan").text(d => d).attr("x", "0")
      .attr("dy", 14).attr("xml:space", "preserve");
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
    });
    edgeLabels.selectAll("circle").data(e => [g.edge(e).label]).join("circle").attr("r", 5).attr("fill", "#FFD700").attr("stroke", "#B8860B")
      .attr("stroke-width", 0.8);
    edgeLabels.selectAll("text").data(e => [g.edge(e).label]).join("text").text(d => d).attr("text-anchor", "middle").attr("dy", "0.35em")
      .attr("font-size", "6px").attr("fill", "black");
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
  // TODO: add the toposort graph here
  document.querySelector(".progress-message").style.display = "none";
  d3.select("#nodes").html("");
  d3.select("#edges").html("");
  document.getElementById("zoom-to-fit-btn").click();
}

// ** zoom and recentering

const zoom = d3.zoom().on("zoom", (e) => d3.select("#render").attr("transform", e.transform));
d3.select("#graph-svg").call(zoom);
// zoom to fit into view
document.getElementById("zoom-to-fit-btn").addEventListener("click", () => {
  const svg = d3.select("#graph-svg");
  svg.call(zoom.transform, d3.zoomIdentity);
  const mainRect = rect(".main-container");
  const x0 = rect(".kernel-list-parent").right;
  const x1 = rect(".metadata-parent").left;
  const pad = 16;
  const R = { x: x0+pad, y: mainRect.top+pad, width: (x1>0 ? x1-x0 : mainRect.width)-2*pad, height: mainRect.height-2*pad };
  const r = rect("#render");
  if (r.width === 0) return;
  const scale = Math.min(R.width/r.width, R.height/r.height);
  const [tx, ty] = [R.x+(R.width-r.width*scale)/2, R.y+(R.height-r.height*scale)/2];
  svg.call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
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
var kernels = null;
const evtSources = [];
const state = {currentKernel:-1, currentUOp:0, currentRewrite:0, expandKernel:false};
function setState(ns) {
  Object.assign(state, ns);
  main();
}
async function main() {
  const { currentKernel, currentUOp, currentRewrite, expandKernel } = state;
  // ** left sidebar kernel list
  if (kernels == null) {
    kernels = await (await fetch("/kernels")).json();
    setState({ currentKernel:-1 });
  }
  const kernelList = document.querySelector(".kernel-list");
  kernelList.innerHTML = "";
  for (const [i,k] of kernels.entries()) {
    const ul = kernelList.appendChild(document.createElement("ul"));
    if (i === currentKernel) {
      ul.className = "active";
      requestAnimationFrame(() => ul.scrollIntoView({ behavior: "auto", block: "nearest" }));
    }
    const p = ul.appendChild(document.createElement("p"));
    p.innerHTML = k[0].replace(/\u001b\[(\d+)m(.*?)\u001b\[0m/g, (_, code, st) => {
      const colors = ['gray','red','green','yellow','blue','magenta','cyan','white'];
      return `<span style="${`color: color-mix(in srgb, ${colors[(parseInt(code)-30+60)%60]} 60%, white)`}">${st}</span>`;
    });
    p.onclick = () => {
      setState(i === currentKernel ? { expandKernel:!expandKernel } : { expandKernel:true, currentKernel:i, currentUOp:0, currentRewrite:0 });
    }
    for (const [j,u] of k[1].entries()) {
      const inner = ul.appendChild(document.createElement("ul"));
      if (i === currentKernel && j === currentUOp) {
        inner.className = "active";
        requestAnimationFrame(() => inner.scrollIntoView({ behavior: "auto", block: "nearest" }));
      }
      inner.innerText = `${u.name ?? u.loc[0].replaceAll("\\", "/").split("/").pop()+':'+u.loc[1]} - ${u.match_count}`;
      inner.style.marginLeft = `${8*u.depth}px`;
      inner.style.display = i === currentKernel && expandKernel ? "block" : "none";
      inner.onclick = (e) => {
        e.stopPropagation();
        setState({ currentUOp:j, currentKernel:i, currentRewrite:0 });
      }
    }
  }
  // ** center graph
  if (currentKernel == -1) return;
  const kernel = kernels[currentKernel][1][currentUOp];
  const cacheKey = `kernel=${currentKernel}&idx=${currentUOp}`;
  // close any pending event sources
  let activeSrc = null;
  for (const e of evtSources) {
    if (e.url.split("?")[1] !== cacheKey) e.close();
    else if (e.readyState === EventSource.OPEN) activeSrc = e;
  }
  if (cacheKey in cache) {
    ret = cache[cacheKey];
  }
  // if we don't have a complete cache yet we start streaming this kernel
  if (!(cacheKey in cache) || (cache[cacheKey].length !== kernel.match_count+1 && activeSrc == null)) {
    ret = [];
    cache[cacheKey] = ret;
    const eventSource = new EventSource(`/kernels?kernel=${currentKernel}&idx=${currentUOp}`);
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
  if (kernel.name == "View Memory Graph") {
    renderMemoryGraph(ret[currentRewrite].graph);
  } else {
    renderDag(ret[currentRewrite].graph, ret[currentRewrite].changed_nodes || [], recenter=currentRewrite === 0);
  }
  // ** right sidebar code blocks
  const metadata = document.querySelector(".metadata");
  const [code, lang] = kernel.kernel_code != null ? [kernel.kernel_code, "cpp"] : [ret[currentRewrite].uop, "python"];
  metadata.replaceChildren(codeBlock(kernel.code_line, "python", { loc:kernel.loc, wrap:true }), codeBlock(code, lang, { wrap:false }));
  // ** rewrite steps
  if (kernel.match_count >= 1) {
    const rewriteList = metadata.appendChild(document.createElement("div"));
    rewriteList.className = "rewrite-list";
    for (let s=0; s<=kernel.match_count; s++) {
      const ul = rewriteList.appendChild(document.createElement("ul"));
      ul.innerText = s;
      ul.id = `rewrite-${s}`;
      ul.onclick = () => setState({ currentRewrite:s });
      ul.className = s > ret.length-1 ? "disabled" : s === currentRewrite ? "active" : "";
      if (s > 0 && s === currentRewrite) {
        const { upat, diff } = ret[s];
        metadata.appendChild(codeBlock(upat[1], "python", { loc:upat[0], wrap:true }));
        const diffCode = metadata.appendChild(document.createElement("pre"));
        diffCode.innerHTML = `<code>`+diff.map((line) => {
          const color = line.startsWith("+") ? "#3aa56d" : line.startsWith("-") ? "#d14b4b" : "#f0f0f5";
          return `<span style="color: ${color};">${line}</span>`;
        }).join("<br>")+`</code>`;
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
appendResizer(document.querySelector(".kernel-list-parent"), { minWidth: 15, maxWidth: 50 }, left=true);
appendResizer(document.querySelector(".metadata-parent"), { minWidth: 20, maxWidth: 50 });

// **** keyboard shortcuts

document.addEventListener("keydown", async function(event) {
  const { currentKernel, currentUOp, currentRewrite, expandKernel } = state;
  // up and down change the UOp or kernel from the list
  if (event.key == "ArrowUp") {
    event.preventDefault();
    if (expandKernel) {
      return setState({ currentRewrite:0, currentUOp:Math.max(0, currentUOp-1) });
    }
    return setState({ currentUOp:0, currentRewrite:0, currentKernel:Math.max(0, currentKernel-1) });
  }
  if (event.key == "ArrowDown") {
    event.preventDefault();
    if (expandKernel) {
      const totalUOps = kernels[currentKernel][1].length-1;
      return setState({ currentRewrite:0, currentUOp:Math.min(totalUOps, currentUOp+1) });
    }
    return setState({ currentUOp:0, currentRewrite:0, currentKernel:Math.min(kernels.length-1, currentKernel+1) });
  }
  // enter toggles focus on a single rewrite stage
  if (event.key == "Enter") {
    event.preventDefault()
    if (state.currentKernel === -1) {
      return setState({ currentKernel:0, expandKernel:true });
    }
    return setState({ currentUOp:0, currentRewrite:0, expandKernel:!expandKernel });
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
