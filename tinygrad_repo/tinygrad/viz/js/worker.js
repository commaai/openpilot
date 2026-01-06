const NODE_PADDING = 10;
const rectDims = (lw, lh) => ({ width:lw+NODE_PADDING*2, height:lh+NODE_PADDING*2, labelWidth:lw, labelHeight:lh });

const LINE_HEIGHT = 14;
const canvas = new OffscreenCanvas(0, 0);
const ctx = canvas.getContext("2d");

onmessage = (e) => {
  const { data, opts } = e.data;
  const g = new dagre.graphlib.Graph({ compound: true }).setDefaultEdgeLabel(function() { return {}; });
  (data.blocks != null ? layoutCfg : layoutUOp)(g, data, opts);
  postMessage(dagre.graphlib.json.write(g));
  self.close();
}

const layoutCfg = (g, { blocks, paths, pc_table, colors }) => {
  g.setGraph({ rankdir:"TD", font:"monospace" });
  ctx.font = `350 ${LINE_HEIGHT}px ${g.graph().font}`;
  // basic blocks render the assembly in nodes
  for (const [lead, members] of Object.entries(blocks)) {
    let [width, height, label] = [0, 0, []];
    for (const m of members) {
      const text = pc_table[m][0];
      width = Math.max(width, ctx.measureText(text).width);
      height += LINE_HEIGHT;
      const [inst, ...operands] = text.split(" ");
      label.push([{st:inst+" ", color:"#7aa2f7"}, {st:operands.join(" "), color:"#9aa5ce"}]);
    }
    g.setNode(lead, { ...rectDims(width, height), label, id:lead, color:"#1a1b26" });
  }
  // paths become edges between basic blocks
  for (const [lead, value] of Object.entries(paths)) {
    for (const [id, color] of Object.entries(value)) g.setEdge(lead, id, {label:{type:"port", text:""}, color:colors[color]});
  }
  dagre.layout(g);
}

const layoutUOp = (g, { graph, change }, opts) => {
  g.setGraph({ rankdir: "LR", font:"sans-serif" });
  ctx.font = `350 ${LINE_HEIGHT}px ${g.graph().font}`;
  if (change?.length) g.setNode("overlay", {label:"", labelWidth:0, labelHeight:0, className:"overlay"});
  for (const [k, {label, src, ref, color, tag }] of Object.entries(graph)) {
    // adjust node dims by label size (excluding escape codes) + add padding
    let [width, height] = [0, 0];
    for (line of label.replace(/\u001B\[(?:K|.*?m)/g, "").split("\n")) {
      width = Math.max(width, ctx.measureText(line).width);
      height += LINE_HEIGHT;
    }
    g.setNode(k, {...rectDims(width, height), label, ref, id:k, color, tag});
    // add edges
    const edgeCounts = {};
    for (const [_, s] of src) edgeCounts[s] = (edgeCounts[s] || 0)+1;
    for (const [port, s] of src) g.setEdge(s, k, { label: edgeCounts[s] > 1 ? {type:"tag", text:edgeCounts[s]} : {type:"port", text:port}});
    if (change?.includes(parseInt(k))) g.setParent(k, "overlay");
  }
  // optionally hide nodes from the layuot
  if (!opts.showIndexing) {
    for (const n of g.nodes()) {
      const node = g.node(n);
      if (node.label.includes("dtypes.index")) g.removeNode(n);
    }
  }
  dagre.layout(g);
  // remove overlay node if it's empty
  if (!g.node("overlay")?.width) g.removeNode("overlay");
}
