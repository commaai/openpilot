const NODE_PADDING = 10;
const rectDims = (lw, lh) => ({ width:lw+NODE_PADDING*2, height:lh+NODE_PADDING*2, labelWidth:lw, labelHeight:lh });

const canvas = new OffscreenCanvas(0, 0);
const ctx = canvas.getContext("2d");

onmessage = (e) => {
  try {
    const { data, opts } = e.data;
    const g = new dagre.graphlib.Graph({ compound: true }).setDefaultEdgeLabel(function() { return {}; });
    (data.blocks != null ? layoutCfg : layoutUOp)(g, data, opts);
    postMessage({result: dagre.graphlib.json.write(g)});
    self.close();
  } catch (err) {
    postMessage({error: err.stack || err.message || String(err)});
    self.close();
  }
}

const layoutCfg = (g, { blocks, paths, pc_tokens }) => {
  const lineHeight = 18;
  g.setGraph({ rankdir:"TD", font:"monospace", lh:lineHeight, textSpace:"1ch" });
  ctx.font = `350 ${lineHeight}px ${g.graph().font}`;
  // basic blocks render the assembly in nodes
  const tokenColors = {0:"#7aa2f7", 1:"#9aa5ce"};
  for (const [lead, members] of Object.entries(blocks)) {
    let [width, height, label] = [0, 0, []];
    for (const m of members) {
      const tokens = pc_tokens[m];
      label.push(tokens.map((t, i) => ({st:t.st, keys:t.keys, color:tokenColors[t.kind]})));
      width = Math.max(width, ctx.measureText(tokens.map((t) => t.st).join("")).width);
      height += lineHeight;
    }
    g.setNode(lead, { ...rectDims(width, height), label, id:lead, color:"#1a1b26" });
  }
  // paths become edges between basic blocks
  const pathColors = {0:"#3f7564", 1:"#7a4540", 2:"#3b5f7e"};
  for (const [lead, value] of Object.entries(paths)) {
    for (const [id, color] of Object.entries(value)) g.setEdge(lead, id, {label:{type:"port", text:""}, color:pathColors[color]});
  }
  dagre.layout(g);
}

const layoutUOp = (g, { graph, change }, opts) => {
  const lineHeight = 14;
  g.setGraph({ rankdir: "LR", font:"sans-serif", lh:lineHeight });
  ctx.font = `350 ${lineHeight}px ${g.graph().font}`;
  if (change?.length) g.setNode("overlay", {label:"", labelWidth:0, labelHeight:0, className:"overlay"});
  let callCount = 0;
  for (const [k, {label, src, ref, color, tag }] of Object.entries(graph)) {
    // adjust node dims by label size (excluding escape codes) + add padding
    let [width, height] = [0, 0];
    for (line of label.replace(/\u001B\[(?:K|.*?m)/g, "").split("\n")) {
      width = Math.max(width, ctx.measureText(line).width);
      height += lineHeight;
    }
    const callNode = label.startsWith("CALL\n") || label.startsWith("FUNCTION\n");
    if (callNode) callCount++;
    g.setNode(k, {...rectDims(width, height), label, ref, id:k, color, tag, callNode});
    // add edges
    const edgeCounts = {};
    for (const [_, s] of src) edgeCounts[s] = (edgeCounts[s] || 0)+1;
    for (const [port, s] of src) g.setEdge(s, k, { label: edgeCounts[s] > 1 ? {type:"tag", text:edgeCounts[s]} : {type:"port", text:port},
      ...(callNode && port === 0 && {color:"#a0a1b8"})});
    if (change?.includes(parseInt(k))) g.setParent(k, "overlay");
  }
  // optionally hide nodes from the layout
  if (!opts.showSink) {
    for (const n of g.nodes()) {
      const node = g.node(n);
      if ((node.label === "SINK" || node.label.startsWith("SINK\n")) && (g.successors(n) || []).length === 0) g.removeNode(n);
    }
  }
  if (!opts.showIndexing) {
    for (const n of g.nodes()) {
      const node = g.node(n);
      if (node.label.includes("dtypes.weakint")) g.removeNode(n);
    }
  }
  if (!opts.showCallSrc || opts.callSrcMask.size > 0) {
    // remove edges from src[0] to CALL nodes, track affected nodes
    const disconnected = new Set();
    for (const n of g.nodes()) {
      const node = g.node(n);
      if (node.callNode && (opts.showCallSrc ? opts.callSrcMask.has(n) : !opts.callSrcMask.has(n))) {
        node.collapsed = true;
        for (const pred of (g.predecessors(n) || [])) {
          const edge = g.edge(pred, n);
          if (edge?.label?.text === 0) {
            g.removeEdge(pred, n);
            disconnected.add(pred);
          }
        }
      }
    }
    // remove nodes that are now disconnected (no successors), only from affected subtree
    let changed = true;
    while (changed) {
      changed = false;
      for (const n of disconnected) {
        if (!g.hasNode(n)) continue;
        if ((g.successors(n) || []).length === 0) {
          for (const pred of (g.predecessors(n) || [])) disconnected.add(pred);
          g.removeNode(n);
          changed = true;
        }
      }
    }
  }
  g.graph().callCount = callCount;
  dagre.layout(g);
  // remove overlay node if it's empty
  if (!g.node("overlay")?.width) g.removeNode("overlay");
}
