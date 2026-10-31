const NODE_PADDING = 10;
const rectDims = (lw, lh) => ({ width:lw+NODE_PADDING*2, height:lh+NODE_PADDING*2, labelWidth:lw, labelHeight:lh });

const canvas = new OffscreenCanvas(0, 0);
const ctx = canvas.getContext("2d");
const monoCanvas = new OffscreenCanvas(0, 0);
const monoCtx = monoCanvas.getContext("2d");
const LINE_HEIGHT = 16;
monoCtx.font = `14px/${LINE_HEIGHT}px monospace`;

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
    // show a smaller view for repeated instructions in the graph
    for (let i=0; i<members.length;) {
      const tokens = pc_tokens[members[i]], signature = JSON.stringify(tokens);
      let j = i+1;
      while (j<members.length && JSON.stringify(pc_tokens[members[j]]) === signature) j++;
      const line = tokens.map((t) => ({st:t.st, keys:t.keys, color:tokenColors[t.kind]}));
      if (j-i > 1) line.push({st:`(${j-i}x)`, keys:[], color:tokenColors[0]});
      label.push(line);
      width = Math.max(width, ctx.measureText(line.map((t) => t.st).join("")).width);
      height += lineHeight;
      i = j;
    }
    g.setNode(lead, { ...rectDims(width, height), label, labelX:0, id:lead, color:"#1a1b26", addrspace:null });
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
  if (change?.length) g.setNode("overlay", {label:"", labelWidth:0, labelHeight:0, labelX:0, className:"overlay"});
  let callCount = 0;
  for (const [k, {label, src, ref, color, tag, exclude, addrspace}] of Object.entries(graph)) {
    // adjust node dims by label size (excluding escape codes) + add padding
    let dims, source, lang;
    if (label.startsWith("SOURCE\n")) {
      source = label.slice("SOURCE\n".length); lang = "cpp";
      const lines = source.split("\n");
      let width = 0;
      for (const line of lines) width = Math.max(width, monoCtx.measureText(line).width);
      dims = rectDims(width, lines.length*LINE_HEIGHT);
    } else {
      let [width, height] = [0, 0];
      for (line of label.replace(/\u001B\[(?:K|.*?m)/g, "").split("\n")) {
        width = Math.max(width, ctx.measureText(line).width);
        height += lineHeight;
      }
      dims = rectDims(width, height);
    }
    const op = label.split("\n", 1)[0];
    const callNode = op === "CALL", programNode = op === "PROGRAM";
    const collapsePorts = callNode ? [0] : programNode ? [0, 1] : null;
    if (callNode) callCount++;
    g.setNode(k, {...dims, label, labelX:0, ref, id:k, color, callNode, collapsePorts, exclude, addrspace, source, lang,
      className:label.startsWith("REWRITE_ERROR") ? "err" : null, tag:tag?.length > 8 ? tag.substring(0, 8) : tag});
    // add edges
    const edgeCounts = {};
    for (const [_, s] of src) edgeCounts[s] = (edgeCounts[s] || 0)+1;
    for (const [port, s] of src) g.setEdge(s, k, { label: edgeCounts[s] > 1 ? {type:"tag", text:edgeCounts[s]} : {type:"port", text:port},
      ...(collapsePorts?.includes(port) && {color:"#a0a1b8"})});
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
  // optionally remove node srcs, track affected nodes
  const disconnected = new Set();
  const CALL_TAG_WIDTH = 14;
  for (const n of g.nodes()) {
    const node = g.node(n);
    for (const consumerId of (g.successors(n) || [])) {
      const consumer = g.node(consumerId);
      // add +- toggle if this consumer has collapsible sources
      const edge = g.edge(n, consumerId);
      const collapsible = consumer.collapsePorts != null ? (consumer.collapsePorts.includes(edge?.label?.text) || node.exclude) : node.exclude;
      if (!collapsible) continue;
      consumer.collapsible = true;
      // increase width of call/function/program nodes to make space for a toggle
      if (consumer.collapsePorts != null) { consumer.width = consumer.labelWidth+NODE_PADDING*2+CALL_TAG_WIDTH; consumer.labelX = CALL_TAG_WIDTH/2; }
      // make sources invisible if UI has toggled it off
      const collapsed = consumer.callNode ? opts.showCallSrc === opts.callSrcMask.has(consumerId) : !opts.expandedNodes.has(consumerId);
      if (!collapsed) continue;
      consumer.collapsed = true;
      g.removeEdge(n, consumerId);
      disconnected.add(n);
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
  g.graph().callCount = callCount;
  dagre.layout(g);
  // remove overlay node if it's empty
  if (!g.node("overlay")?.width) g.removeNode("overlay");
}
