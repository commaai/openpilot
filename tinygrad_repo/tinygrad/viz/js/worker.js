const NODE_PADDING = 10;
const LINE_HEIGHT = 14;
const canvas = new OffscreenCanvas(0, 0);
const ctx = canvas.getContext("2d");
ctx.font = `${LINE_HEIGHT}px sans-serif`;

onmessage = (e) => {
  const { graph, additions, ctxs } = e.data;
  const g = new dagre.graphlib.Graph({ compound: true });
  g.setGraph({ rankdir: "LR" }).setDefaultEdgeLabel(function() { return {}; });
  if (additions.length !== 0) g.setNode("addition", {label:"", style:"fill: rgba(26, 27, 38, 0.5);", padding:0});
  for (let [k, {label, src, ref, ...rest }] of Object.entries(graph)) {
    const idx = ref ? ctxs.findIndex(k => k.ref === ref) : -1;
    // replace colors in label
    if (idx != -1) label += `\ncodegen@${ctxs[idx].name.replace(/\x1b\[\d+m(.*?)\x1b\[0m/g, "$1")}`;
    // adjust node dims by label size + add padding
    let [width, height] = [0, 0];
    for (line of label.split("\n")) {
      width = Math.max(width, ctx.measureText(line).width);
      height += LINE_HEIGHT;
    }
    g.setNode(k, {width:width+NODE_PADDING*2, height:height+NODE_PADDING*2, padding:NODE_PADDING, label, ref:idx==-1 ? null : idx, ...rest});
    // add edges
    const edgeCounts = {}
    for (const s of src) edgeCounts[s] = (edgeCounts[s] || 0)+1;
    for (const s of src) g.setEdge(s, k, { label: edgeCounts[s] > 1 ? edgeCounts[s] : null });
    if (additions.includes(parseInt(k))) g.setParent(k, "addition");
  }
  dagre.layout(g);
  postMessage(dagre.graphlib.json.write(g));
  self.close();
}
