function recenterRects(svg, zoom) {
  const svgBounds = svg.node().getBoundingClientRect();
  for (const rect of svg.node().querySelectorAll("rect")) {
    const rectBounds = rect.getBoundingClientRect();
    const outOfBounds = rectBounds.top < svgBounds.top || rectBounds.left < svgBounds.left ||
      rectBounds.bottom > svgBounds.bottom || rectBounds.right > svgBounds.right;
    // if there's at least one rect in view we don't do anything
    if (!outOfBounds) return;
  }
  svg.call(zoom.transform, d3.zoomIdentity)
}

window.renderGraph = function(graph, additions) {
  // ** initialize the graph
  const g = new dagreD3.dagre.graphlib.Graph({ compound: true });
  g.setGraph({ rankdir: "LR" }).setDefaultEdgeLabel(function() { return {}; });
  g.setNode("addition", {label: "", clusterLabelPos: "top", style: additions.length !== 0 ? "fill: rgba(26, 27, 38, 0.5);" : "display: none;"});
  for (const [k,v] of Object.entries(graph)) {
    const [label, src, color] = v;
    const node = {label, labelType:"text", style:`fill: ${color};`};
    // for PROGRAM UOp we render the node with a code block
    if (label.includes("PROGRAM")) {
      const [name, ...rest] = label.split("\n");
      const labelEl = Object.assign(document.createElement("div"));
      labelEl.appendChild(Object.assign(document.createElement("p"), {innerText: name, className: "label", style: "margin-bottom: 2px" }));
      labelEl.appendChild(highlightedCodeBlock(rest.join("\n"), "cpp", true));
      node.label = labelEl;
      node.labelType = "html";
    }
    g.setNode(k, node);
    for (const s of src) g.setEdge(s, k, {curve: d3.curveBasis});
    if (additions.includes(parseInt(k))) g.setParent(k, "addition");
  }

  // ** select svg render
  const svg = d3.select("#graph-svg");
  const inner = svg.select("g");
  const zoom = d3.zoom().scaleExtent([0.05, 2]).on("zoom", () => {
    const transform = d3.event.transform;
    inner.attr("transform", transform);
  });
  recenterRects(svg, zoom);
  svg.call(zoom);

  // ** calculate layout + render
  const render = new dagreD3.render();
  render(inner, g);
}
