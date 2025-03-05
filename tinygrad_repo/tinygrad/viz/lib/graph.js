function intersectRect(r1, r2) {
  const dx = r2.x-r1.x;
  const dy = r2.y-r1.y;
  if (dx === 0 && dy === 0) throw new Error("Invalid node coordinates, rects must not overlap");
  const scaleX = dx !== 0 ? (r1.width/2)/Math.abs(dx) : Infinity;
  const scaleY = dy !== 0 ? (r1.height/2)/Math.abs(dy) : Infinity;
  const scale = Math.min(scaleX, scaleY);
  return {x:r1.x+dx*scale, y:r1.y+dy*scale};
}

const allWorkers = [];
window.renderGraph = function(graph, additions) {
  while (allWorkers.length) {
    const { worker, timeout } = allWorkers.pop();
    worker.terminate();
    clearTimeout(timeout);
  }

  // ** start calculating the new layout (non-blocking)
  worker = new Worker("/lib/worker.js");
  const progressMessage = document.querySelector(".progress-message");
  const timeout = setTimeout(() => {
    progressMessage.style.display = "block";
  }, 2000);
  allWorkers.push({worker, timeout});
  worker.postMessage({graph, additions});

  worker.onmessage = (e) => {
    progressMessage.style.display = "none";
    clearTimeout(timeout);
    const g = dagre.graphlib.json.read(e.data);
    // ** draw nodes
    const nodeRender = d3.select("#nodes");
    const nodes = nodeRender.selectAll("g").data(g.nodes().map(id => g.node(id)), d => d).join("g")
      .attr("transform", d => `translate(${d.x},${d.y})`);
    nodes.selectAll("rect").data(d => [d]).join("rect").attr("width", d => d.width).attr("height", d => d.height).attr("fill", d => d.color)
      .attr("x", d => -d.width/2).attr("y", d => -d.height/2).attr("style", d => d.style);
    // +labels
    nodes.selectAll("g.label").data(d => [d]).join("g").attr("class", "label").attr("transform", d => {
      const x = (d.width-d.padding*2)/2;
      const y = (d.height-d.padding*2)/2;
      return `translate(-${x}, -${y})`;
     }).selectAll("text").data(d => [d.label.split("\n")]).join("text").selectAll("tspan").data(d => d).join("tspan").text(d => d).attr("x", "1")
       .attr("dy", 14).attr("xml:space", "preserve");

    // ** draw edges
    const line = d3.line().x(d => d.x).y(d => d.y).curve(d3.curveBasis);
    const edgeRender = d3.select("#edges");
    edgeRender.selectAll("path.edgePath").data(g.edges()).join("path").attr("class", "edgePath").attr("d", (e) => {
      const edge = g.edge(e);
      const points = edge.points.slice(1, edge.points.length-1);
      points.unshift(intersectRect(g.node(e.v), points[0]));
      points.push(intersectRect(g.node(e.w), points[points.length-1]));
      return line(points);
    }).attr("marker-end", "url(#arrowhead)");
    // +arrow heads
    d3.select("#render").append("defs").append("marker").attr("id", "arrowhead").attr("viewBox", "0 -5 10 10").attr("refX", 10).attr("refY", 0)
      .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto").append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#4a4b57");
  };
}
