(function(f){if(typeof exports==="object"&&typeof module!=="undefined"){module.exports=f()}else if(typeof define==="function"&&define.amd){define([],f)}else{var g;if(typeof window!=="undefined"){g=window}else if(typeof global!=="undefined"){g=global}else if(typeof self!=="undefined"){g=self}else{g=this}g.dagreD3=f()}})(function(){var define,module,exports;return function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r}()({1:[function(require,module,exports){
/**
 * @license
 * Copyright (c) 2012-2013 Chris Pettitt
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
module.exports={graphlib:require("./lib/graphlib"),dagre:require("./lib/dagre"),intersect:require("./lib/intersect"),render:require("./lib/render"),util:require("./lib/util"),version:require("./lib/version")}},{"./lib/dagre":8,"./lib/graphlib":9,"./lib/intersect":10,"./lib/render":25,"./lib/util":27,"./lib/version":28}],2:[function(require,module,exports){var util=require("./util");module.exports={default:normal,normal:normal,vee:vee,undirected:undirected};function normal(parent,id,edge,type){var marker=parent.append("marker").attr("id",id).attr("viewBox","0 0 10 10").attr("refX",9).attr("refY",5).attr("markerUnits","strokeWidth").attr("markerWidth",8).attr("markerHeight",6).attr("orient","auto");var path=marker.append("path").attr("d","M 0 0 L 10 5 L 0 10 z").style("stroke-width",1).style("stroke-dasharray","1,0");util.applyStyle(path,edge[type+"Style"]);if(edge[type+"Class"]){path.attr("class",edge[type+"Class"])}}function vee(parent,id,edge,type){var marker=parent.append("marker").attr("id",id).attr("viewBox","0 0 10 10").attr("refX",9).attr("refY",5).attr("markerUnits","strokeWidth").attr("markerWidth",8).attr("markerHeight",6).attr("orient","auto");var path=marker.append("path").attr("d","M 0 0 L 10 5 L 0 10 L 4 5 z").style("stroke-width",1).style("stroke-dasharray","1,0");util.applyStyle(path,edge[type+"Style"]);if(edge[type+"Class"]){path.attr("class",edge[type+"Class"])}}function undirected(parent,id,edge,type){var marker=parent.append("marker").attr("id",id).attr("viewBox","0 0 10 10").attr("refX",9).attr("refY",5).attr("markerUnits","strokeWidth").attr("markerWidth",8).attr("markerHeight",6).attr("orient","auto");var path=marker.append("path").attr("d","M 0 5 L 10 5").style("stroke-width",1).style("stroke-dasharray","1,0");util.applyStyle(path,edge[type+"Style"]);if(edge[type+"Class"]){path.attr("class",edge[type+"Class"])}}},{"./util":27}],3:[function(require,module,exports){var util=require("./util");var d3=require("./d3");var addLabel=require("./label/add-label");module.exports=createClusters;function createClusters(selection,g){var clusters=g.nodes().filter(function(v){return util.isSubgraph(g,v)});var svgClusters=selection.selectAll("g.cluster").data(clusters,function(v){return v});svgClusters.selectAll("*").remove();svgClusters.enter().append("g").attr("class","cluster").attr("id",function(v){var node=g.node(v);return node.id}).style("opacity",0);svgClusters=selection.selectAll("g.cluster");util.applyTransition(svgClusters,g).style("opacity",1);svgClusters.each(function(v){var node=g.node(v);var thisGroup=d3.select(this);d3.select(this).append("rect");var labelGroup=thisGroup.append("g").attr("class","label");addLabel(labelGroup,node,node.clusterLabelPos)});svgClusters.selectAll("rect").each(function(c){var node=g.node(c);var domCluster=d3.select(this);util.applyStyle(domCluster,node.style)});var exitSelection;if(svgClusters.exit){exitSelection=svgClusters.exit()}else{exitSelection=svgClusters.selectAll(null);// empty selection
}util.applyTransition(exitSelection,g).style("opacity",0).remove();return svgClusters}},{"./d3":7,"./label/add-label":18,"./util":27}],4:[function(require,module,exports){"use strict";var _=require("./lodash");var addLabel=require("./label/add-label");var util=require("./util");var d3=require("./d3");module.exports=createEdgeLabels;function createEdgeLabels(selection,g){var svgEdgeLabels=selection.selectAll("g.edgeLabel").data(g.edges(),function(e){return util.edgeToId(e)}).classed("update",true);svgEdgeLabels.exit().remove();svgEdgeLabels.enter().append("g").classed("edgeLabel",true).style("opacity",0);svgEdgeLabels=selection.selectAll("g.edgeLabel");svgEdgeLabels.each(function(e){var root=d3.select(this);root.select(".label").remove();var edge=g.edge(e);var label=addLabel(root,g.edge(e),0,0).classed("label",true);var bbox=label.node().getBBox();if(edge.labelId){label.attr("id",edge.labelId)}if(!_.has(edge,"width")){edge.width=bbox.width}if(!_.has(edge,"height")){edge.height=bbox.height}});var exitSelection;if(svgEdgeLabels.exit){exitSelection=svgEdgeLabels.exit()}else{exitSelection=svgEdgeLabels.selectAll(null);// empty selection
}util.applyTransition(exitSelection,g).style("opacity",0).remove();return svgEdgeLabels}},{"./d3":7,"./label/add-label":18,"./lodash":21,"./util":27}],5:[function(require,module,exports){"use strict";var _=require("./lodash");var intersectNode=require("./intersect/intersect-node");var util=require("./util");var d3=require("./d3");module.exports=createEdgePaths;function createEdgePaths(selection,g,arrows){var previousPaths=selection.selectAll("g.edgePath").data(g.edges(),function(e){return util.edgeToId(e)}).classed("update",true);var newPaths=enter(previousPaths,g);exit(previousPaths,g);var svgPaths=previousPaths.merge!==undefined?previousPaths.merge(newPaths):previousPaths;util.applyTransition(svgPaths,g).style("opacity",1);
// Save DOM element in the path group, and set ID and class
svgPaths.each(function(e){var domEdge=d3.select(this);var edge=g.edge(e);edge.elem=this;if(edge.id){domEdge.attr("id",edge.id)}util.applyClass(domEdge,edge["class"],(domEdge.classed("update")?"update ":"")+"edgePath")});svgPaths.selectAll("path.path").each(function(e){var edge=g.edge(e);edge.arrowheadId=_.uniqueId("arrowhead");var domEdge=d3.select(this).attr("marker-end",function(){return"url("+makeFragmentRef(location.href,edge.arrowheadId)+")"}).style("fill","none");util.applyTransition(domEdge,g).attr("d",function(e){return calcPoints(g,e)});util.applyStyle(domEdge,edge.style)});svgPaths.selectAll("defs *").remove();svgPaths.selectAll("defs").each(function(e){var edge=g.edge(e);var arrowhead=arrows[edge.arrowhead];arrowhead(d3.select(this),edge.arrowheadId,edge,"arrowhead")});return svgPaths}function makeFragmentRef(url,fragmentId){var baseUrl=url.split("#")[0];return baseUrl+"#"+fragmentId}function calcPoints(g,e){var edge=g.edge(e);var tail=g.node(e.v);var head=g.node(e.w);var points=edge.points.slice(1,edge.points.length-1);points.unshift(intersectNode(tail,points[0]));points.push(intersectNode(head,points[points.length-1]));return createLine(edge,points)}function createLine(edge,points){var line=(d3.line||d3.svg.line)().x(function(d){return d.x}).y(function(d){return d.y});(line.curve||line.interpolate)(edge.curve);return line(points)}function getCoords(elem){var bbox=elem.getBBox();var matrix=elem.ownerSVGElement.getScreenCTM().inverse().multiply(elem.getScreenCTM()).translate(bbox.width/2,bbox.height/2);return{x:matrix.e,y:matrix.f}}function enter(svgPaths,g){var svgPathsEnter=svgPaths.enter().append("g").attr("class","edgePath").style("opacity",0);svgPathsEnter.append("path").attr("class","path").attr("d",function(e){var edge=g.edge(e);var sourceElem=g.node(e.v).elem;var points=_.range(edge.points.length).map(function(){return getCoords(sourceElem)});return createLine(edge,points)});svgPathsEnter.append("defs");return svgPathsEnter}function exit(svgPaths,g){var svgPathExit=svgPaths.exit();util.applyTransition(svgPathExit,g).style("opacity",0).remove()}},{"./d3":7,"./intersect/intersect-node":14,"./lodash":21,"./util":27}],6:[function(require,module,exports){"use strict";var _=require("./lodash");var addLabel=require("./label/add-label");var util=require("./util");var d3=require("./d3");module.exports=createNodes;function createNodes(selection,g,shapes){var simpleNodes=g.nodes().filter(function(v){return!util.isSubgraph(g,v)});var svgNodes=selection.selectAll("g.node").data(simpleNodes,function(v){return v}).classed("update",true);svgNodes.exit().remove();svgNodes.enter().append("g").attr("class","node").style("opacity",0);svgNodes=selection.selectAll("g.node");svgNodes.each(function(v){var node=g.node(v);var thisGroup=d3.select(this);util.applyClass(thisGroup,node["class"],(thisGroup.classed("update")?"update ":"")+"node");thisGroup.select("g.label").remove();var labelGroup=thisGroup.append("g").attr("class","label");var labelDom=addLabel(labelGroup,node);var shape=shapes[node.shape];var bbox=_.pick(labelDom.node().getBBox(),"width","height");node.elem=this;if(node.id){thisGroup.attr("id",node.id)}if(node.labelId){labelGroup.attr("id",node.labelId)}if(_.has(node,"width")){bbox.width=node.width}if(_.has(node,"height")){bbox.height=node.height}bbox.width+=node.paddingLeft+node.paddingRight;bbox.height+=node.paddingTop+node.paddingBottom;labelGroup.attr("transform","translate("+(node.paddingLeft-node.paddingRight)/2+","+(node.paddingTop-node.paddingBottom)/2+")");var root=d3.select(this);root.select(".label-container").remove();var shapeSvg=shape(root,bbox,node).classed("label-container",true);util.applyStyle(shapeSvg,node.style);var shapeBBox=shapeSvg.node().getBBox();node.width=shapeBBox.width;node.height=shapeBBox.height});var exitSelection;if(svgNodes.exit){exitSelection=svgNodes.exit()}else{exitSelection=svgNodes.selectAll(null);// empty selection
}util.applyTransition(exitSelection,g).style("opacity",0).remove();return svgNodes}},{"./d3":7,"./label/add-label":18,"./lodash":21,"./util":27}],7:[function(require,module,exports){
// Stub to get D3 either via NPM or from the global object
var d3;if(!d3){if(typeof require==="function"){try{d3=require("d3")}catch(e){
// continue regardless of error
}}}if(!d3){d3=window.d3}module.exports=d3},{d3:60}],8:[function(require,module,exports){
/* global window */
var dagre;if(typeof require==="function"){try{dagre=require("dagre")}catch(e){
// continue regardless of error
}}if(!dagre){dagre=window.dagre}module.exports=dagre},{dagre:61}],9:[function(require,module,exports){
/* global window */
var graphlib;if(typeof require==="function"){try{graphlib=require("graphlib")}catch(e){
// continue regardless of error
}}if(!graphlib){graphlib=window.graphlib}module.exports=graphlib},{graphlib:91}],10:[function(require,module,exports){module.exports={node:require("./intersect-node"),circle:require("./intersect-circle"),ellipse:require("./intersect-ellipse"),polygon:require("./intersect-polygon"),rect:require("./intersect-rect")}},{"./intersect-circle":11,"./intersect-ellipse":12,"./intersect-node":14,"./intersect-polygon":15,"./intersect-rect":16}],11:[function(require,module,exports){var intersectEllipse=require("./intersect-ellipse");module.exports=intersectCircle;function intersectCircle(node,rx,point){return intersectEllipse(node,rx,rx,point)}},{"./intersect-ellipse":12}],12:[function(require,module,exports){module.exports=intersectEllipse;function intersectEllipse(node,rx,ry,point){
// Formulae from: http://mathworld.wolfram.com/Ellipse-LineIntersection.html
var cx=node.x;var cy=node.y;var px=cx-point.x;var py=cy-point.y;var det=Math.sqrt(rx*rx*py*py+ry*ry*px*px);var dx=Math.abs(rx*ry*px/det);if(point.x<cx){dx=-dx}var dy=Math.abs(rx*ry*py/det);if(point.y<cy){dy=-dy}return{x:cx+dx,y:cy+dy}}},{}],13:[function(require,module,exports){module.exports=intersectLine;
/*
 * Returns the point at which two lines, p and q, intersect or returns
 * undefined if they do not intersect.
 */function intersectLine(p1,p2,q1,q2){
// Algorithm from J. Avro, (ed.) Graphics Gems, No 2, Morgan Kaufmann, 1994,
// p7 and p473.
var a1,a2,b1,b2,c1,c2;var r1,r2,r3,r4;var denom,offset,num;var x,y;
// Compute a1, b1, c1, where line joining points 1 and 2 is F(x,y) = a1 x +
// b1 y + c1 = 0.
a1=p2.y-p1.y;b1=p1.x-p2.x;c1=p2.x*p1.y-p1.x*p2.y;
// Compute r3 and r4.
r3=a1*q1.x+b1*q1.y+c1;r4=a1*q2.x+b1*q2.y+c1;
// Check signs of r3 and r4. If both point 3 and point 4 lie on
// same side of line 1, the line segments do not intersect.
if(r3!==0&&r4!==0&&sameSign(r3,r4)){return}
// Compute a2, b2, c2 where line joining points 3 and 4 is G(x,y) = a2 x + b2 y + c2 = 0
a2=q2.y-q1.y;b2=q1.x-q2.x;c2=q2.x*q1.y-q1.x*q2.y;
// Compute r1 and r2
r1=a2*p1.x+b2*p1.y+c2;r2=a2*p2.x+b2*p2.y+c2;
// Check signs of r1 and r2. If both point 1 and point 2 lie
// on same side of second line segment, the line segments do
// not intersect.
if(r1!==0&&r2!==0&&sameSign(r1,r2)){return}
// Line segments intersect: compute intersection point.
denom=a1*b2-a2*b1;if(denom===0){return}offset=Math.abs(denom/2);
// The denom/2 is to get rounding instead of truncating. It
// is added or subtracted to the numerator, depending upon the
// sign of the numerator.
num=b1*c2-b2*c1;x=num<0?(num-offset)/denom:(num+offset)/denom;num=a2*c1-a1*c2;y=num<0?(num-offset)/denom:(num+offset)/denom;return{x:x,y:y}}function sameSign(r1,r2){return r1*r2>0}},{}],14:[function(require,module,exports){module.exports=intersectNode;function intersectNode(node,point){return node.intersect(point)}},{}],15:[function(require,module,exports){
/* eslint "no-console": off */
var intersectLine=require("./intersect-line");module.exports=intersectPolygon;
/*
 * Returns the point ({x, y}) at which the point argument intersects with the
 * node argument assuming that it has the shape specified by polygon.
 */function intersectPolygon(node,polyPoints,point){var x1=node.x;var y1=node.y;var intersections=[];var minX=Number.POSITIVE_INFINITY;var minY=Number.POSITIVE_INFINITY;polyPoints.forEach(function(entry){minX=Math.min(minX,entry.x);minY=Math.min(minY,entry.y)});var left=x1-node.width/2-minX;var top=y1-node.height/2-minY;for(var i=0;i<polyPoints.length;i++){var p1=polyPoints[i];var p2=polyPoints[i<polyPoints.length-1?i+1:0];var intersect=intersectLine(node,point,{x:left+p1.x,y:top+p1.y},{x:left+p2.x,y:top+p2.y});if(intersect){intersections.push(intersect)}}if(!intersections.length){console.log("NO INTERSECTION FOUND, RETURN NODE CENTER",node);return node}if(intersections.length>1){
// More intersections, find the one nearest to edge end point
intersections.sort(function(p,q){var pdx=p.x-point.x;var pdy=p.y-point.y;var distp=Math.sqrt(pdx*pdx+pdy*pdy);var qdx=q.x-point.x;var qdy=q.y-point.y;var distq=Math.sqrt(qdx*qdx+qdy*qdy);return distp<distq?-1:distp===distq?0:1})}return intersections[0]}},{"./intersect-line":13}],16:[function(require,module,exports){module.exports=intersectRect;function intersectRect(node,point){var x=node.x;var y=node.y;
// Rectangle intersection algorithm from:
// http://math.stackexchange.com/questions/108113/find-edge-between-two-boxes
var dx=point.x-x;var dy=point.y-y;var w=node.width/2;var h=node.height/2;var sx,sy;if(Math.abs(dy)*w>Math.abs(dx)*h){
// Intersection is top or bottom of rect.
if(dy<0){h=-h}sx=dy===0?0:h*dx/dy;sy=h}else{
// Intersection is left or right of rect.
if(dx<0){w=-w}sx=w;sy=dx===0?0:w*dy/dx}return{x:x+sx,y:y+sy}}},{}],17:[function(require,module,exports){var util=require("../util");module.exports=addHtmlLabel;function addHtmlLabel(root,node){var fo=root.append("foreignObject").attr("width","100000");var div=fo.append("xhtml:div");div.attr("xmlns","http://www.w3.org/1999/xhtml");var label=node.label;switch(typeof label){case"function":div.insert(label);break;case"object":
// Currently we assume this is a DOM object.
div.insert(function(){return label});break;default:div.html(label)}util.applyStyle(div,node.labelStyle);div.style("display","inline-block");
// Fix for firefox
div.style("white-space","nowrap");var client=div.node().getBoundingClientRect();fo.attr("width",client.width).attr("height",client.height);return fo}},{"../util":27}],18:[function(require,module,exports){var addTextLabel=require("./add-text-label");var addHtmlLabel=require("./add-html-label");var addSVGLabel=require("./add-svg-label");module.exports=addLabel;function addLabel(root,node,location){var label=node.label;var labelSvg=root.append("g");
// Allow the label to be a string, a function that returns a DOM element, or
// a DOM element itself.
if(node.labelType==="svg"){addSVGLabel(labelSvg,node)}else if(typeof label!=="string"||node.labelType==="html"){addHtmlLabel(labelSvg,node)}else{addTextLabel(labelSvg,node)}var labelBBox=labelSvg.node().getBBox();var y;switch(location){case"top":y=-node.height/2;break;case"bottom":y=node.height/2-labelBBox.height;break;default:y=-labelBBox.height/2}labelSvg.attr("transform","translate("+-labelBBox.width/2+","+y+")");return labelSvg}},{"./add-html-label":17,"./add-svg-label":19,"./add-text-label":20}],19:[function(require,module,exports){var util=require("../util");module.exports=addSVGLabel;function addSVGLabel(root,node){var domNode=root;domNode.node().appendChild(node.label);util.applyStyle(domNode,node.labelStyle);return domNode}},{"../util":27}],20:[function(require,module,exports){var util=require("../util");module.exports=addTextLabel;
/*
 * Attaches a text label to the specified root. Handles escape sequences.
 */function addTextLabel(root,node){var domNode=root.append("text");var lines=processEscapeSequences(node.label).split("\n");for(var i=0;i<lines.length;i++){domNode.append("tspan").attr("xml:space","preserve").attr("dy","1em").attr("x","1").text(lines[i])}util.applyStyle(domNode,node.labelStyle);return domNode}function processEscapeSequences(text){var newText="";var escaped=false;var ch;for(var i=0;i<text.length;++i){ch=text[i];if(escaped){switch(ch){case"n":newText+="\n";break;default:newText+=ch}escaped=false}else if(ch==="\\"){escaped=true}else{newText+=ch}}return newText}},{"../util":27}],21:[function(require,module,exports){
/* global window */
var lodash;if(typeof require==="function"){try{lodash={defaults:require("lodash/defaults"),each:require("lodash/each"),isFunction:require("lodash/isFunction"),isPlainObject:require("lodash/isPlainObject"),pick:require("lodash/pick"),has:require("lodash/has"),range:require("lodash/range"),uniqueId:require("lodash/uniqueId")}}catch(e){
// continue regardless of error
}}if(!lodash){lodash=window._}module.exports=lodash},{"lodash/defaults":289,"lodash/each":290,"lodash/has":299,"lodash/isFunction":308,"lodash/isPlainObject":313,"lodash/pick":331,"lodash/range":333,"lodash/uniqueId":346}],22:[function(require,module,exports){"use strict";var util=require("./util");var d3=require("./d3");module.exports=positionClusters;function positionClusters(selection,g){var created=selection.filter(function(){return!d3.select(this).classed("update")});function translate(v){var node=g.node(v);return"translate("+node.x+","+node.y+")"}created.attr("transform",translate);util.applyTransition(selection,g).style("opacity",1).attr("transform",translate);util.applyTransition(created.selectAll("rect"),g).attr("width",function(v){return g.node(v).width}).attr("height",function(v){return g.node(v).height}).attr("x",function(v){var node=g.node(v);return-node.width/2}).attr("y",function(v){var node=g.node(v);return-node.height/2})}},{"./d3":7,"./util":27}],23:[function(require,module,exports){"use strict";var util=require("./util");var d3=require("./d3");var _=require("./lodash");module.exports=positionEdgeLabels;function positionEdgeLabels(selection,g){var created=selection.filter(function(){return!d3.select(this).classed("update")});function translate(e){var edge=g.edge(e);return _.has(edge,"x")?"translate("+edge.x+","+edge.y+")":""}created.attr("transform",translate);util.applyTransition(selection,g).style("opacity",1).attr("transform",translate)}},{"./d3":7,"./lodash":21,"./util":27}],24:[function(require,module,exports){"use strict";var util=require("./util");var d3=require("./d3");module.exports=positionNodes;function positionNodes(selection,g){var created=selection.filter(function(){return!d3.select(this).classed("update")});function translate(v){var node=g.node(v);return"translate("+node.x+","+node.y+")"}created.attr("transform",translate);util.applyTransition(selection,g).style("opacity",1).attr("transform",translate)}},{"./d3":7,"./util":27}],25:[function(require,module,exports){var _=require("./lodash");var d3=require("./d3");var layout=require("./dagre").layout;module.exports=render;
// This design is based on http://bost.ocks.org/mike/chart/.
function render(){var createNodes=require("./create-nodes");var createClusters=require("./create-clusters");var createEdgeLabels=require("./create-edge-labels");var createEdgePaths=require("./create-edge-paths");var positionNodes=require("./position-nodes");var positionEdgeLabels=require("./position-edge-labels");var positionClusters=require("./position-clusters");var shapes=require("./shapes");var arrows=require("./arrows");var fn=function(svg,g){preProcessGraph(g);var outputGroup=createOrSelectGroup(svg,"output");var clustersGroup=createOrSelectGroup(outputGroup,"clusters");var edgePathsGroup=createOrSelectGroup(outputGroup,"edgePaths");var edgeLabels=createEdgeLabels(createOrSelectGroup(outputGroup,"edgeLabels"),g);var nodes=createNodes(createOrSelectGroup(outputGroup,"nodes"),g,shapes);layout(g);positionNodes(nodes,g);positionEdgeLabels(edgeLabels,g);createEdgePaths(edgePathsGroup,g,arrows);var clusters=createClusters(clustersGroup,g);positionClusters(clusters,g);postProcessGraph(g)};fn.createNodes=function(value){if(!arguments.length)return createNodes;createNodes=value;return fn};fn.createClusters=function(value){if(!arguments.length)return createClusters;createClusters=value;return fn};fn.createEdgeLabels=function(value){if(!arguments.length)return createEdgeLabels;createEdgeLabels=value;return fn};fn.createEdgePaths=function(value){if(!arguments.length)return createEdgePaths;createEdgePaths=value;return fn};fn.shapes=function(value){if(!arguments.length)return shapes;shapes=value;return fn};fn.arrows=function(value){if(!arguments.length)return arrows;arrows=value;return fn};return fn}var NODE_DEFAULT_ATTRS={paddingLeft:10,paddingRight:10,paddingTop:10,paddingBottom:10,rx:0,ry:0,shape:"rect"};var EDGE_DEFAULT_ATTRS={arrowhead:"normal",curve:d3.curveLinear};function preProcessGraph(g){g.nodes().forEach(function(v){var node=g.node(v);if(!_.has(node,"label")&&!g.children(v).length){node.label=v}if(_.has(node,"paddingX")){_.defaults(node,{paddingLeft:node.paddingX,paddingRight:node.paddingX})}if(_.has(node,"paddingY")){_.defaults(node,{paddingTop:node.paddingY,paddingBottom:node.paddingY})}if(_.has(node,"padding")){_.defaults(node,{paddingLeft:node.padding,paddingRight:node.padding,paddingTop:node.padding,paddingBottom:node.padding})}_.defaults(node,NODE_DEFAULT_ATTRS);_.each(["paddingLeft","paddingRight","paddingTop","paddingBottom"],function(k){node[k]=Number(node[k])});
// Save dimensions for restore during post-processing
if(_.has(node,"width")){node._prevWidth=node.width}if(_.has(node,"height")){node._prevHeight=node.height}});g.edges().forEach(function(e){var edge=g.edge(e);if(!_.has(edge,"label")){edge.label=""}_.defaults(edge,EDGE_DEFAULT_ATTRS)})}function postProcessGraph(g){_.each(g.nodes(),function(v){var node=g.node(v);
// Restore original dimensions
if(_.has(node,"_prevWidth")){node.width=node._prevWidth}else{delete node.width}if(_.has(node,"_prevHeight")){node.height=node._prevHeight}else{delete node.height}delete node._prevWidth;delete node._prevHeight})}function createOrSelectGroup(root,name){var selection=root.select("g."+name);if(selection.empty()){selection=root.append("g").attr("class",name)}return selection}},{"./arrows":2,"./create-clusters":3,"./create-edge-labels":4,"./create-edge-paths":5,"./create-nodes":6,"./d3":7,"./dagre":8,"./lodash":21,"./position-clusters":22,"./position-edge-labels":23,"./position-nodes":24,"./shapes":26}],26:[function(require,module,exports){"use strict";var intersectRect=require("./intersect/intersect-rect");var intersectEllipse=require("./intersect/intersect-ellipse");var intersectCircle=require("./intersect/intersect-circle");var intersectPolygon=require("./intersect/intersect-polygon");module.exports={rect:rect,ellipse:ellipse,circle:circle,diamond:diamond};function rect(parent,bbox,node){var shapeSvg=parent.insert("rect",":first-child").attr("rx",node.rx).attr("ry",node.ry).attr("x",-bbox.width/2).attr("y",-bbox.height/2).attr("width",bbox.width).attr("height",bbox.height);node.intersect=function(point){return intersectRect(node,point)};return shapeSvg}function ellipse(parent,bbox,node){var rx=bbox.width/2;var ry=bbox.height/2;var shapeSvg=parent.insert("ellipse",":first-child").attr("x",-bbox.width/2).attr("y",-bbox.height/2).attr("rx",rx).attr("ry",ry);node.intersect=function(point){return intersectEllipse(node,rx,ry,point)};return shapeSvg}function circle(parent,bbox,node){var r=Math.max(bbox.width,bbox.height)/2;var shapeSvg=parent.insert("circle",":first-child").attr("x",-bbox.width/2).attr("y",-bbox.height/2).attr("r",r);node.intersect=function(point){return intersectCircle(node,r,point)};return shapeSvg}
// Circumscribe an ellipse for the bounding box with a diamond shape. I derived
// the function to calculate the diamond shape from:
// http://mathforum.org/kb/message.jspa?messageID=3750236
function diamond(parent,bbox,node){var w=bbox.width*Math.SQRT2/2;var h=bbox.height*Math.SQRT2/2;var points=[{x:0,y:-h},{x:-w,y:0},{x:0,y:h},{x:w,y:0}];var shapeSvg=parent.insert("polygon",":first-child").attr("points",points.map(function(p){return p.x+","+p.y}).join(" "));node.intersect=function(p){return intersectPolygon(node,points,p)};return shapeSvg}},{"./intersect/intersect-circle":11,"./intersect/intersect-ellipse":12,"./intersect/intersect-polygon":15,"./intersect/intersect-rect":16}],27:[function(require,module,exports){var _=require("./lodash");
// Public utility functions
module.exports={isSubgraph:isSubgraph,edgeToId:edgeToId,applyStyle:applyStyle,applyClass:applyClass,applyTransition:applyTransition};
/*
 * Returns true if the specified node in the graph is a subgraph node. A
 * subgraph node is one that contains other nodes.
 */function isSubgraph(g,v){return!!g.children(v).length}function edgeToId(e){return escapeId(e.v)+":"+escapeId(e.w)+":"+escapeId(e.name)}var ID_DELIM=/:/g;function escapeId(str){return str?String(str).replace(ID_DELIM,"\\:"):""}function applyStyle(dom,styleFn){if(styleFn){dom.attr("style",styleFn)}}function applyClass(dom,classFn,otherClasses){if(classFn){dom.attr("class",classFn).attr("class",otherClasses+" "+dom.attr("class"))}}function applyTransition(selection,g){var graph=g.graph();if(_.isPlainObject(graph)){var transition=graph.transition;if(_.isFunction(transition)){return transition(selection)}}return selection}},{"./lodash":21}],28:[function(require,module,exports){module.exports="0.6.4"},{}],29:[function(require,module,exports){
// https://d3js.org/d3-array/ v1.2.4 Copyright 2018 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):factory(global.d3=global.d3||{})})(this,function(exports){"use strict";function ascending(a,b){return a<b?-1:a>b?1:a>=b?0:NaN}function bisector(compare){if(compare.length===1)compare=ascendingComparator(compare);return{left:function(a,x,lo,hi){if(lo==null)lo=0;if(hi==null)hi=a.length;while(lo<hi){var mid=lo+hi>>>1;if(compare(a[mid],x)<0)lo=mid+1;else hi=mid}return lo},right:function(a,x,lo,hi){if(lo==null)lo=0;if(hi==null)hi=a.length;while(lo<hi){var mid=lo+hi>>>1;if(compare(a[mid],x)>0)hi=mid;else lo=mid+1}return lo}}}function ascendingComparator(f){return function(d,x){return ascending(f(d),x)}}var ascendingBisect=bisector(ascending);var bisectRight=ascendingBisect.right;var bisectLeft=ascendingBisect.left;function pairs(array,f){if(f==null)f=pair;var i=0,n=array.length-1,p=array[0],pairs=new Array(n<0?0:n);while(i<n)pairs[i]=f(p,p=array[++i]);return pairs}function pair(a,b){return[a,b]}function cross(values0,values1,reduce){var n0=values0.length,n1=values1.length,values=new Array(n0*n1),i0,i1,i,value0;if(reduce==null)reduce=pair;for(i0=i=0;i0<n0;++i0){for(value0=values0[i0],i1=0;i1<n1;++i1,++i){values[i]=reduce(value0,values1[i1])}}return values}function descending(a,b){return b<a?-1:b>a?1:b>=a?0:NaN}function number(x){return x===null?NaN:+x}function variance(values,valueof){var n=values.length,m=0,i=-1,mean=0,value,delta,sum=0;if(valueof==null){while(++i<n){if(!isNaN(value=number(values[i]))){delta=value-mean;mean+=delta/++m;sum+=delta*(value-mean)}}}else{while(++i<n){if(!isNaN(value=number(valueof(values[i],i,values)))){delta=value-mean;mean+=delta/++m;sum+=delta*(value-mean)}}}if(m>1)return sum/(m-1)}function deviation(array,f){var v=variance(array,f);return v?Math.sqrt(v):v}function extent(values,valueof){var n=values.length,i=-1,value,min,max;if(valueof==null){while(++i<n){// Find the first comparable value.
if((value=values[i])!=null&&value>=value){min=max=value;while(++i<n){// Compare the remaining values.
if((value=values[i])!=null){if(min>value)min=value;if(max<value)max=value}}}}}else{while(++i<n){// Find the first comparable value.
if((value=valueof(values[i],i,values))!=null&&value>=value){min=max=value;while(++i<n){// Compare the remaining values.
if((value=valueof(values[i],i,values))!=null){if(min>value)min=value;if(max<value)max=value}}}}}return[min,max]}var array=Array.prototype;var slice=array.slice;var map=array.map;function constant(x){return function(){return x}}function identity(x){return x}function range(start,stop,step){start=+start,stop=+stop,step=(n=arguments.length)<2?(stop=start,start=0,1):n<3?1:+step;var i=-1,n=Math.max(0,Math.ceil((stop-start)/step))|0,range=new Array(n);while(++i<n){range[i]=start+i*step}return range}var e10=Math.sqrt(50),e5=Math.sqrt(10),e2=Math.sqrt(2);function ticks(start,stop,count){var reverse,i=-1,n,ticks,step;stop=+stop,start=+start,count=+count;if(start===stop&&count>0)return[start];if(reverse=stop<start)n=start,start=stop,stop=n;if((step=tickIncrement(start,stop,count))===0||!isFinite(step))return[];if(step>0){start=Math.ceil(start/step);stop=Math.floor(stop/step);ticks=new Array(n=Math.ceil(stop-start+1));while(++i<n)ticks[i]=(start+i)*step}else{start=Math.floor(start*step);stop=Math.ceil(stop*step);ticks=new Array(n=Math.ceil(start-stop+1));while(++i<n)ticks[i]=(start-i)/step}if(reverse)ticks.reverse();return ticks}function tickIncrement(start,stop,count){var step=(stop-start)/Math.max(0,count),power=Math.floor(Math.log(step)/Math.LN10),error=step/Math.pow(10,power);return power>=0?(error>=e10?10:error>=e5?5:error>=e2?2:1)*Math.pow(10,power):-Math.pow(10,-power)/(error>=e10?10:error>=e5?5:error>=e2?2:1)}function tickStep(start,stop,count){var step0=Math.abs(stop-start)/Math.max(0,count),step1=Math.pow(10,Math.floor(Math.log(step0)/Math.LN10)),error=step0/step1;if(error>=e10)step1*=10;else if(error>=e5)step1*=5;else if(error>=e2)step1*=2;return stop<start?-step1:step1}function sturges(values){return Math.ceil(Math.log(values.length)/Math.LN2)+1}function histogram(){var value=identity,domain=extent,threshold=sturges;function histogram(data){var i,n=data.length,x,values=new Array(n);for(i=0;i<n;++i){values[i]=value(data[i],i,data)}var xz=domain(values),x0=xz[0],x1=xz[1],tz=threshold(values,x0,x1);
// Convert number of thresholds into uniform thresholds.
if(!Array.isArray(tz)){tz=tickStep(x0,x1,tz);tz=range(Math.ceil(x0/tz)*tz,x1,tz);// exclusive
}
// Remove any thresholds outside the domain.
var m=tz.length;while(tz[0]<=x0)tz.shift(),--m;while(tz[m-1]>x1)tz.pop(),--m;var bins=new Array(m+1),bin;
// Initialize bins.
for(i=0;i<=m;++i){bin=bins[i]=[];bin.x0=i>0?tz[i-1]:x0;bin.x1=i<m?tz[i]:x1}
// Assign data to bins by value, ignoring any outside the domain.
for(i=0;i<n;++i){x=values[i];if(x0<=x&&x<=x1){bins[bisectRight(tz,x,0,m)].push(data[i])}}return bins}histogram.value=function(_){return arguments.length?(value=typeof _==="function"?_:constant(_),histogram):value};histogram.domain=function(_){return arguments.length?(domain=typeof _==="function"?_:constant([_[0],_[1]]),histogram):domain};histogram.thresholds=function(_){return arguments.length?(threshold=typeof _==="function"?_:Array.isArray(_)?constant(slice.call(_)):constant(_),histogram):threshold};return histogram}function quantile(values,p,valueof){if(valueof==null)valueof=number;if(!(n=values.length))return;if((p=+p)<=0||n<2)return+valueof(values[0],0,values);if(p>=1)return+valueof(values[n-1],n-1,values);var n,i=(n-1)*p,i0=Math.floor(i),value0=+valueof(values[i0],i0,values),value1=+valueof(values[i0+1],i0+1,values);return value0+(value1-value0)*(i-i0)}function freedmanDiaconis(values,min,max){values=map.call(values,number).sort(ascending);return Math.ceil((max-min)/(2*(quantile(values,.75)-quantile(values,.25))*Math.pow(values.length,-1/3)))}function scott(values,min,max){return Math.ceil((max-min)/(3.5*deviation(values)*Math.pow(values.length,-1/3)))}function max(values,valueof){var n=values.length,i=-1,value,max;if(valueof==null){while(++i<n){// Find the first comparable value.
if((value=values[i])!=null&&value>=value){max=value;while(++i<n){// Compare the remaining values.
if((value=values[i])!=null&&value>max){max=value}}}}}else{while(++i<n){// Find the first comparable value.
if((value=valueof(values[i],i,values))!=null&&value>=value){max=value;while(++i<n){// Compare the remaining values.
if((value=valueof(values[i],i,values))!=null&&value>max){max=value}}}}}return max}function mean(values,valueof){var n=values.length,m=n,i=-1,value,sum=0;if(valueof==null){while(++i<n){if(!isNaN(value=number(values[i])))sum+=value;else--m}}else{while(++i<n){if(!isNaN(value=number(valueof(values[i],i,values))))sum+=value;else--m}}if(m)return sum/m}function median(values,valueof){var n=values.length,i=-1,value,numbers=[];if(valueof==null){while(++i<n){if(!isNaN(value=number(values[i]))){numbers.push(value)}}}else{while(++i<n){if(!isNaN(value=number(valueof(values[i],i,values)))){numbers.push(value)}}}return quantile(numbers.sort(ascending),.5)}function merge(arrays){var n=arrays.length,m,i=-1,j=0,merged,array;while(++i<n)j+=arrays[i].length;merged=new Array(j);while(--n>=0){array=arrays[n];m=array.length;while(--m>=0){merged[--j]=array[m]}}return merged}function min(values,valueof){var n=values.length,i=-1,value,min;if(valueof==null){while(++i<n){// Find the first comparable value.
if((value=values[i])!=null&&value>=value){min=value;while(++i<n){// Compare the remaining values.
if((value=values[i])!=null&&min>value){min=value}}}}}else{while(++i<n){// Find the first comparable value.
if((value=valueof(values[i],i,values))!=null&&value>=value){min=value;while(++i<n){// Compare the remaining values.
if((value=valueof(values[i],i,values))!=null&&min>value){min=value}}}}}return min}function permute(array,indexes){var i=indexes.length,permutes=new Array(i);while(i--)permutes[i]=array[indexes[i]];return permutes}function scan(values,compare){if(!(n=values.length))return;var n,i=0,j=0,xi,xj=values[j];if(compare==null)compare=ascending;while(++i<n){if(compare(xi=values[i],xj)<0||compare(xj,xj)!==0){xj=xi,j=i}}if(compare(xj,xj)===0)return j}function shuffle(array,i0,i1){var m=(i1==null?array.length:i1)-(i0=i0==null?0:+i0),t,i;while(m){i=Math.random()*m--|0;t=array[m+i0];array[m+i0]=array[i+i0];array[i+i0]=t}return array}function sum(values,valueof){var n=values.length,i=-1,value,sum=0;if(valueof==null){while(++i<n){if(value=+values[i])sum+=value;// Note: zero and null are equivalent.
}}else{while(++i<n){if(value=+valueof(values[i],i,values))sum+=value}}return sum}function transpose(matrix){if(!(n=matrix.length))return[];for(var i=-1,m=min(matrix,length),transpose=new Array(m);++i<m;){for(var j=-1,n,row=transpose[i]=new Array(n);++j<n;){row[j]=matrix[j][i]}}return transpose}function length(d){return d.length}function zip(){return transpose(arguments)}exports.bisect=bisectRight;exports.bisectRight=bisectRight;exports.bisectLeft=bisectLeft;exports.ascending=ascending;exports.bisector=bisector;exports.cross=cross;exports.descending=descending;exports.deviation=deviation;exports.extent=extent;exports.histogram=histogram;exports.thresholdFreedmanDiaconis=freedmanDiaconis;exports.thresholdScott=scott;exports.thresholdSturges=sturges;exports.max=max;exports.mean=mean;exports.median=median;exports.merge=merge;exports.min=min;exports.pairs=pairs;exports.permute=permute;exports.quantile=quantile;exports.range=range;exports.scan=scan;exports.shuffle=shuffle;exports.sum=sum;exports.ticks=ticks;exports.tickIncrement=tickIncrement;exports.tickStep=tickStep;exports.transpose=transpose;exports.variance=variance;exports.zip=zip;Object.defineProperty(exports,"__esModule",{value:true})})},{}],30:[function(require,module,exports){
// https://d3js.org/d3-axis/ v1.0.12 Copyright 2018 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):factory(global.d3=global.d3||{})})(this,function(exports){"use strict";var slice=Array.prototype.slice;function identity(x){return x}var top=1,right=2,bottom=3,left=4,epsilon=1e-6;function translateX(x){return"translate("+(x+.5)+",0)"}function translateY(y){return"translate(0,"+(y+.5)+")"}function number(scale){return function(d){return+scale(d)}}function center(scale){var offset=Math.max(0,scale.bandwidth()-1)/2;// Adjust for 0.5px offset.
if(scale.round())offset=Math.round(offset);return function(d){return+scale(d)+offset}}function entering(){return!this.__axis}function axis(orient,scale){var tickArguments=[],tickValues=null,tickFormat=null,tickSizeInner=6,tickSizeOuter=6,tickPadding=3,k=orient===top||orient===left?-1:1,x=orient===left||orient===right?"x":"y",transform=orient===top||orient===bottom?translateX:translateY;function axis(context){var values=tickValues==null?scale.ticks?scale.ticks.apply(scale,tickArguments):scale.domain():tickValues,format=tickFormat==null?scale.tickFormat?scale.tickFormat.apply(scale,tickArguments):identity:tickFormat,spacing=Math.max(tickSizeInner,0)+tickPadding,range=scale.range(),range0=+range[0]+.5,range1=+range[range.length-1]+.5,position=(scale.bandwidth?center:number)(scale.copy()),selection=context.selection?context.selection():context,path=selection.selectAll(".domain").data([null]),tick=selection.selectAll(".tick").data(values,scale).order(),tickExit=tick.exit(),tickEnter=tick.enter().append("g").attr("class","tick"),line=tick.select("line"),text=tick.select("text");path=path.merge(path.enter().insert("path",".tick").attr("class","domain").attr("stroke","currentColor"));tick=tick.merge(tickEnter);line=line.merge(tickEnter.append("line").attr("stroke","currentColor").attr(x+"2",k*tickSizeInner));text=text.merge(tickEnter.append("text").attr("fill","currentColor").attr(x,k*spacing).attr("dy",orient===top?"0em":orient===bottom?"0.71em":"0.32em"));if(context!==selection){path=path.transition(context);tick=tick.transition(context);line=line.transition(context);text=text.transition(context);tickExit=tickExit.transition(context).attr("opacity",epsilon).attr("transform",function(d){return isFinite(d=position(d))?transform(d):this.getAttribute("transform")});tickEnter.attr("opacity",epsilon).attr("transform",function(d){var p=this.parentNode.__axis;return transform(p&&isFinite(p=p(d))?p:position(d))})}tickExit.remove();path.attr("d",orient===left||orient==right?tickSizeOuter?"M"+k*tickSizeOuter+","+range0+"H0.5V"+range1+"H"+k*tickSizeOuter:"M0.5,"+range0+"V"+range1:tickSizeOuter?"M"+range0+","+k*tickSizeOuter+"V0.5H"+range1+"V"+k*tickSizeOuter:"M"+range0+",0.5H"+range1);tick.attr("opacity",1).attr("transform",function(d){return transform(position(d))});line.attr(x+"2",k*tickSizeInner);text.attr(x,k*spacing).text(format);selection.filter(entering).attr("fill","none").attr("font-size",10).attr("font-family","sans-serif").attr("text-anchor",orient===right?"start":orient===left?"end":"middle");selection.each(function(){this.__axis=position})}axis.scale=function(_){return arguments.length?(scale=_,axis):scale};axis.ticks=function(){return tickArguments=slice.call(arguments),axis};axis.tickArguments=function(_){return arguments.length?(tickArguments=_==null?[]:slice.call(_),axis):tickArguments.slice()};axis.tickValues=function(_){return arguments.length?(tickValues=_==null?null:slice.call(_),axis):tickValues&&tickValues.slice()};axis.tickFormat=function(_){return arguments.length?(tickFormat=_,axis):tickFormat};axis.tickSize=function(_){return arguments.length?(tickSizeInner=tickSizeOuter=+_,axis):tickSizeInner};axis.tickSizeInner=function(_){return arguments.length?(tickSizeInner=+_,axis):tickSizeInner};axis.tickSizeOuter=function(_){return arguments.length?(tickSizeOuter=+_,axis):tickSizeOuter};axis.tickPadding=function(_){return arguments.length?(tickPadding=+_,axis):tickPadding};return axis}function axisTop(scale){return axis(top,scale)}function axisRight(scale){return axis(right,scale)}function axisBottom(scale){return axis(bottom,scale)}function axisLeft(scale){return axis(left,scale)}exports.axisTop=axisTop;exports.axisRight=axisRight;exports.axisBottom=axisBottom;exports.axisLeft=axisLeft;Object.defineProperty(exports,"__esModule",{value:true})})},{}],31:[function(require,module,exports){
// https://d3js.org/d3-brush/ v1.1.5 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports,require("d3-dispatch"),require("d3-drag"),require("d3-interpolate"),require("d3-selection"),require("d3-transition")):typeof define==="function"&&define.amd?define(["exports","d3-dispatch","d3-drag","d3-interpolate","d3-selection","d3-transition"],factory):(global=global||self,factory(global.d3=global.d3||{},global.d3,global.d3,global.d3,global.d3,global.d3))})(this,function(exports,d3Dispatch,d3Drag,d3Interpolate,d3Selection,d3Transition){"use strict";function constant(x){return function(){return x}}function BrushEvent(target,type,selection){this.target=target;this.type=type;this.selection=selection}function nopropagation(){d3Selection.event.stopImmediatePropagation()}function noevent(){d3Selection.event.preventDefault();d3Selection.event.stopImmediatePropagation()}var MODE_DRAG={name:"drag"},MODE_SPACE={name:"space"},MODE_HANDLE={name:"handle"},MODE_CENTER={name:"center"};function number1(e){return[+e[0],+e[1]]}function number2(e){return[number1(e[0]),number1(e[1])]}function toucher(identifier){return function(target){return d3Selection.touch(target,d3Selection.event.touches,identifier)}}var X={name:"x",handles:["w","e"].map(type),input:function(x,e){return x==null?null:[[+x[0],e[0][1]],[+x[1],e[1][1]]]},output:function(xy){return xy&&[xy[0][0],xy[1][0]]}};var Y={name:"y",handles:["n","s"].map(type),input:function(y,e){return y==null?null:[[e[0][0],+y[0]],[e[1][0],+y[1]]]},output:function(xy){return xy&&[xy[0][1],xy[1][1]]}};var XY={name:"xy",handles:["n","w","e","s","nw","ne","sw","se"].map(type),input:function(xy){return xy==null?null:number2(xy)},output:function(xy){return xy}};var cursors={overlay:"crosshair",selection:"move",n:"ns-resize",e:"ew-resize",s:"ns-resize",w:"ew-resize",nw:"nwse-resize",ne:"nesw-resize",se:"nwse-resize",sw:"nesw-resize"};var flipX={e:"w",w:"e",nw:"ne",ne:"nw",se:"sw",sw:"se"};var flipY={n:"s",s:"n",nw:"sw",ne:"se",se:"ne",sw:"nw"};var signsX={overlay:+1,selection:+1,n:null,e:+1,s:null,w:-1,nw:-1,ne:+1,se:+1,sw:-1};var signsY={overlay:+1,selection:+1,n:-1,e:null,s:+1,w:null,nw:-1,ne:-1,se:+1,sw:+1};function type(t){return{type:t}}
// Ignore right-click, since that should open the context menu.
function defaultFilter(){return!d3Selection.event.ctrlKey&&!d3Selection.event.button}function defaultExtent(){var svg=this.ownerSVGElement||this;if(svg.hasAttribute("viewBox")){svg=svg.viewBox.baseVal;return[[svg.x,svg.y],[svg.x+svg.width,svg.y+svg.height]]}return[[0,0],[svg.width.baseVal.value,svg.height.baseVal.value]]}function defaultTouchable(){return navigator.maxTouchPoints||"ontouchstart"in this}
// Like d3.local, but with the name “__brush” rather than auto-generated.
function local(node){while(!node.__brush)if(!(node=node.parentNode))return;return node.__brush}function empty(extent){return extent[0][0]===extent[1][0]||extent[0][1]===extent[1][1]}function brushSelection(node){var state=node.__brush;return state?state.dim.output(state.selection):null}function brushX(){return brush$1(X)}function brushY(){return brush$1(Y)}function brush(){return brush$1(XY)}function brush$1(dim){var extent=defaultExtent,filter=defaultFilter,touchable=defaultTouchable,keys=true,listeners=d3Dispatch.dispatch("start","brush","end"),handleSize=6,touchending;function brush(group){var overlay=group.property("__brush",initialize).selectAll(".overlay").data([type("overlay")]);overlay.enter().append("rect").attr("class","overlay").attr("pointer-events","all").attr("cursor",cursors.overlay).merge(overlay).each(function(){var extent=local(this).extent;d3Selection.select(this).attr("x",extent[0][0]).attr("y",extent[0][1]).attr("width",extent[1][0]-extent[0][0]).attr("height",extent[1][1]-extent[0][1])});group.selectAll(".selection").data([type("selection")]).enter().append("rect").attr("class","selection").attr("cursor",cursors.selection).attr("fill","#777").attr("fill-opacity",.3).attr("stroke","#fff").attr("shape-rendering","crispEdges");var handle=group.selectAll(".handle").data(dim.handles,function(d){return d.type});handle.exit().remove();handle.enter().append("rect").attr("class",function(d){return"handle handle--"+d.type}).attr("cursor",function(d){return cursors[d.type]});group.each(redraw).attr("fill","none").attr("pointer-events","all").on("mousedown.brush",started).filter(touchable).on("touchstart.brush",started).on("touchmove.brush",touchmoved).on("touchend.brush touchcancel.brush",touchended).style("touch-action","none").style("-webkit-tap-highlight-color","rgba(0,0,0,0)")}brush.move=function(group,selection){if(group.selection){group.on("start.brush",function(){emitter(this,arguments).beforestart().start()}).on("interrupt.brush end.brush",function(){emitter(this,arguments).end()}).tween("brush",function(){var that=this,state=that.__brush,emit=emitter(that,arguments),selection0=state.selection,selection1=dim.input(typeof selection==="function"?selection.apply(this,arguments):selection,state.extent),i=d3Interpolate.interpolate(selection0,selection1);function tween(t){state.selection=t===1&&selection1===null?null:i(t);redraw.call(that);emit.brush()}return selection0!==null&&selection1!==null?tween:tween(1)})}else{group.each(function(){var that=this,args=arguments,state=that.__brush,selection1=dim.input(typeof selection==="function"?selection.apply(that,args):selection,state.extent),emit=emitter(that,args).beforestart();d3Transition.interrupt(that);state.selection=selection1===null?null:selection1;redraw.call(that);emit.start().brush().end()})}};brush.clear=function(group){brush.move(group,null)};function redraw(){var group=d3Selection.select(this),selection=local(this).selection;if(selection){group.selectAll(".selection").style("display",null).attr("x",selection[0][0]).attr("y",selection[0][1]).attr("width",selection[1][0]-selection[0][0]).attr("height",selection[1][1]-selection[0][1]);group.selectAll(".handle").style("display",null).attr("x",function(d){return d.type[d.type.length-1]==="e"?selection[1][0]-handleSize/2:selection[0][0]-handleSize/2}).attr("y",function(d){return d.type[0]==="s"?selection[1][1]-handleSize/2:selection[0][1]-handleSize/2}).attr("width",function(d){return d.type==="n"||d.type==="s"?selection[1][0]-selection[0][0]+handleSize:handleSize}).attr("height",function(d){return d.type==="e"||d.type==="w"?selection[1][1]-selection[0][1]+handleSize:handleSize})}else{group.selectAll(".selection,.handle").style("display","none").attr("x",null).attr("y",null).attr("width",null).attr("height",null)}}function emitter(that,args,clean){return!clean&&that.__brush.emitter||new Emitter(that,args)}function Emitter(that,args){this.that=that;this.args=args;this.state=that.__brush;this.active=0}Emitter.prototype={beforestart:function(){if(++this.active===1)this.state.emitter=this,this.starting=true;return this},start:function(){if(this.starting)this.starting=false,this.emit("start");else this.emit("brush");return this},brush:function(){this.emit("brush");return this},end:function(){if(--this.active===0)delete this.state.emitter,this.emit("end");return this},emit:function(type){d3Selection.customEvent(new BrushEvent(brush,type,dim.output(this.state.selection)),listeners.apply,listeners,[type,this.that,this.args])}};function started(){if(touchending&&!d3Selection.event.touches)return;if(!filter.apply(this,arguments))return;var that=this,type=d3Selection.event.target.__data__.type,mode=(keys&&d3Selection.event.metaKey?type="overlay":type)==="selection"?MODE_DRAG:keys&&d3Selection.event.altKey?MODE_CENTER:MODE_HANDLE,signX=dim===Y?null:signsX[type],signY=dim===X?null:signsY[type],state=local(that),extent=state.extent,selection=state.selection,W=extent[0][0],w0,w1,N=extent[0][1],n0,n1,E=extent[1][0],e0,e1,S=extent[1][1],s0,s1,dx=0,dy=0,moving,shifting=signX&&signY&&keys&&d3Selection.event.shiftKey,lockX,lockY,pointer=d3Selection.event.touches?toucher(d3Selection.event.changedTouches[0].identifier):d3Selection.mouse,point0=pointer(that),point=point0,emit=emitter(that,arguments,true).beforestart();if(type==="overlay"){if(selection)moving=true;state.selection=selection=[[w0=dim===Y?W:point0[0],n0=dim===X?N:point0[1]],[e0=dim===Y?E:w0,s0=dim===X?S:n0]]}else{w0=selection[0][0];n0=selection[0][1];e0=selection[1][0];s0=selection[1][1]}w1=w0;n1=n0;e1=e0;s1=s0;var group=d3Selection.select(that).attr("pointer-events","none");var overlay=group.selectAll(".overlay").attr("cursor",cursors[type]);if(d3Selection.event.touches){emit.moved=moved;emit.ended=ended}else{var view=d3Selection.select(d3Selection.event.view).on("mousemove.brush",moved,true).on("mouseup.brush",ended,true);if(keys)view.on("keydown.brush",keydowned,true).on("keyup.brush",keyupped,true);d3Drag.dragDisable(d3Selection.event.view)}nopropagation();d3Transition.interrupt(that);redraw.call(that);emit.start();function moved(){var point1=pointer(that);if(shifting&&!lockX&&!lockY){if(Math.abs(point1[0]-point[0])>Math.abs(point1[1]-point[1]))lockY=true;else lockX=true}point=point1;moving=true;noevent();move()}function move(){var t;dx=point[0]-point0[0];dy=point[1]-point0[1];switch(mode){case MODE_SPACE:case MODE_DRAG:{if(signX)dx=Math.max(W-w0,Math.min(E-e0,dx)),w1=w0+dx,e1=e0+dx;if(signY)dy=Math.max(N-n0,Math.min(S-s0,dy)),n1=n0+dy,s1=s0+dy;break}case MODE_HANDLE:{if(signX<0)dx=Math.max(W-w0,Math.min(E-w0,dx)),w1=w0+dx,e1=e0;else if(signX>0)dx=Math.max(W-e0,Math.min(E-e0,dx)),w1=w0,e1=e0+dx;if(signY<0)dy=Math.max(N-n0,Math.min(S-n0,dy)),n1=n0+dy,s1=s0;else if(signY>0)dy=Math.max(N-s0,Math.min(S-s0,dy)),n1=n0,s1=s0+dy;break}case MODE_CENTER:{if(signX)w1=Math.max(W,Math.min(E,w0-dx*signX)),e1=Math.max(W,Math.min(E,e0+dx*signX));if(signY)n1=Math.max(N,Math.min(S,n0-dy*signY)),s1=Math.max(N,Math.min(S,s0+dy*signY));break}}if(e1<w1){signX*=-1;t=w0,w0=e0,e0=t;t=w1,w1=e1,e1=t;if(type in flipX)overlay.attr("cursor",cursors[type=flipX[type]])}if(s1<n1){signY*=-1;t=n0,n0=s0,s0=t;t=n1,n1=s1,s1=t;if(type in flipY)overlay.attr("cursor",cursors[type=flipY[type]])}if(state.selection)selection=state.selection;// May be set by brush.move!
if(lockX)w1=selection[0][0],e1=selection[1][0];if(lockY)n1=selection[0][1],s1=selection[1][1];if(selection[0][0]!==w1||selection[0][1]!==n1||selection[1][0]!==e1||selection[1][1]!==s1){state.selection=[[w1,n1],[e1,s1]];redraw.call(that);emit.brush()}}function ended(){nopropagation();if(d3Selection.event.touches){if(d3Selection.event.touches.length)return;if(touchending)clearTimeout(touchending);touchending=setTimeout(function(){touchending=null},500);// Ghost clicks are delayed!
}else{d3Drag.dragEnable(d3Selection.event.view,moving);view.on("keydown.brush keyup.brush mousemove.brush mouseup.brush",null)}group.attr("pointer-events","all");overlay.attr("cursor",cursors.overlay);if(state.selection)selection=state.selection;// May be set by brush.move (on start)!
if(empty(selection))state.selection=null,redraw.call(that);emit.end()}function keydowned(){switch(d3Selection.event.keyCode){case 16:{// SHIFT
shifting=signX&&signY;break}case 18:{// ALT
if(mode===MODE_HANDLE){if(signX)e0=e1-dx*signX,w0=w1+dx*signX;if(signY)s0=s1-dy*signY,n0=n1+dy*signY;mode=MODE_CENTER;move()}break}case 32:{// SPACE; takes priority over ALT
if(mode===MODE_HANDLE||mode===MODE_CENTER){if(signX<0)e0=e1-dx;else if(signX>0)w0=w1-dx;if(signY<0)s0=s1-dy;else if(signY>0)n0=n1-dy;mode=MODE_SPACE;overlay.attr("cursor",cursors.selection);move()}break}default:return}noevent()}function keyupped(){switch(d3Selection.event.keyCode){case 16:{// SHIFT
if(shifting){lockX=lockY=shifting=false;move()}break}case 18:{// ALT
if(mode===MODE_CENTER){if(signX<0)e0=e1;else if(signX>0)w0=w1;if(signY<0)s0=s1;else if(signY>0)n0=n1;mode=MODE_HANDLE;move()}break}case 32:{// SPACE
if(mode===MODE_SPACE){if(d3Selection.event.altKey){if(signX)e0=e1-dx*signX,w0=w1+dx*signX;if(signY)s0=s1-dy*signY,n0=n1+dy*signY;mode=MODE_CENTER}else{if(signX<0)e0=e1;else if(signX>0)w0=w1;if(signY<0)s0=s1;else if(signY>0)n0=n1;mode=MODE_HANDLE}overlay.attr("cursor",cursors[type]);move()}break}default:return}noevent()}}function touchmoved(){emitter(this,arguments).moved()}function touchended(){emitter(this,arguments).ended()}function initialize(){var state=this.__brush||{selection:null};state.extent=number2(extent.apply(this,arguments));state.dim=dim;return state}brush.extent=function(_){return arguments.length?(extent=typeof _==="function"?_:constant(number2(_)),brush):extent};brush.filter=function(_){return arguments.length?(filter=typeof _==="function"?_:constant(!!_),brush):filter};brush.touchable=function(_){return arguments.length?(touchable=typeof _==="function"?_:constant(!!_),brush):touchable};brush.handleSize=function(_){return arguments.length?(handleSize=+_,brush):handleSize};brush.keyModifiers=function(_){return arguments.length?(keys=!!_,brush):keys};brush.on=function(){var value=listeners.on.apply(listeners,arguments);return value===listeners?brush:value};return brush}exports.brush=brush;exports.brushSelection=brushSelection;exports.brushX=brushX;exports.brushY=brushY;Object.defineProperty(exports,"__esModule",{value:true})})},{"d3-dispatch":36,"d3-drag":37,"d3-interpolate":45,"d3-selection":52,"d3-transition":57}],32:[function(require,module,exports){
// https://d3js.org/d3-chord/ v1.0.6 Copyright 2018 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports,require("d3-array"),require("d3-path")):typeof define==="function"&&define.amd?define(["exports","d3-array","d3-path"],factory):factory(global.d3=global.d3||{},global.d3,global.d3)})(this,function(exports,d3Array,d3Path){"use strict";var cos=Math.cos;var sin=Math.sin;var pi=Math.PI;var halfPi=pi/2;var tau=pi*2;var max=Math.max;function compareValue(compare){return function(a,b){return compare(a.source.value+a.target.value,b.source.value+b.target.value)}}function chord(){var padAngle=0,sortGroups=null,sortSubgroups=null,sortChords=null;function chord(matrix){var n=matrix.length,groupSums=[],groupIndex=d3Array.range(n),subgroupIndex=[],chords=[],groups=chords.groups=new Array(n),subgroups=new Array(n*n),k,x,x0,dx,i,j;
// Compute the sum.
k=0,i=-1;while(++i<n){x=0,j=-1;while(++j<n){x+=matrix[i][j]}groupSums.push(x);subgroupIndex.push(d3Array.range(n));k+=x}
// Sort groups…
if(sortGroups)groupIndex.sort(function(a,b){return sortGroups(groupSums[a],groupSums[b])});
// Sort subgroups…
if(sortSubgroups)subgroupIndex.forEach(function(d,i){d.sort(function(a,b){return sortSubgroups(matrix[i][a],matrix[i][b])})});
// Convert the sum to scaling factor for [0, 2pi].
// TODO Allow start and end angle to be specified?
// TODO Allow padding to be specified as percentage?
k=max(0,tau-padAngle*n)/k;dx=k?padAngle:tau/n;
// Compute the start and end angle for each group and subgroup.
// Note: Opera has a bug reordering object literal properties!
x=0,i=-1;while(++i<n){x0=x,j=-1;while(++j<n){var di=groupIndex[i],dj=subgroupIndex[di][j],v=matrix[di][dj],a0=x,a1=x+=v*k;subgroups[dj*n+di]={index:di,subindex:dj,startAngle:a0,endAngle:a1,value:v}}groups[di]={index:di,startAngle:x0,endAngle:x,value:groupSums[di]};x+=dx}
// Generate chords for each (non-empty) subgroup-subgroup link.
i=-1;while(++i<n){j=i-1;while(++j<n){var source=subgroups[j*n+i],target=subgroups[i*n+j];if(source.value||target.value){chords.push(source.value<target.value?{source:target,target:source}:{source:source,target:target})}}}return sortChords?chords.sort(sortChords):chords}chord.padAngle=function(_){return arguments.length?(padAngle=max(0,_),chord):padAngle};chord.sortGroups=function(_){return arguments.length?(sortGroups=_,chord):sortGroups};chord.sortSubgroups=function(_){return arguments.length?(sortSubgroups=_,chord):sortSubgroups};chord.sortChords=function(_){return arguments.length?(_==null?sortChords=null:(sortChords=compareValue(_))._=_,chord):sortChords&&sortChords._};return chord}var slice=Array.prototype.slice;function constant(x){return function(){return x}}function defaultSource(d){return d.source}function defaultTarget(d){return d.target}function defaultRadius(d){return d.radius}function defaultStartAngle(d){return d.startAngle}function defaultEndAngle(d){return d.endAngle}function ribbon(){var source=defaultSource,target=defaultTarget,radius=defaultRadius,startAngle=defaultStartAngle,endAngle=defaultEndAngle,context=null;function ribbon(){var buffer,argv=slice.call(arguments),s=source.apply(this,argv),t=target.apply(this,argv),sr=+radius.apply(this,(argv[0]=s,argv)),sa0=startAngle.apply(this,argv)-halfPi,sa1=endAngle.apply(this,argv)-halfPi,sx0=sr*cos(sa0),sy0=sr*sin(sa0),tr=+radius.apply(this,(argv[0]=t,argv)),ta0=startAngle.apply(this,argv)-halfPi,ta1=endAngle.apply(this,argv)-halfPi;if(!context)context=buffer=d3Path.path();context.moveTo(sx0,sy0);context.arc(0,0,sr,sa0,sa1);if(sa0!==ta0||sa1!==ta1){// TODO sr !== tr?
context.quadraticCurveTo(0,0,tr*cos(ta0),tr*sin(ta0));context.arc(0,0,tr,ta0,ta1)}context.quadraticCurveTo(0,0,sx0,sy0);context.closePath();if(buffer)return context=null,buffer+""||null}ribbon.radius=function(_){return arguments.length?(radius=typeof _==="function"?_:constant(+_),ribbon):radius};ribbon.startAngle=function(_){return arguments.length?(startAngle=typeof _==="function"?_:constant(+_),ribbon):startAngle};ribbon.endAngle=function(_){return arguments.length?(endAngle=typeof _==="function"?_:constant(+_),ribbon):endAngle};ribbon.source=function(_){return arguments.length?(source=_,ribbon):source};ribbon.target=function(_){return arguments.length?(target=_,ribbon):target};ribbon.context=function(_){return arguments.length?(context=_==null?null:_,ribbon):context};return ribbon}exports.chord=chord;exports.ribbon=ribbon;Object.defineProperty(exports,"__esModule",{value:true})})},{"d3-array":29,"d3-path":46}],33:[function(require,module,exports){
// https://d3js.org/d3-collection/ v1.0.7 Copyright 2018 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):factory(global.d3=global.d3||{})})(this,function(exports){"use strict";var prefix="$";function Map(){}Map.prototype=map.prototype={constructor:Map,has:function(key){return prefix+key in this},get:function(key){return this[prefix+key]},set:function(key,value){this[prefix+key]=value;return this},remove:function(key){var property=prefix+key;return property in this&&delete this[property]},clear:function(){for(var property in this)if(property[0]===prefix)delete this[property]},keys:function(){var keys=[];for(var property in this)if(property[0]===prefix)keys.push(property.slice(1));return keys},values:function(){var values=[];for(var property in this)if(property[0]===prefix)values.push(this[property]);return values},entries:function(){var entries=[];for(var property in this)if(property[0]===prefix)entries.push({key:property.slice(1),value:this[property]});return entries},size:function(){var size=0;for(var property in this)if(property[0]===prefix)++size;return size},empty:function(){for(var property in this)if(property[0]===prefix)return false;return true},each:function(f){for(var property in this)if(property[0]===prefix)f(this[property],property.slice(1),this)}};function map(object,f){var map=new Map;
// Copy constructor.
if(object instanceof Map)object.each(function(value,key){map.set(key,value)});
// Index array by numeric index or specified key function.
else if(Array.isArray(object)){var i=-1,n=object.length,o;if(f==null)while(++i<n)map.set(i,object[i]);else while(++i<n)map.set(f(o=object[i],i,object),o)}
// Convert object to map.
else if(object)for(var key in object)map.set(key,object[key]);return map}function nest(){var keys=[],sortKeys=[],sortValues,rollup,nest;function apply(array,depth,createResult,setResult){if(depth>=keys.length){if(sortValues!=null)array.sort(sortValues);return rollup!=null?rollup(array):array}var i=-1,n=array.length,key=keys[depth++],keyValue,value,valuesByKey=map(),values,result=createResult();while(++i<n){if(values=valuesByKey.get(keyValue=key(value=array[i])+"")){values.push(value)}else{valuesByKey.set(keyValue,[value])}}valuesByKey.each(function(values,key){setResult(result,key,apply(values,depth,createResult,setResult))});return result}function entries(map$$1,depth){if(++depth>keys.length)return map$$1;var array,sortKey=sortKeys[depth-1];if(rollup!=null&&depth>=keys.length)array=map$$1.entries();else array=[],map$$1.each(function(v,k){array.push({key:k,values:entries(v,depth)})});return sortKey!=null?array.sort(function(a,b){return sortKey(a.key,b.key)}):array}return nest={object:function(array){return apply(array,0,createObject,setObject)},map:function(array){return apply(array,0,createMap,setMap)},entries:function(array){return entries(apply(array,0,createMap,setMap),0)},key:function(d){keys.push(d);return nest},sortKeys:function(order){sortKeys[keys.length-1]=order;return nest},sortValues:function(order){sortValues=order;return nest},rollup:function(f){rollup=f;return nest}}}function createObject(){return{}}function setObject(object,key,value){object[key]=value}function createMap(){return map()}function setMap(map$$1,key,value){map$$1.set(key,value)}function Set(){}var proto=map.prototype;Set.prototype=set.prototype={constructor:Set,has:proto.has,add:function(value){value+="";this[prefix+value]=value;return this},remove:proto.remove,clear:proto.clear,values:proto.keys,size:proto.size,empty:proto.empty,each:proto.each};function set(object,f){var set=new Set;
// Copy constructor.
if(object instanceof Set)object.each(function(value){set.add(value)});
// Otherwise, assume it’s an array.
else if(object){var i=-1,n=object.length;if(f==null)while(++i<n)set.add(object[i]);else while(++i<n)set.add(f(object[i],i,object))}return set}function keys(map){var keys=[];for(var key in map)keys.push(key);return keys}function values(map){var values=[];for(var key in map)values.push(map[key]);return values}function entries(map){var entries=[];for(var key in map)entries.push({key:key,value:map[key]});return entries}exports.nest=nest;exports.set=set;exports.map=map;exports.keys=keys;exports.values=values;exports.entries=entries;Object.defineProperty(exports,"__esModule",{value:true})})},{}],34:[function(require,module,exports){
// https://d3js.org/d3-color/ v1.4.0 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):(global=global||self,factory(global.d3=global.d3||{}))})(this,function(exports){"use strict";function define(constructor,factory,prototype){constructor.prototype=factory.prototype=prototype;prototype.constructor=constructor}function extend(parent,definition){var prototype=Object.create(parent.prototype);for(var key in definition)prototype[key]=definition[key];return prototype}function Color(){}var darker=.7;var brighter=1/darker;var reI="\\s*([+-]?\\d+)\\s*",reN="\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)\\s*",reP="\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*",reHex=/^#([0-9a-f]{3,8})$/,reRgbInteger=new RegExp("^rgb\\("+[reI,reI,reI]+"\\)$"),reRgbPercent=new RegExp("^rgb\\("+[reP,reP,reP]+"\\)$"),reRgbaInteger=new RegExp("^rgba\\("+[reI,reI,reI,reN]+"\\)$"),reRgbaPercent=new RegExp("^rgba\\("+[reP,reP,reP,reN]+"\\)$"),reHslPercent=new RegExp("^hsl\\("+[reN,reP,reP]+"\\)$"),reHslaPercent=new RegExp("^hsla\\("+[reN,reP,reP,reN]+"\\)$");var named={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074};define(Color,color,{copy:function(channels){return Object.assign(new this.constructor,this,channels)},displayable:function(){return this.rgb().displayable()},hex:color_formatHex,// Deprecated! Use color.formatHex.
formatHex:color_formatHex,formatHsl:color_formatHsl,formatRgb:color_formatRgb,toString:color_formatRgb});function color_formatHex(){return this.rgb().formatHex()}function color_formatHsl(){return hslConvert(this).formatHsl()}function color_formatRgb(){return this.rgb().formatRgb()}function color(format){var m,l;format=(format+"").trim().toLowerCase();return(m=reHex.exec(format))?(l=m[1].length,m=parseInt(m[1],16),l===6?rgbn(m):l===3?new Rgb(m>>8&15|m>>4&240,m>>4&15|m&240,(m&15)<<4|m&15,1):l===8?new Rgb(m>>24&255,m>>16&255,m>>8&255,(m&255)/255):l===4?new Rgb(m>>12&15|m>>8&240,m>>8&15|m>>4&240,m>>4&15|m&240,((m&15)<<4|m&15)/255):null):// invalid hex
(m=reRgbInteger.exec(format))?new Rgb(m[1],m[2],m[3],1):(m=reRgbPercent.exec(format))?new Rgb(m[1]*255/100,m[2]*255/100,m[3]*255/100,1):(m=reRgbaInteger.exec(format))?rgba(m[1],m[2],m[3],m[4]):(m=reRgbaPercent.exec(format))?rgba(m[1]*255/100,m[2]*255/100,m[3]*255/100,m[4]):(m=reHslPercent.exec(format))?hsla(m[1],m[2]/100,m[3]/100,1):(m=reHslaPercent.exec(format))?hsla(m[1],m[2]/100,m[3]/100,m[4]):named.hasOwnProperty(format)?rgbn(named[format]):format==="transparent"?new Rgb(NaN,NaN,NaN,0):null}function rgbn(n){return new Rgb(n>>16&255,n>>8&255,n&255,1)}function rgba(r,g,b,a){if(a<=0)r=g=b=NaN;return new Rgb(r,g,b,a)}function rgbConvert(o){if(!(o instanceof Color))o=color(o);if(!o)return new Rgb;o=o.rgb();return new Rgb(o.r,o.g,o.b,o.opacity)}function rgb(r,g,b,opacity){return arguments.length===1?rgbConvert(r):new Rgb(r,g,b,opacity==null?1:opacity)}function Rgb(r,g,b,opacity){this.r=+r;this.g=+g;this.b=+b;this.opacity=+opacity}define(Rgb,rgb,extend(Color,{brighter:function(k){k=k==null?brighter:Math.pow(brighter,k);return new Rgb(this.r*k,this.g*k,this.b*k,this.opacity)},darker:function(k){k=k==null?darker:Math.pow(darker,k);return new Rgb(this.r*k,this.g*k,this.b*k,this.opacity)},rgb:function(){return this},displayable:function(){return-.5<=this.r&&this.r<255.5&&(-.5<=this.g&&this.g<255.5)&&(-.5<=this.b&&this.b<255.5)&&(0<=this.opacity&&this.opacity<=1)},hex:rgb_formatHex,// Deprecated! Use color.formatHex.
formatHex:rgb_formatHex,formatRgb:rgb_formatRgb,toString:rgb_formatRgb}));function rgb_formatHex(){return"#"+hex(this.r)+hex(this.g)+hex(this.b)}function rgb_formatRgb(){var a=this.opacity;a=isNaN(a)?1:Math.max(0,Math.min(1,a));return(a===1?"rgb(":"rgba(")+Math.max(0,Math.min(255,Math.round(this.r)||0))+", "+Math.max(0,Math.min(255,Math.round(this.g)||0))+", "+Math.max(0,Math.min(255,Math.round(this.b)||0))+(a===1?")":", "+a+")")}function hex(value){value=Math.max(0,Math.min(255,Math.round(value)||0));return(value<16?"0":"")+value.toString(16)}function hsla(h,s,l,a){if(a<=0)h=s=l=NaN;else if(l<=0||l>=1)h=s=NaN;else if(s<=0)h=NaN;return new Hsl(h,s,l,a)}function hslConvert(o){if(o instanceof Hsl)return new Hsl(o.h,o.s,o.l,o.opacity);if(!(o instanceof Color))o=color(o);if(!o)return new Hsl;if(o instanceof Hsl)return o;o=o.rgb();var r=o.r/255,g=o.g/255,b=o.b/255,min=Math.min(r,g,b),max=Math.max(r,g,b),h=NaN,s=max-min,l=(max+min)/2;if(s){if(r===max)h=(g-b)/s+(g<b)*6;else if(g===max)h=(b-r)/s+2;else h=(r-g)/s+4;s/=l<.5?max+min:2-max-min;h*=60}else{s=l>0&&l<1?0:h}return new Hsl(h,s,l,o.opacity)}function hsl(h,s,l,opacity){return arguments.length===1?hslConvert(h):new Hsl(h,s,l,opacity==null?1:opacity)}function Hsl(h,s,l,opacity){this.h=+h;this.s=+s;this.l=+l;this.opacity=+opacity}define(Hsl,hsl,extend(Color,{brighter:function(k){k=k==null?brighter:Math.pow(brighter,k);return new Hsl(this.h,this.s,this.l*k,this.opacity)},darker:function(k){k=k==null?darker:Math.pow(darker,k);return new Hsl(this.h,this.s,this.l*k,this.opacity)},rgb:function(){var h=this.h%360+(this.h<0)*360,s=isNaN(h)||isNaN(this.s)?0:this.s,l=this.l,m2=l+(l<.5?l:1-l)*s,m1=2*l-m2;return new Rgb(hsl2rgb(h>=240?h-240:h+120,m1,m2),hsl2rgb(h,m1,m2),hsl2rgb(h<120?h+240:h-120,m1,m2),this.opacity)},displayable:function(){return(0<=this.s&&this.s<=1||isNaN(this.s))&&(0<=this.l&&this.l<=1)&&(0<=this.opacity&&this.opacity<=1)},formatHsl:function(){var a=this.opacity;a=isNaN(a)?1:Math.max(0,Math.min(1,a));return(a===1?"hsl(":"hsla(")+(this.h||0)+", "+(this.s||0)*100+"%, "+(this.l||0)*100+"%"+(a===1?")":", "+a+")")}}));
/* From FvD 13.37, CSS Color Module Level 3 */function hsl2rgb(h,m1,m2){return(h<60?m1+(m2-m1)*h/60:h<180?m2:h<240?m1+(m2-m1)*(240-h)/60:m1)*255}var deg2rad=Math.PI/180;var rad2deg=180/Math.PI;
// https://observablehq.com/@mbostock/lab-and-rgb
var K=18,Xn=.96422,Yn=1,Zn=.82521,t0=4/29,t1=6/29,t2=3*t1*t1,t3=t1*t1*t1;function labConvert(o){if(o instanceof Lab)return new Lab(o.l,o.a,o.b,o.opacity);if(o instanceof Hcl)return hcl2lab(o);if(!(o instanceof Rgb))o=rgbConvert(o);var r=rgb2lrgb(o.r),g=rgb2lrgb(o.g),b=rgb2lrgb(o.b),y=xyz2lab((.2225045*r+.7168786*g+.0606169*b)/Yn),x,z;if(r===g&&g===b)x=z=y;else{x=xyz2lab((.4360747*r+.3850649*g+.1430804*b)/Xn);z=xyz2lab((.0139322*r+.0971045*g+.7141733*b)/Zn)}return new Lab(116*y-16,500*(x-y),200*(y-z),o.opacity)}function gray(l,opacity){return new Lab(l,0,0,opacity==null?1:opacity)}function lab(l,a,b,opacity){return arguments.length===1?labConvert(l):new Lab(l,a,b,opacity==null?1:opacity)}function Lab(l,a,b,opacity){this.l=+l;this.a=+a;this.b=+b;this.opacity=+opacity}define(Lab,lab,extend(Color,{brighter:function(k){return new Lab(this.l+K*(k==null?1:k),this.a,this.b,this.opacity)},darker:function(k){return new Lab(this.l-K*(k==null?1:k),this.a,this.b,this.opacity)},rgb:function(){var y=(this.l+16)/116,x=isNaN(this.a)?y:y+this.a/500,z=isNaN(this.b)?y:y-this.b/200;x=Xn*lab2xyz(x);y=Yn*lab2xyz(y);z=Zn*lab2xyz(z);return new Rgb(lrgb2rgb(3.1338561*x-1.6168667*y-.4906146*z),lrgb2rgb(-.9787684*x+1.9161415*y+.033454*z),lrgb2rgb(.0719453*x-.2289914*y+1.4052427*z),this.opacity)}}));function xyz2lab(t){return t>t3?Math.pow(t,1/3):t/t2+t0}function lab2xyz(t){return t>t1?t*t*t:t2*(t-t0)}function lrgb2rgb(x){return 255*(x<=.0031308?12.92*x:1.055*Math.pow(x,1/2.4)-.055)}function rgb2lrgb(x){return(x/=255)<=.04045?x/12.92:Math.pow((x+.055)/1.055,2.4)}function hclConvert(o){if(o instanceof Hcl)return new Hcl(o.h,o.c,o.l,o.opacity);if(!(o instanceof Lab))o=labConvert(o);if(o.a===0&&o.b===0)return new Hcl(NaN,0<o.l&&o.l<100?0:NaN,o.l,o.opacity);var h=Math.atan2(o.b,o.a)*rad2deg;return new Hcl(h<0?h+360:h,Math.sqrt(o.a*o.a+o.b*o.b),o.l,o.opacity)}function lch(l,c,h,opacity){return arguments.length===1?hclConvert(l):new Hcl(h,c,l,opacity==null?1:opacity)}function hcl(h,c,l,opacity){return arguments.length===1?hclConvert(h):new Hcl(h,c,l,opacity==null?1:opacity)}function Hcl(h,c,l,opacity){this.h=+h;this.c=+c;this.l=+l;this.opacity=+opacity}function hcl2lab(o){if(isNaN(o.h))return new Lab(o.l,0,0,o.opacity);var h=o.h*deg2rad;return new Lab(o.l,Math.cos(h)*o.c,Math.sin(h)*o.c,o.opacity)}define(Hcl,hcl,extend(Color,{brighter:function(k){return new Hcl(this.h,this.c,this.l+K*(k==null?1:k),this.opacity)},darker:function(k){return new Hcl(this.h,this.c,this.l-K*(k==null?1:k),this.opacity)},rgb:function(){return hcl2lab(this).rgb()}}));var A=-.14861,B=+1.78277,C=-.29227,D=-.90649,E=+1.97294,ED=E*D,EB=E*B,BC_DA=B*C-D*A;function cubehelixConvert(o){if(o instanceof Cubehelix)return new Cubehelix(o.h,o.s,o.l,o.opacity);if(!(o instanceof Rgb))o=rgbConvert(o);var r=o.r/255,g=o.g/255,b=o.b/255,l=(BC_DA*b+ED*r-EB*g)/(BC_DA+ED-EB),bl=b-l,k=(E*(g-l)-C*bl)/D,s=Math.sqrt(k*k+bl*bl)/(E*l*(1-l)),// NaN if l=0 or l=1
h=s?Math.atan2(k,bl)*rad2deg-120:NaN;return new Cubehelix(h<0?h+360:h,s,l,o.opacity)}function cubehelix(h,s,l,opacity){return arguments.length===1?cubehelixConvert(h):new Cubehelix(h,s,l,opacity==null?1:opacity)}function Cubehelix(h,s,l,opacity){this.h=+h;this.s=+s;this.l=+l;this.opacity=+opacity}define(Cubehelix,cubehelix,extend(Color,{brighter:function(k){k=k==null?brighter:Math.pow(brighter,k);return new Cubehelix(this.h,this.s,this.l*k,this.opacity)},darker:function(k){k=k==null?darker:Math.pow(darker,k);return new Cubehelix(this.h,this.s,this.l*k,this.opacity)},rgb:function(){var h=isNaN(this.h)?0:(this.h+120)*deg2rad,l=+this.l,a=isNaN(this.s)?0:this.s*l*(1-l),cosh=Math.cos(h),sinh=Math.sin(h);return new Rgb(255*(l+a*(A*cosh+B*sinh)),255*(l+a*(C*cosh+D*sinh)),255*(l+a*(E*cosh)),this.opacity)}}));exports.color=color;exports.cubehelix=cubehelix;exports.gray=gray;exports.hcl=hcl;exports.hsl=hsl;exports.lab=lab;exports.lch=lch;exports.rgb=rgb;Object.defineProperty(exports,"__esModule",{value:true})})},{}],35:[function(require,module,exports){
// https://d3js.org/d3-contour/ v1.3.2 Copyright 2018 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports,require("d3-array")):typeof define==="function"&&define.amd?define(["exports","d3-array"],factory):factory(global.d3=global.d3||{},global.d3)})(this,function(exports,d3Array){"use strict";var array=Array.prototype;var slice=array.slice;function ascending(a,b){return a-b}function area(ring){var i=0,n=ring.length,area=ring[n-1][1]*ring[0][0]-ring[n-1][0]*ring[0][1];while(++i<n)area+=ring[i-1][1]*ring[i][0]-ring[i-1][0]*ring[i][1];return area}function constant(x){return function(){return x}}function contains(ring,hole){var i=-1,n=hole.length,c;while(++i<n)if(c=ringContains(ring,hole[i]))return c;return 0}function ringContains(ring,point){var x=point[0],y=point[1],contains=-1;for(var i=0,n=ring.length,j=n-1;i<n;j=i++){var pi=ring[i],xi=pi[0],yi=pi[1],pj=ring[j],xj=pj[0],yj=pj[1];if(segmentContains(pi,pj,point))return 0;if(yi>y!==yj>y&&x<(xj-xi)*(y-yi)/(yj-yi)+xi)contains=-contains}return contains}function segmentContains(a,b,c){var i;return collinear(a,b,c)&&within(a[i=+(a[0]===b[0])],c[i],b[i])}function collinear(a,b,c){return(b[0]-a[0])*(c[1]-a[1])===(c[0]-a[0])*(b[1]-a[1])}function within(p,q,r){return p<=q&&q<=r||r<=q&&q<=p}function noop(){}var cases=[[],[[[1,1.5],[.5,1]]],[[[1.5,1],[1,1.5]]],[[[1.5,1],[.5,1]]],[[[1,.5],[1.5,1]]],[[[1,1.5],[.5,1]],[[1,.5],[1.5,1]]],[[[1,.5],[1,1.5]]],[[[1,.5],[.5,1]]],[[[.5,1],[1,.5]]],[[[1,1.5],[1,.5]]],[[[.5,1],[1,.5]],[[1.5,1],[1,1.5]]],[[[1.5,1],[1,.5]]],[[[.5,1],[1.5,1]]],[[[1,1.5],[1.5,1]]],[[[.5,1],[1,1.5]]],[]];function contours(){var dx=1,dy=1,threshold=d3Array.thresholdSturges,smooth=smoothLinear;function contours(values){var tz=threshold(values);
// Convert number of thresholds into uniform thresholds.
if(!Array.isArray(tz)){var domain=d3Array.extent(values),start=domain[0],stop=domain[1];tz=d3Array.tickStep(start,stop,tz);tz=d3Array.range(Math.floor(start/tz)*tz,Math.floor(stop/tz)*tz,tz)}else{tz=tz.slice().sort(ascending)}return tz.map(function(value){return contour(values,value)})}
// Accumulate, smooth contour rings, assign holes to exterior rings.
// Based on https://github.com/mbostock/shapefile/blob/v0.6.2/shp/polygon.js
function contour(values,value){var polygons=[],holes=[];isorings(values,value,function(ring){smooth(ring,values,value);if(area(ring)>0)polygons.push([ring]);else holes.push(ring)});holes.forEach(function(hole){for(var i=0,n=polygons.length,polygon;i<n;++i){if(contains((polygon=polygons[i])[0],hole)!==-1){polygon.push(hole);return}}});return{type:"MultiPolygon",value:value,coordinates:polygons}}
// Marching squares with isolines stitched into rings.
// Based on https://github.com/topojson/topojson-client/blob/v3.0.0/src/stitch.js
function isorings(values,value,callback){var fragmentByStart=new Array,fragmentByEnd=new Array,x,y,t0,t1,t2,t3;
// Special case for the first row (y = -1, t2 = t3 = 0).
x=y=-1;t1=values[0]>=value;cases[t1<<1].forEach(stitch);while(++x<dx-1){t0=t1,t1=values[x+1]>=value;cases[t0|t1<<1].forEach(stitch)}cases[t1<<0].forEach(stitch);
// General case for the intermediate rows.
while(++y<dy-1){x=-1;t1=values[y*dx+dx]>=value;t2=values[y*dx]>=value;cases[t1<<1|t2<<2].forEach(stitch);while(++x<dx-1){t0=t1,t1=values[y*dx+dx+x+1]>=value;t3=t2,t2=values[y*dx+x+1]>=value;cases[t0|t1<<1|t2<<2|t3<<3].forEach(stitch)}cases[t1|t2<<3].forEach(stitch)}
// Special case for the last row (y = dy - 1, t0 = t1 = 0).
x=-1;t2=values[y*dx]>=value;cases[t2<<2].forEach(stitch);while(++x<dx-1){t3=t2,t2=values[y*dx+x+1]>=value;cases[t2<<2|t3<<3].forEach(stitch)}cases[t2<<3].forEach(stitch);function stitch(line){var start=[line[0][0]+x,line[0][1]+y],end=[line[1][0]+x,line[1][1]+y],startIndex=index(start),endIndex=index(end),f,g;if(f=fragmentByEnd[startIndex]){if(g=fragmentByStart[endIndex]){delete fragmentByEnd[f.end];delete fragmentByStart[g.start];if(f===g){f.ring.push(end);callback(f.ring)}else{fragmentByStart[f.start]=fragmentByEnd[g.end]={start:f.start,end:g.end,ring:f.ring.concat(g.ring)}}}else{delete fragmentByEnd[f.end];f.ring.push(end);fragmentByEnd[f.end=endIndex]=f}}else if(f=fragmentByStart[endIndex]){if(g=fragmentByEnd[startIndex]){delete fragmentByStart[f.start];delete fragmentByEnd[g.end];if(f===g){f.ring.push(end);callback(f.ring)}else{fragmentByStart[g.start]=fragmentByEnd[f.end]={start:g.start,end:f.end,ring:g.ring.concat(f.ring)}}}else{delete fragmentByStart[f.start];f.ring.unshift(start);fragmentByStart[f.start=startIndex]=f}}else{fragmentByStart[startIndex]=fragmentByEnd[endIndex]={start:startIndex,end:endIndex,ring:[start,end]}}}}function index(point){return point[0]*2+point[1]*(dx+1)*4}function smoothLinear(ring,values,value){ring.forEach(function(point){var x=point[0],y=point[1],xt=x|0,yt=y|0,v0,v1=values[yt*dx+xt];if(x>0&&x<dx&&xt===x){v0=values[yt*dx+xt-1];point[0]=x+(value-v0)/(v1-v0)-.5}if(y>0&&y<dy&&yt===y){v0=values[(yt-1)*dx+xt];point[1]=y+(value-v0)/(v1-v0)-.5}})}contours.contour=contour;contours.size=function(_){if(!arguments.length)return[dx,dy];var _0=Math.ceil(_[0]),_1=Math.ceil(_[1]);if(!(_0>0)||!(_1>0))throw new Error("invalid size");return dx=_0,dy=_1,contours};contours.thresholds=function(_){return arguments.length?(threshold=typeof _==="function"?_:Array.isArray(_)?constant(slice.call(_)):constant(_),contours):threshold};contours.smooth=function(_){return arguments.length?(smooth=_?smoothLinear:noop,contours):smooth===smoothLinear};return contours}
// TODO Optimize edge cases.
// TODO Optimize index calculation.
// TODO Optimize arguments.
function blurX(source,target,r){var n=source.width,m=source.height,w=(r<<1)+1;for(var j=0;j<m;++j){for(var i=0,sr=0;i<n+r;++i){if(i<n){sr+=source.data[i+j*n]}if(i>=r){if(i>=w){sr-=source.data[i-w+j*n]}target.data[i-r+j*n]=sr/Math.min(i+1,n-1+w-i,w)}}}}
// TODO Optimize edge cases.
// TODO Optimize index calculation.
// TODO Optimize arguments.
function blurY(source,target,r){var n=source.width,m=source.height,w=(r<<1)+1;for(var i=0;i<n;++i){for(var j=0,sr=0;j<m+r;++j){if(j<m){sr+=source.data[i+j*n]}if(j>=r){if(j>=w){sr-=source.data[i+(j-w)*n]}target.data[i+(j-r)*n]=sr/Math.min(j+1,m-1+w-j,w)}}}}function defaultX(d){return d[0]}function defaultY(d){return d[1]}function defaultWeight(){return 1}function density(){var x=defaultX,y=defaultY,weight=defaultWeight,dx=960,dy=500,r=20,// blur radius
k=2,// log2(grid cell size)
o=r*3,// grid offset, to pad for blur
n=dx+o*2>>k,// grid width
m=dy+o*2>>k,// grid height
threshold=constant(20);function density(data){var values0=new Float32Array(n*m),values1=new Float32Array(n*m);data.forEach(function(d,i,data){var xi=+x(d,i,data)+o>>k,yi=+y(d,i,data)+o>>k,wi=+weight(d,i,data);if(xi>=0&&xi<n&&yi>=0&&yi<m){values0[xi+yi*n]+=wi}});
// TODO Optimize.
blurX({width:n,height:m,data:values0},{width:n,height:m,data:values1},r>>k);blurY({width:n,height:m,data:values1},{width:n,height:m,data:values0},r>>k);blurX({width:n,height:m,data:values0},{width:n,height:m,data:values1},r>>k);blurY({width:n,height:m,data:values1},{width:n,height:m,data:values0},r>>k);blurX({width:n,height:m,data:values0},{width:n,height:m,data:values1},r>>k);blurY({width:n,height:m,data:values1},{width:n,height:m,data:values0},r>>k);var tz=threshold(values0);
// Convert number of thresholds into uniform thresholds.
if(!Array.isArray(tz)){var stop=d3Array.max(values0);tz=d3Array.tickStep(0,stop,tz);tz=d3Array.range(0,Math.floor(stop/tz)*tz,tz);tz.shift()}return contours().thresholds(tz).size([n,m])(values0).map(transform)}function transform(geometry){geometry.value*=Math.pow(2,-2*k);// Density in points per square pixel.
geometry.coordinates.forEach(transformPolygon);return geometry}function transformPolygon(coordinates){coordinates.forEach(transformRing)}function transformRing(coordinates){coordinates.forEach(transformPoint)}
// TODO Optimize.
function transformPoint(coordinates){coordinates[0]=coordinates[0]*Math.pow(2,k)-o;coordinates[1]=coordinates[1]*Math.pow(2,k)-o}function resize(){o=r*3;n=dx+o*2>>k;m=dy+o*2>>k;return density}density.x=function(_){return arguments.length?(x=typeof _==="function"?_:constant(+_),density):x};density.y=function(_){return arguments.length?(y=typeof _==="function"?_:constant(+_),density):y};density.weight=function(_){return arguments.length?(weight=typeof _==="function"?_:constant(+_),density):weight};density.size=function(_){if(!arguments.length)return[dx,dy];var _0=Math.ceil(_[0]),_1=Math.ceil(_[1]);if(!(_0>=0)&&!(_0>=0))throw new Error("invalid size");return dx=_0,dy=_1,resize()};density.cellSize=function(_){if(!arguments.length)return 1<<k;if(!((_=+_)>=1))throw new Error("invalid cell size");return k=Math.floor(Math.log(_)/Math.LN2),resize()};density.thresholds=function(_){return arguments.length?(threshold=typeof _==="function"?_:Array.isArray(_)?constant(slice.call(_)):constant(_),density):threshold};density.bandwidth=function(_){if(!arguments.length)return Math.sqrt(r*(r+1));if(!((_=+_)>=0))throw new Error("invalid bandwidth");return r=Math.round((Math.sqrt(4*_*_+1)-1)/2),resize()};return density}exports.contours=contours;exports.contourDensity=density;Object.defineProperty(exports,"__esModule",{value:true})})},{"d3-array":29}],36:[function(require,module,exports){
// https://d3js.org/d3-dispatch/ v1.0.6 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):(global=global||self,factory(global.d3=global.d3||{}))})(this,function(exports){"use strict";var noop={value:function(){}};function dispatch(){for(var i=0,n=arguments.length,_={},t;i<n;++i){if(!(t=arguments[i]+"")||t in _||/[\s.]/.test(t))throw new Error("illegal type: "+t);_[t]=[]}return new Dispatch(_)}function Dispatch(_){this._=_}function parseTypenames(typenames,types){return typenames.trim().split(/^|\s+/).map(function(t){var name="",i=t.indexOf(".");if(i>=0)name=t.slice(i+1),t=t.slice(0,i);if(t&&!types.hasOwnProperty(t))throw new Error("unknown type: "+t);return{type:t,name:name}})}Dispatch.prototype=dispatch.prototype={constructor:Dispatch,on:function(typename,callback){var _=this._,T=parseTypenames(typename+"",_),t,i=-1,n=T.length;
// If no callback was specified, return the callback of the given type and name.
if(arguments.length<2){while(++i<n)if((t=(typename=T[i]).type)&&(t=get(_[t],typename.name)))return t;return}
// If a type was specified, set the callback for the given type and name.
// Otherwise, if a null callback was specified, remove callbacks of the given name.
if(callback!=null&&typeof callback!=="function")throw new Error("invalid callback: "+callback);while(++i<n){if(t=(typename=T[i]).type)_[t]=set(_[t],typename.name,callback);else if(callback==null)for(t in _)_[t]=set(_[t],typename.name,null)}return this},copy:function(){var copy={},_=this._;for(var t in _)copy[t]=_[t].slice();return new Dispatch(copy)},call:function(type,that){if((n=arguments.length-2)>0)for(var args=new Array(n),i=0,n,t;i<n;++i)args[i]=arguments[i+2];if(!this._.hasOwnProperty(type))throw new Error("unknown type: "+type);for(t=this._[type],i=0,n=t.length;i<n;++i)t[i].value.apply(that,args)},apply:function(type,that,args){if(!this._.hasOwnProperty(type))throw new Error("unknown type: "+type);for(var t=this._[type],i=0,n=t.length;i<n;++i)t[i].value.apply(that,args)}};function get(type,name){for(var i=0,n=type.length,c;i<n;++i){if((c=type[i]).name===name){return c.value}}}function set(type,name,callback){for(var i=0,n=type.length;i<n;++i){if(type[i].name===name){type[i]=noop,type=type.slice(0,i).concat(type.slice(i+1));break}}if(callback!=null)type.push({name:name,value:callback});return type}exports.dispatch=dispatch;Object.defineProperty(exports,"__esModule",{value:true})})},{}],37:[function(require,module,exports){
// https://d3js.org/d3-drag/ v1.2.5 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports,require("d3-dispatch"),require("d3-selection")):typeof define==="function"&&define.amd?define(["exports","d3-dispatch","d3-selection"],factory):(global=global||self,factory(global.d3=global.d3||{},global.d3,global.d3))})(this,function(exports,d3Dispatch,d3Selection){"use strict";function nopropagation(){d3Selection.event.stopImmediatePropagation()}function noevent(){d3Selection.event.preventDefault();d3Selection.event.stopImmediatePropagation()}function nodrag(view){var root=view.document.documentElement,selection=d3Selection.select(view).on("dragstart.drag",noevent,true);if("onselectstart"in root){selection.on("selectstart.drag",noevent,true)}else{root.__noselect=root.style.MozUserSelect;root.style.MozUserSelect="none"}}function yesdrag(view,noclick){var root=view.document.documentElement,selection=d3Selection.select(view).on("dragstart.drag",null);if(noclick){selection.on("click.drag",noevent,true);setTimeout(function(){selection.on("click.drag",null)},0)}if("onselectstart"in root){selection.on("selectstart.drag",null)}else{root.style.MozUserSelect=root.__noselect;delete root.__noselect}}function constant(x){return function(){return x}}function DragEvent(target,type,subject,id,active,x,y,dx,dy,dispatch){this.target=target;this.type=type;this.subject=subject;this.identifier=id;this.active=active;this.x=x;this.y=y;this.dx=dx;this.dy=dy;this._=dispatch}DragEvent.prototype.on=function(){var value=this._.on.apply(this._,arguments);return value===this._?this:value};
// Ignore right-click, since that should open the context menu.
function defaultFilter(){return!d3Selection.event.ctrlKey&&!d3Selection.event.button}function defaultContainer(){return this.parentNode}function defaultSubject(d){return d==null?{x:d3Selection.event.x,y:d3Selection.event.y}:d}function defaultTouchable(){return navigator.maxTouchPoints||"ontouchstart"in this}function drag(){var filter=defaultFilter,container=defaultContainer,subject=defaultSubject,touchable=defaultTouchable,gestures={},listeners=d3Dispatch.dispatch("start","drag","end"),active=0,mousedownx,mousedowny,mousemoving,touchending,clickDistance2=0;function drag(selection){selection.on("mousedown.drag",mousedowned).filter(touchable).on("touchstart.drag",touchstarted).on("touchmove.drag",touchmoved).on("touchend.drag touchcancel.drag",touchended).style("touch-action","none").style("-webkit-tap-highlight-color","rgba(0,0,0,0)")}function mousedowned(){if(touchending||!filter.apply(this,arguments))return;var gesture=beforestart("mouse",container.apply(this,arguments),d3Selection.mouse,this,arguments);if(!gesture)return;d3Selection.select(d3Selection.event.view).on("mousemove.drag",mousemoved,true).on("mouseup.drag",mouseupped,true);nodrag(d3Selection.event.view);nopropagation();mousemoving=false;mousedownx=d3Selection.event.clientX;mousedowny=d3Selection.event.clientY;gesture("start")}function mousemoved(){noevent();if(!mousemoving){var dx=d3Selection.event.clientX-mousedownx,dy=d3Selection.event.clientY-mousedowny;mousemoving=dx*dx+dy*dy>clickDistance2}gestures.mouse("drag")}function mouseupped(){d3Selection.select(d3Selection.event.view).on("mousemove.drag mouseup.drag",null);yesdrag(d3Selection.event.view,mousemoving);noevent();gestures.mouse("end")}function touchstarted(){if(!filter.apply(this,arguments))return;var touches=d3Selection.event.changedTouches,c=container.apply(this,arguments),n=touches.length,i,gesture;for(i=0;i<n;++i){if(gesture=beforestart(touches[i].identifier,c,d3Selection.touch,this,arguments)){nopropagation();gesture("start")}}}function touchmoved(){var touches=d3Selection.event.changedTouches,n=touches.length,i,gesture;for(i=0;i<n;++i){if(gesture=gestures[touches[i].identifier]){noevent();gesture("drag")}}}function touchended(){var touches=d3Selection.event.changedTouches,n=touches.length,i,gesture;if(touchending)clearTimeout(touchending);touchending=setTimeout(function(){touchending=null},500);// Ghost clicks are delayed!
for(i=0;i<n;++i){if(gesture=gestures[touches[i].identifier]){nopropagation();gesture("end")}}}function beforestart(id,container,point,that,args){var p=point(container,id),s,dx,dy,sublisteners=listeners.copy();if(!d3Selection.customEvent(new DragEvent(drag,"beforestart",s,id,active,p[0],p[1],0,0,sublisteners),function(){if((d3Selection.event.subject=s=subject.apply(that,args))==null)return false;dx=s.x-p[0]||0;dy=s.y-p[1]||0;return true}))return;return function gesture(type){var p0=p,n;switch(type){case"start":gestures[id]=gesture,n=active++;break;case"end":delete gestures[id],--active;// nobreak
case"drag":p=point(container,id),n=active;break}d3Selection.customEvent(new DragEvent(drag,type,s,id,n,p[0]+dx,p[1]+dy,p[0]-p0[0],p[1]-p0[1],sublisteners),sublisteners.apply,sublisteners,[type,that,args])}}drag.filter=function(_){return arguments.length?(filter=typeof _==="function"?_:constant(!!_),drag):filter};drag.container=function(_){return arguments.length?(container=typeof _==="function"?_:constant(_),drag):container};drag.subject=function(_){return arguments.length?(subject=typeof _==="function"?_:constant(_),drag):subject};drag.touchable=function(_){return arguments.length?(touchable=typeof _==="function"?_:constant(!!_),drag):touchable};drag.on=function(){var value=listeners.on.apply(listeners,arguments);return value===listeners?drag:value};drag.clickDistance=function(_){return arguments.length?(clickDistance2=(_=+_)*_,drag):Math.sqrt(clickDistance2)};return drag}exports.drag=drag;exports.dragDisable=nodrag;exports.dragEnable=yesdrag;Object.defineProperty(exports,"__esModule",{value:true})})},{"d3-dispatch":36,"d3-selection":52}],38:[function(require,module,exports){
// https://d3js.org/d3-dsv/ v1.2.0 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):(global=global||self,factory(global.d3=global.d3||{}))})(this,function(exports){"use strict";var EOL={},EOF={},QUOTE=34,NEWLINE=10,RETURN=13;function objectConverter(columns){return new Function("d","return {"+columns.map(function(name,i){return JSON.stringify(name)+": d["+i+'] || ""'}).join(",")+"}")}function customConverter(columns,f){var object=objectConverter(columns);return function(row,i){return f(object(row),i,columns)}}
// Compute unique columns in order of discovery.
function inferColumns(rows){var columnSet=Object.create(null),columns=[];rows.forEach(function(row){for(var column in row){if(!(column in columnSet)){columns.push(columnSet[column]=column)}}});return columns}function pad(value,width){var s=value+"",length=s.length;return length<width?new Array(width-length+1).join(0)+s:s}function formatYear(year){return year<0?"-"+pad(-year,6):year>9999?"+"+pad(year,6):pad(year,4)}function formatDate(date){var hours=date.getUTCHours(),minutes=date.getUTCMinutes(),seconds=date.getUTCSeconds(),milliseconds=date.getUTCMilliseconds();return isNaN(date)?"Invalid Date":formatYear(date.getUTCFullYear())+"-"+pad(date.getUTCMonth()+1,2)+"-"+pad(date.getUTCDate(),2)+(milliseconds?"T"+pad(hours,2)+":"+pad(minutes,2)+":"+pad(seconds,2)+"."+pad(milliseconds,3)+"Z":seconds?"T"+pad(hours,2)+":"+pad(minutes,2)+":"+pad(seconds,2)+"Z":minutes||hours?"T"+pad(hours,2)+":"+pad(minutes,2)+"Z":"")}function dsv(delimiter){var reFormat=new RegExp('["'+delimiter+"\n\r]"),DELIMITER=delimiter.charCodeAt(0);function parse(text,f){var convert,columns,rows=parseRows(text,function(row,i){if(convert)return convert(row,i-1);columns=row,convert=f?customConverter(row,f):objectConverter(row)});rows.columns=columns||[];return rows}function parseRows(text,f){var rows=[],// output rows
N=text.length,I=0,// current character index
n=0,// current line number
t,// current token
eof=N<=0,// current token followed by EOF?
eol=false;// current token followed by EOL?
// Strip the trailing newline.
if(text.charCodeAt(N-1)===NEWLINE)--N;if(text.charCodeAt(N-1)===RETURN)--N;function token(){if(eof)return EOF;if(eol)return eol=false,EOL;
// Unescape quotes.
var i,j=I,c;if(text.charCodeAt(j)===QUOTE){while(I++<N&&text.charCodeAt(I)!==QUOTE||text.charCodeAt(++I)===QUOTE);if((i=I)>=N)eof=true;else if((c=text.charCodeAt(I++))===NEWLINE)eol=true;else if(c===RETURN){eol=true;if(text.charCodeAt(I)===NEWLINE)++I}return text.slice(j+1,i-1).replace(/""/g,'"')}
// Find next delimiter or newline.
while(I<N){if((c=text.charCodeAt(i=I++))===NEWLINE)eol=true;else if(c===RETURN){eol=true;if(text.charCodeAt(I)===NEWLINE)++I}else if(c!==DELIMITER)continue;return text.slice(j,i)}
// Return last token before EOF.
return eof=true,text.slice(j,N)}while((t=token())!==EOF){var row=[];while(t!==EOL&&t!==EOF)row.push(t),t=token();if(f&&(row=f(row,n++))==null)continue;rows.push(row)}return rows}function preformatBody(rows,columns){return rows.map(function(row){return columns.map(function(column){return formatValue(row[column])}).join(delimiter)})}function format(rows,columns){if(columns==null)columns=inferColumns(rows);return[columns.map(formatValue).join(delimiter)].concat(preformatBody(rows,columns)).join("\n")}function formatBody(rows,columns){if(columns==null)columns=inferColumns(rows);return preformatBody(rows,columns).join("\n")}function formatRows(rows){return rows.map(formatRow).join("\n")}function formatRow(row){return row.map(formatValue).join(delimiter)}function formatValue(value){return value==null?"":value instanceof Date?formatDate(value):reFormat.test(value+="")?'"'+value.replace(/"/g,'""')+'"':value}return{parse:parse,parseRows:parseRows,format:format,formatBody:formatBody,formatRows:formatRows,formatRow:formatRow,formatValue:formatValue}}var csv=dsv(",");var csvParse=csv.parse;var csvParseRows=csv.parseRows;var csvFormat=csv.format;var csvFormatBody=csv.formatBody;var csvFormatRows=csv.formatRows;var csvFormatRow=csv.formatRow;var csvFormatValue=csv.formatValue;var tsv=dsv("\t");var tsvParse=tsv.parse;var tsvParseRows=tsv.parseRows;var tsvFormat=tsv.format;var tsvFormatBody=tsv.formatBody;var tsvFormatRows=tsv.formatRows;var tsvFormatRow=tsv.formatRow;var tsvFormatValue=tsv.formatValue;function autoType(object){for(var key in object){var value=object[key].trim(),number,m;if(!value)value=null;else if(value==="true")value=true;else if(value==="false")value=false;else if(value==="NaN")value=NaN;else if(!isNaN(number=+value))value=number;else if(m=value.match(/^([-+]\d{2})?\d{4}(-\d{2}(-\d{2})?)?(T\d{2}:\d{2}(:\d{2}(\.\d{3})?)?(Z|[-+]\d{2}:\d{2})?)?$/)){if(fixtz&&!!m[4]&&!m[7])value=value.replace(/-/g,"/").replace(/T/," ");value=new Date(value)}else continue;object[key]=value}return object}
// https://github.com/d3/d3-dsv/issues/45
var fixtz=new Date("2019-01-01T00:00").getHours()||new Date("2019-07-01T00:00").getHours();exports.autoType=autoType;exports.csvFormat=csvFormat;exports.csvFormatBody=csvFormatBody;exports.csvFormatRow=csvFormatRow;exports.csvFormatRows=csvFormatRows;exports.csvFormatValue=csvFormatValue;exports.csvParse=csvParse;exports.csvParseRows=csvParseRows;exports.dsvFormat=dsv;exports.tsvFormat=tsvFormat;exports.tsvFormatBody=tsvFormatBody;exports.tsvFormatRow=tsvFormatRow;exports.tsvFormatRows=tsvFormatRows;exports.tsvFormatValue=tsvFormatValue;exports.tsvParse=tsvParse;exports.tsvParseRows=tsvParseRows;Object.defineProperty(exports,"__esModule",{value:true})})},{}],39:[function(require,module,exports){
// https://d3js.org/d3-ease/ v1.0.6 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):(global=global||self,factory(global.d3=global.d3||{}))})(this,function(exports){"use strict";function linear(t){return+t}function quadIn(t){return t*t}function quadOut(t){return t*(2-t)}function quadInOut(t){return((t*=2)<=1?t*t:--t*(2-t)+1)/2}function cubicIn(t){return t*t*t}function cubicOut(t){return--t*t*t+1}function cubicInOut(t){return((t*=2)<=1?t*t*t:(t-=2)*t*t+2)/2}var exponent=3;var polyIn=function custom(e){e=+e;function polyIn(t){return Math.pow(t,e)}polyIn.exponent=custom;return polyIn}(exponent);var polyOut=function custom(e){e=+e;function polyOut(t){return 1-Math.pow(1-t,e)}polyOut.exponent=custom;return polyOut}(exponent);var polyInOut=function custom(e){e=+e;function polyInOut(t){return((t*=2)<=1?Math.pow(t,e):2-Math.pow(2-t,e))/2}polyInOut.exponent=custom;return polyInOut}(exponent);var pi=Math.PI,halfPi=pi/2;function sinIn(t){return 1-Math.cos(t*halfPi)}function sinOut(t){return Math.sin(t*halfPi)}function sinInOut(t){return(1-Math.cos(pi*t))/2}function expIn(t){return Math.pow(2,10*t-10)}function expOut(t){return 1-Math.pow(2,-10*t)}function expInOut(t){return((t*=2)<=1?Math.pow(2,10*t-10):2-Math.pow(2,10-10*t))/2}function circleIn(t){return 1-Math.sqrt(1-t*t)}function circleOut(t){return Math.sqrt(1- --t*t)}function circleInOut(t){return((t*=2)<=1?1-Math.sqrt(1-t*t):Math.sqrt(1-(t-=2)*t)+1)/2}var b1=4/11,b2=6/11,b3=8/11,b4=3/4,b5=9/11,b6=10/11,b7=15/16,b8=21/22,b9=63/64,b0=1/b1/b1;function bounceIn(t){return 1-bounceOut(1-t)}function bounceOut(t){return(t=+t)<b1?b0*t*t:t<b3?b0*(t-=b2)*t+b4:t<b6?b0*(t-=b5)*t+b7:b0*(t-=b8)*t+b9}function bounceInOut(t){return((t*=2)<=1?1-bounceOut(1-t):bounceOut(t-1)+1)/2}var overshoot=1.70158;var backIn=function custom(s){s=+s;function backIn(t){return t*t*((s+1)*t-s)}backIn.overshoot=custom;return backIn}(overshoot);var backOut=function custom(s){s=+s;function backOut(t){return--t*t*((s+1)*t+s)+1}backOut.overshoot=custom;return backOut}(overshoot);var backInOut=function custom(s){s=+s;function backInOut(t){return((t*=2)<1?t*t*((s+1)*t-s):(t-=2)*t*((s+1)*t+s)+2)/2}backInOut.overshoot=custom;return backInOut}(overshoot);var tau=2*Math.PI,amplitude=1,period=.3;var elasticIn=function custom(a,p){var s=Math.asin(1/(a=Math.max(1,a)))*(p/=tau);function elasticIn(t){return a*Math.pow(2,10*--t)*Math.sin((s-t)/p)}elasticIn.amplitude=function(a){return custom(a,p*tau)};elasticIn.period=function(p){return custom(a,p)};return elasticIn}(amplitude,period);var elasticOut=function custom(a,p){var s=Math.asin(1/(a=Math.max(1,a)))*(p/=tau);function elasticOut(t){return 1-a*Math.pow(2,-10*(t=+t))*Math.sin((t+s)/p)}elasticOut.amplitude=function(a){return custom(a,p*tau)};elasticOut.period=function(p){return custom(a,p)};return elasticOut}(amplitude,period);var elasticInOut=function custom(a,p){var s=Math.asin(1/(a=Math.max(1,a)))*(p/=tau);function elasticInOut(t){return((t=t*2-1)<0?a*Math.pow(2,10*t)*Math.sin((s-t)/p):2-a*Math.pow(2,-10*t)*Math.sin((s+t)/p))/2}elasticInOut.amplitude=function(a){return custom(a,p*tau)};elasticInOut.period=function(p){return custom(a,p)};return elasticInOut}(amplitude,period);exports.easeBack=backInOut;exports.easeBackIn=backIn;exports.easeBackInOut=backInOut;exports.easeBackOut=backOut;exports.easeBounce=bounceOut;exports.easeBounceIn=bounceIn;exports.easeBounceInOut=bounceInOut;exports.easeBounceOut=bounceOut;exports.easeCircle=circleInOut;exports.easeCircleIn=circleIn;exports.easeCircleInOut=circleInOut;exports.easeCircleOut=circleOut;exports.easeCubic=cubicInOut;exports.easeCubicIn=cubicIn;exports.easeCubicInOut=cubicInOut;exports.easeCubicOut=cubicOut;exports.easeElastic=elasticOut;exports.easeElasticIn=elasticIn;exports.easeElasticInOut=elasticInOut;exports.easeElasticOut=elasticOut;exports.easeExp=expInOut;exports.easeExpIn=expIn;exports.easeExpInOut=expInOut;exports.easeExpOut=expOut;exports.easeLinear=linear;exports.easePoly=polyInOut;exports.easePolyIn=polyIn;exports.easePolyInOut=polyInOut;exports.easePolyOut=polyOut;exports.easeQuad=quadInOut;exports.easeQuadIn=quadIn;exports.easeQuadInOut=quadInOut;exports.easeQuadOut=quadOut;exports.easeSin=sinInOut;exports.easeSinIn=sinIn;exports.easeSinInOut=sinInOut;exports.easeSinOut=sinOut;Object.defineProperty(exports,"__esModule",{value:true})})},{}],40:[function(require,module,exports){
// https://d3js.org/d3-fetch/ v1.1.2 Copyright 2018 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports,require("d3-dsv")):typeof define==="function"&&define.amd?define(["exports","d3-dsv"],factory):factory(global.d3=global.d3||{},global.d3)})(this,function(exports,d3Dsv){"use strict";function responseBlob(response){if(!response.ok)throw new Error(response.status+" "+response.statusText);return response.blob()}function blob(input,init){return fetch(input,init).then(responseBlob)}function responseArrayBuffer(response){if(!response.ok)throw new Error(response.status+" "+response.statusText);return response.arrayBuffer()}function buffer(input,init){return fetch(input,init).then(responseArrayBuffer)}function responseText(response){if(!response.ok)throw new Error(response.status+" "+response.statusText);return response.text()}function text(input,init){return fetch(input,init).then(responseText)}function dsvParse(parse){return function(input,init,row){if(arguments.length===2&&typeof init==="function")row=init,init=undefined;return text(input,init).then(function(response){return parse(response,row)})}}function dsv(delimiter,input,init,row){if(arguments.length===3&&typeof init==="function")row=init,init=undefined;var format=d3Dsv.dsvFormat(delimiter);return text(input,init).then(function(response){return format.parse(response,row)})}var csv=dsvParse(d3Dsv.csvParse);var tsv=dsvParse(d3Dsv.tsvParse);function image(input,init){return new Promise(function(resolve,reject){var image=new Image;for(var key in init)image[key]=init[key];image.onerror=reject;image.onload=function(){resolve(image)};image.src=input})}function responseJson(response){if(!response.ok)throw new Error(response.status+" "+response.statusText);return response.json()}function json(input,init){return fetch(input,init).then(responseJson)}function parser(type){return function(input,init){return text(input,init).then(function(text$$1){return(new DOMParser).parseFromString(text$$1,type)})}}var xml=parser("application/xml");var html=parser("text/html");var svg=parser("image/svg+xml");exports.blob=blob;exports.buffer=buffer;exports.dsv=dsv;exports.csv=csv;exports.tsv=tsv;exports.image=image;exports.json=json;exports.text=text;exports.xml=xml;exports.html=html;exports.svg=svg;Object.defineProperty(exports,"__esModule",{value:true})})},{"d3-dsv":38}],41:[function(require,module,exports){
// https://d3js.org/d3-force/ v1.2.1 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports,require("d3-quadtree"),require("d3-collection"),require("d3-dispatch"),require("d3-timer")):typeof define==="function"&&define.amd?define(["exports","d3-quadtree","d3-collection","d3-dispatch","d3-timer"],factory):factory(global.d3=global.d3||{},global.d3,global.d3,global.d3,global.d3)})(this,function(exports,d3Quadtree,d3Collection,d3Dispatch,d3Timer){"use strict";function center(x,y){var nodes;if(x==null)x=0;if(y==null)y=0;function force(){var i,n=nodes.length,node,sx=0,sy=0;for(i=0;i<n;++i){node=nodes[i],sx+=node.x,sy+=node.y}for(sx=sx/n-x,sy=sy/n-y,i=0;i<n;++i){node=nodes[i],node.x-=sx,node.y-=sy}}force.initialize=function(_){nodes=_};force.x=function(_){return arguments.length?(x=+_,force):x};force.y=function(_){return arguments.length?(y=+_,force):y};return force}function constant(x){return function(){return x}}function jiggle(){return(Math.random()-.5)*1e-6}function x(d){return d.x+d.vx}function y(d){return d.y+d.vy}function collide(radius){var nodes,radii,strength=1,iterations=1;if(typeof radius!=="function")radius=constant(radius==null?1:+radius);function force(){var i,n=nodes.length,tree,node,xi,yi,ri,ri2;for(var k=0;k<iterations;++k){tree=d3Quadtree.quadtree(nodes,x,y).visitAfter(prepare);for(i=0;i<n;++i){node=nodes[i];ri=radii[node.index],ri2=ri*ri;xi=node.x+node.vx;yi=node.y+node.vy;tree.visit(apply)}}function apply(quad,x0,y0,x1,y1){var data=quad.data,rj=quad.r,r=ri+rj;if(data){if(data.index>node.index){var x=xi-data.x-data.vx,y=yi-data.y-data.vy,l=x*x+y*y;if(l<r*r){if(x===0)x=jiggle(),l+=x*x;if(y===0)y=jiggle(),l+=y*y;l=(r-(l=Math.sqrt(l)))/l*strength;node.vx+=(x*=l)*(r=(rj*=rj)/(ri2+rj));node.vy+=(y*=l)*r;data.vx-=x*(r=1-r);data.vy-=y*r}}return}return x0>xi+r||x1<xi-r||y0>yi+r||y1<yi-r}}function prepare(quad){if(quad.data)return quad.r=radii[quad.data.index];for(var i=quad.r=0;i<4;++i){if(quad[i]&&quad[i].r>quad.r){quad.r=quad[i].r}}}function initialize(){if(!nodes)return;var i,n=nodes.length,node;radii=new Array(n);for(i=0;i<n;++i)node=nodes[i],radii[node.index]=+radius(node,i,nodes)}force.initialize=function(_){nodes=_;initialize()};force.iterations=function(_){return arguments.length?(iterations=+_,force):iterations};force.strength=function(_){return arguments.length?(strength=+_,force):strength};force.radius=function(_){return arguments.length?(radius=typeof _==="function"?_:constant(+_),initialize(),force):radius};return force}function index(d){return d.index}function find(nodeById,nodeId){var node=nodeById.get(nodeId);if(!node)throw new Error("missing: "+nodeId);return node}function link(links){var id=index,strength=defaultStrength,strengths,distance=constant(30),distances,nodes,count,bias,iterations=1;if(links==null)links=[];function defaultStrength(link){return 1/Math.min(count[link.source.index],count[link.target.index])}function force(alpha){for(var k=0,n=links.length;k<iterations;++k){for(var i=0,link,source,target,x,y,l,b;i<n;++i){link=links[i],source=link.source,target=link.target;x=target.x+target.vx-source.x-source.vx||jiggle();y=target.y+target.vy-source.y-source.vy||jiggle();l=Math.sqrt(x*x+y*y);l=(l-distances[i])/l*alpha*strengths[i];x*=l,y*=l;target.vx-=x*(b=bias[i]);target.vy-=y*b;source.vx+=x*(b=1-b);source.vy+=y*b}}}function initialize(){if(!nodes)return;var i,n=nodes.length,m=links.length,nodeById=d3Collection.map(nodes,id),link;for(i=0,count=new Array(n);i<m;++i){link=links[i],link.index=i;if(typeof link.source!=="object")link.source=find(nodeById,link.source);if(typeof link.target!=="object")link.target=find(nodeById,link.target);count[link.source.index]=(count[link.source.index]||0)+1;count[link.target.index]=(count[link.target.index]||0)+1}for(i=0,bias=new Array(m);i<m;++i){link=links[i],bias[i]=count[link.source.index]/(count[link.source.index]+count[link.target.index])}strengths=new Array(m),initializeStrength();distances=new Array(m),initializeDistance()}function initializeStrength(){if(!nodes)return;for(var i=0,n=links.length;i<n;++i){strengths[i]=+strength(links[i],i,links)}}function initializeDistance(){if(!nodes)return;for(var i=0,n=links.length;i<n;++i){distances[i]=+distance(links[i],i,links)}}force.initialize=function(_){nodes=_;initialize()};force.links=function(_){return arguments.length?(links=_,initialize(),force):links};force.id=function(_){return arguments.length?(id=_,force):id};force.iterations=function(_){return arguments.length?(iterations=+_,force):iterations};force.strength=function(_){return arguments.length?(strength=typeof _==="function"?_:constant(+_),initializeStrength(),force):strength};force.distance=function(_){return arguments.length?(distance=typeof _==="function"?_:constant(+_),initializeDistance(),force):distance};return force}function x$1(d){return d.x}function y$1(d){return d.y}var initialRadius=10,initialAngle=Math.PI*(3-Math.sqrt(5));function simulation(nodes){var simulation,alpha=1,alphaMin=.001,alphaDecay=1-Math.pow(alphaMin,1/300),alphaTarget=0,velocityDecay=.6,forces=d3Collection.map(),stepper=d3Timer.timer(step),event=d3Dispatch.dispatch("tick","end");if(nodes==null)nodes=[];function step(){tick();event.call("tick",simulation);if(alpha<alphaMin){stepper.stop();event.call("end",simulation)}}function tick(iterations){var i,n=nodes.length,node;if(iterations===undefined)iterations=1;for(var k=0;k<iterations;++k){alpha+=(alphaTarget-alpha)*alphaDecay;forces.each(function(force){force(alpha)});for(i=0;i<n;++i){node=nodes[i];if(node.fx==null)node.x+=node.vx*=velocityDecay;else node.x=node.fx,node.vx=0;if(node.fy==null)node.y+=node.vy*=velocityDecay;else node.y=node.fy,node.vy=0}}return simulation}function initializeNodes(){for(var i=0,n=nodes.length,node;i<n;++i){node=nodes[i],node.index=i;if(node.fx!=null)node.x=node.fx;if(node.fy!=null)node.y=node.fy;if(isNaN(node.x)||isNaN(node.y)){var radius=initialRadius*Math.sqrt(i),angle=i*initialAngle;node.x=radius*Math.cos(angle);node.y=radius*Math.sin(angle)}if(isNaN(node.vx)||isNaN(node.vy)){node.vx=node.vy=0}}}function initializeForce(force){if(force.initialize)force.initialize(nodes);return force}initializeNodes();return simulation={tick:tick,restart:function(){return stepper.restart(step),simulation},stop:function(){return stepper.stop(),simulation},nodes:function(_){return arguments.length?(nodes=_,initializeNodes(),forces.each(initializeForce),simulation):nodes},alpha:function(_){return arguments.length?(alpha=+_,simulation):alpha},alphaMin:function(_){return arguments.length?(alphaMin=+_,simulation):alphaMin},alphaDecay:function(_){return arguments.length?(alphaDecay=+_,simulation):+alphaDecay},alphaTarget:function(_){return arguments.length?(alphaTarget=+_,simulation):alphaTarget},velocityDecay:function(_){return arguments.length?(velocityDecay=1-_,simulation):1-velocityDecay},force:function(name,_){return arguments.length>1?(_==null?forces.remove(name):forces.set(name,initializeForce(_)),simulation):forces.get(name)},find:function(x,y,radius){var i=0,n=nodes.length,dx,dy,d2,node,closest;if(radius==null)radius=Infinity;else radius*=radius;for(i=0;i<n;++i){node=nodes[i];dx=x-node.x;dy=y-node.y;d2=dx*dx+dy*dy;if(d2<radius)closest=node,radius=d2}return closest},on:function(name,_){return arguments.length>1?(event.on(name,_),simulation):event.on(name)}}}function manyBody(){var nodes,node,alpha,strength=constant(-30),strengths,distanceMin2=1,distanceMax2=Infinity,theta2=.81;function force(_){var i,n=nodes.length,tree=d3Quadtree.quadtree(nodes,x$1,y$1).visitAfter(accumulate);for(alpha=_,i=0;i<n;++i)node=nodes[i],tree.visit(apply)}function initialize(){if(!nodes)return;var i,n=nodes.length,node;strengths=new Array(n);for(i=0;i<n;++i)node=nodes[i],strengths[node.index]=+strength(node,i,nodes)}function accumulate(quad){var strength=0,q,c,weight=0,x,y,i;
// For internal nodes, accumulate forces from child quadrants.
if(quad.length){for(x=y=i=0;i<4;++i){if((q=quad[i])&&(c=Math.abs(q.value))){strength+=q.value,weight+=c,x+=c*q.x,y+=c*q.y}}quad.x=x/weight;quad.y=y/weight}
// For leaf nodes, accumulate forces from coincident quadrants.
else{q=quad;q.x=q.data.x;q.y=q.data.y;do{strength+=strengths[q.data.index]}while(q=q.next)}quad.value=strength}function apply(quad,x1,_,x2){if(!quad.value)return true;var x=quad.x-node.x,y=quad.y-node.y,w=x2-x1,l=x*x+y*y;
// Apply the Barnes-Hut approximation if possible.
// Limit forces for very close nodes; randomize direction if coincident.
if(w*w/theta2<l){if(l<distanceMax2){if(x===0)x=jiggle(),l+=x*x;if(y===0)y=jiggle(),l+=y*y;if(l<distanceMin2)l=Math.sqrt(distanceMin2*l);node.vx+=x*quad.value*alpha/l;node.vy+=y*quad.value*alpha/l}return true}
// Otherwise, process points directly.
else if(quad.length||l>=distanceMax2)return;
// Limit forces for very close nodes; randomize direction if coincident.
if(quad.data!==node||quad.next){if(x===0)x=jiggle(),l+=x*x;if(y===0)y=jiggle(),l+=y*y;if(l<distanceMin2)l=Math.sqrt(distanceMin2*l)}do{if(quad.data!==node){w=strengths[quad.data.index]*alpha/l;node.vx+=x*w;node.vy+=y*w}}while(quad=quad.next)}force.initialize=function(_){nodes=_;initialize()};force.strength=function(_){return arguments.length?(strength=typeof _==="function"?_:constant(+_),initialize(),force):strength};force.distanceMin=function(_){return arguments.length?(distanceMin2=_*_,force):Math.sqrt(distanceMin2)};force.distanceMax=function(_){return arguments.length?(distanceMax2=_*_,force):Math.sqrt(distanceMax2)};force.theta=function(_){return arguments.length?(theta2=_*_,force):Math.sqrt(theta2)};return force}function radial(radius,x,y){var nodes,strength=constant(.1),strengths,radiuses;if(typeof radius!=="function")radius=constant(+radius);if(x==null)x=0;if(y==null)y=0;function force(alpha){for(var i=0,n=nodes.length;i<n;++i){var node=nodes[i],dx=node.x-x||1e-6,dy=node.y-y||1e-6,r=Math.sqrt(dx*dx+dy*dy),k=(radiuses[i]-r)*strengths[i]*alpha/r;node.vx+=dx*k;node.vy+=dy*k}}function initialize(){if(!nodes)return;var i,n=nodes.length;strengths=new Array(n);radiuses=new Array(n);for(i=0;i<n;++i){radiuses[i]=+radius(nodes[i],i,nodes);strengths[i]=isNaN(radiuses[i])?0:+strength(nodes[i],i,nodes)}}force.initialize=function(_){nodes=_,initialize()};force.strength=function(_){return arguments.length?(strength=typeof _==="function"?_:constant(+_),initialize(),force):strength};force.radius=function(_){return arguments.length?(radius=typeof _==="function"?_:constant(+_),initialize(),force):radius};force.x=function(_){return arguments.length?(x=+_,force):x};force.y=function(_){return arguments.length?(y=+_,force):y};return force}function x$2(x){var strength=constant(.1),nodes,strengths,xz;if(typeof x!=="function")x=constant(x==null?0:+x);function force(alpha){for(var i=0,n=nodes.length,node;i<n;++i){node=nodes[i],node.vx+=(xz[i]-node.x)*strengths[i]*alpha}}function initialize(){if(!nodes)return;var i,n=nodes.length;strengths=new Array(n);xz=new Array(n);for(i=0;i<n;++i){strengths[i]=isNaN(xz[i]=+x(nodes[i],i,nodes))?0:+strength(nodes[i],i,nodes)}}force.initialize=function(_){nodes=_;initialize()};force.strength=function(_){return arguments.length?(strength=typeof _==="function"?_:constant(+_),initialize(),force):strength};force.x=function(_){return arguments.length?(x=typeof _==="function"?_:constant(+_),initialize(),force):x};return force}function y$2(y){var strength=constant(.1),nodes,strengths,yz;if(typeof y!=="function")y=constant(y==null?0:+y);function force(alpha){for(var i=0,n=nodes.length,node;i<n;++i){node=nodes[i],node.vy+=(yz[i]-node.y)*strengths[i]*alpha}}function initialize(){if(!nodes)return;var i,n=nodes.length;strengths=new Array(n);yz=new Array(n);for(i=0;i<n;++i){strengths[i]=isNaN(yz[i]=+y(nodes[i],i,nodes))?0:+strength(nodes[i],i,nodes)}}force.initialize=function(_){nodes=_;initialize()};force.strength=function(_){return arguments.length?(strength=typeof _==="function"?_:constant(+_),initialize(),force):strength};force.y=function(_){return arguments.length?(y=typeof _==="function"?_:constant(+_),initialize(),force):y};return force}exports.forceCenter=center;exports.forceCollide=collide;exports.forceLink=link;exports.forceManyBody=manyBody;exports.forceRadial=radial;exports.forceSimulation=simulation;exports.forceX=x$2;exports.forceY=y$2;Object.defineProperty(exports,"__esModule",{value:true})})},{"d3-collection":33,"d3-dispatch":36,"d3-quadtree":48,"d3-timer":56}],42:[function(require,module,exports){
// https://d3js.org/d3-format/ v1.4.2 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):(global=global||self,factory(global.d3=global.d3||{}))})(this,function(exports){"use strict";
// Computes the decimal coefficient and exponent of the specified number x with
// significant digits p, where x is positive and p is in [1, 21] or undefined.
// For example, formatDecimal(1.23) returns ["123", 0].
function formatDecimal(x,p){if((i=(x=p?x.toExponential(p-1):x.toExponential()).indexOf("e"))<0)return null;// NaN, ±Infinity
var i,coefficient=x.slice(0,i);
// The string returned by toExponential either has the form \d\.\d+e[-+]\d+
// (e.g., 1.2e+3) or the form \de[-+]\d+ (e.g., 1e+3).
return[coefficient.length>1?coefficient[0]+coefficient.slice(2):coefficient,+x.slice(i+1)]}function exponent(x){return x=formatDecimal(Math.abs(x)),x?x[1]:NaN}function formatGroup(grouping,thousands){return function(value,width){var i=value.length,t=[],j=0,g=grouping[0],length=0;while(i>0&&g>0){if(length+g+1>width)g=Math.max(1,width-length);t.push(value.substring(i-=g,i+g));if((length+=g+1)>width)break;g=grouping[j=(j+1)%grouping.length]}return t.reverse().join(thousands)}}function formatNumerals(numerals){return function(value){return value.replace(/[0-9]/g,function(i){return numerals[+i]})}}
// [[fill]align][sign][symbol][0][width][,][.precision][~][type]
var re=/^(?:(.)?([<>=^]))?([+\-( ])?([$#])?(0)?(\d+)?(,)?(\.\d+)?(~)?([a-z%])?$/i;function formatSpecifier(specifier){if(!(match=re.exec(specifier)))throw new Error("invalid format: "+specifier);var match;return new FormatSpecifier({fill:match[1],align:match[2],sign:match[3],symbol:match[4],zero:match[5],width:match[6],comma:match[7],precision:match[8]&&match[8].slice(1),trim:match[9],type:match[10]})}formatSpecifier.prototype=FormatSpecifier.prototype;// instanceof
function FormatSpecifier(specifier){this.fill=specifier.fill===undefined?" ":specifier.fill+"";this.align=specifier.align===undefined?">":specifier.align+"";this.sign=specifier.sign===undefined?"-":specifier.sign+"";this.symbol=specifier.symbol===undefined?"":specifier.symbol+"";this.zero=!!specifier.zero;this.width=specifier.width===undefined?undefined:+specifier.width;this.comma=!!specifier.comma;this.precision=specifier.precision===undefined?undefined:+specifier.precision;this.trim=!!specifier.trim;this.type=specifier.type===undefined?"":specifier.type+""}FormatSpecifier.prototype.toString=function(){return this.fill+this.align+this.sign+this.symbol+(this.zero?"0":"")+(this.width===undefined?"":Math.max(1,this.width|0))+(this.comma?",":"")+(this.precision===undefined?"":"."+Math.max(0,this.precision|0))+(this.trim?"~":"")+this.type};
// Trims insignificant zeros, e.g., replaces 1.2000k with 1.2k.
function formatTrim(s){out:for(var n=s.length,i=1,i0=-1,i1;i<n;++i){switch(s[i]){case".":i0=i1=i;break;case"0":if(i0===0)i0=i;i1=i;break;default:if(i0>0){if(!+s[i])break out;i0=0}break}}return i0>0?s.slice(0,i0)+s.slice(i1+1):s}var prefixExponent;function formatPrefixAuto(x,p){var d=formatDecimal(x,p);if(!d)return x+"";var coefficient=d[0],exponent=d[1],i=exponent-(prefixExponent=Math.max(-8,Math.min(8,Math.floor(exponent/3)))*3)+1,n=coefficient.length;return i===n?coefficient:i>n?coefficient+new Array(i-n+1).join("0"):i>0?coefficient.slice(0,i)+"."+coefficient.slice(i):"0."+new Array(1-i).join("0")+formatDecimal(x,Math.max(0,p+i-1))[0];// less than 1y!
}function formatRounded(x,p){var d=formatDecimal(x,p);if(!d)return x+"";var coefficient=d[0],exponent=d[1];return exponent<0?"0."+new Array(-exponent).join("0")+coefficient:coefficient.length>exponent+1?coefficient.slice(0,exponent+1)+"."+coefficient.slice(exponent+1):coefficient+new Array(exponent-coefficient.length+2).join("0")}var formatTypes={"%":function(x,p){return(x*100).toFixed(p)},b:function(x){return Math.round(x).toString(2)},c:function(x){return x+""},d:function(x){return Math.round(x).toString(10)},e:function(x,p){return x.toExponential(p)},f:function(x,p){return x.toFixed(p)},g:function(x,p){return x.toPrecision(p)},o:function(x){return Math.round(x).toString(8)},p:function(x,p){return formatRounded(x*100,p)},r:formatRounded,s:formatPrefixAuto,X:function(x){return Math.round(x).toString(16).toUpperCase()},x:function(x){return Math.round(x).toString(16)}};function identity(x){return x}var map=Array.prototype.map,prefixes=["y","z","a","f","p","n","µ","m","","k","M","G","T","P","E","Z","Y"];function formatLocale(locale){var group=locale.grouping===undefined||locale.thousands===undefined?identity:formatGroup(map.call(locale.grouping,Number),locale.thousands+""),currencyPrefix=locale.currency===undefined?"":locale.currency[0]+"",currencySuffix=locale.currency===undefined?"":locale.currency[1]+"",decimal=locale.decimal===undefined?".":locale.decimal+"",numerals=locale.numerals===undefined?identity:formatNumerals(map.call(locale.numerals,String)),percent=locale.percent===undefined?"%":locale.percent+"",minus=locale.minus===undefined?"-":locale.minus+"",nan=locale.nan===undefined?"NaN":locale.nan+"";function newFormat(specifier){specifier=formatSpecifier(specifier);var fill=specifier.fill,align=specifier.align,sign=specifier.sign,symbol=specifier.symbol,zero=specifier.zero,width=specifier.width,comma=specifier.comma,precision=specifier.precision,trim=specifier.trim,type=specifier.type;
// The "n" type is an alias for ",g".
if(type==="n")comma=true,type="g";
// The "" type, and any invalid type, is an alias for ".12~g".
else if(!formatTypes[type])precision===undefined&&(precision=12),trim=true,type="g";
// If zero fill is specified, padding goes after sign and before digits.
if(zero||fill==="0"&&align==="=")zero=true,fill="0",align="=";
// Compute the prefix and suffix.
// For SI-prefix, the suffix is lazily computed.
var prefix=symbol==="$"?currencyPrefix:symbol==="#"&&/[boxX]/.test(type)?"0"+type.toLowerCase():"",suffix=symbol==="$"?currencySuffix:/[%p]/.test(type)?percent:"";
// What format function should we use?
// Is this an integer type?
// Can this type generate exponential notation?
var formatType=formatTypes[type],maybeSuffix=/[defgprs%]/.test(type);
// Set the default precision if not specified,
// or clamp the specified precision to the supported range.
// For significant precision, it must be in [1, 21].
// For fixed precision, it must be in [0, 20].
precision=precision===undefined?6:/[gprs]/.test(type)?Math.max(1,Math.min(21,precision)):Math.max(0,Math.min(20,precision));function format(value){var valuePrefix=prefix,valueSuffix=suffix,i,n,c;if(type==="c"){valueSuffix=formatType(value)+valueSuffix;value=""}else{value=+value;
// Perform the initial formatting.
var valueNegative=value<0;value=isNaN(value)?nan:formatType(Math.abs(value),precision);
// Trim insignificant zeros.
if(trim)value=formatTrim(value);
// If a negative value rounds to zero during formatting, treat as positive.
if(valueNegative&&+value===0)valueNegative=false;
// Compute the prefix and suffix.
valuePrefix=(valueNegative?sign==="("?sign:minus:sign==="-"||sign==="("?"":sign)+valuePrefix;valueSuffix=(type==="s"?prefixes[8+prefixExponent/3]:"")+valueSuffix+(valueNegative&&sign==="("?")":"");
// Break the formatted value into the integer “value” part that can be
// grouped, and fractional or exponential “suffix” part that is not.
if(maybeSuffix){i=-1,n=value.length;while(++i<n){if(c=value.charCodeAt(i),48>c||c>57){valueSuffix=(c===46?decimal+value.slice(i+1):value.slice(i))+valueSuffix;value=value.slice(0,i);break}}}}
// If the fill character is not "0", grouping is applied before padding.
if(comma&&!zero)value=group(value,Infinity);
// Compute the padding.
var length=valuePrefix.length+value.length+valueSuffix.length,padding=length<width?new Array(width-length+1).join(fill):"";
// If the fill character is "0", grouping is applied after padding.
if(comma&&zero)value=group(padding+value,padding.length?width-valueSuffix.length:Infinity),padding="";
// Reconstruct the final output based on the desired alignment.
switch(align){case"<":value=valuePrefix+value+valueSuffix+padding;break;case"=":value=valuePrefix+padding+value+valueSuffix;break;case"^":value=padding.slice(0,length=padding.length>>1)+valuePrefix+value+valueSuffix+padding.slice(length);break;default:value=padding+valuePrefix+value+valueSuffix;break}return numerals(value)}format.toString=function(){return specifier+""};return format}function formatPrefix(specifier,value){var f=newFormat((specifier=formatSpecifier(specifier),specifier.type="f",specifier)),e=Math.max(-8,Math.min(8,Math.floor(exponent(value)/3)))*3,k=Math.pow(10,-e),prefix=prefixes[8+e/3];return function(value){return f(k*value)+prefix}}return{format:newFormat,formatPrefix:formatPrefix}}var locale;defaultLocale({decimal:".",thousands:",",grouping:[3],currency:["$",""],minus:"-"});function defaultLocale(definition){locale=formatLocale(definition);exports.format=locale.format;exports.formatPrefix=locale.formatPrefix;return locale}function precisionFixed(step){return Math.max(0,-exponent(Math.abs(step)))}function precisionPrefix(step,value){return Math.max(0,Math.max(-8,Math.min(8,Math.floor(exponent(value)/3)))*3-exponent(Math.abs(step)))}function precisionRound(step,max){step=Math.abs(step),max=Math.abs(max)-step;return Math.max(0,exponent(max)-exponent(step))+1}exports.FormatSpecifier=FormatSpecifier;exports.formatDefaultLocale=defaultLocale;exports.formatLocale=formatLocale;exports.formatSpecifier=formatSpecifier;exports.precisionFixed=precisionFixed;exports.precisionPrefix=precisionPrefix;exports.precisionRound=precisionRound;Object.defineProperty(exports,"__esModule",{value:true})})},{}],43:[function(require,module,exports){
// https://d3js.org/d3-geo/ v1.11.9 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports,require("d3-array")):typeof define==="function"&&define.amd?define(["exports","d3-array"],factory):(global=global||self,factory(global.d3=global.d3||{},global.d3))})(this,function(exports,d3Array){"use strict";
// Adds floating point numbers with twice the normal precision.
// Reference: J. R. Shewchuk, Adaptive Precision Floating-Point Arithmetic and
// Fast Robust Geometric Predicates, Discrete & Computational Geometry 18(3)
// 305–363 (1997).
// Code adapted from GeographicLib by Charles F. F. Karney,
// http://geographiclib.sourceforge.net/
function adder(){return new Adder}function Adder(){this.reset()}Adder.prototype={constructor:Adder,reset:function(){this.s=// rounded value
this.t=0;// exact error
},add:function(y){add(temp,y,this.t);add(this,temp.s,this.s);if(this.s)this.t+=temp.t;else this.s=temp.t},valueOf:function(){return this.s}};var temp=new Adder;function add(adder,a,b){var x=adder.s=a+b,bv=x-a,av=x-bv;adder.t=a-av+(b-bv)}var epsilon=1e-6;var epsilon2=1e-12;var pi=Math.PI;var halfPi=pi/2;var quarterPi=pi/4;var tau=pi*2;var degrees=180/pi;var radians=pi/180;var abs=Math.abs;var atan=Math.atan;var atan2=Math.atan2;var cos=Math.cos;var ceil=Math.ceil;var exp=Math.exp;var log=Math.log;var pow=Math.pow;var sin=Math.sin;var sign=Math.sign||function(x){return x>0?1:x<0?-1:0};var sqrt=Math.sqrt;var tan=Math.tan;function acos(x){return x>1?0:x<-1?pi:Math.acos(x)}function asin(x){return x>1?halfPi:x<-1?-halfPi:Math.asin(x)}function haversin(x){return(x=sin(x/2))*x}function noop(){}function streamGeometry(geometry,stream){if(geometry&&streamGeometryType.hasOwnProperty(geometry.type)){streamGeometryType[geometry.type](geometry,stream)}}var streamObjectType={Feature:function(object,stream){streamGeometry(object.geometry,stream)},FeatureCollection:function(object,stream){var features=object.features,i=-1,n=features.length;while(++i<n)streamGeometry(features[i].geometry,stream)}};var streamGeometryType={Sphere:function(object,stream){stream.sphere()},Point:function(object,stream){object=object.coordinates;stream.point(object[0],object[1],object[2])},MultiPoint:function(object,stream){var coordinates=object.coordinates,i=-1,n=coordinates.length;while(++i<n)object=coordinates[i],stream.point(object[0],object[1],object[2])},LineString:function(object,stream){streamLine(object.coordinates,stream,0)},MultiLineString:function(object,stream){var coordinates=object.coordinates,i=-1,n=coordinates.length;while(++i<n)streamLine(coordinates[i],stream,0)},Polygon:function(object,stream){streamPolygon(object.coordinates,stream)},MultiPolygon:function(object,stream){var coordinates=object.coordinates,i=-1,n=coordinates.length;while(++i<n)streamPolygon(coordinates[i],stream)},GeometryCollection:function(object,stream){var geometries=object.geometries,i=-1,n=geometries.length;while(++i<n)streamGeometry(geometries[i],stream)}};function streamLine(coordinates,stream,closed){var i=-1,n=coordinates.length-closed,coordinate;stream.lineStart();while(++i<n)coordinate=coordinates[i],stream.point(coordinate[0],coordinate[1],coordinate[2]);stream.lineEnd()}function streamPolygon(coordinates,stream){var i=-1,n=coordinates.length;stream.polygonStart();while(++i<n)streamLine(coordinates[i],stream,1);stream.polygonEnd()}function geoStream(object,stream){if(object&&streamObjectType.hasOwnProperty(object.type)){streamObjectType[object.type](object,stream)}else{streamGeometry(object,stream)}}var areaRingSum=adder();var areaSum=adder(),lambda00,phi00,lambda0,cosPhi0,sinPhi0;var areaStream={point:noop,lineStart:noop,lineEnd:noop,polygonStart:function(){areaRingSum.reset();areaStream.lineStart=areaRingStart;areaStream.lineEnd=areaRingEnd},polygonEnd:function(){var areaRing=+areaRingSum;areaSum.add(areaRing<0?tau+areaRing:areaRing);this.lineStart=this.lineEnd=this.point=noop},sphere:function(){areaSum.add(tau)}};function areaRingStart(){areaStream.point=areaPointFirst}function areaRingEnd(){areaPoint(lambda00,phi00)}function areaPointFirst(lambda,phi){areaStream.point=areaPoint;lambda00=lambda,phi00=phi;lambda*=radians,phi*=radians;lambda0=lambda,cosPhi0=cos(phi=phi/2+quarterPi),sinPhi0=sin(phi)}function areaPoint(lambda,phi){lambda*=radians,phi*=radians;phi=phi/2+quarterPi;// half the angular distance from south pole
// Spherical excess E for a spherical triangle with vertices: south pole,
// previous point, current point.  Uses a formula derived from Cagnoli’s
// theorem.  See Todhunter, Spherical Trig. (1871), Sec. 103, Eq. (2).
var dLambda=lambda-lambda0,sdLambda=dLambda>=0?1:-1,adLambda=sdLambda*dLambda,cosPhi=cos(phi),sinPhi=sin(phi),k=sinPhi0*sinPhi,u=cosPhi0*cosPhi+k*cos(adLambda),v=k*sdLambda*sin(adLambda);areaRingSum.add(atan2(v,u));
// Advance the previous points.
lambda0=lambda,cosPhi0=cosPhi,sinPhi0=sinPhi}function area(object){areaSum.reset();geoStream(object,areaStream);return areaSum*2}function spherical(cartesian){return[atan2(cartesian[1],cartesian[0]),asin(cartesian[2])]}function cartesian(spherical){var lambda=spherical[0],phi=spherical[1],cosPhi=cos(phi);return[cosPhi*cos(lambda),cosPhi*sin(lambda),sin(phi)]}function cartesianDot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}function cartesianCross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
// TODO return a
function cartesianAddInPlace(a,b){a[0]+=b[0],a[1]+=b[1],a[2]+=b[2]}function cartesianScale(vector,k){return[vector[0]*k,vector[1]*k,vector[2]*k]}
// TODO return d
function cartesianNormalizeInPlace(d){var l=sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);d[0]/=l,d[1]/=l,d[2]/=l}var lambda0$1,phi0,lambda1,phi1,// bounds
lambda2,// previous lambda-coordinate
lambda00$1,phi00$1,// first point
p0,// previous 3D point
deltaSum=adder(),ranges,range;var boundsStream={point:boundsPoint,lineStart:boundsLineStart,lineEnd:boundsLineEnd,polygonStart:function(){boundsStream.point=boundsRingPoint;boundsStream.lineStart=boundsRingStart;boundsStream.lineEnd=boundsRingEnd;deltaSum.reset();areaStream.polygonStart()},polygonEnd:function(){areaStream.polygonEnd();boundsStream.point=boundsPoint;boundsStream.lineStart=boundsLineStart;boundsStream.lineEnd=boundsLineEnd;if(areaRingSum<0)lambda0$1=-(lambda1=180),phi0=-(phi1=90);else if(deltaSum>epsilon)phi1=90;else if(deltaSum<-epsilon)phi0=-90;range[0]=lambda0$1,range[1]=lambda1},sphere:function(){lambda0$1=-(lambda1=180),phi0=-(phi1=90)}};function boundsPoint(lambda,phi){ranges.push(range=[lambda0$1=lambda,lambda1=lambda]);if(phi<phi0)phi0=phi;if(phi>phi1)phi1=phi}function linePoint(lambda,phi){var p=cartesian([lambda*radians,phi*radians]);if(p0){var normal=cartesianCross(p0,p),equatorial=[normal[1],-normal[0],0],inflection=cartesianCross(equatorial,normal);cartesianNormalizeInPlace(inflection);inflection=spherical(inflection);var delta=lambda-lambda2,sign=delta>0?1:-1,lambdai=inflection[0]*degrees*sign,phii,antimeridian=abs(delta)>180;if(antimeridian^(sign*lambda2<lambdai&&lambdai<sign*lambda)){phii=inflection[1]*degrees;if(phii>phi1)phi1=phii}else if(lambdai=(lambdai+360)%360-180,antimeridian^(sign*lambda2<lambdai&&lambdai<sign*lambda)){phii=-inflection[1]*degrees;if(phii<phi0)phi0=phii}else{if(phi<phi0)phi0=phi;if(phi>phi1)phi1=phi}if(antimeridian){if(lambda<lambda2){if(angle(lambda0$1,lambda)>angle(lambda0$1,lambda1))lambda1=lambda}else{if(angle(lambda,lambda1)>angle(lambda0$1,lambda1))lambda0$1=lambda}}else{if(lambda1>=lambda0$1){if(lambda<lambda0$1)lambda0$1=lambda;if(lambda>lambda1)lambda1=lambda}else{if(lambda>lambda2){if(angle(lambda0$1,lambda)>angle(lambda0$1,lambda1))lambda1=lambda}else{if(angle(lambda,lambda1)>angle(lambda0$1,lambda1))lambda0$1=lambda}}}}else{ranges.push(range=[lambda0$1=lambda,lambda1=lambda])}if(phi<phi0)phi0=phi;if(phi>phi1)phi1=phi;p0=p,lambda2=lambda}function boundsLineStart(){boundsStream.point=linePoint}function boundsLineEnd(){range[0]=lambda0$1,range[1]=lambda1;boundsStream.point=boundsPoint;p0=null}function boundsRingPoint(lambda,phi){if(p0){var delta=lambda-lambda2;deltaSum.add(abs(delta)>180?delta+(delta>0?360:-360):delta)}else{lambda00$1=lambda,phi00$1=phi}areaStream.point(lambda,phi);linePoint(lambda,phi)}function boundsRingStart(){areaStream.lineStart()}function boundsRingEnd(){boundsRingPoint(lambda00$1,phi00$1);areaStream.lineEnd();if(abs(deltaSum)>epsilon)lambda0$1=-(lambda1=180);range[0]=lambda0$1,range[1]=lambda1;p0=null}
// Finds the left-right distance between two longitudes.
// This is almost the same as (lambda1 - lambda0 + 360°) % 360°, except that we want
// the distance between ±180° to be 360°.
function angle(lambda0,lambda1){return(lambda1-=lambda0)<0?lambda1+360:lambda1}function rangeCompare(a,b){return a[0]-b[0]}function rangeContains(range,x){return range[0]<=range[1]?range[0]<=x&&x<=range[1]:x<range[0]||range[1]<x}function bounds(feature){var i,n,a,b,merged,deltaMax,delta;phi1=lambda1=-(lambda0$1=phi0=Infinity);ranges=[];geoStream(feature,boundsStream);
// First, sort ranges by their minimum longitudes.
if(n=ranges.length){ranges.sort(rangeCompare);
// Then, merge any ranges that overlap.
for(i=1,a=ranges[0],merged=[a];i<n;++i){b=ranges[i];if(rangeContains(a,b[0])||rangeContains(a,b[1])){if(angle(a[0],b[1])>angle(a[0],a[1]))a[1]=b[1];if(angle(b[0],a[1])>angle(a[0],a[1]))a[0]=b[0]}else{merged.push(a=b)}}
// Finally, find the largest gap between the merged ranges.
// The final bounding box will be the inverse of this gap.
for(deltaMax=-Infinity,n=merged.length-1,i=0,a=merged[n];i<=n;a=b,++i){b=merged[i];if((delta=angle(a[1],b[0]))>deltaMax)deltaMax=delta,lambda0$1=b[0],lambda1=a[1]}}ranges=range=null;return lambda0$1===Infinity||phi0===Infinity?[[NaN,NaN],[NaN,NaN]]:[[lambda0$1,phi0],[lambda1,phi1]]}var W0,W1,X0,Y0,Z0,X1,Y1,Z1,X2,Y2,Z2,lambda00$2,phi00$2,// first point
x0,y0,z0;// previous point
var centroidStream={sphere:noop,point:centroidPoint,lineStart:centroidLineStart,lineEnd:centroidLineEnd,polygonStart:function(){centroidStream.lineStart=centroidRingStart;centroidStream.lineEnd=centroidRingEnd},polygonEnd:function(){centroidStream.lineStart=centroidLineStart;centroidStream.lineEnd=centroidLineEnd}};
// Arithmetic mean of Cartesian vectors.
function centroidPoint(lambda,phi){lambda*=radians,phi*=radians;var cosPhi=cos(phi);centroidPointCartesian(cosPhi*cos(lambda),cosPhi*sin(lambda),sin(phi))}function centroidPointCartesian(x,y,z){++W0;X0+=(x-X0)/W0;Y0+=(y-Y0)/W0;Z0+=(z-Z0)/W0}function centroidLineStart(){centroidStream.point=centroidLinePointFirst}function centroidLinePointFirst(lambda,phi){lambda*=radians,phi*=radians;var cosPhi=cos(phi);x0=cosPhi*cos(lambda);y0=cosPhi*sin(lambda);z0=sin(phi);centroidStream.point=centroidLinePoint;centroidPointCartesian(x0,y0,z0)}function centroidLinePoint(lambda,phi){lambda*=radians,phi*=radians;var cosPhi=cos(phi),x=cosPhi*cos(lambda),y=cosPhi*sin(lambda),z=sin(phi),w=atan2(sqrt((w=y0*z-z0*y)*w+(w=z0*x-x0*z)*w+(w=x0*y-y0*x)*w),x0*x+y0*y+z0*z);W1+=w;X1+=w*(x0+(x0=x));Y1+=w*(y0+(y0=y));Z1+=w*(z0+(z0=z));centroidPointCartesian(x0,y0,z0)}function centroidLineEnd(){centroidStream.point=centroidPoint}
// See J. E. Brock, The Inertia Tensor for a Spherical Triangle,
// J. Applied Mechanics 42, 239 (1975).
function centroidRingStart(){centroidStream.point=centroidRingPointFirst}function centroidRingEnd(){centroidRingPoint(lambda00$2,phi00$2);centroidStream.point=centroidPoint}function centroidRingPointFirst(lambda,phi){lambda00$2=lambda,phi00$2=phi;lambda*=radians,phi*=radians;centroidStream.point=centroidRingPoint;var cosPhi=cos(phi);x0=cosPhi*cos(lambda);y0=cosPhi*sin(lambda);z0=sin(phi);centroidPointCartesian(x0,y0,z0)}function centroidRingPoint(lambda,phi){lambda*=radians,phi*=radians;var cosPhi=cos(phi),x=cosPhi*cos(lambda),y=cosPhi*sin(lambda),z=sin(phi),cx=y0*z-z0*y,cy=z0*x-x0*z,cz=x0*y-y0*x,m=sqrt(cx*cx+cy*cy+cz*cz),w=asin(m),// line weight = angle
v=m&&-w/m;// area weight multiplier
X2+=v*cx;Y2+=v*cy;Z2+=v*cz;W1+=w;X1+=w*(x0+(x0=x));Y1+=w*(y0+(y0=y));Z1+=w*(z0+(z0=z));centroidPointCartesian(x0,y0,z0)}function centroid(object){W0=W1=X0=Y0=Z0=X1=Y1=Z1=X2=Y2=Z2=0;geoStream(object,centroidStream);var x=X2,y=Y2,z=Z2,m=x*x+y*y+z*z;
// If the area-weighted ccentroid is undefined, fall back to length-weighted ccentroid.
if(m<epsilon2){x=X1,y=Y1,z=Z1;
// If the feature has zero length, fall back to arithmetic mean of point vectors.
if(W1<epsilon)x=X0,y=Y0,z=Z0;m=x*x+y*y+z*z;
// If the feature still has an undefined ccentroid, then return.
if(m<epsilon2)return[NaN,NaN]}return[atan2(y,x)*degrees,asin(z/sqrt(m))*degrees]}function constant(x){return function(){return x}}function compose(a,b){function compose(x,y){return x=a(x,y),b(x[0],x[1])}if(a.invert&&b.invert)compose.invert=function(x,y){return x=b.invert(x,y),x&&a.invert(x[0],x[1])};return compose}function rotationIdentity(lambda,phi){return[abs(lambda)>pi?lambda+Math.round(-lambda/tau)*tau:lambda,phi]}rotationIdentity.invert=rotationIdentity;function rotateRadians(deltaLambda,deltaPhi,deltaGamma){return(deltaLambda%=tau)?deltaPhi||deltaGamma?compose(rotationLambda(deltaLambda),rotationPhiGamma(deltaPhi,deltaGamma)):rotationLambda(deltaLambda):deltaPhi||deltaGamma?rotationPhiGamma(deltaPhi,deltaGamma):rotationIdentity}function forwardRotationLambda(deltaLambda){return function(lambda,phi){return lambda+=deltaLambda,[lambda>pi?lambda-tau:lambda<-pi?lambda+tau:lambda,phi]}}function rotationLambda(deltaLambda){var rotation=forwardRotationLambda(deltaLambda);rotation.invert=forwardRotationLambda(-deltaLambda);return rotation}function rotationPhiGamma(deltaPhi,deltaGamma){var cosDeltaPhi=cos(deltaPhi),sinDeltaPhi=sin(deltaPhi),cosDeltaGamma=cos(deltaGamma),sinDeltaGamma=sin(deltaGamma);function rotation(lambda,phi){var cosPhi=cos(phi),x=cos(lambda)*cosPhi,y=sin(lambda)*cosPhi,z=sin(phi),k=z*cosDeltaPhi+x*sinDeltaPhi;return[atan2(y*cosDeltaGamma-k*sinDeltaGamma,x*cosDeltaPhi-z*sinDeltaPhi),asin(k*cosDeltaGamma+y*sinDeltaGamma)]}rotation.invert=function(lambda,phi){var cosPhi=cos(phi),x=cos(lambda)*cosPhi,y=sin(lambda)*cosPhi,z=sin(phi),k=z*cosDeltaGamma-y*sinDeltaGamma;return[atan2(y*cosDeltaGamma+z*sinDeltaGamma,x*cosDeltaPhi+k*sinDeltaPhi),asin(k*cosDeltaPhi-x*sinDeltaPhi)]};return rotation}function rotation(rotate){rotate=rotateRadians(rotate[0]*radians,rotate[1]*radians,rotate.length>2?rotate[2]*radians:0);function forward(coordinates){coordinates=rotate(coordinates[0]*radians,coordinates[1]*radians);return coordinates[0]*=degrees,coordinates[1]*=degrees,coordinates}forward.invert=function(coordinates){coordinates=rotate.invert(coordinates[0]*radians,coordinates[1]*radians);return coordinates[0]*=degrees,coordinates[1]*=degrees,coordinates};return forward}
// Generates a circle centered at [0°, 0°], with a given radius and precision.
function circleStream(stream,radius,delta,direction,t0,t1){if(!delta)return;var cosRadius=cos(radius),sinRadius=sin(radius),step=direction*delta;if(t0==null){t0=radius+direction*tau;t1=radius-step/2}else{t0=circleRadius(cosRadius,t0);t1=circleRadius(cosRadius,t1);if(direction>0?t0<t1:t0>t1)t0+=direction*tau}for(var point,t=t0;direction>0?t>t1:t<t1;t-=step){point=spherical([cosRadius,-sinRadius*cos(t),-sinRadius*sin(t)]);stream.point(point[0],point[1])}}
// Returns the signed angle of a cartesian point relative to [cosRadius, 0, 0].
function circleRadius(cosRadius,point){point=cartesian(point),point[0]-=cosRadius;cartesianNormalizeInPlace(point);var radius=acos(-point[1]);return((-point[2]<0?-radius:radius)+tau-epsilon)%tau}function circle(){var center=constant([0,0]),radius=constant(90),precision=constant(6),ring,rotate,stream={point:point};function point(x,y){ring.push(x=rotate(x,y));x[0]*=degrees,x[1]*=degrees}function circle(){var c=center.apply(this,arguments),r=radius.apply(this,arguments)*radians,p=precision.apply(this,arguments)*radians;ring=[];rotate=rotateRadians(-c[0]*radians,-c[1]*radians,0).invert;circleStream(stream,r,p,1);c={type:"Polygon",coordinates:[ring]};ring=rotate=null;return c}circle.center=function(_){return arguments.length?(center=typeof _==="function"?_:constant([+_[0],+_[1]]),circle):center};circle.radius=function(_){return arguments.length?(radius=typeof _==="function"?_:constant(+_),circle):radius};circle.precision=function(_){return arguments.length?(precision=typeof _==="function"?_:constant(+_),circle):precision};return circle}function clipBuffer(){var lines=[],line;return{point:function(x,y){line.push([x,y])},lineStart:function(){lines.push(line=[])},lineEnd:noop,rejoin:function(){if(lines.length>1)lines.push(lines.pop().concat(lines.shift()))},result:function(){var result=lines;lines=[];line=null;return result}}}function pointEqual(a,b){return abs(a[0]-b[0])<epsilon&&abs(a[1]-b[1])<epsilon}function Intersection(point,points,other,entry){this.x=point;this.z=points;this.o=other;// another intersection
this.e=entry;// is an entry?
this.v=false;// visited
this.n=this.p=null;// next & previous
}
// A generalized polygon clipping algorithm: given a polygon that has been cut
// into its visible line segments, and rejoins the segments by interpolating
// along the clip edge.
function clipRejoin(segments,compareIntersection,startInside,interpolate,stream){var subject=[],clip=[],i,n;segments.forEach(function(segment){if((n=segment.length-1)<=0)return;var n,p0=segment[0],p1=segment[n],x;
// If the first and last points of a segment are coincident, then treat as a
// closed ring. TODO if all rings are closed, then the winding order of the
// exterior ring should be checked.
if(pointEqual(p0,p1)){stream.lineStart();for(i=0;i<n;++i)stream.point((p0=segment[i])[0],p0[1]);stream.lineEnd();return}subject.push(x=new Intersection(p0,segment,null,true));clip.push(x.o=new Intersection(p0,null,x,false));subject.push(x=new Intersection(p1,segment,null,false));clip.push(x.o=new Intersection(p1,null,x,true))});if(!subject.length)return;clip.sort(compareIntersection);link(subject);link(clip);for(i=0,n=clip.length;i<n;++i){clip[i].e=startInside=!startInside}var start=subject[0],points,point;while(1){
// Find first unvisited intersection.
var current=start,isSubject=true;while(current.v)if((current=current.n)===start)return;points=current.z;stream.lineStart();do{current.v=current.o.v=true;if(current.e){if(isSubject){for(i=0,n=points.length;i<n;++i)stream.point((point=points[i])[0],point[1])}else{interpolate(current.x,current.n.x,1,stream)}current=current.n}else{if(isSubject){points=current.p.z;for(i=points.length-1;i>=0;--i)stream.point((point=points[i])[0],point[1])}else{interpolate(current.x,current.p.x,-1,stream)}current=current.p}current=current.o;points=current.z;isSubject=!isSubject}while(!current.v);stream.lineEnd()}}function link(array){if(!(n=array.length))return;var n,i=0,a=array[0],b;while(++i<n){a.n=b=array[i];b.p=a;a=b}a.n=b=array[0];b.p=a}var sum=adder();function longitude(point){if(abs(point[0])<=pi)return point[0];else return sign(point[0])*((abs(point[0])+pi)%tau-pi)}function polygonContains(polygon,point){var lambda=longitude(point),phi=point[1],sinPhi=sin(phi),normal=[sin(lambda),-cos(lambda),0],angle=0,winding=0;sum.reset();if(sinPhi===1)phi=halfPi+epsilon;else if(sinPhi===-1)phi=-halfPi-epsilon;for(var i=0,n=polygon.length;i<n;++i){if(!(m=(ring=polygon[i]).length))continue;var ring,m,point0=ring[m-1],lambda0=longitude(point0),phi0=point0[1]/2+quarterPi,sinPhi0=sin(phi0),cosPhi0=cos(phi0);for(var j=0;j<m;++j,lambda0=lambda1,sinPhi0=sinPhi1,cosPhi0=cosPhi1,point0=point1){var point1=ring[j],lambda1=longitude(point1),phi1=point1[1]/2+quarterPi,sinPhi1=sin(phi1),cosPhi1=cos(phi1),delta=lambda1-lambda0,sign=delta>=0?1:-1,absDelta=sign*delta,antimeridian=absDelta>pi,k=sinPhi0*sinPhi1;sum.add(atan2(k*sign*sin(absDelta),cosPhi0*cosPhi1+k*cos(absDelta)));angle+=antimeridian?delta+sign*tau:delta;
// Are the longitudes either side of the point’s meridian (lambda),
// and are the latitudes smaller than the parallel (phi)?
if(antimeridian^lambda0>=lambda^lambda1>=lambda){var arc=cartesianCross(cartesian(point0),cartesian(point1));cartesianNormalizeInPlace(arc);var intersection=cartesianCross(normal,arc);cartesianNormalizeInPlace(intersection);var phiArc=(antimeridian^delta>=0?-1:1)*asin(intersection[2]);if(phi>phiArc||phi===phiArc&&(arc[0]||arc[1])){winding+=antimeridian^delta>=0?1:-1}}}}
// First, determine whether the South pole is inside or outside:
//
// It is inside if:
// * the polygon winds around it in a clockwise direction.
// * the polygon does not (cumulatively) wind around it, but has a negative
//   (counter-clockwise) area.
//
// Second, count the (signed) number of times a segment crosses a lambda
// from the point to the South pole.  If it is zero, then the point is the
// same side as the South pole.
return(angle<-epsilon||angle<epsilon&&sum<-epsilon)^winding&1}function clip(pointVisible,clipLine,interpolate,start){return function(sink){var line=clipLine(sink),ringBuffer=clipBuffer(),ringSink=clipLine(ringBuffer),polygonStarted=false,polygon,segments,ring;var clip={point:point,lineStart:lineStart,lineEnd:lineEnd,polygonStart:function(){clip.point=pointRing;clip.lineStart=ringStart;clip.lineEnd=ringEnd;segments=[];polygon=[]},polygonEnd:function(){clip.point=point;clip.lineStart=lineStart;clip.lineEnd=lineEnd;segments=d3Array.merge(segments);var startInside=polygonContains(polygon,start);if(segments.length){if(!polygonStarted)sink.polygonStart(),polygonStarted=true;clipRejoin(segments,compareIntersection,startInside,interpolate,sink)}else if(startInside){if(!polygonStarted)sink.polygonStart(),polygonStarted=true;sink.lineStart();interpolate(null,null,1,sink);sink.lineEnd()}if(polygonStarted)sink.polygonEnd(),polygonStarted=false;segments=polygon=null},sphere:function(){sink.polygonStart();sink.lineStart();interpolate(null,null,1,sink);sink.lineEnd();sink.polygonEnd()}};function point(lambda,phi){if(pointVisible(lambda,phi))sink.point(lambda,phi)}function pointLine(lambda,phi){line.point(lambda,phi)}function lineStart(){clip.point=pointLine;line.lineStart()}function lineEnd(){clip.point=point;line.lineEnd()}function pointRing(lambda,phi){ring.push([lambda,phi]);ringSink.point(lambda,phi)}function ringStart(){ringSink.lineStart();ring=[]}function ringEnd(){pointRing(ring[0][0],ring[0][1]);ringSink.lineEnd();var clean=ringSink.clean(),ringSegments=ringBuffer.result(),i,n=ringSegments.length,m,segment,point;ring.pop();polygon.push(ring);ring=null;if(!n)return;
// No intersections.
if(clean&1){segment=ringSegments[0];if((m=segment.length-1)>0){if(!polygonStarted)sink.polygonStart(),polygonStarted=true;sink.lineStart();for(i=0;i<m;++i)sink.point((point=segment[i])[0],point[1]);sink.lineEnd()}return}
// Rejoin connected segments.
// TODO reuse ringBuffer.rejoin()?
if(n>1&&clean&2)ringSegments.push(ringSegments.pop().concat(ringSegments.shift()));segments.push(ringSegments.filter(validSegment))}return clip}}function validSegment(segment){return segment.length>1}
// Intersections are sorted along the clip edge. For both antimeridian cutting
// and circle clipping, the same comparison is used.
function compareIntersection(a,b){return((a=a.x)[0]<0?a[1]-halfPi-epsilon:halfPi-a[1])-((b=b.x)[0]<0?b[1]-halfPi-epsilon:halfPi-b[1])}var clipAntimeridian=clip(function(){return true},clipAntimeridianLine,clipAntimeridianInterpolate,[-pi,-halfPi]);
// Takes a line and cuts into visible segments. Return values: 0 - there were
// intersections or the line was empty; 1 - no intersections; 2 - there were
// intersections, and the first and last segments should be rejoined.
function clipAntimeridianLine(stream){var lambda0=NaN,phi0=NaN,sign0=NaN,clean;// no intersections
return{lineStart:function(){stream.lineStart();clean=1},point:function(lambda1,phi1){var sign1=lambda1>0?pi:-pi,delta=abs(lambda1-lambda0);if(abs(delta-pi)<epsilon){// line crosses a pole
stream.point(lambda0,phi0=(phi0+phi1)/2>0?halfPi:-halfPi);stream.point(sign0,phi0);stream.lineEnd();stream.lineStart();stream.point(sign1,phi0);stream.point(lambda1,phi0);clean=0}else if(sign0!==sign1&&delta>=pi){// line crosses antimeridian
if(abs(lambda0-sign0)<epsilon)lambda0-=sign0*epsilon;// handle degeneracies
if(abs(lambda1-sign1)<epsilon)lambda1-=sign1*epsilon;phi0=clipAntimeridianIntersect(lambda0,phi0,lambda1,phi1);stream.point(sign0,phi0);stream.lineEnd();stream.lineStart();stream.point(sign1,phi0);clean=0}stream.point(lambda0=lambda1,phi0=phi1);sign0=sign1},lineEnd:function(){stream.lineEnd();lambda0=phi0=NaN},clean:function(){return 2-clean;// if intersections, rejoin first and last segments
}}}function clipAntimeridianIntersect(lambda0,phi0,lambda1,phi1){var cosPhi0,cosPhi1,sinLambda0Lambda1=sin(lambda0-lambda1);return abs(sinLambda0Lambda1)>epsilon?atan((sin(phi0)*(cosPhi1=cos(phi1))*sin(lambda1)-sin(phi1)*(cosPhi0=cos(phi0))*sin(lambda0))/(cosPhi0*cosPhi1*sinLambda0Lambda1)):(phi0+phi1)/2}function clipAntimeridianInterpolate(from,to,direction,stream){var phi;if(from==null){phi=direction*halfPi;stream.point(-pi,phi);stream.point(0,phi);stream.point(pi,phi);stream.point(pi,0);stream.point(pi,-phi);stream.point(0,-phi);stream.point(-pi,-phi);stream.point(-pi,0);stream.point(-pi,phi)}else if(abs(from[0]-to[0])>epsilon){var lambda=from[0]<to[0]?pi:-pi;phi=direction*lambda/2;stream.point(-lambda,phi);stream.point(0,phi);stream.point(lambda,phi)}else{stream.point(to[0],to[1])}}function clipCircle(radius){var cr=cos(radius),delta=6*radians,smallRadius=cr>0,notHemisphere=abs(cr)>epsilon;// TODO optimise for this common case
function interpolate(from,to,direction,stream){circleStream(stream,radius,delta,direction,from,to)}function visible(lambda,phi){return cos(lambda)*cos(phi)>cr}
// Takes a line and cuts into visible segments. Return values used for polygon
// clipping: 0 - there were intersections or the line was empty; 1 - no
// intersections 2 - there were intersections, and the first and last segments
// should be rejoined.
function clipLine(stream){var point0,// previous point
c0,// code for previous point
v0,// visibility of previous point
v00,// visibility of first point
clean;// no intersections
return{lineStart:function(){v00=v0=false;clean=1},point:function(lambda,phi){var point1=[lambda,phi],point2,v=visible(lambda,phi),c=smallRadius?v?0:code(lambda,phi):v?code(lambda+(lambda<0?pi:-pi),phi):0;if(!point0&&(v00=v0=v))stream.lineStart();
// Handle degeneracies.
// TODO ignore if not clipping polygons.
if(v!==v0){point2=intersect(point0,point1);if(!point2||pointEqual(point0,point2)||pointEqual(point1,point2)){point1[0]+=epsilon;point1[1]+=epsilon;v=visible(point1[0],point1[1])}}if(v!==v0){clean=0;if(v){
// outside going in
stream.lineStart();point2=intersect(point1,point0);stream.point(point2[0],point2[1])}else{
// inside going out
point2=intersect(point0,point1);stream.point(point2[0],point2[1]);stream.lineEnd()}point0=point2}else if(notHemisphere&&point0&&smallRadius^v){var t;
// If the codes for two points are different, or are both zero,
// and there this segment intersects with the small circle.
if(!(c&c0)&&(t=intersect(point1,point0,true))){clean=0;if(smallRadius){stream.lineStart();stream.point(t[0][0],t[0][1]);stream.point(t[1][0],t[1][1]);stream.lineEnd()}else{stream.point(t[1][0],t[1][1]);stream.lineEnd();stream.lineStart();stream.point(t[0][0],t[0][1])}}}if(v&&(!point0||!pointEqual(point0,point1))){stream.point(point1[0],point1[1])}point0=point1,v0=v,c0=c},lineEnd:function(){if(v0)stream.lineEnd();point0=null},
// Rejoin first and last segments if there were intersections and the first
// and last points were visible.
clean:function(){return clean|(v00&&v0)<<1}}}
// Intersects the great circle between a and b with the clip circle.
function intersect(a,b,two){var pa=cartesian(a),pb=cartesian(b);
// We have two planes, n1.p = d1 and n2.p = d2.
// Find intersection line p(t) = c1 n1 + c2 n2 + t (n1 ⨯ n2).
var n1=[1,0,0],// normal
n2=cartesianCross(pa,pb),n2n2=cartesianDot(n2,n2),n1n2=n2[0],// cartesianDot(n1, n2),
determinant=n2n2-n1n2*n1n2;
// Two polar points.
if(!determinant)return!two&&a;var c1=cr*n2n2/determinant,c2=-cr*n1n2/determinant,n1xn2=cartesianCross(n1,n2),A=cartesianScale(n1,c1),B=cartesianScale(n2,c2);cartesianAddInPlace(A,B);
// Solve |p(t)|^2 = 1.
var u=n1xn2,w=cartesianDot(A,u),uu=cartesianDot(u,u),t2=w*w-uu*(cartesianDot(A,A)-1);if(t2<0)return;var t=sqrt(t2),q=cartesianScale(u,(-w-t)/uu);cartesianAddInPlace(q,A);q=spherical(q);if(!two)return q;
// Two intersection points.
var lambda0=a[0],lambda1=b[0],phi0=a[1],phi1=b[1],z;if(lambda1<lambda0)z=lambda0,lambda0=lambda1,lambda1=z;var delta=lambda1-lambda0,polar=abs(delta-pi)<epsilon,meridian=polar||delta<epsilon;if(!polar&&phi1<phi0)z=phi0,phi0=phi1,phi1=z;
// Check that the first point is between a and b.
if(meridian?polar?phi0+phi1>0^q[1]<(abs(q[0]-lambda0)<epsilon?phi0:phi1):phi0<=q[1]&&q[1]<=phi1:delta>pi^(lambda0<=q[0]&&q[0]<=lambda1)){var q1=cartesianScale(u,(-w+t)/uu);cartesianAddInPlace(q1,A);return[q,spherical(q1)]}}
// Generates a 4-bit vector representing the location of a point relative to
// the small circle's bounding box.
function code(lambda,phi){var r=smallRadius?radius:pi-radius,code=0;if(lambda<-r)code|=1;// left
else if(lambda>r)code|=2;// right
if(phi<-r)code|=4;// below
else if(phi>r)code|=8;// above
return code}return clip(visible,clipLine,interpolate,smallRadius?[0,-radius]:[-pi,radius-pi])}function clipLine(a,b,x0,y0,x1,y1){var ax=a[0],ay=a[1],bx=b[0],by=b[1],t0=0,t1=1,dx=bx-ax,dy=by-ay,r;r=x0-ax;if(!dx&&r>0)return;r/=dx;if(dx<0){if(r<t0)return;if(r<t1)t1=r}else if(dx>0){if(r>t1)return;if(r>t0)t0=r}r=x1-ax;if(!dx&&r<0)return;r/=dx;if(dx<0){if(r>t1)return;if(r>t0)t0=r}else if(dx>0){if(r<t0)return;if(r<t1)t1=r}r=y0-ay;if(!dy&&r>0)return;r/=dy;if(dy<0){if(r<t0)return;if(r<t1)t1=r}else if(dy>0){if(r>t1)return;if(r>t0)t0=r}r=y1-ay;if(!dy&&r<0)return;r/=dy;if(dy<0){if(r>t1)return;if(r>t0)t0=r}else if(dy>0){if(r<t0)return;if(r<t1)t1=r}if(t0>0)a[0]=ax+t0*dx,a[1]=ay+t0*dy;if(t1<1)b[0]=ax+t1*dx,b[1]=ay+t1*dy;return true}var clipMax=1e9,clipMin=-clipMax;
// TODO Use d3-polygon’s polygonContains here for the ring check?
// TODO Eliminate duplicate buffering in clipBuffer and polygon.push?
function clipRectangle(x0,y0,x1,y1){function visible(x,y){return x0<=x&&x<=x1&&y0<=y&&y<=y1}function interpolate(from,to,direction,stream){var a=0,a1=0;if(from==null||(a=corner(from,direction))!==(a1=corner(to,direction))||comparePoint(from,to)<0^direction>0){do{stream.point(a===0||a===3?x0:x1,a>1?y1:y0)}while((a=(a+direction+4)%4)!==a1)}else{stream.point(to[0],to[1])}}function corner(p,direction){return abs(p[0]-x0)<epsilon?direction>0?0:3:abs(p[0]-x1)<epsilon?direction>0?2:1:abs(p[1]-y0)<epsilon?direction>0?1:0:direction>0?3:2;// abs(p[1] - y1) < epsilon
}function compareIntersection(a,b){return comparePoint(a.x,b.x)}function comparePoint(a,b){var ca=corner(a,1),cb=corner(b,1);return ca!==cb?ca-cb:ca===0?b[1]-a[1]:ca===1?a[0]-b[0]:ca===2?a[1]-b[1]:b[0]-a[0]}return function(stream){var activeStream=stream,bufferStream=clipBuffer(),segments,polygon,ring,x__,y__,v__,// first point
x_,y_,v_,// previous point
first,clean;var clipStream={point:point,lineStart:lineStart,lineEnd:lineEnd,polygonStart:polygonStart,polygonEnd:polygonEnd};function point(x,y){if(visible(x,y))activeStream.point(x,y)}function polygonInside(){var winding=0;for(var i=0,n=polygon.length;i<n;++i){for(var ring=polygon[i],j=1,m=ring.length,point=ring[0],a0,a1,b0=point[0],b1=point[1];j<m;++j){a0=b0,a1=b1,point=ring[j],b0=point[0],b1=point[1];if(a1<=y1){if(b1>y1&&(b0-a0)*(y1-a1)>(b1-a1)*(x0-a0))++winding}else{if(b1<=y1&&(b0-a0)*(y1-a1)<(b1-a1)*(x0-a0))--winding}}}return winding}
// Buffer geometry within a polygon and then clip it en masse.
function polygonStart(){activeStream=bufferStream,segments=[],polygon=[],clean=true}function polygonEnd(){var startInside=polygonInside(),cleanInside=clean&&startInside,visible=(segments=d3Array.merge(segments)).length;if(cleanInside||visible){stream.polygonStart();if(cleanInside){stream.lineStart();interpolate(null,null,1,stream);stream.lineEnd()}if(visible){clipRejoin(segments,compareIntersection,startInside,interpolate,stream)}stream.polygonEnd()}activeStream=stream,segments=polygon=ring=null}function lineStart(){clipStream.point=linePoint;if(polygon)polygon.push(ring=[]);first=true;v_=false;x_=y_=NaN}
// TODO rather than special-case polygons, simply handle them separately.
// Ideally, coincident intersection points should be jittered to avoid
// clipping issues.
function lineEnd(){if(segments){linePoint(x__,y__);if(v__&&v_)bufferStream.rejoin();segments.push(bufferStream.result())}clipStream.point=point;if(v_)activeStream.lineEnd()}function linePoint(x,y){var v=visible(x,y);if(polygon)ring.push([x,y]);if(first){x__=x,y__=y,v__=v;first=false;if(v){activeStream.lineStart();activeStream.point(x,y)}}else{if(v&&v_)activeStream.point(x,y);else{var a=[x_=Math.max(clipMin,Math.min(clipMax,x_)),y_=Math.max(clipMin,Math.min(clipMax,y_))],b=[x=Math.max(clipMin,Math.min(clipMax,x)),y=Math.max(clipMin,Math.min(clipMax,y))];if(clipLine(a,b,x0,y0,x1,y1)){if(!v_){activeStream.lineStart();activeStream.point(a[0],a[1])}activeStream.point(b[0],b[1]);if(!v)activeStream.lineEnd();clean=false}else if(v){activeStream.lineStart();activeStream.point(x,y);clean=false}}}x_=x,y_=y,v_=v}return clipStream}}function extent(){var x0=0,y0=0,x1=960,y1=500,cache,cacheStream,clip;return clip={stream:function(stream){return cache&&cacheStream===stream?cache:cache=clipRectangle(x0,y0,x1,y1)(cacheStream=stream)},extent:function(_){return arguments.length?(x0=+_[0][0],y0=+_[0][1],x1=+_[1][0],y1=+_[1][1],cache=cacheStream=null,clip):[[x0,y0],[x1,y1]]}}}var lengthSum=adder(),lambda0$2,sinPhi0$1,cosPhi0$1;var lengthStream={sphere:noop,point:noop,lineStart:lengthLineStart,lineEnd:noop,polygonStart:noop,polygonEnd:noop};function lengthLineStart(){lengthStream.point=lengthPointFirst;lengthStream.lineEnd=lengthLineEnd}function lengthLineEnd(){lengthStream.point=lengthStream.lineEnd=noop}function lengthPointFirst(lambda,phi){lambda*=radians,phi*=radians;lambda0$2=lambda,sinPhi0$1=sin(phi),cosPhi0$1=cos(phi);lengthStream.point=lengthPoint}function lengthPoint(lambda,phi){lambda*=radians,phi*=radians;var sinPhi=sin(phi),cosPhi=cos(phi),delta=abs(lambda-lambda0$2),cosDelta=cos(delta),sinDelta=sin(delta),x=cosPhi*sinDelta,y=cosPhi0$1*sinPhi-sinPhi0$1*cosPhi*cosDelta,z=sinPhi0$1*sinPhi+cosPhi0$1*cosPhi*cosDelta;lengthSum.add(atan2(sqrt(x*x+y*y),z));lambda0$2=lambda,sinPhi0$1=sinPhi,cosPhi0$1=cosPhi}function length(object){lengthSum.reset();geoStream(object,lengthStream);return+lengthSum}var coordinates=[null,null],object={type:"LineString",coordinates:coordinates};function distance(a,b){coordinates[0]=a;coordinates[1]=b;return length(object)}var containsObjectType={Feature:function(object,point){return containsGeometry(object.geometry,point)},FeatureCollection:function(object,point){var features=object.features,i=-1,n=features.length;while(++i<n)if(containsGeometry(features[i].geometry,point))return true;return false}};var containsGeometryType={Sphere:function(){return true},Point:function(object,point){return containsPoint(object.coordinates,point)},MultiPoint:function(object,point){var coordinates=object.coordinates,i=-1,n=coordinates.length;while(++i<n)if(containsPoint(coordinates[i],point))return true;return false},LineString:function(object,point){return containsLine(object.coordinates,point)},MultiLineString:function(object,point){var coordinates=object.coordinates,i=-1,n=coordinates.length;while(++i<n)if(containsLine(coordinates[i],point))return true;return false},Polygon:function(object,point){return containsPolygon(object.coordinates,point)},MultiPolygon:function(object,point){var coordinates=object.coordinates,i=-1,n=coordinates.length;while(++i<n)if(containsPolygon(coordinates[i],point))return true;return false},GeometryCollection:function(object,point){var geometries=object.geometries,i=-1,n=geometries.length;while(++i<n)if(containsGeometry(geometries[i],point))return true;return false}};function containsGeometry(geometry,point){return geometry&&containsGeometryType.hasOwnProperty(geometry.type)?containsGeometryType[geometry.type](geometry,point):false}function containsPoint(coordinates,point){return distance(coordinates,point)===0}function containsLine(coordinates,point){var ao,bo,ab;for(var i=0,n=coordinates.length;i<n;i++){bo=distance(coordinates[i],point);if(bo===0)return true;if(i>0){ab=distance(coordinates[i],coordinates[i-1]);if(ab>0&&ao<=ab&&bo<=ab&&(ao+bo-ab)*(1-Math.pow((ao-bo)/ab,2))<epsilon2*ab)return true}ao=bo}return false}function containsPolygon(coordinates,point){return!!polygonContains(coordinates.map(ringRadians),pointRadians(point))}function ringRadians(ring){return ring=ring.map(pointRadians),ring.pop(),ring}function pointRadians(point){return[point[0]*radians,point[1]*radians]}function contains(object,point){return(object&&containsObjectType.hasOwnProperty(object.type)?containsObjectType[object.type]:containsGeometry)(object,point)}function graticuleX(y0,y1,dy){var y=d3Array.range(y0,y1-epsilon,dy).concat(y1);return function(x){return y.map(function(y){return[x,y]})}}function graticuleY(x0,x1,dx){var x=d3Array.range(x0,x1-epsilon,dx).concat(x1);return function(y){return x.map(function(x){return[x,y]})}}function graticule(){var x1,x0,X1,X0,y1,y0,Y1,Y0,dx=10,dy=dx,DX=90,DY=360,x,y,X,Y,precision=2.5;function graticule(){return{type:"MultiLineString",coordinates:lines()}}function lines(){return d3Array.range(ceil(X0/DX)*DX,X1,DX).map(X).concat(d3Array.range(ceil(Y0/DY)*DY,Y1,DY).map(Y)).concat(d3Array.range(ceil(x0/dx)*dx,x1,dx).filter(function(x){return abs(x%DX)>epsilon}).map(x)).concat(d3Array.range(ceil(y0/dy)*dy,y1,dy).filter(function(y){return abs(y%DY)>epsilon}).map(y))}graticule.lines=function(){return lines().map(function(coordinates){return{type:"LineString",coordinates:coordinates}})};graticule.outline=function(){return{type:"Polygon",coordinates:[X(X0).concat(Y(Y1).slice(1),X(X1).reverse().slice(1),Y(Y0).reverse().slice(1))]}};graticule.extent=function(_){if(!arguments.length)return graticule.extentMinor();return graticule.extentMajor(_).extentMinor(_)};graticule.extentMajor=function(_){if(!arguments.length)return[[X0,Y0],[X1,Y1]];X0=+_[0][0],X1=+_[1][0];Y0=+_[0][1],Y1=+_[1][1];if(X0>X1)_=X0,X0=X1,X1=_;if(Y0>Y1)_=Y0,Y0=Y1,Y1=_;return graticule.precision(precision)};graticule.extentMinor=function(_){if(!arguments.length)return[[x0,y0],[x1,y1]];x0=+_[0][0],x1=+_[1][0];y0=+_[0][1],y1=+_[1][1];if(x0>x1)_=x0,x0=x1,x1=_;if(y0>y1)_=y0,y0=y1,y1=_;return graticule.precision(precision)};graticule.step=function(_){if(!arguments.length)return graticule.stepMinor();return graticule.stepMajor(_).stepMinor(_)};graticule.stepMajor=function(_){if(!arguments.length)return[DX,DY];DX=+_[0],DY=+_[1];return graticule};graticule.stepMinor=function(_){if(!arguments.length)return[dx,dy];dx=+_[0],dy=+_[1];return graticule};graticule.precision=function(_){if(!arguments.length)return precision;precision=+_;x=graticuleX(y0,y1,90);y=graticuleY(x0,x1,precision);X=graticuleX(Y0,Y1,90);Y=graticuleY(X0,X1,precision);return graticule};return graticule.extentMajor([[-180,-90+epsilon],[180,90-epsilon]]).extentMinor([[-180,-80-epsilon],[180,80+epsilon]])}function graticule10(){return graticule()()}function interpolate(a,b){var x0=a[0]*radians,y0=a[1]*radians,x1=b[0]*radians,y1=b[1]*radians,cy0=cos(y0),sy0=sin(y0),cy1=cos(y1),sy1=sin(y1),kx0=cy0*cos(x0),ky0=cy0*sin(x0),kx1=cy1*cos(x1),ky1=cy1*sin(x1),d=2*asin(sqrt(haversin(y1-y0)+cy0*cy1*haversin(x1-x0))),k=sin(d);var interpolate=d?function(t){var B=sin(t*=d)/k,A=sin(d-t)/k,x=A*kx0+B*kx1,y=A*ky0+B*ky1,z=A*sy0+B*sy1;return[atan2(y,x)*degrees,atan2(z,sqrt(x*x+y*y))*degrees]}:function(){return[x0*degrees,y0*degrees]};interpolate.distance=d;return interpolate}function identity(x){return x}var areaSum$1=adder(),areaRingSum$1=adder(),x00,y00,x0$1,y0$1;var areaStream$1={point:noop,lineStart:noop,lineEnd:noop,polygonStart:function(){areaStream$1.lineStart=areaRingStart$1;areaStream$1.lineEnd=areaRingEnd$1},polygonEnd:function(){areaStream$1.lineStart=areaStream$1.lineEnd=areaStream$1.point=noop;areaSum$1.add(abs(areaRingSum$1));areaRingSum$1.reset()},result:function(){var area=areaSum$1/2;areaSum$1.reset();return area}};function areaRingStart$1(){areaStream$1.point=areaPointFirst$1}function areaPointFirst$1(x,y){areaStream$1.point=areaPoint$1;x00=x0$1=x,y00=y0$1=y}function areaPoint$1(x,y){areaRingSum$1.add(y0$1*x-x0$1*y);x0$1=x,y0$1=y}function areaRingEnd$1(){areaPoint$1(x00,y00)}var x0$2=Infinity,y0$2=x0$2,x1=-x0$2,y1=x1;var boundsStream$1={point:boundsPoint$1,lineStart:noop,lineEnd:noop,polygonStart:noop,polygonEnd:noop,result:function(){var bounds=[[x0$2,y0$2],[x1,y1]];x1=y1=-(y0$2=x0$2=Infinity);return bounds}};function boundsPoint$1(x,y){if(x<x0$2)x0$2=x;if(x>x1)x1=x;if(y<y0$2)y0$2=y;if(y>y1)y1=y}
// TODO Enforce positive area for exterior, negative area for interior?
var X0$1=0,Y0$1=0,Z0$1=0,X1$1=0,Y1$1=0,Z1$1=0,X2$1=0,Y2$1=0,Z2$1=0,x00$1,y00$1,x0$3,y0$3;var centroidStream$1={point:centroidPoint$1,lineStart:centroidLineStart$1,lineEnd:centroidLineEnd$1,polygonStart:function(){centroidStream$1.lineStart=centroidRingStart$1;centroidStream$1.lineEnd=centroidRingEnd$1},polygonEnd:function(){centroidStream$1.point=centroidPoint$1;centroidStream$1.lineStart=centroidLineStart$1;centroidStream$1.lineEnd=centroidLineEnd$1},result:function(){var centroid=Z2$1?[X2$1/Z2$1,Y2$1/Z2$1]:Z1$1?[X1$1/Z1$1,Y1$1/Z1$1]:Z0$1?[X0$1/Z0$1,Y0$1/Z0$1]:[NaN,NaN];X0$1=Y0$1=Z0$1=X1$1=Y1$1=Z1$1=X2$1=Y2$1=Z2$1=0;return centroid}};function centroidPoint$1(x,y){X0$1+=x;Y0$1+=y;++Z0$1}function centroidLineStart$1(){centroidStream$1.point=centroidPointFirstLine}function centroidPointFirstLine(x,y){centroidStream$1.point=centroidPointLine;centroidPoint$1(x0$3=x,y0$3=y)}function centroidPointLine(x,y){var dx=x-x0$3,dy=y-y0$3,z=sqrt(dx*dx+dy*dy);X1$1+=z*(x0$3+x)/2;Y1$1+=z*(y0$3+y)/2;Z1$1+=z;centroidPoint$1(x0$3=x,y0$3=y)}function centroidLineEnd$1(){centroidStream$1.point=centroidPoint$1}function centroidRingStart$1(){centroidStream$1.point=centroidPointFirstRing}function centroidRingEnd$1(){centroidPointRing(x00$1,y00$1)}function centroidPointFirstRing(x,y){centroidStream$1.point=centroidPointRing;centroidPoint$1(x00$1=x0$3=x,y00$1=y0$3=y)}function centroidPointRing(x,y){var dx=x-x0$3,dy=y-y0$3,z=sqrt(dx*dx+dy*dy);X1$1+=z*(x0$3+x)/2;Y1$1+=z*(y0$3+y)/2;Z1$1+=z;z=y0$3*x-x0$3*y;X2$1+=z*(x0$3+x);Y2$1+=z*(y0$3+y);Z2$1+=z*3;centroidPoint$1(x0$3=x,y0$3=y)}function PathContext(context){this._context=context}PathContext.prototype={_radius:4.5,pointRadius:function(_){return this._radius=_,this},polygonStart:function(){this._line=0},polygonEnd:function(){this._line=NaN},lineStart:function(){this._point=0},lineEnd:function(){if(this._line===0)this._context.closePath();this._point=NaN},point:function(x,y){switch(this._point){case 0:{this._context.moveTo(x,y);this._point=1;break}case 1:{this._context.lineTo(x,y);break}default:{this._context.moveTo(x+this._radius,y);this._context.arc(x,y,this._radius,0,tau);break}}},result:noop};var lengthSum$1=adder(),lengthRing,x00$2,y00$2,x0$4,y0$4;var lengthStream$1={point:noop,lineStart:function(){lengthStream$1.point=lengthPointFirst$1},lineEnd:function(){if(lengthRing)lengthPoint$1(x00$2,y00$2);lengthStream$1.point=noop},polygonStart:function(){lengthRing=true},polygonEnd:function(){lengthRing=null},result:function(){var length=+lengthSum$1;lengthSum$1.reset();return length}};function lengthPointFirst$1(x,y){lengthStream$1.point=lengthPoint$1;x00$2=x0$4=x,y00$2=y0$4=y}function lengthPoint$1(x,y){x0$4-=x,y0$4-=y;lengthSum$1.add(sqrt(x0$4*x0$4+y0$4*y0$4));x0$4=x,y0$4=y}function PathString(){this._string=[]}PathString.prototype={_radius:4.5,_circle:circle$1(4.5),pointRadius:function(_){if((_=+_)!==this._radius)this._radius=_,this._circle=null;return this},polygonStart:function(){this._line=0},polygonEnd:function(){this._line=NaN},lineStart:function(){this._point=0},lineEnd:function(){if(this._line===0)this._string.push("Z");this._point=NaN},point:function(x,y){switch(this._point){case 0:{this._string.push("M",x,",",y);this._point=1;break}case 1:{this._string.push("L",x,",",y);break}default:{if(this._circle==null)this._circle=circle$1(this._radius);this._string.push("M",x,",",y,this._circle);break}}},result:function(){if(this._string.length){var result=this._string.join("");this._string=[];return result}else{return null}}};function circle$1(radius){return"m0,"+radius+"a"+radius+","+radius+" 0 1,1 0,"+-2*radius+"a"+radius+","+radius+" 0 1,1 0,"+2*radius+"z"}function index(projection,context){var pointRadius=4.5,projectionStream,contextStream;function path(object){if(object){if(typeof pointRadius==="function")contextStream.pointRadius(+pointRadius.apply(this,arguments));geoStream(object,projectionStream(contextStream))}return contextStream.result()}path.area=function(object){geoStream(object,projectionStream(areaStream$1));return areaStream$1.result()};path.measure=function(object){geoStream(object,projectionStream(lengthStream$1));return lengthStream$1.result()};path.bounds=function(object){geoStream(object,projectionStream(boundsStream$1));return boundsStream$1.result()};path.centroid=function(object){geoStream(object,projectionStream(centroidStream$1));return centroidStream$1.result()};path.projection=function(_){return arguments.length?(projectionStream=_==null?(projection=null,identity):(projection=_).stream,path):projection};path.context=function(_){if(!arguments.length)return context;contextStream=_==null?(context=null,new PathString):new PathContext(context=_);if(typeof pointRadius!=="function")contextStream.pointRadius(pointRadius);return path};path.pointRadius=function(_){if(!arguments.length)return pointRadius;pointRadius=typeof _==="function"?_:(contextStream.pointRadius(+_),+_);return path};return path.projection(projection).context(context)}function transform(methods){return{stream:transformer(methods)}}function transformer(methods){return function(stream){var s=new TransformStream;for(var key in methods)s[key]=methods[key];s.stream=stream;return s}}function TransformStream(){}TransformStream.prototype={constructor:TransformStream,point:function(x,y){this.stream.point(x,y)},sphere:function(){this.stream.sphere()},lineStart:function(){this.stream.lineStart()},lineEnd:function(){this.stream.lineEnd()},polygonStart:function(){this.stream.polygonStart()},polygonEnd:function(){this.stream.polygonEnd()}};function fit(projection,fitBounds,object){var clip=projection.clipExtent&&projection.clipExtent();projection.scale(150).translate([0,0]);if(clip!=null)projection.clipExtent(null);geoStream(object,projection.stream(boundsStream$1));fitBounds(boundsStream$1.result());if(clip!=null)projection.clipExtent(clip);return projection}function fitExtent(projection,extent,object){return fit(projection,function(b){var w=extent[1][0]-extent[0][0],h=extent[1][1]-extent[0][1],k=Math.min(w/(b[1][0]-b[0][0]),h/(b[1][1]-b[0][1])),x=+extent[0][0]+(w-k*(b[1][0]+b[0][0]))/2,y=+extent[0][1]+(h-k*(b[1][1]+b[0][1]))/2;projection.scale(150*k).translate([x,y])},object)}function fitSize(projection,size,object){return fitExtent(projection,[[0,0],size],object)}function fitWidth(projection,width,object){return fit(projection,function(b){var w=+width,k=w/(b[1][0]-b[0][0]),x=(w-k*(b[1][0]+b[0][0]))/2,y=-k*b[0][1];projection.scale(150*k).translate([x,y])},object)}function fitHeight(projection,height,object){return fit(projection,function(b){var h=+height,k=h/(b[1][1]-b[0][1]),x=-k*b[0][0],y=(h-k*(b[1][1]+b[0][1]))/2;projection.scale(150*k).translate([x,y])},object)}var maxDepth=16,// maximum depth of subdivision
cosMinDistance=cos(30*radians);// cos(minimum angular distance)
function resample(project,delta2){return+delta2?resample$1(project,delta2):resampleNone(project)}function resampleNone(project){return transformer({point:function(x,y){x=project(x,y);this.stream.point(x[0],x[1])}})}function resample$1(project,delta2){function resampleLineTo(x0,y0,lambda0,a0,b0,c0,x1,y1,lambda1,a1,b1,c1,depth,stream){var dx=x1-x0,dy=y1-y0,d2=dx*dx+dy*dy;if(d2>4*delta2&&depth--){var a=a0+a1,b=b0+b1,c=c0+c1,m=sqrt(a*a+b*b+c*c),phi2=asin(c/=m),lambda2=abs(abs(c)-1)<epsilon||abs(lambda0-lambda1)<epsilon?(lambda0+lambda1)/2:atan2(b,a),p=project(lambda2,phi2),x2=p[0],y2=p[1],dx2=x2-x0,dy2=y2-y0,dz=dy*dx2-dx*dy2;if(dz*dz/d2>delta2||abs((dx*dx2+dy*dy2)/d2-.5)>.3||a0*a1+b0*b1+c0*c1<cosMinDistance){// angular distance
resampleLineTo(x0,y0,lambda0,a0,b0,c0,x2,y2,lambda2,a/=m,b/=m,c,depth,stream);stream.point(x2,y2);resampleLineTo(x2,y2,lambda2,a,b,c,x1,y1,lambda1,a1,b1,c1,depth,stream)}}}return function(stream){var lambda00,x00,y00,a00,b00,c00,// first point
lambda0,x0,y0,a0,b0,c0;// previous point
var resampleStream={point:point,lineStart:lineStart,lineEnd:lineEnd,polygonStart:function(){stream.polygonStart();resampleStream.lineStart=ringStart},polygonEnd:function(){stream.polygonEnd();resampleStream.lineStart=lineStart}};function point(x,y){x=project(x,y);stream.point(x[0],x[1])}function lineStart(){x0=NaN;resampleStream.point=linePoint;stream.lineStart()}function linePoint(lambda,phi){var c=cartesian([lambda,phi]),p=project(lambda,phi);resampleLineTo(x0,y0,lambda0,a0,b0,c0,x0=p[0],y0=p[1],lambda0=lambda,a0=c[0],b0=c[1],c0=c[2],maxDepth,stream);stream.point(x0,y0)}function lineEnd(){resampleStream.point=point;stream.lineEnd()}function ringStart(){lineStart();resampleStream.point=ringPoint;resampleStream.lineEnd=ringEnd}function ringPoint(lambda,phi){linePoint(lambda00=lambda,phi),x00=x0,y00=y0,a00=a0,b00=b0,c00=c0;resampleStream.point=linePoint}function ringEnd(){resampleLineTo(x0,y0,lambda0,a0,b0,c0,x00,y00,lambda00,a00,b00,c00,maxDepth,stream);resampleStream.lineEnd=lineEnd;lineEnd()}return resampleStream}}var transformRadians=transformer({point:function(x,y){this.stream.point(x*radians,y*radians)}});function transformRotate(rotate){return transformer({point:function(x,y){var r=rotate(x,y);return this.stream.point(r[0],r[1])}})}function scaleTranslate(k,dx,dy){function transform(x,y){return[dx+k*x,dy-k*y]}transform.invert=function(x,y){return[(x-dx)/k,(dy-y)/k]};return transform}function scaleTranslateRotate(k,dx,dy,alpha){var cosAlpha=cos(alpha),sinAlpha=sin(alpha),a=cosAlpha*k,b=sinAlpha*k,ai=cosAlpha/k,bi=sinAlpha/k,ci=(sinAlpha*dy-cosAlpha*dx)/k,fi=(sinAlpha*dx+cosAlpha*dy)/k;function transform(x,y){return[a*x-b*y+dx,dy-b*x-a*y]}transform.invert=function(x,y){return[ai*x-bi*y+ci,fi-bi*x-ai*y]};return transform}function projection(project){return projectionMutator(function(){return project})()}function projectionMutator(projectAt){var project,k=150,// scale
x=480,y=250,// translate
lambda=0,phi=0,// center
deltaLambda=0,deltaPhi=0,deltaGamma=0,rotate,// pre-rotate
alpha=0,// post-rotate
theta=null,preclip=clipAntimeridian,// pre-clip angle
x0=null,y0,x1,y1,postclip=identity,// post-clip extent
delta2=.5,// precision
projectResample,projectTransform,projectRotateTransform,cache,cacheStream;function projection(point){return projectRotateTransform(point[0]*radians,point[1]*radians)}function invert(point){point=projectRotateTransform.invert(point[0],point[1]);return point&&[point[0]*degrees,point[1]*degrees]}projection.stream=function(stream){return cache&&cacheStream===stream?cache:cache=transformRadians(transformRotate(rotate)(preclip(projectResample(postclip(cacheStream=stream)))))};projection.preclip=function(_){return arguments.length?(preclip=_,theta=undefined,reset()):preclip};projection.postclip=function(_){return arguments.length?(postclip=_,x0=y0=x1=y1=null,reset()):postclip};projection.clipAngle=function(_){return arguments.length?(preclip=+_?clipCircle(theta=_*radians):(theta=null,clipAntimeridian),reset()):theta*degrees};projection.clipExtent=function(_){return arguments.length?(postclip=_==null?(x0=y0=x1=y1=null,identity):clipRectangle(x0=+_[0][0],y0=+_[0][1],x1=+_[1][0],y1=+_[1][1]),reset()):x0==null?null:[[x0,y0],[x1,y1]]};projection.scale=function(_){return arguments.length?(k=+_,recenter()):k};projection.translate=function(_){return arguments.length?(x=+_[0],y=+_[1],recenter()):[x,y]};projection.center=function(_){return arguments.length?(lambda=_[0]%360*radians,phi=_[1]%360*radians,recenter()):[lambda*degrees,phi*degrees]};projection.rotate=function(_){return arguments.length?(deltaLambda=_[0]%360*radians,deltaPhi=_[1]%360*radians,deltaGamma=_.length>2?_[2]%360*radians:0,recenter()):[deltaLambda*degrees,deltaPhi*degrees,deltaGamma*degrees]};projection.angle=function(_){return arguments.length?(alpha=_%360*radians,recenter()):alpha*degrees};projection.precision=function(_){return arguments.length?(projectResample=resample(projectTransform,delta2=_*_),reset()):sqrt(delta2)};projection.fitExtent=function(extent,object){return fitExtent(projection,extent,object)};projection.fitSize=function(size,object){return fitSize(projection,size,object)};projection.fitWidth=function(width,object){return fitWidth(projection,width,object)};projection.fitHeight=function(height,object){return fitHeight(projection,height,object)};function recenter(){var center=scaleTranslateRotate(k,0,0,alpha).apply(null,project(lambda,phi)),transform=(alpha?scaleTranslateRotate:scaleTranslate)(k,x-center[0],y-center[1],alpha);rotate=rotateRadians(deltaLambda,deltaPhi,deltaGamma);projectTransform=compose(project,transform);projectRotateTransform=compose(rotate,projectTransform);projectResample=resample(projectTransform,delta2);return reset()}function reset(){cache=cacheStream=null;return projection}return function(){project=projectAt.apply(this,arguments);projection.invert=project.invert&&invert;return recenter()}}function conicProjection(projectAt){var phi0=0,phi1=pi/3,m=projectionMutator(projectAt),p=m(phi0,phi1);p.parallels=function(_){return arguments.length?m(phi0=_[0]*radians,phi1=_[1]*radians):[phi0*degrees,phi1*degrees]};return p}function cylindricalEqualAreaRaw(phi0){var cosPhi0=cos(phi0);function forward(lambda,phi){return[lambda*cosPhi0,sin(phi)/cosPhi0]}forward.invert=function(x,y){return[x/cosPhi0,asin(y*cosPhi0)]};return forward}function conicEqualAreaRaw(y0,y1){var sy0=sin(y0),n=(sy0+sin(y1))/2;
// Are the parallels symmetrical around the Equator?
if(abs(n)<epsilon)return cylindricalEqualAreaRaw(y0);var c=1+sy0*(2*n-sy0),r0=sqrt(c)/n;function project(x,y){var r=sqrt(c-2*n*sin(y))/n;return[r*sin(x*=n),r0-r*cos(x)]}project.invert=function(x,y){var r0y=r0-y;return[atan2(x,abs(r0y))/n*sign(r0y),asin((c-(x*x+r0y*r0y)*n*n)/(2*n))]};return project}function conicEqualArea(){return conicProjection(conicEqualAreaRaw).scale(155.424).center([0,33.6442])}function albers(){return conicEqualArea().parallels([29.5,45.5]).scale(1070).translate([480,250]).rotate([96,0]).center([-.6,38.7])}
// The projections must have mutually exclusive clip regions on the sphere,
// as this will avoid emitting interleaving lines and polygons.
function multiplex(streams){var n=streams.length;return{point:function(x,y){var i=-1;while(++i<n)streams[i].point(x,y)},sphere:function(){var i=-1;while(++i<n)streams[i].sphere()},lineStart:function(){var i=-1;while(++i<n)streams[i].lineStart()},lineEnd:function(){var i=-1;while(++i<n)streams[i].lineEnd()},polygonStart:function(){var i=-1;while(++i<n)streams[i].polygonStart()},polygonEnd:function(){var i=-1;while(++i<n)streams[i].polygonEnd()}}}
// A composite projection for the United States, configured by default for
// 960×500. The projection also works quite well at 960×600 if you change the
// scale to 1285 and adjust the translate accordingly. The set of standard
// parallels for each region comes from USGS, which is published here:
// http://egsc.usgs.gov/isb/pubs/MapProjections/projections.html#albers
function albersUsa(){var cache,cacheStream,lower48=albers(),lower48Point,alaska=conicEqualArea().rotate([154,0]).center([-2,58.5]).parallels([55,65]),alaskaPoint,// EPSG:3338
hawaii=conicEqualArea().rotate([157,0]).center([-3,19.9]).parallels([8,18]),hawaiiPoint,// ESRI:102007
point,pointStream={point:function(x,y){point=[x,y]}};function albersUsa(coordinates){var x=coordinates[0],y=coordinates[1];return point=null,(lower48Point.point(x,y),point)||(alaskaPoint.point(x,y),point)||(hawaiiPoint.point(x,y),point)}albersUsa.invert=function(coordinates){var k=lower48.scale(),t=lower48.translate(),x=(coordinates[0]-t[0])/k,y=(coordinates[1]-t[1])/k;return(y>=.12&&y<.234&&x>=-.425&&x<-.214?alaska:y>=.166&&y<.234&&x>=-.214&&x<-.115?hawaii:lower48).invert(coordinates)};albersUsa.stream=function(stream){return cache&&cacheStream===stream?cache:cache=multiplex([lower48.stream(cacheStream=stream),alaska.stream(stream),hawaii.stream(stream)])};albersUsa.precision=function(_){if(!arguments.length)return lower48.precision();lower48.precision(_),alaska.precision(_),hawaii.precision(_);return reset()};albersUsa.scale=function(_){if(!arguments.length)return lower48.scale();lower48.scale(_),alaska.scale(_*.35),hawaii.scale(_);return albersUsa.translate(lower48.translate())};albersUsa.translate=function(_){if(!arguments.length)return lower48.translate();var k=lower48.scale(),x=+_[0],y=+_[1];lower48Point=lower48.translate(_).clipExtent([[x-.455*k,y-.238*k],[x+.455*k,y+.238*k]]).stream(pointStream);alaskaPoint=alaska.translate([x-.307*k,y+.201*k]).clipExtent([[x-.425*k+epsilon,y+.12*k+epsilon],[x-.214*k-epsilon,y+.234*k-epsilon]]).stream(pointStream);hawaiiPoint=hawaii.translate([x-.205*k,y+.212*k]).clipExtent([[x-.214*k+epsilon,y+.166*k+epsilon],[x-.115*k-epsilon,y+.234*k-epsilon]]).stream(pointStream);return reset()};albersUsa.fitExtent=function(extent,object){return fitExtent(albersUsa,extent,object)};albersUsa.fitSize=function(size,object){return fitSize(albersUsa,size,object)};albersUsa.fitWidth=function(width,object){return fitWidth(albersUsa,width,object)};albersUsa.fitHeight=function(height,object){return fitHeight(albersUsa,height,object)};function reset(){cache=cacheStream=null;return albersUsa}return albersUsa.scale(1070)}function azimuthalRaw(scale){return function(x,y){var cx=cos(x),cy=cos(y),k=scale(cx*cy);return[k*cy*sin(x),k*sin(y)]}}function azimuthalInvert(angle){return function(x,y){var z=sqrt(x*x+y*y),c=angle(z),sc=sin(c),cc=cos(c);return[atan2(x*sc,z*cc),asin(z&&y*sc/z)]}}var azimuthalEqualAreaRaw=azimuthalRaw(function(cxcy){return sqrt(2/(1+cxcy))});azimuthalEqualAreaRaw.invert=azimuthalInvert(function(z){return 2*asin(z/2)});function azimuthalEqualArea(){return projection(azimuthalEqualAreaRaw).scale(124.75).clipAngle(180-.001)}var azimuthalEquidistantRaw=azimuthalRaw(function(c){return(c=acos(c))&&c/sin(c)});azimuthalEquidistantRaw.invert=azimuthalInvert(function(z){return z});function azimuthalEquidistant(){return projection(azimuthalEquidistantRaw).scale(79.4188).clipAngle(180-.001)}function mercatorRaw(lambda,phi){return[lambda,log(tan((halfPi+phi)/2))]}mercatorRaw.invert=function(x,y){return[x,2*atan(exp(y))-halfPi]};function mercator(){return mercatorProjection(mercatorRaw).scale(961/tau)}function mercatorProjection(project){var m=projection(project),center=m.center,scale=m.scale,translate=m.translate,clipExtent=m.clipExtent,x0=null,y0,x1,y1;// clip extent
m.scale=function(_){return arguments.length?(scale(_),reclip()):scale()};m.translate=function(_){return arguments.length?(translate(_),reclip()):translate()};m.center=function(_){return arguments.length?(center(_),reclip()):center()};m.clipExtent=function(_){return arguments.length?(_==null?x0=y0=x1=y1=null:(x0=+_[0][0],y0=+_[0][1],x1=+_[1][0],y1=+_[1][1]),reclip()):x0==null?null:[[x0,y0],[x1,y1]]};function reclip(){var k=pi*scale(),t=m(rotation(m.rotate()).invert([0,0]));return clipExtent(x0==null?[[t[0]-k,t[1]-k],[t[0]+k,t[1]+k]]:project===mercatorRaw?[[Math.max(t[0]-k,x0),y0],[Math.min(t[0]+k,x1),y1]]:[[x0,Math.max(t[1]-k,y0)],[x1,Math.min(t[1]+k,y1)]])}return reclip()}function tany(y){return tan((halfPi+y)/2)}function conicConformalRaw(y0,y1){var cy0=cos(y0),n=y0===y1?sin(y0):log(cy0/cos(y1))/log(tany(y1)/tany(y0)),f=cy0*pow(tany(y0),n)/n;if(!n)return mercatorRaw;function project(x,y){if(f>0){if(y<-halfPi+epsilon)y=-halfPi+epsilon}else{if(y>halfPi-epsilon)y=halfPi-epsilon}var r=f/pow(tany(y),n);return[r*sin(n*x),f-r*cos(n*x)]}project.invert=function(x,y){var fy=f-y,r=sign(n)*sqrt(x*x+fy*fy);return[atan2(x,abs(fy))/n*sign(fy),2*atan(pow(f/r,1/n))-halfPi]};return project}function conicConformal(){return conicProjection(conicConformalRaw).scale(109.5).parallels([30,30])}function equirectangularRaw(lambda,phi){return[lambda,phi]}equirectangularRaw.invert=equirectangularRaw;function equirectangular(){return projection(equirectangularRaw).scale(152.63)}function conicEquidistantRaw(y0,y1){var cy0=cos(y0),n=y0===y1?sin(y0):(cy0-cos(y1))/(y1-y0),g=cy0/n+y0;if(abs(n)<epsilon)return equirectangularRaw;function project(x,y){var gy=g-y,nx=n*x;return[gy*sin(nx),g-gy*cos(nx)]}project.invert=function(x,y){var gy=g-y;return[atan2(x,abs(gy))/n*sign(gy),g-sign(n)*sqrt(x*x+gy*gy)]};return project}function conicEquidistant(){return conicProjection(conicEquidistantRaw).scale(131.154).center([0,13.9389])}var A1=1.340264,A2=-.081106,A3=893e-6,A4=.003796,M=sqrt(3)/2,iterations=12;function equalEarthRaw(lambda,phi){var l=asin(M*sin(phi)),l2=l*l,l6=l2*l2*l2;return[lambda*cos(l)/(M*(A1+3*A2*l2+l6*(7*A3+9*A4*l2))),l*(A1+A2*l2+l6*(A3+A4*l2))]}equalEarthRaw.invert=function(x,y){var l=y,l2=l*l,l6=l2*l2*l2;for(var i=0,delta,fy,fpy;i<iterations;++i){fy=l*(A1+A2*l2+l6*(A3+A4*l2))-y;fpy=A1+3*A2*l2+l6*(7*A3+9*A4*l2);l-=delta=fy/fpy,l2=l*l,l6=l2*l2*l2;if(abs(delta)<epsilon2)break}return[M*x*(A1+3*A2*l2+l6*(7*A3+9*A4*l2))/cos(l),asin(sin(l)/M)]};function equalEarth(){return projection(equalEarthRaw).scale(177.158)}function gnomonicRaw(x,y){var cy=cos(y),k=cos(x)*cy;return[cy*sin(x)/k,sin(y)/k]}gnomonicRaw.invert=azimuthalInvert(atan);function gnomonic(){return projection(gnomonicRaw).scale(144.049).clipAngle(60)}function scaleTranslate$1(kx,ky,tx,ty){return kx===1&&ky===1&&tx===0&&ty===0?identity:transformer({point:function(x,y){this.stream.point(x*kx+tx,y*ky+ty)}})}function identity$1(){var k=1,tx=0,ty=0,sx=1,sy=1,transform=identity,// scale, translate and reflect
x0=null,y0,x1,y1,// clip extent
postclip=identity,cache,cacheStream,projection;function reset(){cache=cacheStream=null;return projection}return projection={stream:function(stream){return cache&&cacheStream===stream?cache:cache=transform(postclip(cacheStream=stream))},postclip:function(_){return arguments.length?(postclip=_,x0=y0=x1=y1=null,reset()):postclip},clipExtent:function(_){return arguments.length?(postclip=_==null?(x0=y0=x1=y1=null,identity):clipRectangle(x0=+_[0][0],y0=+_[0][1],x1=+_[1][0],y1=+_[1][1]),reset()):x0==null?null:[[x0,y0],[x1,y1]]},scale:function(_){return arguments.length?(transform=scaleTranslate$1((k=+_)*sx,k*sy,tx,ty),reset()):k},translate:function(_){return arguments.length?(transform=scaleTranslate$1(k*sx,k*sy,tx=+_[0],ty=+_[1]),reset()):[tx,ty]},reflectX:function(_){return arguments.length?(transform=scaleTranslate$1(k*(sx=_?-1:1),k*sy,tx,ty),reset()):sx<0},reflectY:function(_){return arguments.length?(transform=scaleTranslate$1(k*sx,k*(sy=_?-1:1),tx,ty),reset()):sy<0},fitExtent:function(extent,object){return fitExtent(projection,extent,object)},fitSize:function(size,object){return fitSize(projection,size,object)},fitWidth:function(width,object){return fitWidth(projection,width,object)},fitHeight:function(height,object){return fitHeight(projection,height,object)}}}function naturalEarth1Raw(lambda,phi){var phi2=phi*phi,phi4=phi2*phi2;return[lambda*(.8707-.131979*phi2+phi4*(-.013791+phi4*(.003971*phi2-.001529*phi4))),phi*(1.007226+phi2*(.015085+phi4*(-.044475+.028874*phi2-.005916*phi4)))]}naturalEarth1Raw.invert=function(x,y){var phi=y,i=25,delta;do{var phi2=phi*phi,phi4=phi2*phi2;phi-=delta=(phi*(1.007226+phi2*(.015085+phi4*(-.044475+.028874*phi2-.005916*phi4)))-y)/(1.007226+phi2*(.015085*3+phi4*(-.044475*7+.028874*9*phi2-.005916*11*phi4)))}while(abs(delta)>epsilon&&--i>0);return[x/(.8707+(phi2=phi*phi)*(-.131979+phi2*(-.013791+phi2*phi2*phi2*(.003971-.001529*phi2)))),phi]};function naturalEarth1(){return projection(naturalEarth1Raw).scale(175.295)}function orthographicRaw(x,y){return[cos(y)*sin(x),sin(y)]}orthographicRaw.invert=azimuthalInvert(asin);function orthographic(){return projection(orthographicRaw).scale(249.5).clipAngle(90+epsilon)}function stereographicRaw(x,y){var cy=cos(y),k=1+cos(x)*cy;return[cy*sin(x)/k,sin(y)/k]}stereographicRaw.invert=azimuthalInvert(function(z){return 2*atan(z)});function stereographic(){return projection(stereographicRaw).scale(250).clipAngle(142)}function transverseMercatorRaw(lambda,phi){return[log(tan((halfPi+phi)/2)),-lambda]}transverseMercatorRaw.invert=function(x,y){return[-y,2*atan(exp(x))-halfPi]};function transverseMercator(){var m=mercatorProjection(transverseMercatorRaw),center=m.center,rotate=m.rotate;m.center=function(_){return arguments.length?center([-_[1],_[0]]):(_=center(),[_[1],-_[0]])};m.rotate=function(_){return arguments.length?rotate([_[0],_[1],_.length>2?_[2]+90:90]):(_=rotate(),[_[0],_[1],_[2]-90])};return rotate([0,0,90]).scale(159.155)}exports.geoAlbers=albers;exports.geoAlbersUsa=albersUsa;exports.geoArea=area;exports.geoAzimuthalEqualArea=azimuthalEqualArea;exports.geoAzimuthalEqualAreaRaw=azimuthalEqualAreaRaw;exports.geoAzimuthalEquidistant=azimuthalEquidistant;exports.geoAzimuthalEquidistantRaw=azimuthalEquidistantRaw;exports.geoBounds=bounds;exports.geoCentroid=centroid;exports.geoCircle=circle;exports.geoClipAntimeridian=clipAntimeridian;exports.geoClipCircle=clipCircle;exports.geoClipExtent=extent;exports.geoClipRectangle=clipRectangle;exports.geoConicConformal=conicConformal;exports.geoConicConformalRaw=conicConformalRaw;exports.geoConicEqualArea=conicEqualArea;exports.geoConicEqualAreaRaw=conicEqualAreaRaw;exports.geoConicEquidistant=conicEquidistant;exports.geoConicEquidistantRaw=conicEquidistantRaw;exports.geoContains=contains;exports.geoDistance=distance;exports.geoEqualEarth=equalEarth;exports.geoEqualEarthRaw=equalEarthRaw;exports.geoEquirectangular=equirectangular;exports.geoEquirectangularRaw=equirectangularRaw;exports.geoGnomonic=gnomonic;exports.geoGnomonicRaw=gnomonicRaw;exports.geoGraticule=graticule;exports.geoGraticule10=graticule10;exports.geoIdentity=identity$1;exports.geoInterpolate=interpolate;exports.geoLength=length;exports.geoMercator=mercator;exports.geoMercatorRaw=mercatorRaw;exports.geoNaturalEarth1=naturalEarth1;exports.geoNaturalEarth1Raw=naturalEarth1Raw;exports.geoOrthographic=orthographic;exports.geoOrthographicRaw=orthographicRaw;exports.geoPath=index;exports.geoProjection=projection;exports.geoProjectionMutator=projectionMutator;exports.geoRotation=rotation;exports.geoStereographic=stereographic;exports.geoStereographicRaw=stereographicRaw;exports.geoStream=geoStream;exports.geoTransform=transform;exports.geoTransverseMercator=transverseMercator;exports.geoTransverseMercatorRaw=transverseMercatorRaw;Object.defineProperty(exports,"__esModule",{value:true})})},{"d3-array":29}],44:[function(require,module,exports){
// https://d3js.org/d3-hierarchy/ v1.1.9 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):(global=global||self,factory(global.d3=global.d3||{}))})(this,function(exports){"use strict";function defaultSeparation(a,b){return a.parent===b.parent?1:2}function meanX(children){return children.reduce(meanXReduce,0)/children.length}function meanXReduce(x,c){return x+c.x}function maxY(children){return 1+children.reduce(maxYReduce,0)}function maxYReduce(y,c){return Math.max(y,c.y)}function leafLeft(node){var children;while(children=node.children)node=children[0];return node}function leafRight(node){var children;while(children=node.children)node=children[children.length-1];return node}function cluster(){var separation=defaultSeparation,dx=1,dy=1,nodeSize=false;function cluster(root){var previousNode,x=0;
// First walk, computing the initial x & y values.
root.eachAfter(function(node){var children=node.children;if(children){node.x=meanX(children);node.y=maxY(children)}else{node.x=previousNode?x+=separation(node,previousNode):0;node.y=0;previousNode=node}});var left=leafLeft(root),right=leafRight(root),x0=left.x-separation(left,right)/2,x1=right.x+separation(right,left)/2;
// Second walk, normalizing x & y to the desired size.
return root.eachAfter(nodeSize?function(node){node.x=(node.x-root.x)*dx;node.y=(root.y-node.y)*dy}:function(node){node.x=(node.x-x0)/(x1-x0)*dx;node.y=(1-(root.y?node.y/root.y:1))*dy})}cluster.separation=function(x){return arguments.length?(separation=x,cluster):separation};cluster.size=function(x){return arguments.length?(nodeSize=false,dx=+x[0],dy=+x[1],cluster):nodeSize?null:[dx,dy]};cluster.nodeSize=function(x){return arguments.length?(nodeSize=true,dx=+x[0],dy=+x[1],cluster):nodeSize?[dx,dy]:null};return cluster}function count(node){var sum=0,children=node.children,i=children&&children.length;if(!i)sum=1;else while(--i>=0)sum+=children[i].value;node.value=sum}function node_count(){return this.eachAfter(count)}function node_each(callback){var node=this,current,next=[node],children,i,n;do{current=next.reverse(),next=[];while(node=current.pop()){callback(node),children=node.children;if(children)for(i=0,n=children.length;i<n;++i){next.push(children[i])}}}while(next.length);return this}function node_eachBefore(callback){var node=this,nodes=[node],children,i;while(node=nodes.pop()){callback(node),children=node.children;if(children)for(i=children.length-1;i>=0;--i){nodes.push(children[i])}}return this}function node_eachAfter(callback){var node=this,nodes=[node],next=[],children,i,n;while(node=nodes.pop()){next.push(node),children=node.children;if(children)for(i=0,n=children.length;i<n;++i){nodes.push(children[i])}}while(node=next.pop()){callback(node)}return this}function node_sum(value){return this.eachAfter(function(node){var sum=+value(node.data)||0,children=node.children,i=children&&children.length;while(--i>=0)sum+=children[i].value;node.value=sum})}function node_sort(compare){return this.eachBefore(function(node){if(node.children){node.children.sort(compare)}})}function node_path(end){var start=this,ancestor=leastCommonAncestor(start,end),nodes=[start];while(start!==ancestor){start=start.parent;nodes.push(start)}var k=nodes.length;while(end!==ancestor){nodes.splice(k,0,end);end=end.parent}return nodes}function leastCommonAncestor(a,b){if(a===b)return a;var aNodes=a.ancestors(),bNodes=b.ancestors(),c=null;a=aNodes.pop();b=bNodes.pop();while(a===b){c=a;a=aNodes.pop();b=bNodes.pop()}return c}function node_ancestors(){var node=this,nodes=[node];while(node=node.parent){nodes.push(node)}return nodes}function node_descendants(){var nodes=[];this.each(function(node){nodes.push(node)});return nodes}function node_leaves(){var leaves=[];this.eachBefore(function(node){if(!node.children){leaves.push(node)}});return leaves}function node_links(){var root=this,links=[];root.each(function(node){if(node!==root){// Don’t include the root’s parent, if any.
links.push({source:node.parent,target:node})}});return links}function hierarchy(data,children){var root=new Node(data),valued=+data.value&&(root.value=data.value),node,nodes=[root],child,childs,i,n;if(children==null)children=defaultChildren;while(node=nodes.pop()){if(valued)node.value=+node.data.value;if((childs=children(node.data))&&(n=childs.length)){node.children=new Array(n);for(i=n-1;i>=0;--i){nodes.push(child=node.children[i]=new Node(childs[i]));child.parent=node;child.depth=node.depth+1}}}return root.eachBefore(computeHeight)}function node_copy(){return hierarchy(this).eachBefore(copyData)}function defaultChildren(d){return d.children}function copyData(node){node.data=node.data.data}function computeHeight(node){var height=0;do{node.height=height}while((node=node.parent)&&node.height<++height)}function Node(data){this.data=data;this.depth=this.height=0;this.parent=null}Node.prototype=hierarchy.prototype={constructor:Node,count:node_count,each:node_each,eachAfter:node_eachAfter,eachBefore:node_eachBefore,sum:node_sum,sort:node_sort,path:node_path,ancestors:node_ancestors,descendants:node_descendants,leaves:node_leaves,links:node_links,copy:node_copy};var slice=Array.prototype.slice;function shuffle(array){var m=array.length,t,i;while(m){i=Math.random()*m--|0;t=array[m];array[m]=array[i];array[i]=t}return array}function enclose(circles){var i=0,n=(circles=shuffle(slice.call(circles))).length,B=[],p,e;while(i<n){p=circles[i];if(e&&enclosesWeak(e,p))++i;else e=encloseBasis(B=extendBasis(B,p)),i=0}return e}function extendBasis(B,p){var i,j;if(enclosesWeakAll(p,B))return[p];
// If we get here then B must have at least one element.
for(i=0;i<B.length;++i){if(enclosesNot(p,B[i])&&enclosesWeakAll(encloseBasis2(B[i],p),B)){return[B[i],p]}}
// If we get here then B must have at least two elements.
for(i=0;i<B.length-1;++i){for(j=i+1;j<B.length;++j){if(enclosesNot(encloseBasis2(B[i],B[j]),p)&&enclosesNot(encloseBasis2(B[i],p),B[j])&&enclosesNot(encloseBasis2(B[j],p),B[i])&&enclosesWeakAll(encloseBasis3(B[i],B[j],p),B)){return[B[i],B[j],p]}}}
// If we get here then something is very wrong.
throw new Error}function enclosesNot(a,b){var dr=a.r-b.r,dx=b.x-a.x,dy=b.y-a.y;return dr<0||dr*dr<dx*dx+dy*dy}function enclosesWeak(a,b){var dr=a.r-b.r+1e-6,dx=b.x-a.x,dy=b.y-a.y;return dr>0&&dr*dr>dx*dx+dy*dy}function enclosesWeakAll(a,B){for(var i=0;i<B.length;++i){if(!enclosesWeak(a,B[i])){return false}}return true}function encloseBasis(B){switch(B.length){case 1:return encloseBasis1(B[0]);case 2:return encloseBasis2(B[0],B[1]);case 3:return encloseBasis3(B[0],B[1],B[2])}}function encloseBasis1(a){return{x:a.x,y:a.y,r:a.r}}function encloseBasis2(a,b){var x1=a.x,y1=a.y,r1=a.r,x2=b.x,y2=b.y,r2=b.r,x21=x2-x1,y21=y2-y1,r21=r2-r1,l=Math.sqrt(x21*x21+y21*y21);return{x:(x1+x2+x21/l*r21)/2,y:(y1+y2+y21/l*r21)/2,r:(l+r1+r2)/2}}function encloseBasis3(a,b,c){var x1=a.x,y1=a.y,r1=a.r,x2=b.x,y2=b.y,r2=b.r,x3=c.x,y3=c.y,r3=c.r,a2=x1-x2,a3=x1-x3,b2=y1-y2,b3=y1-y3,c2=r2-r1,c3=r3-r1,d1=x1*x1+y1*y1-r1*r1,d2=d1-x2*x2-y2*y2+r2*r2,d3=d1-x3*x3-y3*y3+r3*r3,ab=a3*b2-a2*b3,xa=(b2*d3-b3*d2)/(ab*2)-x1,xb=(b3*c2-b2*c3)/ab,ya=(a3*d2-a2*d3)/(ab*2)-y1,yb=(a2*c3-a3*c2)/ab,A=xb*xb+yb*yb-1,B=2*(r1+xa*xb+ya*yb),C=xa*xa+ya*ya-r1*r1,r=-(A?(B+Math.sqrt(B*B-4*A*C))/(2*A):C/B);return{x:x1+xa+xb*r,y:y1+ya+yb*r,r:r}}function place(b,a,c){var dx=b.x-a.x,x,a2,dy=b.y-a.y,y,b2,d2=dx*dx+dy*dy;if(d2){a2=a.r+c.r,a2*=a2;b2=b.r+c.r,b2*=b2;if(a2>b2){x=(d2+b2-a2)/(2*d2);y=Math.sqrt(Math.max(0,b2/d2-x*x));c.x=b.x-x*dx-y*dy;c.y=b.y-x*dy+y*dx}else{x=(d2+a2-b2)/(2*d2);y=Math.sqrt(Math.max(0,a2/d2-x*x));c.x=a.x+x*dx-y*dy;c.y=a.y+x*dy+y*dx}}else{c.x=a.x+c.r;c.y=a.y}}function intersects(a,b){var dr=a.r+b.r-1e-6,dx=b.x-a.x,dy=b.y-a.y;return dr>0&&dr*dr>dx*dx+dy*dy}function score(node){var a=node._,b=node.next._,ab=a.r+b.r,dx=(a.x*b.r+b.x*a.r)/ab,dy=(a.y*b.r+b.y*a.r)/ab;return dx*dx+dy*dy}function Node$1(circle){this._=circle;this.next=null;this.previous=null}function packEnclose(circles){if(!(n=circles.length))return 0;var a,b,c,n,aa,ca,i,j,k,sj,sk;
// Place the first circle.
a=circles[0],a.x=0,a.y=0;if(!(n>1))return a.r;
// Place the second circle.
b=circles[1],a.x=-b.r,b.x=a.r,b.y=0;if(!(n>2))return a.r+b.r;
// Place the third circle.
place(b,a,c=circles[2]);
// Initialize the front-chain using the first three circles a, b and c.
a=new Node$1(a),b=new Node$1(b),c=new Node$1(c);a.next=c.previous=b;b.next=a.previous=c;c.next=b.previous=a;
// Attempt to place each remaining circle…
pack:for(i=3;i<n;++i){place(a._,b._,c=circles[i]),c=new Node$1(c);
// Find the closest intersecting circle on the front-chain, if any.
// “Closeness” is determined by linear distance along the front-chain.
// “Ahead” or “behind” is likewise determined by linear distance.
j=b.next,k=a.previous,sj=b._.r,sk=a._.r;do{if(sj<=sk){if(intersects(j._,c._)){b=j,a.next=b,b.previous=a,--i;continue pack}sj+=j._.r,j=j.next}else{if(intersects(k._,c._)){a=k,a.next=b,b.previous=a,--i;continue pack}sk+=k._.r,k=k.previous}}while(j!==k.next);
// Success! Insert the new circle c between a and b.
c.previous=a,c.next=b,a.next=b.previous=b=c;
// Compute the new closest circle pair to the centroid.
aa=score(a);while((c=c.next)!==b){if((ca=score(c))<aa){a=c,aa=ca}}b=a.next}
// Compute the enclosing circle of the front chain.
a=[b._],c=b;while((c=c.next)!==b)a.push(c._);c=enclose(a);
// Translate the circles to put the enclosing circle around the origin.
for(i=0;i<n;++i)a=circles[i],a.x-=c.x,a.y-=c.y;return c.r}function siblings(circles){packEnclose(circles);return circles}function optional(f){return f==null?null:required(f)}function required(f){if(typeof f!=="function")throw new Error;return f}function constantZero(){return 0}function constant(x){return function(){return x}}function defaultRadius(d){return Math.sqrt(d.value)}function index(){var radius=null,dx=1,dy=1,padding=constantZero;function pack(root){root.x=dx/2,root.y=dy/2;if(radius){root.eachBefore(radiusLeaf(radius)).eachAfter(packChildren(padding,.5)).eachBefore(translateChild(1))}else{root.eachBefore(radiusLeaf(defaultRadius)).eachAfter(packChildren(constantZero,1)).eachAfter(packChildren(padding,root.r/Math.min(dx,dy))).eachBefore(translateChild(Math.min(dx,dy)/(2*root.r)))}return root}pack.radius=function(x){return arguments.length?(radius=optional(x),pack):radius};pack.size=function(x){return arguments.length?(dx=+x[0],dy=+x[1],pack):[dx,dy]};pack.padding=function(x){return arguments.length?(padding=typeof x==="function"?x:constant(+x),pack):padding};return pack}function radiusLeaf(radius){return function(node){if(!node.children){node.r=Math.max(0,+radius(node)||0)}}}function packChildren(padding,k){return function(node){if(children=node.children){var children,i,n=children.length,r=padding(node)*k||0,e;if(r)for(i=0;i<n;++i)children[i].r+=r;e=packEnclose(children);if(r)for(i=0;i<n;++i)children[i].r-=r;node.r=e+r}}}function translateChild(k){return function(node){var parent=node.parent;node.r*=k;if(parent){node.x=parent.x+k*node.x;node.y=parent.y+k*node.y}}}function roundNode(node){node.x0=Math.round(node.x0);node.y0=Math.round(node.y0);node.x1=Math.round(node.x1);node.y1=Math.round(node.y1)}function treemapDice(parent,x0,y0,x1,y1){var nodes=parent.children,node,i=-1,n=nodes.length,k=parent.value&&(x1-x0)/parent.value;while(++i<n){node=nodes[i],node.y0=y0,node.y1=y1;node.x0=x0,node.x1=x0+=node.value*k}}function partition(){var dx=1,dy=1,padding=0,round=false;function partition(root){var n=root.height+1;root.x0=root.y0=padding;root.x1=dx;root.y1=dy/n;root.eachBefore(positionNode(dy,n));if(round)root.eachBefore(roundNode);return root}function positionNode(dy,n){return function(node){if(node.children){treemapDice(node,node.x0,dy*(node.depth+1)/n,node.x1,dy*(node.depth+2)/n)}var x0=node.x0,y0=node.y0,x1=node.x1-padding,y1=node.y1-padding;if(x1<x0)x0=x1=(x0+x1)/2;if(y1<y0)y0=y1=(y0+y1)/2;node.x0=x0;node.y0=y0;node.x1=x1;node.y1=y1}}partition.round=function(x){return arguments.length?(round=!!x,partition):round};partition.size=function(x){return arguments.length?(dx=+x[0],dy=+x[1],partition):[dx,dy]};partition.padding=function(x){return arguments.length?(padding=+x,partition):padding};return partition}var keyPrefix="$",// Protect against keys like “__proto__”.
preroot={depth:-1},ambiguous={};function defaultId(d){return d.id}function defaultParentId(d){return d.parentId}function stratify(){var id=defaultId,parentId=defaultParentId;function stratify(data){var d,i,n=data.length,root,parent,node,nodes=new Array(n),nodeId,nodeKey,nodeByKey={};for(i=0;i<n;++i){d=data[i],node=nodes[i]=new Node(d);if((nodeId=id(d,i,data))!=null&&(nodeId+="")){nodeKey=keyPrefix+(node.id=nodeId);nodeByKey[nodeKey]=nodeKey in nodeByKey?ambiguous:node}}for(i=0;i<n;++i){node=nodes[i],nodeId=parentId(data[i],i,data);if(nodeId==null||!(nodeId+="")){if(root)throw new Error("multiple roots");root=node}else{parent=nodeByKey[keyPrefix+nodeId];if(!parent)throw new Error("missing: "+nodeId);if(parent===ambiguous)throw new Error("ambiguous: "+nodeId);if(parent.children)parent.children.push(node);else parent.children=[node];node.parent=parent}}if(!root)throw new Error("no root");root.parent=preroot;root.eachBefore(function(node){node.depth=node.parent.depth+1;--n}).eachBefore(computeHeight);root.parent=null;if(n>0)throw new Error("cycle");return root}stratify.id=function(x){return arguments.length?(id=required(x),stratify):id};stratify.parentId=function(x){return arguments.length?(parentId=required(x),stratify):parentId};return stratify}function defaultSeparation$1(a,b){return a.parent===b.parent?1:2}
// function radialSeparation(a, b) {
//   return (a.parent === b.parent ? 1 : 2) / a.depth;
// }
// This function is used to traverse the left contour of a subtree (or
// subforest). It returns the successor of v on this contour. This successor is
// either given by the leftmost child of v or by the thread of v. The function
// returns null if and only if v is on the highest level of its subtree.
function nextLeft(v){var children=v.children;return children?children[0]:v.t}
// This function works analogously to nextLeft.
function nextRight(v){var children=v.children;return children?children[children.length-1]:v.t}
// Shifts the current subtree rooted at w+. This is done by increasing
// prelim(w+) and mod(w+) by shift.
function moveSubtree(wm,wp,shift){var change=shift/(wp.i-wm.i);wp.c-=change;wp.s+=shift;wm.c+=change;wp.z+=shift;wp.m+=shift}
// All other shifts, applied to the smaller subtrees between w- and w+, are
// performed by this function. To prepare the shifts, we have to adjust
// change(w+), shift(w+), and change(w-).
function executeShifts(v){var shift=0,change=0,children=v.children,i=children.length,w;while(--i>=0){w=children[i];w.z+=shift;w.m+=shift;shift+=w.s+(change+=w.c)}}
// If vi-’s ancestor is a sibling of v, returns vi-’s ancestor. Otherwise,
// returns the specified (default) ancestor.
function nextAncestor(vim,v,ancestor){return vim.a.parent===v.parent?vim.a:ancestor}function TreeNode(node,i){this._=node;this.parent=null;this.children=null;this.A=null;// default ancestor
this.a=this;// ancestor
this.z=0;// prelim
this.m=0;// mod
this.c=0;// change
this.s=0;// shift
this.t=null;// thread
this.i=i;// number
}TreeNode.prototype=Object.create(Node.prototype);function treeRoot(root){var tree=new TreeNode(root,0),node,nodes=[tree],child,children,i,n;while(node=nodes.pop()){if(children=node._.children){node.children=new Array(n=children.length);for(i=n-1;i>=0;--i){nodes.push(child=node.children[i]=new TreeNode(children[i],i));child.parent=node}}}(tree.parent=new TreeNode(null,0)).children=[tree];return tree}
// Node-link tree diagram using the Reingold-Tilford "tidy" algorithm
function tree(){var separation=defaultSeparation$1,dx=1,dy=1,nodeSize=null;function tree(root){var t=treeRoot(root);
// Compute the layout using Buchheim et al.’s algorithm.
t.eachAfter(firstWalk),t.parent.m=-t.z;t.eachBefore(secondWalk);
// If a fixed node size is specified, scale x and y.
if(nodeSize)root.eachBefore(sizeNode);
// If a fixed tree size is specified, scale x and y based on the extent.
// Compute the left-most, right-most, and depth-most nodes for extents.
else{var left=root,right=root,bottom=root;root.eachBefore(function(node){if(node.x<left.x)left=node;if(node.x>right.x)right=node;if(node.depth>bottom.depth)bottom=node});var s=left===right?1:separation(left,right)/2,tx=s-left.x,kx=dx/(right.x+s+tx),ky=dy/(bottom.depth||1);root.eachBefore(function(node){node.x=(node.x+tx)*kx;node.y=node.depth*ky})}return root}
// Computes a preliminary x-coordinate for v. Before that, FIRST WALK is
// applied recursively to the children of v, as well as the function
// APPORTION. After spacing out the children by calling EXECUTE SHIFTS, the
// node v is placed to the midpoint of its outermost children.
function firstWalk(v){var children=v.children,siblings=v.parent.children,w=v.i?siblings[v.i-1]:null;if(children){executeShifts(v);var midpoint=(children[0].z+children[children.length-1].z)/2;if(w){v.z=w.z+separation(v._,w._);v.m=v.z-midpoint}else{v.z=midpoint}}else if(w){v.z=w.z+separation(v._,w._)}v.parent.A=apportion(v,w,v.parent.A||siblings[0])}
// Computes all real x-coordinates by summing up the modifiers recursively.
function secondWalk(v){v._.x=v.z+v.parent.m;v.m+=v.parent.m}
// The core of the algorithm. Here, a new subtree is combined with the
// previous subtrees. Threads are used to traverse the inside and outside
// contours of the left and right subtree up to the highest common level. The
// vertices used for the traversals are vi+, vi-, vo-, and vo+, where the
// superscript o means outside and i means inside, the subscript - means left
// subtree and + means right subtree. For summing up the modifiers along the
// contour, we use respective variables si+, si-, so-, and so+. Whenever two
// nodes of the inside contours conflict, we compute the left one of the
// greatest uncommon ancestors using the function ANCESTOR and call MOVE
// SUBTREE to shift the subtree and prepare the shifts of smaller subtrees.
// Finally, we add a new thread (if necessary).
function apportion(v,w,ancestor){if(w){var vip=v,vop=v,vim=w,vom=vip.parent.children[0],sip=vip.m,sop=vop.m,sim=vim.m,som=vom.m,shift;while(vim=nextRight(vim),vip=nextLeft(vip),vim&&vip){vom=nextLeft(vom);vop=nextRight(vop);vop.a=v;shift=vim.z+sim-vip.z-sip+separation(vim._,vip._);if(shift>0){moveSubtree(nextAncestor(vim,v,ancestor),v,shift);sip+=shift;sop+=shift}sim+=vim.m;sip+=vip.m;som+=vom.m;sop+=vop.m}if(vim&&!nextRight(vop)){vop.t=vim;vop.m+=sim-sop}if(vip&&!nextLeft(vom)){vom.t=vip;vom.m+=sip-som;ancestor=v}}return ancestor}function sizeNode(node){node.x*=dx;node.y=node.depth*dy}tree.separation=function(x){return arguments.length?(separation=x,tree):separation};tree.size=function(x){return arguments.length?(nodeSize=false,dx=+x[0],dy=+x[1],tree):nodeSize?null:[dx,dy]};tree.nodeSize=function(x){return arguments.length?(nodeSize=true,dx=+x[0],dy=+x[1],tree):nodeSize?[dx,dy]:null};return tree}function treemapSlice(parent,x0,y0,x1,y1){var nodes=parent.children,node,i=-1,n=nodes.length,k=parent.value&&(y1-y0)/parent.value;while(++i<n){node=nodes[i],node.x0=x0,node.x1=x1;node.y0=y0,node.y1=y0+=node.value*k}}var phi=(1+Math.sqrt(5))/2;function squarifyRatio(ratio,parent,x0,y0,x1,y1){var rows=[],nodes=parent.children,row,nodeValue,i0=0,i1=0,n=nodes.length,dx,dy,value=parent.value,sumValue,minValue,maxValue,newRatio,minRatio,alpha,beta;while(i0<n){dx=x1-x0,dy=y1-y0;
// Find the next non-empty node.
do{sumValue=nodes[i1++].value}while(!sumValue&&i1<n);minValue=maxValue=sumValue;alpha=Math.max(dy/dx,dx/dy)/(value*ratio);beta=sumValue*sumValue*alpha;minRatio=Math.max(maxValue/beta,beta/minValue);
// Keep adding nodes while the aspect ratio maintains or improves.
for(;i1<n;++i1){sumValue+=nodeValue=nodes[i1].value;if(nodeValue<minValue)minValue=nodeValue;if(nodeValue>maxValue)maxValue=nodeValue;beta=sumValue*sumValue*alpha;newRatio=Math.max(maxValue/beta,beta/minValue);if(newRatio>minRatio){sumValue-=nodeValue;break}minRatio=newRatio}
// Position and record the row orientation.
rows.push(row={value:sumValue,dice:dx<dy,children:nodes.slice(i0,i1)});if(row.dice)treemapDice(row,x0,y0,x1,value?y0+=dy*sumValue/value:y1);else treemapSlice(row,x0,y0,value?x0+=dx*sumValue/value:x1,y1);value-=sumValue,i0=i1}return rows}var squarify=function custom(ratio){function squarify(parent,x0,y0,x1,y1){squarifyRatio(ratio,parent,x0,y0,x1,y1)}squarify.ratio=function(x){return custom((x=+x)>1?x:1)};return squarify}(phi);function index$1(){var tile=squarify,round=false,dx=1,dy=1,paddingStack=[0],paddingInner=constantZero,paddingTop=constantZero,paddingRight=constantZero,paddingBottom=constantZero,paddingLeft=constantZero;function treemap(root){root.x0=root.y0=0;root.x1=dx;root.y1=dy;root.eachBefore(positionNode);paddingStack=[0];if(round)root.eachBefore(roundNode);return root}function positionNode(node){var p=paddingStack[node.depth],x0=node.x0+p,y0=node.y0+p,x1=node.x1-p,y1=node.y1-p;if(x1<x0)x0=x1=(x0+x1)/2;if(y1<y0)y0=y1=(y0+y1)/2;node.x0=x0;node.y0=y0;node.x1=x1;node.y1=y1;if(node.children){p=paddingStack[node.depth+1]=paddingInner(node)/2;x0+=paddingLeft(node)-p;y0+=paddingTop(node)-p;x1-=paddingRight(node)-p;y1-=paddingBottom(node)-p;if(x1<x0)x0=x1=(x0+x1)/2;if(y1<y0)y0=y1=(y0+y1)/2;tile(node,x0,y0,x1,y1)}}treemap.round=function(x){return arguments.length?(round=!!x,treemap):round};treemap.size=function(x){return arguments.length?(dx=+x[0],dy=+x[1],treemap):[dx,dy]};treemap.tile=function(x){return arguments.length?(tile=required(x),treemap):tile};treemap.padding=function(x){return arguments.length?treemap.paddingInner(x).paddingOuter(x):treemap.paddingInner()};treemap.paddingInner=function(x){return arguments.length?(paddingInner=typeof x==="function"?x:constant(+x),treemap):paddingInner};treemap.paddingOuter=function(x){return arguments.length?treemap.paddingTop(x).paddingRight(x).paddingBottom(x).paddingLeft(x):treemap.paddingTop()};treemap.paddingTop=function(x){return arguments.length?(paddingTop=typeof x==="function"?x:constant(+x),treemap):paddingTop};treemap.paddingRight=function(x){return arguments.length?(paddingRight=typeof x==="function"?x:constant(+x),treemap):paddingRight};treemap.paddingBottom=function(x){return arguments.length?(paddingBottom=typeof x==="function"?x:constant(+x),treemap):paddingBottom};treemap.paddingLeft=function(x){return arguments.length?(paddingLeft=typeof x==="function"?x:constant(+x),treemap):paddingLeft};return treemap}function binary(parent,x0,y0,x1,y1){var nodes=parent.children,i,n=nodes.length,sum,sums=new Array(n+1);for(sums[0]=sum=i=0;i<n;++i){sums[i+1]=sum+=nodes[i].value}partition(0,n,parent.value,x0,y0,x1,y1);function partition(i,j,value,x0,y0,x1,y1){if(i>=j-1){var node=nodes[i];node.x0=x0,node.y0=y0;node.x1=x1,node.y1=y1;return}var valueOffset=sums[i],valueTarget=value/2+valueOffset,k=i+1,hi=j-1;while(k<hi){var mid=k+hi>>>1;if(sums[mid]<valueTarget)k=mid+1;else hi=mid}if(valueTarget-sums[k-1]<sums[k]-valueTarget&&i+1<k)--k;var valueLeft=sums[k]-valueOffset,valueRight=value-valueLeft;if(x1-x0>y1-y0){var xk=(x0*valueRight+x1*valueLeft)/value;partition(i,k,valueLeft,x0,y0,xk,y1);partition(k,j,valueRight,xk,y0,x1,y1)}else{var yk=(y0*valueRight+y1*valueLeft)/value;partition(i,k,valueLeft,x0,y0,x1,yk);partition(k,j,valueRight,x0,yk,x1,y1)}}}function sliceDice(parent,x0,y0,x1,y1){(parent.depth&1?treemapSlice:treemapDice)(parent,x0,y0,x1,y1)}var resquarify=function custom(ratio){function resquarify(parent,x0,y0,x1,y1){if((rows=parent._squarify)&&rows.ratio===ratio){var rows,row,nodes,i,j=-1,n,m=rows.length,value=parent.value;while(++j<m){row=rows[j],nodes=row.children;for(i=row.value=0,n=nodes.length;i<n;++i)row.value+=nodes[i].value;if(row.dice)treemapDice(row,x0,y0,x1,y0+=(y1-y0)*row.value/value);else treemapSlice(row,x0,y0,x0+=(x1-x0)*row.value/value,y1);value-=row.value}}else{parent._squarify=rows=squarifyRatio(ratio,parent,x0,y0,x1,y1);rows.ratio=ratio}}resquarify.ratio=function(x){return custom((x=+x)>1?x:1)};return resquarify}(phi);exports.cluster=cluster;exports.hierarchy=hierarchy;exports.pack=index;exports.packEnclose=enclose;exports.packSiblings=siblings;exports.partition=partition;exports.stratify=stratify;exports.tree=tree;exports.treemap=index$1;exports.treemapBinary=binary;exports.treemapDice=treemapDice;exports.treemapResquarify=resquarify;exports.treemapSlice=treemapSlice;exports.treemapSliceDice=sliceDice;exports.treemapSquarify=squarify;Object.defineProperty(exports,"__esModule",{value:true})})},{}],45:[function(require,module,exports){
// https://d3js.org/d3-interpolate/ v1.4.0 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports,require("d3-color")):typeof define==="function"&&define.amd?define(["exports","d3-color"],factory):(global=global||self,factory(global.d3=global.d3||{},global.d3))})(this,function(exports,d3Color){"use strict";function basis(t1,v0,v1,v2,v3){var t2=t1*t1,t3=t2*t1;return((1-3*t1+3*t2-t3)*v0+(4-6*t2+3*t3)*v1+(1+3*t1+3*t2-3*t3)*v2+t3*v3)/6}function basis$1(values){var n=values.length-1;return function(t){var i=t<=0?t=0:t>=1?(t=1,n-1):Math.floor(t*n),v1=values[i],v2=values[i+1],v0=i>0?values[i-1]:2*v1-v2,v3=i<n-1?values[i+2]:2*v2-v1;return basis((t-i/n)*n,v0,v1,v2,v3)}}function basisClosed(values){var n=values.length;return function(t){var i=Math.floor(((t%=1)<0?++t:t)*n),v0=values[(i+n-1)%n],v1=values[i%n],v2=values[(i+1)%n],v3=values[(i+2)%n];return basis((t-i/n)*n,v0,v1,v2,v3)}}function constant(x){return function(){return x}}function linear(a,d){return function(t){return a+t*d}}function exponential(a,b,y){return a=Math.pow(a,y),b=Math.pow(b,y)-a,y=1/y,function(t){return Math.pow(a+t*b,y)}}function hue(a,b){var d=b-a;return d?linear(a,d>180||d<-180?d-360*Math.round(d/360):d):constant(isNaN(a)?b:a)}function gamma(y){return(y=+y)===1?nogamma:function(a,b){return b-a?exponential(a,b,y):constant(isNaN(a)?b:a)}}function nogamma(a,b){var d=b-a;return d?linear(a,d):constant(isNaN(a)?b:a)}var rgb=function rgbGamma(y){var color=gamma(y);function rgb(start,end){var r=color((start=d3Color.rgb(start)).r,(end=d3Color.rgb(end)).r),g=color(start.g,end.g),b=color(start.b,end.b),opacity=nogamma(start.opacity,end.opacity);return function(t){start.r=r(t);start.g=g(t);start.b=b(t);start.opacity=opacity(t);return start+""}}rgb.gamma=rgbGamma;return rgb}(1);function rgbSpline(spline){return function(colors){var n=colors.length,r=new Array(n),g=new Array(n),b=new Array(n),i,color;for(i=0;i<n;++i){color=d3Color.rgb(colors[i]);r[i]=color.r||0;g[i]=color.g||0;b[i]=color.b||0}r=spline(r);g=spline(g);b=spline(b);color.opacity=1;return function(t){color.r=r(t);color.g=g(t);color.b=b(t);return color+""}}}var rgbBasis=rgbSpline(basis$1);var rgbBasisClosed=rgbSpline(basisClosed);function numberArray(a,b){if(!b)b=[];var n=a?Math.min(b.length,a.length):0,c=b.slice(),i;return function(t){for(i=0;i<n;++i)c[i]=a[i]*(1-t)+b[i]*t;return c}}function isNumberArray(x){return ArrayBuffer.isView(x)&&!(x instanceof DataView)}function array(a,b){return(isNumberArray(b)?numberArray:genericArray)(a,b)}function genericArray(a,b){var nb=b?b.length:0,na=a?Math.min(nb,a.length):0,x=new Array(na),c=new Array(nb),i;for(i=0;i<na;++i)x[i]=value(a[i],b[i]);for(;i<nb;++i)c[i]=b[i];return function(t){for(i=0;i<na;++i)c[i]=x[i](t);return c}}function date(a,b){var d=new Date;return a=+a,b=+b,function(t){return d.setTime(a*(1-t)+b*t),d}}function number(a,b){return a=+a,b=+b,function(t){return a*(1-t)+b*t}}function object(a,b){var i={},c={},k;if(a===null||typeof a!=="object")a={};if(b===null||typeof b!=="object")b={};for(k in b){if(k in a){i[k]=value(a[k],b[k])}else{c[k]=b[k]}}return function(t){for(k in i)c[k]=i[k](t);return c}}var reA=/[-+]?(?:\d+\.?\d*|\.?\d+)(?:[eE][-+]?\d+)?/g,reB=new RegExp(reA.source,"g");function zero(b){return function(){return b}}function one(b){return function(t){return b(t)+""}}function string(a,b){var bi=reA.lastIndex=reB.lastIndex=0,// scan index for next number in b
am,// current match in a
bm,// current match in b
bs,// string preceding current number in b, if any
i=-1,// index in s
s=[],// string constants and placeholders
q=[];// number interpolators
// Coerce inputs to strings.
a=a+"",b=b+"";
// Interpolate pairs of numbers in a & b.
while((am=reA.exec(a))&&(bm=reB.exec(b))){if((bs=bm.index)>bi){// a string precedes the next number in b
bs=b.slice(bi,bs);if(s[i])s[i]+=bs;// coalesce with previous string
else s[++i]=bs}if((am=am[0])===(bm=bm[0])){// numbers in a & b match
if(s[i])s[i]+=bm;// coalesce with previous string
else s[++i]=bm}else{// interpolate non-matching numbers
s[++i]=null;q.push({i:i,x:number(am,bm)})}bi=reB.lastIndex}
// Add remains of b.
if(bi<b.length){bs=b.slice(bi);if(s[i])s[i]+=bs;// coalesce with previous string
else s[++i]=bs}
// Special optimization for only a single match.
// Otherwise, interpolate each of the numbers and rejoin the string.
return s.length<2?q[0]?one(q[0].x):zero(b):(b=q.length,function(t){for(var i=0,o;i<b;++i)s[(o=q[i]).i]=o.x(t);return s.join("")})}function value(a,b){var t=typeof b,c;return b==null||t==="boolean"?constant(b):(t==="number"?number:t==="string"?(c=d3Color.color(b))?(b=c,rgb):string:b instanceof d3Color.color?rgb:b instanceof Date?date:isNumberArray(b)?numberArray:Array.isArray(b)?genericArray:typeof b.valueOf!=="function"&&typeof b.toString!=="function"||isNaN(b)?object:number)(a,b)}function discrete(range){var n=range.length;return function(t){return range[Math.max(0,Math.min(n-1,Math.floor(t*n)))]}}function hue$1(a,b){var i=hue(+a,+b);return function(t){var x=i(t);return x-360*Math.floor(x/360)}}function round(a,b){return a=+a,b=+b,function(t){return Math.round(a*(1-t)+b*t)}}var degrees=180/Math.PI;var identity={translateX:0,translateY:0,rotate:0,skewX:0,scaleX:1,scaleY:1};function decompose(a,b,c,d,e,f){var scaleX,scaleY,skewX;if(scaleX=Math.sqrt(a*a+b*b))a/=scaleX,b/=scaleX;if(skewX=a*c+b*d)c-=a*skewX,d-=b*skewX;if(scaleY=Math.sqrt(c*c+d*d))c/=scaleY,d/=scaleY,skewX/=scaleY;if(a*d<b*c)a=-a,b=-b,skewX=-skewX,scaleX=-scaleX;return{translateX:e,translateY:f,rotate:Math.atan2(b,a)*degrees,skewX:Math.atan(skewX)*degrees,scaleX:scaleX,scaleY:scaleY}}var cssNode,cssRoot,cssView,svgNode;function parseCss(value){if(value==="none")return identity;if(!cssNode)cssNode=document.createElement("DIV"),cssRoot=document.documentElement,cssView=document.defaultView;cssNode.style.transform=value;value=cssView.getComputedStyle(cssRoot.appendChild(cssNode),null).getPropertyValue("transform");cssRoot.removeChild(cssNode);value=value.slice(7,-1).split(",");return decompose(+value[0],+value[1],+value[2],+value[3],+value[4],+value[5])}function parseSvg(value){if(value==null)return identity;if(!svgNode)svgNode=document.createElementNS("http://www.w3.org/2000/svg","g");svgNode.setAttribute("transform",value);if(!(value=svgNode.transform.baseVal.consolidate()))return identity;value=value.matrix;return decompose(value.a,value.b,value.c,value.d,value.e,value.f)}function interpolateTransform(parse,pxComma,pxParen,degParen){function pop(s){return s.length?s.pop()+" ":""}function translate(xa,ya,xb,yb,s,q){if(xa!==xb||ya!==yb){var i=s.push("translate(",null,pxComma,null,pxParen);q.push({i:i-4,x:number(xa,xb)},{i:i-2,x:number(ya,yb)})}else if(xb||yb){s.push("translate("+xb+pxComma+yb+pxParen)}}function rotate(a,b,s,q){if(a!==b){if(a-b>180)b+=360;else if(b-a>180)a+=360;// shortest path
q.push({i:s.push(pop(s)+"rotate(",null,degParen)-2,x:number(a,b)})}else if(b){s.push(pop(s)+"rotate("+b+degParen)}}function skewX(a,b,s,q){if(a!==b){q.push({i:s.push(pop(s)+"skewX(",null,degParen)-2,x:number(a,b)})}else if(b){s.push(pop(s)+"skewX("+b+degParen)}}function scale(xa,ya,xb,yb,s,q){if(xa!==xb||ya!==yb){var i=s.push(pop(s)+"scale(",null,",",null,")");q.push({i:i-4,x:number(xa,xb)},{i:i-2,x:number(ya,yb)})}else if(xb!==1||yb!==1){s.push(pop(s)+"scale("+xb+","+yb+")")}}return function(a,b){var s=[],// string constants and placeholders
q=[];// number interpolators
a=parse(a),b=parse(b);translate(a.translateX,a.translateY,b.translateX,b.translateY,s,q);rotate(a.rotate,b.rotate,s,q);skewX(a.skewX,b.skewX,s,q);scale(a.scaleX,a.scaleY,b.scaleX,b.scaleY,s,q);a=b=null;// gc
return function(t){var i=-1,n=q.length,o;while(++i<n)s[(o=q[i]).i]=o.x(t);return s.join("")}}}var interpolateTransformCss=interpolateTransform(parseCss,"px, ","px)","deg)");var interpolateTransformSvg=interpolateTransform(parseSvg,", ",")",")");var rho=Math.SQRT2,rho2=2,rho4=4,epsilon2=1e-12;function cosh(x){return((x=Math.exp(x))+1/x)/2}function sinh(x){return((x=Math.exp(x))-1/x)/2}function tanh(x){return((x=Math.exp(2*x))-1)/(x+1)}
// p0 = [ux0, uy0, w0]
// p1 = [ux1, uy1, w1]
function zoom(p0,p1){var ux0=p0[0],uy0=p0[1],w0=p0[2],ux1=p1[0],uy1=p1[1],w1=p1[2],dx=ux1-ux0,dy=uy1-uy0,d2=dx*dx+dy*dy,i,S;
// Special case for u0 ≅ u1.
if(d2<epsilon2){S=Math.log(w1/w0)/rho;i=function(t){return[ux0+t*dx,uy0+t*dy,w0*Math.exp(rho*t*S)]}}
// General case.
else{var d1=Math.sqrt(d2),b0=(w1*w1-w0*w0+rho4*d2)/(2*w0*rho2*d1),b1=(w1*w1-w0*w0-rho4*d2)/(2*w1*rho2*d1),r0=Math.log(Math.sqrt(b0*b0+1)-b0),r1=Math.log(Math.sqrt(b1*b1+1)-b1);S=(r1-r0)/rho;i=function(t){var s=t*S,coshr0=cosh(r0),u=w0/(rho2*d1)*(coshr0*tanh(rho*s+r0)-sinh(r0));return[ux0+u*dx,uy0+u*dy,w0*coshr0/cosh(rho*s+r0)]}}i.duration=S*1e3;return i}function hsl(hue){return function(start,end){var h=hue((start=d3Color.hsl(start)).h,(end=d3Color.hsl(end)).h),s=nogamma(start.s,end.s),l=nogamma(start.l,end.l),opacity=nogamma(start.opacity,end.opacity);return function(t){start.h=h(t);start.s=s(t);start.l=l(t);start.opacity=opacity(t);return start+""}}}var hsl$1=hsl(hue);var hslLong=hsl(nogamma);function lab(start,end){var l=nogamma((start=d3Color.lab(start)).l,(end=d3Color.lab(end)).l),a=nogamma(start.a,end.a),b=nogamma(start.b,end.b),opacity=nogamma(start.opacity,end.opacity);return function(t){start.l=l(t);start.a=a(t);start.b=b(t);start.opacity=opacity(t);return start+""}}function hcl(hue){return function(start,end){var h=hue((start=d3Color.hcl(start)).h,(end=d3Color.hcl(end)).h),c=nogamma(start.c,end.c),l=nogamma(start.l,end.l),opacity=nogamma(start.opacity,end.opacity);return function(t){start.h=h(t);start.c=c(t);start.l=l(t);start.opacity=opacity(t);return start+""}}}var hcl$1=hcl(hue);var hclLong=hcl(nogamma);function cubehelix(hue){return function cubehelixGamma(y){y=+y;function cubehelix(start,end){var h=hue((start=d3Color.cubehelix(start)).h,(end=d3Color.cubehelix(end)).h),s=nogamma(start.s,end.s),l=nogamma(start.l,end.l),opacity=nogamma(start.opacity,end.opacity);return function(t){start.h=h(t);start.s=s(t);start.l=l(Math.pow(t,y));start.opacity=opacity(t);return start+""}}cubehelix.gamma=cubehelixGamma;return cubehelix}(1)}var cubehelix$1=cubehelix(hue);var cubehelixLong=cubehelix(nogamma);function piecewise(interpolate,values){var i=0,n=values.length-1,v=values[0],I=new Array(n<0?0:n);while(i<n)I[i]=interpolate(v,v=values[++i]);return function(t){var i=Math.max(0,Math.min(n-1,Math.floor(t*=n)));return I[i](t-i)}}function quantize(interpolator,n){var samples=new Array(n);for(var i=0;i<n;++i)samples[i]=interpolator(i/(n-1));return samples}exports.interpolate=value;exports.interpolateArray=array;exports.interpolateBasis=basis$1;exports.interpolateBasisClosed=basisClosed;exports.interpolateCubehelix=cubehelix$1;exports.interpolateCubehelixLong=cubehelixLong;exports.interpolateDate=date;exports.interpolateDiscrete=discrete;exports.interpolateHcl=hcl$1;exports.interpolateHclLong=hclLong;exports.interpolateHsl=hsl$1;exports.interpolateHslLong=hslLong;exports.interpolateHue=hue$1;exports.interpolateLab=lab;exports.interpolateNumber=number;exports.interpolateNumberArray=numberArray;exports.interpolateObject=object;exports.interpolateRgb=rgb;exports.interpolateRgbBasis=rgbBasis;exports.interpolateRgbBasisClosed=rgbBasisClosed;exports.interpolateRound=round;exports.interpolateString=string;exports.interpolateTransformCss=interpolateTransformCss;exports.interpolateTransformSvg=interpolateTransformSvg;exports.interpolateZoom=zoom;exports.piecewise=piecewise;exports.quantize=quantize;Object.defineProperty(exports,"__esModule",{value:true})})},{"d3-color":34}],46:[function(require,module,exports){
// https://d3js.org/d3-path/ v1.0.9 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):(global=global||self,factory(global.d3=global.d3||{}))})(this,function(exports){"use strict";var pi=Math.PI,tau=2*pi,epsilon=1e-6,tauEpsilon=tau-epsilon;function Path(){this._x0=this._y0=// start of current subpath
this._x1=this._y1=null;// end of current subpath
this._=""}function path(){return new Path}Path.prototype=path.prototype={constructor:Path,moveTo:function(x,y){this._+="M"+(this._x0=this._x1=+x)+","+(this._y0=this._y1=+y)},closePath:function(){if(this._x1!==null){this._x1=this._x0,this._y1=this._y0;this._+="Z"}},lineTo:function(x,y){this._+="L"+(this._x1=+x)+","+(this._y1=+y)},quadraticCurveTo:function(x1,y1,x,y){this._+="Q"+ +x1+","+ +y1+","+(this._x1=+x)+","+(this._y1=+y)},bezierCurveTo:function(x1,y1,x2,y2,x,y){this._+="C"+ +x1+","+ +y1+","+ +x2+","+ +y2+","+(this._x1=+x)+","+(this._y1=+y)},arcTo:function(x1,y1,x2,y2,r){x1=+x1,y1=+y1,x2=+x2,y2=+y2,r=+r;var x0=this._x1,y0=this._y1,x21=x2-x1,y21=y2-y1,x01=x0-x1,y01=y0-y1,l01_2=x01*x01+y01*y01;
// Is the radius negative? Error.
if(r<0)throw new Error("negative radius: "+r);
// Is this path empty? Move to (x1,y1).
if(this._x1===null){this._+="M"+(this._x1=x1)+","+(this._y1=y1)}
// Or, is (x1,y1) coincident with (x0,y0)? Do nothing.
else if(!(l01_2>epsilon));else if(!(Math.abs(y01*x21-y21*x01)>epsilon)||!r){this._+="L"+(this._x1=x1)+","+(this._y1=y1)}
// Otherwise, draw an arc!
else{var x20=x2-x0,y20=y2-y0,l21_2=x21*x21+y21*y21,l20_2=x20*x20+y20*y20,l21=Math.sqrt(l21_2),l01=Math.sqrt(l01_2),l=r*Math.tan((pi-Math.acos((l21_2+l01_2-l20_2)/(2*l21*l01)))/2),t01=l/l01,t21=l/l21;
// If the start tangent is not coincident with (x0,y0), line to.
if(Math.abs(t01-1)>epsilon){this._+="L"+(x1+t01*x01)+","+(y1+t01*y01)}this._+="A"+r+","+r+",0,0,"+ +(y01*x20>x01*y20)+","+(this._x1=x1+t21*x21)+","+(this._y1=y1+t21*y21)}},arc:function(x,y,r,a0,a1,ccw){x=+x,y=+y,r=+r,ccw=!!ccw;var dx=r*Math.cos(a0),dy=r*Math.sin(a0),x0=x+dx,y0=y+dy,cw=1^ccw,da=ccw?a0-a1:a1-a0;
// Is the radius negative? Error.
if(r<0)throw new Error("negative radius: "+r);
// Is this path empty? Move to (x0,y0).
if(this._x1===null){this._+="M"+x0+","+y0}
// Or, is (x0,y0) not coincident with the previous point? Line to (x0,y0).
else if(Math.abs(this._x1-x0)>epsilon||Math.abs(this._y1-y0)>epsilon){this._+="L"+x0+","+y0}
// Is this arc empty? We’re done.
if(!r)return;
// Does the angle go the wrong way? Flip the direction.
if(da<0)da=da%tau+tau;
// Is this a complete circle? Draw two arcs to complete the circle.
if(da>tauEpsilon){this._+="A"+r+","+r+",0,1,"+cw+","+(x-dx)+","+(y-dy)+"A"+r+","+r+",0,1,"+cw+","+(this._x1=x0)+","+(this._y1=y0)}
// Is this arc non-empty? Draw an arc!
else if(da>epsilon){this._+="A"+r+","+r+",0,"+ +(da>=pi)+","+cw+","+(this._x1=x+r*Math.cos(a1))+","+(this._y1=y+r*Math.sin(a1))}},rect:function(x,y,w,h){this._+="M"+(this._x0=this._x1=+x)+","+(this._y0=this._y1=+y)+"h"+ +w+"v"+ +h+"h"+-w+"Z"},toString:function(){return this._}};exports.path=path;Object.defineProperty(exports,"__esModule",{value:true})})},{}],47:[function(require,module,exports){
// https://d3js.org/d3-polygon/ v1.0.6 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):(global=global||self,factory(global.d3=global.d3||{}))})(this,function(exports){"use strict";function area(polygon){var i=-1,n=polygon.length,a,b=polygon[n-1],area=0;while(++i<n){a=b;b=polygon[i];area+=a[1]*b[0]-a[0]*b[1]}return area/2}function centroid(polygon){var i=-1,n=polygon.length,x=0,y=0,a,b=polygon[n-1],c,k=0;while(++i<n){a=b;b=polygon[i];k+=c=a[0]*b[1]-b[0]*a[1];x+=(a[0]+b[0])*c;y+=(a[1]+b[1])*c}return k*=3,[x/k,y/k]}
// Returns the 2D cross product of AB and AC vectors, i.e., the z-component of
// the 3D cross product in a quadrant I Cartesian coordinate system (+x is
// right, +y is up). Returns a positive value if ABC is counter-clockwise,
// negative if clockwise, and zero if the points are collinear.
function cross(a,b,c){return(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])}function lexicographicOrder(a,b){return a[0]-b[0]||a[1]-b[1]}
// Computes the upper convex hull per the monotone chain algorithm.
// Assumes points.length >= 3, is sorted by x, unique in y.
// Returns an array of indices into points in left-to-right order.
function computeUpperHullIndexes(points){var n=points.length,indexes=[0,1],size=2;for(var i=2;i<n;++i){while(size>1&&cross(points[indexes[size-2]],points[indexes[size-1]],points[i])<=0)--size;indexes[size++]=i}return indexes.slice(0,size);// remove popped points
}function hull(points){if((n=points.length)<3)return null;var i,n,sortedPoints=new Array(n),flippedPoints=new Array(n);for(i=0;i<n;++i)sortedPoints[i]=[+points[i][0],+points[i][1],i];sortedPoints.sort(lexicographicOrder);for(i=0;i<n;++i)flippedPoints[i]=[sortedPoints[i][0],-sortedPoints[i][1]];var upperIndexes=computeUpperHullIndexes(sortedPoints),lowerIndexes=computeUpperHullIndexes(flippedPoints);
// Construct the hull polygon, removing possible duplicate endpoints.
var skipLeft=lowerIndexes[0]===upperIndexes[0],skipRight=lowerIndexes[lowerIndexes.length-1]===upperIndexes[upperIndexes.length-1],hull=[];
// Add upper hull in right-to-l order.
// Then add lower hull in left-to-right order.
for(i=upperIndexes.length-1;i>=0;--i)hull.push(points[sortedPoints[upperIndexes[i]][2]]);for(i=+skipLeft;i<lowerIndexes.length-skipRight;++i)hull.push(points[sortedPoints[lowerIndexes[i]][2]]);return hull}function contains(polygon,point){var n=polygon.length,p=polygon[n-1],x=point[0],y=point[1],x0=p[0],y0=p[1],x1,y1,inside=false;for(var i=0;i<n;++i){p=polygon[i],x1=p[0],y1=p[1];if(y1>y!==y0>y&&x<(x0-x1)*(y-y1)/(y0-y1)+x1)inside=!inside;x0=x1,y0=y1}return inside}function length(polygon){var i=-1,n=polygon.length,b=polygon[n-1],xa,ya,xb=b[0],yb=b[1],perimeter=0;while(++i<n){xa=xb;ya=yb;b=polygon[i];xb=b[0];yb=b[1];xa-=xb;ya-=yb;perimeter+=Math.sqrt(xa*xa+ya*ya)}return perimeter}exports.polygonArea=area;exports.polygonCentroid=centroid;exports.polygonContains=contains;exports.polygonHull=hull;exports.polygonLength=length;Object.defineProperty(exports,"__esModule",{value:true})})},{}],48:[function(require,module,exports){
// https://d3js.org/d3-quadtree/ v1.0.7 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):(global=global||self,factory(global.d3=global.d3||{}))})(this,function(exports){"use strict";function tree_add(d){var x=+this._x.call(null,d),y=+this._y.call(null,d);return add(this.cover(x,y),x,y,d)}function add(tree,x,y,d){if(isNaN(x)||isNaN(y))return tree;// ignore invalid points
var parent,node=tree._root,leaf={data:d},x0=tree._x0,y0=tree._y0,x1=tree._x1,y1=tree._y1,xm,ym,xp,yp,right,bottom,i,j;
// If the tree is empty, initialize the root as a leaf.
if(!node)return tree._root=leaf,tree;
// Find the existing leaf for the new point, or add it.
while(node.length){if(right=x>=(xm=(x0+x1)/2))x0=xm;else x1=xm;if(bottom=y>=(ym=(y0+y1)/2))y0=ym;else y1=ym;if(parent=node,!(node=node[i=bottom<<1|right]))return parent[i]=leaf,tree}
// Is the new point is exactly coincident with the existing point?
xp=+tree._x.call(null,node.data);yp=+tree._y.call(null,node.data);if(x===xp&&y===yp)return leaf.next=node,parent?parent[i]=leaf:tree._root=leaf,tree;
// Otherwise, split the leaf node until the old and new point are separated.
do{parent=parent?parent[i]=new Array(4):tree._root=new Array(4);if(right=x>=(xm=(x0+x1)/2))x0=xm;else x1=xm;if(bottom=y>=(ym=(y0+y1)/2))y0=ym;else y1=ym}while((i=bottom<<1|right)===(j=(yp>=ym)<<1|xp>=xm));return parent[j]=node,parent[i]=leaf,tree}function addAll(data){var d,i,n=data.length,x,y,xz=new Array(n),yz=new Array(n),x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
// Compute the points and their extent.
for(i=0;i<n;++i){if(isNaN(x=+this._x.call(null,d=data[i]))||isNaN(y=+this._y.call(null,d)))continue;xz[i]=x;yz[i]=y;if(x<x0)x0=x;if(x>x1)x1=x;if(y<y0)y0=y;if(y>y1)y1=y}
// If there were no (valid) points, abort.
if(x0>x1||y0>y1)return this;
// Expand the tree to cover the new points.
this.cover(x0,y0).cover(x1,y1);
// Add the new points.
for(i=0;i<n;++i){add(this,xz[i],yz[i],data[i])}return this}function tree_cover(x,y){if(isNaN(x=+x)||isNaN(y=+y))return this;// ignore invalid points
var x0=this._x0,y0=this._y0,x1=this._x1,y1=this._y1;
// If the quadtree has no extent, initialize them.
// Integer extent are necessary so that if we later double the extent,
// the existing quadrant boundaries don’t change due to floating point error!
if(isNaN(x0)){x1=(x0=Math.floor(x))+1;y1=(y0=Math.floor(y))+1}
// Otherwise, double repeatedly to cover.
else{var z=x1-x0,node=this._root,parent,i;while(x0>x||x>=x1||y0>y||y>=y1){i=(y<y0)<<1|x<x0;parent=new Array(4),parent[i]=node,node=parent,z*=2;switch(i){case 0:x1=x0+z,y1=y0+z;break;case 1:x0=x1-z,y1=y0+z;break;case 2:x1=x0+z,y0=y1-z;break;case 3:x0=x1-z,y0=y1-z;break}}if(this._root&&this._root.length)this._root=node}this._x0=x0;this._y0=y0;this._x1=x1;this._y1=y1;return this}function tree_data(){var data=[];this.visit(function(node){if(!node.length)do{data.push(node.data)}while(node=node.next)});return data}function tree_extent(_){return arguments.length?this.cover(+_[0][0],+_[0][1]).cover(+_[1][0],+_[1][1]):isNaN(this._x0)?undefined:[[this._x0,this._y0],[this._x1,this._y1]]}function Quad(node,x0,y0,x1,y1){this.node=node;this.x0=x0;this.y0=y0;this.x1=x1;this.y1=y1}function tree_find(x,y,radius){var data,x0=this._x0,y0=this._y0,x1,y1,x2,y2,x3=this._x1,y3=this._y1,quads=[],node=this._root,q,i;if(node)quads.push(new Quad(node,x0,y0,x3,y3));if(radius==null)radius=Infinity;else{x0=x-radius,y0=y-radius;x3=x+radius,y3=y+radius;radius*=radius}while(q=quads.pop()){
// Stop searching if this quadrant can’t contain a closer node.
if(!(node=q.node)||(x1=q.x0)>x3||(y1=q.y0)>y3||(x2=q.x1)<x0||(y2=q.y1)<y0)continue;
// Bisect the current quadrant.
if(node.length){var xm=(x1+x2)/2,ym=(y1+y2)/2;quads.push(new Quad(node[3],xm,ym,x2,y2),new Quad(node[2],x1,ym,xm,y2),new Quad(node[1],xm,y1,x2,ym),new Quad(node[0],x1,y1,xm,ym));
// Visit the closest quadrant first.
if(i=(y>=ym)<<1|x>=xm){q=quads[quads.length-1];quads[quads.length-1]=quads[quads.length-1-i];quads[quads.length-1-i]=q}}
// Visit this point. (Visiting coincident points isn’t necessary!)
else{var dx=x-+this._x.call(null,node.data),dy=y-+this._y.call(null,node.data),d2=dx*dx+dy*dy;if(d2<radius){var d=Math.sqrt(radius=d2);x0=x-d,y0=y-d;x3=x+d,y3=y+d;data=node.data}}}return data}function tree_remove(d){if(isNaN(x=+this._x.call(null,d))||isNaN(y=+this._y.call(null,d)))return this;// ignore invalid points
var parent,node=this._root,retainer,previous,next,x0=this._x0,y0=this._y0,x1=this._x1,y1=this._y1,x,y,xm,ym,right,bottom,i,j;
// If the tree is empty, initialize the root as a leaf.
if(!node)return this;
// Find the leaf node for the point.
// While descending, also retain the deepest parent with a non-removed sibling.
if(node.length)while(true){if(right=x>=(xm=(x0+x1)/2))x0=xm;else x1=xm;if(bottom=y>=(ym=(y0+y1)/2))y0=ym;else y1=ym;if(!(parent=node,node=node[i=bottom<<1|right]))return this;if(!node.length)break;if(parent[i+1&3]||parent[i+2&3]||parent[i+3&3])retainer=parent,j=i}
// Find the point to remove.
while(node.data!==d)if(!(previous=node,node=node.next))return this;if(next=node.next)delete node.next;
// If there are multiple coincident points, remove just the point.
if(previous)return next?previous.next=next:delete previous.next,this;
// If this is the root point, remove it.
if(!parent)return this._root=next,this;
// Remove this leaf.
next?parent[i]=next:delete parent[i];
// If the parent now contains exactly one leaf, collapse superfluous parents.
if((node=parent[0]||parent[1]||parent[2]||parent[3])&&node===(parent[3]||parent[2]||parent[1]||parent[0])&&!node.length){if(retainer)retainer[j]=node;else this._root=node}return this}function removeAll(data){for(var i=0,n=data.length;i<n;++i)this.remove(data[i]);return this}function tree_root(){return this._root}function tree_size(){var size=0;this.visit(function(node){if(!node.length)do{++size}while(node=node.next)});return size}function tree_visit(callback){var quads=[],q,node=this._root,child,x0,y0,x1,y1;if(node)quads.push(new Quad(node,this._x0,this._y0,this._x1,this._y1));while(q=quads.pop()){if(!callback(node=q.node,x0=q.x0,y0=q.y0,x1=q.x1,y1=q.y1)&&node.length){var xm=(x0+x1)/2,ym=(y0+y1)/2;if(child=node[3])quads.push(new Quad(child,xm,ym,x1,y1));if(child=node[2])quads.push(new Quad(child,x0,ym,xm,y1));if(child=node[1])quads.push(new Quad(child,xm,y0,x1,ym));if(child=node[0])quads.push(new Quad(child,x0,y0,xm,ym))}}return this}function tree_visitAfter(callback){var quads=[],next=[],q;if(this._root)quads.push(new Quad(this._root,this._x0,this._y0,this._x1,this._y1));while(q=quads.pop()){var node=q.node;if(node.length){var child,x0=q.x0,y0=q.y0,x1=q.x1,y1=q.y1,xm=(x0+x1)/2,ym=(y0+y1)/2;if(child=node[0])quads.push(new Quad(child,x0,y0,xm,ym));if(child=node[1])quads.push(new Quad(child,xm,y0,x1,ym));if(child=node[2])quads.push(new Quad(child,x0,ym,xm,y1));if(child=node[3])quads.push(new Quad(child,xm,ym,x1,y1))}next.push(q)}while(q=next.pop()){callback(q.node,q.x0,q.y0,q.x1,q.y1)}return this}function defaultX(d){return d[0]}function tree_x(_){return arguments.length?(this._x=_,this):this._x}function defaultY(d){return d[1]}function tree_y(_){return arguments.length?(this._y=_,this):this._y}function quadtree(nodes,x,y){var tree=new Quadtree(x==null?defaultX:x,y==null?defaultY:y,NaN,NaN,NaN,NaN);return nodes==null?tree:tree.addAll(nodes)}function Quadtree(x,y,x0,y0,x1,y1){this._x=x;this._y=y;this._x0=x0;this._y0=y0;this._x1=x1;this._y1=y1;this._root=undefined}function leaf_copy(leaf){var copy={data:leaf.data},next=copy;while(leaf=leaf.next)next=next.next={data:leaf.data};return copy}var treeProto=quadtree.prototype=Quadtree.prototype;treeProto.copy=function(){var copy=new Quadtree(this._x,this._y,this._x0,this._y0,this._x1,this._y1),node=this._root,nodes,child;if(!node)return copy;if(!node.length)return copy._root=leaf_copy(node),copy;nodes=[{source:node,target:copy._root=new Array(4)}];while(node=nodes.pop()){for(var i=0;i<4;++i){if(child=node.source[i]){if(child.length)nodes.push({source:child,target:node.target[i]=new Array(4)});else node.target[i]=leaf_copy(child)}}}return copy};treeProto.add=tree_add;treeProto.addAll=addAll;treeProto.cover=tree_cover;treeProto.data=tree_data;treeProto.extent=tree_extent;treeProto.find=tree_find;treeProto.remove=tree_remove;treeProto.removeAll=removeAll;treeProto.root=tree_root;treeProto.size=tree_size;treeProto.visit=tree_visit;treeProto.visitAfter=tree_visitAfter;treeProto.x=tree_x;treeProto.y=tree_y;exports.quadtree=quadtree;Object.defineProperty(exports,"__esModule",{value:true})})},{}],49:[function(require,module,exports){
// https://d3js.org/d3-random/ v1.1.2 Copyright 2018 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):factory(global.d3=global.d3||{})})(this,function(exports){"use strict";function defaultSource(){return Math.random()}var uniform=function sourceRandomUniform(source){function randomUniform(min,max){min=min==null?0:+min;max=max==null?1:+max;if(arguments.length===1)max=min,min=0;else max-=min;return function(){return source()*max+min}}randomUniform.source=sourceRandomUniform;return randomUniform}(defaultSource);var normal=function sourceRandomNormal(source){function randomNormal(mu,sigma){var x,r;mu=mu==null?0:+mu;sigma=sigma==null?1:+sigma;return function(){var y;
// If available, use the second previously-generated uniform random.
if(x!=null)y=x,x=null;
// Otherwise, generate a new x and y.
else do{x=source()*2-1;y=source()*2-1;r=x*x+y*y}while(!r||r>1);return mu+sigma*y*Math.sqrt(-2*Math.log(r)/r)}}randomNormal.source=sourceRandomNormal;return randomNormal}(defaultSource);var logNormal=function sourceRandomLogNormal(source){function randomLogNormal(){var randomNormal=normal.source(source).apply(this,arguments);return function(){return Math.exp(randomNormal())}}randomLogNormal.source=sourceRandomLogNormal;return randomLogNormal}(defaultSource);var irwinHall=function sourceRandomIrwinHall(source){function randomIrwinHall(n){return function(){for(var sum=0,i=0;i<n;++i)sum+=source();return sum}}randomIrwinHall.source=sourceRandomIrwinHall;return randomIrwinHall}(defaultSource);var bates=function sourceRandomBates(source){function randomBates(n){var randomIrwinHall=irwinHall.source(source)(n);return function(){return randomIrwinHall()/n}}randomBates.source=sourceRandomBates;return randomBates}(defaultSource);var exponential=function sourceRandomExponential(source){function randomExponential(lambda){return function(){return-Math.log(1-source())/lambda}}randomExponential.source=sourceRandomExponential;return randomExponential}(defaultSource);exports.randomUniform=uniform;exports.randomNormal=normal;exports.randomLogNormal=logNormal;exports.randomBates=bates;exports.randomIrwinHall=irwinHall;exports.randomExponential=exponential;Object.defineProperty(exports,"__esModule",{value:true})})},{}],50:[function(require,module,exports){
// https://d3js.org/d3-scale-chromatic/ v1.5.0 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports,require("d3-interpolate"),require("d3-color")):typeof define==="function"&&define.amd?define(["exports","d3-interpolate","d3-color"],factory):(global=global||self,factory(global.d3=global.d3||{},global.d3,global.d3))})(this,function(exports,d3Interpolate,d3Color){"use strict";function colors(specifier){var n=specifier.length/6|0,colors=new Array(n),i=0;while(i<n)colors[i]="#"+specifier.slice(i*6,++i*6);return colors}var category10=colors("1f77b4ff7f0e2ca02cd627289467bd8c564be377c27f7f7fbcbd2217becf");var Accent=colors("7fc97fbeaed4fdc086ffff99386cb0f0027fbf5b17666666");var Dark2=colors("1b9e77d95f027570b3e7298a66a61ee6ab02a6761d666666");var Paired=colors("a6cee31f78b4b2df8a33a02cfb9a99e31a1cfdbf6fff7f00cab2d66a3d9affff99b15928");var Pastel1=colors("fbb4aeb3cde3ccebc5decbe4fed9a6ffffcce5d8bdfddaecf2f2f2");var Pastel2=colors("b3e2cdfdcdaccbd5e8f4cae4e6f5c9fff2aef1e2cccccccc");var Set1=colors("e41a1c377eb84daf4a984ea3ff7f00ffff33a65628f781bf999999");var Set2=colors("66c2a5fc8d628da0cbe78ac3a6d854ffd92fe5c494b3b3b3");var Set3=colors("8dd3c7ffffb3bebadafb807280b1d3fdb462b3de69fccde5d9d9d9bc80bdccebc5ffed6f");var Tableau10=colors("4e79a7f28e2ce1575976b7b259a14fedc949af7aa1ff9da79c755fbab0ab");function ramp(scheme){return d3Interpolate.interpolateRgbBasis(scheme[scheme.length-1])}var scheme=new Array(3).concat("d8b365f5f5f55ab4ac","a6611adfc27d80cdc1018571","a6611adfc27df5f5f580cdc1018571","8c510ad8b365f6e8c3c7eae55ab4ac01665e","8c510ad8b365f6e8c3f5f5f5c7eae55ab4ac01665e","8c510abf812ddfc27df6e8c3c7eae580cdc135978f01665e","8c510abf812ddfc27df6e8c3f5f5f5c7eae580cdc135978f01665e","5430058c510abf812ddfc27df6e8c3c7eae580cdc135978f01665e003c30","5430058c510abf812ddfc27df6e8c3f5f5f5c7eae580cdc135978f01665e003c30").map(colors);var BrBG=ramp(scheme);var scheme$1=new Array(3).concat("af8dc3f7f7f77fbf7b","7b3294c2a5cfa6dba0008837","7b3294c2a5cff7f7f7a6dba0008837","762a83af8dc3e7d4e8d9f0d37fbf7b1b7837","762a83af8dc3e7d4e8f7f7f7d9f0d37fbf7b1b7837","762a839970abc2a5cfe7d4e8d9f0d3a6dba05aae611b7837","762a839970abc2a5cfe7d4e8f7f7f7d9f0d3a6dba05aae611b7837","40004b762a839970abc2a5cfe7d4e8d9f0d3a6dba05aae611b783700441b","40004b762a839970abc2a5cfe7d4e8f7f7f7d9f0d3a6dba05aae611b783700441b").map(colors);var PRGn=ramp(scheme$1);var scheme$2=new Array(3).concat("e9a3c9f7f7f7a1d76a","d01c8bf1b6dab8e1864dac26","d01c8bf1b6daf7f7f7b8e1864dac26","c51b7de9a3c9fde0efe6f5d0a1d76a4d9221","c51b7de9a3c9fde0eff7f7f7e6f5d0a1d76a4d9221","c51b7dde77aef1b6dafde0efe6f5d0b8e1867fbc414d9221","c51b7dde77aef1b6dafde0eff7f7f7e6f5d0b8e1867fbc414d9221","8e0152c51b7dde77aef1b6dafde0efe6f5d0b8e1867fbc414d9221276419","8e0152c51b7dde77aef1b6dafde0eff7f7f7e6f5d0b8e1867fbc414d9221276419").map(colors);var PiYG=ramp(scheme$2);var scheme$3=new Array(3).concat("998ec3f7f7f7f1a340","5e3c99b2abd2fdb863e66101","5e3c99b2abd2f7f7f7fdb863e66101","542788998ec3d8daebfee0b6f1a340b35806","542788998ec3d8daebf7f7f7fee0b6f1a340b35806","5427888073acb2abd2d8daebfee0b6fdb863e08214b35806","5427888073acb2abd2d8daebf7f7f7fee0b6fdb863e08214b35806","2d004b5427888073acb2abd2d8daebfee0b6fdb863e08214b358067f3b08","2d004b5427888073acb2abd2d8daebf7f7f7fee0b6fdb863e08214b358067f3b08").map(colors);var PuOr=ramp(scheme$3);var scheme$4=new Array(3).concat("ef8a62f7f7f767a9cf","ca0020f4a58292c5de0571b0","ca0020f4a582f7f7f792c5de0571b0","b2182bef8a62fddbc7d1e5f067a9cf2166ac","b2182bef8a62fddbc7f7f7f7d1e5f067a9cf2166ac","b2182bd6604df4a582fddbc7d1e5f092c5de4393c32166ac","b2182bd6604df4a582fddbc7f7f7f7d1e5f092c5de4393c32166ac","67001fb2182bd6604df4a582fddbc7d1e5f092c5de4393c32166ac053061","67001fb2182bd6604df4a582fddbc7f7f7f7d1e5f092c5de4393c32166ac053061").map(colors);var RdBu=ramp(scheme$4);var scheme$5=new Array(3).concat("ef8a62ffffff999999","ca0020f4a582bababa404040","ca0020f4a582ffffffbababa404040","b2182bef8a62fddbc7e0e0e09999994d4d4d","b2182bef8a62fddbc7ffffffe0e0e09999994d4d4d","b2182bd6604df4a582fddbc7e0e0e0bababa8787874d4d4d","b2182bd6604df4a582fddbc7ffffffe0e0e0bababa8787874d4d4d","67001fb2182bd6604df4a582fddbc7e0e0e0bababa8787874d4d4d1a1a1a","67001fb2182bd6604df4a582fddbc7ffffffe0e0e0bababa8787874d4d4d1a1a1a").map(colors);var RdGy=ramp(scheme$5);var scheme$6=new Array(3).concat("fc8d59ffffbf91bfdb","d7191cfdae61abd9e92c7bb6","d7191cfdae61ffffbfabd9e92c7bb6","d73027fc8d59fee090e0f3f891bfdb4575b4","d73027fc8d59fee090ffffbfe0f3f891bfdb4575b4","d73027f46d43fdae61fee090e0f3f8abd9e974add14575b4","d73027f46d43fdae61fee090ffffbfe0f3f8abd9e974add14575b4","a50026d73027f46d43fdae61fee090e0f3f8abd9e974add14575b4313695","a50026d73027f46d43fdae61fee090ffffbfe0f3f8abd9e974add14575b4313695").map(colors);var RdYlBu=ramp(scheme$6);var scheme$7=new Array(3).concat("fc8d59ffffbf91cf60","d7191cfdae61a6d96a1a9641","d7191cfdae61ffffbfa6d96a1a9641","d73027fc8d59fee08bd9ef8b91cf601a9850","d73027fc8d59fee08bffffbfd9ef8b91cf601a9850","d73027f46d43fdae61fee08bd9ef8ba6d96a66bd631a9850","d73027f46d43fdae61fee08bffffbfd9ef8ba6d96a66bd631a9850","a50026d73027f46d43fdae61fee08bd9ef8ba6d96a66bd631a9850006837","a50026d73027f46d43fdae61fee08bffffbfd9ef8ba6d96a66bd631a9850006837").map(colors);var RdYlGn=ramp(scheme$7);var scheme$8=new Array(3).concat("fc8d59ffffbf99d594","d7191cfdae61abdda42b83ba","d7191cfdae61ffffbfabdda42b83ba","d53e4ffc8d59fee08be6f59899d5943288bd","d53e4ffc8d59fee08bffffbfe6f59899d5943288bd","d53e4ff46d43fdae61fee08be6f598abdda466c2a53288bd","d53e4ff46d43fdae61fee08bffffbfe6f598abdda466c2a53288bd","9e0142d53e4ff46d43fdae61fee08be6f598abdda466c2a53288bd5e4fa2","9e0142d53e4ff46d43fdae61fee08bffffbfe6f598abdda466c2a53288bd5e4fa2").map(colors);var Spectral=ramp(scheme$8);var scheme$9=new Array(3).concat("e5f5f999d8c92ca25f","edf8fbb2e2e266c2a4238b45","edf8fbb2e2e266c2a42ca25f006d2c","edf8fbccece699d8c966c2a42ca25f006d2c","edf8fbccece699d8c966c2a441ae76238b45005824","f7fcfde5f5f9ccece699d8c966c2a441ae76238b45005824","f7fcfde5f5f9ccece699d8c966c2a441ae76238b45006d2c00441b").map(colors);var BuGn=ramp(scheme$9);var scheme$a=new Array(3).concat("e0ecf49ebcda8856a7","edf8fbb3cde38c96c688419d","edf8fbb3cde38c96c68856a7810f7c","edf8fbbfd3e69ebcda8c96c68856a7810f7c","edf8fbbfd3e69ebcda8c96c68c6bb188419d6e016b","f7fcfde0ecf4bfd3e69ebcda8c96c68c6bb188419d6e016b","f7fcfde0ecf4bfd3e69ebcda8c96c68c6bb188419d810f7c4d004b").map(colors);var BuPu=ramp(scheme$a);var scheme$b=new Array(3).concat("e0f3dba8ddb543a2ca","f0f9e8bae4bc7bccc42b8cbe","f0f9e8bae4bc7bccc443a2ca0868ac","f0f9e8ccebc5a8ddb57bccc443a2ca0868ac","f0f9e8ccebc5a8ddb57bccc44eb3d32b8cbe08589e","f7fcf0e0f3dbccebc5a8ddb57bccc44eb3d32b8cbe08589e","f7fcf0e0f3dbccebc5a8ddb57bccc44eb3d32b8cbe0868ac084081").map(colors);var GnBu=ramp(scheme$b);var scheme$c=new Array(3).concat("fee8c8fdbb84e34a33","fef0d9fdcc8afc8d59d7301f","fef0d9fdcc8afc8d59e34a33b30000","fef0d9fdd49efdbb84fc8d59e34a33b30000","fef0d9fdd49efdbb84fc8d59ef6548d7301f990000","fff7ecfee8c8fdd49efdbb84fc8d59ef6548d7301f990000","fff7ecfee8c8fdd49efdbb84fc8d59ef6548d7301fb300007f0000").map(colors);var OrRd=ramp(scheme$c);var scheme$d=new Array(3).concat("ece2f0a6bddb1c9099","f6eff7bdc9e167a9cf02818a","f6eff7bdc9e167a9cf1c9099016c59","f6eff7d0d1e6a6bddb67a9cf1c9099016c59","f6eff7d0d1e6a6bddb67a9cf3690c002818a016450","fff7fbece2f0d0d1e6a6bddb67a9cf3690c002818a016450","fff7fbece2f0d0d1e6a6bddb67a9cf3690c002818a016c59014636").map(colors);var PuBuGn=ramp(scheme$d);var scheme$e=new Array(3).concat("ece7f2a6bddb2b8cbe","f1eef6bdc9e174a9cf0570b0","f1eef6bdc9e174a9cf2b8cbe045a8d","f1eef6d0d1e6a6bddb74a9cf2b8cbe045a8d","f1eef6d0d1e6a6bddb74a9cf3690c00570b0034e7b","fff7fbece7f2d0d1e6a6bddb74a9cf3690c00570b0034e7b","fff7fbece7f2d0d1e6a6bddb74a9cf3690c00570b0045a8d023858").map(colors);var PuBu=ramp(scheme$e);var scheme$f=new Array(3).concat("e7e1efc994c7dd1c77","f1eef6d7b5d8df65b0ce1256","f1eef6d7b5d8df65b0dd1c77980043","f1eef6d4b9dac994c7df65b0dd1c77980043","f1eef6d4b9dac994c7df65b0e7298ace125691003f","f7f4f9e7e1efd4b9dac994c7df65b0e7298ace125691003f","f7f4f9e7e1efd4b9dac994c7df65b0e7298ace125698004367001f").map(colors);var PuRd=ramp(scheme$f);var scheme$g=new Array(3).concat("fde0ddfa9fb5c51b8a","feebe2fbb4b9f768a1ae017e","feebe2fbb4b9f768a1c51b8a7a0177","feebe2fcc5c0fa9fb5f768a1c51b8a7a0177","feebe2fcc5c0fa9fb5f768a1dd3497ae017e7a0177","fff7f3fde0ddfcc5c0fa9fb5f768a1dd3497ae017e7a0177","fff7f3fde0ddfcc5c0fa9fb5f768a1dd3497ae017e7a017749006a").map(colors);var RdPu=ramp(scheme$g);var scheme$h=new Array(3).concat("edf8b17fcdbb2c7fb8","ffffcca1dab441b6c4225ea8","ffffcca1dab441b6c42c7fb8253494","ffffccc7e9b47fcdbb41b6c42c7fb8253494","ffffccc7e9b47fcdbb41b6c41d91c0225ea80c2c84","ffffd9edf8b1c7e9b47fcdbb41b6c41d91c0225ea80c2c84","ffffd9edf8b1c7e9b47fcdbb41b6c41d91c0225ea8253494081d58").map(colors);var YlGnBu=ramp(scheme$h);var scheme$i=new Array(3).concat("f7fcb9addd8e31a354","ffffccc2e69978c679238443","ffffccc2e69978c67931a354006837","ffffccd9f0a3addd8e78c67931a354006837","ffffccd9f0a3addd8e78c67941ab5d238443005a32","ffffe5f7fcb9d9f0a3addd8e78c67941ab5d238443005a32","ffffe5f7fcb9d9f0a3addd8e78c67941ab5d238443006837004529").map(colors);var YlGn=ramp(scheme$i);var scheme$j=new Array(3).concat("fff7bcfec44fd95f0e","ffffd4fed98efe9929cc4c02","ffffd4fed98efe9929d95f0e993404","ffffd4fee391fec44ffe9929d95f0e993404","ffffd4fee391fec44ffe9929ec7014cc4c028c2d04","ffffe5fff7bcfee391fec44ffe9929ec7014cc4c028c2d04","ffffe5fff7bcfee391fec44ffe9929ec7014cc4c02993404662506").map(colors);var YlOrBr=ramp(scheme$j);var scheme$k=new Array(3).concat("ffeda0feb24cf03b20","ffffb2fecc5cfd8d3ce31a1c","ffffb2fecc5cfd8d3cf03b20bd0026","ffffb2fed976feb24cfd8d3cf03b20bd0026","ffffb2fed976feb24cfd8d3cfc4e2ae31a1cb10026","ffffccffeda0fed976feb24cfd8d3cfc4e2ae31a1cb10026","ffffccffeda0fed976feb24cfd8d3cfc4e2ae31a1cbd0026800026").map(colors);var YlOrRd=ramp(scheme$k);var scheme$l=new Array(3).concat("deebf79ecae13182bd","eff3ffbdd7e76baed62171b5","eff3ffbdd7e76baed63182bd08519c","eff3ffc6dbef9ecae16baed63182bd08519c","eff3ffc6dbef9ecae16baed64292c62171b5084594","f7fbffdeebf7c6dbef9ecae16baed64292c62171b5084594","f7fbffdeebf7c6dbef9ecae16baed64292c62171b508519c08306b").map(colors);var Blues=ramp(scheme$l);var scheme$m=new Array(3).concat("e5f5e0a1d99b31a354","edf8e9bae4b374c476238b45","edf8e9bae4b374c47631a354006d2c","edf8e9c7e9c0a1d99b74c47631a354006d2c","edf8e9c7e9c0a1d99b74c47641ab5d238b45005a32","f7fcf5e5f5e0c7e9c0a1d99b74c47641ab5d238b45005a32","f7fcf5e5f5e0c7e9c0a1d99b74c47641ab5d238b45006d2c00441b").map(colors);var Greens=ramp(scheme$m);var scheme$n=new Array(3).concat("f0f0f0bdbdbd636363","f7f7f7cccccc969696525252","f7f7f7cccccc969696636363252525","f7f7f7d9d9d9bdbdbd969696636363252525","f7f7f7d9d9d9bdbdbd969696737373525252252525","fffffff0f0f0d9d9d9bdbdbd969696737373525252252525","fffffff0f0f0d9d9d9bdbdbd969696737373525252252525000000").map(colors);var Greys=ramp(scheme$n);var scheme$o=new Array(3).concat("efedf5bcbddc756bb1","f2f0f7cbc9e29e9ac86a51a3","f2f0f7cbc9e29e9ac8756bb154278f","f2f0f7dadaebbcbddc9e9ac8756bb154278f","f2f0f7dadaebbcbddc9e9ac8807dba6a51a34a1486","fcfbfdefedf5dadaebbcbddc9e9ac8807dba6a51a34a1486","fcfbfdefedf5dadaebbcbddc9e9ac8807dba6a51a354278f3f007d").map(colors);var Purples=ramp(scheme$o);var scheme$p=new Array(3).concat("fee0d2fc9272de2d26","fee5d9fcae91fb6a4acb181d","fee5d9fcae91fb6a4ade2d26a50f15","fee5d9fcbba1fc9272fb6a4ade2d26a50f15","fee5d9fcbba1fc9272fb6a4aef3b2ccb181d99000d","fff5f0fee0d2fcbba1fc9272fb6a4aef3b2ccb181d99000d","fff5f0fee0d2fcbba1fc9272fb6a4aef3b2ccb181da50f1567000d").map(colors);var Reds=ramp(scheme$p);var scheme$q=new Array(3).concat("fee6cefdae6be6550d","feeddefdbe85fd8d3cd94701","feeddefdbe85fd8d3ce6550da63603","feeddefdd0a2fdae6bfd8d3ce6550da63603","feeddefdd0a2fdae6bfd8d3cf16913d948018c2d04","fff5ebfee6cefdd0a2fdae6bfd8d3cf16913d948018c2d04","fff5ebfee6cefdd0a2fdae6bfd8d3cf16913d94801a636037f2704").map(colors);var Oranges=ramp(scheme$q);function cividis(t){t=Math.max(0,Math.min(1,t));return"rgb("+Math.max(0,Math.min(255,Math.round(-4.54-t*(35.34-t*(2381.73-t*(6402.7-t*(7024.72-t*2710.57)))))))+", "+Math.max(0,Math.min(255,Math.round(32.49+t*(170.73+t*(52.82-t*(131.46-t*(176.58-t*67.37)))))))+", "+Math.max(0,Math.min(255,Math.round(81.24+t*(442.36-t*(2482.43-t*(6167.24-t*(6614.94-t*2475.67)))))))+")"}var cubehelix=d3Interpolate.interpolateCubehelixLong(d3Color.cubehelix(300,.5,0),d3Color.cubehelix(-240,.5,1));var warm=d3Interpolate.interpolateCubehelixLong(d3Color.cubehelix(-100,.75,.35),d3Color.cubehelix(80,1.5,.8));var cool=d3Interpolate.interpolateCubehelixLong(d3Color.cubehelix(260,.75,.35),d3Color.cubehelix(80,1.5,.8));var c=d3Color.cubehelix();function rainbow(t){if(t<0||t>1)t-=Math.floor(t);var ts=Math.abs(t-.5);c.h=360*t-100;c.s=1.5-1.5*ts;c.l=.8-.9*ts;return c+""}var c$1=d3Color.rgb(),pi_1_3=Math.PI/3,pi_2_3=Math.PI*2/3;function sinebow(t){var x;t=(.5-t)*Math.PI;c$1.r=255*(x=Math.sin(t))*x;c$1.g=255*(x=Math.sin(t+pi_1_3))*x;c$1.b=255*(x=Math.sin(t+pi_2_3))*x;return c$1+""}function turbo(t){t=Math.max(0,Math.min(1,t));return"rgb("+Math.max(0,Math.min(255,Math.round(34.61+t*(1172.33-t*(10793.56-t*(33300.12-t*(38394.49-t*14825.05)))))))+", "+Math.max(0,Math.min(255,Math.round(23.31+t*(557.33+t*(1225.33-t*(3574.96-t*(1073.77+t*707.56)))))))+", "+Math.max(0,Math.min(255,Math.round(27.2+t*(3211.1-t*(15327.97-t*(27814-t*(22569.18-t*6838.66)))))))+")"}function ramp$1(range){var n=range.length;return function(t){return range[Math.max(0,Math.min(n-1,Math.floor(t*n)))]}}var viridis=ramp$1(colors("44015444025645045745055946075a46085c460a5d460b5e470d60470e6147106347116447136548146748166848176948186a481a6c481b6d481c6e481d6f481f70482071482173482374482475482576482677482878482979472a7a472c7a472d7b472e7c472f7d46307e46327e46337f463480453581453781453882443983443a83443b84433d84433e85423f854240864241864142874144874045884046883f47883f48893e49893e4a893e4c8a3d4d8a3d4e8a3c4f8a3c508b3b518b3b528b3a538b3a548c39558c39568c38588c38598c375a8c375b8d365c8d365d8d355e8d355f8d34608d34618d33628d33638d32648e32658e31668e31678e31688e30698e306a8e2f6b8e2f6c8e2e6d8e2e6e8e2e6f8e2d708e2d718e2c718e2c728e2c738e2b748e2b758e2a768e2a778e2a788e29798e297a8e297b8e287c8e287d8e277e8e277f8e27808e26818e26828e26828e25838e25848e25858e24868e24878e23888e23898e238a8d228b8d228c8d228d8d218e8d218f8d21908d21918c20928c20928c20938c1f948c1f958b1f968b1f978b1f988b1f998a1f9a8a1e9b8a1e9c891e9d891f9e891f9f881fa0881fa1881fa1871fa28720a38620a48621a58521a68522a78522a88423a98324aa8325ab8225ac8226ad8127ad8128ae8029af7f2ab07f2cb17e2db27d2eb37c2fb47c31b57b32b67a34b67935b77937b87838b9773aba763bbb753dbc743fbc7340bd7242be7144bf7046c06f48c16e4ac16d4cc26c4ec36b50c46a52c56954c56856c66758c7655ac8645cc8635ec96260ca6063cb5f65cb5e67cc5c69cd5b6ccd5a6ece5870cf5773d05675d05477d1537ad1517cd2507fd34e81d34d84d44b86d54989d5488bd6468ed64590d74393d74195d84098d83e9bd93c9dd93ba0da39a2da37a5db36a8db34aadc32addc30b0dd2fb2dd2db5de2bb8de29bade28bddf26c0df25c2df23c5e021c8e020cae11fcde11dd0e11cd2e21bd5e21ad8e219dae319dde318dfe318e2e418e5e419e7e419eae51aece51befe51cf1e51df4e61ef6e620f8e621fbe723fde725"));var magma=ramp$1(colors("00000401000501010601010802010902020b02020d03030f03031204041405041606051806051a07061c08071e0907200a08220b09240c09260d0a290e0b2b100b2d110c2f120d31130d34140e36150e38160f3b180f3d19103f1a10421c10441d11471e114920114b21114e22115024125325125527125829115a2a115c2c115f2d11612f116331116533106734106936106b38106c390f6e3b0f703d0f713f0f72400f74420f75440f764510774710784910784a10794c117a4e117b4f127b51127c52137c54137d56147d57157e59157e5a167e5c167f5d177f5f187f601880621980641a80651a80671b80681c816a1c816b1d816d1d816e1e81701f81721f817320817521817621817822817922827b23827c23827e24828025828125818326818426818627818827818928818b29818c29818e2a81902a81912b81932b80942c80962c80982d80992d809b2e7f9c2e7f9e2f7fa02f7fa1307ea3307ea5317ea6317da8327daa337dab337cad347cae347bb0357bb2357bb3367ab5367ab73779b83779ba3878bc3978bd3977bf3a77c03a76c23b75c43c75c53c74c73d73c83e73ca3e72cc3f71cd4071cf4070d0416fd2426fd3436ed5446dd6456cd8456cd9466bdb476adc4869de4968df4a68e04c67e24d66e34e65e44f64e55064e75263e85362e95462ea5661eb5760ec5860ed5a5fee5b5eef5d5ef05f5ef1605df2625df2645cf3655cf4675cf4695cf56b5cf66c5cf66e5cf7705cf7725cf8745cf8765cf9785df9795df97b5dfa7d5efa7f5efa815ffb835ffb8560fb8761fc8961fc8a62fc8c63fc8e64fc9065fd9266fd9467fd9668fd9869fd9a6afd9b6bfe9d6cfe9f6dfea16efea36ffea571fea772fea973feaa74feac76feae77feb078feb27afeb47bfeb67cfeb77efeb97ffebb81febd82febf84fec185fec287fec488fec68afec88cfeca8dfecc8ffecd90fecf92fed194fed395fed597fed799fed89afdda9cfddc9efddea0fde0a1fde2a3fde3a5fde5a7fde7a9fde9aafdebacfcecaefceeb0fcf0b2fcf2b4fcf4b6fcf6b8fcf7b9fcf9bbfcfbbdfcfdbf"));var inferno=ramp$1(colors("00000401000501010601010802010a02020c02020e03021004031204031405041706041907051b08051d09061f0a07220b07240c08260d08290e092b10092d110a30120a32140b34150b37160b39180c3c190c3e1b0c411c0c431e0c451f0c48210c4a230c4c240c4f260c51280b53290b552b0b572d0b592f0a5b310a5c320a5e340a5f3609613809623909633b09643d09653e0966400a67420a68440a68450a69470b6a490b6a4a0c6b4c0c6b4d0d6c4f0d6c510e6c520e6d540f6d550f6d57106e59106e5a116e5c126e5d126e5f136e61136e62146e64156e65156e67166e69166e6a176e6c186e6d186e6f196e71196e721a6e741a6e751b6e771c6d781c6d7a1d6d7c1d6d7d1e6d7f1e6c801f6c82206c84206b85216b87216b88226a8a226a8c23698d23698f24699025689225689326679526679727669827669a28659b29649d29649f2a63a02a63a22b62a32c61a52c60a62d60a82e5fa92e5eab2f5ead305dae305cb0315bb1325ab3325ab43359b63458b73557b93556ba3655bc3754bd3853bf3952c03a51c13a50c33b4fc43c4ec63d4dc73e4cc83f4bca404acb4149cc4248ce4347cf4446d04545d24644d34743d44842d54a41d74b3fd84c3ed94d3dda4e3cdb503bdd513ade5238df5337e05536e15635e25734e35933e45a31e55c30e65d2fe75e2ee8602de9612bea632aeb6429eb6628ec6726ed6925ee6a24ef6c23ef6e21f06f20f1711ff1731df2741cf3761bf37819f47918f57b17f57d15f67e14f68013f78212f78410f8850ff8870ef8890cf98b0bf98c0af98e09fa9008fa9207fa9407fb9606fb9706fb9906fb9b06fb9d07fc9f07fca108fca309fca50afca60cfca80dfcaa0ffcac11fcae12fcb014fcb216fcb418fbb61afbb81dfbba1ffbbc21fbbe23fac026fac228fac42afac62df9c72ff9c932f9cb35f8cd37f8cf3af7d13df7d340f6d543f6d746f5d949f5db4cf4dd4ff4df53f4e156f3e35af3e55df2e661f2e865f2ea69f1ec6df1ed71f1ef75f1f179f2f27df2f482f3f586f3f68af4f88ef5f992f6fa96f8fb9af9fc9dfafda1fcffa4"));var plasma=ramp$1(colors("0d088710078813078916078a19068c1b068d1d068e20068f2206902406912605912805922a05932c05942e05952f059631059733059735049837049938049a3a049a3c049b3e049c3f049c41049d43039e44039e46039f48039f4903a04b03a14c02a14e02a25002a25102a35302a35502a45601a45801a45901a55b01a55c01a65e01a66001a66100a76300a76400a76600a76700a86900a86a00a86c00a86e00a86f00a87100a87201a87401a87501a87701a87801a87a02a87b02a87d03a87e03a88004a88104a78305a78405a78606a68707a68808a68a09a58b0aa58d0ba58e0ca48f0da4910ea3920fa39410a29511a19613a19814a099159f9a169f9c179e9d189d9e199da01a9ca11b9ba21d9aa31e9aa51f99a62098a72197a82296aa2395ab2494ac2694ad2793ae2892b02991b12a90b22b8fb32c8eb42e8db52f8cb6308bb7318ab83289ba3388bb3488bc3587bd3786be3885bf3984c03a83c13b82c23c81c33d80c43e7fc5407ec6417dc7427cc8437bc9447aca457acb4679cc4778cc4977cd4a76ce4b75cf4c74d04d73d14e72d24f71d35171d45270d5536fd5546ed6556dd7566cd8576bd9586ada5a6ada5b69db5c68dc5d67dd5e66de5f65de6164df6263e06363e16462e26561e26660e3685fe4695ee56a5de56b5de66c5ce76e5be76f5ae87059e97158e97257ea7457eb7556eb7655ec7754ed7953ed7a52ee7b51ef7c51ef7e50f07f4ff0804ef1814df1834cf2844bf3854bf3874af48849f48948f58b47f58c46f68d45f68f44f79044f79143f79342f89441f89540f9973ff9983ef99a3efa9b3dfa9c3cfa9e3bfb9f3afba139fba238fca338fca537fca636fca835fca934fdab33fdac33fdae32fdaf31fdb130fdb22ffdb42ffdb52efeb72dfeb82cfeba2cfebb2bfebd2afebe2afec029fdc229fdc328fdc527fdc627fdc827fdca26fdcb26fccd25fcce25fcd025fcd225fbd324fbd524fbd724fad824fada24f9dc24f9dd25f8df25f8e125f7e225f7e425f6e626f6e826f5e926f5eb27f4ed27f3ee27f3f027f2f227f1f426f1f525f0f724f0f921"));exports.interpolateBlues=Blues;exports.interpolateBrBG=BrBG;exports.interpolateBuGn=BuGn;exports.interpolateBuPu=BuPu;exports.interpolateCividis=cividis;exports.interpolateCool=cool;exports.interpolateCubehelixDefault=cubehelix;exports.interpolateGnBu=GnBu;exports.interpolateGreens=Greens;exports.interpolateGreys=Greys;exports.interpolateInferno=inferno;exports.interpolateMagma=magma;exports.interpolateOrRd=OrRd;exports.interpolateOranges=Oranges;exports.interpolatePRGn=PRGn;exports.interpolatePiYG=PiYG;exports.interpolatePlasma=plasma;exports.interpolatePuBu=PuBu;exports.interpolatePuBuGn=PuBuGn;exports.interpolatePuOr=PuOr;exports.interpolatePuRd=PuRd;exports.interpolatePurples=Purples;exports.interpolateRainbow=rainbow;exports.interpolateRdBu=RdBu;exports.interpolateRdGy=RdGy;exports.interpolateRdPu=RdPu;exports.interpolateRdYlBu=RdYlBu;exports.interpolateRdYlGn=RdYlGn;exports.interpolateReds=Reds;exports.interpolateSinebow=sinebow;exports.interpolateSpectral=Spectral;exports.interpolateTurbo=turbo;exports.interpolateViridis=viridis;exports.interpolateWarm=warm;exports.interpolateYlGn=YlGn;exports.interpolateYlGnBu=YlGnBu;exports.interpolateYlOrBr=YlOrBr;exports.interpolateYlOrRd=YlOrRd;exports.schemeAccent=Accent;exports.schemeBlues=scheme$l;exports.schemeBrBG=scheme;exports.schemeBuGn=scheme$9;exports.schemeBuPu=scheme$a;exports.schemeCategory10=category10;exports.schemeDark2=Dark2;exports.schemeGnBu=scheme$b;exports.schemeGreens=scheme$m;exports.schemeGreys=scheme$n;exports.schemeOrRd=scheme$c;exports.schemeOranges=scheme$q;exports.schemePRGn=scheme$1;exports.schemePaired=Paired;exports.schemePastel1=Pastel1;exports.schemePastel2=Pastel2;exports.schemePiYG=scheme$2;exports.schemePuBu=scheme$e;exports.schemePuBuGn=scheme$d;exports.schemePuOr=scheme$3;exports.schemePuRd=scheme$f;exports.schemePurples=scheme$o;exports.schemeRdBu=scheme$4;exports.schemeRdGy=scheme$5;exports.schemeRdPu=scheme$g;exports.schemeRdYlBu=scheme$6;exports.schemeRdYlGn=scheme$7;exports.schemeReds=scheme$p;exports.schemeSet1=Set1;exports.schemeSet2=Set2;exports.schemeSet3=Set3;exports.schemeSpectral=scheme$8;exports.schemeTableau10=Tableau10;exports.schemeYlGn=scheme$i;exports.schemeYlGnBu=scheme$h;exports.schemeYlOrBr=scheme$j;exports.schemeYlOrRd=scheme$k;Object.defineProperty(exports,"__esModule",{value:true})})},{"d3-color":34,"d3-interpolate":45}],51:[function(require,module,exports){
// https://d3js.org/d3-scale/ v2.2.2 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports,require("d3-collection"),require("d3-array"),require("d3-interpolate"),require("d3-format"),require("d3-time"),require("d3-time-format")):typeof define==="function"&&define.amd?define(["exports","d3-collection","d3-array","d3-interpolate","d3-format","d3-time","d3-time-format"],factory):factory(global.d3=global.d3||{},global.d3,global.d3,global.d3,global.d3,global.d3,global.d3)})(this,function(exports,d3Collection,d3Array,d3Interpolate,d3Format,d3Time,d3TimeFormat){"use strict";function initRange(domain,range){switch(arguments.length){case 0:break;case 1:this.range(domain);break;default:this.range(range).domain(domain);break}return this}function initInterpolator(domain,interpolator){switch(arguments.length){case 0:break;case 1:this.interpolator(domain);break;default:this.interpolator(interpolator).domain(domain);break}return this}var array=Array.prototype;var map=array.map;var slice=array.slice;var implicit={name:"implicit"};function ordinal(){var index=d3Collection.map(),domain=[],range=[],unknown=implicit;function scale(d){var key=d+"",i=index.get(key);if(!i){if(unknown!==implicit)return unknown;index.set(key,i=domain.push(d))}return range[(i-1)%range.length]}scale.domain=function(_){if(!arguments.length)return domain.slice();domain=[],index=d3Collection.map();var i=-1,n=_.length,d,key;while(++i<n)if(!index.has(key=(d=_[i])+""))index.set(key,domain.push(d));return scale};scale.range=function(_){return arguments.length?(range=slice.call(_),scale):range.slice()};scale.unknown=function(_){return arguments.length?(unknown=_,scale):unknown};scale.copy=function(){return ordinal(domain,range).unknown(unknown)};initRange.apply(scale,arguments);return scale}function band(){var scale=ordinal().unknown(undefined),domain=scale.domain,ordinalRange=scale.range,range=[0,1],step,bandwidth,round=false,paddingInner=0,paddingOuter=0,align=.5;delete scale.unknown;function rescale(){var n=domain().length,reverse=range[1]<range[0],start=range[reverse-0],stop=range[1-reverse];step=(stop-start)/Math.max(1,n-paddingInner+paddingOuter*2);if(round)step=Math.floor(step);start+=(stop-start-step*(n-paddingInner))*align;bandwidth=step*(1-paddingInner);if(round)start=Math.round(start),bandwidth=Math.round(bandwidth);var values=d3Array.range(n).map(function(i){return start+step*i});return ordinalRange(reverse?values.reverse():values)}scale.domain=function(_){return arguments.length?(domain(_),rescale()):domain()};scale.range=function(_){return arguments.length?(range=[+_[0],+_[1]],rescale()):range.slice()};scale.rangeRound=function(_){return range=[+_[0],+_[1]],round=true,rescale()};scale.bandwidth=function(){return bandwidth};scale.step=function(){return step};scale.round=function(_){return arguments.length?(round=!!_,rescale()):round};scale.padding=function(_){return arguments.length?(paddingInner=Math.min(1,paddingOuter=+_),rescale()):paddingInner};scale.paddingInner=function(_){return arguments.length?(paddingInner=Math.min(1,_),rescale()):paddingInner};scale.paddingOuter=function(_){return arguments.length?(paddingOuter=+_,rescale()):paddingOuter};scale.align=function(_){return arguments.length?(align=Math.max(0,Math.min(1,_)),rescale()):align};scale.copy=function(){return band(domain(),range).round(round).paddingInner(paddingInner).paddingOuter(paddingOuter).align(align)};return initRange.apply(rescale(),arguments)}function pointish(scale){var copy=scale.copy;scale.padding=scale.paddingOuter;delete scale.paddingInner;delete scale.paddingOuter;scale.copy=function(){return pointish(copy())};return scale}function point(){return pointish(band.apply(null,arguments).paddingInner(1))}function constant(x){return function(){return x}}function number(x){return+x}var unit=[0,1];function identity(x){return x}function normalize(a,b){return(b-=a=+a)?function(x){return(x-a)/b}:constant(isNaN(b)?NaN:.5)}function clamper(domain){var a=domain[0],b=domain[domain.length-1],t;if(a>b)t=a,a=b,b=t;return function(x){return Math.max(a,Math.min(b,x))}}
// normalize(a, b)(x) takes a domain value x in [a,b] and returns the corresponding parameter t in [0,1].
// interpolate(a, b)(t) takes a parameter t in [0,1] and returns the corresponding range value x in [a,b].
function bimap(domain,range,interpolate){var d0=domain[0],d1=domain[1],r0=range[0],r1=range[1];if(d1<d0)d0=normalize(d1,d0),r0=interpolate(r1,r0);else d0=normalize(d0,d1),r0=interpolate(r0,r1);return function(x){return r0(d0(x))}}function polymap(domain,range,interpolate){var j=Math.min(domain.length,range.length)-1,d=new Array(j),r=new Array(j),i=-1;
// Reverse descending domains.
if(domain[j]<domain[0]){domain=domain.slice().reverse();range=range.slice().reverse()}while(++i<j){d[i]=normalize(domain[i],domain[i+1]);r[i]=interpolate(range[i],range[i+1])}return function(x){var i=d3Array.bisect(domain,x,1,j)-1;return r[i](d[i](x))}}function copy(source,target){return target.domain(source.domain()).range(source.range()).interpolate(source.interpolate()).clamp(source.clamp()).unknown(source.unknown())}function transformer(){var domain=unit,range=unit,interpolate=d3Interpolate.interpolate,transform,untransform,unknown,clamp=identity,piecewise,output,input;function rescale(){piecewise=Math.min(domain.length,range.length)>2?polymap:bimap;output=input=null;return scale}function scale(x){return isNaN(x=+x)?unknown:(output||(output=piecewise(domain.map(transform),range,interpolate)))(transform(clamp(x)))}scale.invert=function(y){return clamp(untransform((input||(input=piecewise(range,domain.map(transform),d3Interpolate.interpolateNumber)))(y)))};scale.domain=function(_){return arguments.length?(domain=map.call(_,number),clamp===identity||(clamp=clamper(domain)),rescale()):domain.slice()};scale.range=function(_){return arguments.length?(range=slice.call(_),rescale()):range.slice()};scale.rangeRound=function(_){return range=slice.call(_),interpolate=d3Interpolate.interpolateRound,rescale()};scale.clamp=function(_){return arguments.length?(clamp=_?clamper(domain):identity,scale):clamp!==identity};scale.interpolate=function(_){return arguments.length?(interpolate=_,rescale()):interpolate};scale.unknown=function(_){return arguments.length?(unknown=_,scale):unknown};return function(t,u){transform=t,untransform=u;return rescale()}}function continuous(transform,untransform){return transformer()(transform,untransform)}function tickFormat(start,stop,count,specifier){var step=d3Array.tickStep(start,stop,count),precision;specifier=d3Format.formatSpecifier(specifier==null?",f":specifier);switch(specifier.type){case"s":{var value=Math.max(Math.abs(start),Math.abs(stop));if(specifier.precision==null&&!isNaN(precision=d3Format.precisionPrefix(step,value)))specifier.precision=precision;return d3Format.formatPrefix(specifier,value)}case"":case"e":case"g":case"p":case"r":{if(specifier.precision==null&&!isNaN(precision=d3Format.precisionRound(step,Math.max(Math.abs(start),Math.abs(stop)))))specifier.precision=precision-(specifier.type==="e");break}case"f":case"%":{if(specifier.precision==null&&!isNaN(precision=d3Format.precisionFixed(step)))specifier.precision=precision-(specifier.type==="%")*2;break}}return d3Format.format(specifier)}function linearish(scale){var domain=scale.domain;scale.ticks=function(count){var d=domain();return d3Array.ticks(d[0],d[d.length-1],count==null?10:count)};scale.tickFormat=function(count,specifier){var d=domain();return tickFormat(d[0],d[d.length-1],count==null?10:count,specifier)};scale.nice=function(count){if(count==null)count=10;var d=domain(),i0=0,i1=d.length-1,start=d[i0],stop=d[i1],step;if(stop<start){step=start,start=stop,stop=step;step=i0,i0=i1,i1=step}step=d3Array.tickIncrement(start,stop,count);if(step>0){start=Math.floor(start/step)*step;stop=Math.ceil(stop/step)*step;step=d3Array.tickIncrement(start,stop,count)}else if(step<0){start=Math.ceil(start*step)/step;stop=Math.floor(stop*step)/step;step=d3Array.tickIncrement(start,stop,count)}if(step>0){d[i0]=Math.floor(start/step)*step;d[i1]=Math.ceil(stop/step)*step;domain(d)}else if(step<0){d[i0]=Math.ceil(start*step)/step;d[i1]=Math.floor(stop*step)/step;domain(d)}return scale};return scale}function linear(){var scale=continuous(identity,identity);scale.copy=function(){return copy(scale,linear())};initRange.apply(scale,arguments);return linearish(scale)}function identity$1(domain){var unknown;function scale(x){return isNaN(x=+x)?unknown:x}scale.invert=scale;scale.domain=scale.range=function(_){return arguments.length?(domain=map.call(_,number),scale):domain.slice()};scale.unknown=function(_){return arguments.length?(unknown=_,scale):unknown};scale.copy=function(){return identity$1(domain).unknown(unknown)};domain=arguments.length?map.call(domain,number):[0,1];return linearish(scale)}function nice(domain,interval){domain=domain.slice();var i0=0,i1=domain.length-1,x0=domain[i0],x1=domain[i1],t;if(x1<x0){t=i0,i0=i1,i1=t;t=x0,x0=x1,x1=t}domain[i0]=interval.floor(x0);domain[i1]=interval.ceil(x1);return domain}function transformLog(x){return Math.log(x)}function transformExp(x){return Math.exp(x)}function transformLogn(x){return-Math.log(-x)}function transformExpn(x){return-Math.exp(-x)}function pow10(x){return isFinite(x)?+("1e"+x):x<0?0:x}function powp(base){return base===10?pow10:base===Math.E?Math.exp:function(x){return Math.pow(base,x)}}function logp(base){return base===Math.E?Math.log:base===10&&Math.log10||base===2&&Math.log2||(base=Math.log(base),function(x){return Math.log(x)/base})}function reflect(f){return function(x){return-f(-x)}}function loggish(transform){var scale=transform(transformLog,transformExp),domain=scale.domain,base=10,logs,pows;function rescale(){logs=logp(base),pows=powp(base);if(domain()[0]<0){logs=reflect(logs),pows=reflect(pows);transform(transformLogn,transformExpn)}else{transform(transformLog,transformExp)}return scale}scale.base=function(_){return arguments.length?(base=+_,rescale()):base};scale.domain=function(_){return arguments.length?(domain(_),rescale()):domain()};scale.ticks=function(count){var d=domain(),u=d[0],v=d[d.length-1],r;if(r=v<u)i=u,u=v,v=i;var i=logs(u),j=logs(v),p,k,t,n=count==null?10:+count,z=[];if(!(base%1)&&j-i<n){i=Math.round(i)-1,j=Math.round(j)+1;if(u>0)for(;i<j;++i){for(k=1,p=pows(i);k<base;++k){t=p*k;if(t<u)continue;if(t>v)break;z.push(t)}}else for(;i<j;++i){for(k=base-1,p=pows(i);k>=1;--k){t=p*k;if(t<u)continue;if(t>v)break;z.push(t)}}}else{z=d3Array.ticks(i,j,Math.min(j-i,n)).map(pows)}return r?z.reverse():z};scale.tickFormat=function(count,specifier){if(specifier==null)specifier=base===10?".0e":",";if(typeof specifier!=="function")specifier=d3Format.format(specifier);if(count===Infinity)return specifier;if(count==null)count=10;var k=Math.max(1,base*count/scale.ticks().length);// TODO fast estimate?
return function(d){var i=d/pows(Math.round(logs(d)));if(i*base<base-.5)i*=base;return i<=k?specifier(d):""}};scale.nice=function(){return domain(nice(domain(),{floor:function(x){return pows(Math.floor(logs(x)))},ceil:function(x){return pows(Math.ceil(logs(x)))}}))};return scale}function log(){var scale=loggish(transformer()).domain([1,10]);scale.copy=function(){return copy(scale,log()).base(scale.base())};initRange.apply(scale,arguments);return scale}function transformSymlog(c){return function(x){return Math.sign(x)*Math.log1p(Math.abs(x/c))}}function transformSymexp(c){return function(x){return Math.sign(x)*Math.expm1(Math.abs(x))*c}}function symlogish(transform){var c=1,scale=transform(transformSymlog(c),transformSymexp(c));scale.constant=function(_){return arguments.length?transform(transformSymlog(c=+_),transformSymexp(c)):c};return linearish(scale)}function symlog(){var scale=symlogish(transformer());scale.copy=function(){return copy(scale,symlog()).constant(scale.constant())};return initRange.apply(scale,arguments)}function transformPow(exponent){return function(x){return x<0?-Math.pow(-x,exponent):Math.pow(x,exponent)}}function transformSqrt(x){return x<0?-Math.sqrt(-x):Math.sqrt(x)}function transformSquare(x){return x<0?-x*x:x*x}function powish(transform){var scale=transform(identity,identity),exponent=1;function rescale(){return exponent===1?transform(identity,identity):exponent===.5?transform(transformSqrt,transformSquare):transform(transformPow(exponent),transformPow(1/exponent))}scale.exponent=function(_){return arguments.length?(exponent=+_,rescale()):exponent};return linearish(scale)}function pow(){var scale=powish(transformer());scale.copy=function(){return copy(scale,pow()).exponent(scale.exponent())};initRange.apply(scale,arguments);return scale}function sqrt(){return pow.apply(null,arguments).exponent(.5)}function quantile(){var domain=[],range=[],thresholds=[],unknown;function rescale(){var i=0,n=Math.max(1,range.length);thresholds=new Array(n-1);while(++i<n)thresholds[i-1]=d3Array.quantile(domain,i/n);return scale}function scale(x){return isNaN(x=+x)?unknown:range[d3Array.bisect(thresholds,x)]}scale.invertExtent=function(y){var i=range.indexOf(y);return i<0?[NaN,NaN]:[i>0?thresholds[i-1]:domain[0],i<thresholds.length?thresholds[i]:domain[domain.length-1]]};scale.domain=function(_){if(!arguments.length)return domain.slice();domain=[];for(var i=0,n=_.length,d;i<n;++i)if(d=_[i],d!=null&&!isNaN(d=+d))domain.push(d);domain.sort(d3Array.ascending);return rescale()};scale.range=function(_){return arguments.length?(range=slice.call(_),rescale()):range.slice()};scale.unknown=function(_){return arguments.length?(unknown=_,scale):unknown};scale.quantiles=function(){return thresholds.slice()};scale.copy=function(){return quantile().domain(domain).range(range).unknown(unknown)};return initRange.apply(scale,arguments)}function quantize(){var x0=0,x1=1,n=1,domain=[.5],range=[0,1],unknown;function scale(x){return x<=x?range[d3Array.bisect(domain,x,0,n)]:unknown}function rescale(){var i=-1;domain=new Array(n);while(++i<n)domain[i]=((i+1)*x1-(i-n)*x0)/(n+1);return scale}scale.domain=function(_){return arguments.length?(x0=+_[0],x1=+_[1],rescale()):[x0,x1]};scale.range=function(_){return arguments.length?(n=(range=slice.call(_)).length-1,rescale()):range.slice()};scale.invertExtent=function(y){var i=range.indexOf(y);return i<0?[NaN,NaN]:i<1?[x0,domain[0]]:i>=n?[domain[n-1],x1]:[domain[i-1],domain[i]]};scale.unknown=function(_){return arguments.length?(unknown=_,scale):scale};scale.thresholds=function(){return domain.slice()};scale.copy=function(){return quantize().domain([x0,x1]).range(range).unknown(unknown)};return initRange.apply(linearish(scale),arguments)}function threshold(){var domain=[.5],range=[0,1],unknown,n=1;function scale(x){return x<=x?range[d3Array.bisect(domain,x,0,n)]:unknown}scale.domain=function(_){return arguments.length?(domain=slice.call(_),n=Math.min(domain.length,range.length-1),scale):domain.slice()};scale.range=function(_){return arguments.length?(range=slice.call(_),n=Math.min(domain.length,range.length-1),scale):range.slice()};scale.invertExtent=function(y){var i=range.indexOf(y);return[domain[i-1],domain[i]]};scale.unknown=function(_){return arguments.length?(unknown=_,scale):unknown};scale.copy=function(){return threshold().domain(domain).range(range).unknown(unknown)};return initRange.apply(scale,arguments)}var durationSecond=1e3,durationMinute=durationSecond*60,durationHour=durationMinute*60,durationDay=durationHour*24,durationWeek=durationDay*7,durationMonth=durationDay*30,durationYear=durationDay*365;function date(t){return new Date(t)}function number$1(t){return t instanceof Date?+t:+new Date(+t)}function calendar(year,month,week,day,hour,minute,second,millisecond,format){var scale=continuous(identity,identity),invert=scale.invert,domain=scale.domain;var formatMillisecond=format(".%L"),formatSecond=format(":%S"),formatMinute=format("%I:%M"),formatHour=format("%I %p"),formatDay=format("%a %d"),formatWeek=format("%b %d"),formatMonth=format("%B"),formatYear=format("%Y");var tickIntervals=[[second,1,durationSecond],[second,5,5*durationSecond],[second,15,15*durationSecond],[second,30,30*durationSecond],[minute,1,durationMinute],[minute,5,5*durationMinute],[minute,15,15*durationMinute],[minute,30,30*durationMinute],[hour,1,durationHour],[hour,3,3*durationHour],[hour,6,6*durationHour],[hour,12,12*durationHour],[day,1,durationDay],[day,2,2*durationDay],[week,1,durationWeek],[month,1,durationMonth],[month,3,3*durationMonth],[year,1,durationYear]];function tickFormat(date){return(second(date)<date?formatMillisecond:minute(date)<date?formatSecond:hour(date)<date?formatMinute:day(date)<date?formatHour:month(date)<date?week(date)<date?formatDay:formatWeek:year(date)<date?formatMonth:formatYear)(date)}function tickInterval(interval,start,stop,step){if(interval==null)interval=10;
// If a desired tick count is specified, pick a reasonable tick interval
// based on the extent of the domain and a rough estimate of tick size.
// Otherwise, assume interval is already a time interval and use it.
if(typeof interval==="number"){var target=Math.abs(stop-start)/interval,i=d3Array.bisector(function(i){return i[2]}).right(tickIntervals,target);if(i===tickIntervals.length){step=d3Array.tickStep(start/durationYear,stop/durationYear,interval);interval=year}else if(i){i=tickIntervals[target/tickIntervals[i-1][2]<tickIntervals[i][2]/target?i-1:i];step=i[1];interval=i[0]}else{step=Math.max(d3Array.tickStep(start,stop,interval),1);interval=millisecond}}return step==null?interval:interval.every(step)}scale.invert=function(y){return new Date(invert(y))};scale.domain=function(_){return arguments.length?domain(map.call(_,number$1)):domain().map(date)};scale.ticks=function(interval,step){var d=domain(),t0=d[0],t1=d[d.length-1],r=t1<t0,t;if(r)t=t0,t0=t1,t1=t;t=tickInterval(interval,t0,t1,step);t=t?t.range(t0,t1+1):[];// inclusive stop
return r?t.reverse():t};scale.tickFormat=function(count,specifier){return specifier==null?tickFormat:format(specifier)};scale.nice=function(interval,step){var d=domain();return(interval=tickInterval(interval,d[0],d[d.length-1],step))?domain(nice(d,interval)):scale};scale.copy=function(){return copy(scale,calendar(year,month,week,day,hour,minute,second,millisecond,format))};return scale}function time(){return initRange.apply(calendar(d3Time.timeYear,d3Time.timeMonth,d3Time.timeWeek,d3Time.timeDay,d3Time.timeHour,d3Time.timeMinute,d3Time.timeSecond,d3Time.timeMillisecond,d3TimeFormat.timeFormat).domain([new Date(2e3,0,1),new Date(2e3,0,2)]),arguments)}function utcTime(){return initRange.apply(calendar(d3Time.utcYear,d3Time.utcMonth,d3Time.utcWeek,d3Time.utcDay,d3Time.utcHour,d3Time.utcMinute,d3Time.utcSecond,d3Time.utcMillisecond,d3TimeFormat.utcFormat).domain([Date.UTC(2e3,0,1),Date.UTC(2e3,0,2)]),arguments)}function transformer$1(){var x0=0,x1=1,t0,t1,k10,transform,interpolator=identity,clamp=false,unknown;function scale(x){return isNaN(x=+x)?unknown:interpolator(k10===0?.5:(x=(transform(x)-t0)*k10,clamp?Math.max(0,Math.min(1,x)):x))}scale.domain=function(_){return arguments.length?(t0=transform(x0=+_[0]),t1=transform(x1=+_[1]),k10=t0===t1?0:1/(t1-t0),scale):[x0,x1]};scale.clamp=function(_){return arguments.length?(clamp=!!_,scale):clamp};scale.interpolator=function(_){return arguments.length?(interpolator=_,scale):interpolator};scale.unknown=function(_){return arguments.length?(unknown=_,scale):unknown};return function(t){transform=t,t0=t(x0),t1=t(x1),k10=t0===t1?0:1/(t1-t0);return scale}}function copy$1(source,target){return target.domain(source.domain()).interpolator(source.interpolator()).clamp(source.clamp()).unknown(source.unknown())}function sequential(){var scale=linearish(transformer$1()(identity));scale.copy=function(){return copy$1(scale,sequential())};return initInterpolator.apply(scale,arguments)}function sequentialLog(){var scale=loggish(transformer$1()).domain([1,10]);scale.copy=function(){return copy$1(scale,sequentialLog()).base(scale.base())};return initInterpolator.apply(scale,arguments)}function sequentialSymlog(){var scale=symlogish(transformer$1());scale.copy=function(){return copy$1(scale,sequentialSymlog()).constant(scale.constant())};return initInterpolator.apply(scale,arguments)}function sequentialPow(){var scale=powish(transformer$1());scale.copy=function(){return copy$1(scale,sequentialPow()).exponent(scale.exponent())};return initInterpolator.apply(scale,arguments)}function sequentialSqrt(){return sequentialPow.apply(null,arguments).exponent(.5)}function sequentialQuantile(){var domain=[],interpolator=identity;function scale(x){if(!isNaN(x=+x))return interpolator((d3Array.bisect(domain,x)-1)/(domain.length-1))}scale.domain=function(_){if(!arguments.length)return domain.slice();domain=[];for(var i=0,n=_.length,d;i<n;++i)if(d=_[i],d!=null&&!isNaN(d=+d))domain.push(d);domain.sort(d3Array.ascending);return scale};scale.interpolator=function(_){return arguments.length?(interpolator=_,scale):interpolator};scale.copy=function(){return sequentialQuantile(interpolator).domain(domain)};return initInterpolator.apply(scale,arguments)}function transformer$2(){var x0=0,x1=.5,x2=1,t0,t1,t2,k10,k21,interpolator=identity,transform,clamp=false,unknown;function scale(x){return isNaN(x=+x)?unknown:(x=.5+((x=+transform(x))-t1)*(x<t1?k10:k21),interpolator(clamp?Math.max(0,Math.min(1,x)):x))}scale.domain=function(_){return arguments.length?(t0=transform(x0=+_[0]),t1=transform(x1=+_[1]),t2=transform(x2=+_[2]),k10=t0===t1?0:.5/(t1-t0),k21=t1===t2?0:.5/(t2-t1),scale):[x0,x1,x2]};scale.clamp=function(_){return arguments.length?(clamp=!!_,scale):clamp};scale.interpolator=function(_){return arguments.length?(interpolator=_,scale):interpolator};scale.unknown=function(_){return arguments.length?(unknown=_,scale):unknown};return function(t){transform=t,t0=t(x0),t1=t(x1),t2=t(x2),k10=t0===t1?0:.5/(t1-t0),k21=t1===t2?0:.5/(t2-t1);return scale}}function diverging(){var scale=linearish(transformer$2()(identity));scale.copy=function(){return copy$1(scale,diverging())};return initInterpolator.apply(scale,arguments)}function divergingLog(){var scale=loggish(transformer$2()).domain([.1,1,10]);scale.copy=function(){return copy$1(scale,divergingLog()).base(scale.base())};return initInterpolator.apply(scale,arguments)}function divergingSymlog(){var scale=symlogish(transformer$2());scale.copy=function(){return copy$1(scale,divergingSymlog()).constant(scale.constant())};return initInterpolator.apply(scale,arguments)}function divergingPow(){var scale=powish(transformer$2());scale.copy=function(){return copy$1(scale,divergingPow()).exponent(scale.exponent())};return initInterpolator.apply(scale,arguments)}function divergingSqrt(){return divergingPow.apply(null,arguments).exponent(.5)}exports.scaleBand=band;exports.scalePoint=point;exports.scaleIdentity=identity$1;exports.scaleLinear=linear;exports.scaleLog=log;exports.scaleSymlog=symlog;exports.scaleOrdinal=ordinal;exports.scaleImplicit=implicit;exports.scalePow=pow;exports.scaleSqrt=sqrt;exports.scaleQuantile=quantile;exports.scaleQuantize=quantize;exports.scaleThreshold=threshold;exports.scaleTime=time;exports.scaleUtc=utcTime;exports.scaleSequential=sequential;exports.scaleSequentialLog=sequentialLog;exports.scaleSequentialPow=sequentialPow;exports.scaleSequentialSqrt=sequentialSqrt;exports.scaleSequentialSymlog=sequentialSymlog;exports.scaleSequentialQuantile=sequentialQuantile;exports.scaleDiverging=diverging;exports.scaleDivergingLog=divergingLog;exports.scaleDivergingPow=divergingPow;exports.scaleDivergingSqrt=divergingSqrt;exports.scaleDivergingSymlog=divergingSymlog;exports.tickFormat=tickFormat;Object.defineProperty(exports,"__esModule",{value:true})})},{"d3-array":29,"d3-collection":33,"d3-format":42,"d3-interpolate":45,"d3-time":55,"d3-time-format":54}],52:[function(require,module,exports){
// https://d3js.org/d3-selection/ v1.4.1 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):(global=global||self,factory(global.d3=global.d3||{}))})(this,function(exports){"use strict";var xhtml="http://www.w3.org/1999/xhtml";var namespaces={svg:"http://www.w3.org/2000/svg",xhtml:xhtml,xlink:"http://www.w3.org/1999/xlink",xml:"http://www.w3.org/XML/1998/namespace",xmlns:"http://www.w3.org/2000/xmlns/"};function namespace(name){var prefix=name+="",i=prefix.indexOf(":");if(i>=0&&(prefix=name.slice(0,i))!=="xmlns")name=name.slice(i+1);return namespaces.hasOwnProperty(prefix)?{space:namespaces[prefix],local:name}:name}function creatorInherit(name){return function(){var document=this.ownerDocument,uri=this.namespaceURI;return uri===xhtml&&document.documentElement.namespaceURI===xhtml?document.createElement(name):document.createElementNS(uri,name)}}function creatorFixed(fullname){return function(){return this.ownerDocument.createElementNS(fullname.space,fullname.local)}}function creator(name){var fullname=namespace(name);return(fullname.local?creatorFixed:creatorInherit)(fullname)}function none(){}function selector(selector){return selector==null?none:function(){return this.querySelector(selector)}}function selection_select(select){if(typeof select!=="function")select=selector(select);for(var groups=this._groups,m=groups.length,subgroups=new Array(m),j=0;j<m;++j){for(var group=groups[j],n=group.length,subgroup=subgroups[j]=new Array(n),node,subnode,i=0;i<n;++i){if((node=group[i])&&(subnode=select.call(node,node.__data__,i,group))){if("__data__"in node)subnode.__data__=node.__data__;subgroup[i]=subnode}}}return new Selection(subgroups,this._parents)}function empty(){return[]}function selectorAll(selector){return selector==null?empty:function(){return this.querySelectorAll(selector)}}function selection_selectAll(select){if(typeof select!=="function")select=selectorAll(select);for(var groups=this._groups,m=groups.length,subgroups=[],parents=[],j=0;j<m;++j){for(var group=groups[j],n=group.length,node,i=0;i<n;++i){if(node=group[i]){subgroups.push(select.call(node,node.__data__,i,group));parents.push(node)}}}return new Selection(subgroups,parents)}function matcher(selector){return function(){return this.matches(selector)}}function selection_filter(match){if(typeof match!=="function")match=matcher(match);for(var groups=this._groups,m=groups.length,subgroups=new Array(m),j=0;j<m;++j){for(var group=groups[j],n=group.length,subgroup=subgroups[j]=[],node,i=0;i<n;++i){if((node=group[i])&&match.call(node,node.__data__,i,group)){subgroup.push(node)}}}return new Selection(subgroups,this._parents)}function sparse(update){return new Array(update.length)}function selection_enter(){return new Selection(this._enter||this._groups.map(sparse),this._parents)}function EnterNode(parent,datum){this.ownerDocument=parent.ownerDocument;this.namespaceURI=parent.namespaceURI;this._next=null;this._parent=parent;this.__data__=datum}EnterNode.prototype={constructor:EnterNode,appendChild:function(child){return this._parent.insertBefore(child,this._next)},insertBefore:function(child,next){return this._parent.insertBefore(child,next)},querySelector:function(selector){return this._parent.querySelector(selector)},querySelectorAll:function(selector){return this._parent.querySelectorAll(selector)}};function constant(x){return function(){return x}}var keyPrefix="$";// Protect against keys like “__proto__”.
function bindIndex(parent,group,enter,update,exit,data){var i=0,node,groupLength=group.length,dataLength=data.length;
// Put any non-null nodes that fit into update.
// Put any null nodes into enter.
// Put any remaining data into enter.
for(;i<dataLength;++i){if(node=group[i]){node.__data__=data[i];update[i]=node}else{enter[i]=new EnterNode(parent,data[i])}}
// Put any non-null nodes that don’t fit into exit.
for(;i<groupLength;++i){if(node=group[i]){exit[i]=node}}}function bindKey(parent,group,enter,update,exit,data,key){var i,node,nodeByKeyValue={},groupLength=group.length,dataLength=data.length,keyValues=new Array(groupLength),keyValue;
// Compute the key for each node.
// If multiple nodes have the same key, the duplicates are added to exit.
for(i=0;i<groupLength;++i){if(node=group[i]){keyValues[i]=keyValue=keyPrefix+key.call(node,node.__data__,i,group);if(keyValue in nodeByKeyValue){exit[i]=node}else{nodeByKeyValue[keyValue]=node}}}
// Compute the key for each datum.
// If there a node associated with this key, join and add it to update.
// If there is not (or the key is a duplicate), add it to enter.
for(i=0;i<dataLength;++i){keyValue=keyPrefix+key.call(parent,data[i],i,data);if(node=nodeByKeyValue[keyValue]){update[i]=node;node.__data__=data[i];nodeByKeyValue[keyValue]=null}else{enter[i]=new EnterNode(parent,data[i])}}
// Add any remaining nodes that were not bound to data to exit.
for(i=0;i<groupLength;++i){if((node=group[i])&&nodeByKeyValue[keyValues[i]]===node){exit[i]=node}}}function selection_data(value,key){if(!value){data=new Array(this.size()),j=-1;this.each(function(d){data[++j]=d});return data}var bind=key?bindKey:bindIndex,parents=this._parents,groups=this._groups;if(typeof value!=="function")value=constant(value);for(var m=groups.length,update=new Array(m),enter=new Array(m),exit=new Array(m),j=0;j<m;++j){var parent=parents[j],group=groups[j],groupLength=group.length,data=value.call(parent,parent&&parent.__data__,j,parents),dataLength=data.length,enterGroup=enter[j]=new Array(dataLength),updateGroup=update[j]=new Array(dataLength),exitGroup=exit[j]=new Array(groupLength);bind(parent,group,enterGroup,updateGroup,exitGroup,data,key);
// Now connect the enter nodes to their following update node, such that
// appendChild can insert the materialized enter node before this node,
// rather than at the end of the parent node.
for(var i0=0,i1=0,previous,next;i0<dataLength;++i0){if(previous=enterGroup[i0]){if(i0>=i1)i1=i0+1;while(!(next=updateGroup[i1])&&++i1<dataLength);previous._next=next||null}}}update=new Selection(update,parents);update._enter=enter;update._exit=exit;return update}function selection_exit(){return new Selection(this._exit||this._groups.map(sparse),this._parents)}function selection_join(onenter,onupdate,onexit){var enter=this.enter(),update=this,exit=this.exit();enter=typeof onenter==="function"?onenter(enter):enter.append(onenter+"");if(onupdate!=null)update=onupdate(update);if(onexit==null)exit.remove();else onexit(exit);return enter&&update?enter.merge(update).order():update}function selection_merge(selection){for(var groups0=this._groups,groups1=selection._groups,m0=groups0.length,m1=groups1.length,m=Math.min(m0,m1),merges=new Array(m0),j=0;j<m;++j){for(var group0=groups0[j],group1=groups1[j],n=group0.length,merge=merges[j]=new Array(n),node,i=0;i<n;++i){if(node=group0[i]||group1[i]){merge[i]=node}}}for(;j<m0;++j){merges[j]=groups0[j]}return new Selection(merges,this._parents)}function selection_order(){for(var groups=this._groups,j=-1,m=groups.length;++j<m;){for(var group=groups[j],i=group.length-1,next=group[i],node;--i>=0;){if(node=group[i]){if(next&&node.compareDocumentPosition(next)^4)next.parentNode.insertBefore(node,next);next=node}}}return this}function selection_sort(compare){if(!compare)compare=ascending;function compareNode(a,b){return a&&b?compare(a.__data__,b.__data__):!a-!b}for(var groups=this._groups,m=groups.length,sortgroups=new Array(m),j=0;j<m;++j){for(var group=groups[j],n=group.length,sortgroup=sortgroups[j]=new Array(n),node,i=0;i<n;++i){if(node=group[i]){sortgroup[i]=node}}sortgroup.sort(compareNode)}return new Selection(sortgroups,this._parents).order()}function ascending(a,b){return a<b?-1:a>b?1:a>=b?0:NaN}function selection_call(){var callback=arguments[0];arguments[0]=this;callback.apply(null,arguments);return this}function selection_nodes(){var nodes=new Array(this.size()),i=-1;this.each(function(){nodes[++i]=this});return nodes}function selection_node(){for(var groups=this._groups,j=0,m=groups.length;j<m;++j){for(var group=groups[j],i=0,n=group.length;i<n;++i){var node=group[i];if(node)return node}}return null}function selection_size(){var size=0;this.each(function(){++size});return size}function selection_empty(){return!this.node()}function selection_each(callback){for(var groups=this._groups,j=0,m=groups.length;j<m;++j){for(var group=groups[j],i=0,n=group.length,node;i<n;++i){if(node=group[i])callback.call(node,node.__data__,i,group)}}return this}function attrRemove(name){return function(){this.removeAttribute(name)}}function attrRemoveNS(fullname){return function(){this.removeAttributeNS(fullname.space,fullname.local)}}function attrConstant(name,value){return function(){this.setAttribute(name,value)}}function attrConstantNS(fullname,value){return function(){this.setAttributeNS(fullname.space,fullname.local,value)}}function attrFunction(name,value){return function(){var v=value.apply(this,arguments);if(v==null)this.removeAttribute(name);else this.setAttribute(name,v)}}function attrFunctionNS(fullname,value){return function(){var v=value.apply(this,arguments);if(v==null)this.removeAttributeNS(fullname.space,fullname.local);else this.setAttributeNS(fullname.space,fullname.local,v)}}function selection_attr(name,value){var fullname=namespace(name);if(arguments.length<2){var node=this.node();return fullname.local?node.getAttributeNS(fullname.space,fullname.local):node.getAttribute(fullname)}return this.each((value==null?fullname.local?attrRemoveNS:attrRemove:typeof value==="function"?fullname.local?attrFunctionNS:attrFunction:fullname.local?attrConstantNS:attrConstant)(fullname,value))}function defaultView(node){return node.ownerDocument&&node.ownerDocument.defaultView||node.document&&node||node.defaultView;// node is a Document
}function styleRemove(name){return function(){this.style.removeProperty(name)}}function styleConstant(name,value,priority){return function(){this.style.setProperty(name,value,priority)}}function styleFunction(name,value,priority){return function(){var v=value.apply(this,arguments);if(v==null)this.style.removeProperty(name);else this.style.setProperty(name,v,priority)}}function selection_style(name,value,priority){return arguments.length>1?this.each((value==null?styleRemove:typeof value==="function"?styleFunction:styleConstant)(name,value,priority==null?"":priority)):styleValue(this.node(),name)}function styleValue(node,name){return node.style.getPropertyValue(name)||defaultView(node).getComputedStyle(node,null).getPropertyValue(name)}function propertyRemove(name){return function(){delete this[name]}}function propertyConstant(name,value){return function(){this[name]=value}}function propertyFunction(name,value){return function(){var v=value.apply(this,arguments);if(v==null)delete this[name];else this[name]=v}}function selection_property(name,value){return arguments.length>1?this.each((value==null?propertyRemove:typeof value==="function"?propertyFunction:propertyConstant)(name,value)):this.node()[name]}function classArray(string){return string.trim().split(/^|\s+/)}function classList(node){return node.classList||new ClassList(node)}function ClassList(node){this._node=node;this._names=classArray(node.getAttribute("class")||"")}ClassList.prototype={add:function(name){var i=this._names.indexOf(name);if(i<0){this._names.push(name);this._node.setAttribute("class",this._names.join(" "))}},remove:function(name){var i=this._names.indexOf(name);if(i>=0){this._names.splice(i,1);this._node.setAttribute("class",this._names.join(" "))}},contains:function(name){return this._names.indexOf(name)>=0}};function classedAdd(node,names){var list=classList(node),i=-1,n=names.length;while(++i<n)list.add(names[i])}function classedRemove(node,names){var list=classList(node),i=-1,n=names.length;while(++i<n)list.remove(names[i])}function classedTrue(names){return function(){classedAdd(this,names)}}function classedFalse(names){return function(){classedRemove(this,names)}}function classedFunction(names,value){return function(){(value.apply(this,arguments)?classedAdd:classedRemove)(this,names)}}function selection_classed(name,value){var names=classArray(name+"");if(arguments.length<2){var list=classList(this.node()),i=-1,n=names.length;while(++i<n)if(!list.contains(names[i]))return false;return true}return this.each((typeof value==="function"?classedFunction:value?classedTrue:classedFalse)(names,value))}function textRemove(){this.textContent=""}function textConstant(value){return function(){this.textContent=value}}function textFunction(value){return function(){var v=value.apply(this,arguments);this.textContent=v==null?"":v}}function selection_text(value){return arguments.length?this.each(value==null?textRemove:(typeof value==="function"?textFunction:textConstant)(value)):this.node().textContent}function htmlRemove(){this.innerHTML=""}function htmlConstant(value){return function(){this.innerHTML=value}}function htmlFunction(value){return function(){var v=value.apply(this,arguments);this.innerHTML=v==null?"":v}}function selection_html(value){return arguments.length?this.each(value==null?htmlRemove:(typeof value==="function"?htmlFunction:htmlConstant)(value)):this.node().innerHTML}function raise(){if(this.nextSibling)this.parentNode.appendChild(this)}function selection_raise(){return this.each(raise)}function lower(){if(this.previousSibling)this.parentNode.insertBefore(this,this.parentNode.firstChild)}function selection_lower(){return this.each(lower)}function selection_append(name){var create=typeof name==="function"?name:creator(name);return this.select(function(){return this.appendChild(create.apply(this,arguments))})}function constantNull(){return null}function selection_insert(name,before){var create=typeof name==="function"?name:creator(name),select=before==null?constantNull:typeof before==="function"?before:selector(before);return this.select(function(){return this.insertBefore(create.apply(this,arguments),select.apply(this,arguments)||null)})}function remove(){var parent=this.parentNode;if(parent)parent.removeChild(this)}function selection_remove(){return this.each(remove)}function selection_cloneShallow(){var clone=this.cloneNode(false),parent=this.parentNode;return parent?parent.insertBefore(clone,this.nextSibling):clone}function selection_cloneDeep(){var clone=this.cloneNode(true),parent=this.parentNode;return parent?parent.insertBefore(clone,this.nextSibling):clone}function selection_clone(deep){return this.select(deep?selection_cloneDeep:selection_cloneShallow)}function selection_datum(value){return arguments.length?this.property("__data__",value):this.node().__data__}var filterEvents={};exports.event=null;if(typeof document!=="undefined"){var element=document.documentElement;if(!("onmouseenter"in element)){filterEvents={mouseenter:"mouseover",mouseleave:"mouseout"}}}function filterContextListener(listener,index,group){listener=contextListener(listener,index,group);return function(event){var related=event.relatedTarget;if(!related||related!==this&&!(related.compareDocumentPosition(this)&8)){listener.call(this,event)}}}function contextListener(listener,index,group){return function(event1){var event0=exports.event;// Events can be reentrant (e.g., focus).
exports.event=event1;try{listener.call(this,this.__data__,index,group)}finally{exports.event=event0}}}function parseTypenames(typenames){return typenames.trim().split(/^|\s+/).map(function(t){var name="",i=t.indexOf(".");if(i>=0)name=t.slice(i+1),t=t.slice(0,i);return{type:t,name:name}})}function onRemove(typename){return function(){var on=this.__on;if(!on)return;for(var j=0,i=-1,m=on.length,o;j<m;++j){if(o=on[j],(!typename.type||o.type===typename.type)&&o.name===typename.name){this.removeEventListener(o.type,o.listener,o.capture)}else{on[++i]=o}}if(++i)on.length=i;else delete this.__on}}function onAdd(typename,value,capture){var wrap=filterEvents.hasOwnProperty(typename.type)?filterContextListener:contextListener;return function(d,i,group){var on=this.__on,o,listener=wrap(value,i,group);if(on)for(var j=0,m=on.length;j<m;++j){if((o=on[j]).type===typename.type&&o.name===typename.name){this.removeEventListener(o.type,o.listener,o.capture);this.addEventListener(o.type,o.listener=listener,o.capture=capture);o.value=value;return}}this.addEventListener(typename.type,listener,capture);o={type:typename.type,name:typename.name,value:value,listener:listener,capture:capture};if(!on)this.__on=[o];else on.push(o)}}function selection_on(typename,value,capture){var typenames=parseTypenames(typename+""),i,n=typenames.length,t;if(arguments.length<2){var on=this.node().__on;if(on)for(var j=0,m=on.length,o;j<m;++j){for(i=0,o=on[j];i<n;++i){if((t=typenames[i]).type===o.type&&t.name===o.name){return o.value}}}return}on=value?onAdd:onRemove;if(capture==null)capture=false;for(i=0;i<n;++i)this.each(on(typenames[i],value,capture));return this}function customEvent(event1,listener,that,args){var event0=exports.event;event1.sourceEvent=exports.event;exports.event=event1;try{return listener.apply(that,args)}finally{exports.event=event0}}function dispatchEvent(node,type,params){var window=defaultView(node),event=window.CustomEvent;if(typeof event==="function"){event=new event(type,params)}else{event=window.document.createEvent("Event");if(params)event.initEvent(type,params.bubbles,params.cancelable),event.detail=params.detail;else event.initEvent(type,false,false)}node.dispatchEvent(event)}function dispatchConstant(type,params){return function(){return dispatchEvent(this,type,params)}}function dispatchFunction(type,params){return function(){return dispatchEvent(this,type,params.apply(this,arguments))}}function selection_dispatch(type,params){return this.each((typeof params==="function"?dispatchFunction:dispatchConstant)(type,params))}var root=[null];function Selection(groups,parents){this._groups=groups;this._parents=parents}function selection(){return new Selection([[document.documentElement]],root)}Selection.prototype=selection.prototype={constructor:Selection,select:selection_select,selectAll:selection_selectAll,filter:selection_filter,data:selection_data,enter:selection_enter,exit:selection_exit,join:selection_join,merge:selection_merge,order:selection_order,sort:selection_sort,call:selection_call,nodes:selection_nodes,node:selection_node,size:selection_size,empty:selection_empty,each:selection_each,attr:selection_attr,style:selection_style,property:selection_property,classed:selection_classed,text:selection_text,html:selection_html,raise:selection_raise,lower:selection_lower,append:selection_append,insert:selection_insert,remove:selection_remove,clone:selection_clone,datum:selection_datum,on:selection_on,dispatch:selection_dispatch};function select(selector){return typeof selector==="string"?new Selection([[document.querySelector(selector)]],[document.documentElement]):new Selection([[selector]],root)}function create(name){return select(creator(name).call(document.documentElement))}var nextId=0;function local(){return new Local}function Local(){this._="@"+(++nextId).toString(36)}Local.prototype=local.prototype={constructor:Local,get:function(node){var id=this._;while(!(id in node))if(!(node=node.parentNode))return;return node[id]},set:function(node,value){return node[this._]=value},remove:function(node){return this._ in node&&delete node[this._]},toString:function(){return this._}};function sourceEvent(){var current=exports.event,source;while(source=current.sourceEvent)current=source;return current}function point(node,event){var svg=node.ownerSVGElement||node;if(svg.createSVGPoint){var point=svg.createSVGPoint();point.x=event.clientX,point.y=event.clientY;point=point.matrixTransform(node.getScreenCTM().inverse());return[point.x,point.y]}var rect=node.getBoundingClientRect();return[event.clientX-rect.left-node.clientLeft,event.clientY-rect.top-node.clientTop]}function mouse(node){var event=sourceEvent();if(event.changedTouches)event=event.changedTouches[0];return point(node,event)}function selectAll(selector){return typeof selector==="string"?new Selection([document.querySelectorAll(selector)],[document.documentElement]):new Selection([selector==null?[]:selector],root)}function touch(node,touches,identifier){if(arguments.length<3)identifier=touches,touches=sourceEvent().changedTouches;for(var i=0,n=touches?touches.length:0,touch;i<n;++i){if((touch=touches[i]).identifier===identifier){return point(node,touch)}}return null}function touches(node,touches){if(touches==null)touches=sourceEvent().touches;for(var i=0,n=touches?touches.length:0,points=new Array(n);i<n;++i){points[i]=point(node,touches[i])}return points}exports.clientPoint=point;exports.create=create;exports.creator=creator;exports.customEvent=customEvent;exports.local=local;exports.matcher=matcher;exports.mouse=mouse;exports.namespace=namespace;exports.namespaces=namespaces;exports.select=select;exports.selectAll=selectAll;exports.selection=selection;exports.selector=selector;exports.selectorAll=selectorAll;exports.style=styleValue;exports.touch=touch;exports.touches=touches;exports.window=defaultView;Object.defineProperty(exports,"__esModule",{value:true})})},{}],53:[function(require,module,exports){
// https://d3js.org/d3-shape/ v1.3.7 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports,require("d3-path")):typeof define==="function"&&define.amd?define(["exports","d3-path"],factory):(global=global||self,factory(global.d3=global.d3||{},global.d3))})(this,function(exports,d3Path){"use strict";function constant(x){return function constant(){return x}}var abs=Math.abs;var atan2=Math.atan2;var cos=Math.cos;var max=Math.max;var min=Math.min;var sin=Math.sin;var sqrt=Math.sqrt;var epsilon=1e-12;var pi=Math.PI;var halfPi=pi/2;var tau=2*pi;function acos(x){return x>1?0:x<-1?pi:Math.acos(x)}function asin(x){return x>=1?halfPi:x<=-1?-halfPi:Math.asin(x)}function arcInnerRadius(d){return d.innerRadius}function arcOuterRadius(d){return d.outerRadius}function arcStartAngle(d){return d.startAngle}function arcEndAngle(d){return d.endAngle}function arcPadAngle(d){return d&&d.padAngle;// Note: optional!
}function intersect(x0,y0,x1,y1,x2,y2,x3,y3){var x10=x1-x0,y10=y1-y0,x32=x3-x2,y32=y3-y2,t=y32*x10-x32*y10;if(t*t<epsilon)return;t=(x32*(y0-y2)-y32*(x0-x2))/t;return[x0+t*x10,y0+t*y10]}
// Compute perpendicular offset line of length rc.
// http://mathworld.wolfram.com/Circle-LineIntersection.html
function cornerTangents(x0,y0,x1,y1,r1,rc,cw){var x01=x0-x1,y01=y0-y1,lo=(cw?rc:-rc)/sqrt(x01*x01+y01*y01),ox=lo*y01,oy=-lo*x01,x11=x0+ox,y11=y0+oy,x10=x1+ox,y10=y1+oy,x00=(x11+x10)/2,y00=(y11+y10)/2,dx=x10-x11,dy=y10-y11,d2=dx*dx+dy*dy,r=r1-rc,D=x11*y10-x10*y11,d=(dy<0?-1:1)*sqrt(max(0,r*r*d2-D*D)),cx0=(D*dy-dx*d)/d2,cy0=(-D*dx-dy*d)/d2,cx1=(D*dy+dx*d)/d2,cy1=(-D*dx+dy*d)/d2,dx0=cx0-x00,dy0=cy0-y00,dx1=cx1-x00,dy1=cy1-y00;
// Pick the closer of the two intersection points.
// TODO Is there a faster way to determine which intersection to use?
if(dx0*dx0+dy0*dy0>dx1*dx1+dy1*dy1)cx0=cx1,cy0=cy1;return{cx:cx0,cy:cy0,x01:-ox,y01:-oy,x11:cx0*(r1/r-1),y11:cy0*(r1/r-1)}}function arc(){var innerRadius=arcInnerRadius,outerRadius=arcOuterRadius,cornerRadius=constant(0),padRadius=null,startAngle=arcStartAngle,endAngle=arcEndAngle,padAngle=arcPadAngle,context=null;function arc(){var buffer,r,r0=+innerRadius.apply(this,arguments),r1=+outerRadius.apply(this,arguments),a0=startAngle.apply(this,arguments)-halfPi,a1=endAngle.apply(this,arguments)-halfPi,da=abs(a1-a0),cw=a1>a0;if(!context)context=buffer=d3Path.path();
// Ensure that the outer radius is always larger than the inner radius.
if(r1<r0)r=r1,r1=r0,r0=r;
// Is it a point?
if(!(r1>epsilon))context.moveTo(0,0);
// Or is it a circle or annulus?
else if(da>tau-epsilon){context.moveTo(r1*cos(a0),r1*sin(a0));context.arc(0,0,r1,a0,a1,!cw);if(r0>epsilon){context.moveTo(r0*cos(a1),r0*sin(a1));context.arc(0,0,r0,a1,a0,cw)}}
// Or is it a circular or annular sector?
else{var a01=a0,a11=a1,a00=a0,a10=a1,da0=da,da1=da,ap=padAngle.apply(this,arguments)/2,rp=ap>epsilon&&(padRadius?+padRadius.apply(this,arguments):sqrt(r0*r0+r1*r1)),rc=min(abs(r1-r0)/2,+cornerRadius.apply(this,arguments)),rc0=rc,rc1=rc,t0,t1;
// Apply padding? Note that since r1 ≥ r0, da1 ≥ da0.
if(rp>epsilon){var p0=asin(rp/r0*sin(ap)),p1=asin(rp/r1*sin(ap));if((da0-=p0*2)>epsilon)p0*=cw?1:-1,a00+=p0,a10-=p0;else da0=0,a00=a10=(a0+a1)/2;if((da1-=p1*2)>epsilon)p1*=cw?1:-1,a01+=p1,a11-=p1;else da1=0,a01=a11=(a0+a1)/2}var x01=r1*cos(a01),y01=r1*sin(a01),x10=r0*cos(a10),y10=r0*sin(a10);
// Apply rounded corners?
if(rc>epsilon){var x11=r1*cos(a11),y11=r1*sin(a11),x00=r0*cos(a00),y00=r0*sin(a00),oc;
// Restrict the corner radius according to the sector angle.
if(da<pi&&(oc=intersect(x01,y01,x00,y00,x11,y11,x10,y10))){var ax=x01-oc[0],ay=y01-oc[1],bx=x11-oc[0],by=y11-oc[1],kc=1/sin(acos((ax*bx+ay*by)/(sqrt(ax*ax+ay*ay)*sqrt(bx*bx+by*by)))/2),lc=sqrt(oc[0]*oc[0]+oc[1]*oc[1]);rc0=min(rc,(r0-lc)/(kc-1));rc1=min(rc,(r1-lc)/(kc+1))}}
// Is the sector collapsed to a line?
if(!(da1>epsilon))context.moveTo(x01,y01);
// Does the sector’s outer ring have rounded corners?
else if(rc1>epsilon){t0=cornerTangents(x00,y00,x01,y01,r1,rc1,cw);t1=cornerTangents(x11,y11,x10,y10,r1,rc1,cw);context.moveTo(t0.cx+t0.x01,t0.cy+t0.y01);
// Have the corners merged?
if(rc1<rc)context.arc(t0.cx,t0.cy,rc1,atan2(t0.y01,t0.x01),atan2(t1.y01,t1.x01),!cw);
// Otherwise, draw the two corners and the ring.
else{context.arc(t0.cx,t0.cy,rc1,atan2(t0.y01,t0.x01),atan2(t0.y11,t0.x11),!cw);context.arc(0,0,r1,atan2(t0.cy+t0.y11,t0.cx+t0.x11),atan2(t1.cy+t1.y11,t1.cx+t1.x11),!cw);context.arc(t1.cx,t1.cy,rc1,atan2(t1.y11,t1.x11),atan2(t1.y01,t1.x01),!cw)}}
// Or is the outer ring just a circular arc?
else context.moveTo(x01,y01),context.arc(0,0,r1,a01,a11,!cw);
// Is there no inner ring, and it’s a circular sector?
// Or perhaps it’s an annular sector collapsed due to padding?
if(!(r0>epsilon)||!(da0>epsilon))context.lineTo(x10,y10);
// Does the sector’s inner ring (or point) have rounded corners?
else if(rc0>epsilon){t0=cornerTangents(x10,y10,x11,y11,r0,-rc0,cw);t1=cornerTangents(x01,y01,x00,y00,r0,-rc0,cw);context.lineTo(t0.cx+t0.x01,t0.cy+t0.y01);
// Have the corners merged?
if(rc0<rc)context.arc(t0.cx,t0.cy,rc0,atan2(t0.y01,t0.x01),atan2(t1.y01,t1.x01),!cw);
// Otherwise, draw the two corners and the ring.
else{context.arc(t0.cx,t0.cy,rc0,atan2(t0.y01,t0.x01),atan2(t0.y11,t0.x11),!cw);context.arc(0,0,r0,atan2(t0.cy+t0.y11,t0.cx+t0.x11),atan2(t1.cy+t1.y11,t1.cx+t1.x11),cw);context.arc(t1.cx,t1.cy,rc0,atan2(t1.y11,t1.x11),atan2(t1.y01,t1.x01),!cw)}}
// Or is the inner ring just a circular arc?
else context.arc(0,0,r0,a10,a00,cw)}context.closePath();if(buffer)return context=null,buffer+""||null}arc.centroid=function(){var r=(+innerRadius.apply(this,arguments)+ +outerRadius.apply(this,arguments))/2,a=(+startAngle.apply(this,arguments)+ +endAngle.apply(this,arguments))/2-pi/2;return[cos(a)*r,sin(a)*r]};arc.innerRadius=function(_){return arguments.length?(innerRadius=typeof _==="function"?_:constant(+_),arc):innerRadius};arc.outerRadius=function(_){return arguments.length?(outerRadius=typeof _==="function"?_:constant(+_),arc):outerRadius};arc.cornerRadius=function(_){return arguments.length?(cornerRadius=typeof _==="function"?_:constant(+_),arc):cornerRadius};arc.padRadius=function(_){return arguments.length?(padRadius=_==null?null:typeof _==="function"?_:constant(+_),arc):padRadius};arc.startAngle=function(_){return arguments.length?(startAngle=typeof _==="function"?_:constant(+_),arc):startAngle};arc.endAngle=function(_){return arguments.length?(endAngle=typeof _==="function"?_:constant(+_),arc):endAngle};arc.padAngle=function(_){return arguments.length?(padAngle=typeof _==="function"?_:constant(+_),arc):padAngle};arc.context=function(_){return arguments.length?(context=_==null?null:_,arc):context};return arc}function Linear(context){this._context=context}Linear.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._point=0},lineEnd:function(){if(this._line||this._line!==0&&this._point===1)this._context.closePath();this._line=1-this._line},point:function(x,y){x=+x,y=+y;switch(this._point){case 0:this._point=1;this._line?this._context.lineTo(x,y):this._context.moveTo(x,y);break;case 1:this._point=2;// proceed
default:this._context.lineTo(x,y);break}}};function curveLinear(context){return new Linear(context)}function x(p){return p[0]}function y(p){return p[1]}function line(){var x$1=x,y$1=y,defined=constant(true),context=null,curve=curveLinear,output=null;function line(data){var i,n=data.length,d,defined0=false,buffer;if(context==null)output=curve(buffer=d3Path.path());for(i=0;i<=n;++i){if(!(i<n&&defined(d=data[i],i,data))===defined0){if(defined0=!defined0)output.lineStart();else output.lineEnd()}if(defined0)output.point(+x$1(d,i,data),+y$1(d,i,data))}if(buffer)return output=null,buffer+""||null}line.x=function(_){return arguments.length?(x$1=typeof _==="function"?_:constant(+_),line):x$1};line.y=function(_){return arguments.length?(y$1=typeof _==="function"?_:constant(+_),line):y$1};line.defined=function(_){return arguments.length?(defined=typeof _==="function"?_:constant(!!_),line):defined};line.curve=function(_){return arguments.length?(curve=_,context!=null&&(output=curve(context)),line):curve};line.context=function(_){return arguments.length?(_==null?context=output=null:output=curve(context=_),line):context};return line}function area(){var x0=x,x1=null,y0=constant(0),y1=y,defined=constant(true),context=null,curve=curveLinear,output=null;function area(data){var i,j,k,n=data.length,d,defined0=false,buffer,x0z=new Array(n),y0z=new Array(n);if(context==null)output=curve(buffer=d3Path.path());for(i=0;i<=n;++i){if(!(i<n&&defined(d=data[i],i,data))===defined0){if(defined0=!defined0){j=i;output.areaStart();output.lineStart()}else{output.lineEnd();output.lineStart();for(k=i-1;k>=j;--k){output.point(x0z[k],y0z[k])}output.lineEnd();output.areaEnd()}}if(defined0){x0z[i]=+x0(d,i,data),y0z[i]=+y0(d,i,data);output.point(x1?+x1(d,i,data):x0z[i],y1?+y1(d,i,data):y0z[i])}}if(buffer)return output=null,buffer+""||null}function arealine(){return line().defined(defined).curve(curve).context(context)}area.x=function(_){return arguments.length?(x0=typeof _==="function"?_:constant(+_),x1=null,area):x0};area.x0=function(_){return arguments.length?(x0=typeof _==="function"?_:constant(+_),area):x0};area.x1=function(_){return arguments.length?(x1=_==null?null:typeof _==="function"?_:constant(+_),area):x1};area.y=function(_){return arguments.length?(y0=typeof _==="function"?_:constant(+_),y1=null,area):y0};area.y0=function(_){return arguments.length?(y0=typeof _==="function"?_:constant(+_),area):y0};area.y1=function(_){return arguments.length?(y1=_==null?null:typeof _==="function"?_:constant(+_),area):y1};area.lineX0=area.lineY0=function(){return arealine().x(x0).y(y0)};area.lineY1=function(){return arealine().x(x0).y(y1)};area.lineX1=function(){return arealine().x(x1).y(y0)};area.defined=function(_){return arguments.length?(defined=typeof _==="function"?_:constant(!!_),area):defined};area.curve=function(_){return arguments.length?(curve=_,context!=null&&(output=curve(context)),area):curve};area.context=function(_){return arguments.length?(_==null?context=output=null:output=curve(context=_),area):context};return area}function descending(a,b){return b<a?-1:b>a?1:b>=a?0:NaN}function identity(d){return d}function pie(){var value=identity,sortValues=descending,sort=null,startAngle=constant(0),endAngle=constant(tau),padAngle=constant(0);function pie(data){var i,n=data.length,j,k,sum=0,index=new Array(n),arcs=new Array(n),a0=+startAngle.apply(this,arguments),da=Math.min(tau,Math.max(-tau,endAngle.apply(this,arguments)-a0)),a1,p=Math.min(Math.abs(da)/n,padAngle.apply(this,arguments)),pa=p*(da<0?-1:1),v;for(i=0;i<n;++i){if((v=arcs[index[i]=i]=+value(data[i],i,data))>0){sum+=v}}
// Optionally sort the arcs by previously-computed values or by data.
if(sortValues!=null)index.sort(function(i,j){return sortValues(arcs[i],arcs[j])});else if(sort!=null)index.sort(function(i,j){return sort(data[i],data[j])});
// Compute the arcs! They are stored in the original data's order.
for(i=0,k=sum?(da-n*pa)/sum:0;i<n;++i,a0=a1){j=index[i],v=arcs[j],a1=a0+(v>0?v*k:0)+pa,arcs[j]={data:data[j],index:i,value:v,startAngle:a0,endAngle:a1,padAngle:p}}return arcs}pie.value=function(_){return arguments.length?(value=typeof _==="function"?_:constant(+_),pie):value};pie.sortValues=function(_){return arguments.length?(sortValues=_,sort=null,pie):sortValues};pie.sort=function(_){return arguments.length?(sort=_,sortValues=null,pie):sort};pie.startAngle=function(_){return arguments.length?(startAngle=typeof _==="function"?_:constant(+_),pie):startAngle};pie.endAngle=function(_){return arguments.length?(endAngle=typeof _==="function"?_:constant(+_),pie):endAngle};pie.padAngle=function(_){return arguments.length?(padAngle=typeof _==="function"?_:constant(+_),pie):padAngle};return pie}var curveRadialLinear=curveRadial(curveLinear);function Radial(curve){this._curve=curve}Radial.prototype={areaStart:function(){this._curve.areaStart()},areaEnd:function(){this._curve.areaEnd()},lineStart:function(){this._curve.lineStart()},lineEnd:function(){this._curve.lineEnd()},point:function(a,r){this._curve.point(r*Math.sin(a),r*-Math.cos(a))}};function curveRadial(curve){function radial(context){return new Radial(curve(context))}radial._curve=curve;return radial}function lineRadial(l){var c=l.curve;l.angle=l.x,delete l.x;l.radius=l.y,delete l.y;l.curve=function(_){return arguments.length?c(curveRadial(_)):c()._curve};return l}function lineRadial$1(){return lineRadial(line().curve(curveRadialLinear))}function areaRadial(){var a=area().curve(curveRadialLinear),c=a.curve,x0=a.lineX0,x1=a.lineX1,y0=a.lineY0,y1=a.lineY1;a.angle=a.x,delete a.x;a.startAngle=a.x0,delete a.x0;a.endAngle=a.x1,delete a.x1;a.radius=a.y,delete a.y;a.innerRadius=a.y0,delete a.y0;a.outerRadius=a.y1,delete a.y1;a.lineStartAngle=function(){return lineRadial(x0())},delete a.lineX0;a.lineEndAngle=function(){return lineRadial(x1())},delete a.lineX1;a.lineInnerRadius=function(){return lineRadial(y0())},delete a.lineY0;a.lineOuterRadius=function(){return lineRadial(y1())},delete a.lineY1;a.curve=function(_){return arguments.length?c(curveRadial(_)):c()._curve};return a}function pointRadial(x,y){return[(y=+y)*Math.cos(x-=Math.PI/2),y*Math.sin(x)]}var slice=Array.prototype.slice;function linkSource(d){return d.source}function linkTarget(d){return d.target}function link(curve){var source=linkSource,target=linkTarget,x$1=x,y$1=y,context=null;function link(){var buffer,argv=slice.call(arguments),s=source.apply(this,argv),t=target.apply(this,argv);if(!context)context=buffer=d3Path.path();curve(context,+x$1.apply(this,(argv[0]=s,argv)),+y$1.apply(this,argv),+x$1.apply(this,(argv[0]=t,argv)),+y$1.apply(this,argv));if(buffer)return context=null,buffer+""||null}link.source=function(_){return arguments.length?(source=_,link):source};link.target=function(_){return arguments.length?(target=_,link):target};link.x=function(_){return arguments.length?(x$1=typeof _==="function"?_:constant(+_),link):x$1};link.y=function(_){return arguments.length?(y$1=typeof _==="function"?_:constant(+_),link):y$1};link.context=function(_){return arguments.length?(context=_==null?null:_,link):context};return link}function curveHorizontal(context,x0,y0,x1,y1){context.moveTo(x0,y0);context.bezierCurveTo(x0=(x0+x1)/2,y0,x0,y1,x1,y1)}function curveVertical(context,x0,y0,x1,y1){context.moveTo(x0,y0);context.bezierCurveTo(x0,y0=(y0+y1)/2,x1,y0,x1,y1)}function curveRadial$1(context,x0,y0,x1,y1){var p0=pointRadial(x0,y0),p1=pointRadial(x0,y0=(y0+y1)/2),p2=pointRadial(x1,y0),p3=pointRadial(x1,y1);context.moveTo(p0[0],p0[1]);context.bezierCurveTo(p1[0],p1[1],p2[0],p2[1],p3[0],p3[1])}function linkHorizontal(){return link(curveHorizontal)}function linkVertical(){return link(curveVertical)}function linkRadial(){var l=link(curveRadial$1);l.angle=l.x,delete l.x;l.radius=l.y,delete l.y;return l}var circle={draw:function(context,size){var r=Math.sqrt(size/pi);context.moveTo(r,0);context.arc(0,0,r,0,tau)}};var cross={draw:function(context,size){var r=Math.sqrt(size/5)/2;context.moveTo(-3*r,-r);context.lineTo(-r,-r);context.lineTo(-r,-3*r);context.lineTo(r,-3*r);context.lineTo(r,-r);context.lineTo(3*r,-r);context.lineTo(3*r,r);context.lineTo(r,r);context.lineTo(r,3*r);context.lineTo(-r,3*r);context.lineTo(-r,r);context.lineTo(-3*r,r);context.closePath()}};var tan30=Math.sqrt(1/3),tan30_2=tan30*2;var diamond={draw:function(context,size){var y=Math.sqrt(size/tan30_2),x=y*tan30;context.moveTo(0,-y);context.lineTo(x,0);context.lineTo(0,y);context.lineTo(-x,0);context.closePath()}};var ka=.8908130915292852,kr=Math.sin(pi/10)/Math.sin(7*pi/10),kx=Math.sin(tau/10)*kr,ky=-Math.cos(tau/10)*kr;var star={draw:function(context,size){var r=Math.sqrt(size*ka),x=kx*r,y=ky*r;context.moveTo(0,-r);context.lineTo(x,y);for(var i=1;i<5;++i){var a=tau*i/5,c=Math.cos(a),s=Math.sin(a);context.lineTo(s*r,-c*r);context.lineTo(c*x-s*y,s*x+c*y)}context.closePath()}};var square={draw:function(context,size){var w=Math.sqrt(size),x=-w/2;context.rect(x,x,w,w)}};var sqrt3=Math.sqrt(3);var triangle={draw:function(context,size){var y=-Math.sqrt(size/(sqrt3*3));context.moveTo(0,y*2);context.lineTo(-sqrt3*y,-y);context.lineTo(sqrt3*y,-y);context.closePath()}};var c=-.5,s=Math.sqrt(3)/2,k=1/Math.sqrt(12),a=(k/2+1)*3;var wye={draw:function(context,size){var r=Math.sqrt(size/a),x0=r/2,y0=r*k,x1=x0,y1=r*k+r,x2=-x1,y2=y1;context.moveTo(x0,y0);context.lineTo(x1,y1);context.lineTo(x2,y2);context.lineTo(c*x0-s*y0,s*x0+c*y0);context.lineTo(c*x1-s*y1,s*x1+c*y1);context.lineTo(c*x2-s*y2,s*x2+c*y2);context.lineTo(c*x0+s*y0,c*y0-s*x0);context.lineTo(c*x1+s*y1,c*y1-s*x1);context.lineTo(c*x2+s*y2,c*y2-s*x2);context.closePath()}};var symbols=[circle,cross,diamond,square,star,triangle,wye];function symbol(){var type=constant(circle),size=constant(64),context=null;function symbol(){var buffer;if(!context)context=buffer=d3Path.path();type.apply(this,arguments).draw(context,+size.apply(this,arguments));if(buffer)return context=null,buffer+""||null}symbol.type=function(_){return arguments.length?(type=typeof _==="function"?_:constant(_),symbol):type};symbol.size=function(_){return arguments.length?(size=typeof _==="function"?_:constant(+_),symbol):size};symbol.context=function(_){return arguments.length?(context=_==null?null:_,symbol):context};return symbol}function noop(){}function point(that,x,y){that._context.bezierCurveTo((2*that._x0+that._x1)/3,(2*that._y0+that._y1)/3,(that._x0+2*that._x1)/3,(that._y0+2*that._y1)/3,(that._x0+4*that._x1+x)/6,(that._y0+4*that._y1+y)/6)}function Basis(context){this._context=context}Basis.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._y0=this._y1=NaN;this._point=0},lineEnd:function(){switch(this._point){case 3:point(this,this._x1,this._y1);// proceed
case 2:this._context.lineTo(this._x1,this._y1);break}if(this._line||this._line!==0&&this._point===1)this._context.closePath();this._line=1-this._line},point:function(x,y){x=+x,y=+y;switch(this._point){case 0:this._point=1;this._line?this._context.lineTo(x,y):this._context.moveTo(x,y);break;case 1:this._point=2;break;case 2:this._point=3;this._context.lineTo((5*this._x0+this._x1)/6,(5*this._y0+this._y1)/6);// proceed
default:point(this,x,y);break}this._x0=this._x1,this._x1=x;this._y0=this._y1,this._y1=y}};function basis(context){return new Basis(context)}function BasisClosed(context){this._context=context}BasisClosed.prototype={areaStart:noop,areaEnd:noop,lineStart:function(){this._x0=this._x1=this._x2=this._x3=this._x4=this._y0=this._y1=this._y2=this._y3=this._y4=NaN;this._point=0},lineEnd:function(){switch(this._point){case 1:{this._context.moveTo(this._x2,this._y2);this._context.closePath();break}case 2:{this._context.moveTo((this._x2+2*this._x3)/3,(this._y2+2*this._y3)/3);this._context.lineTo((this._x3+2*this._x2)/3,(this._y3+2*this._y2)/3);this._context.closePath();break}case 3:{this.point(this._x2,this._y2);this.point(this._x3,this._y3);this.point(this._x4,this._y4);break}}},point:function(x,y){x=+x,y=+y;switch(this._point){case 0:this._point=1;this._x2=x,this._y2=y;break;case 1:this._point=2;this._x3=x,this._y3=y;break;case 2:this._point=3;this._x4=x,this._y4=y;this._context.moveTo((this._x0+4*this._x1+x)/6,(this._y0+4*this._y1+y)/6);break;default:point(this,x,y);break}this._x0=this._x1,this._x1=x;this._y0=this._y1,this._y1=y}};function basisClosed(context){return new BasisClosed(context)}function BasisOpen(context){this._context=context}BasisOpen.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._y0=this._y1=NaN;this._point=0},lineEnd:function(){if(this._line||this._line!==0&&this._point===3)this._context.closePath();this._line=1-this._line},point:function(x,y){x=+x,y=+y;switch(this._point){case 0:this._point=1;break;case 1:this._point=2;break;case 2:this._point=3;var x0=(this._x0+4*this._x1+x)/6,y0=(this._y0+4*this._y1+y)/6;this._line?this._context.lineTo(x0,y0):this._context.moveTo(x0,y0);break;case 3:this._point=4;// proceed
default:point(this,x,y);break}this._x0=this._x1,this._x1=x;this._y0=this._y1,this._y1=y}};function basisOpen(context){return new BasisOpen(context)}function Bundle(context,beta){this._basis=new Basis(context);this._beta=beta}Bundle.prototype={lineStart:function(){this._x=[];this._y=[];this._basis.lineStart()},lineEnd:function(){var x=this._x,y=this._y,j=x.length-1;if(j>0){var x0=x[0],y0=y[0],dx=x[j]-x0,dy=y[j]-y0,i=-1,t;while(++i<=j){t=i/j;this._basis.point(this._beta*x[i]+(1-this._beta)*(x0+t*dx),this._beta*y[i]+(1-this._beta)*(y0+t*dy))}}this._x=this._y=null;this._basis.lineEnd()},point:function(x,y){this._x.push(+x);this._y.push(+y)}};var bundle=function custom(beta){function bundle(context){return beta===1?new Basis(context):new Bundle(context,beta)}bundle.beta=function(beta){return custom(+beta)};return bundle}(.85);function point$1(that,x,y){that._context.bezierCurveTo(that._x1+that._k*(that._x2-that._x0),that._y1+that._k*(that._y2-that._y0),that._x2+that._k*(that._x1-x),that._y2+that._k*(that._y1-y),that._x2,that._y2)}function Cardinal(context,tension){this._context=context;this._k=(1-tension)/6}Cardinal.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN;this._point=0},lineEnd:function(){switch(this._point){case 2:this._context.lineTo(this._x2,this._y2);break;case 3:point$1(this,this._x1,this._y1);break}if(this._line||this._line!==0&&this._point===1)this._context.closePath();this._line=1-this._line},point:function(x,y){x=+x,y=+y;switch(this._point){case 0:this._point=1;this._line?this._context.lineTo(x,y):this._context.moveTo(x,y);break;case 1:this._point=2;this._x1=x,this._y1=y;break;case 2:this._point=3;// proceed
default:point$1(this,x,y);break}this._x0=this._x1,this._x1=this._x2,this._x2=x;this._y0=this._y1,this._y1=this._y2,this._y2=y}};var cardinal=function custom(tension){function cardinal(context){return new Cardinal(context,tension)}cardinal.tension=function(tension){return custom(+tension)};return cardinal}(0);function CardinalClosed(context,tension){this._context=context;this._k=(1-tension)/6}CardinalClosed.prototype={areaStart:noop,areaEnd:noop,lineStart:function(){this._x0=this._x1=this._x2=this._x3=this._x4=this._x5=this._y0=this._y1=this._y2=this._y3=this._y4=this._y5=NaN;this._point=0},lineEnd:function(){switch(this._point){case 1:{this._context.moveTo(this._x3,this._y3);this._context.closePath();break}case 2:{this._context.lineTo(this._x3,this._y3);this._context.closePath();break}case 3:{this.point(this._x3,this._y3);this.point(this._x4,this._y4);this.point(this._x5,this._y5);break}}},point:function(x,y){x=+x,y=+y;switch(this._point){case 0:this._point=1;this._x3=x,this._y3=y;break;case 1:this._point=2;this._context.moveTo(this._x4=x,this._y4=y);break;case 2:this._point=3;this._x5=x,this._y5=y;break;default:point$1(this,x,y);break}this._x0=this._x1,this._x1=this._x2,this._x2=x;this._y0=this._y1,this._y1=this._y2,this._y2=y}};var cardinalClosed=function custom(tension){function cardinal(context){return new CardinalClosed(context,tension)}cardinal.tension=function(tension){return custom(+tension)};return cardinal}(0);function CardinalOpen(context,tension){this._context=context;this._k=(1-tension)/6}CardinalOpen.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN;this._point=0},lineEnd:function(){if(this._line||this._line!==0&&this._point===3)this._context.closePath();this._line=1-this._line},point:function(x,y){x=+x,y=+y;switch(this._point){case 0:this._point=1;break;case 1:this._point=2;break;case 2:this._point=3;this._line?this._context.lineTo(this._x2,this._y2):this._context.moveTo(this._x2,this._y2);break;case 3:this._point=4;// proceed
default:point$1(this,x,y);break}this._x0=this._x1,this._x1=this._x2,this._x2=x;this._y0=this._y1,this._y1=this._y2,this._y2=y}};var cardinalOpen=function custom(tension){function cardinal(context){return new CardinalOpen(context,tension)}cardinal.tension=function(tension){return custom(+tension)};return cardinal}(0);function point$2(that,x,y){var x1=that._x1,y1=that._y1,x2=that._x2,y2=that._y2;if(that._l01_a>epsilon){var a=2*that._l01_2a+3*that._l01_a*that._l12_a+that._l12_2a,n=3*that._l01_a*(that._l01_a+that._l12_a);x1=(x1*a-that._x0*that._l12_2a+that._x2*that._l01_2a)/n;y1=(y1*a-that._y0*that._l12_2a+that._y2*that._l01_2a)/n}if(that._l23_a>epsilon){var b=2*that._l23_2a+3*that._l23_a*that._l12_a+that._l12_2a,m=3*that._l23_a*(that._l23_a+that._l12_a);x2=(x2*b+that._x1*that._l23_2a-x*that._l12_2a)/m;y2=(y2*b+that._y1*that._l23_2a-y*that._l12_2a)/m}that._context.bezierCurveTo(x1,y1,x2,y2,that._x2,that._y2)}function CatmullRom(context,alpha){this._context=context;this._alpha=alpha}CatmullRom.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN;this._l01_a=this._l12_a=this._l23_a=this._l01_2a=this._l12_2a=this._l23_2a=this._point=0},lineEnd:function(){switch(this._point){case 2:this._context.lineTo(this._x2,this._y2);break;case 3:this.point(this._x2,this._y2);break}if(this._line||this._line!==0&&this._point===1)this._context.closePath();this._line=1-this._line},point:function(x,y){x=+x,y=+y;if(this._point){var x23=this._x2-x,y23=this._y2-y;this._l23_a=Math.sqrt(this._l23_2a=Math.pow(x23*x23+y23*y23,this._alpha))}switch(this._point){case 0:this._point=1;this._line?this._context.lineTo(x,y):this._context.moveTo(x,y);break;case 1:this._point=2;break;case 2:this._point=3;// proceed
default:point$2(this,x,y);break}this._l01_a=this._l12_a,this._l12_a=this._l23_a;this._l01_2a=this._l12_2a,this._l12_2a=this._l23_2a;this._x0=this._x1,this._x1=this._x2,this._x2=x;this._y0=this._y1,this._y1=this._y2,this._y2=y}};var catmullRom=function custom(alpha){function catmullRom(context){return alpha?new CatmullRom(context,alpha):new Cardinal(context,0)}catmullRom.alpha=function(alpha){return custom(+alpha)};return catmullRom}(.5);function CatmullRomClosed(context,alpha){this._context=context;this._alpha=alpha}CatmullRomClosed.prototype={areaStart:noop,areaEnd:noop,lineStart:function(){this._x0=this._x1=this._x2=this._x3=this._x4=this._x5=this._y0=this._y1=this._y2=this._y3=this._y4=this._y5=NaN;this._l01_a=this._l12_a=this._l23_a=this._l01_2a=this._l12_2a=this._l23_2a=this._point=0},lineEnd:function(){switch(this._point){case 1:{this._context.moveTo(this._x3,this._y3);this._context.closePath();break}case 2:{this._context.lineTo(this._x3,this._y3);this._context.closePath();break}case 3:{this.point(this._x3,this._y3);this.point(this._x4,this._y4);this.point(this._x5,this._y5);break}}},point:function(x,y){x=+x,y=+y;if(this._point){var x23=this._x2-x,y23=this._y2-y;this._l23_a=Math.sqrt(this._l23_2a=Math.pow(x23*x23+y23*y23,this._alpha))}switch(this._point){case 0:this._point=1;this._x3=x,this._y3=y;break;case 1:this._point=2;this._context.moveTo(this._x4=x,this._y4=y);break;case 2:this._point=3;this._x5=x,this._y5=y;break;default:point$2(this,x,y);break}this._l01_a=this._l12_a,this._l12_a=this._l23_a;this._l01_2a=this._l12_2a,this._l12_2a=this._l23_2a;this._x0=this._x1,this._x1=this._x2,this._x2=x;this._y0=this._y1,this._y1=this._y2,this._y2=y}};var catmullRomClosed=function custom(alpha){function catmullRom(context){return alpha?new CatmullRomClosed(context,alpha):new CardinalClosed(context,0)}catmullRom.alpha=function(alpha){return custom(+alpha)};return catmullRom}(.5);function CatmullRomOpen(context,alpha){this._context=context;this._alpha=alpha}CatmullRomOpen.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN;this._l01_a=this._l12_a=this._l23_a=this._l01_2a=this._l12_2a=this._l23_2a=this._point=0},lineEnd:function(){if(this._line||this._line!==0&&this._point===3)this._context.closePath();this._line=1-this._line},point:function(x,y){x=+x,y=+y;if(this._point){var x23=this._x2-x,y23=this._y2-y;this._l23_a=Math.sqrt(this._l23_2a=Math.pow(x23*x23+y23*y23,this._alpha))}switch(this._point){case 0:this._point=1;break;case 1:this._point=2;break;case 2:this._point=3;this._line?this._context.lineTo(this._x2,this._y2):this._context.moveTo(this._x2,this._y2);break;case 3:this._point=4;// proceed
default:point$2(this,x,y);break}this._l01_a=this._l12_a,this._l12_a=this._l23_a;this._l01_2a=this._l12_2a,this._l12_2a=this._l23_2a;this._x0=this._x1,this._x1=this._x2,this._x2=x;this._y0=this._y1,this._y1=this._y2,this._y2=y}};var catmullRomOpen=function custom(alpha){function catmullRom(context){return alpha?new CatmullRomOpen(context,alpha):new CardinalOpen(context,0)}catmullRom.alpha=function(alpha){return custom(+alpha)};return catmullRom}(.5);function LinearClosed(context){this._context=context}LinearClosed.prototype={areaStart:noop,areaEnd:noop,lineStart:function(){this._point=0},lineEnd:function(){if(this._point)this._context.closePath()},point:function(x,y){x=+x,y=+y;if(this._point)this._context.lineTo(x,y);else this._point=1,this._context.moveTo(x,y)}};function linearClosed(context){return new LinearClosed(context)}function sign(x){return x<0?-1:1}
// Calculate the slopes of the tangents (Hermite-type interpolation) based on
// the following paper: Steffen, M. 1990. A Simple Method for Monotonic
// Interpolation in One Dimension. Astronomy and Astrophysics, Vol. 239, NO.
// NOV(II), P. 443, 1990.
function slope3(that,x2,y2){var h0=that._x1-that._x0,h1=x2-that._x1,s0=(that._y1-that._y0)/(h0||h1<0&&-0),s1=(y2-that._y1)/(h1||h0<0&&-0),p=(s0*h1+s1*h0)/(h0+h1);return(sign(s0)+sign(s1))*Math.min(Math.abs(s0),Math.abs(s1),.5*Math.abs(p))||0}
// Calculate a one-sided slope.
function slope2(that,t){var h=that._x1-that._x0;return h?(3*(that._y1-that._y0)/h-t)/2:t}
// According to https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Representations
// "you can express cubic Hermite interpolation in terms of cubic Bézier curves
// with respect to the four values p0, p0 + m0 / 3, p1 - m1 / 3, p1".
function point$3(that,t0,t1){var x0=that._x0,y0=that._y0,x1=that._x1,y1=that._y1,dx=(x1-x0)/3;that._context.bezierCurveTo(x0+dx,y0+dx*t0,x1-dx,y1-dx*t1,x1,y1)}function MonotoneX(context){this._context=context}MonotoneX.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._y0=this._y1=this._t0=NaN;this._point=0},lineEnd:function(){switch(this._point){case 2:this._context.lineTo(this._x1,this._y1);break;case 3:point$3(this,this._t0,slope2(this,this._t0));break}if(this._line||this._line!==0&&this._point===1)this._context.closePath();this._line=1-this._line},point:function(x,y){var t1=NaN;x=+x,y=+y;if(x===this._x1&&y===this._y1)return;// Ignore coincident points.
switch(this._point){case 0:this._point=1;this._line?this._context.lineTo(x,y):this._context.moveTo(x,y);break;case 1:this._point=2;break;case 2:this._point=3;point$3(this,slope2(this,t1=slope3(this,x,y)),t1);break;default:point$3(this,this._t0,t1=slope3(this,x,y));break}this._x0=this._x1,this._x1=x;this._y0=this._y1,this._y1=y;this._t0=t1}};function MonotoneY(context){this._context=new ReflectContext(context)}(MonotoneY.prototype=Object.create(MonotoneX.prototype)).point=function(x,y){MonotoneX.prototype.point.call(this,y,x)};function ReflectContext(context){this._context=context}ReflectContext.prototype={moveTo:function(x,y){this._context.moveTo(y,x)},closePath:function(){this._context.closePath()},lineTo:function(x,y){this._context.lineTo(y,x)},bezierCurveTo:function(x1,y1,x2,y2,x,y){this._context.bezierCurveTo(y1,x1,y2,x2,y,x)}};function monotoneX(context){return new MonotoneX(context)}function monotoneY(context){return new MonotoneY(context)}function Natural(context){this._context=context}Natural.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x=[];this._y=[]},lineEnd:function(){var x=this._x,y=this._y,n=x.length;if(n){this._line?this._context.lineTo(x[0],y[0]):this._context.moveTo(x[0],y[0]);if(n===2){this._context.lineTo(x[1],y[1])}else{var px=controlPoints(x),py=controlPoints(y);for(var i0=0,i1=1;i1<n;++i0,++i1){this._context.bezierCurveTo(px[0][i0],py[0][i0],px[1][i0],py[1][i0],x[i1],y[i1])}}}if(this._line||this._line!==0&&n===1)this._context.closePath();this._line=1-this._line;this._x=this._y=null},point:function(x,y){this._x.push(+x);this._y.push(+y)}};
// See https://www.particleincell.com/2012/bezier-splines/ for derivation.
function controlPoints(x){var i,n=x.length-1,m,a=new Array(n),b=new Array(n),r=new Array(n);a[0]=0,b[0]=2,r[0]=x[0]+2*x[1];for(i=1;i<n-1;++i)a[i]=1,b[i]=4,r[i]=4*x[i]+2*x[i+1];a[n-1]=2,b[n-1]=7,r[n-1]=8*x[n-1]+x[n];for(i=1;i<n;++i)m=a[i]/b[i-1],b[i]-=m,r[i]-=m*r[i-1];a[n-1]=r[n-1]/b[n-1];for(i=n-2;i>=0;--i)a[i]=(r[i]-a[i+1])/b[i];b[n-1]=(x[n]+a[n-1])/2;for(i=0;i<n-1;++i)b[i]=2*x[i+1]-a[i+1];return[a,b]}function natural(context){return new Natural(context)}function Step(context,t){this._context=context;this._t=t}Step.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x=this._y=NaN;this._point=0},lineEnd:function(){if(0<this._t&&this._t<1&&this._point===2)this._context.lineTo(this._x,this._y);if(this._line||this._line!==0&&this._point===1)this._context.closePath();if(this._line>=0)this._t=1-this._t,this._line=1-this._line},point:function(x,y){x=+x,y=+y;switch(this._point){case 0:this._point=1;this._line?this._context.lineTo(x,y):this._context.moveTo(x,y);break;case 1:this._point=2;// proceed
default:{if(this._t<=0){this._context.lineTo(this._x,y);this._context.lineTo(x,y)}else{var x1=this._x*(1-this._t)+x*this._t;this._context.lineTo(x1,this._y);this._context.lineTo(x1,y)}break}}this._x=x,this._y=y}};function step(context){return new Step(context,.5)}function stepBefore(context){return new Step(context,0)}function stepAfter(context){return new Step(context,1)}function none(series,order){if(!((n=series.length)>1))return;for(var i=1,j,s0,s1=series[order[0]],n,m=s1.length;i<n;++i){s0=s1,s1=series[order[i]];for(j=0;j<m;++j){s1[j][1]+=s1[j][0]=isNaN(s0[j][1])?s0[j][0]:s0[j][1]}}}function none$1(series){var n=series.length,o=new Array(n);while(--n>=0)o[n]=n;return o}function stackValue(d,key){return d[key]}function stack(){var keys=constant([]),order=none$1,offset=none,value=stackValue;function stack(data){var kz=keys.apply(this,arguments),i,m=data.length,n=kz.length,sz=new Array(n),oz;for(i=0;i<n;++i){for(var ki=kz[i],si=sz[i]=new Array(m),j=0,sij;j<m;++j){si[j]=sij=[0,+value(data[j],ki,j,data)];sij.data=data[j]}si.key=ki}for(i=0,oz=order(sz);i<n;++i){sz[oz[i]].index=i}offset(sz,oz);return sz}stack.keys=function(_){return arguments.length?(keys=typeof _==="function"?_:constant(slice.call(_)),stack):keys};stack.value=function(_){return arguments.length?(value=typeof _==="function"?_:constant(+_),stack):value};stack.order=function(_){return arguments.length?(order=_==null?none$1:typeof _==="function"?_:constant(slice.call(_)),stack):order};stack.offset=function(_){return arguments.length?(offset=_==null?none:_,stack):offset};return stack}function expand(series,order){if(!((n=series.length)>0))return;for(var i,n,j=0,m=series[0].length,y;j<m;++j){for(y=i=0;i<n;++i)y+=series[i][j][1]||0;if(y)for(i=0;i<n;++i)series[i][j][1]/=y}none(series,order)}function diverging(series,order){if(!((n=series.length)>0))return;for(var i,j=0,d,dy,yp,yn,n,m=series[order[0]].length;j<m;++j){for(yp=yn=0,i=0;i<n;++i){if((dy=(d=series[order[i]][j])[1]-d[0])>0){d[0]=yp,d[1]=yp+=dy}else if(dy<0){d[1]=yn,d[0]=yn+=dy}else{d[0]=0,d[1]=dy}}}}function silhouette(series,order){if(!((n=series.length)>0))return;for(var j=0,s0=series[order[0]],n,m=s0.length;j<m;++j){for(var i=0,y=0;i<n;++i)y+=series[i][j][1]||0;s0[j][1]+=s0[j][0]=-y/2}none(series,order)}function wiggle(series,order){if(!((n=series.length)>0)||!((m=(s0=series[order[0]]).length)>0))return;for(var y=0,j=1,s0,m,n;j<m;++j){for(var i=0,s1=0,s2=0;i<n;++i){var si=series[order[i]],sij0=si[j][1]||0,sij1=si[j-1][1]||0,s3=(sij0-sij1)/2;for(var k=0;k<i;++k){var sk=series[order[k]],skj0=sk[j][1]||0,skj1=sk[j-1][1]||0;s3+=skj0-skj1}s1+=sij0,s2+=s3*sij0}s0[j-1][1]+=s0[j-1][0]=y;if(s1)y-=s2/s1}s0[j-1][1]+=s0[j-1][0]=y;none(series,order)}function appearance(series){var peaks=series.map(peak);return none$1(series).sort(function(a,b){return peaks[a]-peaks[b]})}function peak(series){var i=-1,j=0,n=series.length,vi,vj=-Infinity;while(++i<n)if((vi=+series[i][1])>vj)vj=vi,j=i;return j}function ascending(series){var sums=series.map(sum);return none$1(series).sort(function(a,b){return sums[a]-sums[b]})}function sum(series){var s=0,i=-1,n=series.length,v;while(++i<n)if(v=+series[i][1])s+=v;return s}function descending$1(series){return ascending(series).reverse()}function insideOut(series){var n=series.length,i,j,sums=series.map(sum),order=appearance(series),top=0,bottom=0,tops=[],bottoms=[];for(i=0;i<n;++i){j=order[i];if(top<bottom){top+=sums[j];tops.push(j)}else{bottom+=sums[j];bottoms.push(j)}}return bottoms.reverse().concat(tops)}function reverse(series){return none$1(series).reverse()}exports.arc=arc;exports.area=area;exports.areaRadial=areaRadial;exports.curveBasis=basis;exports.curveBasisClosed=basisClosed;exports.curveBasisOpen=basisOpen;exports.curveBundle=bundle;exports.curveCardinal=cardinal;exports.curveCardinalClosed=cardinalClosed;exports.curveCardinalOpen=cardinalOpen;exports.curveCatmullRom=catmullRom;exports.curveCatmullRomClosed=catmullRomClosed;exports.curveCatmullRomOpen=catmullRomOpen;exports.curveLinear=curveLinear;exports.curveLinearClosed=linearClosed;exports.curveMonotoneX=monotoneX;exports.curveMonotoneY=monotoneY;exports.curveNatural=natural;exports.curveStep=step;exports.curveStepAfter=stepAfter;exports.curveStepBefore=stepBefore;exports.line=line;exports.lineRadial=lineRadial$1;exports.linkHorizontal=linkHorizontal;exports.linkRadial=linkRadial;exports.linkVertical=linkVertical;exports.pie=pie;exports.pointRadial=pointRadial;exports.radialArea=areaRadial;exports.radialLine=lineRadial$1;exports.stack=stack;exports.stackOffsetDiverging=diverging;exports.stackOffsetExpand=expand;exports.stackOffsetNone=none;exports.stackOffsetSilhouette=silhouette;exports.stackOffsetWiggle=wiggle;exports.stackOrderAppearance=appearance;exports.stackOrderAscending=ascending;exports.stackOrderDescending=descending$1;exports.stackOrderInsideOut=insideOut;exports.stackOrderNone=none$1;exports.stackOrderReverse=reverse;exports.symbol=symbol;exports.symbolCircle=circle;exports.symbolCross=cross;exports.symbolDiamond=diamond;exports.symbolSquare=square;exports.symbolStar=star;exports.symbolTriangle=triangle;exports.symbolWye=wye;exports.symbols=symbols;Object.defineProperty(exports,"__esModule",{value:true})})},{"d3-path":46}],54:[function(require,module,exports){
// https://d3js.org/d3-time-format/ v2.2.2 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports,require("d3-time")):typeof define==="function"&&define.amd?define(["exports","d3-time"],factory):(global=global||self,factory(global.d3=global.d3||{},global.d3))})(this,function(exports,d3Time){"use strict";function localDate(d){if(0<=d.y&&d.y<100){var date=new Date(-1,d.m,d.d,d.H,d.M,d.S,d.L);date.setFullYear(d.y);return date}return new Date(d.y,d.m,d.d,d.H,d.M,d.S,d.L)}function utcDate(d){if(0<=d.y&&d.y<100){var date=new Date(Date.UTC(-1,d.m,d.d,d.H,d.M,d.S,d.L));date.setUTCFullYear(d.y);return date}return new Date(Date.UTC(d.y,d.m,d.d,d.H,d.M,d.S,d.L))}function newDate(y,m,d){return{y:y,m:m,d:d,H:0,M:0,S:0,L:0}}function formatLocale(locale){var locale_dateTime=locale.dateTime,locale_date=locale.date,locale_time=locale.time,locale_periods=locale.periods,locale_weekdays=locale.days,locale_shortWeekdays=locale.shortDays,locale_months=locale.months,locale_shortMonths=locale.shortMonths;var periodRe=formatRe(locale_periods),periodLookup=formatLookup(locale_periods),weekdayRe=formatRe(locale_weekdays),weekdayLookup=formatLookup(locale_weekdays),shortWeekdayRe=formatRe(locale_shortWeekdays),shortWeekdayLookup=formatLookup(locale_shortWeekdays),monthRe=formatRe(locale_months),monthLookup=formatLookup(locale_months),shortMonthRe=formatRe(locale_shortMonths),shortMonthLookup=formatLookup(locale_shortMonths);var formats={a:formatShortWeekday,A:formatWeekday,b:formatShortMonth,B:formatMonth,c:null,d:formatDayOfMonth,e:formatDayOfMonth,f:formatMicroseconds,H:formatHour24,I:formatHour12,j:formatDayOfYear,L:formatMilliseconds,m:formatMonthNumber,M:formatMinutes,p:formatPeriod,q:formatQuarter,Q:formatUnixTimestamp,s:formatUnixTimestampSeconds,S:formatSeconds,u:formatWeekdayNumberMonday,U:formatWeekNumberSunday,V:formatWeekNumberISO,w:formatWeekdayNumberSunday,W:formatWeekNumberMonday,x:null,X:null,y:formatYear,Y:formatFullYear,Z:formatZone,"%":formatLiteralPercent};var utcFormats={a:formatUTCShortWeekday,A:formatUTCWeekday,b:formatUTCShortMonth,B:formatUTCMonth,c:null,d:formatUTCDayOfMonth,e:formatUTCDayOfMonth,f:formatUTCMicroseconds,H:formatUTCHour24,I:formatUTCHour12,j:formatUTCDayOfYear,L:formatUTCMilliseconds,m:formatUTCMonthNumber,M:formatUTCMinutes,p:formatUTCPeriod,q:formatUTCQuarter,Q:formatUnixTimestamp,s:formatUnixTimestampSeconds,S:formatUTCSeconds,u:formatUTCWeekdayNumberMonday,U:formatUTCWeekNumberSunday,V:formatUTCWeekNumberISO,w:formatUTCWeekdayNumberSunday,W:formatUTCWeekNumberMonday,x:null,X:null,y:formatUTCYear,Y:formatUTCFullYear,Z:formatUTCZone,"%":formatLiteralPercent};var parses={a:parseShortWeekday,A:parseWeekday,b:parseShortMonth,B:parseMonth,c:parseLocaleDateTime,d:parseDayOfMonth,e:parseDayOfMonth,f:parseMicroseconds,H:parseHour24,I:parseHour24,j:parseDayOfYear,L:parseMilliseconds,m:parseMonthNumber,M:parseMinutes,p:parsePeriod,q:parseQuarter,Q:parseUnixTimestamp,s:parseUnixTimestampSeconds,S:parseSeconds,u:parseWeekdayNumberMonday,U:parseWeekNumberSunday,V:parseWeekNumberISO,w:parseWeekdayNumberSunday,W:parseWeekNumberMonday,x:parseLocaleDate,X:parseLocaleTime,y:parseYear,Y:parseFullYear,Z:parseZone,"%":parseLiteralPercent};
// These recursive directive definitions must be deferred.
formats.x=newFormat(locale_date,formats);formats.X=newFormat(locale_time,formats);formats.c=newFormat(locale_dateTime,formats);utcFormats.x=newFormat(locale_date,utcFormats);utcFormats.X=newFormat(locale_time,utcFormats);utcFormats.c=newFormat(locale_dateTime,utcFormats);function newFormat(specifier,formats){return function(date){var string=[],i=-1,j=0,n=specifier.length,c,pad,format;if(!(date instanceof Date))date=new Date(+date);while(++i<n){if(specifier.charCodeAt(i)===37){string.push(specifier.slice(j,i));if((pad=pads[c=specifier.charAt(++i)])!=null)c=specifier.charAt(++i);else pad=c==="e"?" ":"0";if(format=formats[c])c=format(date,pad);string.push(c);j=i+1}}string.push(specifier.slice(j,i));return string.join("")}}function newParse(specifier,Z){return function(string){var d=newDate(1900,undefined,1),i=parseSpecifier(d,specifier,string+="",0),week,day;if(i!=string.length)return null;
// If a UNIX timestamp is specified, return it.
if("Q"in d)return new Date(d.Q);if("s"in d)return new Date(d.s*1e3+("L"in d?d.L:0));
// If this is utcParse, never use the local timezone.
if(Z&&!("Z"in d))d.Z=0;
// The am-pm flag is 0 for AM, and 1 for PM.
if("p"in d)d.H=d.H%12+d.p*12;
// If the month was not specified, inherit from the quarter.
if(d.m===undefined)d.m="q"in d?d.q:0;
// Convert day-of-week and week-of-year to day-of-year.
if("V"in d){if(d.V<1||d.V>53)return null;if(!("w"in d))d.w=1;if("Z"in d){week=utcDate(newDate(d.y,0,1)),day=week.getUTCDay();week=day>4||day===0?d3Time.utcMonday.ceil(week):d3Time.utcMonday(week);week=d3Time.utcDay.offset(week,(d.V-1)*7);d.y=week.getUTCFullYear();d.m=week.getUTCMonth();d.d=week.getUTCDate()+(d.w+6)%7}else{week=localDate(newDate(d.y,0,1)),day=week.getDay();week=day>4||day===0?d3Time.timeMonday.ceil(week):d3Time.timeMonday(week);week=d3Time.timeDay.offset(week,(d.V-1)*7);d.y=week.getFullYear();d.m=week.getMonth();d.d=week.getDate()+(d.w+6)%7}}else if("W"in d||"U"in d){if(!("w"in d))d.w="u"in d?d.u%7:"W"in d?1:0;day="Z"in d?utcDate(newDate(d.y,0,1)).getUTCDay():localDate(newDate(d.y,0,1)).getDay();d.m=0;d.d="W"in d?(d.w+6)%7+d.W*7-(day+5)%7:d.w+d.U*7-(day+6)%7}
// If a time zone is specified, all fields are interpreted as UTC and then
// offset according to the specified time zone.
if("Z"in d){d.H+=d.Z/100|0;d.M+=d.Z%100;return utcDate(d)}
// Otherwise, all fields are in local time.
return localDate(d)}}function parseSpecifier(d,specifier,string,j){var i=0,n=specifier.length,m=string.length,c,parse;while(i<n){if(j>=m)return-1;c=specifier.charCodeAt(i++);if(c===37){c=specifier.charAt(i++);parse=parses[c in pads?specifier.charAt(i++):c];if(!parse||(j=parse(d,string,j))<0)return-1}else if(c!=string.charCodeAt(j++)){return-1}}return j}function parsePeriod(d,string,i){var n=periodRe.exec(string.slice(i));return n?(d.p=periodLookup[n[0].toLowerCase()],i+n[0].length):-1}function parseShortWeekday(d,string,i){var n=shortWeekdayRe.exec(string.slice(i));return n?(d.w=shortWeekdayLookup[n[0].toLowerCase()],i+n[0].length):-1}function parseWeekday(d,string,i){var n=weekdayRe.exec(string.slice(i));return n?(d.w=weekdayLookup[n[0].toLowerCase()],i+n[0].length):-1}function parseShortMonth(d,string,i){var n=shortMonthRe.exec(string.slice(i));return n?(d.m=shortMonthLookup[n[0].toLowerCase()],i+n[0].length):-1}function parseMonth(d,string,i){var n=monthRe.exec(string.slice(i));return n?(d.m=monthLookup[n[0].toLowerCase()],i+n[0].length):-1}function parseLocaleDateTime(d,string,i){return parseSpecifier(d,locale_dateTime,string,i)}function parseLocaleDate(d,string,i){return parseSpecifier(d,locale_date,string,i)}function parseLocaleTime(d,string,i){return parseSpecifier(d,locale_time,string,i)}function formatShortWeekday(d){return locale_shortWeekdays[d.getDay()]}function formatWeekday(d){return locale_weekdays[d.getDay()]}function formatShortMonth(d){return locale_shortMonths[d.getMonth()]}function formatMonth(d){return locale_months[d.getMonth()]}function formatPeriod(d){return locale_periods[+(d.getHours()>=12)]}function formatQuarter(d){return 1+~~(d.getMonth()/3)}function formatUTCShortWeekday(d){return locale_shortWeekdays[d.getUTCDay()]}function formatUTCWeekday(d){return locale_weekdays[d.getUTCDay()]}function formatUTCShortMonth(d){return locale_shortMonths[d.getUTCMonth()]}function formatUTCMonth(d){return locale_months[d.getUTCMonth()]}function formatUTCPeriod(d){return locale_periods[+(d.getUTCHours()>=12)]}function formatUTCQuarter(d){return 1+~~(d.getUTCMonth()/3)}return{format:function(specifier){var f=newFormat(specifier+="",formats);f.toString=function(){return specifier};return f},parse:function(specifier){var p=newParse(specifier+="",false);p.toString=function(){return specifier};return p},utcFormat:function(specifier){var f=newFormat(specifier+="",utcFormats);f.toString=function(){return specifier};return f},utcParse:function(specifier){var p=newParse(specifier+="",true);p.toString=function(){return specifier};return p}}}var pads={"-":"",_:" ",0:"0"},numberRe=/^\s*\d+/,// note: ignores next directive
percentRe=/^%/,requoteRe=/[\\^$*+?|[\]().{}]/g;function pad(value,fill,width){var sign=value<0?"-":"",string=(sign?-value:value)+"",length=string.length;return sign+(length<width?new Array(width-length+1).join(fill)+string:string)}function requote(s){return s.replace(requoteRe,"\\$&")}function formatRe(names){return new RegExp("^(?:"+names.map(requote).join("|")+")","i")}function formatLookup(names){var map={},i=-1,n=names.length;while(++i<n)map[names[i].toLowerCase()]=i;return map}function parseWeekdayNumberSunday(d,string,i){var n=numberRe.exec(string.slice(i,i+1));return n?(d.w=+n[0],i+n[0].length):-1}function parseWeekdayNumberMonday(d,string,i){var n=numberRe.exec(string.slice(i,i+1));return n?(d.u=+n[0],i+n[0].length):-1}function parseWeekNumberSunday(d,string,i){var n=numberRe.exec(string.slice(i,i+2));return n?(d.U=+n[0],i+n[0].length):-1}function parseWeekNumberISO(d,string,i){var n=numberRe.exec(string.slice(i,i+2));return n?(d.V=+n[0],i+n[0].length):-1}function parseWeekNumberMonday(d,string,i){var n=numberRe.exec(string.slice(i,i+2));return n?(d.W=+n[0],i+n[0].length):-1}function parseFullYear(d,string,i){var n=numberRe.exec(string.slice(i,i+4));return n?(d.y=+n[0],i+n[0].length):-1}function parseYear(d,string,i){var n=numberRe.exec(string.slice(i,i+2));return n?(d.y=+n[0]+(+n[0]>68?1900:2e3),i+n[0].length):-1}function parseZone(d,string,i){var n=/^(Z)|([+-]\d\d)(?::?(\d\d))?/.exec(string.slice(i,i+6));return n?(d.Z=n[1]?0:-(n[2]+(n[3]||"00")),i+n[0].length):-1}function parseQuarter(d,string,i){var n=numberRe.exec(string.slice(i,i+1));return n?(d.q=n[0]*3-3,i+n[0].length):-1}function parseMonthNumber(d,string,i){var n=numberRe.exec(string.slice(i,i+2));return n?(d.m=n[0]-1,i+n[0].length):-1}function parseDayOfMonth(d,string,i){var n=numberRe.exec(string.slice(i,i+2));return n?(d.d=+n[0],i+n[0].length):-1}function parseDayOfYear(d,string,i){var n=numberRe.exec(string.slice(i,i+3));return n?(d.m=0,d.d=+n[0],i+n[0].length):-1}function parseHour24(d,string,i){var n=numberRe.exec(string.slice(i,i+2));return n?(d.H=+n[0],i+n[0].length):-1}function parseMinutes(d,string,i){var n=numberRe.exec(string.slice(i,i+2));return n?(d.M=+n[0],i+n[0].length):-1}function parseSeconds(d,string,i){var n=numberRe.exec(string.slice(i,i+2));return n?(d.S=+n[0],i+n[0].length):-1}function parseMilliseconds(d,string,i){var n=numberRe.exec(string.slice(i,i+3));return n?(d.L=+n[0],i+n[0].length):-1}function parseMicroseconds(d,string,i){var n=numberRe.exec(string.slice(i,i+6));return n?(d.L=Math.floor(n[0]/1e3),i+n[0].length):-1}function parseLiteralPercent(d,string,i){var n=percentRe.exec(string.slice(i,i+1));return n?i+n[0].length:-1}function parseUnixTimestamp(d,string,i){var n=numberRe.exec(string.slice(i));return n?(d.Q=+n[0],i+n[0].length):-1}function parseUnixTimestampSeconds(d,string,i){var n=numberRe.exec(string.slice(i));return n?(d.s=+n[0],i+n[0].length):-1}function formatDayOfMonth(d,p){return pad(d.getDate(),p,2)}function formatHour24(d,p){return pad(d.getHours(),p,2)}function formatHour12(d,p){return pad(d.getHours()%12||12,p,2)}function formatDayOfYear(d,p){return pad(1+d3Time.timeDay.count(d3Time.timeYear(d),d),p,3)}function formatMilliseconds(d,p){return pad(d.getMilliseconds(),p,3)}function formatMicroseconds(d,p){return formatMilliseconds(d,p)+"000"}function formatMonthNumber(d,p){return pad(d.getMonth()+1,p,2)}function formatMinutes(d,p){return pad(d.getMinutes(),p,2)}function formatSeconds(d,p){return pad(d.getSeconds(),p,2)}function formatWeekdayNumberMonday(d){var day=d.getDay();return day===0?7:day}function formatWeekNumberSunday(d,p){return pad(d3Time.timeSunday.count(d3Time.timeYear(d)-1,d),p,2)}function formatWeekNumberISO(d,p){var day=d.getDay();d=day>=4||day===0?d3Time.timeThursday(d):d3Time.timeThursday.ceil(d);return pad(d3Time.timeThursday.count(d3Time.timeYear(d),d)+(d3Time.timeYear(d).getDay()===4),p,2)}function formatWeekdayNumberSunday(d){return d.getDay()}function formatWeekNumberMonday(d,p){return pad(d3Time.timeMonday.count(d3Time.timeYear(d)-1,d),p,2)}function formatYear(d,p){return pad(d.getFullYear()%100,p,2)}function formatFullYear(d,p){return pad(d.getFullYear()%1e4,p,4)}function formatZone(d){var z=d.getTimezoneOffset();return(z>0?"-":(z*=-1,"+"))+pad(z/60|0,"0",2)+pad(z%60,"0",2)}function formatUTCDayOfMonth(d,p){return pad(d.getUTCDate(),p,2)}function formatUTCHour24(d,p){return pad(d.getUTCHours(),p,2)}function formatUTCHour12(d,p){return pad(d.getUTCHours()%12||12,p,2)}function formatUTCDayOfYear(d,p){return pad(1+d3Time.utcDay.count(d3Time.utcYear(d),d),p,3)}function formatUTCMilliseconds(d,p){return pad(d.getUTCMilliseconds(),p,3)}function formatUTCMicroseconds(d,p){return formatUTCMilliseconds(d,p)+"000"}function formatUTCMonthNumber(d,p){return pad(d.getUTCMonth()+1,p,2)}function formatUTCMinutes(d,p){return pad(d.getUTCMinutes(),p,2)}function formatUTCSeconds(d,p){return pad(d.getUTCSeconds(),p,2)}function formatUTCWeekdayNumberMonday(d){var dow=d.getUTCDay();return dow===0?7:dow}function formatUTCWeekNumberSunday(d,p){return pad(d3Time.utcSunday.count(d3Time.utcYear(d)-1,d),p,2)}function formatUTCWeekNumberISO(d,p){var day=d.getUTCDay();d=day>=4||day===0?d3Time.utcThursday(d):d3Time.utcThursday.ceil(d);return pad(d3Time.utcThursday.count(d3Time.utcYear(d),d)+(d3Time.utcYear(d).getUTCDay()===4),p,2)}function formatUTCWeekdayNumberSunday(d){return d.getUTCDay()}function formatUTCWeekNumberMonday(d,p){return pad(d3Time.utcMonday.count(d3Time.utcYear(d)-1,d),p,2)}function formatUTCYear(d,p){return pad(d.getUTCFullYear()%100,p,2)}function formatUTCFullYear(d,p){return pad(d.getUTCFullYear()%1e4,p,4)}function formatUTCZone(){return"+0000"}function formatLiteralPercent(){return"%"}function formatUnixTimestamp(d){return+d}function formatUnixTimestampSeconds(d){return Math.floor(+d/1e3)}var locale;defaultLocale({dateTime:"%x, %X",date:"%-m/%-d/%Y",time:"%-I:%M:%S %p",periods:["AM","PM"],days:["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"],shortDays:["Sun","Mon","Tue","Wed","Thu","Fri","Sat"],months:["January","February","March","April","May","June","July","August","September","October","November","December"],shortMonths:["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]});function defaultLocale(definition){locale=formatLocale(definition);exports.timeFormat=locale.format;exports.timeParse=locale.parse;exports.utcFormat=locale.utcFormat;exports.utcParse=locale.utcParse;return locale}var isoSpecifier="%Y-%m-%dT%H:%M:%S.%LZ";function formatIsoNative(date){return date.toISOString()}var formatIso=Date.prototype.toISOString?formatIsoNative:exports.utcFormat(isoSpecifier);function parseIsoNative(string){var date=new Date(string);return isNaN(date)?null:date}var parseIso=+new Date("2000-01-01T00:00:00.000Z")?parseIsoNative:exports.utcParse(isoSpecifier);exports.isoFormat=formatIso;exports.isoParse=parseIso;exports.timeFormatDefaultLocale=defaultLocale;exports.timeFormatLocale=formatLocale;Object.defineProperty(exports,"__esModule",{value:true})})},{"d3-time":55}],55:[function(require,module,exports){
// https://d3js.org/d3-time/ v1.1.0 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):(global=global||self,factory(global.d3=global.d3||{}))})(this,function(exports){"use strict";var t0=new Date,t1=new Date;function newInterval(floori,offseti,count,field){function interval(date){return floori(date=arguments.length===0?new Date:new Date(+date)),date}interval.floor=function(date){return floori(date=new Date(+date)),date};interval.ceil=function(date){return floori(date=new Date(date-1)),offseti(date,1),floori(date),date};interval.round=function(date){var d0=interval(date),d1=interval.ceil(date);return date-d0<d1-date?d0:d1};interval.offset=function(date,step){return offseti(date=new Date(+date),step==null?1:Math.floor(step)),date};interval.range=function(start,stop,step){var range=[],previous;start=interval.ceil(start);step=step==null?1:Math.floor(step);if(!(start<stop)||!(step>0))return range;// also handles Invalid Date
do{range.push(previous=new Date(+start)),offseti(start,step),floori(start)}while(previous<start&&start<stop);return range};interval.filter=function(test){return newInterval(function(date){if(date>=date)while(floori(date),!test(date))date.setTime(date-1)},function(date,step){if(date>=date){if(step<0)while(++step<=0){while(offseti(date,-1),!test(date)){}// eslint-disable-line no-empty
}else while(--step>=0){while(offseti(date,+1),!test(date)){}// eslint-disable-line no-empty
}}})};if(count){interval.count=function(start,end){t0.setTime(+start),t1.setTime(+end);floori(t0),floori(t1);return Math.floor(count(t0,t1))};interval.every=function(step){step=Math.floor(step);return!isFinite(step)||!(step>0)?null:!(step>1)?interval:interval.filter(field?function(d){return field(d)%step===0}:function(d){return interval.count(0,d)%step===0})}}return interval}var millisecond=newInterval(function(){
// noop
},function(date,step){date.setTime(+date+step)},function(start,end){return end-start});
// An optimized implementation for this simple case.
millisecond.every=function(k){k=Math.floor(k);if(!isFinite(k)||!(k>0))return null;if(!(k>1))return millisecond;return newInterval(function(date){date.setTime(Math.floor(date/k)*k)},function(date,step){date.setTime(+date+step*k)},function(start,end){return(end-start)/k})};var milliseconds=millisecond.range;var durationSecond=1e3;var durationMinute=6e4;var durationHour=36e5;var durationDay=864e5;var durationWeek=6048e5;var second=newInterval(function(date){date.setTime(date-date.getMilliseconds())},function(date,step){date.setTime(+date+step*durationSecond)},function(start,end){return(end-start)/durationSecond},function(date){return date.getUTCSeconds()});var seconds=second.range;var minute=newInterval(function(date){date.setTime(date-date.getMilliseconds()-date.getSeconds()*durationSecond)},function(date,step){date.setTime(+date+step*durationMinute)},function(start,end){return(end-start)/durationMinute},function(date){return date.getMinutes()});var minutes=minute.range;var hour=newInterval(function(date){date.setTime(date-date.getMilliseconds()-date.getSeconds()*durationSecond-date.getMinutes()*durationMinute)},function(date,step){date.setTime(+date+step*durationHour)},function(start,end){return(end-start)/durationHour},function(date){return date.getHours()});var hours=hour.range;var day=newInterval(function(date){date.setHours(0,0,0,0)},function(date,step){date.setDate(date.getDate()+step)},function(start,end){return(end-start-(end.getTimezoneOffset()-start.getTimezoneOffset())*durationMinute)/durationDay},function(date){return date.getDate()-1});var days=day.range;function weekday(i){return newInterval(function(date){date.setDate(date.getDate()-(date.getDay()+7-i)%7);date.setHours(0,0,0,0)},function(date,step){date.setDate(date.getDate()+step*7)},function(start,end){return(end-start-(end.getTimezoneOffset()-start.getTimezoneOffset())*durationMinute)/durationWeek})}var sunday=weekday(0);var monday=weekday(1);var tuesday=weekday(2);var wednesday=weekday(3);var thursday=weekday(4);var friday=weekday(5);var saturday=weekday(6);var sundays=sunday.range;var mondays=monday.range;var tuesdays=tuesday.range;var wednesdays=wednesday.range;var thursdays=thursday.range;var fridays=friday.range;var saturdays=saturday.range;var month=newInterval(function(date){date.setDate(1);date.setHours(0,0,0,0)},function(date,step){date.setMonth(date.getMonth()+step)},function(start,end){return end.getMonth()-start.getMonth()+(end.getFullYear()-start.getFullYear())*12},function(date){return date.getMonth()});var months=month.range;var year=newInterval(function(date){date.setMonth(0,1);date.setHours(0,0,0,0)},function(date,step){date.setFullYear(date.getFullYear()+step)},function(start,end){return end.getFullYear()-start.getFullYear()},function(date){return date.getFullYear()});
// An optimized implementation for this simple case.
year.every=function(k){return!isFinite(k=Math.floor(k))||!(k>0)?null:newInterval(function(date){date.setFullYear(Math.floor(date.getFullYear()/k)*k);date.setMonth(0,1);date.setHours(0,0,0,0)},function(date,step){date.setFullYear(date.getFullYear()+step*k)})};var years=year.range;var utcMinute=newInterval(function(date){date.setUTCSeconds(0,0)},function(date,step){date.setTime(+date+step*durationMinute)},function(start,end){return(end-start)/durationMinute},function(date){return date.getUTCMinutes()});var utcMinutes=utcMinute.range;var utcHour=newInterval(function(date){date.setUTCMinutes(0,0,0)},function(date,step){date.setTime(+date+step*durationHour)},function(start,end){return(end-start)/durationHour},function(date){return date.getUTCHours()});var utcHours=utcHour.range;var utcDay=newInterval(function(date){date.setUTCHours(0,0,0,0)},function(date,step){date.setUTCDate(date.getUTCDate()+step)},function(start,end){return(end-start)/durationDay},function(date){return date.getUTCDate()-1});var utcDays=utcDay.range;function utcWeekday(i){return newInterval(function(date){date.setUTCDate(date.getUTCDate()-(date.getUTCDay()+7-i)%7);date.setUTCHours(0,0,0,0)},function(date,step){date.setUTCDate(date.getUTCDate()+step*7)},function(start,end){return(end-start)/durationWeek})}var utcSunday=utcWeekday(0);var utcMonday=utcWeekday(1);var utcTuesday=utcWeekday(2);var utcWednesday=utcWeekday(3);var utcThursday=utcWeekday(4);var utcFriday=utcWeekday(5);var utcSaturday=utcWeekday(6);var utcSundays=utcSunday.range;var utcMondays=utcMonday.range;var utcTuesdays=utcTuesday.range;var utcWednesdays=utcWednesday.range;var utcThursdays=utcThursday.range;var utcFridays=utcFriday.range;var utcSaturdays=utcSaturday.range;var utcMonth=newInterval(function(date){date.setUTCDate(1);date.setUTCHours(0,0,0,0)},function(date,step){date.setUTCMonth(date.getUTCMonth()+step)},function(start,end){return end.getUTCMonth()-start.getUTCMonth()+(end.getUTCFullYear()-start.getUTCFullYear())*12},function(date){return date.getUTCMonth()});var utcMonths=utcMonth.range;var utcYear=newInterval(function(date){date.setUTCMonth(0,1);date.setUTCHours(0,0,0,0)},function(date,step){date.setUTCFullYear(date.getUTCFullYear()+step)},function(start,end){return end.getUTCFullYear()-start.getUTCFullYear()},function(date){return date.getUTCFullYear()});
// An optimized implementation for this simple case.
utcYear.every=function(k){return!isFinite(k=Math.floor(k))||!(k>0)?null:newInterval(function(date){date.setUTCFullYear(Math.floor(date.getUTCFullYear()/k)*k);date.setUTCMonth(0,1);date.setUTCHours(0,0,0,0)},function(date,step){date.setUTCFullYear(date.getUTCFullYear()+step*k)})};var utcYears=utcYear.range;exports.timeDay=day;exports.timeDays=days;exports.timeFriday=friday;exports.timeFridays=fridays;exports.timeHour=hour;exports.timeHours=hours;exports.timeInterval=newInterval;exports.timeMillisecond=millisecond;exports.timeMilliseconds=milliseconds;exports.timeMinute=minute;exports.timeMinutes=minutes;exports.timeMonday=monday;exports.timeMondays=mondays;exports.timeMonth=month;exports.timeMonths=months;exports.timeSaturday=saturday;exports.timeSaturdays=saturdays;exports.timeSecond=second;exports.timeSeconds=seconds;exports.timeSunday=sunday;exports.timeSundays=sundays;exports.timeThursday=thursday;exports.timeThursdays=thursdays;exports.timeTuesday=tuesday;exports.timeTuesdays=tuesdays;exports.timeWednesday=wednesday;exports.timeWednesdays=wednesdays;exports.timeWeek=sunday;exports.timeWeeks=sundays;exports.timeYear=year;exports.timeYears=years;exports.utcDay=utcDay;exports.utcDays=utcDays;exports.utcFriday=utcFriday;exports.utcFridays=utcFridays;exports.utcHour=utcHour;exports.utcHours=utcHours;exports.utcMillisecond=millisecond;exports.utcMilliseconds=milliseconds;exports.utcMinute=utcMinute;exports.utcMinutes=utcMinutes;exports.utcMonday=utcMonday;exports.utcMondays=utcMondays;exports.utcMonth=utcMonth;exports.utcMonths=utcMonths;exports.utcSaturday=utcSaturday;exports.utcSaturdays=utcSaturdays;exports.utcSecond=second;exports.utcSeconds=seconds;exports.utcSunday=utcSunday;exports.utcSundays=utcSundays;exports.utcThursday=utcThursday;exports.utcThursdays=utcThursdays;exports.utcTuesday=utcTuesday;exports.utcTuesdays=utcTuesdays;exports.utcWednesday=utcWednesday;exports.utcWednesdays=utcWednesdays;exports.utcWeek=utcSunday;exports.utcWeeks=utcSundays;exports.utcYear=utcYear;exports.utcYears=utcYears;Object.defineProperty(exports,"__esModule",{value:true})})},{}],56:[function(require,module,exports){
// https://d3js.org/d3-timer/ v1.0.10 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):(global=global||self,factory(global.d3=global.d3||{}))})(this,function(exports){"use strict";var frame=0,// is an animation frame pending?
timeout=0,// is a timeout pending?
interval=0,// are any timers active?
pokeDelay=1e3,// how frequently we check for clock skew
taskHead,taskTail,clockLast=0,clockNow=0,clockSkew=0,clock=typeof performance==="object"&&performance.now?performance:Date,setFrame=typeof window==="object"&&window.requestAnimationFrame?window.requestAnimationFrame.bind(window):function(f){setTimeout(f,17)};function now(){return clockNow||(setFrame(clearNow),clockNow=clock.now()+clockSkew)}function clearNow(){clockNow=0}function Timer(){this._call=this._time=this._next=null}Timer.prototype=timer.prototype={constructor:Timer,restart:function(callback,delay,time){if(typeof callback!=="function")throw new TypeError("callback is not a function");time=(time==null?now():+time)+(delay==null?0:+delay);if(!this._next&&taskTail!==this){if(taskTail)taskTail._next=this;else taskHead=this;taskTail=this}this._call=callback;this._time=time;sleep()},stop:function(){if(this._call){this._call=null;this._time=Infinity;sleep()}}};function timer(callback,delay,time){var t=new Timer;t.restart(callback,delay,time);return t}function timerFlush(){now();// Get the current time, if not already set.
++frame;// Pretend we’ve set an alarm, if we haven’t already.
var t=taskHead,e;while(t){if((e=clockNow-t._time)>=0)t._call.call(null,e);t=t._next}--frame}function wake(){clockNow=(clockLast=clock.now())+clockSkew;frame=timeout=0;try{timerFlush()}finally{frame=0;nap();clockNow=0}}function poke(){var now=clock.now(),delay=now-clockLast;if(delay>pokeDelay)clockSkew-=delay,clockLast=now}function nap(){var t0,t1=taskHead,t2,time=Infinity;while(t1){if(t1._call){if(time>t1._time)time=t1._time;t0=t1,t1=t1._next}else{t2=t1._next,t1._next=null;t1=t0?t0._next=t2:taskHead=t2}}taskTail=t0;sleep(time)}function sleep(time){if(frame)return;// Soonest alarm already set, or will be.
if(timeout)timeout=clearTimeout(timeout);var delay=time-clockNow;// Strictly less than if we recomputed clockNow.
if(delay>24){if(time<Infinity)timeout=setTimeout(wake,time-clock.now()-clockSkew);if(interval)interval=clearInterval(interval)}else{if(!interval)clockLast=clock.now(),interval=setInterval(poke,pokeDelay);frame=1,setFrame(wake)}}function timeout$1(callback,delay,time){var t=new Timer;delay=delay==null?0:+delay;t.restart(function(elapsed){t.stop();callback(elapsed+delay)},delay,time);return t}function interval$1(callback,delay,time){var t=new Timer,total=delay;if(delay==null)return t.restart(callback,delay,time),t;delay=+delay,time=time==null?now():+time;t.restart(function tick(elapsed){elapsed+=total;t.restart(tick,total+=delay,time);callback(elapsed)},delay,time);return t}exports.interval=interval$1;exports.now=now;exports.timeout=timeout$1;exports.timer=timer;exports.timerFlush=timerFlush;Object.defineProperty(exports,"__esModule",{value:true})})},{}],57:[function(require,module,exports){
// https://d3js.org/d3-transition/ v1.3.2 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports,require("d3-selection"),require("d3-dispatch"),require("d3-timer"),require("d3-interpolate"),require("d3-color"),require("d3-ease")):typeof define==="function"&&define.amd?define(["exports","d3-selection","d3-dispatch","d3-timer","d3-interpolate","d3-color","d3-ease"],factory):(global=global||self,factory(global.d3=global.d3||{},global.d3,global.d3,global.d3,global.d3,global.d3,global.d3))})(this,function(exports,d3Selection,d3Dispatch,d3Timer,d3Interpolate,d3Color,d3Ease){"use strict";var emptyOn=d3Dispatch.dispatch("start","end","cancel","interrupt");var emptyTween=[];var CREATED=0;var SCHEDULED=1;var STARTING=2;var STARTED=3;var RUNNING=4;var ENDING=5;var ENDED=6;function schedule(node,name,id,index,group,timing){var schedules=node.__transition;if(!schedules)node.__transition={};else if(id in schedules)return;create(node,id,{name:name,index:index,// For context during callback.
group:group,// For context during callback.
on:emptyOn,tween:emptyTween,time:timing.time,delay:timing.delay,duration:timing.duration,ease:timing.ease,timer:null,state:CREATED})}function init(node,id){var schedule=get(node,id);if(schedule.state>CREATED)throw new Error("too late; already scheduled");return schedule}function set(node,id){var schedule=get(node,id);if(schedule.state>STARTED)throw new Error("too late; already running");return schedule}function get(node,id){var schedule=node.__transition;if(!schedule||!(schedule=schedule[id]))throw new Error("transition not found");return schedule}function create(node,id,self){var schedules=node.__transition,tween;
// Initialize the self timer when the transition is created.
// Note the actual delay is not known until the first callback!
schedules[id]=self;self.timer=d3Timer.timer(schedule,0,self.time);function schedule(elapsed){self.state=SCHEDULED;self.timer.restart(start,self.delay,self.time);
// If the elapsed delay is less than our first sleep, start immediately.
if(self.delay<=elapsed)start(elapsed-self.delay)}function start(elapsed){var i,j,n,o;
// If the state is not SCHEDULED, then we previously errored on start.
if(self.state!==SCHEDULED)return stop();for(i in schedules){o=schedules[i];if(o.name!==self.name)continue;
// While this element already has a starting transition during this frame,
// defer starting an interrupting transition until that transition has a
// chance to tick (and possibly end); see d3/d3-transition#54!
if(o.state===STARTED)return d3Timer.timeout(start);
// Interrupt the active transition, if any.
if(o.state===RUNNING){o.state=ENDED;o.timer.stop();o.on.call("interrupt",node,node.__data__,o.index,o.group);delete schedules[i]}
// Cancel any pre-empted transitions.
else if(+i<id){o.state=ENDED;o.timer.stop();o.on.call("cancel",node,node.__data__,o.index,o.group);delete schedules[i]}}
// Defer the first tick to end of the current frame; see d3/d3#1576.
// Note the transition may be canceled after start and before the first tick!
// Note this must be scheduled before the start event; see d3/d3-transition#16!
// Assuming this is successful, subsequent callbacks go straight to tick.
d3Timer.timeout(function(){if(self.state===STARTED){self.state=RUNNING;self.timer.restart(tick,self.delay,self.time);tick(elapsed)}});
// Dispatch the start event.
// Note this must be done before the tween are initialized.
self.state=STARTING;self.on.call("start",node,node.__data__,self.index,self.group);if(self.state!==STARTING)return;// interrupted
self.state=STARTED;
// Initialize the tween, deleting null tween.
tween=new Array(n=self.tween.length);for(i=0,j=-1;i<n;++i){if(o=self.tween[i].value.call(node,node.__data__,self.index,self.group)){tween[++j]=o}}tween.length=j+1}function tick(elapsed){var t=elapsed<self.duration?self.ease.call(null,elapsed/self.duration):(self.timer.restart(stop),self.state=ENDING,1),i=-1,n=tween.length;while(++i<n){tween[i].call(node,t)}
// Dispatch the end event.
if(self.state===ENDING){self.on.call("end",node,node.__data__,self.index,self.group);stop()}}function stop(){self.state=ENDED;self.timer.stop();delete schedules[id];for(var i in schedules)return;// eslint-disable-line no-unused-vars
delete node.__transition}}function interrupt(node,name){var schedules=node.__transition,schedule,active,empty=true,i;if(!schedules)return;name=name==null?null:name+"";for(i in schedules){if((schedule=schedules[i]).name!==name){empty=false;continue}active=schedule.state>STARTING&&schedule.state<ENDING;schedule.state=ENDED;schedule.timer.stop();schedule.on.call(active?"interrupt":"cancel",node,node.__data__,schedule.index,schedule.group);delete schedules[i]}if(empty)delete node.__transition}function selection_interrupt(name){return this.each(function(){interrupt(this,name)})}function tweenRemove(id,name){var tween0,tween1;return function(){var schedule=set(this,id),tween=schedule.tween;
// If this node shared tween with the previous node,
// just assign the updated shared tween and we’re done!
// Otherwise, copy-on-write.
if(tween!==tween0){tween1=tween0=tween;for(var i=0,n=tween1.length;i<n;++i){if(tween1[i].name===name){tween1=tween1.slice();tween1.splice(i,1);break}}}schedule.tween=tween1}}function tweenFunction(id,name,value){var tween0,tween1;if(typeof value!=="function")throw new Error;return function(){var schedule=set(this,id),tween=schedule.tween;
// If this node shared tween with the previous node,
// just assign the updated shared tween and we’re done!
// Otherwise, copy-on-write.
if(tween!==tween0){tween1=(tween0=tween).slice();for(var t={name:name,value:value},i=0,n=tween1.length;i<n;++i){if(tween1[i].name===name){tween1[i]=t;break}}if(i===n)tween1.push(t)}schedule.tween=tween1}}function transition_tween(name,value){var id=this._id;name+="";if(arguments.length<2){var tween=get(this.node(),id).tween;for(var i=0,n=tween.length,t;i<n;++i){if((t=tween[i]).name===name){return t.value}}return null}return this.each((value==null?tweenRemove:tweenFunction)(id,name,value))}function tweenValue(transition,name,value){var id=transition._id;transition.each(function(){var schedule=set(this,id);(schedule.value||(schedule.value={}))[name]=value.apply(this,arguments)});return function(node){return get(node,id).value[name]}}function interpolate(a,b){var c;return(typeof b==="number"?d3Interpolate.interpolateNumber:b instanceof d3Color.color?d3Interpolate.interpolateRgb:(c=d3Color.color(b))?(b=c,d3Interpolate.interpolateRgb):d3Interpolate.interpolateString)(a,b)}function attrRemove(name){return function(){this.removeAttribute(name)}}function attrRemoveNS(fullname){return function(){this.removeAttributeNS(fullname.space,fullname.local)}}function attrConstant(name,interpolate,value1){var string00,string1=value1+"",interpolate0;return function(){var string0=this.getAttribute(name);return string0===string1?null:string0===string00?interpolate0:interpolate0=interpolate(string00=string0,value1)}}function attrConstantNS(fullname,interpolate,value1){var string00,string1=value1+"",interpolate0;return function(){var string0=this.getAttributeNS(fullname.space,fullname.local);return string0===string1?null:string0===string00?interpolate0:interpolate0=interpolate(string00=string0,value1)}}function attrFunction(name,interpolate,value){var string00,string10,interpolate0;return function(){var string0,value1=value(this),string1;if(value1==null)return void this.removeAttribute(name);string0=this.getAttribute(name);string1=value1+"";return string0===string1?null:string0===string00&&string1===string10?interpolate0:(string10=string1,interpolate0=interpolate(string00=string0,value1))}}function attrFunctionNS(fullname,interpolate,value){var string00,string10,interpolate0;return function(){var string0,value1=value(this),string1;if(value1==null)return void this.removeAttributeNS(fullname.space,fullname.local);string0=this.getAttributeNS(fullname.space,fullname.local);string1=value1+"";return string0===string1?null:string0===string00&&string1===string10?interpolate0:(string10=string1,interpolate0=interpolate(string00=string0,value1))}}function transition_attr(name,value){var fullname=d3Selection.namespace(name),i=fullname==="transform"?d3Interpolate.interpolateTransformSvg:interpolate;return this.attrTween(name,typeof value==="function"?(fullname.local?attrFunctionNS:attrFunction)(fullname,i,tweenValue(this,"attr."+name,value)):value==null?(fullname.local?attrRemoveNS:attrRemove)(fullname):(fullname.local?attrConstantNS:attrConstant)(fullname,i,value))}function attrInterpolate(name,i){return function(t){this.setAttribute(name,i.call(this,t))}}function attrInterpolateNS(fullname,i){return function(t){this.setAttributeNS(fullname.space,fullname.local,i.call(this,t))}}function attrTweenNS(fullname,value){var t0,i0;function tween(){var i=value.apply(this,arguments);if(i!==i0)t0=(i0=i)&&attrInterpolateNS(fullname,i);return t0}tween._value=value;return tween}function attrTween(name,value){var t0,i0;function tween(){var i=value.apply(this,arguments);if(i!==i0)t0=(i0=i)&&attrInterpolate(name,i);return t0}tween._value=value;return tween}function transition_attrTween(name,value){var key="attr."+name;if(arguments.length<2)return(key=this.tween(key))&&key._value;if(value==null)return this.tween(key,null);if(typeof value!=="function")throw new Error;var fullname=d3Selection.namespace(name);return this.tween(key,(fullname.local?attrTweenNS:attrTween)(fullname,value))}function delayFunction(id,value){return function(){init(this,id).delay=+value.apply(this,arguments)}}function delayConstant(id,value){return value=+value,function(){init(this,id).delay=value}}function transition_delay(value){var id=this._id;return arguments.length?this.each((typeof value==="function"?delayFunction:delayConstant)(id,value)):get(this.node(),id).delay}function durationFunction(id,value){return function(){set(this,id).duration=+value.apply(this,arguments)}}function durationConstant(id,value){return value=+value,function(){set(this,id).duration=value}}function transition_duration(value){var id=this._id;return arguments.length?this.each((typeof value==="function"?durationFunction:durationConstant)(id,value)):get(this.node(),id).duration}function easeConstant(id,value){if(typeof value!=="function")throw new Error;return function(){set(this,id).ease=value}}function transition_ease(value){var id=this._id;return arguments.length?this.each(easeConstant(id,value)):get(this.node(),id).ease}function transition_filter(match){if(typeof match!=="function")match=d3Selection.matcher(match);for(var groups=this._groups,m=groups.length,subgroups=new Array(m),j=0;j<m;++j){for(var group=groups[j],n=group.length,subgroup=subgroups[j]=[],node,i=0;i<n;++i){if((node=group[i])&&match.call(node,node.__data__,i,group)){subgroup.push(node)}}}return new Transition(subgroups,this._parents,this._name,this._id)}function transition_merge(transition){if(transition._id!==this._id)throw new Error;for(var groups0=this._groups,groups1=transition._groups,m0=groups0.length,m1=groups1.length,m=Math.min(m0,m1),merges=new Array(m0),j=0;j<m;++j){for(var group0=groups0[j],group1=groups1[j],n=group0.length,merge=merges[j]=new Array(n),node,i=0;i<n;++i){if(node=group0[i]||group1[i]){merge[i]=node}}}for(;j<m0;++j){merges[j]=groups0[j]}return new Transition(merges,this._parents,this._name,this._id)}function start(name){return(name+"").trim().split(/^|\s+/).every(function(t){var i=t.indexOf(".");if(i>=0)t=t.slice(0,i);return!t||t==="start"})}function onFunction(id,name,listener){var on0,on1,sit=start(name)?init:set;return function(){var schedule=sit(this,id),on=schedule.on;
// If this node shared a dispatch with the previous node,
// just assign the updated shared dispatch and we’re done!
// Otherwise, copy-on-write.
if(on!==on0)(on1=(on0=on).copy()).on(name,listener);schedule.on=on1}}function transition_on(name,listener){var id=this._id;return arguments.length<2?get(this.node(),id).on.on(name):this.each(onFunction(id,name,listener))}function removeFunction(id){return function(){var parent=this.parentNode;for(var i in this.__transition)if(+i!==id)return;if(parent)parent.removeChild(this)}}function transition_remove(){return this.on("end.remove",removeFunction(this._id))}function transition_select(select){var name=this._name,id=this._id;if(typeof select!=="function")select=d3Selection.selector(select);for(var groups=this._groups,m=groups.length,subgroups=new Array(m),j=0;j<m;++j){for(var group=groups[j],n=group.length,subgroup=subgroups[j]=new Array(n),node,subnode,i=0;i<n;++i){if((node=group[i])&&(subnode=select.call(node,node.__data__,i,group))){if("__data__"in node)subnode.__data__=node.__data__;subgroup[i]=subnode;schedule(subgroup[i],name,id,i,subgroup,get(node,id))}}}return new Transition(subgroups,this._parents,name,id)}function transition_selectAll(select){var name=this._name,id=this._id;if(typeof select!=="function")select=d3Selection.selectorAll(select);for(var groups=this._groups,m=groups.length,subgroups=[],parents=[],j=0;j<m;++j){for(var group=groups[j],n=group.length,node,i=0;i<n;++i){if(node=group[i]){for(var children=select.call(node,node.__data__,i,group),child,inherit=get(node,id),k=0,l=children.length;k<l;++k){if(child=children[k]){schedule(child,name,id,k,children,inherit)}}subgroups.push(children);parents.push(node)}}}return new Transition(subgroups,parents,name,id)}var Selection=d3Selection.selection.prototype.constructor;function transition_selection(){return new Selection(this._groups,this._parents)}function styleNull(name,interpolate){var string00,string10,interpolate0;return function(){var string0=d3Selection.style(this,name),string1=(this.style.removeProperty(name),d3Selection.style(this,name));return string0===string1?null:string0===string00&&string1===string10?interpolate0:interpolate0=interpolate(string00=string0,string10=string1)}}function styleRemove(name){return function(){this.style.removeProperty(name)}}function styleConstant(name,interpolate,value1){var string00,string1=value1+"",interpolate0;return function(){var string0=d3Selection.style(this,name);return string0===string1?null:string0===string00?interpolate0:interpolate0=interpolate(string00=string0,value1)}}function styleFunction(name,interpolate,value){var string00,string10,interpolate0;return function(){var string0=d3Selection.style(this,name),value1=value(this),string1=value1+"";if(value1==null)string1=value1=(this.style.removeProperty(name),d3Selection.style(this,name));return string0===string1?null:string0===string00&&string1===string10?interpolate0:(string10=string1,interpolate0=interpolate(string00=string0,value1))}}function styleMaybeRemove(id,name){var on0,on1,listener0,key="style."+name,event="end."+key,remove;return function(){var schedule=set(this,id),on=schedule.on,listener=schedule.value[key]==null?remove||(remove=styleRemove(name)):undefined;
// If this node shared a dispatch with the previous node,
// just assign the updated shared dispatch and we’re done!
// Otherwise, copy-on-write.
if(on!==on0||listener0!==listener)(on1=(on0=on).copy()).on(event,listener0=listener);schedule.on=on1}}function transition_style(name,value,priority){var i=(name+="")==="transform"?d3Interpolate.interpolateTransformCss:interpolate;return value==null?this.styleTween(name,styleNull(name,i)).on("end.style."+name,styleRemove(name)):typeof value==="function"?this.styleTween(name,styleFunction(name,i,tweenValue(this,"style."+name,value))).each(styleMaybeRemove(this._id,name)):this.styleTween(name,styleConstant(name,i,value),priority).on("end.style."+name,null)}function styleInterpolate(name,i,priority){return function(t){this.style.setProperty(name,i.call(this,t),priority)}}function styleTween(name,value,priority){var t,i0;function tween(){var i=value.apply(this,arguments);if(i!==i0)t=(i0=i)&&styleInterpolate(name,i,priority);return t}tween._value=value;return tween}function transition_styleTween(name,value,priority){var key="style."+(name+="");if(arguments.length<2)return(key=this.tween(key))&&key._value;if(value==null)return this.tween(key,null);if(typeof value!=="function")throw new Error;return this.tween(key,styleTween(name,value,priority==null?"":priority))}function textConstant(value){return function(){this.textContent=value}}function textFunction(value){return function(){var value1=value(this);this.textContent=value1==null?"":value1}}function transition_text(value){return this.tween("text",typeof value==="function"?textFunction(tweenValue(this,"text",value)):textConstant(value==null?"":value+""))}function textInterpolate(i){return function(t){this.textContent=i.call(this,t)}}function textTween(value){var t0,i0;function tween(){var i=value.apply(this,arguments);if(i!==i0)t0=(i0=i)&&textInterpolate(i);return t0}tween._value=value;return tween}function transition_textTween(value){var key="text";if(arguments.length<1)return(key=this.tween(key))&&key._value;if(value==null)return this.tween(key,null);if(typeof value!=="function")throw new Error;return this.tween(key,textTween(value))}function transition_transition(){var name=this._name,id0=this._id,id1=newId();for(var groups=this._groups,m=groups.length,j=0;j<m;++j){for(var group=groups[j],n=group.length,node,i=0;i<n;++i){if(node=group[i]){var inherit=get(node,id0);schedule(node,name,id1,i,group,{time:inherit.time+inherit.delay+inherit.duration,delay:0,duration:inherit.duration,ease:inherit.ease})}}}return new Transition(groups,this._parents,name,id1)}function transition_end(){var on0,on1,that=this,id=that._id,size=that.size();return new Promise(function(resolve,reject){var cancel={value:reject},end={value:function(){if(--size===0)resolve()}};that.each(function(){var schedule=set(this,id),on=schedule.on;
// If this node shared a dispatch with the previous node,
// just assign the updated shared dispatch and we’re done!
// Otherwise, copy-on-write.
if(on!==on0){on1=(on0=on).copy();on1._.cancel.push(cancel);on1._.interrupt.push(cancel);on1._.end.push(end)}schedule.on=on1})})}var id=0;function Transition(groups,parents,name,id){this._groups=groups;this._parents=parents;this._name=name;this._id=id}function transition(name){return d3Selection.selection().transition(name)}function newId(){return++id}var selection_prototype=d3Selection.selection.prototype;Transition.prototype=transition.prototype={constructor:Transition,select:transition_select,selectAll:transition_selectAll,filter:transition_filter,merge:transition_merge,selection:transition_selection,transition:transition_transition,call:selection_prototype.call,nodes:selection_prototype.nodes,node:selection_prototype.node,size:selection_prototype.size,empty:selection_prototype.empty,each:selection_prototype.each,on:transition_on,attr:transition_attr,attrTween:transition_attrTween,style:transition_style,styleTween:transition_styleTween,text:transition_text,textTween:transition_textTween,remove:transition_remove,tween:transition_tween,delay:transition_delay,duration:transition_duration,ease:transition_ease,end:transition_end};var defaultTiming={time:null,// Set on use.
delay:0,duration:250,ease:d3Ease.easeCubicInOut};function inherit(node,id){var timing;while(!(timing=node.__transition)||!(timing=timing[id])){if(!(node=node.parentNode)){return defaultTiming.time=d3Timer.now(),defaultTiming}}return timing}function selection_transition(name){var id,timing;if(name instanceof Transition){id=name._id,name=name._name}else{id=newId(),(timing=defaultTiming).time=d3Timer.now(),name=name==null?null:name+""}for(var groups=this._groups,m=groups.length,j=0;j<m;++j){for(var group=groups[j],n=group.length,node,i=0;i<n;++i){if(node=group[i]){schedule(node,name,id,i,group,timing||inherit(node,id))}}}return new Transition(groups,this._parents,name,id)}d3Selection.selection.prototype.interrupt=selection_interrupt;d3Selection.selection.prototype.transition=selection_transition;var root=[null];function active(node,name){var schedules=node.__transition,schedule,i;if(schedules){name=name==null?null:name+"";for(i in schedules){if((schedule=schedules[i]).state>SCHEDULED&&schedule.name===name){return new Transition([[node]],root,name,+i)}}}return null}exports.active=active;exports.interrupt=interrupt;exports.transition=transition;Object.defineProperty(exports,"__esModule",{value:true})})},{"d3-color":34,"d3-dispatch":36,"d3-ease":39,"d3-interpolate":45,"d3-selection":52,"d3-timer":56}],58:[function(require,module,exports){
// https://d3js.org/d3-voronoi/ v1.1.4 Copyright 2018 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports):typeof define==="function"&&define.amd?define(["exports"],factory):factory(global.d3=global.d3||{})})(this,function(exports){"use strict";function constant(x){return function(){return x}}function x(d){return d[0]}function y(d){return d[1]}function RedBlackTree(){this._=null;// root node
}function RedBlackNode(node){node.U=// parent node
node.C=// color - true for red, false for black
node.L=// left node
node.R=// right node
node.P=// previous node
node.N=null;// next node
}RedBlackTree.prototype={constructor:RedBlackTree,insert:function(after,node){var parent,grandpa,uncle;if(after){node.P=after;node.N=after.N;if(after.N)after.N.P=node;after.N=node;if(after.R){after=after.R;while(after.L)after=after.L;after.L=node}else{after.R=node}parent=after}else if(this._){after=RedBlackFirst(this._);node.P=null;node.N=after;after.P=after.L=node;parent=after}else{node.P=node.N=null;this._=node;parent=null}node.L=node.R=null;node.U=parent;node.C=true;after=node;while(parent&&parent.C){grandpa=parent.U;if(parent===grandpa.L){uncle=grandpa.R;if(uncle&&uncle.C){parent.C=uncle.C=false;grandpa.C=true;after=grandpa}else{if(after===parent.R){RedBlackRotateLeft(this,parent);after=parent;parent=after.U}parent.C=false;grandpa.C=true;RedBlackRotateRight(this,grandpa)}}else{uncle=grandpa.L;if(uncle&&uncle.C){parent.C=uncle.C=false;grandpa.C=true;after=grandpa}else{if(after===parent.L){RedBlackRotateRight(this,parent);after=parent;parent=after.U}parent.C=false;grandpa.C=true;RedBlackRotateLeft(this,grandpa)}}parent=after.U}this._.C=false},remove:function(node){if(node.N)node.N.P=node.P;if(node.P)node.P.N=node.N;node.N=node.P=null;var parent=node.U,sibling,left=node.L,right=node.R,next,red;if(!left)next=right;else if(!right)next=left;else next=RedBlackFirst(right);if(parent){if(parent.L===node)parent.L=next;else parent.R=next}else{this._=next}if(left&&right){red=next.C;next.C=node.C;next.L=left;left.U=next;if(next!==right){parent=next.U;next.U=node.U;node=next.R;parent.L=node;next.R=right;right.U=next}else{next.U=parent;parent=next;node=next.R}}else{red=node.C;node=next}if(node)node.U=parent;if(red)return;if(node&&node.C){node.C=false;return}do{if(node===this._)break;if(node===parent.L){sibling=parent.R;if(sibling.C){sibling.C=false;parent.C=true;RedBlackRotateLeft(this,parent);sibling=parent.R}if(sibling.L&&sibling.L.C||sibling.R&&sibling.R.C){if(!sibling.R||!sibling.R.C){sibling.L.C=false;sibling.C=true;RedBlackRotateRight(this,sibling);sibling=parent.R}sibling.C=parent.C;parent.C=sibling.R.C=false;RedBlackRotateLeft(this,parent);node=this._;break}}else{sibling=parent.L;if(sibling.C){sibling.C=false;parent.C=true;RedBlackRotateRight(this,parent);sibling=parent.L}if(sibling.L&&sibling.L.C||sibling.R&&sibling.R.C){if(!sibling.L||!sibling.L.C){sibling.R.C=false;sibling.C=true;RedBlackRotateLeft(this,sibling);sibling=parent.L}sibling.C=parent.C;parent.C=sibling.L.C=false;RedBlackRotateRight(this,parent);node=this._;break}}sibling.C=true;node=parent;parent=parent.U}while(!node.C);if(node)node.C=false}};function RedBlackRotateLeft(tree,node){var p=node,q=node.R,parent=p.U;if(parent){if(parent.L===p)parent.L=q;else parent.R=q}else{tree._=q}q.U=parent;p.U=q;p.R=q.L;if(p.R)p.R.U=p;q.L=p}function RedBlackRotateRight(tree,node){var p=node,q=node.L,parent=p.U;if(parent){if(parent.L===p)parent.L=q;else parent.R=q}else{tree._=q}q.U=parent;p.U=q;p.L=q.R;if(p.L)p.L.U=p;q.R=p}function RedBlackFirst(node){while(node.L)node=node.L;return node}function createEdge(left,right,v0,v1){var edge=[null,null],index=edges.push(edge)-1;edge.left=left;edge.right=right;if(v0)setEdgeEnd(edge,left,right,v0);if(v1)setEdgeEnd(edge,right,left,v1);cells[left.index].halfedges.push(index);cells[right.index].halfedges.push(index);return edge}function createBorderEdge(left,v0,v1){var edge=[v0,v1];edge.left=left;return edge}function setEdgeEnd(edge,left,right,vertex){if(!edge[0]&&!edge[1]){edge[0]=vertex;edge.left=left;edge.right=right}else if(edge.left===right){edge[1]=vertex}else{edge[0]=vertex}}
// Liang–Barsky line clipping.
function clipEdge(edge,x0,y0,x1,y1){var a=edge[0],b=edge[1],ax=a[0],ay=a[1],bx=b[0],by=b[1],t0=0,t1=1,dx=bx-ax,dy=by-ay,r;r=x0-ax;if(!dx&&r>0)return;r/=dx;if(dx<0){if(r<t0)return;if(r<t1)t1=r}else if(dx>0){if(r>t1)return;if(r>t0)t0=r}r=x1-ax;if(!dx&&r<0)return;r/=dx;if(dx<0){if(r>t1)return;if(r>t0)t0=r}else if(dx>0){if(r<t0)return;if(r<t1)t1=r}r=y0-ay;if(!dy&&r>0)return;r/=dy;if(dy<0){if(r<t0)return;if(r<t1)t1=r}else if(dy>0){if(r>t1)return;if(r>t0)t0=r}r=y1-ay;if(!dy&&r<0)return;r/=dy;if(dy<0){if(r>t1)return;if(r>t0)t0=r}else if(dy>0){if(r<t0)return;if(r<t1)t1=r}if(!(t0>0)&&!(t1<1))return true;// TODO Better check?
if(t0>0)edge[0]=[ax+t0*dx,ay+t0*dy];if(t1<1)edge[1]=[ax+t1*dx,ay+t1*dy];return true}function connectEdge(edge,x0,y0,x1,y1){var v1=edge[1];if(v1)return true;var v0=edge[0],left=edge.left,right=edge.right,lx=left[0],ly=left[1],rx=right[0],ry=right[1],fx=(lx+rx)/2,fy=(ly+ry)/2,fm,fb;if(ry===ly){if(fx<x0||fx>=x1)return;if(lx>rx){if(!v0)v0=[fx,y0];else if(v0[1]>=y1)return;v1=[fx,y1]}else{if(!v0)v0=[fx,y1];else if(v0[1]<y0)return;v1=[fx,y0]}}else{fm=(lx-rx)/(ry-ly);fb=fy-fm*fx;if(fm<-1||fm>1){if(lx>rx){if(!v0)v0=[(y0-fb)/fm,y0];else if(v0[1]>=y1)return;v1=[(y1-fb)/fm,y1]}else{if(!v0)v0=[(y1-fb)/fm,y1];else if(v0[1]<y0)return;v1=[(y0-fb)/fm,y0]}}else{if(ly<ry){if(!v0)v0=[x0,fm*x0+fb];else if(v0[0]>=x1)return;v1=[x1,fm*x1+fb]}else{if(!v0)v0=[x1,fm*x1+fb];else if(v0[0]<x0)return;v1=[x0,fm*x0+fb]}}}edge[0]=v0;edge[1]=v1;return true}function clipEdges(x0,y0,x1,y1){var i=edges.length,edge;while(i--){if(!connectEdge(edge=edges[i],x0,y0,x1,y1)||!clipEdge(edge,x0,y0,x1,y1)||!(Math.abs(edge[0][0]-edge[1][0])>epsilon||Math.abs(edge[0][1]-edge[1][1])>epsilon)){delete edges[i]}}}function createCell(site){return cells[site.index]={site:site,halfedges:[]}}function cellHalfedgeAngle(cell,edge){var site=cell.site,va=edge.left,vb=edge.right;if(site===vb)vb=va,va=site;if(vb)return Math.atan2(vb[1]-va[1],vb[0]-va[0]);if(site===va)va=edge[1],vb=edge[0];else va=edge[0],vb=edge[1];return Math.atan2(va[0]-vb[0],vb[1]-va[1])}function cellHalfedgeStart(cell,edge){return edge[+(edge.left!==cell.site)]}function cellHalfedgeEnd(cell,edge){return edge[+(edge.left===cell.site)]}function sortCellHalfedges(){for(var i=0,n=cells.length,cell,halfedges,j,m;i<n;++i){if((cell=cells[i])&&(m=(halfedges=cell.halfedges).length)){var index=new Array(m),array=new Array(m);for(j=0;j<m;++j)index[j]=j,array[j]=cellHalfedgeAngle(cell,edges[halfedges[j]]);index.sort(function(i,j){return array[j]-array[i]});for(j=0;j<m;++j)array[j]=halfedges[index[j]];for(j=0;j<m;++j)halfedges[j]=array[j]}}}function clipCells(x0,y0,x1,y1){var nCells=cells.length,iCell,cell,site,iHalfedge,halfedges,nHalfedges,start,startX,startY,end,endX,endY,cover=true;for(iCell=0;iCell<nCells;++iCell){if(cell=cells[iCell]){site=cell.site;halfedges=cell.halfedges;iHalfedge=halfedges.length;
// Remove any dangling clipped edges.
while(iHalfedge--){if(!edges[halfedges[iHalfedge]]){halfedges.splice(iHalfedge,1)}}
// Insert any border edges as necessary.
iHalfedge=0,nHalfedges=halfedges.length;while(iHalfedge<nHalfedges){end=cellHalfedgeEnd(cell,edges[halfedges[iHalfedge]]),endX=end[0],endY=end[1];start=cellHalfedgeStart(cell,edges[halfedges[++iHalfedge%nHalfedges]]),startX=start[0],startY=start[1];if(Math.abs(endX-startX)>epsilon||Math.abs(endY-startY)>epsilon){halfedges.splice(iHalfedge,0,edges.push(createBorderEdge(site,end,Math.abs(endX-x0)<epsilon&&y1-endY>epsilon?[x0,Math.abs(startX-x0)<epsilon?startY:y1]:Math.abs(endY-y1)<epsilon&&x1-endX>epsilon?[Math.abs(startY-y1)<epsilon?startX:x1,y1]:Math.abs(endX-x1)<epsilon&&endY-y0>epsilon?[x1,Math.abs(startX-x1)<epsilon?startY:y0]:Math.abs(endY-y0)<epsilon&&endX-x0>epsilon?[Math.abs(startY-y0)<epsilon?startX:x0,y0]:null))-1);++nHalfedges}}if(nHalfedges)cover=false}}
// If there weren’t any edges, have the closest site cover the extent.
// It doesn’t matter which corner of the extent we measure!
if(cover){var dx,dy,d2,dc=Infinity;for(iCell=0,cover=null;iCell<nCells;++iCell){if(cell=cells[iCell]){site=cell.site;dx=site[0]-x0;dy=site[1]-y0;d2=dx*dx+dy*dy;if(d2<dc)dc=d2,cover=cell}}if(cover){var v00=[x0,y0],v01=[x0,y1],v11=[x1,y1],v10=[x1,y0];cover.halfedges.push(edges.push(createBorderEdge(site=cover.site,v00,v01))-1,edges.push(createBorderEdge(site,v01,v11))-1,edges.push(createBorderEdge(site,v11,v10))-1,edges.push(createBorderEdge(site,v10,v00))-1)}}
// Lastly delete any cells with no edges; these were entirely clipped.
for(iCell=0;iCell<nCells;++iCell){if(cell=cells[iCell]){if(!cell.halfedges.length){delete cells[iCell]}}}}var circlePool=[];var firstCircle;function Circle(){RedBlackNode(this);this.x=this.y=this.arc=this.site=this.cy=null}function attachCircle(arc){var lArc=arc.P,rArc=arc.N;if(!lArc||!rArc)return;var lSite=lArc.site,cSite=arc.site,rSite=rArc.site;if(lSite===rSite)return;var bx=cSite[0],by=cSite[1],ax=lSite[0]-bx,ay=lSite[1]-by,cx=rSite[0]-bx,cy=rSite[1]-by;var d=2*(ax*cy-ay*cx);if(d>=-epsilon2)return;var ha=ax*ax+ay*ay,hc=cx*cx+cy*cy,x=(cy*ha-ay*hc)/d,y=(ax*hc-cx*ha)/d;var circle=circlePool.pop()||new Circle;circle.arc=arc;circle.site=cSite;circle.x=x+bx;circle.y=(circle.cy=y+by)+Math.sqrt(x*x+y*y);// y bottom
arc.circle=circle;var before=null,node=circles._;while(node){if(circle.y<node.y||circle.y===node.y&&circle.x<=node.x){if(node.L)node=node.L;else{before=node.P;break}}else{if(node.R)node=node.R;else{before=node;break}}}circles.insert(before,circle);if(!before)firstCircle=circle}function detachCircle(arc){var circle=arc.circle;if(circle){if(!circle.P)firstCircle=circle.N;circles.remove(circle);circlePool.push(circle);RedBlackNode(circle);arc.circle=null}}var beachPool=[];function Beach(){RedBlackNode(this);this.edge=this.site=this.circle=null}function createBeach(site){var beach=beachPool.pop()||new Beach;beach.site=site;return beach}function detachBeach(beach){detachCircle(beach);beaches.remove(beach);beachPool.push(beach);RedBlackNode(beach)}function removeBeach(beach){var circle=beach.circle,x=circle.x,y=circle.cy,vertex=[x,y],previous=beach.P,next=beach.N,disappearing=[beach];detachBeach(beach);var lArc=previous;while(lArc.circle&&Math.abs(x-lArc.circle.x)<epsilon&&Math.abs(y-lArc.circle.cy)<epsilon){previous=lArc.P;disappearing.unshift(lArc);detachBeach(lArc);lArc=previous}disappearing.unshift(lArc);detachCircle(lArc);var rArc=next;while(rArc.circle&&Math.abs(x-rArc.circle.x)<epsilon&&Math.abs(y-rArc.circle.cy)<epsilon){next=rArc.N;disappearing.push(rArc);detachBeach(rArc);rArc=next}disappearing.push(rArc);detachCircle(rArc);var nArcs=disappearing.length,iArc;for(iArc=1;iArc<nArcs;++iArc){rArc=disappearing[iArc];lArc=disappearing[iArc-1];setEdgeEnd(rArc.edge,lArc.site,rArc.site,vertex)}lArc=disappearing[0];rArc=disappearing[nArcs-1];rArc.edge=createEdge(lArc.site,rArc.site,null,vertex);attachCircle(lArc);attachCircle(rArc)}function addBeach(site){var x=site[0],directrix=site[1],lArc,rArc,dxl,dxr,node=beaches._;while(node){dxl=leftBreakPoint(node,directrix)-x;if(dxl>epsilon)node=node.L;else{dxr=x-rightBreakPoint(node,directrix);if(dxr>epsilon){if(!node.R){lArc=node;break}node=node.R}else{if(dxl>-epsilon){lArc=node.P;rArc=node}else if(dxr>-epsilon){lArc=node;rArc=node.N}else{lArc=rArc=node}break}}}createCell(site);var newArc=createBeach(site);beaches.insert(lArc,newArc);if(!lArc&&!rArc)return;if(lArc===rArc){detachCircle(lArc);rArc=createBeach(lArc.site);beaches.insert(newArc,rArc);newArc.edge=rArc.edge=createEdge(lArc.site,newArc.site);attachCircle(lArc);attachCircle(rArc);return}if(!rArc){// && lArc
newArc.edge=createEdge(lArc.site,newArc.site);return}
// else lArc !== rArc
detachCircle(lArc);detachCircle(rArc);var lSite=lArc.site,ax=lSite[0],ay=lSite[1],bx=site[0]-ax,by=site[1]-ay,rSite=rArc.site,cx=rSite[0]-ax,cy=rSite[1]-ay,d=2*(bx*cy-by*cx),hb=bx*bx+by*by,hc=cx*cx+cy*cy,vertex=[(cy*hb-by*hc)/d+ax,(bx*hc-cx*hb)/d+ay];setEdgeEnd(rArc.edge,lSite,rSite,vertex);newArc.edge=createEdge(lSite,site,null,vertex);rArc.edge=createEdge(site,rSite,null,vertex);attachCircle(lArc);attachCircle(rArc)}function leftBreakPoint(arc,directrix){var site=arc.site,rfocx=site[0],rfocy=site[1],pby2=rfocy-directrix;if(!pby2)return rfocx;var lArc=arc.P;if(!lArc)return-Infinity;site=lArc.site;var lfocx=site[0],lfocy=site[1],plby2=lfocy-directrix;if(!plby2)return lfocx;var hl=lfocx-rfocx,aby2=1/pby2-1/plby2,b=hl/plby2;if(aby2)return(-b+Math.sqrt(b*b-2*aby2*(hl*hl/(-2*plby2)-lfocy+plby2/2+rfocy-pby2/2)))/aby2+rfocx;return(rfocx+lfocx)/2}function rightBreakPoint(arc,directrix){var rArc=arc.N;if(rArc)return leftBreakPoint(rArc,directrix);var site=arc.site;return site[1]===directrix?site[0]:Infinity}var epsilon=1e-6;var epsilon2=1e-12;var beaches;var cells;var circles;var edges;function triangleArea(a,b,c){return(a[0]-c[0])*(b[1]-a[1])-(a[0]-b[0])*(c[1]-a[1])}function lexicographic(a,b){return b[1]-a[1]||b[0]-a[0]}function Diagram(sites,extent){var site=sites.sort(lexicographic).pop(),x,y,circle;edges=[];cells=new Array(sites.length);beaches=new RedBlackTree;circles=new RedBlackTree;while(true){circle=firstCircle;if(site&&(!circle||site[1]<circle.y||site[1]===circle.y&&site[0]<circle.x)){if(site[0]!==x||site[1]!==y){addBeach(site);x=site[0],y=site[1]}site=sites.pop()}else if(circle){removeBeach(circle.arc)}else{break}}sortCellHalfedges();if(extent){var x0=+extent[0][0],y0=+extent[0][1],x1=+extent[1][0],y1=+extent[1][1];clipEdges(x0,y0,x1,y1);clipCells(x0,y0,x1,y1)}this.edges=edges;this.cells=cells;beaches=circles=edges=cells=null}Diagram.prototype={constructor:Diagram,polygons:function(){var edges=this.edges;return this.cells.map(function(cell){var polygon=cell.halfedges.map(function(i){return cellHalfedgeStart(cell,edges[i])});polygon.data=cell.site.data;return polygon})},triangles:function(){var triangles=[],edges=this.edges;this.cells.forEach(function(cell,i){if(!(m=(halfedges=cell.halfedges).length))return;var site=cell.site,halfedges,j=-1,m,s0,e1=edges[halfedges[m-1]],s1=e1.left===site?e1.right:e1.left;while(++j<m){s0=s1;e1=edges[halfedges[j]];s1=e1.left===site?e1.right:e1.left;if(s0&&s1&&i<s0.index&&i<s1.index&&triangleArea(site,s0,s1)<0){triangles.push([site.data,s0.data,s1.data])}}});return triangles},links:function(){return this.edges.filter(function(edge){return edge.right}).map(function(edge){return{source:edge.left.data,target:edge.right.data}})},find:function(x,y,radius){var that=this,i0,i1=that._found||0,n=that.cells.length,cell;
// Use the previously-found cell, or start with an arbitrary one.
while(!(cell=that.cells[i1]))if(++i1>=n)return null;var dx=x-cell.site[0],dy=y-cell.site[1],d2=dx*dx+dy*dy;
// Traverse the half-edges to find a closer cell, if any.
do{cell=that.cells[i0=i1],i1=null;cell.halfedges.forEach(function(e){var edge=that.edges[e],v=edge.left;if((v===cell.site||!v)&&!(v=edge.right))return;var vx=x-v[0],vy=y-v[1],v2=vx*vx+vy*vy;if(v2<d2)d2=v2,i1=v.index})}while(i1!==null);that._found=i0;return radius==null||d2<=radius*radius?cell.site:null}};function voronoi(){var x$$1=x,y$$1=y,extent=null;function voronoi(data){return new Diagram(data.map(function(d,i){var s=[Math.round(x$$1(d,i,data)/epsilon)*epsilon,Math.round(y$$1(d,i,data)/epsilon)*epsilon];s.index=i;s.data=d;return s}),extent)}voronoi.polygons=function(data){return voronoi(data).polygons()};voronoi.links=function(data){return voronoi(data).links()};voronoi.triangles=function(data){return voronoi(data).triangles()};voronoi.x=function(_){return arguments.length?(x$$1=typeof _==="function"?_:constant(+_),voronoi):x$$1};voronoi.y=function(_){return arguments.length?(y$$1=typeof _==="function"?_:constant(+_),voronoi):y$$1};voronoi.extent=function(_){return arguments.length?(extent=_==null?null:[[+_[0][0],+_[0][1]],[+_[1][0],+_[1][1]]],voronoi):extent&&[[extent[0][0],extent[0][1]],[extent[1][0],extent[1][1]]]};voronoi.size=function(_){return arguments.length?(extent=_==null?null:[[0,0],[+_[0],+_[1]]],voronoi):extent&&[extent[1][0]-extent[0][0],extent[1][1]-extent[0][1]]};return voronoi}exports.voronoi=voronoi;Object.defineProperty(exports,"__esModule",{value:true})})},{}],59:[function(require,module,exports){
// https://d3js.org/d3-zoom/ v1.8.3 Copyright 2019 Mike Bostock
(function(global,factory){typeof exports==="object"&&typeof module!=="undefined"?factory(exports,require("d3-dispatch"),require("d3-drag"),require("d3-interpolate"),require("d3-selection"),require("d3-transition")):typeof define==="function"&&define.amd?define(["exports","d3-dispatch","d3-drag","d3-interpolate","d3-selection","d3-transition"],factory):(global=global||self,factory(global.d3=global.d3||{},global.d3,global.d3,global.d3,global.d3,global.d3))})(this,function(exports,d3Dispatch,d3Drag,d3Interpolate,d3Selection,d3Transition){"use strict";function constant(x){return function(){return x}}function ZoomEvent(target,type,transform){this.target=target;this.type=type;this.transform=transform}function Transform(k,x,y){this.k=k;this.x=x;this.y=y}Transform.prototype={constructor:Transform,scale:function(k){return k===1?this:new Transform(this.k*k,this.x,this.y)},translate:function(x,y){return x===0&y===0?this:new Transform(this.k,this.x+this.k*x,this.y+this.k*y)},apply:function(point){return[point[0]*this.k+this.x,point[1]*this.k+this.y]},applyX:function(x){return x*this.k+this.x},applyY:function(y){return y*this.k+this.y},invert:function(location){return[(location[0]-this.x)/this.k,(location[1]-this.y)/this.k]},invertX:function(x){return(x-this.x)/this.k},invertY:function(y){return(y-this.y)/this.k},rescaleX:function(x){return x.copy().domain(x.range().map(this.invertX,this).map(x.invert,x))},rescaleY:function(y){return y.copy().domain(y.range().map(this.invertY,this).map(y.invert,y))},toString:function(){return"translate("+this.x+","+this.y+") scale("+this.k+")"}};var identity=new Transform(1,0,0);transform.prototype=Transform.prototype;function transform(node){while(!node.__zoom)if(!(node=node.parentNode))return identity;return node.__zoom}function nopropagation(){d3Selection.event.stopImmediatePropagation()}function noevent(){d3Selection.event.preventDefault();d3Selection.event.stopImmediatePropagation()}
// Ignore right-click, since that should open the context menu.
function defaultFilter(){return!d3Selection.event.ctrlKey&&!d3Selection.event.button}function defaultExtent(){var e=this;if(e instanceof SVGElement){e=e.ownerSVGElement||e;if(e.hasAttribute("viewBox")){e=e.viewBox.baseVal;return[[e.x,e.y],[e.x+e.width,e.y+e.height]]}return[[0,0],[e.width.baseVal.value,e.height.baseVal.value]]}return[[0,0],[e.clientWidth,e.clientHeight]]}function defaultTransform(){return this.__zoom||identity}function defaultWheelDelta(){return-d3Selection.event.deltaY*(d3Selection.event.deltaMode===1?.05:d3Selection.event.deltaMode?1:.002)}function defaultTouchable(){return navigator.maxTouchPoints||"ontouchstart"in this}function defaultConstrain(transform,extent,translateExtent){var dx0=transform.invertX(extent[0][0])-translateExtent[0][0],dx1=transform.invertX(extent[1][0])-translateExtent[1][0],dy0=transform.invertY(extent[0][1])-translateExtent[0][1],dy1=transform.invertY(extent[1][1])-translateExtent[1][1];return transform.translate(dx1>dx0?(dx0+dx1)/2:Math.min(0,dx0)||Math.max(0,dx1),dy1>dy0?(dy0+dy1)/2:Math.min(0,dy0)||Math.max(0,dy1))}function zoom(){var filter=defaultFilter,extent=defaultExtent,constrain=defaultConstrain,wheelDelta=defaultWheelDelta,touchable=defaultTouchable,scaleExtent=[0,Infinity],translateExtent=[[-Infinity,-Infinity],[Infinity,Infinity]],duration=250,interpolate=d3Interpolate.interpolateZoom,listeners=d3Dispatch.dispatch("start","zoom","end"),touchstarting,touchending,touchDelay=500,wheelDelay=150,clickDistance2=0;function zoom(selection){selection.property("__zoom",defaultTransform).on("wheel.zoom",wheeled).on("mousedown.zoom",mousedowned).on("dblclick.zoom",dblclicked).filter(touchable).on("touchstart.zoom",touchstarted).on("touchmove.zoom",touchmoved).on("touchend.zoom touchcancel.zoom",touchended).style("touch-action","none").style("-webkit-tap-highlight-color","rgba(0,0,0,0)")}zoom.transform=function(collection,transform,point){var selection=collection.selection?collection.selection():collection;selection.property("__zoom",defaultTransform);if(collection!==selection){schedule(collection,transform,point)}else{selection.interrupt().each(function(){gesture(this,arguments).start().zoom(null,typeof transform==="function"?transform.apply(this,arguments):transform).end()})}};zoom.scaleBy=function(selection,k,p){zoom.scaleTo(selection,function(){var k0=this.__zoom.k,k1=typeof k==="function"?k.apply(this,arguments):k;return k0*k1},p)};zoom.scaleTo=function(selection,k,p){zoom.transform(selection,function(){var e=extent.apply(this,arguments),t0=this.__zoom,p0=p==null?centroid(e):typeof p==="function"?p.apply(this,arguments):p,p1=t0.invert(p0),k1=typeof k==="function"?k.apply(this,arguments):k;return constrain(translate(scale(t0,k1),p0,p1),e,translateExtent)},p)};zoom.translateBy=function(selection,x,y){zoom.transform(selection,function(){return constrain(this.__zoom.translate(typeof x==="function"?x.apply(this,arguments):x,typeof y==="function"?y.apply(this,arguments):y),extent.apply(this,arguments),translateExtent)})};zoom.translateTo=function(selection,x,y,p){zoom.transform(selection,function(){var e=extent.apply(this,arguments),t=this.__zoom,p0=p==null?centroid(e):typeof p==="function"?p.apply(this,arguments):p;return constrain(identity.translate(p0[0],p0[1]).scale(t.k).translate(typeof x==="function"?-x.apply(this,arguments):-x,typeof y==="function"?-y.apply(this,arguments):-y),e,translateExtent)},p)};function scale(transform,k){k=Math.max(scaleExtent[0],Math.min(scaleExtent[1],k));return k===transform.k?transform:new Transform(k,transform.x,transform.y)}function translate(transform,p0,p1){var x=p0[0]-p1[0]*transform.k,y=p0[1]-p1[1]*transform.k;return x===transform.x&&y===transform.y?transform:new Transform(transform.k,x,y)}function centroid(extent){return[(+extent[0][0]+ +extent[1][0])/2,(+extent[0][1]+ +extent[1][1])/2]}function schedule(transition,transform,point){transition.on("start.zoom",function(){gesture(this,arguments).start()}).on("interrupt.zoom end.zoom",function(){gesture(this,arguments).end()}).tween("zoom",function(){var that=this,args=arguments,g=gesture(that,args),e=extent.apply(that,args),p=point==null?centroid(e):typeof point==="function"?point.apply(that,args):point,w=Math.max(e[1][0]-e[0][0],e[1][1]-e[0][1]),a=that.__zoom,b=typeof transform==="function"?transform.apply(that,args):transform,i=interpolate(a.invert(p).concat(w/a.k),b.invert(p).concat(w/b.k));return function(t){if(t===1)t=b;// Avoid rounding error on end.
else{var l=i(t),k=w/l[2];t=new Transform(k,p[0]-l[0]*k,p[1]-l[1]*k)}g.zoom(null,t)}})}function gesture(that,args,clean){return!clean&&that.__zooming||new Gesture(that,args)}function Gesture(that,args){this.that=that;this.args=args;this.active=0;this.extent=extent.apply(that,args);this.taps=0}Gesture.prototype={start:function(){if(++this.active===1){this.that.__zooming=this;this.emit("start")}return this},zoom:function(key,transform){if(this.mouse&&key!=="mouse")this.mouse[1]=transform.invert(this.mouse[0]);if(this.touch0&&key!=="touch")this.touch0[1]=transform.invert(this.touch0[0]);if(this.touch1&&key!=="touch")this.touch1[1]=transform.invert(this.touch1[0]);this.that.__zoom=transform;this.emit("zoom");return this},end:function(){if(--this.active===0){delete this.that.__zooming;this.emit("end")}return this},emit:function(type){d3Selection.customEvent(new ZoomEvent(zoom,type,this.that.__zoom),listeners.apply,listeners,[type,this.that,this.args])}};function wheeled(){if(!filter.apply(this,arguments))return;var g=gesture(this,arguments),t=this.__zoom,k=Math.max(scaleExtent[0],Math.min(scaleExtent[1],t.k*Math.pow(2,wheelDelta.apply(this,arguments)))),p=d3Selection.mouse(this);
// If the mouse is in the same location as before, reuse it.
// If there were recent wheel events, reset the wheel idle timeout.
if(g.wheel){if(g.mouse[0][0]!==p[0]||g.mouse[0][1]!==p[1]){g.mouse[1]=t.invert(g.mouse[0]=p)}clearTimeout(g.wheel)}
// If this wheel event won’t trigger a transform change, ignore it.
else if(t.k===k)return;
// Otherwise, capture the mouse point and location at the start.
else{g.mouse=[p,t.invert(p)];d3Transition.interrupt(this);g.start()}noevent();g.wheel=setTimeout(wheelidled,wheelDelay);g.zoom("mouse",constrain(translate(scale(t,k),g.mouse[0],g.mouse[1]),g.extent,translateExtent));function wheelidled(){g.wheel=null;g.end()}}function mousedowned(){if(touchending||!filter.apply(this,arguments))return;var g=gesture(this,arguments,true),v=d3Selection.select(d3Selection.event.view).on("mousemove.zoom",mousemoved,true).on("mouseup.zoom",mouseupped,true),p=d3Selection.mouse(this),x0=d3Selection.event.clientX,y0=d3Selection.event.clientY;d3Drag.dragDisable(d3Selection.event.view);nopropagation();g.mouse=[p,this.__zoom.invert(p)];d3Transition.interrupt(this);g.start();function mousemoved(){noevent();if(!g.moved){var dx=d3Selection.event.clientX-x0,dy=d3Selection.event.clientY-y0;g.moved=dx*dx+dy*dy>clickDistance2}g.zoom("mouse",constrain(translate(g.that.__zoom,g.mouse[0]=d3Selection.mouse(g.that),g.mouse[1]),g.extent,translateExtent))}function mouseupped(){v.on("mousemove.zoom mouseup.zoom",null);d3Drag.dragEnable(d3Selection.event.view,g.moved);noevent();g.end()}}function dblclicked(){if(!filter.apply(this,arguments))return;var t0=this.__zoom,p0=d3Selection.mouse(this),p1=t0.invert(p0),k1=t0.k*(d3Selection.event.shiftKey?.5:2),t1=constrain(translate(scale(t0,k1),p0,p1),extent.apply(this,arguments),translateExtent);noevent();if(duration>0)d3Selection.select(this).transition().duration(duration).call(schedule,t1,p0);else d3Selection.select(this).call(zoom.transform,t1)}function touchstarted(){if(!filter.apply(this,arguments))return;var touches=d3Selection.event.touches,n=touches.length,g=gesture(this,arguments,d3Selection.event.changedTouches.length===n),started,i,t,p;nopropagation();for(i=0;i<n;++i){t=touches[i],p=d3Selection.touch(this,touches,t.identifier);p=[p,this.__zoom.invert(p),t.identifier];if(!g.touch0)g.touch0=p,started=true,g.taps=1+!!touchstarting;else if(!g.touch1&&g.touch0[2]!==p[2])g.touch1=p,g.taps=0}if(touchstarting)touchstarting=clearTimeout(touchstarting);if(started){if(g.taps<2)touchstarting=setTimeout(function(){touchstarting=null},touchDelay);d3Transition.interrupt(this);g.start()}}function touchmoved(){if(!this.__zooming)return;var g=gesture(this,arguments),touches=d3Selection.event.changedTouches,n=touches.length,i,t,p,l;noevent();if(touchstarting)touchstarting=clearTimeout(touchstarting);g.taps=0;for(i=0;i<n;++i){t=touches[i],p=d3Selection.touch(this,touches,t.identifier);if(g.touch0&&g.touch0[2]===t.identifier)g.touch0[0]=p;else if(g.touch1&&g.touch1[2]===t.identifier)g.touch1[0]=p}t=g.that.__zoom;if(g.touch1){var p0=g.touch0[0],l0=g.touch0[1],p1=g.touch1[0],l1=g.touch1[1],dp=(dp=p1[0]-p0[0])*dp+(dp=p1[1]-p0[1])*dp,dl=(dl=l1[0]-l0[0])*dl+(dl=l1[1]-l0[1])*dl;t=scale(t,Math.sqrt(dp/dl));p=[(p0[0]+p1[0])/2,(p0[1]+p1[1])/2];l=[(l0[0]+l1[0])/2,(l0[1]+l1[1])/2]}else if(g.touch0)p=g.touch0[0],l=g.touch0[1];else return;g.zoom("touch",constrain(translate(t,p,l),g.extent,translateExtent))}function touchended(){if(!this.__zooming)return;var g=gesture(this,arguments),touches=d3Selection.event.changedTouches,n=touches.length,i,t;nopropagation();if(touchending)clearTimeout(touchending);touchending=setTimeout(function(){touchending=null},touchDelay);for(i=0;i<n;++i){t=touches[i];if(g.touch0&&g.touch0[2]===t.identifier)delete g.touch0;else if(g.touch1&&g.touch1[2]===t.identifier)delete g.touch1}if(g.touch1&&!g.touch0)g.touch0=g.touch1,delete g.touch1;if(g.touch0)g.touch0[1]=this.__zoom.invert(g.touch0[0]);else{g.end();
// If this was a dbltap, reroute to the (optional) dblclick.zoom handler.
if(g.taps===2){var p=d3Selection.select(this).on("dblclick.zoom");if(p)p.apply(this,arguments)}}}zoom.wheelDelta=function(_){return arguments.length?(wheelDelta=typeof _==="function"?_:constant(+_),zoom):wheelDelta};zoom.filter=function(_){return arguments.length?(filter=typeof _==="function"?_:constant(!!_),zoom):filter};zoom.touchable=function(_){return arguments.length?(touchable=typeof _==="function"?_:constant(!!_),zoom):touchable};zoom.extent=function(_){return arguments.length?(extent=typeof _==="function"?_:constant([[+_[0][0],+_[0][1]],[+_[1][0],+_[1][1]]]),zoom):extent};zoom.scaleExtent=function(_){return arguments.length?(scaleExtent[0]=+_[0],scaleExtent[1]=+_[1],zoom):[scaleExtent[0],scaleExtent[1]]};zoom.translateExtent=function(_){return arguments.length?(translateExtent[0][0]=+_[0][0],translateExtent[1][0]=+_[1][0],translateExtent[0][1]=+_[0][1],translateExtent[1][1]=+_[1][1],zoom):[[translateExtent[0][0],translateExtent[0][1]],[translateExtent[1][0],translateExtent[1][1]]]};zoom.constrain=function(_){return arguments.length?(constrain=_,zoom):constrain};zoom.duration=function(_){return arguments.length?(duration=+_,zoom):duration};zoom.interpolate=function(_){return arguments.length?(interpolate=_,zoom):interpolate};zoom.on=function(){var value=listeners.on.apply(listeners,arguments);return value===listeners?zoom:value};zoom.clickDistance=function(_){return arguments.length?(clickDistance2=(_=+_)*_,zoom):Math.sqrt(clickDistance2)};return zoom}exports.zoom=zoom;exports.zoomIdentity=identity;exports.zoomTransform=transform;Object.defineProperty(exports,"__esModule",{value:true})})},{"d3-dispatch":36,"d3-drag":37,"d3-interpolate":45,"d3-selection":52,"d3-transition":57}],60:[function(require,module,exports){"use strict";Object.defineProperty(exports,"__esModule",{value:true});var d3Array=require("d3-array");var d3Axis=require("d3-axis");var d3Brush=require("d3-brush");var d3Chord=require("d3-chord");var d3Collection=require("d3-collection");var d3Color=require("d3-color");var d3Contour=require("d3-contour");var d3Dispatch=require("d3-dispatch");var d3Drag=require("d3-drag");var d3Dsv=require("d3-dsv");var d3Ease=require("d3-ease");var d3Fetch=require("d3-fetch");var d3Force=require("d3-force");var d3Format=require("d3-format");var d3Geo=require("d3-geo");var d3Hierarchy=require("d3-hierarchy");var d3Interpolate=require("d3-interpolate");var d3Path=require("d3-path");var d3Polygon=require("d3-polygon");var d3Quadtree=require("d3-quadtree");var d3Random=require("d3-random");var d3Scale=require("d3-scale");var d3ScaleChromatic=require("d3-scale-chromatic");var d3Selection=require("d3-selection");var d3Shape=require("d3-shape");var d3Time=require("d3-time");var d3TimeFormat=require("d3-time-format");var d3Timer=require("d3-timer");var d3Transition=require("d3-transition");var d3Voronoi=require("d3-voronoi");var d3Zoom=require("d3-zoom");var version="5.14.2";Object.keys(d3Array).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Array[k]}})});Object.keys(d3Axis).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Axis[k]}})});Object.keys(d3Brush).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Brush[k]}})});Object.keys(d3Chord).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Chord[k]}})});Object.keys(d3Collection).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Collection[k]}})});Object.keys(d3Color).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Color[k]}})});Object.keys(d3Contour).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Contour[k]}})});Object.keys(d3Dispatch).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Dispatch[k]}})});Object.keys(d3Drag).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Drag[k]}})});Object.keys(d3Dsv).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Dsv[k]}})});Object.keys(d3Ease).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Ease[k]}})});Object.keys(d3Fetch).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Fetch[k]}})});Object.keys(d3Force).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Force[k]}})});Object.keys(d3Format).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Format[k]}})});Object.keys(d3Geo).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Geo[k]}})});Object.keys(d3Hierarchy).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Hierarchy[k]}})});Object.keys(d3Interpolate).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Interpolate[k]}})});Object.keys(d3Path).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Path[k]}})});Object.keys(d3Polygon).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Polygon[k]}})});Object.keys(d3Quadtree).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Quadtree[k]}})});Object.keys(d3Random).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Random[k]}})});Object.keys(d3Scale).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Scale[k]}})});Object.keys(d3ScaleChromatic).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3ScaleChromatic[k]}})});Object.keys(d3Selection).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Selection[k]}})});Object.keys(d3Shape).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Shape[k]}})});Object.keys(d3Time).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Time[k]}})});Object.keys(d3TimeFormat).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3TimeFormat[k]}})});Object.keys(d3Timer).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Timer[k]}})});Object.keys(d3Transition).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Transition[k]}})});Object.keys(d3Voronoi).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Voronoi[k]}})});Object.keys(d3Zoom).forEach(function(k){if(k!=="default")Object.defineProperty(exports,k,{enumerable:true,get:function(){return d3Zoom[k]}})});exports.version=version},{"d3-array":29,"d3-axis":30,"d3-brush":31,"d3-chord":32,"d3-collection":33,"d3-color":34,"d3-contour":35,"d3-dispatch":36,"d3-drag":37,"d3-dsv":38,"d3-ease":39,"d3-fetch":40,"d3-force":41,"d3-format":42,"d3-geo":43,"d3-hierarchy":44,"d3-interpolate":45,"d3-path":46,"d3-polygon":47,"d3-quadtree":48,"d3-random":49,"d3-scale":51,"d3-scale-chromatic":50,"d3-selection":52,"d3-shape":53,"d3-time":55,"d3-time-format":54,"d3-timer":56,"d3-transition":57,"d3-voronoi":58,"d3-zoom":59}],61:[function(require,module,exports){
/*
Copyright (c) 2012-2014 Chris Pettitt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
module.exports={graphlib:require("./lib/graphlib"),layout:require("./lib/layout"),debug:require("./lib/debug"),util:{time:require("./lib/util").time,notime:require("./lib/util").notime},version:require("./lib/version")}},{"./lib/debug":66,"./lib/graphlib":67,"./lib/layout":69,"./lib/util":89,"./lib/version":90}],62:[function(require,module,exports){"use strict";var _=require("./lodash");var greedyFAS=require("./greedy-fas");module.exports={run:run,undo:undo};function run(g){var fas=g.graph().acyclicer==="greedy"?greedyFAS(g,weightFn(g)):dfsFAS(g);_.forEach(fas,function(e){var label=g.edge(e);g.removeEdge(e);label.forwardName=e.name;label.reversed=true;g.setEdge(e.w,e.v,label,_.uniqueId("rev"))});function weightFn(g){return function(e){return g.edge(e).weight}}}function dfsFAS(g){var fas=[];var stack={};var visited={};function dfs(v){if(_.has(visited,v)){return}visited[v]=true;stack[v]=true;_.forEach(g.outEdges(v),function(e){if(_.has(stack,e.w)){fas.push(e)}else{dfs(e.w)}});delete stack[v]}_.forEach(g.nodes(),dfs);return fas}function undo(g){_.forEach(g.edges(),function(e){var label=g.edge(e);if(label.reversed){g.removeEdge(e);var forwardName=label.forwardName;delete label.reversed;delete label.forwardName;g.setEdge(e.w,e.v,label,forwardName)}})}},{"./greedy-fas":68,"./lodash":70}],63:[function(require,module,exports){var _=require("./lodash");var util=require("./util");module.exports=addBorderSegments;function addBorderSegments(g){function dfs(v){var children=g.children(v);var node=g.node(v);if(children.length){_.forEach(children,dfs)}if(_.has(node,"minRank")){node.borderLeft=[];node.borderRight=[];for(var rank=node.minRank,maxRank=node.maxRank+1;rank<maxRank;++rank){addBorderNode(g,"borderLeft","_bl",v,node,rank);addBorderNode(g,"borderRight","_br",v,node,rank)}}}_.forEach(g.children(),dfs)}function addBorderNode(g,prop,prefix,sg,sgNode,rank){var label={width:0,height:0,rank:rank,borderType:prop};var prev=sgNode[prop][rank-1];var curr=util.addDummyNode(g,"border",label,prefix);sgNode[prop][rank]=curr;g.setParent(curr,sg);if(prev){g.setEdge(prev,curr,{weight:1})}}},{"./lodash":70,"./util":89}],64:[function(require,module,exports){"use strict";var _=require("./lodash");module.exports={adjust:adjust,undo:undo};function adjust(g){var rankDir=g.graph().rankdir.toLowerCase();if(rankDir==="lr"||rankDir==="rl"){swapWidthHeight(g)}}function undo(g){var rankDir=g.graph().rankdir.toLowerCase();if(rankDir==="bt"||rankDir==="rl"){reverseY(g)}if(rankDir==="lr"||rankDir==="rl"){swapXY(g);swapWidthHeight(g)}}function swapWidthHeight(g){_.forEach(g.nodes(),function(v){swapWidthHeightOne(g.node(v))});_.forEach(g.edges(),function(e){swapWidthHeightOne(g.edge(e))})}function swapWidthHeightOne(attrs){var w=attrs.width;attrs.width=attrs.height;attrs.height=w}function reverseY(g){_.forEach(g.nodes(),function(v){reverseYOne(g.node(v))});_.forEach(g.edges(),function(e){var edge=g.edge(e);_.forEach(edge.points,reverseYOne);if(_.has(edge,"y")){reverseYOne(edge)}})}function reverseYOne(attrs){attrs.y=-attrs.y}function swapXY(g){_.forEach(g.nodes(),function(v){swapXYOne(g.node(v))});_.forEach(g.edges(),function(e){var edge=g.edge(e);_.forEach(edge.points,swapXYOne);if(_.has(edge,"x")){swapXYOne(edge)}})}function swapXYOne(attrs){var x=attrs.x;attrs.x=attrs.y;attrs.y=x}},{"./lodash":70}],65:[function(require,module,exports){
/*
 * Simple doubly linked list implementation derived from Cormen, et al.,
 * "Introduction to Algorithms".
 */
module.exports=List;function List(){var sentinel={};sentinel._next=sentinel._prev=sentinel;this._sentinel=sentinel}List.prototype.dequeue=function(){var sentinel=this._sentinel;var entry=sentinel._prev;if(entry!==sentinel){unlink(entry);return entry}};List.prototype.enqueue=function(entry){var sentinel=this._sentinel;if(entry._prev&&entry._next){unlink(entry)}entry._next=sentinel._next;sentinel._next._prev=entry;sentinel._next=entry;entry._prev=sentinel};List.prototype.toString=function(){var strs=[];var sentinel=this._sentinel;var curr=sentinel._prev;while(curr!==sentinel){strs.push(JSON.stringify(curr,filterOutLinks));curr=curr._prev}return"["+strs.join(", ")+"]"};function unlink(entry){entry._prev._next=entry._next;entry._next._prev=entry._prev;delete entry._next;delete entry._prev}function filterOutLinks(k,v){if(k!=="_next"&&k!=="_prev"){return v}}},{}],66:[function(require,module,exports){var _=require("./lodash");var util=require("./util");var Graph=require("./graphlib").Graph;module.exports={debugOrdering:debugOrdering};
/* istanbul ignore next */function debugOrdering(g){var layerMatrix=util.buildLayerMatrix(g);var h=new Graph({compound:true,multigraph:true}).setGraph({});_.forEach(g.nodes(),function(v){h.setNode(v,{label:v});h.setParent(v,"layer"+g.node(v).rank)});_.forEach(g.edges(),function(e){h.setEdge(e.v,e.w,{},e.name)});_.forEach(layerMatrix,function(layer,i){var layerV="layer"+i;h.setNode(layerV,{rank:"same"});_.reduce(layer,function(u,v){h.setEdge(u,v,{style:"invis"});return v})});return h}},{"./graphlib":67,"./lodash":70,"./util":89}],67:[function(require,module,exports){
/* global window */
var graphlib;if(typeof require==="function"){try{graphlib=require("graphlib")}catch(e){
// continue regardless of error
}}if(!graphlib){graphlib=window.graphlib}module.exports=graphlib},{graphlib:91}],68:[function(require,module,exports){var _=require("./lodash");var Graph=require("./graphlib").Graph;var List=require("./data/list");
/*
 * A greedy heuristic for finding a feedback arc set for a graph. A feedback
 * arc set is a set of edges that can be removed to make a graph acyclic.
 * The algorithm comes from: P. Eades, X. Lin, and W. F. Smyth, "A fast and
 * effective heuristic for the feedback arc set problem." This implementation
 * adjusts that from the paper to allow for weighted edges.
 */module.exports=greedyFAS;var DEFAULT_WEIGHT_FN=_.constant(1);function greedyFAS(g,weightFn){if(g.nodeCount()<=1){return[]}var state=buildState(g,weightFn||DEFAULT_WEIGHT_FN);var results=doGreedyFAS(state.graph,state.buckets,state.zeroIdx);
// Expand multi-edges
return _.flatten(_.map(results,function(e){return g.outEdges(e.v,e.w)}),true)}function doGreedyFAS(g,buckets,zeroIdx){var results=[];var sources=buckets[buckets.length-1];var sinks=buckets[0];var entry;while(g.nodeCount()){while(entry=sinks.dequeue()){removeNode(g,buckets,zeroIdx,entry)}while(entry=sources.dequeue()){removeNode(g,buckets,zeroIdx,entry)}if(g.nodeCount()){for(var i=buckets.length-2;i>0;--i){entry=buckets[i].dequeue();if(entry){results=results.concat(removeNode(g,buckets,zeroIdx,entry,true));break}}}}return results}function removeNode(g,buckets,zeroIdx,entry,collectPredecessors){var results=collectPredecessors?[]:undefined;_.forEach(g.inEdges(entry.v),function(edge){var weight=g.edge(edge);var uEntry=g.node(edge.v);if(collectPredecessors){results.push({v:edge.v,w:edge.w})}uEntry.out-=weight;assignBucket(buckets,zeroIdx,uEntry)});_.forEach(g.outEdges(entry.v),function(edge){var weight=g.edge(edge);var w=edge.w;var wEntry=g.node(w);wEntry["in"]-=weight;assignBucket(buckets,zeroIdx,wEntry)});g.removeNode(entry.v);return results}function buildState(g,weightFn){var fasGraph=new Graph;var maxIn=0;var maxOut=0;_.forEach(g.nodes(),function(v){fasGraph.setNode(v,{v:v,in:0,out:0})});
// Aggregate weights on nodes, but also sum the weights across multi-edges
// into a single edge for the fasGraph.
_.forEach(g.edges(),function(e){var prevWeight=fasGraph.edge(e.v,e.w)||0;var weight=weightFn(e);var edgeWeight=prevWeight+weight;fasGraph.setEdge(e.v,e.w,edgeWeight);maxOut=Math.max(maxOut,fasGraph.node(e.v).out+=weight);maxIn=Math.max(maxIn,fasGraph.node(e.w)["in"]+=weight)});var buckets=_.range(maxOut+maxIn+3).map(function(){return new List});var zeroIdx=maxIn+1;_.forEach(fasGraph.nodes(),function(v){assignBucket(buckets,zeroIdx,fasGraph.node(v))});return{graph:fasGraph,buckets:buckets,zeroIdx:zeroIdx}}function assignBucket(buckets,zeroIdx,entry){if(!entry.out){buckets[0].enqueue(entry)}else if(!entry["in"]){buckets[buckets.length-1].enqueue(entry)}else{buckets[entry.out-entry["in"]+zeroIdx].enqueue(entry)}}},{"./data/list":65,"./graphlib":67,"./lodash":70}],69:[function(require,module,exports){"use strict";var _=require("./lodash");var acyclic=require("./acyclic");var normalize=require("./normalize");var rank=require("./rank");var normalizeRanks=require("./util").normalizeRanks;var parentDummyChains=require("./parent-dummy-chains");var removeEmptyRanks=require("./util").removeEmptyRanks;var nestingGraph=require("./nesting-graph");var addBorderSegments=require("./add-border-segments");var coordinateSystem=require("./coordinate-system");var order=require("./order");var position=require("./position");var util=require("./util");var Graph=require("./graphlib").Graph;module.exports=layout;function layout(g,opts){var time=opts&&opts.debugTiming?util.time:util.notime;time("layout",function(){var layoutGraph=time("  buildLayoutGraph",function(){return buildLayoutGraph(g)});time("  runLayout",function(){runLayout(layoutGraph,time)});time("  updateInputGraph",function(){updateInputGraph(g,layoutGraph)})})}function runLayout(g,time){time("    makeSpaceForEdgeLabels",function(){makeSpaceForEdgeLabels(g)});time("    removeSelfEdges",function(){removeSelfEdges(g)});time("    acyclic",function(){acyclic.run(g)});time("    nestingGraph.run",function(){nestingGraph.run(g)});time("    rank",function(){rank(util.asNonCompoundGraph(g))});time("    injectEdgeLabelProxies",function(){injectEdgeLabelProxies(g)});time("    removeEmptyRanks",function(){removeEmptyRanks(g)});time("    nestingGraph.cleanup",function(){nestingGraph.cleanup(g)});time("    normalizeRanks",function(){normalizeRanks(g)});time("    assignRankMinMax",function(){assignRankMinMax(g)});time("    removeEdgeLabelProxies",function(){removeEdgeLabelProxies(g)});time("    normalize.run",function(){normalize.run(g)});time("    parentDummyChains",function(){parentDummyChains(g)});time("    addBorderSegments",function(){addBorderSegments(g)});time("    order",function(){order(g)});time("    insertSelfEdges",function(){insertSelfEdges(g)});time("    adjustCoordinateSystem",function(){coordinateSystem.adjust(g)});time("    position",function(){position(g)});time("    positionSelfEdges",function(){positionSelfEdges(g)});time("    removeBorderNodes",function(){removeBorderNodes(g)});time("    normalize.undo",function(){normalize.undo(g)});time("    fixupEdgeLabelCoords",function(){fixupEdgeLabelCoords(g)});time("    undoCoordinateSystem",function(){coordinateSystem.undo(g)});time("    translateGraph",function(){translateGraph(g)});time("    assignNodeIntersects",function(){assignNodeIntersects(g)});time("    reversePoints",function(){reversePointsForReversedEdges(g)});time("    acyclic.undo",function(){acyclic.undo(g)})}
/*
 * Copies final layout information from the layout graph back to the input
 * graph. This process only copies whitelisted attributes from the layout graph
 * to the input graph, so it serves as a good place to determine what
 * attributes can influence layout.
 */function updateInputGraph(inputGraph,layoutGraph){_.forEach(inputGraph.nodes(),function(v){var inputLabel=inputGraph.node(v);var layoutLabel=layoutGraph.node(v);if(inputLabel){inputLabel.x=layoutLabel.x;inputLabel.y=layoutLabel.y;if(layoutGraph.children(v).length){inputLabel.width=layoutLabel.width;inputLabel.height=layoutLabel.height}}});_.forEach(inputGraph.edges(),function(e){var inputLabel=inputGraph.edge(e);var layoutLabel=layoutGraph.edge(e);inputLabel.points=layoutLabel.points;if(_.has(layoutLabel,"x")){inputLabel.x=layoutLabel.x;inputLabel.y=layoutLabel.y}});inputGraph.graph().width=layoutGraph.graph().width;inputGraph.graph().height=layoutGraph.graph().height}var graphNumAttrs=["nodesep","edgesep","ranksep","marginx","marginy"];var graphDefaults={ranksep:50,edgesep:20,nodesep:50,rankdir:"tb"};var graphAttrs=["acyclicer","ranker","rankdir","align"];var nodeNumAttrs=["width","height"];var nodeDefaults={width:0,height:0};var edgeNumAttrs=["minlen","weight","width","height","labeloffset"];var edgeDefaults={minlen:1,weight:1,width:0,height:0,labeloffset:10,labelpos:"r"};var edgeAttrs=["labelpos"];
/*
 * Constructs a new graph from the input graph, which can be used for layout.
 * This process copies only whitelisted attributes from the input graph to the
 * layout graph. Thus this function serves as a good place to determine what
 * attributes can influence layout.
 */function buildLayoutGraph(inputGraph){var g=new Graph({multigraph:true,compound:true});var graph=canonicalize(inputGraph.graph());g.setGraph(_.merge({},graphDefaults,selectNumberAttrs(graph,graphNumAttrs),_.pick(graph,graphAttrs)));_.forEach(inputGraph.nodes(),function(v){var node=canonicalize(inputGraph.node(v));g.setNode(v,_.defaults(selectNumberAttrs(node,nodeNumAttrs),nodeDefaults));g.setParent(v,inputGraph.parent(v))});_.forEach(inputGraph.edges(),function(e){var edge=canonicalize(inputGraph.edge(e));g.setEdge(e,_.merge({},edgeDefaults,selectNumberAttrs(edge,edgeNumAttrs),_.pick(edge,edgeAttrs)))});return g}
/*
 * This idea comes from the Gansner paper: to account for edge labels in our
 * layout we split each rank in half by doubling minlen and halving ranksep.
 * Then we can place labels at these mid-points between nodes.
 *
 * We also add some minimal padding to the width to push the label for the edge
 * away from the edge itself a bit.
 */function makeSpaceForEdgeLabels(g){var graph=g.graph();graph.ranksep/=2;_.forEach(g.edges(),function(e){var edge=g.edge(e);edge.minlen*=2;if(edge.labelpos.toLowerCase()!=="c"){if(graph.rankdir==="TB"||graph.rankdir==="BT"){edge.width+=edge.labeloffset}else{edge.height+=edge.labeloffset}}})}
/*
 * Creates temporary dummy nodes that capture the rank in which each edge's
 * label is going to, if it has one of non-zero width and height. We do this
 * so that we can safely remove empty ranks while preserving balance for the
 * label's position.
 */function injectEdgeLabelProxies(g){_.forEach(g.edges(),function(e){var edge=g.edge(e);if(edge.width&&edge.height){var v=g.node(e.v);var w=g.node(e.w);var label={rank:(w.rank-v.rank)/2+v.rank,e:e};util.addDummyNode(g,"edge-proxy",label,"_ep")}})}function assignRankMinMax(g){var maxRank=0;_.forEach(g.nodes(),function(v){var node=g.node(v);if(node.borderTop){node.minRank=g.node(node.borderTop).rank;node.maxRank=g.node(node.borderBottom).rank;maxRank=_.max(maxRank,node.maxRank)}});g.graph().maxRank=maxRank}function removeEdgeLabelProxies(g){_.forEach(g.nodes(),function(v){var node=g.node(v);if(node.dummy==="edge-proxy"){g.edge(node.e).labelRank=node.rank;g.removeNode(v)}})}function translateGraph(g){var minX=Number.POSITIVE_INFINITY;var maxX=0;var minY=Number.POSITIVE_INFINITY;var maxY=0;var graphLabel=g.graph();var marginX=graphLabel.marginx||0;var marginY=graphLabel.marginy||0;function getExtremes(attrs){var x=attrs.x;var y=attrs.y;var w=attrs.width;var h=attrs.height;minX=Math.min(minX,x-w/2);maxX=Math.max(maxX,x+w/2);minY=Math.min(minY,y-h/2);maxY=Math.max(maxY,y+h/2)}_.forEach(g.nodes(),function(v){getExtremes(g.node(v))});_.forEach(g.edges(),function(e){var edge=g.edge(e);if(_.has(edge,"x")){getExtremes(edge)}});minX-=marginX;minY-=marginY;_.forEach(g.nodes(),function(v){var node=g.node(v);node.x-=minX;node.y-=minY});_.forEach(g.edges(),function(e){var edge=g.edge(e);_.forEach(edge.points,function(p){p.x-=minX;p.y-=minY});if(_.has(edge,"x")){edge.x-=minX}if(_.has(edge,"y")){edge.y-=minY}});graphLabel.width=maxX-minX+marginX;graphLabel.height=maxY-minY+marginY}function assignNodeIntersects(g){_.forEach(g.edges(),function(e){var edge=g.edge(e);var nodeV=g.node(e.v);var nodeW=g.node(e.w);var p1,p2;if(!edge.points){edge.points=[];p1=nodeW;p2=nodeV}else{p1=edge.points[0];p2=edge.points[edge.points.length-1]}edge.points.unshift(util.intersectRect(nodeV,p1));edge.points.push(util.intersectRect(nodeW,p2))})}function fixupEdgeLabelCoords(g){_.forEach(g.edges(),function(e){var edge=g.edge(e);if(_.has(edge,"x")){if(edge.labelpos==="l"||edge.labelpos==="r"){edge.width-=edge.labeloffset}switch(edge.labelpos){case"l":edge.x-=edge.width/2+edge.labeloffset;break;case"r":edge.x+=edge.width/2+edge.labeloffset;break}}})}function reversePointsForReversedEdges(g){_.forEach(g.edges(),function(e){var edge=g.edge(e);if(edge.reversed){edge.points.reverse()}})}function removeBorderNodes(g){_.forEach(g.nodes(),function(v){if(g.children(v).length){var node=g.node(v);var t=g.node(node.borderTop);var b=g.node(node.borderBottom);var l=g.node(_.last(node.borderLeft));var r=g.node(_.last(node.borderRight));node.width=Math.abs(r.x-l.x);node.height=Math.abs(b.y-t.y);node.x=l.x+node.width/2;node.y=t.y+node.height/2}});_.forEach(g.nodes(),function(v){if(g.node(v).dummy==="border"){g.removeNode(v)}})}function removeSelfEdges(g){_.forEach(g.edges(),function(e){if(e.v===e.w){var node=g.node(e.v);if(!node.selfEdges){node.selfEdges=[]}node.selfEdges.push({e:e,label:g.edge(e)});g.removeEdge(e)}})}function insertSelfEdges(g){var layers=util.buildLayerMatrix(g);_.forEach(layers,function(layer){var orderShift=0;_.forEach(layer,function(v,i){var node=g.node(v);node.order=i+orderShift;_.forEach(node.selfEdges,function(selfEdge){util.addDummyNode(g,"selfedge",{width:selfEdge.label.width,height:selfEdge.label.height,rank:node.rank,order:i+ ++orderShift,e:selfEdge.e,label:selfEdge.label},"_se")});delete node.selfEdges})})}function positionSelfEdges(g){_.forEach(g.nodes(),function(v){var node=g.node(v);if(node.dummy==="selfedge"){var selfNode=g.node(node.e.v);var x=selfNode.x+selfNode.width/2;var y=selfNode.y;var dx=node.x-x;var dy=selfNode.height/2;g.setEdge(node.e,node.label);g.removeNode(v);node.label.points=[{x:x+2*dx/3,y:y-dy},{x:x+5*dx/6,y:y-dy},{x:x+dx,y:y},{x:x+5*dx/6,y:y+dy},{x:x+2*dx/3,y:y+dy}];node.label.x=node.x;node.label.y=node.y}})}function selectNumberAttrs(obj,attrs){return _.mapValues(_.pick(obj,attrs),Number)}function canonicalize(attrs){var newAttrs={};_.forEach(attrs,function(v,k){newAttrs[k.toLowerCase()]=v});return newAttrs}},{"./acyclic":62,"./add-border-segments":63,"./coordinate-system":64,"./graphlib":67,"./lodash":70,"./nesting-graph":71,"./normalize":72,"./order":77,"./parent-dummy-chains":82,"./position":84,"./rank":86,"./util":89}],70:[function(require,module,exports){
/* global window */
var lodash;if(typeof require==="function"){try{lodash={cloneDeep:require("lodash/cloneDeep"),constant:require("lodash/constant"),defaults:require("lodash/defaults"),each:require("lodash/each"),filter:require("lodash/filter"),find:require("lodash/find"),flatten:require("lodash/flatten"),forEach:require("lodash/forEach"),forIn:require("lodash/forIn"),has:require("lodash/has"),isUndefined:require("lodash/isUndefined"),last:require("lodash/last"),map:require("lodash/map"),mapValues:require("lodash/mapValues"),max:require("lodash/max"),merge:require("lodash/merge"),min:require("lodash/min"),minBy:require("lodash/minBy"),now:require("lodash/now"),pick:require("lodash/pick"),range:require("lodash/range"),reduce:require("lodash/reduce"),sortBy:require("lodash/sortBy"),uniqueId:require("lodash/uniqueId"),values:require("lodash/values"),zipObject:require("lodash/zipObject")}}catch(e){
// continue regardless of error
}}if(!lodash){lodash=window._}module.exports=lodash},{"lodash/cloneDeep":287,"lodash/constant":288,"lodash/defaults":289,"lodash/each":290,"lodash/filter":292,"lodash/find":293,"lodash/flatten":295,"lodash/forEach":296,"lodash/forIn":297,"lodash/has":299,"lodash/isUndefined":318,"lodash/last":321,"lodash/map":322,"lodash/mapValues":323,"lodash/max":324,"lodash/merge":326,"lodash/min":327,"lodash/minBy":328,"lodash/now":330,"lodash/pick":331,"lodash/range":333,"lodash/reduce":334,"lodash/sortBy":336,"lodash/uniqueId":346,"lodash/values":347,"lodash/zipObject":348}],71:[function(require,module,exports){var _=require("./lodash");var util=require("./util");module.exports={run:run,cleanup:cleanup};
/*
 * A nesting graph creates dummy nodes for the tops and bottoms of subgraphs,
 * adds appropriate edges to ensure that all cluster nodes are placed between
 * these boundries, and ensures that the graph is connected.
 *
 * In addition we ensure, through the use of the minlen property, that nodes
 * and subgraph border nodes to not end up on the same rank.
 *
 * Preconditions:
 *
 *    1. Input graph is a DAG
 *    2. Nodes in the input graph has a minlen attribute
 *
 * Postconditions:
 *
 *    1. Input graph is connected.
 *    2. Dummy nodes are added for the tops and bottoms of subgraphs.
 *    3. The minlen attribute for nodes is adjusted to ensure nodes do not
 *       get placed on the same rank as subgraph border nodes.
 *
 * The nesting graph idea comes from Sander, "Layout of Compound Directed
 * Graphs."
 */function run(g){var root=util.addDummyNode(g,"root",{},"_root");var depths=treeDepths(g);var height=_.max(_.values(depths))-1;// Note: depths is an Object not an array
var nodeSep=2*height+1;g.graph().nestingRoot=root;
// Multiply minlen by nodeSep to align nodes on non-border ranks.
_.forEach(g.edges(),function(e){g.edge(e).minlen*=nodeSep});
// Calculate a weight that is sufficient to keep subgraphs vertically compact
var weight=sumWeights(g)+1;
// Create border nodes and link them up
_.forEach(g.children(),function(child){dfs(g,root,nodeSep,weight,height,depths,child)});
// Save the multiplier for node layers for later removal of empty border
// layers.
g.graph().nodeRankFactor=nodeSep}function dfs(g,root,nodeSep,weight,height,depths,v){var children=g.children(v);if(!children.length){if(v!==root){g.setEdge(root,v,{weight:0,minlen:nodeSep})}return}var top=util.addBorderNode(g,"_bt");var bottom=util.addBorderNode(g,"_bb");var label=g.node(v);g.setParent(top,v);label.borderTop=top;g.setParent(bottom,v);label.borderBottom=bottom;_.forEach(children,function(child){dfs(g,root,nodeSep,weight,height,depths,child);var childNode=g.node(child);var childTop=childNode.borderTop?childNode.borderTop:child;var childBottom=childNode.borderBottom?childNode.borderBottom:child;var thisWeight=childNode.borderTop?weight:2*weight;var minlen=childTop!==childBottom?1:height-depths[v]+1;g.setEdge(top,childTop,{weight:thisWeight,minlen:minlen,nestingEdge:true});g.setEdge(childBottom,bottom,{weight:thisWeight,minlen:minlen,nestingEdge:true})});if(!g.parent(v)){g.setEdge(root,top,{weight:0,minlen:height+depths[v]})}}function treeDepths(g){var depths={};function dfs(v,depth){var children=g.children(v);if(children&&children.length){_.forEach(children,function(child){dfs(child,depth+1)})}depths[v]=depth}_.forEach(g.children(),function(v){dfs(v,1)});return depths}function sumWeights(g){return _.reduce(g.edges(),function(acc,e){return acc+g.edge(e).weight},0)}function cleanup(g){var graphLabel=g.graph();g.removeNode(graphLabel.nestingRoot);delete graphLabel.nestingRoot;_.forEach(g.edges(),function(e){var edge=g.edge(e);if(edge.nestingEdge){g.removeEdge(e)}})}},{"./lodash":70,"./util":89}],72:[function(require,module,exports){"use strict";var _=require("./lodash");var util=require("./util");module.exports={run:run,undo:undo};
/*
 * Breaks any long edges in the graph into short segments that span 1 layer
 * each. This operation is undoable with the denormalize function.
 *
 * Pre-conditions:
 *
 *    1. The input graph is a DAG.
 *    2. Each node in the graph has a "rank" property.
 *
 * Post-condition:
 *
 *    1. All edges in the graph have a length of 1.
 *    2. Dummy nodes are added where edges have been split into segments.
 *    3. The graph is augmented with a "dummyChains" attribute which contains
 *       the first dummy in each chain of dummy nodes produced.
 */function run(g){g.graph().dummyChains=[];_.forEach(g.edges(),function(edge){normalizeEdge(g,edge)})}function normalizeEdge(g,e){var v=e.v;var vRank=g.node(v).rank;var w=e.w;var wRank=g.node(w).rank;var name=e.name;var edgeLabel=g.edge(e);var labelRank=edgeLabel.labelRank;if(wRank===vRank+1)return;g.removeEdge(e);var dummy,attrs,i;for(i=0,++vRank;vRank<wRank;++i,++vRank){edgeLabel.points=[];attrs={width:0,height:0,edgeLabel:edgeLabel,edgeObj:e,rank:vRank};dummy=util.addDummyNode(g,"edge",attrs,"_d");if(vRank===labelRank){attrs.width=edgeLabel.width;attrs.height=edgeLabel.height;attrs.dummy="edge-label";attrs.labelpos=edgeLabel.labelpos}g.setEdge(v,dummy,{weight:edgeLabel.weight},name);if(i===0){g.graph().dummyChains.push(dummy)}v=dummy}g.setEdge(v,w,{weight:edgeLabel.weight},name)}function undo(g){_.forEach(g.graph().dummyChains,function(v){var node=g.node(v);var origLabel=node.edgeLabel;var w;g.setEdge(node.edgeObj,origLabel);while(node.dummy){w=g.successors(v)[0];g.removeNode(v);origLabel.points.push({x:node.x,y:node.y});if(node.dummy==="edge-label"){origLabel.x=node.x;origLabel.y=node.y;origLabel.width=node.width;origLabel.height=node.height}v=w;node=g.node(v)}})}},{"./lodash":70,"./util":89}],73:[function(require,module,exports){var _=require("../lodash");module.exports=addSubgraphConstraints;function addSubgraphConstraints(g,cg,vs){var prev={},rootPrev;_.forEach(vs,function(v){var child=g.parent(v),parent,prevChild;while(child){parent=g.parent(child);if(parent){prevChild=prev[parent];prev[parent]=child}else{prevChild=rootPrev;rootPrev=child}if(prevChild&&prevChild!==child){cg.setEdge(prevChild,child);return}child=parent}});
/*
  function dfs(v) {
    var children = v ? g.children(v) : g.children();
    if (children.length) {
      var min = Number.POSITIVE_INFINITY,
          subgraphs = [];
      _.each(children, function(child) {
        var childMin = dfs(child);
        if (g.children(child).length) {
          subgraphs.push({ v: child, order: childMin });
        }
        min = Math.min(min, childMin);
      });
      _.reduce(_.sortBy(subgraphs, "order"), function(prev, curr) {
        cg.setEdge(prev.v, curr.v);
        return curr;
      });
      return min;
    }
    return g.node(v).order;
  }
  dfs(undefined);
  */}},{"../lodash":70}],74:[function(require,module,exports){var _=require("../lodash");module.exports=barycenter;function barycenter(g,movable){return _.map(movable,function(v){var inV=g.inEdges(v);if(!inV.length){return{v:v}}else{var result=_.reduce(inV,function(acc,e){var edge=g.edge(e),nodeU=g.node(e.v);return{sum:acc.sum+edge.weight*nodeU.order,weight:acc.weight+edge.weight}},{sum:0,weight:0});return{v:v,barycenter:result.sum/result.weight,weight:result.weight}}})}},{"../lodash":70}],75:[function(require,module,exports){var _=require("../lodash");var Graph=require("../graphlib").Graph;module.exports=buildLayerGraph;
/*
 * Constructs a graph that can be used to sort a layer of nodes. The graph will
 * contain all base and subgraph nodes from the request layer in their original
 * hierarchy and any edges that are incident on these nodes and are of the type
 * requested by the "relationship" parameter.
 *
 * Nodes from the requested rank that do not have parents are assigned a root
 * node in the output graph, which is set in the root graph attribute. This
 * makes it easy to walk the hierarchy of movable nodes during ordering.
 *
 * Pre-conditions:
 *
 *    1. Input graph is a DAG
 *    2. Base nodes in the input graph have a rank attribute
 *    3. Subgraph nodes in the input graph has minRank and maxRank attributes
 *    4. Edges have an assigned weight
 *
 * Post-conditions:
 *
 *    1. Output graph has all nodes in the movable rank with preserved
 *       hierarchy.
 *    2. Root nodes in the movable layer are made children of the node
 *       indicated by the root attribute of the graph.
 *    3. Non-movable nodes incident on movable nodes, selected by the
 *       relationship parameter, are included in the graph (without hierarchy).
 *    4. Edges incident on movable nodes, selected by the relationship
 *       parameter, are added to the output graph.
 *    5. The weights for copied edges are aggregated as need, since the output
 *       graph is not a multi-graph.
 */function buildLayerGraph(g,rank,relationship){var root=createRootNode(g),result=new Graph({compound:true}).setGraph({root:root}).setDefaultNodeLabel(function(v){return g.node(v)});_.forEach(g.nodes(),function(v){var node=g.node(v),parent=g.parent(v);if(node.rank===rank||node.minRank<=rank&&rank<=node.maxRank){result.setNode(v);result.setParent(v,parent||root);
// This assumes we have only short edges!
_.forEach(g[relationship](v),function(e){var u=e.v===v?e.w:e.v,edge=result.edge(u,v),weight=!_.isUndefined(edge)?edge.weight:0;result.setEdge(u,v,{weight:g.edge(e).weight+weight})});if(_.has(node,"minRank")){result.setNode(v,{borderLeft:node.borderLeft[rank],borderRight:node.borderRight[rank]})}}});return result}function createRootNode(g){var v;while(g.hasNode(v=_.uniqueId("_root")));return v}},{"../graphlib":67,"../lodash":70}],76:[function(require,module,exports){"use strict";var _=require("../lodash");module.exports=crossCount;
/*
 * A function that takes a layering (an array of layers, each with an array of
 * ordererd nodes) and a graph and returns a weighted crossing count.
 *
 * Pre-conditions:
 *
 *    1. Input graph must be simple (not a multigraph), directed, and include
 *       only simple edges.
 *    2. Edges in the input graph must have assigned weights.
 *
 * Post-conditions:
 *
 *    1. The graph and layering matrix are left unchanged.
 *
 * This algorithm is derived from Barth, et al., "Bilayer Cross Counting."
 */function crossCount(g,layering){var cc=0;for(var i=1;i<layering.length;++i){cc+=twoLayerCrossCount(g,layering[i-1],layering[i])}return cc}function twoLayerCrossCount(g,northLayer,southLayer){
// Sort all of the edges between the north and south layers by their position
// in the north layer and then the south. Map these edges to the position of
// their head in the south layer.
var southPos=_.zipObject(southLayer,_.map(southLayer,function(v,i){return i}));var southEntries=_.flatten(_.map(northLayer,function(v){return _.sortBy(_.map(g.outEdges(v),function(e){return{pos:southPos[e.w],weight:g.edge(e).weight}}),"pos")}),true);
// Build the accumulator tree
var firstIndex=1;while(firstIndex<southLayer.length)firstIndex<<=1;var treeSize=2*firstIndex-1;firstIndex-=1;var tree=_.map(new Array(treeSize),function(){return 0});
// Calculate the weighted crossings
var cc=0;_.forEach(southEntries.forEach(function(entry){var index=entry.pos+firstIndex;tree[index]+=entry.weight;var weightSum=0;while(index>0){if(index%2){weightSum+=tree[index+1]}index=index-1>>1;tree[index]+=entry.weight}cc+=entry.weight*weightSum}));return cc}},{"../lodash":70}],77:[function(require,module,exports){"use strict";var _=require("../lodash");var initOrder=require("./init-order");var crossCount=require("./cross-count");var sortSubgraph=require("./sort-subgraph");var buildLayerGraph=require("./build-layer-graph");var addSubgraphConstraints=require("./add-subgraph-constraints");var Graph=require("../graphlib").Graph;var util=require("../util");module.exports=order;
/*
 * Applies heuristics to minimize edge crossings in the graph and sets the best
 * order solution as an order attribute on each node.
 *
 * Pre-conditions:
 *
 *    1. Graph must be DAG
 *    2. Graph nodes must be objects with a "rank" attribute
 *    3. Graph edges must have the "weight" attribute
 *
 * Post-conditions:
 *
 *    1. Graph nodes will have an "order" attribute based on the results of the
 *       algorithm.
 */function order(g){var maxRank=util.maxRank(g),downLayerGraphs=buildLayerGraphs(g,_.range(1,maxRank+1),"inEdges"),upLayerGraphs=buildLayerGraphs(g,_.range(maxRank-1,-1,-1),"outEdges");var layering=initOrder(g);assignOrder(g,layering);var bestCC=Number.POSITIVE_INFINITY,best;for(var i=0,lastBest=0;lastBest<4;++i,++lastBest){sweepLayerGraphs(i%2?downLayerGraphs:upLayerGraphs,i%4>=2);layering=util.buildLayerMatrix(g);var cc=crossCount(g,layering);if(cc<bestCC){lastBest=0;best=_.cloneDeep(layering);bestCC=cc}}assignOrder(g,best)}function buildLayerGraphs(g,ranks,relationship){return _.map(ranks,function(rank){return buildLayerGraph(g,rank,relationship)})}function sweepLayerGraphs(layerGraphs,biasRight){var cg=new Graph;_.forEach(layerGraphs,function(lg){var root=lg.graph().root;var sorted=sortSubgraph(lg,root,cg,biasRight);_.forEach(sorted.vs,function(v,i){lg.node(v).order=i});addSubgraphConstraints(lg,cg,sorted.vs)})}function assignOrder(g,layering){_.forEach(layering,function(layer){_.forEach(layer,function(v,i){g.node(v).order=i})})}},{"../graphlib":67,"../lodash":70,"../util":89,"./add-subgraph-constraints":73,"./build-layer-graph":75,"./cross-count":76,"./init-order":78,"./sort-subgraph":80}],78:[function(require,module,exports){"use strict";var _=require("../lodash");module.exports=initOrder;
/*
 * Assigns an initial order value for each node by performing a DFS search
 * starting from nodes in the first rank. Nodes are assigned an order in their
 * rank as they are first visited.
 *
 * This approach comes from Gansner, et al., "A Technique for Drawing Directed
 * Graphs."
 *
 * Returns a layering matrix with an array per layer and each layer sorted by
 * the order of its nodes.
 */function initOrder(g){var visited={};var simpleNodes=_.filter(g.nodes(),function(v){return!g.children(v).length});var maxRank=_.max(_.map(simpleNodes,function(v){return g.node(v).rank}));var layers=_.map(_.range(maxRank+1),function(){return[]});function dfs(v){if(_.has(visited,v))return;visited[v]=true;var node=g.node(v);layers[node.rank].push(v);_.forEach(g.successors(v),dfs)}var orderedVs=_.sortBy(simpleNodes,function(v){return g.node(v).rank});_.forEach(orderedVs,dfs);return layers}},{"../lodash":70}],79:[function(require,module,exports){"use strict";var _=require("../lodash");module.exports=resolveConflicts;
/*
 * Given a list of entries of the form {v, barycenter, weight} and a
 * constraint graph this function will resolve any conflicts between the
 * constraint graph and the barycenters for the entries. If the barycenters for
 * an entry would violate a constraint in the constraint graph then we coalesce
 * the nodes in the conflict into a new node that respects the contraint and
 * aggregates barycenter and weight information.
 *
 * This implementation is based on the description in Forster, "A Fast and
 * Simple Hueristic for Constrained Two-Level Crossing Reduction," thought it
 * differs in some specific details.
 *
 * Pre-conditions:
 *
 *    1. Each entry has the form {v, barycenter, weight}, or if the node has
 *       no barycenter, then {v}.
 *
 * Returns:
 *
 *    A new list of entries of the form {vs, i, barycenter, weight}. The list
 *    `vs` may either be a singleton or it may be an aggregation of nodes
 *    ordered such that they do not violate constraints from the constraint
 *    graph. The property `i` is the lowest original index of any of the
 *    elements in `vs`.
 */function resolveConflicts(entries,cg){var mappedEntries={};_.forEach(entries,function(entry,i){var tmp=mappedEntries[entry.v]={indegree:0,in:[],out:[],vs:[entry.v],i:i};if(!_.isUndefined(entry.barycenter)){tmp.barycenter=entry.barycenter;tmp.weight=entry.weight}});_.forEach(cg.edges(),function(e){var entryV=mappedEntries[e.v];var entryW=mappedEntries[e.w];if(!_.isUndefined(entryV)&&!_.isUndefined(entryW)){entryW.indegree++;entryV.out.push(mappedEntries[e.w])}});var sourceSet=_.filter(mappedEntries,function(entry){return!entry.indegree});return doResolveConflicts(sourceSet)}function doResolveConflicts(sourceSet){var entries=[];function handleIn(vEntry){return function(uEntry){if(uEntry.merged){return}if(_.isUndefined(uEntry.barycenter)||_.isUndefined(vEntry.barycenter)||uEntry.barycenter>=vEntry.barycenter){mergeEntries(vEntry,uEntry)}}}function handleOut(vEntry){return function(wEntry){wEntry["in"].push(vEntry);if(--wEntry.indegree===0){sourceSet.push(wEntry)}}}while(sourceSet.length){var entry=sourceSet.pop();entries.push(entry);_.forEach(entry["in"].reverse(),handleIn(entry));_.forEach(entry.out,handleOut(entry))}return _.map(_.filter(entries,function(entry){return!entry.merged}),function(entry){return _.pick(entry,["vs","i","barycenter","weight"])})}function mergeEntries(target,source){var sum=0;var weight=0;if(target.weight){sum+=target.barycenter*target.weight;weight+=target.weight}if(source.weight){sum+=source.barycenter*source.weight;weight+=source.weight}target.vs=source.vs.concat(target.vs);target.barycenter=sum/weight;target.weight=weight;target.i=Math.min(source.i,target.i);source.merged=true}},{"../lodash":70}],80:[function(require,module,exports){var _=require("../lodash");var barycenter=require("./barycenter");var resolveConflicts=require("./resolve-conflicts");var sort=require("./sort");module.exports=sortSubgraph;function sortSubgraph(g,v,cg,biasRight){var movable=g.children(v);var node=g.node(v);var bl=node?node.borderLeft:undefined;var br=node?node.borderRight:undefined;var subgraphs={};if(bl){movable=_.filter(movable,function(w){return w!==bl&&w!==br})}var barycenters=barycenter(g,movable);_.forEach(barycenters,function(entry){if(g.children(entry.v).length){var subgraphResult=sortSubgraph(g,entry.v,cg,biasRight);subgraphs[entry.v]=subgraphResult;if(_.has(subgraphResult,"barycenter")){mergeBarycenters(entry,subgraphResult)}}});var entries=resolveConflicts(barycenters,cg);expandSubgraphs(entries,subgraphs);var result=sort(entries,biasRight);if(bl){result.vs=_.flatten([bl,result.vs,br],true);if(g.predecessors(bl).length){var blPred=g.node(g.predecessors(bl)[0]),brPred=g.node(g.predecessors(br)[0]);if(!_.has(result,"barycenter")){result.barycenter=0;result.weight=0}result.barycenter=(result.barycenter*result.weight+blPred.order+brPred.order)/(result.weight+2);result.weight+=2}}return result}function expandSubgraphs(entries,subgraphs){_.forEach(entries,function(entry){entry.vs=_.flatten(entry.vs.map(function(v){if(subgraphs[v]){return subgraphs[v].vs}return v}),true)})}function mergeBarycenters(target,other){if(!_.isUndefined(target.barycenter)){target.barycenter=(target.barycenter*target.weight+other.barycenter*other.weight)/(target.weight+other.weight);target.weight+=other.weight}else{target.barycenter=other.barycenter;target.weight=other.weight}}},{"../lodash":70,"./barycenter":74,"./resolve-conflicts":79,"./sort":81}],81:[function(require,module,exports){var _=require("../lodash");var util=require("../util");module.exports=sort;function sort(entries,biasRight){var parts=util.partition(entries,function(entry){return _.has(entry,"barycenter")});var sortable=parts.lhs,unsortable=_.sortBy(parts.rhs,function(entry){return-entry.i}),vs=[],sum=0,weight=0,vsIndex=0;sortable.sort(compareWithBias(!!biasRight));vsIndex=consumeUnsortable(vs,unsortable,vsIndex);_.forEach(sortable,function(entry){vsIndex+=entry.vs.length;vs.push(entry.vs);sum+=entry.barycenter*entry.weight;weight+=entry.weight;vsIndex=consumeUnsortable(vs,unsortable,vsIndex)});var result={vs:_.flatten(vs,true)};if(weight){result.barycenter=sum/weight;result.weight=weight}return result}function consumeUnsortable(vs,unsortable,index){var last;while(unsortable.length&&(last=_.last(unsortable)).i<=index){unsortable.pop();vs.push(last.vs);index++}return index}function compareWithBias(bias){return function(entryV,entryW){if(entryV.barycenter<entryW.barycenter){return-1}else if(entryV.barycenter>entryW.barycenter){return 1}return!bias?entryV.i-entryW.i:entryW.i-entryV.i}}},{"../lodash":70,"../util":89}],82:[function(require,module,exports){var _=require("./lodash");module.exports=parentDummyChains;function parentDummyChains(g){var postorderNums=postorder(g);_.forEach(g.graph().dummyChains,function(v){var node=g.node(v);var edgeObj=node.edgeObj;var pathData=findPath(g,postorderNums,edgeObj.v,edgeObj.w);var path=pathData.path;var lca=pathData.lca;var pathIdx=0;var pathV=path[pathIdx];var ascending=true;while(v!==edgeObj.w){node=g.node(v);if(ascending){while((pathV=path[pathIdx])!==lca&&g.node(pathV).maxRank<node.rank){pathIdx++}if(pathV===lca){ascending=false}}if(!ascending){while(pathIdx<path.length-1&&g.node(pathV=path[pathIdx+1]).minRank<=node.rank){pathIdx++}pathV=path[pathIdx]}g.setParent(v,pathV);v=g.successors(v)[0]}})}
// Find a path from v to w through the lowest common ancestor (LCA). Return the
// full path and the LCA.
function findPath(g,postorderNums,v,w){var vPath=[];var wPath=[];var low=Math.min(postorderNums[v].low,postorderNums[w].low);var lim=Math.max(postorderNums[v].lim,postorderNums[w].lim);var parent;var lca;
// Traverse up from v to find the LCA
parent=v;do{parent=g.parent(parent);vPath.push(parent)}while(parent&&(postorderNums[parent].low>low||lim>postorderNums[parent].lim));lca=parent;
// Traverse from w to LCA
parent=w;while((parent=g.parent(parent))!==lca){wPath.push(parent)}return{path:vPath.concat(wPath.reverse()),lca:lca}}function postorder(g){var result={};var lim=0;function dfs(v){var low=lim;_.forEach(g.children(v),dfs);result[v]={low:low,lim:lim++}}_.forEach(g.children(),dfs);return result}},{"./lodash":70}],83:[function(require,module,exports){"use strict";var _=require("../lodash");var Graph=require("../graphlib").Graph;var util=require("../util");
/*
 * This module provides coordinate assignment based on Brandes and Köpf, "Fast
 * and Simple Horizontal Coordinate Assignment."
 */module.exports={positionX:positionX,findType1Conflicts:findType1Conflicts,findType2Conflicts:findType2Conflicts,addConflict:addConflict,hasConflict:hasConflict,verticalAlignment:verticalAlignment,horizontalCompaction:horizontalCompaction,alignCoordinates:alignCoordinates,findSmallestWidthAlignment:findSmallestWidthAlignment,balance:balance};
/*
 * Marks all edges in the graph with a type-1 conflict with the "type1Conflict"
 * property. A type-1 conflict is one where a non-inner segment crosses an
 * inner segment. An inner segment is an edge with both incident nodes marked
 * with the "dummy" property.
 *
 * This algorithm scans layer by layer, starting with the second, for type-1
 * conflicts between the current layer and the previous layer. For each layer
 * it scans the nodes from left to right until it reaches one that is incident
 * on an inner segment. It then scans predecessors to determine if they have
 * edges that cross that inner segment. At the end a final scan is done for all
 * nodes on the current rank to see if they cross the last visited inner
 * segment.
 *
 * This algorithm (safely) assumes that a dummy node will only be incident on a
 * single node in the layers being scanned.
 */function findType1Conflicts(g,layering){var conflicts={};function visitLayer(prevLayer,layer){var
// last visited node in the previous layer that is incident on an inner
// segment.
k0=0,
// Tracks the last node in this layer scanned for crossings with a type-1
// segment.
scanPos=0,prevLayerLength=prevLayer.length,lastNode=_.last(layer);_.forEach(layer,function(v,i){var w=findOtherInnerSegmentNode(g,v),k1=w?g.node(w).order:prevLayerLength;if(w||v===lastNode){_.forEach(layer.slice(scanPos,i+1),function(scanNode){_.forEach(g.predecessors(scanNode),function(u){var uLabel=g.node(u),uPos=uLabel.order;if((uPos<k0||k1<uPos)&&!(uLabel.dummy&&g.node(scanNode).dummy)){addConflict(conflicts,u,scanNode)}})});scanPos=i+1;k0=k1}});return layer}_.reduce(layering,visitLayer);return conflicts}function findType2Conflicts(g,layering){var conflicts={};function scan(south,southPos,southEnd,prevNorthBorder,nextNorthBorder){var v;_.forEach(_.range(southPos,southEnd),function(i){v=south[i];if(g.node(v).dummy){_.forEach(g.predecessors(v),function(u){var uNode=g.node(u);if(uNode.dummy&&(uNode.order<prevNorthBorder||uNode.order>nextNorthBorder)){addConflict(conflicts,u,v)}})}})}function visitLayer(north,south){var prevNorthPos=-1,nextNorthPos,southPos=0;_.forEach(south,function(v,southLookahead){if(g.node(v).dummy==="border"){var predecessors=g.predecessors(v);if(predecessors.length){nextNorthPos=g.node(predecessors[0]).order;scan(south,southPos,southLookahead,prevNorthPos,nextNorthPos);southPos=southLookahead;prevNorthPos=nextNorthPos}}scan(south,southPos,south.length,nextNorthPos,north.length)});return south}_.reduce(layering,visitLayer);return conflicts}function findOtherInnerSegmentNode(g,v){if(g.node(v).dummy){return _.find(g.predecessors(v),function(u){return g.node(u).dummy})}}function addConflict(conflicts,v,w){if(v>w){var tmp=v;v=w;w=tmp}var conflictsV=conflicts[v];if(!conflictsV){conflicts[v]=conflictsV={}}conflictsV[w]=true}function hasConflict(conflicts,v,w){if(v>w){var tmp=v;v=w;w=tmp}return _.has(conflicts[v],w)}
/*
 * Try to align nodes into vertical "blocks" where possible. This algorithm
 * attempts to align a node with one of its median neighbors. If the edge
 * connecting a neighbor is a type-1 conflict then we ignore that possibility.
 * If a previous node has already formed a block with a node after the node
 * we're trying to form a block with, we also ignore that possibility - our
 * blocks would be split in that scenario.
 */function verticalAlignment(g,layering,conflicts,neighborFn){var root={},align={},pos={};
// We cache the position here based on the layering because the graph and
// layering may be out of sync. The layering matrix is manipulated to
// generate different extreme alignments.
_.forEach(layering,function(layer){_.forEach(layer,function(v,order){root[v]=v;align[v]=v;pos[v]=order})});_.forEach(layering,function(layer){var prevIdx=-1;_.forEach(layer,function(v){var ws=neighborFn(v);if(ws.length){ws=_.sortBy(ws,function(w){return pos[w]});var mp=(ws.length-1)/2;for(var i=Math.floor(mp),il=Math.ceil(mp);i<=il;++i){var w=ws[i];if(align[v]===v&&prevIdx<pos[w]&&!hasConflict(conflicts,v,w)){align[w]=v;align[v]=root[v]=root[w];prevIdx=pos[w]}}}})});return{root:root,align:align}}function horizontalCompaction(g,layering,root,align,reverseSep){
// This portion of the algorithm differs from BK due to a number of problems.
// Instead of their algorithm we construct a new block graph and do two
// sweeps. The first sweep places blocks with the smallest possible
// coordinates. The second sweep removes unused space by moving blocks to the
// greatest coordinates without violating separation.
var xs={},blockG=buildBlockGraph(g,layering,root,reverseSep),borderType=reverseSep?"borderLeft":"borderRight";function iterate(setXsFunc,nextNodesFunc){var stack=blockG.nodes();var elem=stack.pop();var visited={};while(elem){if(visited[elem]){setXsFunc(elem)}else{visited[elem]=true;stack.push(elem);stack=stack.concat(nextNodesFunc(elem))}elem=stack.pop()}}
// First pass, assign smallest coordinates
function pass1(elem){xs[elem]=blockG.inEdges(elem).reduce(function(acc,e){return Math.max(acc,xs[e.v]+blockG.edge(e))},0)}
// Second pass, assign greatest coordinates
function pass2(elem){var min=blockG.outEdges(elem).reduce(function(acc,e){return Math.min(acc,xs[e.w]-blockG.edge(e))},Number.POSITIVE_INFINITY);var node=g.node(elem);if(min!==Number.POSITIVE_INFINITY&&node.borderType!==borderType){xs[elem]=Math.max(xs[elem],min)}}iterate(pass1,blockG.predecessors.bind(blockG));iterate(pass2,blockG.successors.bind(blockG));
// Assign x coordinates to all nodes
_.forEach(align,function(v){xs[v]=xs[root[v]]});return xs}function buildBlockGraph(g,layering,root,reverseSep){var blockGraph=new Graph,graphLabel=g.graph(),sepFn=sep(graphLabel.nodesep,graphLabel.edgesep,reverseSep);_.forEach(layering,function(layer){var u;_.forEach(layer,function(v){var vRoot=root[v];blockGraph.setNode(vRoot);if(u){var uRoot=root[u],prevMax=blockGraph.edge(uRoot,vRoot);blockGraph.setEdge(uRoot,vRoot,Math.max(sepFn(g,v,u),prevMax||0))}u=v})});return blockGraph}
/*
 * Returns the alignment that has the smallest width of the given alignments.
 */function findSmallestWidthAlignment(g,xss){return _.minBy(_.values(xss),function(xs){var max=Number.NEGATIVE_INFINITY;var min=Number.POSITIVE_INFINITY;_.forIn(xs,function(x,v){var halfWidth=width(g,v)/2;max=Math.max(x+halfWidth,max);min=Math.min(x-halfWidth,min)});return max-min})}
/*
 * Align the coordinates of each of the layout alignments such that
 * left-biased alignments have their minimum coordinate at the same point as
 * the minimum coordinate of the smallest width alignment and right-biased
 * alignments have their maximum coordinate at the same point as the maximum
 * coordinate of the smallest width alignment.
 */function alignCoordinates(xss,alignTo){var alignToVals=_.values(alignTo),alignToMin=_.min(alignToVals),alignToMax=_.max(alignToVals);_.forEach(["u","d"],function(vert){_.forEach(["l","r"],function(horiz){var alignment=vert+horiz,xs=xss[alignment],delta;if(xs===alignTo)return;var xsVals=_.values(xs);delta=horiz==="l"?alignToMin-_.min(xsVals):alignToMax-_.max(xsVals);if(delta){xss[alignment]=_.mapValues(xs,function(x){return x+delta})}})})}function balance(xss,align){return _.mapValues(xss.ul,function(ignore,v){if(align){return xss[align.toLowerCase()][v]}else{var xs=_.sortBy(_.map(xss,v));return(xs[1]+xs[2])/2}})}function positionX(g){var layering=util.buildLayerMatrix(g);var conflicts=_.merge(findType1Conflicts(g,layering),findType2Conflicts(g,layering));var xss={};var adjustedLayering;_.forEach(["u","d"],function(vert){adjustedLayering=vert==="u"?layering:_.values(layering).reverse();_.forEach(["l","r"],function(horiz){if(horiz==="r"){adjustedLayering=_.map(adjustedLayering,function(inner){return _.values(inner).reverse()})}var neighborFn=(vert==="u"?g.predecessors:g.successors).bind(g);var align=verticalAlignment(g,adjustedLayering,conflicts,neighborFn);var xs=horizontalCompaction(g,adjustedLayering,align.root,align.align,horiz==="r");if(horiz==="r"){xs=_.mapValues(xs,function(x){return-x})}xss[vert+horiz]=xs})});var smallestWidth=findSmallestWidthAlignment(g,xss);alignCoordinates(xss,smallestWidth);return balance(xss,g.graph().align)}function sep(nodeSep,edgeSep,reverseSep){return function(g,v,w){var vLabel=g.node(v);var wLabel=g.node(w);var sum=0;var delta;sum+=vLabel.width/2;if(_.has(vLabel,"labelpos")){switch(vLabel.labelpos.toLowerCase()){case"l":delta=-vLabel.width/2;break;case"r":delta=vLabel.width/2;break}}if(delta){sum+=reverseSep?delta:-delta}delta=0;sum+=(vLabel.dummy?edgeSep:nodeSep)/2;sum+=(wLabel.dummy?edgeSep:nodeSep)/2;sum+=wLabel.width/2;if(_.has(wLabel,"labelpos")){switch(wLabel.labelpos.toLowerCase()){case"l":delta=wLabel.width/2;break;case"r":delta=-wLabel.width/2;break}}if(delta){sum+=reverseSep?delta:-delta}delta=0;return sum}}function width(g,v){return g.node(v).width}},{"../graphlib":67,"../lodash":70,"../util":89}],84:[function(require,module,exports){"use strict";var _=require("../lodash");var util=require("../util");var positionX=require("./bk").positionX;module.exports=position;function position(g){g=util.asNonCompoundGraph(g);positionY(g);_.forEach(positionX(g),function(x,v){g.node(v).x=x})}function positionY(g){var layering=util.buildLayerMatrix(g);var rankSep=g.graph().ranksep;var prevY=0;_.forEach(layering,function(layer){var maxHeight=_.max(_.map(layer,function(v){return g.node(v).height}));_.forEach(layer,function(v){g.node(v).y=prevY+maxHeight/2});prevY+=maxHeight+rankSep})}},{"../lodash":70,"../util":89,"./bk":83}],85:[function(require,module,exports){"use strict";var _=require("../lodash");var Graph=require("../graphlib").Graph;var slack=require("./util").slack;module.exports=feasibleTree;
/*
 * Constructs a spanning tree with tight edges and adjusted the input node's
 * ranks to achieve this. A tight edge is one that is has a length that matches
 * its "minlen" attribute.
 *
 * The basic structure for this function is derived from Gansner, et al., "A
 * Technique for Drawing Directed Graphs."
 *
 * Pre-conditions:
 *
 *    1. Graph must be a DAG.
 *    2. Graph must be connected.
 *    3. Graph must have at least one node.
 *    5. Graph nodes must have been previously assigned a "rank" property that
 *       respects the "minlen" property of incident edges.
 *    6. Graph edges must have a "minlen" property.
 *
 * Post-conditions:
 *
 *    - Graph nodes will have their rank adjusted to ensure that all edges are
 *      tight.
 *
 * Returns a tree (undirected graph) that is constructed using only "tight"
 * edges.
 */function feasibleTree(g){var t=new Graph({directed:false});
// Choose arbitrary node from which to start our tree
var start=g.nodes()[0];var size=g.nodeCount();t.setNode(start,{});var edge,delta;while(tightTree(t,g)<size){edge=findMinSlackEdge(t,g);delta=t.hasNode(edge.v)?slack(g,edge):-slack(g,edge);shiftRanks(t,g,delta)}return t}
/*
 * Finds a maximal tree of tight edges and returns the number of nodes in the
 * tree.
 */function tightTree(t,g){function dfs(v){_.forEach(g.nodeEdges(v),function(e){var edgeV=e.v,w=v===edgeV?e.w:edgeV;if(!t.hasNode(w)&&!slack(g,e)){t.setNode(w,{});t.setEdge(v,w,{});dfs(w)}})}_.forEach(t.nodes(),dfs);return t.nodeCount()}
/*
 * Finds the edge with the smallest slack that is incident on tree and returns
 * it.
 */function findMinSlackEdge(t,g){return _.minBy(g.edges(),function(e){if(t.hasNode(e.v)!==t.hasNode(e.w)){return slack(g,e)}})}function shiftRanks(t,g,delta){_.forEach(t.nodes(),function(v){g.node(v).rank+=delta})}},{"../graphlib":67,"../lodash":70,"./util":88}],86:[function(require,module,exports){"use strict";var rankUtil=require("./util");var longestPath=rankUtil.longestPath;var feasibleTree=require("./feasible-tree");var networkSimplex=require("./network-simplex");module.exports=rank;
/*
 * Assigns a rank to each node in the input graph that respects the "minlen"
 * constraint specified on edges between nodes.
 *
 * This basic structure is derived from Gansner, et al., "A Technique for
 * Drawing Directed Graphs."
 *
 * Pre-conditions:
 *
 *    1. Graph must be a connected DAG
 *    2. Graph nodes must be objects
 *    3. Graph edges must have "weight" and "minlen" attributes
 *
 * Post-conditions:
 *
 *    1. Graph nodes will have a "rank" attribute based on the results of the
 *       algorithm. Ranks can start at any index (including negative), we'll
 *       fix them up later.
 */function rank(g){switch(g.graph().ranker){case"network-simplex":networkSimplexRanker(g);break;case"tight-tree":tightTreeRanker(g);break;case"longest-path":longestPathRanker(g);break;default:networkSimplexRanker(g)}}
// A fast and simple ranker, but results are far from optimal.
var longestPathRanker=longestPath;function tightTreeRanker(g){longestPath(g);feasibleTree(g)}function networkSimplexRanker(g){networkSimplex(g)}},{"./feasible-tree":85,"./network-simplex":87,"./util":88}],87:[function(require,module,exports){"use strict";var _=require("../lodash");var feasibleTree=require("./feasible-tree");var slack=require("./util").slack;var initRank=require("./util").longestPath;var preorder=require("../graphlib").alg.preorder;var postorder=require("../graphlib").alg.postorder;var simplify=require("../util").simplify;module.exports=networkSimplex;
// Expose some internals for testing purposes
networkSimplex.initLowLimValues=initLowLimValues;networkSimplex.initCutValues=initCutValues;networkSimplex.calcCutValue=calcCutValue;networkSimplex.leaveEdge=leaveEdge;networkSimplex.enterEdge=enterEdge;networkSimplex.exchangeEdges=exchangeEdges;
/*
 * The network simplex algorithm assigns ranks to each node in the input graph
 * and iteratively improves the ranking to reduce the length of edges.
 *
 * Preconditions:
 *
 *    1. The input graph must be a DAG.
 *    2. All nodes in the graph must have an object value.
 *    3. All edges in the graph must have "minlen" and "weight" attributes.
 *
 * Postconditions:
 *
 *    1. All nodes in the graph will have an assigned "rank" attribute that has
 *       been optimized by the network simplex algorithm. Ranks start at 0.
 *
 *
 * A rough sketch of the algorithm is as follows:
 *
 *    1. Assign initial ranks to each node. We use the longest path algorithm,
 *       which assigns ranks to the lowest position possible. In general this
 *       leads to very wide bottom ranks and unnecessarily long edges.
 *    2. Construct a feasible tight tree. A tight tree is one such that all
 *       edges in the tree have no slack (difference between length of edge
 *       and minlen for the edge). This by itself greatly improves the assigned
 *       rankings by shorting edges.
 *    3. Iteratively find edges that have negative cut values. Generally a
 *       negative cut value indicates that the edge could be removed and a new
 *       tree edge could be added to produce a more compact graph.
 *
 * Much of the algorithms here are derived from Gansner, et al., "A Technique
 * for Drawing Directed Graphs." The structure of the file roughly follows the
 * structure of the overall algorithm.
 */function networkSimplex(g){g=simplify(g);initRank(g);var t=feasibleTree(g);initLowLimValues(t);initCutValues(t,g);var e,f;while(e=leaveEdge(t)){f=enterEdge(t,g,e);exchangeEdges(t,g,e,f)}}
/*
 * Initializes cut values for all edges in the tree.
 */function initCutValues(t,g){var vs=postorder(t,t.nodes());vs=vs.slice(0,vs.length-1);_.forEach(vs,function(v){assignCutValue(t,g,v)})}function assignCutValue(t,g,child){var childLab=t.node(child);var parent=childLab.parent;t.edge(child,parent).cutvalue=calcCutValue(t,g,child)}
/*
 * Given the tight tree, its graph, and a child in the graph calculate and
 * return the cut value for the edge between the child and its parent.
 */function calcCutValue(t,g,child){var childLab=t.node(child);var parent=childLab.parent;
// True if the child is on the tail end of the edge in the directed graph
var childIsTail=true;
// The graph's view of the tree edge we're inspecting
var graphEdge=g.edge(child,parent);
// The accumulated cut value for the edge between this node and its parent
var cutValue=0;if(!graphEdge){childIsTail=false;graphEdge=g.edge(parent,child)}cutValue=graphEdge.weight;_.forEach(g.nodeEdges(child),function(e){var isOutEdge=e.v===child,other=isOutEdge?e.w:e.v;if(other!==parent){var pointsToHead=isOutEdge===childIsTail,otherWeight=g.edge(e).weight;cutValue+=pointsToHead?otherWeight:-otherWeight;if(isTreeEdge(t,child,other)){var otherCutValue=t.edge(child,other).cutvalue;cutValue+=pointsToHead?-otherCutValue:otherCutValue}}});return cutValue}function initLowLimValues(tree,root){if(arguments.length<2){root=tree.nodes()[0]}dfsAssignLowLim(tree,{},1,root)}function dfsAssignLowLim(tree,visited,nextLim,v,parent){var low=nextLim;var label=tree.node(v);visited[v]=true;_.forEach(tree.neighbors(v),function(w){if(!_.has(visited,w)){nextLim=dfsAssignLowLim(tree,visited,nextLim,w,v)}});label.low=low;label.lim=nextLim++;if(parent){label.parent=parent}else{
// TODO should be able to remove this when we incrementally update low lim
delete label.parent}return nextLim}function leaveEdge(tree){return _.find(tree.edges(),function(e){return tree.edge(e).cutvalue<0})}function enterEdge(t,g,edge){var v=edge.v;var w=edge.w;
// For the rest of this function we assume that v is the tail and w is the
// head, so if we don't have this edge in the graph we should flip it to
// match the correct orientation.
if(!g.hasEdge(v,w)){v=edge.w;w=edge.v}var vLabel=t.node(v);var wLabel=t.node(w);var tailLabel=vLabel;var flip=false;
// If the root is in the tail of the edge then we need to flip the logic that
// checks for the head and tail nodes in the candidates function below.
if(vLabel.lim>wLabel.lim){tailLabel=wLabel;flip=true}var candidates=_.filter(g.edges(),function(edge){return flip===isDescendant(t,t.node(edge.v),tailLabel)&&flip!==isDescendant(t,t.node(edge.w),tailLabel)});return _.minBy(candidates,function(edge){return slack(g,edge)})}function exchangeEdges(t,g,e,f){var v=e.v;var w=e.w;t.removeEdge(v,w);t.setEdge(f.v,f.w,{});initLowLimValues(t);initCutValues(t,g);updateRanks(t,g)}function updateRanks(t,g){var root=_.find(t.nodes(),function(v){return!g.node(v).parent});var vs=preorder(t,root);vs=vs.slice(1);_.forEach(vs,function(v){var parent=t.node(v).parent,edge=g.edge(v,parent),flipped=false;if(!edge){edge=g.edge(parent,v);flipped=true}g.node(v).rank=g.node(parent).rank+(flipped?edge.minlen:-edge.minlen)})}
/*
 * Returns true if the edge is in the tree.
 */function isTreeEdge(tree,u,v){return tree.hasEdge(u,v)}
/*
 * Returns true if the specified node is descendant of the root node per the
 * assigned low and lim attributes in the tree.
 */function isDescendant(tree,vLabel,rootLabel){return rootLabel.low<=vLabel.lim&&vLabel.lim<=rootLabel.lim}},{"../graphlib":67,"../lodash":70,"../util":89,"./feasible-tree":85,"./util":88}],88:[function(require,module,exports){"use strict";var _=require("../lodash");module.exports={longestPath:longestPath,slack:slack};
/*
 * Initializes ranks for the input graph using the longest path algorithm. This
 * algorithm scales well and is fast in practice, it yields rather poor
 * solutions. Nodes are pushed to the lowest layer possible, leaving the bottom
 * ranks wide and leaving edges longer than necessary. However, due to its
 * speed, this algorithm is good for getting an initial ranking that can be fed
 * into other algorithms.
 *
 * This algorithm does not normalize layers because it will be used by other
 * algorithms in most cases. If using this algorithm directly, be sure to
 * run normalize at the end.
 *
 * Pre-conditions:
 *
 *    1. Input graph is a DAG.
 *    2. Input graph node labels can be assigned properties.
 *
 * Post-conditions:
 *
 *    1. Each node will be assign an (unnormalized) "rank" property.
 */function longestPath(g){var visited={};function dfs(v){var label=g.node(v);if(_.has(visited,v)){return label.rank}visited[v]=true;var rank=_.min(_.map(g.outEdges(v),function(e){return dfs(e.w)-g.edge(e).minlen}));if(rank===Number.POSITIVE_INFINITY||// return value of _.map([]) for Lodash 3
rank===undefined||// return value of _.map([]) for Lodash 4
rank===null){// return value of _.map([null])
rank=0}return label.rank=rank}_.forEach(g.sources(),dfs)}
/*
 * Returns the amount of slack for the given edge. The slack is defined as the
 * difference between the length of the edge and its minimum length.
 */function slack(g,e){return g.node(e.w).rank-g.node(e.v).rank-g.edge(e).minlen}},{"../lodash":70}],89:[function(require,module,exports){
/* eslint "no-console": off */
"use strict";var _=require("./lodash");var Graph=require("./graphlib").Graph;module.exports={addDummyNode:addDummyNode,simplify:simplify,asNonCompoundGraph:asNonCompoundGraph,successorWeights:successorWeights,predecessorWeights:predecessorWeights,intersectRect:intersectRect,buildLayerMatrix:buildLayerMatrix,normalizeRanks:normalizeRanks,removeEmptyRanks:removeEmptyRanks,addBorderNode:addBorderNode,maxRank:maxRank,partition:partition,time:time,notime:notime};
/*
 * Adds a dummy node to the graph and return v.
 */function addDummyNode(g,type,attrs,name){var v;do{v=_.uniqueId(name)}while(g.hasNode(v));attrs.dummy=type;g.setNode(v,attrs);return v}
/*
 * Returns a new graph with only simple edges. Handles aggregation of data
 * associated with multi-edges.
 */function simplify(g){var simplified=(new Graph).setGraph(g.graph());_.forEach(g.nodes(),function(v){simplified.setNode(v,g.node(v))});_.forEach(g.edges(),function(e){var simpleLabel=simplified.edge(e.v,e.w)||{weight:0,minlen:1};var label=g.edge(e);simplified.setEdge(e.v,e.w,{weight:simpleLabel.weight+label.weight,minlen:Math.max(simpleLabel.minlen,label.minlen)})});return simplified}function asNonCompoundGraph(g){var simplified=new Graph({multigraph:g.isMultigraph()}).setGraph(g.graph());_.forEach(g.nodes(),function(v){if(!g.children(v).length){simplified.setNode(v,g.node(v))}});_.forEach(g.edges(),function(e){simplified.setEdge(e,g.edge(e))});return simplified}function successorWeights(g){var weightMap=_.map(g.nodes(),function(v){var sucs={};_.forEach(g.outEdges(v),function(e){sucs[e.w]=(sucs[e.w]||0)+g.edge(e).weight});return sucs});return _.zipObject(g.nodes(),weightMap)}function predecessorWeights(g){var weightMap=_.map(g.nodes(),function(v){var preds={};_.forEach(g.inEdges(v),function(e){preds[e.v]=(preds[e.v]||0)+g.edge(e).weight});return preds});return _.zipObject(g.nodes(),weightMap)}
/*
 * Finds where a line starting at point ({x, y}) would intersect a rectangle
 * ({x, y, width, height}) if it were pointing at the rectangle's center.
 */function intersectRect(rect,point){var x=rect.x;var y=rect.y;
// Rectangle intersection algorithm from:
// http://math.stackexchange.com/questions/108113/find-edge-between-two-boxes
var dx=point.x-x;var dy=point.y-y;var w=rect.width/2;var h=rect.height/2;if(!dx&&!dy){throw new Error("Not possible to find intersection inside of the rectangle")}var sx,sy;if(Math.abs(dy)*w>Math.abs(dx)*h){
// Intersection is top or bottom of rect.
if(dy<0){h=-h}sx=h*dx/dy;sy=h}else{
// Intersection is left or right of rect.
if(dx<0){w=-w}sx=w;sy=w*dy/dx}return{x:x+sx,y:y+sy}}
/*
 * Given a DAG with each node assigned "rank" and "order" properties, this
 * function will produce a matrix with the ids of each node.
 */function buildLayerMatrix(g){var layering=_.map(_.range(maxRank(g)+1),function(){return[]});_.forEach(g.nodes(),function(v){var node=g.node(v);var rank=node.rank;if(!_.isUndefined(rank)){layering[rank][node.order]=v}});return layering}
/*
 * Adjusts the ranks for all nodes in the graph such that all nodes v have
 * rank(v) >= 0 and at least one node w has rank(w) = 0.
 */function normalizeRanks(g){var min=_.min(_.map(g.nodes(),function(v){return g.node(v).rank}));_.forEach(g.nodes(),function(v){var node=g.node(v);if(_.has(node,"rank")){node.rank-=min}})}function removeEmptyRanks(g){
// Ranks may not start at 0, so we need to offset them
var offset=_.min(_.map(g.nodes(),function(v){return g.node(v).rank}));var layers=[];_.forEach(g.nodes(),function(v){var rank=g.node(v).rank-offset;if(!layers[rank]){layers[rank]=[]}layers[rank].push(v)});var delta=0;var nodeRankFactor=g.graph().nodeRankFactor;_.forEach(layers,function(vs,i){if(_.isUndefined(vs)&&i%nodeRankFactor!==0){--delta}else if(delta){_.forEach(vs,function(v){g.node(v).rank+=delta})}})}function addBorderNode(g,prefix,rank,order){var node={width:0,height:0};if(arguments.length>=4){node.rank=rank;node.order=order}return addDummyNode(g,"border",node,prefix)}function maxRank(g){return _.max(_.map(g.nodes(),function(v){var rank=g.node(v).rank;if(!_.isUndefined(rank)){return rank}}))}
/*
 * Partition a collection into two groups: `lhs` and `rhs`. If the supplied
 * function returns true for an entry it goes into `lhs`. Otherwise it goes
 * into `rhs.
 */function partition(collection,fn){var result={lhs:[],rhs:[]};_.forEach(collection,function(value){if(fn(value)){result.lhs.push(value)}else{result.rhs.push(value)}});return result}
/*
 * Returns a new function that wraps `fn` with a timer. The wrapper logs the
 * time it takes to execute the function.
 */function time(name,fn){var start=_.now();try{return fn()}finally{console.log(name+" time: "+(_.now()-start)+"ms")}}function notime(name,fn){return fn()}},{"./graphlib":67,"./lodash":70}],90:[function(require,module,exports){module.exports="0.8.5"},{}],91:[function(require,module,exports){
/**
 * Copyright (c) 2014, Chris Pettitt
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
var lib=require("./lib");module.exports={Graph:lib.Graph,json:require("./lib/json"),alg:require("./lib/alg"),version:lib.version}},{"./lib":107,"./lib/alg":98,"./lib/json":108}],92:[function(require,module,exports){var _=require("../lodash");module.exports=components;function components(g){var visited={};var cmpts=[];var cmpt;function dfs(v){if(_.has(visited,v))return;visited[v]=true;cmpt.push(v);_.each(g.successors(v),dfs);_.each(g.predecessors(v),dfs)}_.each(g.nodes(),function(v){cmpt=[];dfs(v);if(cmpt.length){cmpts.push(cmpt)}});return cmpts}},{"../lodash":109}],93:[function(require,module,exports){var _=require("../lodash");module.exports=dfs;
/*
 * A helper that preforms a pre- or post-order traversal on the input graph
 * and returns the nodes in the order they were visited. If the graph is
 * undirected then this algorithm will navigate using neighbors. If the graph
 * is directed then this algorithm will navigate using successors.
 *
 * Order must be one of "pre" or "post".
 */function dfs(g,vs,order){if(!_.isArray(vs)){vs=[vs]}var navigation=(g.isDirected()?g.successors:g.neighbors).bind(g);var acc=[];var visited={};_.each(vs,function(v){if(!g.hasNode(v)){throw new Error("Graph does not have node: "+v)}doDfs(g,v,order==="post",visited,navigation,acc)});return acc}function doDfs(g,v,postorder,visited,navigation,acc){if(!_.has(visited,v)){visited[v]=true;if(!postorder){acc.push(v)}_.each(navigation(v),function(w){doDfs(g,w,postorder,visited,navigation,acc)});if(postorder){acc.push(v)}}}},{"../lodash":109}],94:[function(require,module,exports){var dijkstra=require("./dijkstra");var _=require("../lodash");module.exports=dijkstraAll;function dijkstraAll(g,weightFunc,edgeFunc){return _.transform(g.nodes(),function(acc,v){acc[v]=dijkstra(g,v,weightFunc,edgeFunc)},{})}},{"../lodash":109,"./dijkstra":95}],95:[function(require,module,exports){var _=require("../lodash");var PriorityQueue=require("../data/priority-queue");module.exports=dijkstra;var DEFAULT_WEIGHT_FUNC=_.constant(1);function dijkstra(g,source,weightFn,edgeFn){return runDijkstra(g,String(source),weightFn||DEFAULT_WEIGHT_FUNC,edgeFn||function(v){return g.outEdges(v)})}function runDijkstra(g,source,weightFn,edgeFn){var results={};var pq=new PriorityQueue;var v,vEntry;var updateNeighbors=function(edge){var w=edge.v!==v?edge.v:edge.w;var wEntry=results[w];var weight=weightFn(edge);var distance=vEntry.distance+weight;if(weight<0){throw new Error("dijkstra does not allow negative edge weights. "+"Bad edge: "+edge+" Weight: "+weight)}if(distance<wEntry.distance){wEntry.distance=distance;wEntry.predecessor=v;pq.decrease(w,distance)}};g.nodes().forEach(function(v){var distance=v===source?0:Number.POSITIVE_INFINITY;results[v]={distance:distance};pq.add(v,distance)});while(pq.size()>0){v=pq.removeMin();vEntry=results[v];if(vEntry.distance===Number.POSITIVE_INFINITY){break}edgeFn(v).forEach(updateNeighbors)}return results}},{"../data/priority-queue":105,"../lodash":109}],96:[function(require,module,exports){var _=require("../lodash");var tarjan=require("./tarjan");module.exports=findCycles;function findCycles(g){return _.filter(tarjan(g),function(cmpt){return cmpt.length>1||cmpt.length===1&&g.hasEdge(cmpt[0],cmpt[0])})}},{"../lodash":109,"./tarjan":103}],97:[function(require,module,exports){var _=require("../lodash");module.exports=floydWarshall;var DEFAULT_WEIGHT_FUNC=_.constant(1);function floydWarshall(g,weightFn,edgeFn){return runFloydWarshall(g,weightFn||DEFAULT_WEIGHT_FUNC,edgeFn||function(v){return g.outEdges(v)})}function runFloydWarshall(g,weightFn,edgeFn){var results={};var nodes=g.nodes();nodes.forEach(function(v){results[v]={};results[v][v]={distance:0};nodes.forEach(function(w){if(v!==w){results[v][w]={distance:Number.POSITIVE_INFINITY}}});edgeFn(v).forEach(function(edge){var w=edge.v===v?edge.w:edge.v;var d=weightFn(edge);results[v][w]={distance:d,predecessor:v}})});nodes.forEach(function(k){var rowK=results[k];nodes.forEach(function(i){var rowI=results[i];nodes.forEach(function(j){var ik=rowI[k];var kj=rowK[j];var ij=rowI[j];var altDistance=ik.distance+kj.distance;if(altDistance<ij.distance){ij.distance=altDistance;ij.predecessor=kj.predecessor}})})});return results}},{"../lodash":109}],98:[function(require,module,exports){module.exports={components:require("./components"),dijkstra:require("./dijkstra"),dijkstraAll:require("./dijkstra-all"),findCycles:require("./find-cycles"),floydWarshall:require("./floyd-warshall"),isAcyclic:require("./is-acyclic"),postorder:require("./postorder"),preorder:require("./preorder"),prim:require("./prim"),tarjan:require("./tarjan"),topsort:require("./topsort")}},{"./components":92,"./dijkstra":95,"./dijkstra-all":94,"./find-cycles":96,"./floyd-warshall":97,"./is-acyclic":99,"./postorder":100,"./preorder":101,"./prim":102,"./tarjan":103,"./topsort":104}],99:[function(require,module,exports){var topsort=require("./topsort");module.exports=isAcyclic;function isAcyclic(g){try{topsort(g)}catch(e){if(e instanceof topsort.CycleException){return false}throw e}return true}},{"./topsort":104}],100:[function(require,module,exports){var dfs=require("./dfs");module.exports=postorder;function postorder(g,vs){return dfs(g,vs,"post")}},{"./dfs":93}],101:[function(require,module,exports){var dfs=require("./dfs");module.exports=preorder;function preorder(g,vs){return dfs(g,vs,"pre")}},{"./dfs":93}],102:[function(require,module,exports){var _=require("../lodash");var Graph=require("../graph");var PriorityQueue=require("../data/priority-queue");module.exports=prim;function prim(g,weightFunc){var result=new Graph;var parents={};var pq=new PriorityQueue;var v;function updateNeighbors(edge){var w=edge.v===v?edge.w:edge.v;var pri=pq.priority(w);if(pri!==undefined){var edgeWeight=weightFunc(edge);if(edgeWeight<pri){parents[w]=v;pq.decrease(w,edgeWeight)}}}if(g.nodeCount()===0){return result}_.each(g.nodes(),function(v){pq.add(v,Number.POSITIVE_INFINITY);result.setNode(v)});
// Start from an arbitrary node
pq.decrease(g.nodes()[0],0);var init=false;while(pq.size()>0){v=pq.removeMin();if(_.has(parents,v)){result.setEdge(v,parents[v])}else if(init){throw new Error("Input graph is not connected: "+g)}else{init=true}g.nodeEdges(v).forEach(updateNeighbors)}return result}},{"../data/priority-queue":105,"../graph":106,"../lodash":109}],103:[function(require,module,exports){var _=require("../lodash");module.exports=tarjan;function tarjan(g){var index=0;var stack=[];var visited={};// node id -> { onStack, lowlink, index }
var results=[];function dfs(v){var entry=visited[v]={onStack:true,lowlink:index,index:index++};stack.push(v);g.successors(v).forEach(function(w){if(!_.has(visited,w)){dfs(w);entry.lowlink=Math.min(entry.lowlink,visited[w].lowlink)}else if(visited[w].onStack){entry.lowlink=Math.min(entry.lowlink,visited[w].index)}});if(entry.lowlink===entry.index){var cmpt=[];var w;do{w=stack.pop();visited[w].onStack=false;cmpt.push(w)}while(v!==w);results.push(cmpt)}}g.nodes().forEach(function(v){if(!_.has(visited,v)){dfs(v)}});return results}},{"../lodash":109}],104:[function(require,module,exports){var _=require("../lodash");module.exports=topsort;topsort.CycleException=CycleException;function topsort(g){var visited={};var stack={};var results=[];function visit(node){if(_.has(stack,node)){throw new CycleException}if(!_.has(visited,node)){stack[node]=true;visited[node]=true;_.each(g.predecessors(node),visit);delete stack[node];results.push(node)}}_.each(g.sinks(),visit);if(_.size(visited)!==g.nodeCount()){throw new CycleException}return results}function CycleException(){}CycleException.prototype=new Error;// must be an instance of Error to pass testing
},{"../lodash":109}],105:[function(require,module,exports){var _=require("../lodash");module.exports=PriorityQueue;
/**
 * A min-priority queue data structure. This algorithm is derived from Cormen,
 * et al., "Introduction to Algorithms". The basic idea of a min-priority
 * queue is that you can efficiently (in O(1) time) get the smallest key in
 * the queue. Adding and removing elements takes O(log n) time. A key can
 * have its priority decreased in O(log n) time.
 */function PriorityQueue(){this._arr=[];this._keyIndices={}}
/**
 * Returns the number of elements in the queue. Takes `O(1)` time.
 */PriorityQueue.prototype.size=function(){return this._arr.length};
/**
 * Returns the keys that are in the queue. Takes `O(n)` time.
 */PriorityQueue.prototype.keys=function(){return this._arr.map(function(x){return x.key})};
/**
 * Returns `true` if **key** is in the queue and `false` if not.
 */PriorityQueue.prototype.has=function(key){return _.has(this._keyIndices,key)};
/**
 * Returns the priority for **key**. If **key** is not present in the queue
 * then this function returns `undefined`. Takes `O(1)` time.
 *
 * @param {Object} key
 */PriorityQueue.prototype.priority=function(key){var index=this._keyIndices[key];if(index!==undefined){return this._arr[index].priority}};
/**
 * Returns the key for the minimum element in this queue. If the queue is
 * empty this function throws an Error. Takes `O(1)` time.
 */PriorityQueue.prototype.min=function(){if(this.size()===0){throw new Error("Queue underflow")}return this._arr[0].key};
/**
 * Inserts a new key into the priority queue. If the key already exists in
 * the queue this function returns `false`; otherwise it will return `true`.
 * Takes `O(n)` time.
 *
 * @param {Object} key the key to add
 * @param {Number} priority the initial priority for the key
 */PriorityQueue.prototype.add=function(key,priority){var keyIndices=this._keyIndices;key=String(key);if(!_.has(keyIndices,key)){var arr=this._arr;var index=arr.length;keyIndices[key]=index;arr.push({key:key,priority:priority});this._decrease(index);return true}return false};
/**
 * Removes and returns the smallest key in the queue. Takes `O(log n)` time.
 */PriorityQueue.prototype.removeMin=function(){this._swap(0,this._arr.length-1);var min=this._arr.pop();delete this._keyIndices[min.key];this._heapify(0);return min.key};
/**
 * Decreases the priority for **key** to **priority**. If the new priority is
 * greater than the previous priority, this function will throw an Error.
 *
 * @param {Object} key the key for which to raise priority
 * @param {Number} priority the new priority for the key
 */PriorityQueue.prototype.decrease=function(key,priority){var index=this._keyIndices[key];if(priority>this._arr[index].priority){throw new Error("New priority is greater than current priority. "+"Key: "+key+" Old: "+this._arr[index].priority+" New: "+priority)}this._arr[index].priority=priority;this._decrease(index)};PriorityQueue.prototype._heapify=function(i){var arr=this._arr;var l=2*i;var r=l+1;var largest=i;if(l<arr.length){largest=arr[l].priority<arr[largest].priority?l:largest;if(r<arr.length){largest=arr[r].priority<arr[largest].priority?r:largest}if(largest!==i){this._swap(i,largest);this._heapify(largest)}}};PriorityQueue.prototype._decrease=function(index){var arr=this._arr;var priority=arr[index].priority;var parent;while(index!==0){parent=index>>1;if(arr[parent].priority<priority){break}this._swap(index,parent);index=parent}};PriorityQueue.prototype._swap=function(i,j){var arr=this._arr;var keyIndices=this._keyIndices;var origArrI=arr[i];var origArrJ=arr[j];arr[i]=origArrJ;arr[j]=origArrI;keyIndices[origArrJ.key]=i;keyIndices[origArrI.key]=j}},{"../lodash":109}],106:[function(require,module,exports){"use strict";var _=require("./lodash");module.exports=Graph;var DEFAULT_EDGE_NAME="\0";var GRAPH_NODE="\0";var EDGE_KEY_DELIM="";
// Implementation notes:
//
//  * Node id query functions should return string ids for the nodes
//  * Edge id query functions should return an "edgeObj", edge object, that is
//    composed of enough information to uniquely identify an edge: {v, w, name}.
//  * Internally we use an "edgeId", a stringified form of the edgeObj, to
//    reference edges. This is because we need a performant way to look these
//    edges up and, object properties, which have string keys, are the closest
//    we're going to get to a performant hashtable in JavaScript.
function Graph(opts){this._isDirected=_.has(opts,"directed")?opts.directed:true;this._isMultigraph=_.has(opts,"multigraph")?opts.multigraph:false;this._isCompound=_.has(opts,"compound")?opts.compound:false;
// Label for the graph itself
this._label=undefined;
// Defaults to be set when creating a new node
this._defaultNodeLabelFn=_.constant(undefined);
// Defaults to be set when creating a new edge
this._defaultEdgeLabelFn=_.constant(undefined);
// v -> label
this._nodes={};if(this._isCompound){
// v -> parent
this._parent={};
// v -> children
this._children={};this._children[GRAPH_NODE]={}}
// v -> edgeObj
this._in={};
// u -> v -> Number
this._preds={};
// v -> edgeObj
this._out={};
// v -> w -> Number
this._sucs={};
// e -> edgeObj
this._edgeObjs={};
// e -> label
this._edgeLabels={}}
/* Number of nodes in the graph. Should only be changed by the implementation. */Graph.prototype._nodeCount=0;
/* Number of edges in the graph. Should only be changed by the implementation. */Graph.prototype._edgeCount=0;
/* === Graph functions ========= */Graph.prototype.isDirected=function(){return this._isDirected};Graph.prototype.isMultigraph=function(){return this._isMultigraph};Graph.prototype.isCompound=function(){return this._isCompound};Graph.prototype.setGraph=function(label){this._label=label;return this};Graph.prototype.graph=function(){return this._label};
/* === Node functions ========== */Graph.prototype.setDefaultNodeLabel=function(newDefault){if(!_.isFunction(newDefault)){newDefault=_.constant(newDefault)}this._defaultNodeLabelFn=newDefault;return this};Graph.prototype.nodeCount=function(){return this._nodeCount};Graph.prototype.nodes=function(){return _.keys(this._nodes)};Graph.prototype.sources=function(){var self=this;return _.filter(this.nodes(),function(v){return _.isEmpty(self._in[v])})};Graph.prototype.sinks=function(){var self=this;return _.filter(this.nodes(),function(v){return _.isEmpty(self._out[v])})};Graph.prototype.setNodes=function(vs,value){var args=arguments;var self=this;_.each(vs,function(v){if(args.length>1){self.setNode(v,value)}else{self.setNode(v)}});return this};Graph.prototype.setNode=function(v,value){if(_.has(this._nodes,v)){if(arguments.length>1){this._nodes[v]=value}return this}this._nodes[v]=arguments.length>1?value:this._defaultNodeLabelFn(v);if(this._isCompound){this._parent[v]=GRAPH_NODE;this._children[v]={};this._children[GRAPH_NODE][v]=true}this._in[v]={};this._preds[v]={};this._out[v]={};this._sucs[v]={};++this._nodeCount;return this};Graph.prototype.node=function(v){return this._nodes[v]};Graph.prototype.hasNode=function(v){return _.has(this._nodes,v)};Graph.prototype.removeNode=function(v){var self=this;if(_.has(this._nodes,v)){var removeEdge=function(e){self.removeEdge(self._edgeObjs[e])};delete this._nodes[v];if(this._isCompound){this._removeFromParentsChildList(v);delete this._parent[v];_.each(this.children(v),function(child){self.setParent(child)});delete this._children[v]}_.each(_.keys(this._in[v]),removeEdge);delete this._in[v];delete this._preds[v];_.each(_.keys(this._out[v]),removeEdge);delete this._out[v];delete this._sucs[v];--this._nodeCount}return this};Graph.prototype.setParent=function(v,parent){if(!this._isCompound){throw new Error("Cannot set parent in a non-compound graph")}if(_.isUndefined(parent)){parent=GRAPH_NODE}else{
// Coerce parent to string
parent+="";for(var ancestor=parent;!_.isUndefined(ancestor);ancestor=this.parent(ancestor)){if(ancestor===v){throw new Error("Setting "+parent+" as parent of "+v+" would create a cycle")}}this.setNode(parent)}this.setNode(v);this._removeFromParentsChildList(v);this._parent[v]=parent;this._children[parent][v]=true;return this};Graph.prototype._removeFromParentsChildList=function(v){delete this._children[this._parent[v]][v]};Graph.prototype.parent=function(v){if(this._isCompound){var parent=this._parent[v];if(parent!==GRAPH_NODE){return parent}}};Graph.prototype.children=function(v){if(_.isUndefined(v)){v=GRAPH_NODE}if(this._isCompound){var children=this._children[v];if(children){return _.keys(children)}}else if(v===GRAPH_NODE){return this.nodes()}else if(this.hasNode(v)){return[]}};Graph.prototype.predecessors=function(v){var predsV=this._preds[v];if(predsV){return _.keys(predsV)}};Graph.prototype.successors=function(v){var sucsV=this._sucs[v];if(sucsV){return _.keys(sucsV)}};Graph.prototype.neighbors=function(v){var preds=this.predecessors(v);if(preds){return _.union(preds,this.successors(v))}};Graph.prototype.isLeaf=function(v){var neighbors;if(this.isDirected()){neighbors=this.successors(v)}else{neighbors=this.neighbors(v)}return neighbors.length===0};Graph.prototype.filterNodes=function(filter){var copy=new this.constructor({directed:this._isDirected,multigraph:this._isMultigraph,compound:this._isCompound});copy.setGraph(this.graph());var self=this;_.each(this._nodes,function(value,v){if(filter(v)){copy.setNode(v,value)}});_.each(this._edgeObjs,function(e){if(copy.hasNode(e.v)&&copy.hasNode(e.w)){copy.setEdge(e,self.edge(e))}});var parents={};function findParent(v){var parent=self.parent(v);if(parent===undefined||copy.hasNode(parent)){parents[v]=parent;return parent}else if(parent in parents){return parents[parent]}else{return findParent(parent)}}if(this._isCompound){_.each(copy.nodes(),function(v){copy.setParent(v,findParent(v))})}return copy};
/* === Edge functions ========== */Graph.prototype.setDefaultEdgeLabel=function(newDefault){if(!_.isFunction(newDefault)){newDefault=_.constant(newDefault)}this._defaultEdgeLabelFn=newDefault;return this};Graph.prototype.edgeCount=function(){return this._edgeCount};Graph.prototype.edges=function(){return _.values(this._edgeObjs)};Graph.prototype.setPath=function(vs,value){var self=this;var args=arguments;_.reduce(vs,function(v,w){if(args.length>1){self.setEdge(v,w,value)}else{self.setEdge(v,w)}return w});return this};
/*
 * setEdge(v, w, [value, [name]])
 * setEdge({ v, w, [name] }, [value])
 */Graph.prototype.setEdge=function(){var v,w,name,value;var valueSpecified=false;var arg0=arguments[0];if(typeof arg0==="object"&&arg0!==null&&"v"in arg0){v=arg0.v;w=arg0.w;name=arg0.name;if(arguments.length===2){value=arguments[1];valueSpecified=true}}else{v=arg0;w=arguments[1];name=arguments[3];if(arguments.length>2){value=arguments[2];valueSpecified=true}}v=""+v;w=""+w;if(!_.isUndefined(name)){name=""+name}var e=edgeArgsToId(this._isDirected,v,w,name);if(_.has(this._edgeLabels,e)){if(valueSpecified){this._edgeLabels[e]=value}return this}if(!_.isUndefined(name)&&!this._isMultigraph){throw new Error("Cannot set a named edge when isMultigraph = false")}
// It didn't exist, so we need to create it.
// First ensure the nodes exist.
this.setNode(v);this.setNode(w);this._edgeLabels[e]=valueSpecified?value:this._defaultEdgeLabelFn(v,w,name);var edgeObj=edgeArgsToObj(this._isDirected,v,w,name);
// Ensure we add undirected edges in a consistent way.
v=edgeObj.v;w=edgeObj.w;Object.freeze(edgeObj);this._edgeObjs[e]=edgeObj;incrementOrInitEntry(this._preds[w],v);incrementOrInitEntry(this._sucs[v],w);this._in[w][e]=edgeObj;this._out[v][e]=edgeObj;this._edgeCount++;return this};Graph.prototype.edge=function(v,w,name){var e=arguments.length===1?edgeObjToId(this._isDirected,arguments[0]):edgeArgsToId(this._isDirected,v,w,name);return this._edgeLabels[e]};Graph.prototype.hasEdge=function(v,w,name){var e=arguments.length===1?edgeObjToId(this._isDirected,arguments[0]):edgeArgsToId(this._isDirected,v,w,name);return _.has(this._edgeLabels,e)};Graph.prototype.removeEdge=function(v,w,name){var e=arguments.length===1?edgeObjToId(this._isDirected,arguments[0]):edgeArgsToId(this._isDirected,v,w,name);var edge=this._edgeObjs[e];if(edge){v=edge.v;w=edge.w;delete this._edgeLabels[e];delete this._edgeObjs[e];decrementOrRemoveEntry(this._preds[w],v);decrementOrRemoveEntry(this._sucs[v],w);delete this._in[w][e];delete this._out[v][e];this._edgeCount--}return this};Graph.prototype.inEdges=function(v,u){var inV=this._in[v];if(inV){var edges=_.values(inV);if(!u){return edges}return _.filter(edges,function(edge){return edge.v===u})}};Graph.prototype.outEdges=function(v,w){var outV=this._out[v];if(outV){var edges=_.values(outV);if(!w){return edges}return _.filter(edges,function(edge){return edge.w===w})}};Graph.prototype.nodeEdges=function(v,w){var inEdges=this.inEdges(v,w);if(inEdges){return inEdges.concat(this.outEdges(v,w))}};function incrementOrInitEntry(map,k){if(map[k]){map[k]++}else{map[k]=1}}function decrementOrRemoveEntry(map,k){if(!--map[k]){delete map[k]}}function edgeArgsToId(isDirected,v_,w_,name){var v=""+v_;var w=""+w_;if(!isDirected&&v>w){var tmp=v;v=w;w=tmp}return v+EDGE_KEY_DELIM+w+EDGE_KEY_DELIM+(_.isUndefined(name)?DEFAULT_EDGE_NAME:name)}function edgeArgsToObj(isDirected,v_,w_,name){var v=""+v_;var w=""+w_;if(!isDirected&&v>w){var tmp=v;v=w;w=tmp}var edgeObj={v:v,w:w};if(name){edgeObj.name=name}return edgeObj}function edgeObjToId(isDirected,edgeObj){return edgeArgsToId(isDirected,edgeObj.v,edgeObj.w,edgeObj.name)}},{"./lodash":109}],107:[function(require,module,exports){
// Includes only the "core" of graphlib
module.exports={Graph:require("./graph"),version:require("./version")}},{"./graph":106,"./version":110}],108:[function(require,module,exports){var _=require("./lodash");var Graph=require("./graph");module.exports={write:write,read:read};function write(g){var json={options:{directed:g.isDirected(),multigraph:g.isMultigraph(),compound:g.isCompound()},nodes:writeNodes(g),edges:writeEdges(g)};if(!_.isUndefined(g.graph())){json.value=_.clone(g.graph())}return json}function writeNodes(g){return _.map(g.nodes(),function(v){var nodeValue=g.node(v);var parent=g.parent(v);var node={v:v};if(!_.isUndefined(nodeValue)){node.value=nodeValue}if(!_.isUndefined(parent)){node.parent=parent}return node})}function writeEdges(g){return _.map(g.edges(),function(e){var edgeValue=g.edge(e);var edge={v:e.v,w:e.w};if(!_.isUndefined(e.name)){edge.name=e.name}if(!_.isUndefined(edgeValue)){edge.value=edgeValue}return edge})}function read(json){var g=new Graph(json.options).setGraph(json.value);_.each(json.nodes,function(entry){g.setNode(entry.v,entry.value);if(entry.parent){g.setParent(entry.v,entry.parent)}});_.each(json.edges,function(entry){g.setEdge({v:entry.v,w:entry.w,name:entry.name},entry.value)});return g}},{"./graph":106,"./lodash":109}],109:[function(require,module,exports){
/* global window */
var lodash;if(typeof require==="function"){try{lodash={clone:require("lodash/clone"),constant:require("lodash/constant"),each:require("lodash/each"),filter:require("lodash/filter"),has:require("lodash/has"),isArray:require("lodash/isArray"),isEmpty:require("lodash/isEmpty"),isFunction:require("lodash/isFunction"),isUndefined:require("lodash/isUndefined"),keys:require("lodash/keys"),map:require("lodash/map"),reduce:require("lodash/reduce"),size:require("lodash/size"),transform:require("lodash/transform"),union:require("lodash/union"),values:require("lodash/values")}}catch(e){
// continue regardless of error
}}if(!lodash){lodash=window._}module.exports=lodash},{"lodash/clone":286,"lodash/constant":288,"lodash/each":290,"lodash/filter":292,"lodash/has":299,"lodash/isArray":303,"lodash/isEmpty":307,"lodash/isFunction":308,"lodash/isUndefined":318,"lodash/keys":319,"lodash/map":322,"lodash/reduce":334,"lodash/size":335,"lodash/transform":344,"lodash/union":345,"lodash/values":347}],110:[function(require,module,exports){module.exports="2.1.8"},{}],111:[function(require,module,exports){var getNative=require("./_getNative"),root=require("./_root");
/* Built-in method references that are verified to be native. */var DataView=getNative(root,"DataView");module.exports=DataView},{"./_getNative":223,"./_root":268}],112:[function(require,module,exports){var hashClear=require("./_hashClear"),hashDelete=require("./_hashDelete"),hashGet=require("./_hashGet"),hashHas=require("./_hashHas"),hashSet=require("./_hashSet");
/**
 * Creates a hash object.
 *
 * @private
 * @constructor
 * @param {Array} [entries] The key-value pairs to cache.
 */function Hash(entries){var index=-1,length=entries==null?0:entries.length;this.clear();while(++index<length){var entry=entries[index];this.set(entry[0],entry[1])}}
// Add methods to `Hash`.
Hash.prototype.clear=hashClear;Hash.prototype["delete"]=hashDelete;Hash.prototype.get=hashGet;Hash.prototype.has=hashHas;Hash.prototype.set=hashSet;module.exports=Hash},{"./_hashClear":232,"./_hashDelete":233,"./_hashGet":234,"./_hashHas":235,"./_hashSet":236}],113:[function(require,module,exports){var listCacheClear=require("./_listCacheClear"),listCacheDelete=require("./_listCacheDelete"),listCacheGet=require("./_listCacheGet"),listCacheHas=require("./_listCacheHas"),listCacheSet=require("./_listCacheSet");
/**
 * Creates an list cache object.
 *
 * @private
 * @constructor
 * @param {Array} [entries] The key-value pairs to cache.
 */function ListCache(entries){var index=-1,length=entries==null?0:entries.length;this.clear();while(++index<length){var entry=entries[index];this.set(entry[0],entry[1])}}
// Add methods to `ListCache`.
ListCache.prototype.clear=listCacheClear;ListCache.prototype["delete"]=listCacheDelete;ListCache.prototype.get=listCacheGet;ListCache.prototype.has=listCacheHas;ListCache.prototype.set=listCacheSet;module.exports=ListCache},{"./_listCacheClear":248,"./_listCacheDelete":249,"./_listCacheGet":250,"./_listCacheHas":251,"./_listCacheSet":252}],114:[function(require,module,exports){var getNative=require("./_getNative"),root=require("./_root");
/* Built-in method references that are verified to be native. */var Map=getNative(root,"Map");module.exports=Map},{"./_getNative":223,"./_root":268}],115:[function(require,module,exports){var mapCacheClear=require("./_mapCacheClear"),mapCacheDelete=require("./_mapCacheDelete"),mapCacheGet=require("./_mapCacheGet"),mapCacheHas=require("./_mapCacheHas"),mapCacheSet=require("./_mapCacheSet");
/**
 * Creates a map cache object to store key-value pairs.
 *
 * @private
 * @constructor
 * @param {Array} [entries] The key-value pairs to cache.
 */function MapCache(entries){var index=-1,length=entries==null?0:entries.length;this.clear();while(++index<length){var entry=entries[index];this.set(entry[0],entry[1])}}
// Add methods to `MapCache`.
MapCache.prototype.clear=mapCacheClear;MapCache.prototype["delete"]=mapCacheDelete;MapCache.prototype.get=mapCacheGet;MapCache.prototype.has=mapCacheHas;MapCache.prototype.set=mapCacheSet;module.exports=MapCache},{"./_mapCacheClear":253,"./_mapCacheDelete":254,"./_mapCacheGet":255,"./_mapCacheHas":256,"./_mapCacheSet":257}],116:[function(require,module,exports){var getNative=require("./_getNative"),root=require("./_root");
/* Built-in method references that are verified to be native. */var Promise=getNative(root,"Promise");module.exports=Promise},{"./_getNative":223,"./_root":268}],117:[function(require,module,exports){var getNative=require("./_getNative"),root=require("./_root");
/* Built-in method references that are verified to be native. */var Set=getNative(root,"Set");module.exports=Set},{"./_getNative":223,"./_root":268}],118:[function(require,module,exports){var MapCache=require("./_MapCache"),setCacheAdd=require("./_setCacheAdd"),setCacheHas=require("./_setCacheHas");
/**
 *
 * Creates an array cache object to store unique values.
 *
 * @private
 * @constructor
 * @param {Array} [values] The values to cache.
 */function SetCache(values){var index=-1,length=values==null?0:values.length;this.__data__=new MapCache;while(++index<length){this.add(values[index])}}
// Add methods to `SetCache`.
SetCache.prototype.add=SetCache.prototype.push=setCacheAdd;SetCache.prototype.has=setCacheHas;module.exports=SetCache},{"./_MapCache":115,"./_setCacheAdd":270,"./_setCacheHas":271}],119:[function(require,module,exports){var ListCache=require("./_ListCache"),stackClear=require("./_stackClear"),stackDelete=require("./_stackDelete"),stackGet=require("./_stackGet"),stackHas=require("./_stackHas"),stackSet=require("./_stackSet");
/**
 * Creates a stack cache object to store key-value pairs.
 *
 * @private
 * @constructor
 * @param {Array} [entries] The key-value pairs to cache.
 */function Stack(entries){var data=this.__data__=new ListCache(entries);this.size=data.size}
// Add methods to `Stack`.
Stack.prototype.clear=stackClear;Stack.prototype["delete"]=stackDelete;Stack.prototype.get=stackGet;Stack.prototype.has=stackHas;Stack.prototype.set=stackSet;module.exports=Stack},{"./_ListCache":113,"./_stackClear":275,"./_stackDelete":276,"./_stackGet":277,"./_stackHas":278,"./_stackSet":279}],120:[function(require,module,exports){var root=require("./_root");
/** Built-in value references. */var Symbol=root.Symbol;module.exports=Symbol},{"./_root":268}],121:[function(require,module,exports){var root=require("./_root");
/** Built-in value references. */var Uint8Array=root.Uint8Array;module.exports=Uint8Array},{"./_root":268}],122:[function(require,module,exports){var getNative=require("./_getNative"),root=require("./_root");
/* Built-in method references that are verified to be native. */var WeakMap=getNative(root,"WeakMap");module.exports=WeakMap},{"./_getNative":223,"./_root":268}],123:[function(require,module,exports){
/**
 * A faster alternative to `Function#apply`, this function invokes `func`
 * with the `this` binding of `thisArg` and the arguments of `args`.
 *
 * @private
 * @param {Function} func The function to invoke.
 * @param {*} thisArg The `this` binding of `func`.
 * @param {Array} args The arguments to invoke `func` with.
 * @returns {*} Returns the result of `func`.
 */
function apply(func,thisArg,args){switch(args.length){case 0:return func.call(thisArg);case 1:return func.call(thisArg,args[0]);case 2:return func.call(thisArg,args[0],args[1]);case 3:return func.call(thisArg,args[0],args[1],args[2])}return func.apply(thisArg,args)}module.exports=apply},{}],124:[function(require,module,exports){
/**
 * A specialized version of `_.forEach` for arrays without support for
 * iteratee shorthands.
 *
 * @private
 * @param {Array} [array] The array to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @returns {Array} Returns `array`.
 */
function arrayEach(array,iteratee){var index=-1,length=array==null?0:array.length;while(++index<length){if(iteratee(array[index],index,array)===false){break}}return array}module.exports=arrayEach},{}],125:[function(require,module,exports){
/**
 * A specialized version of `_.filter` for arrays without support for
 * iteratee shorthands.
 *
 * @private
 * @param {Array} [array] The array to iterate over.
 * @param {Function} predicate The function invoked per iteration.
 * @returns {Array} Returns the new filtered array.
 */
function arrayFilter(array,predicate){var index=-1,length=array==null?0:array.length,resIndex=0,result=[];while(++index<length){var value=array[index];if(predicate(value,index,array)){result[resIndex++]=value}}return result}module.exports=arrayFilter},{}],126:[function(require,module,exports){var baseIndexOf=require("./_baseIndexOf");
/**
 * A specialized version of `_.includes` for arrays without support for
 * specifying an index to search from.
 *
 * @private
 * @param {Array} [array] The array to inspect.
 * @param {*} target The value to search for.
 * @returns {boolean} Returns `true` if `target` is found, else `false`.
 */function arrayIncludes(array,value){var length=array==null?0:array.length;return!!length&&baseIndexOf(array,value,0)>-1}module.exports=arrayIncludes},{"./_baseIndexOf":155}],127:[function(require,module,exports){
/**
 * This function is like `arrayIncludes` except that it accepts a comparator.
 *
 * @private
 * @param {Array} [array] The array to inspect.
 * @param {*} target The value to search for.
 * @param {Function} comparator The comparator invoked per element.
 * @returns {boolean} Returns `true` if `target` is found, else `false`.
 */
function arrayIncludesWith(array,value,comparator){var index=-1,length=array==null?0:array.length;while(++index<length){if(comparator(value,array[index])){return true}}return false}module.exports=arrayIncludesWith},{}],128:[function(require,module,exports){var baseTimes=require("./_baseTimes"),isArguments=require("./isArguments"),isArray=require("./isArray"),isBuffer=require("./isBuffer"),isIndex=require("./_isIndex"),isTypedArray=require("./isTypedArray");
/** Used for built-in method references. */var objectProto=Object.prototype;
/** Used to check objects for own properties. */var hasOwnProperty=objectProto.hasOwnProperty;
/**
 * Creates an array of the enumerable property names of the array-like `value`.
 *
 * @private
 * @param {*} value The value to query.
 * @param {boolean} inherited Specify returning inherited property names.
 * @returns {Array} Returns the array of property names.
 */function arrayLikeKeys(value,inherited){var isArr=isArray(value),isArg=!isArr&&isArguments(value),isBuff=!isArr&&!isArg&&isBuffer(value),isType=!isArr&&!isArg&&!isBuff&&isTypedArray(value),skipIndexes=isArr||isArg||isBuff||isType,result=skipIndexes?baseTimes(value.length,String):[],length=result.length;for(var key in value){if((inherited||hasOwnProperty.call(value,key))&&!(skipIndexes&&(
// Safari 9 has enumerable `arguments.length` in strict mode.
key=="length"||
// Node.js 0.10 has enumerable non-index properties on buffers.
isBuff&&(key=="offset"||key=="parent")||
// PhantomJS 2 has enumerable non-index properties on typed arrays.
isType&&(key=="buffer"||key=="byteLength"||key=="byteOffset")||
// Skip index properties.
isIndex(key,length)))){result.push(key)}}return result}module.exports=arrayLikeKeys},{"./_baseTimes":185,"./_isIndex":241,"./isArguments":302,"./isArray":303,"./isBuffer":306,"./isTypedArray":317}],129:[function(require,module,exports){
/**
 * A specialized version of `_.map` for arrays without support for iteratee
 * shorthands.
 *
 * @private
 * @param {Array} [array] The array to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @returns {Array} Returns the new mapped array.
 */
function arrayMap(array,iteratee){var index=-1,length=array==null?0:array.length,result=Array(length);while(++index<length){result[index]=iteratee(array[index],index,array)}return result}module.exports=arrayMap},{}],130:[function(require,module,exports){
/**
 * Appends the elements of `values` to `array`.
 *
 * @private
 * @param {Array} array The array to modify.
 * @param {Array} values The values to append.
 * @returns {Array} Returns `array`.
 */
function arrayPush(array,values){var index=-1,length=values.length,offset=array.length;while(++index<length){array[offset+index]=values[index]}return array}module.exports=arrayPush},{}],131:[function(require,module,exports){
/**
 * A specialized version of `_.reduce` for arrays without support for
 * iteratee shorthands.
 *
 * @private
 * @param {Array} [array] The array to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @param {*} [accumulator] The initial value.
 * @param {boolean} [initAccum] Specify using the first element of `array` as
 *  the initial value.
 * @returns {*} Returns the accumulated value.
 */
function arrayReduce(array,iteratee,accumulator,initAccum){var index=-1,length=array==null?0:array.length;if(initAccum&&length){accumulator=array[++index]}while(++index<length){accumulator=iteratee(accumulator,array[index],index,array)}return accumulator}module.exports=arrayReduce},{}],132:[function(require,module,exports){
/**
 * A specialized version of `_.some` for arrays without support for iteratee
 * shorthands.
 *
 * @private
 * @param {Array} [array] The array to iterate over.
 * @param {Function} predicate The function invoked per iteration.
 * @returns {boolean} Returns `true` if any element passes the predicate check,
 *  else `false`.
 */
function arraySome(array,predicate){var index=-1,length=array==null?0:array.length;while(++index<length){if(predicate(array[index],index,array)){return true}}return false}module.exports=arraySome},{}],133:[function(require,module,exports){var baseProperty=require("./_baseProperty");
/**
 * Gets the size of an ASCII `string`.
 *
 * @private
 * @param {string} string The string inspect.
 * @returns {number} Returns the string size.
 */var asciiSize=baseProperty("length");module.exports=asciiSize},{"./_baseProperty":177}],134:[function(require,module,exports){var baseAssignValue=require("./_baseAssignValue"),eq=require("./eq");
/**
 * This function is like `assignValue` except that it doesn't assign
 * `undefined` values.
 *
 * @private
 * @param {Object} object The object to modify.
 * @param {string} key The key of the property to assign.
 * @param {*} value The value to assign.
 */function assignMergeValue(object,key,value){if(value!==undefined&&!eq(object[key],value)||value===undefined&&!(key in object)){baseAssignValue(object,key,value)}}module.exports=assignMergeValue},{"./_baseAssignValue":139,"./eq":291}],135:[function(require,module,exports){var baseAssignValue=require("./_baseAssignValue"),eq=require("./eq");
/** Used for built-in method references. */var objectProto=Object.prototype;
/** Used to check objects for own properties. */var hasOwnProperty=objectProto.hasOwnProperty;
/**
 * Assigns `value` to `key` of `object` if the existing value is not equivalent
 * using [`SameValueZero`](http://ecma-international.org/ecma-262/7.0/#sec-samevaluezero)
 * for equality comparisons.
 *
 * @private
 * @param {Object} object The object to modify.
 * @param {string} key The key of the property to assign.
 * @param {*} value The value to assign.
 */function assignValue(object,key,value){var objValue=object[key];if(!(hasOwnProperty.call(object,key)&&eq(objValue,value))||value===undefined&&!(key in object)){baseAssignValue(object,key,value)}}module.exports=assignValue},{"./_baseAssignValue":139,"./eq":291}],136:[function(require,module,exports){var eq=require("./eq");
/**
 * Gets the index at which the `key` is found in `array` of key-value pairs.
 *
 * @private
 * @param {Array} array The array to inspect.
 * @param {*} key The key to search for.
 * @returns {number} Returns the index of the matched value, else `-1`.
 */function assocIndexOf(array,key){var length=array.length;while(length--){if(eq(array[length][0],key)){return length}}return-1}module.exports=assocIndexOf},{"./eq":291}],137:[function(require,module,exports){var copyObject=require("./_copyObject"),keys=require("./keys");
/**
 * The base implementation of `_.assign` without support for multiple sources
 * or `customizer` functions.
 *
 * @private
 * @param {Object} object The destination object.
 * @param {Object} source The source object.
 * @returns {Object} Returns `object`.
 */function baseAssign(object,source){return object&&copyObject(source,keys(source),object)}module.exports=baseAssign},{"./_copyObject":203,"./keys":319}],138:[function(require,module,exports){var copyObject=require("./_copyObject"),keysIn=require("./keysIn");
/**
 * The base implementation of `_.assignIn` without support for multiple sources
 * or `customizer` functions.
 *
 * @private
 * @param {Object} object The destination object.
 * @param {Object} source The source object.
 * @returns {Object} Returns `object`.
 */function baseAssignIn(object,source){return object&&copyObject(source,keysIn(source),object)}module.exports=baseAssignIn},{"./_copyObject":203,"./keysIn":320}],139:[function(require,module,exports){var defineProperty=require("./_defineProperty");
/**
 * The base implementation of `assignValue` and `assignMergeValue` without
 * value checks.
 *
 * @private
 * @param {Object} object The object to modify.
 * @param {string} key The key of the property to assign.
 * @param {*} value The value to assign.
 */function baseAssignValue(object,key,value){if(key=="__proto__"&&defineProperty){defineProperty(object,key,{configurable:true,enumerable:true,value:value,writable:true})}else{object[key]=value}}module.exports=baseAssignValue},{"./_defineProperty":213}],140:[function(require,module,exports){var Stack=require("./_Stack"),arrayEach=require("./_arrayEach"),assignValue=require("./_assignValue"),baseAssign=require("./_baseAssign"),baseAssignIn=require("./_baseAssignIn"),cloneBuffer=require("./_cloneBuffer"),copyArray=require("./_copyArray"),copySymbols=require("./_copySymbols"),copySymbolsIn=require("./_copySymbolsIn"),getAllKeys=require("./_getAllKeys"),getAllKeysIn=require("./_getAllKeysIn"),getTag=require("./_getTag"),initCloneArray=require("./_initCloneArray"),initCloneByTag=require("./_initCloneByTag"),initCloneObject=require("./_initCloneObject"),isArray=require("./isArray"),isBuffer=require("./isBuffer"),isMap=require("./isMap"),isObject=require("./isObject"),isSet=require("./isSet"),keys=require("./keys");
/** Used to compose bitmasks for cloning. */var CLONE_DEEP_FLAG=1,CLONE_FLAT_FLAG=2,CLONE_SYMBOLS_FLAG=4;
/** `Object#toString` result references. */var argsTag="[object Arguments]",arrayTag="[object Array]",boolTag="[object Boolean]",dateTag="[object Date]",errorTag="[object Error]",funcTag="[object Function]",genTag="[object GeneratorFunction]",mapTag="[object Map]",numberTag="[object Number]",objectTag="[object Object]",regexpTag="[object RegExp]",setTag="[object Set]",stringTag="[object String]",symbolTag="[object Symbol]",weakMapTag="[object WeakMap]";var arrayBufferTag="[object ArrayBuffer]",dataViewTag="[object DataView]",float32Tag="[object Float32Array]",float64Tag="[object Float64Array]",int8Tag="[object Int8Array]",int16Tag="[object Int16Array]",int32Tag="[object Int32Array]",uint8Tag="[object Uint8Array]",uint8ClampedTag="[object Uint8ClampedArray]",uint16Tag="[object Uint16Array]",uint32Tag="[object Uint32Array]";
/** Used to identify `toStringTag` values supported by `_.clone`. */var cloneableTags={};cloneableTags[argsTag]=cloneableTags[arrayTag]=cloneableTags[arrayBufferTag]=cloneableTags[dataViewTag]=cloneableTags[boolTag]=cloneableTags[dateTag]=cloneableTags[float32Tag]=cloneableTags[float64Tag]=cloneableTags[int8Tag]=cloneableTags[int16Tag]=cloneableTags[int32Tag]=cloneableTags[mapTag]=cloneableTags[numberTag]=cloneableTags[objectTag]=cloneableTags[regexpTag]=cloneableTags[setTag]=cloneableTags[stringTag]=cloneableTags[symbolTag]=cloneableTags[uint8Tag]=cloneableTags[uint8ClampedTag]=cloneableTags[uint16Tag]=cloneableTags[uint32Tag]=true;cloneableTags[errorTag]=cloneableTags[funcTag]=cloneableTags[weakMapTag]=false;
/**
 * The base implementation of `_.clone` and `_.cloneDeep` which tracks
 * traversed objects.
 *
 * @private
 * @param {*} value The value to clone.
 * @param {boolean} bitmask The bitmask flags.
 *  1 - Deep clone
 *  2 - Flatten inherited properties
 *  4 - Clone symbols
 * @param {Function} [customizer] The function to customize cloning.
 * @param {string} [key] The key of `value`.
 * @param {Object} [object] The parent object of `value`.
 * @param {Object} [stack] Tracks traversed objects and their clone counterparts.
 * @returns {*} Returns the cloned value.
 */function baseClone(value,bitmask,customizer,key,object,stack){var result,isDeep=bitmask&CLONE_DEEP_FLAG,isFlat=bitmask&CLONE_FLAT_FLAG,isFull=bitmask&CLONE_SYMBOLS_FLAG;if(customizer){result=object?customizer(value,key,object,stack):customizer(value)}if(result!==undefined){return result}if(!isObject(value)){return value}var isArr=isArray(value);if(isArr){result=initCloneArray(value);if(!isDeep){return copyArray(value,result)}}else{var tag=getTag(value),isFunc=tag==funcTag||tag==genTag;if(isBuffer(value)){return cloneBuffer(value,isDeep)}if(tag==objectTag||tag==argsTag||isFunc&&!object){result=isFlat||isFunc?{}:initCloneObject(value);if(!isDeep){return isFlat?copySymbolsIn(value,baseAssignIn(result,value)):copySymbols(value,baseAssign(result,value))}}else{if(!cloneableTags[tag]){return object?value:{}}result=initCloneByTag(value,tag,isDeep)}}
// Check for circular references and return its corresponding clone.
stack||(stack=new Stack);var stacked=stack.get(value);if(stacked){return stacked}stack.set(value,result);if(isSet(value)){value.forEach(function(subValue){result.add(baseClone(subValue,bitmask,customizer,subValue,value,stack))})}else if(isMap(value)){value.forEach(function(subValue,key){result.set(key,baseClone(subValue,bitmask,customizer,key,value,stack))})}var keysFunc=isFull?isFlat?getAllKeysIn:getAllKeys:isFlat?keysIn:keys;var props=isArr?undefined:keysFunc(value);arrayEach(props||value,function(subValue,key){if(props){key=subValue;subValue=value[key]}
// Recursively populate clone (susceptible to call stack limits).
assignValue(result,key,baseClone(subValue,bitmask,customizer,key,value,stack))});return result}module.exports=baseClone},{"./_Stack":119,"./_arrayEach":124,"./_assignValue":135,"./_baseAssign":137,"./_baseAssignIn":138,"./_cloneBuffer":195,"./_copyArray":202,"./_copySymbols":204,"./_copySymbolsIn":205,"./_getAllKeys":219,"./_getAllKeysIn":220,"./_getTag":228,"./_initCloneArray":237,"./_initCloneByTag":238,"./_initCloneObject":239,"./isArray":303,"./isBuffer":306,"./isMap":310,"./isObject":311,"./isSet":314,"./keys":319}],141:[function(require,module,exports){var isObject=require("./isObject");
/** Built-in value references. */var objectCreate=Object.create;
/**
 * The base implementation of `_.create` without support for assigning
 * properties to the created object.
 *
 * @private
 * @param {Object} proto The object to inherit from.
 * @returns {Object} Returns the new object.
 */var baseCreate=function(){function object(){}return function(proto){if(!isObject(proto)){return{}}if(objectCreate){return objectCreate(proto)}object.prototype=proto;var result=new object;object.prototype=undefined;return result}}();module.exports=baseCreate},{"./isObject":311}],142:[function(require,module,exports){var baseForOwn=require("./_baseForOwn"),createBaseEach=require("./_createBaseEach");
/**
 * The base implementation of `_.forEach` without support for iteratee shorthands.
 *
 * @private
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @returns {Array|Object} Returns `collection`.
 */var baseEach=createBaseEach(baseForOwn);module.exports=baseEach},{"./_baseForOwn":148,"./_createBaseEach":208}],143:[function(require,module,exports){var isSymbol=require("./isSymbol");
/**
 * The base implementation of methods like `_.max` and `_.min` which accepts a
 * `comparator` to determine the extremum value.
 *
 * @private
 * @param {Array} array The array to iterate over.
 * @param {Function} iteratee The iteratee invoked per iteration.
 * @param {Function} comparator The comparator used to compare values.
 * @returns {*} Returns the extremum value.
 */function baseExtremum(array,iteratee,comparator){var index=-1,length=array.length;while(++index<length){var value=array[index],current=iteratee(value);if(current!=null&&(computed===undefined?current===current&&!isSymbol(current):comparator(current,computed))){var computed=current,result=value}}return result}module.exports=baseExtremum},{"./isSymbol":316}],144:[function(require,module,exports){var baseEach=require("./_baseEach");
/**
 * The base implementation of `_.filter` without support for iteratee shorthands.
 *
 * @private
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function} predicate The function invoked per iteration.
 * @returns {Array} Returns the new filtered array.
 */function baseFilter(collection,predicate){var result=[];baseEach(collection,function(value,index,collection){if(predicate(value,index,collection)){result.push(value)}});return result}module.exports=baseFilter},{"./_baseEach":142}],145:[function(require,module,exports){
/**
 * The base implementation of `_.findIndex` and `_.findLastIndex` without
 * support for iteratee shorthands.
 *
 * @private
 * @param {Array} array The array to inspect.
 * @param {Function} predicate The function invoked per iteration.
 * @param {number} fromIndex The index to search from.
 * @param {boolean} [fromRight] Specify iterating from right to left.
 * @returns {number} Returns the index of the matched value, else `-1`.
 */
function baseFindIndex(array,predicate,fromIndex,fromRight){var length=array.length,index=fromIndex+(fromRight?1:-1);while(fromRight?index--:++index<length){if(predicate(array[index],index,array)){return index}}return-1}module.exports=baseFindIndex},{}],146:[function(require,module,exports){var arrayPush=require("./_arrayPush"),isFlattenable=require("./_isFlattenable");
/**
 * The base implementation of `_.flatten` with support for restricting flattening.
 *
 * @private
 * @param {Array} array The array to flatten.
 * @param {number} depth The maximum recursion depth.
 * @param {boolean} [predicate=isFlattenable] The function invoked per iteration.
 * @param {boolean} [isStrict] Restrict to values that pass `predicate` checks.
 * @param {Array} [result=[]] The initial result value.
 * @returns {Array} Returns the new flattened array.
 */function baseFlatten(array,depth,predicate,isStrict,result){var index=-1,length=array.length;predicate||(predicate=isFlattenable);result||(result=[]);while(++index<length){var value=array[index];if(depth>0&&predicate(value)){if(depth>1){
// Recursively flatten arrays (susceptible to call stack limits).
baseFlatten(value,depth-1,predicate,isStrict,result)}else{arrayPush(result,value)}}else if(!isStrict){result[result.length]=value}}return result}module.exports=baseFlatten},{"./_arrayPush":130,"./_isFlattenable":240}],147:[function(require,module,exports){var createBaseFor=require("./_createBaseFor");
/**
 * The base implementation of `baseForOwn` which iterates over `object`
 * properties returned by `keysFunc` and invokes `iteratee` for each property.
 * Iteratee functions may exit iteration early by explicitly returning `false`.
 *
 * @private
 * @param {Object} object The object to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @param {Function} keysFunc The function to get the keys of `object`.
 * @returns {Object} Returns `object`.
 */var baseFor=createBaseFor();module.exports=baseFor},{"./_createBaseFor":209}],148:[function(require,module,exports){var baseFor=require("./_baseFor"),keys=require("./keys");
/**
 * The base implementation of `_.forOwn` without support for iteratee shorthands.
 *
 * @private
 * @param {Object} object The object to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @returns {Object} Returns `object`.
 */function baseForOwn(object,iteratee){return object&&baseFor(object,iteratee,keys)}module.exports=baseForOwn},{"./_baseFor":147,"./keys":319}],149:[function(require,module,exports){var castPath=require("./_castPath"),toKey=require("./_toKey");
/**
 * The base implementation of `_.get` without support for default values.
 *
 * @private
 * @param {Object} object The object to query.
 * @param {Array|string} path The path of the property to get.
 * @returns {*} Returns the resolved value.
 */function baseGet(object,path){path=castPath(path,object);var index=0,length=path.length;while(object!=null&&index<length){object=object[toKey(path[index++])]}return index&&index==length?object:undefined}module.exports=baseGet},{"./_castPath":193,"./_toKey":283}],150:[function(require,module,exports){var arrayPush=require("./_arrayPush"),isArray=require("./isArray");
/**
 * The base implementation of `getAllKeys` and `getAllKeysIn` which uses
 * `keysFunc` and `symbolsFunc` to get the enumerable property names and
 * symbols of `object`.
 *
 * @private
 * @param {Object} object The object to query.
 * @param {Function} keysFunc The function to get the keys of `object`.
 * @param {Function} symbolsFunc The function to get the symbols of `object`.
 * @returns {Array} Returns the array of property names and symbols.
 */function baseGetAllKeys(object,keysFunc,symbolsFunc){var result=keysFunc(object);return isArray(object)?result:arrayPush(result,symbolsFunc(object))}module.exports=baseGetAllKeys},{"./_arrayPush":130,"./isArray":303}],151:[function(require,module,exports){var Symbol=require("./_Symbol"),getRawTag=require("./_getRawTag"),objectToString=require("./_objectToString");
/** `Object#toString` result references. */var nullTag="[object Null]",undefinedTag="[object Undefined]";
/** Built-in value references. */var symToStringTag=Symbol?Symbol.toStringTag:undefined;
/**
 * The base implementation of `getTag` without fallbacks for buggy environments.
 *
 * @private
 * @param {*} value The value to query.
 * @returns {string} Returns the `toStringTag`.
 */function baseGetTag(value){if(value==null){return value===undefined?undefinedTag:nullTag}return symToStringTag&&symToStringTag in Object(value)?getRawTag(value):objectToString(value)}module.exports=baseGetTag},{"./_Symbol":120,"./_getRawTag":225,"./_objectToString":265}],152:[function(require,module,exports){
/**
 * The base implementation of `_.gt` which doesn't coerce arguments.
 *
 * @private
 * @param {*} value The value to compare.
 * @param {*} other The other value to compare.
 * @returns {boolean} Returns `true` if `value` is greater than `other`,
 *  else `false`.
 */
function baseGt(value,other){return value>other}module.exports=baseGt},{}],153:[function(require,module,exports){
/** Used for built-in method references. */
var objectProto=Object.prototype;
/** Used to check objects for own properties. */var hasOwnProperty=objectProto.hasOwnProperty;
/**
 * The base implementation of `_.has` without support for deep paths.
 *
 * @private
 * @param {Object} [object] The object to query.
 * @param {Array|string} key The key to check.
 * @returns {boolean} Returns `true` if `key` exists, else `false`.
 */function baseHas(object,key){return object!=null&&hasOwnProperty.call(object,key)}module.exports=baseHas},{}],154:[function(require,module,exports){
/**
 * The base implementation of `_.hasIn` without support for deep paths.
 *
 * @private
 * @param {Object} [object] The object to query.
 * @param {Array|string} key The key to check.
 * @returns {boolean} Returns `true` if `key` exists, else `false`.
 */
function baseHasIn(object,key){return object!=null&&key in Object(object)}module.exports=baseHasIn},{}],155:[function(require,module,exports){var baseFindIndex=require("./_baseFindIndex"),baseIsNaN=require("./_baseIsNaN"),strictIndexOf=require("./_strictIndexOf");
/**
 * The base implementation of `_.indexOf` without `fromIndex` bounds checks.
 *
 * @private
 * @param {Array} array The array to inspect.
 * @param {*} value The value to search for.
 * @param {number} fromIndex The index to search from.
 * @returns {number} Returns the index of the matched value, else `-1`.
 */function baseIndexOf(array,value,fromIndex){return value===value?strictIndexOf(array,value,fromIndex):baseFindIndex(array,baseIsNaN,fromIndex)}module.exports=baseIndexOf},{"./_baseFindIndex":145,"./_baseIsNaN":161,"./_strictIndexOf":280}],156:[function(require,module,exports){var baseGetTag=require("./_baseGetTag"),isObjectLike=require("./isObjectLike");
/** `Object#toString` result references. */var argsTag="[object Arguments]";
/**
 * The base implementation of `_.isArguments`.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is an `arguments` object,
 */function baseIsArguments(value){return isObjectLike(value)&&baseGetTag(value)==argsTag}module.exports=baseIsArguments},{"./_baseGetTag":151,"./isObjectLike":312}],157:[function(require,module,exports){var baseIsEqualDeep=require("./_baseIsEqualDeep"),isObjectLike=require("./isObjectLike");
/**
 * The base implementation of `_.isEqual` which supports partial comparisons
 * and tracks traversed objects.
 *
 * @private
 * @param {*} value The value to compare.
 * @param {*} other The other value to compare.
 * @param {boolean} bitmask The bitmask flags.
 *  1 - Unordered comparison
 *  2 - Partial comparison
 * @param {Function} [customizer] The function to customize comparisons.
 * @param {Object} [stack] Tracks traversed `value` and `other` objects.
 * @returns {boolean} Returns `true` if the values are equivalent, else `false`.
 */function baseIsEqual(value,other,bitmask,customizer,stack){if(value===other){return true}if(value==null||other==null||!isObjectLike(value)&&!isObjectLike(other)){return value!==value&&other!==other}return baseIsEqualDeep(value,other,bitmask,customizer,baseIsEqual,stack)}module.exports=baseIsEqual},{"./_baseIsEqualDeep":158,"./isObjectLike":312}],158:[function(require,module,exports){var Stack=require("./_Stack"),equalArrays=require("./_equalArrays"),equalByTag=require("./_equalByTag"),equalObjects=require("./_equalObjects"),getTag=require("./_getTag"),isArray=require("./isArray"),isBuffer=require("./isBuffer"),isTypedArray=require("./isTypedArray");
/** Used to compose bitmasks for value comparisons. */var COMPARE_PARTIAL_FLAG=1;
/** `Object#toString` result references. */var argsTag="[object Arguments]",arrayTag="[object Array]",objectTag="[object Object]";
/** Used for built-in method references. */var objectProto=Object.prototype;
/** Used to check objects for own properties. */var hasOwnProperty=objectProto.hasOwnProperty;
/**
 * A specialized version of `baseIsEqual` for arrays and objects which performs
 * deep comparisons and tracks traversed objects enabling objects with circular
 * references to be compared.
 *
 * @private
 * @param {Object} object The object to compare.
 * @param {Object} other The other object to compare.
 * @param {number} bitmask The bitmask flags. See `baseIsEqual` for more details.
 * @param {Function} customizer The function to customize comparisons.
 * @param {Function} equalFunc The function to determine equivalents of values.
 * @param {Object} [stack] Tracks traversed `object` and `other` objects.
 * @returns {boolean} Returns `true` if the objects are equivalent, else `false`.
 */function baseIsEqualDeep(object,other,bitmask,customizer,equalFunc,stack){var objIsArr=isArray(object),othIsArr=isArray(other),objTag=objIsArr?arrayTag:getTag(object),othTag=othIsArr?arrayTag:getTag(other);objTag=objTag==argsTag?objectTag:objTag;othTag=othTag==argsTag?objectTag:othTag;var objIsObj=objTag==objectTag,othIsObj=othTag==objectTag,isSameTag=objTag==othTag;if(isSameTag&&isBuffer(object)){if(!isBuffer(other)){return false}objIsArr=true;objIsObj=false}if(isSameTag&&!objIsObj){stack||(stack=new Stack);return objIsArr||isTypedArray(object)?equalArrays(object,other,bitmask,customizer,equalFunc,stack):equalByTag(object,other,objTag,bitmask,customizer,equalFunc,stack)}if(!(bitmask&COMPARE_PARTIAL_FLAG)){var objIsWrapped=objIsObj&&hasOwnProperty.call(object,"__wrapped__"),othIsWrapped=othIsObj&&hasOwnProperty.call(other,"__wrapped__");if(objIsWrapped||othIsWrapped){var objUnwrapped=objIsWrapped?object.value():object,othUnwrapped=othIsWrapped?other.value():other;stack||(stack=new Stack);return equalFunc(objUnwrapped,othUnwrapped,bitmask,customizer,stack)}}if(!isSameTag){return false}stack||(stack=new Stack);return equalObjects(object,other,bitmask,customizer,equalFunc,stack)}module.exports=baseIsEqualDeep},{"./_Stack":119,"./_equalArrays":214,"./_equalByTag":215,"./_equalObjects":216,"./_getTag":228,"./isArray":303,"./isBuffer":306,"./isTypedArray":317}],159:[function(require,module,exports){var getTag=require("./_getTag"),isObjectLike=require("./isObjectLike");
/** `Object#toString` result references. */var mapTag="[object Map]";
/**
 * The base implementation of `_.isMap` without Node.js optimizations.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a map, else `false`.
 */function baseIsMap(value){return isObjectLike(value)&&getTag(value)==mapTag}module.exports=baseIsMap},{"./_getTag":228,"./isObjectLike":312}],160:[function(require,module,exports){var Stack=require("./_Stack"),baseIsEqual=require("./_baseIsEqual");
/** Used to compose bitmasks for value comparisons. */var COMPARE_PARTIAL_FLAG=1,COMPARE_UNORDERED_FLAG=2;
/**
 * The base implementation of `_.isMatch` without support for iteratee shorthands.
 *
 * @private
 * @param {Object} object The object to inspect.
 * @param {Object} source The object of property values to match.
 * @param {Array} matchData The property names, values, and compare flags to match.
 * @param {Function} [customizer] The function to customize comparisons.
 * @returns {boolean} Returns `true` if `object` is a match, else `false`.
 */function baseIsMatch(object,source,matchData,customizer){var index=matchData.length,length=index,noCustomizer=!customizer;if(object==null){return!length}object=Object(object);while(index--){var data=matchData[index];if(noCustomizer&&data[2]?data[1]!==object[data[0]]:!(data[0]in object)){return false}}while(++index<length){data=matchData[index];var key=data[0],objValue=object[key],srcValue=data[1];if(noCustomizer&&data[2]){if(objValue===undefined&&!(key in object)){return false}}else{var stack=new Stack;if(customizer){var result=customizer(objValue,srcValue,key,object,source,stack)}if(!(result===undefined?baseIsEqual(srcValue,objValue,COMPARE_PARTIAL_FLAG|COMPARE_UNORDERED_FLAG,customizer,stack):result)){return false}}}return true}module.exports=baseIsMatch},{"./_Stack":119,"./_baseIsEqual":157}],161:[function(require,module,exports){
/**
 * The base implementation of `_.isNaN` without support for number objects.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is `NaN`, else `false`.
 */
function baseIsNaN(value){return value!==value}module.exports=baseIsNaN},{}],162:[function(require,module,exports){var isFunction=require("./isFunction"),isMasked=require("./_isMasked"),isObject=require("./isObject"),toSource=require("./_toSource");
/**
 * Used to match `RegExp`
 * [syntax characters](http://ecma-international.org/ecma-262/7.0/#sec-patterns).
 */var reRegExpChar=/[\\^$.*+?()[\]{}|]/g;
/** Used to detect host constructors (Safari). */var reIsHostCtor=/^\[object .+?Constructor\]$/;
/** Used for built-in method references. */var funcProto=Function.prototype,objectProto=Object.prototype;
/** Used to resolve the decompiled source of functions. */var funcToString=funcProto.toString;
/** Used to check objects for own properties. */var hasOwnProperty=objectProto.hasOwnProperty;
/** Used to detect if a method is native. */var reIsNative=RegExp("^"+funcToString.call(hasOwnProperty).replace(reRegExpChar,"\\$&").replace(/hasOwnProperty|(function).*?(?=\\\()| for .+?(?=\\\])/g,"$1.*?")+"$");
/**
 * The base implementation of `_.isNative` without bad shim checks.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a native function,
 *  else `false`.
 */function baseIsNative(value){if(!isObject(value)||isMasked(value)){return false}var pattern=isFunction(value)?reIsNative:reIsHostCtor;return pattern.test(toSource(value))}module.exports=baseIsNative},{"./_isMasked":245,"./_toSource":284,"./isFunction":308,"./isObject":311}],163:[function(require,module,exports){var getTag=require("./_getTag"),isObjectLike=require("./isObjectLike");
/** `Object#toString` result references. */var setTag="[object Set]";
/**
 * The base implementation of `_.isSet` without Node.js optimizations.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a set, else `false`.
 */function baseIsSet(value){return isObjectLike(value)&&getTag(value)==setTag}module.exports=baseIsSet},{"./_getTag":228,"./isObjectLike":312}],164:[function(require,module,exports){var baseGetTag=require("./_baseGetTag"),isLength=require("./isLength"),isObjectLike=require("./isObjectLike");
/** `Object#toString` result references. */var argsTag="[object Arguments]",arrayTag="[object Array]",boolTag="[object Boolean]",dateTag="[object Date]",errorTag="[object Error]",funcTag="[object Function]",mapTag="[object Map]",numberTag="[object Number]",objectTag="[object Object]",regexpTag="[object RegExp]",setTag="[object Set]",stringTag="[object String]",weakMapTag="[object WeakMap]";var arrayBufferTag="[object ArrayBuffer]",dataViewTag="[object DataView]",float32Tag="[object Float32Array]",float64Tag="[object Float64Array]",int8Tag="[object Int8Array]",int16Tag="[object Int16Array]",int32Tag="[object Int32Array]",uint8Tag="[object Uint8Array]",uint8ClampedTag="[object Uint8ClampedArray]",uint16Tag="[object Uint16Array]",uint32Tag="[object Uint32Array]";
/** Used to identify `toStringTag` values of typed arrays. */var typedArrayTags={};typedArrayTags[float32Tag]=typedArrayTags[float64Tag]=typedArrayTags[int8Tag]=typedArrayTags[int16Tag]=typedArrayTags[int32Tag]=typedArrayTags[uint8Tag]=typedArrayTags[uint8ClampedTag]=typedArrayTags[uint16Tag]=typedArrayTags[uint32Tag]=true;typedArrayTags[argsTag]=typedArrayTags[arrayTag]=typedArrayTags[arrayBufferTag]=typedArrayTags[boolTag]=typedArrayTags[dataViewTag]=typedArrayTags[dateTag]=typedArrayTags[errorTag]=typedArrayTags[funcTag]=typedArrayTags[mapTag]=typedArrayTags[numberTag]=typedArrayTags[objectTag]=typedArrayTags[regexpTag]=typedArrayTags[setTag]=typedArrayTags[stringTag]=typedArrayTags[weakMapTag]=false;
/**
 * The base implementation of `_.isTypedArray` without Node.js optimizations.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a typed array, else `false`.
 */function baseIsTypedArray(value){return isObjectLike(value)&&isLength(value.length)&&!!typedArrayTags[baseGetTag(value)]}module.exports=baseIsTypedArray},{"./_baseGetTag":151,"./isLength":309,"./isObjectLike":312}],165:[function(require,module,exports){var baseMatches=require("./_baseMatches"),baseMatchesProperty=require("./_baseMatchesProperty"),identity=require("./identity"),isArray=require("./isArray"),property=require("./property");
/**
 * The base implementation of `_.iteratee`.
 *
 * @private
 * @param {*} [value=_.identity] The value to convert to an iteratee.
 * @returns {Function} Returns the iteratee.
 */function baseIteratee(value){
// Don't store the `typeof` result in a variable to avoid a JIT bug in Safari 9.
// See https://bugs.webkit.org/show_bug.cgi?id=156034 for more details.
if(typeof value=="function"){return value}if(value==null){return identity}if(typeof value=="object"){return isArray(value)?baseMatchesProperty(value[0],value[1]):baseMatches(value)}return property(value)}module.exports=baseIteratee},{"./_baseMatches":170,"./_baseMatchesProperty":171,"./identity":301,"./isArray":303,"./property":332}],166:[function(require,module,exports){var isPrototype=require("./_isPrototype"),nativeKeys=require("./_nativeKeys");
/** Used for built-in method references. */var objectProto=Object.prototype;
/** Used to check objects for own properties. */var hasOwnProperty=objectProto.hasOwnProperty;
/**
 * The base implementation of `_.keys` which doesn't treat sparse arrays as dense.
 *
 * @private
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of property names.
 */function baseKeys(object){if(!isPrototype(object)){return nativeKeys(object)}var result=[];for(var key in Object(object)){if(hasOwnProperty.call(object,key)&&key!="constructor"){result.push(key)}}return result}module.exports=baseKeys},{"./_isPrototype":246,"./_nativeKeys":262}],167:[function(require,module,exports){var isObject=require("./isObject"),isPrototype=require("./_isPrototype"),nativeKeysIn=require("./_nativeKeysIn");
/** Used for built-in method references. */var objectProto=Object.prototype;
/** Used to check objects for own properties. */var hasOwnProperty=objectProto.hasOwnProperty;
/**
 * The base implementation of `_.keysIn` which doesn't treat sparse arrays as dense.
 *
 * @private
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of property names.
 */function baseKeysIn(object){if(!isObject(object)){return nativeKeysIn(object)}var isProto=isPrototype(object),result=[];for(var key in object){if(!(key=="constructor"&&(isProto||!hasOwnProperty.call(object,key)))){result.push(key)}}return result}module.exports=baseKeysIn},{"./_isPrototype":246,"./_nativeKeysIn":263,"./isObject":311}],168:[function(require,module,exports){
/**
 * The base implementation of `_.lt` which doesn't coerce arguments.
 *
 * @private
 * @param {*} value The value to compare.
 * @param {*} other The other value to compare.
 * @returns {boolean} Returns `true` if `value` is less than `other`,
 *  else `false`.
 */
function baseLt(value,other){return value<other}module.exports=baseLt},{}],169:[function(require,module,exports){var baseEach=require("./_baseEach"),isArrayLike=require("./isArrayLike");
/**
 * The base implementation of `_.map` without support for iteratee shorthands.
 *
 * @private
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @returns {Array} Returns the new mapped array.
 */function baseMap(collection,iteratee){var index=-1,result=isArrayLike(collection)?Array(collection.length):[];baseEach(collection,function(value,key,collection){result[++index]=iteratee(value,key,collection)});return result}module.exports=baseMap},{"./_baseEach":142,"./isArrayLike":304}],170:[function(require,module,exports){var baseIsMatch=require("./_baseIsMatch"),getMatchData=require("./_getMatchData"),matchesStrictComparable=require("./_matchesStrictComparable");
/**
 * The base implementation of `_.matches` which doesn't clone `source`.
 *
 * @private
 * @param {Object} source The object of property values to match.
 * @returns {Function} Returns the new spec function.
 */function baseMatches(source){var matchData=getMatchData(source);if(matchData.length==1&&matchData[0][2]){return matchesStrictComparable(matchData[0][0],matchData[0][1])}return function(object){return object===source||baseIsMatch(object,source,matchData)}}module.exports=baseMatches},{"./_baseIsMatch":160,"./_getMatchData":222,"./_matchesStrictComparable":259}],171:[function(require,module,exports){var baseIsEqual=require("./_baseIsEqual"),get=require("./get"),hasIn=require("./hasIn"),isKey=require("./_isKey"),isStrictComparable=require("./_isStrictComparable"),matchesStrictComparable=require("./_matchesStrictComparable"),toKey=require("./_toKey");
/** Used to compose bitmasks for value comparisons. */var COMPARE_PARTIAL_FLAG=1,COMPARE_UNORDERED_FLAG=2;
/**
 * The base implementation of `_.matchesProperty` which doesn't clone `srcValue`.
 *
 * @private
 * @param {string} path The path of the property to get.
 * @param {*} srcValue The value to match.
 * @returns {Function} Returns the new spec function.
 */function baseMatchesProperty(path,srcValue){if(isKey(path)&&isStrictComparable(srcValue)){return matchesStrictComparable(toKey(path),srcValue)}return function(object){var objValue=get(object,path);return objValue===undefined&&objValue===srcValue?hasIn(object,path):baseIsEqual(srcValue,objValue,COMPARE_PARTIAL_FLAG|COMPARE_UNORDERED_FLAG)}}module.exports=baseMatchesProperty},{"./_baseIsEqual":157,"./_isKey":243,"./_isStrictComparable":247,"./_matchesStrictComparable":259,"./_toKey":283,"./get":298,"./hasIn":300}],172:[function(require,module,exports){var Stack=require("./_Stack"),assignMergeValue=require("./_assignMergeValue"),baseFor=require("./_baseFor"),baseMergeDeep=require("./_baseMergeDeep"),isObject=require("./isObject"),keysIn=require("./keysIn"),safeGet=require("./_safeGet");
/**
 * The base implementation of `_.merge` without support for multiple sources.
 *
 * @private
 * @param {Object} object The destination object.
 * @param {Object} source The source object.
 * @param {number} srcIndex The index of `source`.
 * @param {Function} [customizer] The function to customize merged values.
 * @param {Object} [stack] Tracks traversed source values and their merged
 *  counterparts.
 */function baseMerge(object,source,srcIndex,customizer,stack){if(object===source){return}baseFor(source,function(srcValue,key){stack||(stack=new Stack);if(isObject(srcValue)){baseMergeDeep(object,source,key,srcIndex,baseMerge,customizer,stack)}else{var newValue=customizer?customizer(safeGet(object,key),srcValue,key+"",object,source,stack):undefined;if(newValue===undefined){newValue=srcValue}assignMergeValue(object,key,newValue)}},keysIn)}module.exports=baseMerge},{"./_Stack":119,"./_assignMergeValue":134,"./_baseFor":147,"./_baseMergeDeep":173,"./_safeGet":269,"./isObject":311,"./keysIn":320}],173:[function(require,module,exports){var assignMergeValue=require("./_assignMergeValue"),cloneBuffer=require("./_cloneBuffer"),cloneTypedArray=require("./_cloneTypedArray"),copyArray=require("./_copyArray"),initCloneObject=require("./_initCloneObject"),isArguments=require("./isArguments"),isArray=require("./isArray"),isArrayLikeObject=require("./isArrayLikeObject"),isBuffer=require("./isBuffer"),isFunction=require("./isFunction"),isObject=require("./isObject"),isPlainObject=require("./isPlainObject"),isTypedArray=require("./isTypedArray"),safeGet=require("./_safeGet"),toPlainObject=require("./toPlainObject");
/**
 * A specialized version of `baseMerge` for arrays and objects which performs
 * deep merges and tracks traversed objects enabling objects with circular
 * references to be merged.
 *
 * @private
 * @param {Object} object The destination object.
 * @param {Object} source The source object.
 * @param {string} key The key of the value to merge.
 * @param {number} srcIndex The index of `source`.
 * @param {Function} mergeFunc The function to merge values.
 * @param {Function} [customizer] The function to customize assigned values.
 * @param {Object} [stack] Tracks traversed source values and their merged
 *  counterparts.
 */function baseMergeDeep(object,source,key,srcIndex,mergeFunc,customizer,stack){var objValue=safeGet(object,key),srcValue=safeGet(source,key),stacked=stack.get(srcValue);if(stacked){assignMergeValue(object,key,stacked);return}var newValue=customizer?customizer(objValue,srcValue,key+"",object,source,stack):undefined;var isCommon=newValue===undefined;if(isCommon){var isArr=isArray(srcValue),isBuff=!isArr&&isBuffer(srcValue),isTyped=!isArr&&!isBuff&&isTypedArray(srcValue);newValue=srcValue;if(isArr||isBuff||isTyped){if(isArray(objValue)){newValue=objValue}else if(isArrayLikeObject(objValue)){newValue=copyArray(objValue)}else if(isBuff){isCommon=false;newValue=cloneBuffer(srcValue,true)}else if(isTyped){isCommon=false;newValue=cloneTypedArray(srcValue,true)}else{newValue=[]}}else if(isPlainObject(srcValue)||isArguments(srcValue)){newValue=objValue;if(isArguments(objValue)){newValue=toPlainObject(objValue)}else if(!isObject(objValue)||isFunction(objValue)){newValue=initCloneObject(srcValue)}}else{isCommon=false}}if(isCommon){
// Recursively merge objects and arrays (susceptible to call stack limits).
stack.set(srcValue,newValue);mergeFunc(newValue,srcValue,srcIndex,customizer,stack);stack["delete"](srcValue)}assignMergeValue(object,key,newValue)}module.exports=baseMergeDeep},{"./_assignMergeValue":134,"./_cloneBuffer":195,"./_cloneTypedArray":199,"./_copyArray":202,"./_initCloneObject":239,"./_safeGet":269,"./isArguments":302,"./isArray":303,"./isArrayLikeObject":305,"./isBuffer":306,"./isFunction":308,"./isObject":311,"./isPlainObject":313,"./isTypedArray":317,"./toPlainObject":342}],174:[function(require,module,exports){var arrayMap=require("./_arrayMap"),baseIteratee=require("./_baseIteratee"),baseMap=require("./_baseMap"),baseSortBy=require("./_baseSortBy"),baseUnary=require("./_baseUnary"),compareMultiple=require("./_compareMultiple"),identity=require("./identity");
/**
 * The base implementation of `_.orderBy` without param guards.
 *
 * @private
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function[]|Object[]|string[]} iteratees The iteratees to sort by.
 * @param {string[]} orders The sort orders of `iteratees`.
 * @returns {Array} Returns the new sorted array.
 */function baseOrderBy(collection,iteratees,orders){var index=-1;iteratees=arrayMap(iteratees.length?iteratees:[identity],baseUnary(baseIteratee));var result=baseMap(collection,function(value,key,collection){var criteria=arrayMap(iteratees,function(iteratee){return iteratee(value)});return{criteria:criteria,index:++index,value:value}});return baseSortBy(result,function(object,other){return compareMultiple(object,other,orders)})}module.exports=baseOrderBy},{"./_arrayMap":129,"./_baseIteratee":165,"./_baseMap":169,"./_baseSortBy":184,"./_baseUnary":187,"./_compareMultiple":201,"./identity":301}],175:[function(require,module,exports){var basePickBy=require("./_basePickBy"),hasIn=require("./hasIn");
/**
 * The base implementation of `_.pick` without support for individual
 * property identifiers.
 *
 * @private
 * @param {Object} object The source object.
 * @param {string[]} paths The property paths to pick.
 * @returns {Object} Returns the new object.
 */function basePick(object,paths){return basePickBy(object,paths,function(value,path){return hasIn(object,path)})}module.exports=basePick},{"./_basePickBy":176,"./hasIn":300}],176:[function(require,module,exports){var baseGet=require("./_baseGet"),baseSet=require("./_baseSet"),castPath=require("./_castPath");
/**
 * The base implementation of  `_.pickBy` without support for iteratee shorthands.
 *
 * @private
 * @param {Object} object The source object.
 * @param {string[]} paths The property paths to pick.
 * @param {Function} predicate The function invoked per property.
 * @returns {Object} Returns the new object.
 */function basePickBy(object,paths,predicate){var index=-1,length=paths.length,result={};while(++index<length){var path=paths[index],value=baseGet(object,path);if(predicate(value,path)){baseSet(result,castPath(path,object),value)}}return result}module.exports=basePickBy},{"./_baseGet":149,"./_baseSet":182,"./_castPath":193}],177:[function(require,module,exports){
/**
 * The base implementation of `_.property` without support for deep paths.
 *
 * @private
 * @param {string} key The key of the property to get.
 * @returns {Function} Returns the new accessor function.
 */
function baseProperty(key){return function(object){return object==null?undefined:object[key]}}module.exports=baseProperty},{}],178:[function(require,module,exports){var baseGet=require("./_baseGet");
/**
 * A specialized version of `baseProperty` which supports deep paths.
 *
 * @private
 * @param {Array|string} path The path of the property to get.
 * @returns {Function} Returns the new accessor function.
 */function basePropertyDeep(path){return function(object){return baseGet(object,path)}}module.exports=basePropertyDeep},{"./_baseGet":149}],179:[function(require,module,exports){
/* Built-in method references for those with the same name as other `lodash` methods. */
var nativeCeil=Math.ceil,nativeMax=Math.max;
/**
 * The base implementation of `_.range` and `_.rangeRight` which doesn't
 * coerce arguments.
 *
 * @private
 * @param {number} start The start of the range.
 * @param {number} end The end of the range.
 * @param {number} step The value to increment or decrement by.
 * @param {boolean} [fromRight] Specify iterating from right to left.
 * @returns {Array} Returns the range of numbers.
 */function baseRange(start,end,step,fromRight){var index=-1,length=nativeMax(nativeCeil((end-start)/(step||1)),0),result=Array(length);while(length--){result[fromRight?length:++index]=start;start+=step}return result}module.exports=baseRange},{}],180:[function(require,module,exports){
/**
 * The base implementation of `_.reduce` and `_.reduceRight`, without support
 * for iteratee shorthands, which iterates over `collection` using `eachFunc`.
 *
 * @private
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @param {*} accumulator The initial value.
 * @param {boolean} initAccum Specify using the first or last element of
 *  `collection` as the initial value.
 * @param {Function} eachFunc The function to iterate over `collection`.
 * @returns {*} Returns the accumulated value.
 */
function baseReduce(collection,iteratee,accumulator,initAccum,eachFunc){eachFunc(collection,function(value,index,collection){accumulator=initAccum?(initAccum=false,value):iteratee(accumulator,value,index,collection)});return accumulator}module.exports=baseReduce},{}],181:[function(require,module,exports){var identity=require("./identity"),overRest=require("./_overRest"),setToString=require("./_setToString");
/**
 * The base implementation of `_.rest` which doesn't validate or coerce arguments.
 *
 * @private
 * @param {Function} func The function to apply a rest parameter to.
 * @param {number} [start=func.length-1] The start position of the rest parameter.
 * @returns {Function} Returns the new function.
 */function baseRest(func,start){return setToString(overRest(func,start,identity),func+"")}module.exports=baseRest},{"./_overRest":267,"./_setToString":273,"./identity":301}],182:[function(require,module,exports){var assignValue=require("./_assignValue"),castPath=require("./_castPath"),isIndex=require("./_isIndex"),isObject=require("./isObject"),toKey=require("./_toKey");
/**
 * The base implementation of `_.set`.
 *
 * @private
 * @param {Object} object The object to modify.
 * @param {Array|string} path The path of the property to set.
 * @param {*} value The value to set.
 * @param {Function} [customizer] The function to customize path creation.
 * @returns {Object} Returns `object`.
 */function baseSet(object,path,value,customizer){if(!isObject(object)){return object}path=castPath(path,object);var index=-1,length=path.length,lastIndex=length-1,nested=object;while(nested!=null&&++index<length){var key=toKey(path[index]),newValue=value;if(index!=lastIndex){var objValue=nested[key];newValue=customizer?customizer(objValue,key,nested):undefined;if(newValue===undefined){newValue=isObject(objValue)?objValue:isIndex(path[index+1])?[]:{}}}assignValue(nested,key,newValue);nested=nested[key]}return object}module.exports=baseSet},{"./_assignValue":135,"./_castPath":193,"./_isIndex":241,"./_toKey":283,"./isObject":311}],183:[function(require,module,exports){var constant=require("./constant"),defineProperty=require("./_defineProperty"),identity=require("./identity");
/**
 * The base implementation of `setToString` without support for hot loop shorting.
 *
 * @private
 * @param {Function} func The function to modify.
 * @param {Function} string The `toString` result.
 * @returns {Function} Returns `func`.
 */var baseSetToString=!defineProperty?identity:function(func,string){return defineProperty(func,"toString",{configurable:true,enumerable:false,value:constant(string),writable:true})};module.exports=baseSetToString},{"./_defineProperty":213,"./constant":288,"./identity":301}],184:[function(require,module,exports){
/**
 * The base implementation of `_.sortBy` which uses `comparer` to define the
 * sort order of `array` and replaces criteria objects with their corresponding
 * values.
 *
 * @private
 * @param {Array} array The array to sort.
 * @param {Function} comparer The function to define sort order.
 * @returns {Array} Returns `array`.
 */
function baseSortBy(array,comparer){var length=array.length;array.sort(comparer);while(length--){array[length]=array[length].value}return array}module.exports=baseSortBy},{}],185:[function(require,module,exports){
/**
 * The base implementation of `_.times` without support for iteratee shorthands
 * or max array length checks.
 *
 * @private
 * @param {number} n The number of times to invoke `iteratee`.
 * @param {Function} iteratee The function invoked per iteration.
 * @returns {Array} Returns the array of results.
 */
function baseTimes(n,iteratee){var index=-1,result=Array(n);while(++index<n){result[index]=iteratee(index)}return result}module.exports=baseTimes},{}],186:[function(require,module,exports){var Symbol=require("./_Symbol"),arrayMap=require("./_arrayMap"),isArray=require("./isArray"),isSymbol=require("./isSymbol");
/** Used as references for various `Number` constants. */var INFINITY=1/0;
/** Used to convert symbols to primitives and strings. */var symbolProto=Symbol?Symbol.prototype:undefined,symbolToString=symbolProto?symbolProto.toString:undefined;
/**
 * The base implementation of `_.toString` which doesn't convert nullish
 * values to empty strings.
 *
 * @private
 * @param {*} value The value to process.
 * @returns {string} Returns the string.
 */function baseToString(value){
// Exit early for strings to avoid a performance hit in some environments.
if(typeof value=="string"){return value}if(isArray(value)){
// Recursively convert values (susceptible to call stack limits).
return arrayMap(value,baseToString)+""}if(isSymbol(value)){return symbolToString?symbolToString.call(value):""}var result=value+"";return result=="0"&&1/value==-INFINITY?"-0":result}module.exports=baseToString},{"./_Symbol":120,"./_arrayMap":129,"./isArray":303,"./isSymbol":316}],187:[function(require,module,exports){
/**
 * The base implementation of `_.unary` without support for storing metadata.
 *
 * @private
 * @param {Function} func The function to cap arguments for.
 * @returns {Function} Returns the new capped function.
 */
function baseUnary(func){return function(value){return func(value)}}module.exports=baseUnary},{}],188:[function(require,module,exports){var SetCache=require("./_SetCache"),arrayIncludes=require("./_arrayIncludes"),arrayIncludesWith=require("./_arrayIncludesWith"),cacheHas=require("./_cacheHas"),createSet=require("./_createSet"),setToArray=require("./_setToArray");
/** Used as the size to enable large array optimizations. */var LARGE_ARRAY_SIZE=200;
/**
 * The base implementation of `_.uniqBy` without support for iteratee shorthands.
 *
 * @private
 * @param {Array} array The array to inspect.
 * @param {Function} [iteratee] The iteratee invoked per element.
 * @param {Function} [comparator] The comparator invoked per element.
 * @returns {Array} Returns the new duplicate free array.
 */function baseUniq(array,iteratee,comparator){var index=-1,includes=arrayIncludes,length=array.length,isCommon=true,result=[],seen=result;if(comparator){isCommon=false;includes=arrayIncludesWith}else if(length>=LARGE_ARRAY_SIZE){var set=iteratee?null:createSet(array);if(set){return setToArray(set)}isCommon=false;includes=cacheHas;seen=new SetCache}else{seen=iteratee?[]:result}outer:while(++index<length){var value=array[index],computed=iteratee?iteratee(value):value;value=comparator||value!==0?value:0;if(isCommon&&computed===computed){var seenIndex=seen.length;while(seenIndex--){if(seen[seenIndex]===computed){continue outer}}if(iteratee){seen.push(computed)}result.push(value)}else if(!includes(seen,computed,comparator)){if(seen!==result){seen.push(computed)}result.push(value)}}return result}module.exports=baseUniq},{"./_SetCache":118,"./_arrayIncludes":126,"./_arrayIncludesWith":127,"./_cacheHas":191,"./_createSet":212,"./_setToArray":272}],189:[function(require,module,exports){var arrayMap=require("./_arrayMap");
/**
 * The base implementation of `_.values` and `_.valuesIn` which creates an
 * array of `object` property values corresponding to the property names
 * of `props`.
 *
 * @private
 * @param {Object} object The object to query.
 * @param {Array} props The property names to get values for.
 * @returns {Object} Returns the array of property values.
 */function baseValues(object,props){return arrayMap(props,function(key){return object[key]})}module.exports=baseValues},{"./_arrayMap":129}],190:[function(require,module,exports){
/**
 * This base implementation of `_.zipObject` which assigns values using `assignFunc`.
 *
 * @private
 * @param {Array} props The property identifiers.
 * @param {Array} values The property values.
 * @param {Function} assignFunc The function to assign values.
 * @returns {Object} Returns the new object.
 */
function baseZipObject(props,values,assignFunc){var index=-1,length=props.length,valsLength=values.length,result={};while(++index<length){var value=index<valsLength?values[index]:undefined;assignFunc(result,props[index],value)}return result}module.exports=baseZipObject},{}],191:[function(require,module,exports){
/**
 * Checks if a `cache` value for `key` exists.
 *
 * @private
 * @param {Object} cache The cache to query.
 * @param {string} key The key of the entry to check.
 * @returns {boolean} Returns `true` if an entry for `key` exists, else `false`.
 */
function cacheHas(cache,key){return cache.has(key)}module.exports=cacheHas},{}],192:[function(require,module,exports){var identity=require("./identity");
/**
 * Casts `value` to `identity` if it's not a function.
 *
 * @private
 * @param {*} value The value to inspect.
 * @returns {Function} Returns cast function.
 */function castFunction(value){return typeof value=="function"?value:identity}module.exports=castFunction},{"./identity":301}],193:[function(require,module,exports){var isArray=require("./isArray"),isKey=require("./_isKey"),stringToPath=require("./_stringToPath"),toString=require("./toString");
/**
 * Casts `value` to a path array if it's not one.
 *
 * @private
 * @param {*} value The value to inspect.
 * @param {Object} [object] The object to query keys on.
 * @returns {Array} Returns the cast property path array.
 */function castPath(value,object){if(isArray(value)){return value}return isKey(value,object)?[value]:stringToPath(toString(value))}module.exports=castPath},{"./_isKey":243,"./_stringToPath":282,"./isArray":303,"./toString":343}],194:[function(require,module,exports){var Uint8Array=require("./_Uint8Array");
/**
 * Creates a clone of `arrayBuffer`.
 *
 * @private
 * @param {ArrayBuffer} arrayBuffer The array buffer to clone.
 * @returns {ArrayBuffer} Returns the cloned array buffer.
 */function cloneArrayBuffer(arrayBuffer){var result=new arrayBuffer.constructor(arrayBuffer.byteLength);new Uint8Array(result).set(new Uint8Array(arrayBuffer));return result}module.exports=cloneArrayBuffer},{"./_Uint8Array":121}],195:[function(require,module,exports){var root=require("./_root");
/** Detect free variable `exports`. */var freeExports=typeof exports=="object"&&exports&&!exports.nodeType&&exports;
/** Detect free variable `module`. */var freeModule=freeExports&&typeof module=="object"&&module&&!module.nodeType&&module;
/** Detect the popular CommonJS extension `module.exports`. */var moduleExports=freeModule&&freeModule.exports===freeExports;
/** Built-in value references. */var Buffer=moduleExports?root.Buffer:undefined,allocUnsafe=Buffer?Buffer.allocUnsafe:undefined;
/**
 * Creates a clone of  `buffer`.
 *
 * @private
 * @param {Buffer} buffer The buffer to clone.
 * @param {boolean} [isDeep] Specify a deep clone.
 * @returns {Buffer} Returns the cloned buffer.
 */function cloneBuffer(buffer,isDeep){if(isDeep){return buffer.slice()}var length=buffer.length,result=allocUnsafe?allocUnsafe(length):new buffer.constructor(length);buffer.copy(result);return result}module.exports=cloneBuffer},{"./_root":268}],196:[function(require,module,exports){var cloneArrayBuffer=require("./_cloneArrayBuffer");
/**
 * Creates a clone of `dataView`.
 *
 * @private
 * @param {Object} dataView The data view to clone.
 * @param {boolean} [isDeep] Specify a deep clone.
 * @returns {Object} Returns the cloned data view.
 */function cloneDataView(dataView,isDeep){var buffer=isDeep?cloneArrayBuffer(dataView.buffer):dataView.buffer;return new dataView.constructor(buffer,dataView.byteOffset,dataView.byteLength)}module.exports=cloneDataView},{"./_cloneArrayBuffer":194}],197:[function(require,module,exports){
/** Used to match `RegExp` flags from their coerced string values. */
var reFlags=/\w*$/;
/**
 * Creates a clone of `regexp`.
 *
 * @private
 * @param {Object} regexp The regexp to clone.
 * @returns {Object} Returns the cloned regexp.
 */function cloneRegExp(regexp){var result=new regexp.constructor(regexp.source,reFlags.exec(regexp));result.lastIndex=regexp.lastIndex;return result}module.exports=cloneRegExp},{}],198:[function(require,module,exports){var Symbol=require("./_Symbol");
/** Used to convert symbols to primitives and strings. */var symbolProto=Symbol?Symbol.prototype:undefined,symbolValueOf=symbolProto?symbolProto.valueOf:undefined;
/**
 * Creates a clone of the `symbol` object.
 *
 * @private
 * @param {Object} symbol The symbol object to clone.
 * @returns {Object} Returns the cloned symbol object.
 */function cloneSymbol(symbol){return symbolValueOf?Object(symbolValueOf.call(symbol)):{}}module.exports=cloneSymbol},{"./_Symbol":120}],199:[function(require,module,exports){var cloneArrayBuffer=require("./_cloneArrayBuffer");
/**
 * Creates a clone of `typedArray`.
 *
 * @private
 * @param {Object} typedArray The typed array to clone.
 * @param {boolean} [isDeep] Specify a deep clone.
 * @returns {Object} Returns the cloned typed array.
 */function cloneTypedArray(typedArray,isDeep){var buffer=isDeep?cloneArrayBuffer(typedArray.buffer):typedArray.buffer;return new typedArray.constructor(buffer,typedArray.byteOffset,typedArray.length)}module.exports=cloneTypedArray},{"./_cloneArrayBuffer":194}],200:[function(require,module,exports){var isSymbol=require("./isSymbol");
/**
 * Compares values to sort them in ascending order.
 *
 * @private
 * @param {*} value The value to compare.
 * @param {*} other The other value to compare.
 * @returns {number} Returns the sort order indicator for `value`.
 */function compareAscending(value,other){if(value!==other){var valIsDefined=value!==undefined,valIsNull=value===null,valIsReflexive=value===value,valIsSymbol=isSymbol(value);var othIsDefined=other!==undefined,othIsNull=other===null,othIsReflexive=other===other,othIsSymbol=isSymbol(other);if(!othIsNull&&!othIsSymbol&&!valIsSymbol&&value>other||valIsSymbol&&othIsDefined&&othIsReflexive&&!othIsNull&&!othIsSymbol||valIsNull&&othIsDefined&&othIsReflexive||!valIsDefined&&othIsReflexive||!valIsReflexive){return 1}if(!valIsNull&&!valIsSymbol&&!othIsSymbol&&value<other||othIsSymbol&&valIsDefined&&valIsReflexive&&!valIsNull&&!valIsSymbol||othIsNull&&valIsDefined&&valIsReflexive||!othIsDefined&&valIsReflexive||!othIsReflexive){return-1}}return 0}module.exports=compareAscending},{"./isSymbol":316}],201:[function(require,module,exports){var compareAscending=require("./_compareAscending");
/**
 * Used by `_.orderBy` to compare multiple properties of a value to another
 * and stable sort them.
 *
 * If `orders` is unspecified, all values are sorted in ascending order. Otherwise,
 * specify an order of "desc" for descending or "asc" for ascending sort order
 * of corresponding values.
 *
 * @private
 * @param {Object} object The object to compare.
 * @param {Object} other The other object to compare.
 * @param {boolean[]|string[]} orders The order to sort by for each property.
 * @returns {number} Returns the sort order indicator for `object`.
 */function compareMultiple(object,other,orders){var index=-1,objCriteria=object.criteria,othCriteria=other.criteria,length=objCriteria.length,ordersLength=orders.length;while(++index<length){var result=compareAscending(objCriteria[index],othCriteria[index]);if(result){if(index>=ordersLength){return result}var order=orders[index];return result*(order=="desc"?-1:1)}}
// Fixes an `Array#sort` bug in the JS engine embedded in Adobe applications
// that causes it, under certain circumstances, to provide the same value for
// `object` and `other`. See https://github.com/jashkenas/underscore/pull/1247
// for more details.
//
// This also ensures a stable sort in V8 and other engines.
// See https://bugs.chromium.org/p/v8/issues/detail?id=90 for more details.
return object.index-other.index}module.exports=compareMultiple},{"./_compareAscending":200}],202:[function(require,module,exports){
/**
 * Copies the values of `source` to `array`.
 *
 * @private
 * @param {Array} source The array to copy values from.
 * @param {Array} [array=[]] The array to copy values to.
 * @returns {Array} Returns `array`.
 */
function copyArray(source,array){var index=-1,length=source.length;array||(array=Array(length));while(++index<length){array[index]=source[index]}return array}module.exports=copyArray},{}],203:[function(require,module,exports){var assignValue=require("./_assignValue"),baseAssignValue=require("./_baseAssignValue");
/**
 * Copies properties of `source` to `object`.
 *
 * @private
 * @param {Object} source The object to copy properties from.
 * @param {Array} props The property identifiers to copy.
 * @param {Object} [object={}] The object to copy properties to.
 * @param {Function} [customizer] The function to customize copied values.
 * @returns {Object} Returns `object`.
 */function copyObject(source,props,object,customizer){var isNew=!object;object||(object={});var index=-1,length=props.length;while(++index<length){var key=props[index];var newValue=customizer?customizer(object[key],source[key],key,object,source):undefined;if(newValue===undefined){newValue=source[key]}if(isNew){baseAssignValue(object,key,newValue)}else{assignValue(object,key,newValue)}}return object}module.exports=copyObject},{"./_assignValue":135,"./_baseAssignValue":139}],204:[function(require,module,exports){var copyObject=require("./_copyObject"),getSymbols=require("./_getSymbols");
/**
 * Copies own symbols of `source` to `object`.
 *
 * @private
 * @param {Object} source The object to copy symbols from.
 * @param {Object} [object={}] The object to copy symbols to.
 * @returns {Object} Returns `object`.
 */function copySymbols(source,object){return copyObject(source,getSymbols(source),object)}module.exports=copySymbols},{"./_copyObject":203,"./_getSymbols":226}],205:[function(require,module,exports){var copyObject=require("./_copyObject"),getSymbolsIn=require("./_getSymbolsIn");
/**
 * Copies own and inherited symbols of `source` to `object`.
 *
 * @private
 * @param {Object} source The object to copy symbols from.
 * @param {Object} [object={}] The object to copy symbols to.
 * @returns {Object} Returns `object`.
 */function copySymbolsIn(source,object){return copyObject(source,getSymbolsIn(source),object)}module.exports=copySymbolsIn},{"./_copyObject":203,"./_getSymbolsIn":227}],206:[function(require,module,exports){var root=require("./_root");
/** Used to detect overreaching core-js shims. */var coreJsData=root["__core-js_shared__"];module.exports=coreJsData},{"./_root":268}],207:[function(require,module,exports){var baseRest=require("./_baseRest"),isIterateeCall=require("./_isIterateeCall");
/**
 * Creates a function like `_.assign`.
 *
 * @private
 * @param {Function} assigner The function to assign values.
 * @returns {Function} Returns the new assigner function.
 */function createAssigner(assigner){return baseRest(function(object,sources){var index=-1,length=sources.length,customizer=length>1?sources[length-1]:undefined,guard=length>2?sources[2]:undefined;customizer=assigner.length>3&&typeof customizer=="function"?(length--,customizer):undefined;if(guard&&isIterateeCall(sources[0],sources[1],guard)){customizer=length<3?undefined:customizer;length=1}object=Object(object);while(++index<length){var source=sources[index];if(source){assigner(object,source,index,customizer)}}return object})}module.exports=createAssigner},{"./_baseRest":181,"./_isIterateeCall":242}],208:[function(require,module,exports){var isArrayLike=require("./isArrayLike");
/**
 * Creates a `baseEach` or `baseEachRight` function.
 *
 * @private
 * @param {Function} eachFunc The function to iterate over a collection.
 * @param {boolean} [fromRight] Specify iterating from right to left.
 * @returns {Function} Returns the new base function.
 */function createBaseEach(eachFunc,fromRight){return function(collection,iteratee){if(collection==null){return collection}if(!isArrayLike(collection)){return eachFunc(collection,iteratee)}var length=collection.length,index=fromRight?length:-1,iterable=Object(collection);while(fromRight?index--:++index<length){if(iteratee(iterable[index],index,iterable)===false){break}}return collection}}module.exports=createBaseEach},{"./isArrayLike":304}],209:[function(require,module,exports){
/**
 * Creates a base function for methods like `_.forIn` and `_.forOwn`.
 *
 * @private
 * @param {boolean} [fromRight] Specify iterating from right to left.
 * @returns {Function} Returns the new base function.
 */
function createBaseFor(fromRight){return function(object,iteratee,keysFunc){var index=-1,iterable=Object(object),props=keysFunc(object),length=props.length;while(length--){var key=props[fromRight?length:++index];if(iteratee(iterable[key],key,iterable)===false){break}}return object}}module.exports=createBaseFor},{}],210:[function(require,module,exports){var baseIteratee=require("./_baseIteratee"),isArrayLike=require("./isArrayLike"),keys=require("./keys");
/**
 * Creates a `_.find` or `_.findLast` function.
 *
 * @private
 * @param {Function} findIndexFunc The function to find the collection index.
 * @returns {Function} Returns the new find function.
 */function createFind(findIndexFunc){return function(collection,predicate,fromIndex){var iterable=Object(collection);if(!isArrayLike(collection)){var iteratee=baseIteratee(predicate,3);collection=keys(collection);predicate=function(key){return iteratee(iterable[key],key,iterable)}}var index=findIndexFunc(collection,predicate,fromIndex);return index>-1?iterable[iteratee?collection[index]:index]:undefined}}module.exports=createFind},{"./_baseIteratee":165,"./isArrayLike":304,"./keys":319}],211:[function(require,module,exports){var baseRange=require("./_baseRange"),isIterateeCall=require("./_isIterateeCall"),toFinite=require("./toFinite");
/**
 * Creates a `_.range` or `_.rangeRight` function.
 *
 * @private
 * @param {boolean} [fromRight] Specify iterating from right to left.
 * @returns {Function} Returns the new range function.
 */function createRange(fromRight){return function(start,end,step){if(step&&typeof step!="number"&&isIterateeCall(start,end,step)){end=step=undefined}
// Ensure the sign of `-0` is preserved.
start=toFinite(start);if(end===undefined){end=start;start=0}else{end=toFinite(end)}step=step===undefined?start<end?1:-1:toFinite(step);return baseRange(start,end,step,fromRight)}}module.exports=createRange},{"./_baseRange":179,"./_isIterateeCall":242,"./toFinite":339}],212:[function(require,module,exports){var Set=require("./_Set"),noop=require("./noop"),setToArray=require("./_setToArray");
/** Used as references for various `Number` constants. */var INFINITY=1/0;
/**
 * Creates a set object of `values`.
 *
 * @private
 * @param {Array} values The values to add to the set.
 * @returns {Object} Returns the new set.
 */var createSet=!(Set&&1/setToArray(new Set([,-0]))[1]==INFINITY)?noop:function(values){return new Set(values)};module.exports=createSet},{"./_Set":117,"./_setToArray":272,"./noop":329}],213:[function(require,module,exports){var getNative=require("./_getNative");var defineProperty=function(){try{var func=getNative(Object,"defineProperty");func({},"",{});return func}catch(e){}}();module.exports=defineProperty},{"./_getNative":223}],214:[function(require,module,exports){var SetCache=require("./_SetCache"),arraySome=require("./_arraySome"),cacheHas=require("./_cacheHas");
/** Used to compose bitmasks for value comparisons. */var COMPARE_PARTIAL_FLAG=1,COMPARE_UNORDERED_FLAG=2;
/**
 * A specialized version of `baseIsEqualDeep` for arrays with support for
 * partial deep comparisons.
 *
 * @private
 * @param {Array} array The array to compare.
 * @param {Array} other The other array to compare.
 * @param {number} bitmask The bitmask flags. See `baseIsEqual` for more details.
 * @param {Function} customizer The function to customize comparisons.
 * @param {Function} equalFunc The function to determine equivalents of values.
 * @param {Object} stack Tracks traversed `array` and `other` objects.
 * @returns {boolean} Returns `true` if the arrays are equivalent, else `false`.
 */function equalArrays(array,other,bitmask,customizer,equalFunc,stack){var isPartial=bitmask&COMPARE_PARTIAL_FLAG,arrLength=array.length,othLength=other.length;if(arrLength!=othLength&&!(isPartial&&othLength>arrLength)){return false}
// Assume cyclic values are equal.
var stacked=stack.get(array);if(stacked&&stack.get(other)){return stacked==other}var index=-1,result=true,seen=bitmask&COMPARE_UNORDERED_FLAG?new SetCache:undefined;stack.set(array,other);stack.set(other,array);
// Ignore non-index properties.
while(++index<arrLength){var arrValue=array[index],othValue=other[index];if(customizer){var compared=isPartial?customizer(othValue,arrValue,index,other,array,stack):customizer(arrValue,othValue,index,array,other,stack)}if(compared!==undefined){if(compared){continue}result=false;break}
// Recursively compare arrays (susceptible to call stack limits).
if(seen){if(!arraySome(other,function(othValue,othIndex){if(!cacheHas(seen,othIndex)&&(arrValue===othValue||equalFunc(arrValue,othValue,bitmask,customizer,stack))){return seen.push(othIndex)}})){result=false;break}}else if(!(arrValue===othValue||equalFunc(arrValue,othValue,bitmask,customizer,stack))){result=false;break}}stack["delete"](array);stack["delete"](other);return result}module.exports=equalArrays},{"./_SetCache":118,"./_arraySome":132,"./_cacheHas":191}],215:[function(require,module,exports){var Symbol=require("./_Symbol"),Uint8Array=require("./_Uint8Array"),eq=require("./eq"),equalArrays=require("./_equalArrays"),mapToArray=require("./_mapToArray"),setToArray=require("./_setToArray");
/** Used to compose bitmasks for value comparisons. */var COMPARE_PARTIAL_FLAG=1,COMPARE_UNORDERED_FLAG=2;
/** `Object#toString` result references. */var boolTag="[object Boolean]",dateTag="[object Date]",errorTag="[object Error]",mapTag="[object Map]",numberTag="[object Number]",regexpTag="[object RegExp]",setTag="[object Set]",stringTag="[object String]",symbolTag="[object Symbol]";var arrayBufferTag="[object ArrayBuffer]",dataViewTag="[object DataView]";
/** Used to convert symbols to primitives and strings. */var symbolProto=Symbol?Symbol.prototype:undefined,symbolValueOf=symbolProto?symbolProto.valueOf:undefined;
/**
 * A specialized version of `baseIsEqualDeep` for comparing objects of
 * the same `toStringTag`.
 *
 * **Note:** This function only supports comparing values with tags of
 * `Boolean`, `Date`, `Error`, `Number`, `RegExp`, or `String`.
 *
 * @private
 * @param {Object} object The object to compare.
 * @param {Object} other The other object to compare.
 * @param {string} tag The `toStringTag` of the objects to compare.
 * @param {number} bitmask The bitmask flags. See `baseIsEqual` for more details.
 * @param {Function} customizer The function to customize comparisons.
 * @param {Function} equalFunc The function to determine equivalents of values.
 * @param {Object} stack Tracks traversed `object` and `other` objects.
 * @returns {boolean} Returns `true` if the objects are equivalent, else `false`.
 */function equalByTag(object,other,tag,bitmask,customizer,equalFunc,stack){switch(tag){case dataViewTag:if(object.byteLength!=other.byteLength||object.byteOffset!=other.byteOffset){return false}object=object.buffer;other=other.buffer;case arrayBufferTag:if(object.byteLength!=other.byteLength||!equalFunc(new Uint8Array(object),new Uint8Array(other))){return false}return true;case boolTag:case dateTag:case numberTag:
// Coerce booleans to `1` or `0` and dates to milliseconds.
// Invalid dates are coerced to `NaN`.
return eq(+object,+other);case errorTag:return object.name==other.name&&object.message==other.message;case regexpTag:case stringTag:
// Coerce regexes to strings and treat strings, primitives and objects,
// as equal. See http://www.ecma-international.org/ecma-262/7.0/#sec-regexp.prototype.tostring
// for more details.
return object==other+"";case mapTag:var convert=mapToArray;case setTag:var isPartial=bitmask&COMPARE_PARTIAL_FLAG;convert||(convert=setToArray);if(object.size!=other.size&&!isPartial){return false}
// Assume cyclic values are equal.
var stacked=stack.get(object);if(stacked){return stacked==other}bitmask|=COMPARE_UNORDERED_FLAG;
// Recursively compare objects (susceptible to call stack limits).
stack.set(object,other);var result=equalArrays(convert(object),convert(other),bitmask,customizer,equalFunc,stack);stack["delete"](object);return result;case symbolTag:if(symbolValueOf){return symbolValueOf.call(object)==symbolValueOf.call(other)}}return false}module.exports=equalByTag},{"./_Symbol":120,"./_Uint8Array":121,"./_equalArrays":214,"./_mapToArray":258,"./_setToArray":272,"./eq":291}],216:[function(require,module,exports){var getAllKeys=require("./_getAllKeys");
/** Used to compose bitmasks for value comparisons. */var COMPARE_PARTIAL_FLAG=1;
/** Used for built-in method references. */var objectProto=Object.prototype;
/** Used to check objects for own properties. */var hasOwnProperty=objectProto.hasOwnProperty;
/**
 * A specialized version of `baseIsEqualDeep` for objects with support for
 * partial deep comparisons.
 *
 * @private
 * @param {Object} object The object to compare.
 * @param {Object} other The other object to compare.
 * @param {number} bitmask The bitmask flags. See `baseIsEqual` for more details.
 * @param {Function} customizer The function to customize comparisons.
 * @param {Function} equalFunc The function to determine equivalents of values.
 * @param {Object} stack Tracks traversed `object` and `other` objects.
 * @returns {boolean} Returns `true` if the objects are equivalent, else `false`.
 */function equalObjects(object,other,bitmask,customizer,equalFunc,stack){var isPartial=bitmask&COMPARE_PARTIAL_FLAG,objProps=getAllKeys(object),objLength=objProps.length,othProps=getAllKeys(other),othLength=othProps.length;if(objLength!=othLength&&!isPartial){return false}var index=objLength;while(index--){var key=objProps[index];if(!(isPartial?key in other:hasOwnProperty.call(other,key))){return false}}
// Assume cyclic values are equal.
var stacked=stack.get(object);if(stacked&&stack.get(other)){return stacked==other}var result=true;stack.set(object,other);stack.set(other,object);var skipCtor=isPartial;while(++index<objLength){key=objProps[index];var objValue=object[key],othValue=other[key];if(customizer){var compared=isPartial?customizer(othValue,objValue,key,other,object,stack):customizer(objValue,othValue,key,object,other,stack)}
// Recursively compare objects (susceptible to call stack limits).
if(!(compared===undefined?objValue===othValue||equalFunc(objValue,othValue,bitmask,customizer,stack):compared)){result=false;break}skipCtor||(skipCtor=key=="constructor")}if(result&&!skipCtor){var objCtor=object.constructor,othCtor=other.constructor;
// Non `Object` object instances with different constructors are not equal.
if(objCtor!=othCtor&&("constructor"in object&&"constructor"in other)&&!(typeof objCtor=="function"&&objCtor instanceof objCtor&&typeof othCtor=="function"&&othCtor instanceof othCtor)){result=false}}stack["delete"](object);stack["delete"](other);return result}module.exports=equalObjects},{"./_getAllKeys":219}],217:[function(require,module,exports){var flatten=require("./flatten"),overRest=require("./_overRest"),setToString=require("./_setToString");
/**
 * A specialized version of `baseRest` which flattens the rest array.
 *
 * @private
 * @param {Function} func The function to apply a rest parameter to.
 * @returns {Function} Returns the new function.
 */function flatRest(func){return setToString(overRest(func,undefined,flatten),func+"")}module.exports=flatRest},{"./_overRest":267,"./_setToString":273,"./flatten":295}],218:[function(require,module,exports){(function(global){
/** Detect free variable `global` from Node.js. */
var freeGlobal=typeof global=="object"&&global&&global.Object===Object&&global;module.exports=freeGlobal}).call(this,typeof global!=="undefined"?global:typeof self!=="undefined"?self:typeof window!=="undefined"?window:{})},{}],219:[function(require,module,exports){var baseGetAllKeys=require("./_baseGetAllKeys"),getSymbols=require("./_getSymbols"),keys=require("./keys");
/**
 * Creates an array of own enumerable property names and symbols of `object`.
 *
 * @private
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of property names and symbols.
 */function getAllKeys(object){return baseGetAllKeys(object,keys,getSymbols)}module.exports=getAllKeys},{"./_baseGetAllKeys":150,"./_getSymbols":226,"./keys":319}],220:[function(require,module,exports){var baseGetAllKeys=require("./_baseGetAllKeys"),getSymbolsIn=require("./_getSymbolsIn"),keysIn=require("./keysIn");
/**
 * Creates an array of own and inherited enumerable property names and
 * symbols of `object`.
 *
 * @private
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of property names and symbols.
 */function getAllKeysIn(object){return baseGetAllKeys(object,keysIn,getSymbolsIn)}module.exports=getAllKeysIn},{"./_baseGetAllKeys":150,"./_getSymbolsIn":227,"./keysIn":320}],221:[function(require,module,exports){var isKeyable=require("./_isKeyable");
/**
 * Gets the data for `map`.
 *
 * @private
 * @param {Object} map The map to query.
 * @param {string} key The reference key.
 * @returns {*} Returns the map data.
 */function getMapData(map,key){var data=map.__data__;return isKeyable(key)?data[typeof key=="string"?"string":"hash"]:data.map}module.exports=getMapData},{"./_isKeyable":244}],222:[function(require,module,exports){var isStrictComparable=require("./_isStrictComparable"),keys=require("./keys");
/**
 * Gets the property names, values, and compare flags of `object`.
 *
 * @private
 * @param {Object} object The object to query.
 * @returns {Array} Returns the match data of `object`.
 */function getMatchData(object){var result=keys(object),length=result.length;while(length--){var key=result[length],value=object[key];result[length]=[key,value,isStrictComparable(value)]}return result}module.exports=getMatchData},{"./_isStrictComparable":247,"./keys":319}],223:[function(require,module,exports){var baseIsNative=require("./_baseIsNative"),getValue=require("./_getValue");
/**
 * Gets the native function at `key` of `object`.
 *
 * @private
 * @param {Object} object The object to query.
 * @param {string} key The key of the method to get.
 * @returns {*} Returns the function if it's native, else `undefined`.
 */function getNative(object,key){var value=getValue(object,key);return baseIsNative(value)?value:undefined}module.exports=getNative},{"./_baseIsNative":162,"./_getValue":229}],224:[function(require,module,exports){var overArg=require("./_overArg");
/** Built-in value references. */var getPrototype=overArg(Object.getPrototypeOf,Object);module.exports=getPrototype},{"./_overArg":266}],225:[function(require,module,exports){var Symbol=require("./_Symbol");
/** Used for built-in method references. */var objectProto=Object.prototype;
/** Used to check objects for own properties. */var hasOwnProperty=objectProto.hasOwnProperty;
/**
 * Used to resolve the
 * [`toStringTag`](http://ecma-international.org/ecma-262/7.0/#sec-object.prototype.tostring)
 * of values.
 */var nativeObjectToString=objectProto.toString;
/** Built-in value references. */var symToStringTag=Symbol?Symbol.toStringTag:undefined;
/**
 * A specialized version of `baseGetTag` which ignores `Symbol.toStringTag` values.
 *
 * @private
 * @param {*} value The value to query.
 * @returns {string} Returns the raw `toStringTag`.
 */function getRawTag(value){var isOwn=hasOwnProperty.call(value,symToStringTag),tag=value[symToStringTag];try{value[symToStringTag]=undefined;var unmasked=true}catch(e){}var result=nativeObjectToString.call(value);if(unmasked){if(isOwn){value[symToStringTag]=tag}else{delete value[symToStringTag]}}return result}module.exports=getRawTag},{"./_Symbol":120}],226:[function(require,module,exports){var arrayFilter=require("./_arrayFilter"),stubArray=require("./stubArray");
/** Used for built-in method references. */var objectProto=Object.prototype;
/** Built-in value references. */var propertyIsEnumerable=objectProto.propertyIsEnumerable;
/* Built-in method references for those with the same name as other `lodash` methods. */var nativeGetSymbols=Object.getOwnPropertySymbols;
/**
 * Creates an array of the own enumerable symbols of `object`.
 *
 * @private
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of symbols.
 */var getSymbols=!nativeGetSymbols?stubArray:function(object){if(object==null){return[]}object=Object(object);return arrayFilter(nativeGetSymbols(object),function(symbol){return propertyIsEnumerable.call(object,symbol)})};module.exports=getSymbols},{"./_arrayFilter":125,"./stubArray":337}],227:[function(require,module,exports){var arrayPush=require("./_arrayPush"),getPrototype=require("./_getPrototype"),getSymbols=require("./_getSymbols"),stubArray=require("./stubArray");
/* Built-in method references for those with the same name as other `lodash` methods. */var nativeGetSymbols=Object.getOwnPropertySymbols;
/**
 * Creates an array of the own and inherited enumerable symbols of `object`.
 *
 * @private
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of symbols.
 */var getSymbolsIn=!nativeGetSymbols?stubArray:function(object){var result=[];while(object){arrayPush(result,getSymbols(object));object=getPrototype(object)}return result};module.exports=getSymbolsIn},{"./_arrayPush":130,"./_getPrototype":224,"./_getSymbols":226,"./stubArray":337}],228:[function(require,module,exports){var DataView=require("./_DataView"),Map=require("./_Map"),Promise=require("./_Promise"),Set=require("./_Set"),WeakMap=require("./_WeakMap"),baseGetTag=require("./_baseGetTag"),toSource=require("./_toSource");
/** `Object#toString` result references. */var mapTag="[object Map]",objectTag="[object Object]",promiseTag="[object Promise]",setTag="[object Set]",weakMapTag="[object WeakMap]";var dataViewTag="[object DataView]";
/** Used to detect maps, sets, and weakmaps. */var dataViewCtorString=toSource(DataView),mapCtorString=toSource(Map),promiseCtorString=toSource(Promise),setCtorString=toSource(Set),weakMapCtorString=toSource(WeakMap);
/**
 * Gets the `toStringTag` of `value`.
 *
 * @private
 * @param {*} value The value to query.
 * @returns {string} Returns the `toStringTag`.
 */var getTag=baseGetTag;
// Fallback for data views, maps, sets, and weak maps in IE 11 and promises in Node.js < 6.
if(DataView&&getTag(new DataView(new ArrayBuffer(1)))!=dataViewTag||Map&&getTag(new Map)!=mapTag||Promise&&getTag(Promise.resolve())!=promiseTag||Set&&getTag(new Set)!=setTag||WeakMap&&getTag(new WeakMap)!=weakMapTag){getTag=function(value){var result=baseGetTag(value),Ctor=result==objectTag?value.constructor:undefined,ctorString=Ctor?toSource(Ctor):"";if(ctorString){switch(ctorString){case dataViewCtorString:return dataViewTag;case mapCtorString:return mapTag;case promiseCtorString:return promiseTag;case setCtorString:return setTag;case weakMapCtorString:return weakMapTag}}return result}}module.exports=getTag},{"./_DataView":111,"./_Map":114,"./_Promise":116,"./_Set":117,"./_WeakMap":122,"./_baseGetTag":151,"./_toSource":284}],229:[function(require,module,exports){
/**
 * Gets the value at `key` of `object`.
 *
 * @private
 * @param {Object} [object] The object to query.
 * @param {string} key The key of the property to get.
 * @returns {*} Returns the property value.
 */
function getValue(object,key){return object==null?undefined:object[key]}module.exports=getValue},{}],230:[function(require,module,exports){var castPath=require("./_castPath"),isArguments=require("./isArguments"),isArray=require("./isArray"),isIndex=require("./_isIndex"),isLength=require("./isLength"),toKey=require("./_toKey");
/**
 * Checks if `path` exists on `object`.
 *
 * @private
 * @param {Object} object The object to query.
 * @param {Array|string} path The path to check.
 * @param {Function} hasFunc The function to check properties.
 * @returns {boolean} Returns `true` if `path` exists, else `false`.
 */function hasPath(object,path,hasFunc){path=castPath(path,object);var index=-1,length=path.length,result=false;while(++index<length){var key=toKey(path[index]);if(!(result=object!=null&&hasFunc(object,key))){break}object=object[key]}if(result||++index!=length){return result}length=object==null?0:object.length;return!!length&&isLength(length)&&isIndex(key,length)&&(isArray(object)||isArguments(object))}module.exports=hasPath},{"./_castPath":193,"./_isIndex":241,"./_toKey":283,"./isArguments":302,"./isArray":303,"./isLength":309}],231:[function(require,module,exports){
/** Used to compose unicode character classes. */
var rsAstralRange="\\ud800-\\udfff",rsComboMarksRange="\\u0300-\\u036f",reComboHalfMarksRange="\\ufe20-\\ufe2f",rsComboSymbolsRange="\\u20d0-\\u20ff",rsComboRange=rsComboMarksRange+reComboHalfMarksRange+rsComboSymbolsRange,rsVarRange="\\ufe0e\\ufe0f";
/** Used to compose unicode capture groups. */var rsZWJ="\\u200d";
/** Used to detect strings with [zero-width joiners or code points from the astral planes](http://eev.ee/blog/2015/09/12/dark-corners-of-unicode/). */var reHasUnicode=RegExp("["+rsZWJ+rsAstralRange+rsComboRange+rsVarRange+"]");
/**
 * Checks if `string` contains Unicode symbols.
 *
 * @private
 * @param {string} string The string to inspect.
 * @returns {boolean} Returns `true` if a symbol is found, else `false`.
 */function hasUnicode(string){return reHasUnicode.test(string)}module.exports=hasUnicode},{}],232:[function(require,module,exports){var nativeCreate=require("./_nativeCreate");
/**
 * Removes all key-value entries from the hash.
 *
 * @private
 * @name clear
 * @memberOf Hash
 */function hashClear(){this.__data__=nativeCreate?nativeCreate(null):{};this.size=0}module.exports=hashClear},{"./_nativeCreate":261}],233:[function(require,module,exports){
/**
 * Removes `key` and its value from the hash.
 *
 * @private
 * @name delete
 * @memberOf Hash
 * @param {Object} hash The hash to modify.
 * @param {string} key The key of the value to remove.
 * @returns {boolean} Returns `true` if the entry was removed, else `false`.
 */
function hashDelete(key){var result=this.has(key)&&delete this.__data__[key];this.size-=result?1:0;return result}module.exports=hashDelete},{}],234:[function(require,module,exports){var nativeCreate=require("./_nativeCreate");
/** Used to stand-in for `undefined` hash values. */var HASH_UNDEFINED="__lodash_hash_undefined__";
/** Used for built-in method references. */var objectProto=Object.prototype;
/** Used to check objects for own properties. */var hasOwnProperty=objectProto.hasOwnProperty;
/**
 * Gets the hash value for `key`.
 *
 * @private
 * @name get
 * @memberOf Hash
 * @param {string} key The key of the value to get.
 * @returns {*} Returns the entry value.
 */function hashGet(key){var data=this.__data__;if(nativeCreate){var result=data[key];return result===HASH_UNDEFINED?undefined:result}return hasOwnProperty.call(data,key)?data[key]:undefined}module.exports=hashGet},{"./_nativeCreate":261}],235:[function(require,module,exports){var nativeCreate=require("./_nativeCreate");
/** Used for built-in method references. */var objectProto=Object.prototype;
/** Used to check objects for own properties. */var hasOwnProperty=objectProto.hasOwnProperty;
/**
 * Checks if a hash value for `key` exists.
 *
 * @private
 * @name has
 * @memberOf Hash
 * @param {string} key The key of the entry to check.
 * @returns {boolean} Returns `true` if an entry for `key` exists, else `false`.
 */function hashHas(key){var data=this.__data__;return nativeCreate?data[key]!==undefined:hasOwnProperty.call(data,key)}module.exports=hashHas},{"./_nativeCreate":261}],236:[function(require,module,exports){var nativeCreate=require("./_nativeCreate");
/** Used to stand-in for `undefined` hash values. */var HASH_UNDEFINED="__lodash_hash_undefined__";
/**
 * Sets the hash `key` to `value`.
 *
 * @private
 * @name set
 * @memberOf Hash
 * @param {string} key The key of the value to set.
 * @param {*} value The value to set.
 * @returns {Object} Returns the hash instance.
 */function hashSet(key,value){var data=this.__data__;this.size+=this.has(key)?0:1;data[key]=nativeCreate&&value===undefined?HASH_UNDEFINED:value;return this}module.exports=hashSet},{"./_nativeCreate":261}],237:[function(require,module,exports){
/** Used for built-in method references. */
var objectProto=Object.prototype;
/** Used to check objects for own properties. */var hasOwnProperty=objectProto.hasOwnProperty;
/**
 * Initializes an array clone.
 *
 * @private
 * @param {Array} array The array to clone.
 * @returns {Array} Returns the initialized clone.
 */function initCloneArray(array){var length=array.length,result=new array.constructor(length);
// Add properties assigned by `RegExp#exec`.
if(length&&typeof array[0]=="string"&&hasOwnProperty.call(array,"index")){result.index=array.index;result.input=array.input}return result}module.exports=initCloneArray},{}],238:[function(require,module,exports){var cloneArrayBuffer=require("./_cloneArrayBuffer"),cloneDataView=require("./_cloneDataView"),cloneRegExp=require("./_cloneRegExp"),cloneSymbol=require("./_cloneSymbol"),cloneTypedArray=require("./_cloneTypedArray");
/** `Object#toString` result references. */var boolTag="[object Boolean]",dateTag="[object Date]",mapTag="[object Map]",numberTag="[object Number]",regexpTag="[object RegExp]",setTag="[object Set]",stringTag="[object String]",symbolTag="[object Symbol]";var arrayBufferTag="[object ArrayBuffer]",dataViewTag="[object DataView]",float32Tag="[object Float32Array]",float64Tag="[object Float64Array]",int8Tag="[object Int8Array]",int16Tag="[object Int16Array]",int32Tag="[object Int32Array]",uint8Tag="[object Uint8Array]",uint8ClampedTag="[object Uint8ClampedArray]",uint16Tag="[object Uint16Array]",uint32Tag="[object Uint32Array]";
/**
 * Initializes an object clone based on its `toStringTag`.
 *
 * **Note:** This function only supports cloning values with tags of
 * `Boolean`, `Date`, `Error`, `Map`, `Number`, `RegExp`, `Set`, or `String`.
 *
 * @private
 * @param {Object} object The object to clone.
 * @param {string} tag The `toStringTag` of the object to clone.
 * @param {boolean} [isDeep] Specify a deep clone.
 * @returns {Object} Returns the initialized clone.
 */function initCloneByTag(object,tag,isDeep){var Ctor=object.constructor;switch(tag){case arrayBufferTag:return cloneArrayBuffer(object);case boolTag:case dateTag:return new Ctor(+object);case dataViewTag:return cloneDataView(object,isDeep);case float32Tag:case float64Tag:case int8Tag:case int16Tag:case int32Tag:case uint8Tag:case uint8ClampedTag:case uint16Tag:case uint32Tag:return cloneTypedArray(object,isDeep);case mapTag:return new Ctor;case numberTag:case stringTag:return new Ctor(object);case regexpTag:return cloneRegExp(object);case setTag:return new Ctor;case symbolTag:return cloneSymbol(object)}}module.exports=initCloneByTag},{"./_cloneArrayBuffer":194,"./_cloneDataView":196,"./_cloneRegExp":197,"./_cloneSymbol":198,"./_cloneTypedArray":199}],239:[function(require,module,exports){var baseCreate=require("./_baseCreate"),getPrototype=require("./_getPrototype"),isPrototype=require("./_isPrototype");
/**
 * Initializes an object clone.
 *
 * @private
 * @param {Object} object The object to clone.
 * @returns {Object} Returns the initialized clone.
 */function initCloneObject(object){return typeof object.constructor=="function"&&!isPrototype(object)?baseCreate(getPrototype(object)):{}}module.exports=initCloneObject},{"./_baseCreate":141,"./_getPrototype":224,"./_isPrototype":246}],240:[function(require,module,exports){var Symbol=require("./_Symbol"),isArguments=require("./isArguments"),isArray=require("./isArray");
/** Built-in value references. */var spreadableSymbol=Symbol?Symbol.isConcatSpreadable:undefined;
/**
 * Checks if `value` is a flattenable `arguments` object or array.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is flattenable, else `false`.
 */function isFlattenable(value){return isArray(value)||isArguments(value)||!!(spreadableSymbol&&value&&value[spreadableSymbol])}module.exports=isFlattenable},{"./_Symbol":120,"./isArguments":302,"./isArray":303}],241:[function(require,module,exports){
/** Used as references for various `Number` constants. */
var MAX_SAFE_INTEGER=9007199254740991;
/** Used to detect unsigned integer values. */var reIsUint=/^(?:0|[1-9]\d*)$/;
/**
 * Checks if `value` is a valid array-like index.
 *
 * @private
 * @param {*} value The value to check.
 * @param {number} [length=MAX_SAFE_INTEGER] The upper bounds of a valid index.
 * @returns {boolean} Returns `true` if `value` is a valid index, else `false`.
 */function isIndex(value,length){var type=typeof value;length=length==null?MAX_SAFE_INTEGER:length;return!!length&&(type=="number"||type!="symbol"&&reIsUint.test(value))&&(value>-1&&value%1==0&&value<length)}module.exports=isIndex},{}],242:[function(require,module,exports){var eq=require("./eq"),isArrayLike=require("./isArrayLike"),isIndex=require("./_isIndex"),isObject=require("./isObject");
/**
 * Checks if the given arguments are from an iteratee call.
 *
 * @private
 * @param {*} value The potential iteratee value argument.
 * @param {*} index The potential iteratee index or key argument.
 * @param {*} object The potential iteratee object argument.
 * @returns {boolean} Returns `true` if the arguments are from an iteratee call,
 *  else `false`.
 */function isIterateeCall(value,index,object){if(!isObject(object)){return false}var type=typeof index;if(type=="number"?isArrayLike(object)&&isIndex(index,object.length):type=="string"&&index in object){return eq(object[index],value)}return false}module.exports=isIterateeCall},{"./_isIndex":241,"./eq":291,"./isArrayLike":304,"./isObject":311}],243:[function(require,module,exports){var isArray=require("./isArray"),isSymbol=require("./isSymbol");
/** Used to match property names within property paths. */var reIsDeepProp=/\.|\[(?:[^[\]]*|(["'])(?:(?!\1)[^\\]|\\.)*?\1)\]/,reIsPlainProp=/^\w*$/;
/**
 * Checks if `value` is a property name and not a property path.
 *
 * @private
 * @param {*} value The value to check.
 * @param {Object} [object] The object to query keys on.
 * @returns {boolean} Returns `true` if `value` is a property name, else `false`.
 */function isKey(value,object){if(isArray(value)){return false}var type=typeof value;if(type=="number"||type=="symbol"||type=="boolean"||value==null||isSymbol(value)){return true}return reIsPlainProp.test(value)||!reIsDeepProp.test(value)||object!=null&&value in Object(object)}module.exports=isKey},{"./isArray":303,"./isSymbol":316}],244:[function(require,module,exports){
/**
 * Checks if `value` is suitable for use as unique object key.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is suitable, else `false`.
 */
function isKeyable(value){var type=typeof value;return type=="string"||type=="number"||type=="symbol"||type=="boolean"?value!=="__proto__":value===null}module.exports=isKeyable},{}],245:[function(require,module,exports){var coreJsData=require("./_coreJsData");
/** Used to detect methods masquerading as native. */var maskSrcKey=function(){var uid=/[^.]+$/.exec(coreJsData&&coreJsData.keys&&coreJsData.keys.IE_PROTO||"");return uid?"Symbol(src)_1."+uid:""}();
/**
 * Checks if `func` has its source masked.
 *
 * @private
 * @param {Function} func The function to check.
 * @returns {boolean} Returns `true` if `func` is masked, else `false`.
 */function isMasked(func){return!!maskSrcKey&&maskSrcKey in func}module.exports=isMasked},{"./_coreJsData":206}],246:[function(require,module,exports){
/** Used for built-in method references. */
var objectProto=Object.prototype;
/**
 * Checks if `value` is likely a prototype object.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a prototype, else `false`.
 */function isPrototype(value){var Ctor=value&&value.constructor,proto=typeof Ctor=="function"&&Ctor.prototype||objectProto;return value===proto}module.exports=isPrototype},{}],247:[function(require,module,exports){var isObject=require("./isObject");
/**
 * Checks if `value` is suitable for strict equality comparisons, i.e. `===`.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` if suitable for strict
 *  equality comparisons, else `false`.
 */function isStrictComparable(value){return value===value&&!isObject(value)}module.exports=isStrictComparable},{"./isObject":311}],248:[function(require,module,exports){
/**
 * Removes all key-value entries from the list cache.
 *
 * @private
 * @name clear
 * @memberOf ListCache
 */
function listCacheClear(){this.__data__=[];this.size=0}module.exports=listCacheClear},{}],249:[function(require,module,exports){var assocIndexOf=require("./_assocIndexOf");
/** Used for built-in method references. */var arrayProto=Array.prototype;
/** Built-in value references. */var splice=arrayProto.splice;
/**
 * Removes `key` and its value from the list cache.
 *
 * @private
 * @name delete
 * @memberOf ListCache
 * @param {string} key The key of the value to remove.
 * @returns {boolean} Returns `true` if the entry was removed, else `false`.
 */function listCacheDelete(key){var data=this.__data__,index=assocIndexOf(data,key);if(index<0){return false}var lastIndex=data.length-1;if(index==lastIndex){data.pop()}else{splice.call(data,index,1)}--this.size;return true}module.exports=listCacheDelete},{"./_assocIndexOf":136}],250:[function(require,module,exports){var assocIndexOf=require("./_assocIndexOf");
/**
 * Gets the list cache value for `key`.
 *
 * @private
 * @name get
 * @memberOf ListCache
 * @param {string} key The key of the value to get.
 * @returns {*} Returns the entry value.
 */function listCacheGet(key){var data=this.__data__,index=assocIndexOf(data,key);return index<0?undefined:data[index][1]}module.exports=listCacheGet},{"./_assocIndexOf":136}],251:[function(require,module,exports){var assocIndexOf=require("./_assocIndexOf");
/**
 * Checks if a list cache value for `key` exists.
 *
 * @private
 * @name has
 * @memberOf ListCache
 * @param {string} key The key of the entry to check.
 * @returns {boolean} Returns `true` if an entry for `key` exists, else `false`.
 */function listCacheHas(key){return assocIndexOf(this.__data__,key)>-1}module.exports=listCacheHas},{"./_assocIndexOf":136}],252:[function(require,module,exports){var assocIndexOf=require("./_assocIndexOf");
/**
 * Sets the list cache `key` to `value`.
 *
 * @private
 * @name set
 * @memberOf ListCache
 * @param {string} key The key of the value to set.
 * @param {*} value The value to set.
 * @returns {Object} Returns the list cache instance.
 */function listCacheSet(key,value){var data=this.__data__,index=assocIndexOf(data,key);if(index<0){++this.size;data.push([key,value])}else{data[index][1]=value}return this}module.exports=listCacheSet},{"./_assocIndexOf":136}],253:[function(require,module,exports){var Hash=require("./_Hash"),ListCache=require("./_ListCache"),Map=require("./_Map");
/**
 * Removes all key-value entries from the map.
 *
 * @private
 * @name clear
 * @memberOf MapCache
 */function mapCacheClear(){this.size=0;this.__data__={hash:new Hash,map:new(Map||ListCache),string:new Hash}}module.exports=mapCacheClear},{"./_Hash":112,"./_ListCache":113,"./_Map":114}],254:[function(require,module,exports){var getMapData=require("./_getMapData");
/**
 * Removes `key` and its value from the map.
 *
 * @private
 * @name delete
 * @memberOf MapCache
 * @param {string} key The key of the value to remove.
 * @returns {boolean} Returns `true` if the entry was removed, else `false`.
 */function mapCacheDelete(key){var result=getMapData(this,key)["delete"](key);this.size-=result?1:0;return result}module.exports=mapCacheDelete},{"./_getMapData":221}],255:[function(require,module,exports){var getMapData=require("./_getMapData");
/**
 * Gets the map value for `key`.
 *
 * @private
 * @name get
 * @memberOf MapCache
 * @param {string} key The key of the value to get.
 * @returns {*} Returns the entry value.
 */function mapCacheGet(key){return getMapData(this,key).get(key)}module.exports=mapCacheGet},{"./_getMapData":221}],256:[function(require,module,exports){var getMapData=require("./_getMapData");
/**
 * Checks if a map value for `key` exists.
 *
 * @private
 * @name has
 * @memberOf MapCache
 * @param {string} key The key of the entry to check.
 * @returns {boolean} Returns `true` if an entry for `key` exists, else `false`.
 */function mapCacheHas(key){return getMapData(this,key).has(key)}module.exports=mapCacheHas},{"./_getMapData":221}],257:[function(require,module,exports){var getMapData=require("./_getMapData");
/**
 * Sets the map `key` to `value`.
 *
 * @private
 * @name set
 * @memberOf MapCache
 * @param {string} key The key of the value to set.
 * @param {*} value The value to set.
 * @returns {Object} Returns the map cache instance.
 */function mapCacheSet(key,value){var data=getMapData(this,key),size=data.size;data.set(key,value);this.size+=data.size==size?0:1;return this}module.exports=mapCacheSet},{"./_getMapData":221}],258:[function(require,module,exports){
/**
 * Converts `map` to its key-value pairs.
 *
 * @private
 * @param {Object} map The map to convert.
 * @returns {Array} Returns the key-value pairs.
 */
function mapToArray(map){var index=-1,result=Array(map.size);map.forEach(function(value,key){result[++index]=[key,value]});return result}module.exports=mapToArray},{}],259:[function(require,module,exports){
/**
 * A specialized version of `matchesProperty` for source values suitable
 * for strict equality comparisons, i.e. `===`.
 *
 * @private
 * @param {string} key The key of the property to get.
 * @param {*} srcValue The value to match.
 * @returns {Function} Returns the new spec function.
 */
function matchesStrictComparable(key,srcValue){return function(object){if(object==null){return false}return object[key]===srcValue&&(srcValue!==undefined||key in Object(object))}}module.exports=matchesStrictComparable},{}],260:[function(require,module,exports){var memoize=require("./memoize");
/** Used as the maximum memoize cache size. */var MAX_MEMOIZE_SIZE=500;
/**
 * A specialized version of `_.memoize` which clears the memoized function's
 * cache when it exceeds `MAX_MEMOIZE_SIZE`.
 *
 * @private
 * @param {Function} func The function to have its output memoized.
 * @returns {Function} Returns the new memoized function.
 */function memoizeCapped(func){var result=memoize(func,function(key){if(cache.size===MAX_MEMOIZE_SIZE){cache.clear()}return key});var cache=result.cache;return result}module.exports=memoizeCapped},{"./memoize":325}],261:[function(require,module,exports){var getNative=require("./_getNative");
/* Built-in method references that are verified to be native. */var nativeCreate=getNative(Object,"create");module.exports=nativeCreate},{"./_getNative":223}],262:[function(require,module,exports){var overArg=require("./_overArg");
/* Built-in method references for those with the same name as other `lodash` methods. */var nativeKeys=overArg(Object.keys,Object);module.exports=nativeKeys},{"./_overArg":266}],263:[function(require,module,exports){
/**
 * This function is like
 * [`Object.keys`](http://ecma-international.org/ecma-262/7.0/#sec-object.keys)
 * except that it includes inherited enumerable properties.
 *
 * @private
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of property names.
 */
function nativeKeysIn(object){var result=[];if(object!=null){for(var key in Object(object)){result.push(key)}}return result}module.exports=nativeKeysIn},{}],264:[function(require,module,exports){var freeGlobal=require("./_freeGlobal");
/** Detect free variable `exports`. */var freeExports=typeof exports=="object"&&exports&&!exports.nodeType&&exports;
/** Detect free variable `module`. */var freeModule=freeExports&&typeof module=="object"&&module&&!module.nodeType&&module;
/** Detect the popular CommonJS extension `module.exports`. */var moduleExports=freeModule&&freeModule.exports===freeExports;
/** Detect free variable `process` from Node.js. */var freeProcess=moduleExports&&freeGlobal.process;
/** Used to access faster Node.js helpers. */var nodeUtil=function(){try{
// Use `util.types` for Node.js 10+.
var types=freeModule&&freeModule.require&&freeModule.require("util").types;if(types){return types}
// Legacy `process.binding('util')` for Node.js < 10.
return freeProcess&&freeProcess.binding&&freeProcess.binding("util")}catch(e){}}();module.exports=nodeUtil},{"./_freeGlobal":218}],265:[function(require,module,exports){
/** Used for built-in method references. */
var objectProto=Object.prototype;
/**
 * Used to resolve the
 * [`toStringTag`](http://ecma-international.org/ecma-262/7.0/#sec-object.prototype.tostring)
 * of values.
 */var nativeObjectToString=objectProto.toString;
/**
 * Converts `value` to a string using `Object.prototype.toString`.
 *
 * @private
 * @param {*} value The value to convert.
 * @returns {string} Returns the converted string.
 */function objectToString(value){return nativeObjectToString.call(value)}module.exports=objectToString},{}],266:[function(require,module,exports){
/**
 * Creates a unary function that invokes `func` with its argument transformed.
 *
 * @private
 * @param {Function} func The function to wrap.
 * @param {Function} transform The argument transform.
 * @returns {Function} Returns the new function.
 */
function overArg(func,transform){return function(arg){return func(transform(arg))}}module.exports=overArg},{}],267:[function(require,module,exports){var apply=require("./_apply");
/* Built-in method references for those with the same name as other `lodash` methods. */var nativeMax=Math.max;
/**
 * A specialized version of `baseRest` which transforms the rest array.
 *
 * @private
 * @param {Function} func The function to apply a rest parameter to.
 * @param {number} [start=func.length-1] The start position of the rest parameter.
 * @param {Function} transform The rest array transform.
 * @returns {Function} Returns the new function.
 */function overRest(func,start,transform){start=nativeMax(start===undefined?func.length-1:start,0);return function(){var args=arguments,index=-1,length=nativeMax(args.length-start,0),array=Array(length);while(++index<length){array[index]=args[start+index]}index=-1;var otherArgs=Array(start+1);while(++index<start){otherArgs[index]=args[index]}otherArgs[start]=transform(array);return apply(func,this,otherArgs)}}module.exports=overRest},{"./_apply":123}],268:[function(require,module,exports){var freeGlobal=require("./_freeGlobal");
/** Detect free variable `self`. */var freeSelf=typeof self=="object"&&self&&self.Object===Object&&self;
/** Used as a reference to the global object. */var root=freeGlobal||freeSelf||Function("return this")();module.exports=root},{"./_freeGlobal":218}],269:[function(require,module,exports){
/**
 * Gets the value at `key`, unless `key` is "__proto__" or "constructor".
 *
 * @private
 * @param {Object} object The object to query.
 * @param {string} key The key of the property to get.
 * @returns {*} Returns the property value.
 */
function safeGet(object,key){if(key==="constructor"&&typeof object[key]==="function"){return}if(key=="__proto__"){return}return object[key]}module.exports=safeGet},{}],270:[function(require,module,exports){
/** Used to stand-in for `undefined` hash values. */
var HASH_UNDEFINED="__lodash_hash_undefined__";
/**
 * Adds `value` to the array cache.
 *
 * @private
 * @name add
 * @memberOf SetCache
 * @alias push
 * @param {*} value The value to cache.
 * @returns {Object} Returns the cache instance.
 */function setCacheAdd(value){this.__data__.set(value,HASH_UNDEFINED);return this}module.exports=setCacheAdd},{}],271:[function(require,module,exports){
/**
 * Checks if `value` is in the array cache.
 *
 * @private
 * @name has
 * @memberOf SetCache
 * @param {*} value The value to search for.
 * @returns {number} Returns `true` if `value` is found, else `false`.
 */
function setCacheHas(value){return this.__data__.has(value)}module.exports=setCacheHas},{}],272:[function(require,module,exports){
/**
 * Converts `set` to an array of its values.
 *
 * @private
 * @param {Object} set The set to convert.
 * @returns {Array} Returns the values.
 */
function setToArray(set){var index=-1,result=Array(set.size);set.forEach(function(value){result[++index]=value});return result}module.exports=setToArray},{}],273:[function(require,module,exports){var baseSetToString=require("./_baseSetToString"),shortOut=require("./_shortOut");
/**
 * Sets the `toString` method of `func` to return `string`.
 *
 * @private
 * @param {Function} func The function to modify.
 * @param {Function} string The `toString` result.
 * @returns {Function} Returns `func`.
 */var setToString=shortOut(baseSetToString);module.exports=setToString},{"./_baseSetToString":183,"./_shortOut":274}],274:[function(require,module,exports){
/** Used to detect hot functions by number of calls within a span of milliseconds. */
var HOT_COUNT=800,HOT_SPAN=16;
/* Built-in method references for those with the same name as other `lodash` methods. */var nativeNow=Date.now;
/**
 * Creates a function that'll short out and invoke `identity` instead
 * of `func` when it's called `HOT_COUNT` or more times in `HOT_SPAN`
 * milliseconds.
 *
 * @private
 * @param {Function} func The function to restrict.
 * @returns {Function} Returns the new shortable function.
 */function shortOut(func){var count=0,lastCalled=0;return function(){var stamp=nativeNow(),remaining=HOT_SPAN-(stamp-lastCalled);lastCalled=stamp;if(remaining>0){if(++count>=HOT_COUNT){return arguments[0]}}else{count=0}return func.apply(undefined,arguments)}}module.exports=shortOut},{}],275:[function(require,module,exports){var ListCache=require("./_ListCache");
/**
 * Removes all key-value entries from the stack.
 *
 * @private
 * @name clear
 * @memberOf Stack
 */function stackClear(){this.__data__=new ListCache;this.size=0}module.exports=stackClear},{"./_ListCache":113}],276:[function(require,module,exports){
/**
 * Removes `key` and its value from the stack.
 *
 * @private
 * @name delete
 * @memberOf Stack
 * @param {string} key The key of the value to remove.
 * @returns {boolean} Returns `true` if the entry was removed, else `false`.
 */
function stackDelete(key){var data=this.__data__,result=data["delete"](key);this.size=data.size;return result}module.exports=stackDelete},{}],277:[function(require,module,exports){
/**
 * Gets the stack value for `key`.
 *
 * @private
 * @name get
 * @memberOf Stack
 * @param {string} key The key of the value to get.
 * @returns {*} Returns the entry value.
 */
function stackGet(key){return this.__data__.get(key)}module.exports=stackGet},{}],278:[function(require,module,exports){
/**
 * Checks if a stack value for `key` exists.
 *
 * @private
 * @name has
 * @memberOf Stack
 * @param {string} key The key of the entry to check.
 * @returns {boolean} Returns `true` if an entry for `key` exists, else `false`.
 */
function stackHas(key){return this.__data__.has(key)}module.exports=stackHas},{}],279:[function(require,module,exports){var ListCache=require("./_ListCache"),Map=require("./_Map"),MapCache=require("./_MapCache");
/** Used as the size to enable large array optimizations. */var LARGE_ARRAY_SIZE=200;
/**
 * Sets the stack `key` to `value`.
 *
 * @private
 * @name set
 * @memberOf Stack
 * @param {string} key The key of the value to set.
 * @param {*} value The value to set.
 * @returns {Object} Returns the stack cache instance.
 */function stackSet(key,value){var data=this.__data__;if(data instanceof ListCache){var pairs=data.__data__;if(!Map||pairs.length<LARGE_ARRAY_SIZE-1){pairs.push([key,value]);this.size=++data.size;return this}data=this.__data__=new MapCache(pairs)}data.set(key,value);this.size=data.size;return this}module.exports=stackSet},{"./_ListCache":113,"./_Map":114,"./_MapCache":115}],280:[function(require,module,exports){
/**
 * A specialized version of `_.indexOf` which performs strict equality
 * comparisons of values, i.e. `===`.
 *
 * @private
 * @param {Array} array The array to inspect.
 * @param {*} value The value to search for.
 * @param {number} fromIndex The index to search from.
 * @returns {number} Returns the index of the matched value, else `-1`.
 */
function strictIndexOf(array,value,fromIndex){var index=fromIndex-1,length=array.length;while(++index<length){if(array[index]===value){return index}}return-1}module.exports=strictIndexOf},{}],281:[function(require,module,exports){var asciiSize=require("./_asciiSize"),hasUnicode=require("./_hasUnicode"),unicodeSize=require("./_unicodeSize");
/**
 * Gets the number of symbols in `string`.
 *
 * @private
 * @param {string} string The string to inspect.
 * @returns {number} Returns the string size.
 */function stringSize(string){return hasUnicode(string)?unicodeSize(string):asciiSize(string)}module.exports=stringSize},{"./_asciiSize":133,"./_hasUnicode":231,"./_unicodeSize":285}],282:[function(require,module,exports){var memoizeCapped=require("./_memoizeCapped");
/** Used to match property names within property paths. */var rePropName=/[^.[\]]+|\[(?:(-?\d+(?:\.\d+)?)|(["'])((?:(?!\2)[^\\]|\\.)*?)\2)\]|(?=(?:\.|\[\])(?:\.|\[\]|$))/g;
/** Used to match backslashes in property paths. */var reEscapeChar=/\\(\\)?/g;
/**
 * Converts `string` to a property path array.
 *
 * @private
 * @param {string} string The string to convert.
 * @returns {Array} Returns the property path array.
 */var stringToPath=memoizeCapped(function(string){var result=[];if(string.charCodeAt(0)===46/* . */){result.push("")}string.replace(rePropName,function(match,number,quote,subString){result.push(quote?subString.replace(reEscapeChar,"$1"):number||match)});return result});module.exports=stringToPath},{"./_memoizeCapped":260}],283:[function(require,module,exports){var isSymbol=require("./isSymbol");
/** Used as references for various `Number` constants. */var INFINITY=1/0;
/**
 * Converts `value` to a string key if it's not a string or symbol.
 *
 * @private
 * @param {*} value The value to inspect.
 * @returns {string|symbol} Returns the key.
 */function toKey(value){if(typeof value=="string"||isSymbol(value)){return value}var result=value+"";return result=="0"&&1/value==-INFINITY?"-0":result}module.exports=toKey},{"./isSymbol":316}],284:[function(require,module,exports){
/** Used for built-in method references. */
var funcProto=Function.prototype;
/** Used to resolve the decompiled source of functions. */var funcToString=funcProto.toString;
/**
 * Converts `func` to its source code.
 *
 * @private
 * @param {Function} func The function to convert.
 * @returns {string} Returns the source code.
 */function toSource(func){if(func!=null){try{return funcToString.call(func)}catch(e){}try{return func+""}catch(e){}}return""}module.exports=toSource},{}],285:[function(require,module,exports){
/** Used to compose unicode character classes. */
var rsAstralRange="\\ud800-\\udfff",rsComboMarksRange="\\u0300-\\u036f",reComboHalfMarksRange="\\ufe20-\\ufe2f",rsComboSymbolsRange="\\u20d0-\\u20ff",rsComboRange=rsComboMarksRange+reComboHalfMarksRange+rsComboSymbolsRange,rsVarRange="\\ufe0e\\ufe0f";
/** Used to compose unicode capture groups. */var rsAstral="["+rsAstralRange+"]",rsCombo="["+rsComboRange+"]",rsFitz="\\ud83c[\\udffb-\\udfff]",rsModifier="(?:"+rsCombo+"|"+rsFitz+")",rsNonAstral="[^"+rsAstralRange+"]",rsRegional="(?:\\ud83c[\\udde6-\\uddff]){2}",rsSurrPair="[\\ud800-\\udbff][\\udc00-\\udfff]",rsZWJ="\\u200d";
/** Used to compose unicode regexes. */var reOptMod=rsModifier+"?",rsOptVar="["+rsVarRange+"]?",rsOptJoin="(?:"+rsZWJ+"(?:"+[rsNonAstral,rsRegional,rsSurrPair].join("|")+")"+rsOptVar+reOptMod+")*",rsSeq=rsOptVar+reOptMod+rsOptJoin,rsSymbol="(?:"+[rsNonAstral+rsCombo+"?",rsCombo,rsRegional,rsSurrPair,rsAstral].join("|")+")";
/** Used to match [string symbols](https://mathiasbynens.be/notes/javascript-unicode). */var reUnicode=RegExp(rsFitz+"(?="+rsFitz+")|"+rsSymbol+rsSeq,"g");
/**
 * Gets the size of a Unicode `string`.
 *
 * @private
 * @param {string} string The string inspect.
 * @returns {number} Returns the string size.
 */function unicodeSize(string){var result=reUnicode.lastIndex=0;while(reUnicode.test(string)){++result}return result}module.exports=unicodeSize},{}],286:[function(require,module,exports){var baseClone=require("./_baseClone");
/** Used to compose bitmasks for cloning. */var CLONE_SYMBOLS_FLAG=4;
/**
 * Creates a shallow clone of `value`.
 *
 * **Note:** This method is loosely based on the
 * [structured clone algorithm](https://mdn.io/Structured_clone_algorithm)
 * and supports cloning arrays, array buffers, booleans, date objects, maps,
 * numbers, `Object` objects, regexes, sets, strings, symbols, and typed
 * arrays. The own enumerable properties of `arguments` objects are cloned
 * as plain objects. An empty object is returned for uncloneable values such
 * as error objects, functions, DOM nodes, and WeakMaps.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Lang
 * @param {*} value The value to clone.
 * @returns {*} Returns the cloned value.
 * @see _.cloneDeep
 * @example
 *
 * var objects = [{ 'a': 1 }, { 'b': 2 }];
 *
 * var shallow = _.clone(objects);
 * console.log(shallow[0] === objects[0]);
 * // => true
 */function clone(value){return baseClone(value,CLONE_SYMBOLS_FLAG)}module.exports=clone},{"./_baseClone":140}],287:[function(require,module,exports){var baseClone=require("./_baseClone");
/** Used to compose bitmasks for cloning. */var CLONE_DEEP_FLAG=1,CLONE_SYMBOLS_FLAG=4;
/**
 * This method is like `_.clone` except that it recursively clones `value`.
 *
 * @static
 * @memberOf _
 * @since 1.0.0
 * @category Lang
 * @param {*} value The value to recursively clone.
 * @returns {*} Returns the deep cloned value.
 * @see _.clone
 * @example
 *
 * var objects = [{ 'a': 1 }, { 'b': 2 }];
 *
 * var deep = _.cloneDeep(objects);
 * console.log(deep[0] === objects[0]);
 * // => false
 */function cloneDeep(value){return baseClone(value,CLONE_DEEP_FLAG|CLONE_SYMBOLS_FLAG)}module.exports=cloneDeep},{"./_baseClone":140}],288:[function(require,module,exports){
/**
 * Creates a function that returns `value`.
 *
 * @static
 * @memberOf _
 * @since 2.4.0
 * @category Util
 * @param {*} value The value to return from the new function.
 * @returns {Function} Returns the new constant function.
 * @example
 *
 * var objects = _.times(2, _.constant({ 'a': 1 }));
 *
 * console.log(objects);
 * // => [{ 'a': 1 }, { 'a': 1 }]
 *
 * console.log(objects[0] === objects[1]);
 * // => true
 */
function constant(value){return function(){return value}}module.exports=constant},{}],289:[function(require,module,exports){var baseRest=require("./_baseRest"),eq=require("./eq"),isIterateeCall=require("./_isIterateeCall"),keysIn=require("./keysIn");
/** Used for built-in method references. */var objectProto=Object.prototype;
/** Used to check objects for own properties. */var hasOwnProperty=objectProto.hasOwnProperty;
/**
 * Assigns own and inherited enumerable string keyed properties of source
 * objects to the destination object for all destination properties that
 * resolve to `undefined`. Source objects are applied from left to right.
 * Once a property is set, additional values of the same property are ignored.
 *
 * **Note:** This method mutates `object`.
 *
 * @static
 * @since 0.1.0
 * @memberOf _
 * @category Object
 * @param {Object} object The destination object.
 * @param {...Object} [sources] The source objects.
 * @returns {Object} Returns `object`.
 * @see _.defaultsDeep
 * @example
 *
 * _.defaults({ 'a': 1 }, { 'b': 2 }, { 'a': 3 });
 * // => { 'a': 1, 'b': 2 }
 */var defaults=baseRest(function(object,sources){object=Object(object);var index=-1;var length=sources.length;var guard=length>2?sources[2]:undefined;if(guard&&isIterateeCall(sources[0],sources[1],guard)){length=1}while(++index<length){var source=sources[index];var props=keysIn(source);var propsIndex=-1;var propsLength=props.length;while(++propsIndex<propsLength){var key=props[propsIndex];var value=object[key];if(value===undefined||eq(value,objectProto[key])&&!hasOwnProperty.call(object,key)){object[key]=source[key]}}}return object});module.exports=defaults},{"./_baseRest":181,"./_isIterateeCall":242,"./eq":291,"./keysIn":320}],290:[function(require,module,exports){module.exports=require("./forEach")},{"./forEach":296}],291:[function(require,module,exports){
/**
 * Performs a
 * [`SameValueZero`](http://ecma-international.org/ecma-262/7.0/#sec-samevaluezero)
 * comparison between two values to determine if they are equivalent.
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Lang
 * @param {*} value The value to compare.
 * @param {*} other The other value to compare.
 * @returns {boolean} Returns `true` if the values are equivalent, else `false`.
 * @example
 *
 * var object = { 'a': 1 };
 * var other = { 'a': 1 };
 *
 * _.eq(object, object);
 * // => true
 *
 * _.eq(object, other);
 * // => false
 *
 * _.eq('a', 'a');
 * // => true
 *
 * _.eq('a', Object('a'));
 * // => false
 *
 * _.eq(NaN, NaN);
 * // => true
 */
function eq(value,other){return value===other||value!==value&&other!==other}module.exports=eq},{}],292:[function(require,module,exports){var arrayFilter=require("./_arrayFilter"),baseFilter=require("./_baseFilter"),baseIteratee=require("./_baseIteratee"),isArray=require("./isArray");
/**
 * Iterates over elements of `collection`, returning an array of all elements
 * `predicate` returns truthy for. The predicate is invoked with three
 * arguments: (value, index|key, collection).
 *
 * **Note:** Unlike `_.remove`, this method returns a new array.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Collection
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function} [predicate=_.identity] The function invoked per iteration.
 * @returns {Array} Returns the new filtered array.
 * @see _.reject
 * @example
 *
 * var users = [
 *   { 'user': 'barney', 'age': 36, 'active': true },
 *   { 'user': 'fred',   'age': 40, 'active': false }
 * ];
 *
 * _.filter(users, function(o) { return !o.active; });
 * // => objects for ['fred']
 *
 * // The `_.matches` iteratee shorthand.
 * _.filter(users, { 'age': 36, 'active': true });
 * // => objects for ['barney']
 *
 * // The `_.matchesProperty` iteratee shorthand.
 * _.filter(users, ['active', false]);
 * // => objects for ['fred']
 *
 * // The `_.property` iteratee shorthand.
 * _.filter(users, 'active');
 * // => objects for ['barney']
 */function filter(collection,predicate){var func=isArray(collection)?arrayFilter:baseFilter;return func(collection,baseIteratee(predicate,3))}module.exports=filter},{"./_arrayFilter":125,"./_baseFilter":144,"./_baseIteratee":165,"./isArray":303}],293:[function(require,module,exports){var createFind=require("./_createFind"),findIndex=require("./findIndex");
/**
 * Iterates over elements of `collection`, returning the first element
 * `predicate` returns truthy for. The predicate is invoked with three
 * arguments: (value, index|key, collection).
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Collection
 * @param {Array|Object} collection The collection to inspect.
 * @param {Function} [predicate=_.identity] The function invoked per iteration.
 * @param {number} [fromIndex=0] The index to search from.
 * @returns {*} Returns the matched element, else `undefined`.
 * @example
 *
 * var users = [
 *   { 'user': 'barney',  'age': 36, 'active': true },
 *   { 'user': 'fred',    'age': 40, 'active': false },
 *   { 'user': 'pebbles', 'age': 1,  'active': true }
 * ];
 *
 * _.find(users, function(o) { return o.age < 40; });
 * // => object for 'barney'
 *
 * // The `_.matches` iteratee shorthand.
 * _.find(users, { 'age': 1, 'active': true });
 * // => object for 'pebbles'
 *
 * // The `_.matchesProperty` iteratee shorthand.
 * _.find(users, ['active', false]);
 * // => object for 'fred'
 *
 * // The `_.property` iteratee shorthand.
 * _.find(users, 'active');
 * // => object for 'barney'
 */var find=createFind(findIndex);module.exports=find},{"./_createFind":210,"./findIndex":294}],294:[function(require,module,exports){var baseFindIndex=require("./_baseFindIndex"),baseIteratee=require("./_baseIteratee"),toInteger=require("./toInteger");
/* Built-in method references for those with the same name as other `lodash` methods. */var nativeMax=Math.max;
/**
 * This method is like `_.find` except that it returns the index of the first
 * element `predicate` returns truthy for instead of the element itself.
 *
 * @static
 * @memberOf _
 * @since 1.1.0
 * @category Array
 * @param {Array} array The array to inspect.
 * @param {Function} [predicate=_.identity] The function invoked per iteration.
 * @param {number} [fromIndex=0] The index to search from.
 * @returns {number} Returns the index of the found element, else `-1`.
 * @example
 *
 * var users = [
 *   { 'user': 'barney',  'active': false },
 *   { 'user': 'fred',    'active': false },
 *   { 'user': 'pebbles', 'active': true }
 * ];
 *
 * _.findIndex(users, function(o) { return o.user == 'barney'; });
 * // => 0
 *
 * // The `_.matches` iteratee shorthand.
 * _.findIndex(users, { 'user': 'fred', 'active': false });
 * // => 1
 *
 * // The `_.matchesProperty` iteratee shorthand.
 * _.findIndex(users, ['active', false]);
 * // => 0
 *
 * // The `_.property` iteratee shorthand.
 * _.findIndex(users, 'active');
 * // => 2
 */function findIndex(array,predicate,fromIndex){var length=array==null?0:array.length;if(!length){return-1}var index=fromIndex==null?0:toInteger(fromIndex);if(index<0){index=nativeMax(length+index,0)}return baseFindIndex(array,baseIteratee(predicate,3),index)}module.exports=findIndex},{"./_baseFindIndex":145,"./_baseIteratee":165,"./toInteger":340}],295:[function(require,module,exports){var baseFlatten=require("./_baseFlatten");
/**
 * Flattens `array` a single level deep.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Array
 * @param {Array} array The array to flatten.
 * @returns {Array} Returns the new flattened array.
 * @example
 *
 * _.flatten([1, [2, [3, [4]], 5]]);
 * // => [1, 2, [3, [4]], 5]
 */function flatten(array){var length=array==null?0:array.length;return length?baseFlatten(array,1):[]}module.exports=flatten},{"./_baseFlatten":146}],296:[function(require,module,exports){var arrayEach=require("./_arrayEach"),baseEach=require("./_baseEach"),castFunction=require("./_castFunction"),isArray=require("./isArray");
/**
 * Iterates over elements of `collection` and invokes `iteratee` for each element.
 * The iteratee is invoked with three arguments: (value, index|key, collection).
 * Iteratee functions may exit iteration early by explicitly returning `false`.
 *
 * **Note:** As with other "Collections" methods, objects with a "length"
 * property are iterated like arrays. To avoid this behavior use `_.forIn`
 * or `_.forOwn` for object iteration.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @alias each
 * @category Collection
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function} [iteratee=_.identity] The function invoked per iteration.
 * @returns {Array|Object} Returns `collection`.
 * @see _.forEachRight
 * @example
 *
 * _.forEach([1, 2], function(value) {
 *   console.log(value);
 * });
 * // => Logs `1` then `2`.
 *
 * _.forEach({ 'a': 1, 'b': 2 }, function(value, key) {
 *   console.log(key);
 * });
 * // => Logs 'a' then 'b' (iteration order is not guaranteed).
 */function forEach(collection,iteratee){var func=isArray(collection)?arrayEach:baseEach;return func(collection,castFunction(iteratee))}module.exports=forEach},{"./_arrayEach":124,"./_baseEach":142,"./_castFunction":192,"./isArray":303}],297:[function(require,module,exports){var baseFor=require("./_baseFor"),castFunction=require("./_castFunction"),keysIn=require("./keysIn");
/**
 * Iterates over own and inherited enumerable string keyed properties of an
 * object and invokes `iteratee` for each property. The iteratee is invoked
 * with three arguments: (value, key, object). Iteratee functions may exit
 * iteration early by explicitly returning `false`.
 *
 * @static
 * @memberOf _
 * @since 0.3.0
 * @category Object
 * @param {Object} object The object to iterate over.
 * @param {Function} [iteratee=_.identity] The function invoked per iteration.
 * @returns {Object} Returns `object`.
 * @see _.forInRight
 * @example
 *
 * function Foo() {
 *   this.a = 1;
 *   this.b = 2;
 * }
 *
 * Foo.prototype.c = 3;
 *
 * _.forIn(new Foo, function(value, key) {
 *   console.log(key);
 * });
 * // => Logs 'a', 'b', then 'c' (iteration order is not guaranteed).
 */function forIn(object,iteratee){return object==null?object:baseFor(object,castFunction(iteratee),keysIn)}module.exports=forIn},{"./_baseFor":147,"./_castFunction":192,"./keysIn":320}],298:[function(require,module,exports){var baseGet=require("./_baseGet");
/**
 * Gets the value at `path` of `object`. If the resolved value is
 * `undefined`, the `defaultValue` is returned in its place.
 *
 * @static
 * @memberOf _
 * @since 3.7.0
 * @category Object
 * @param {Object} object The object to query.
 * @param {Array|string} path The path of the property to get.
 * @param {*} [defaultValue] The value returned for `undefined` resolved values.
 * @returns {*} Returns the resolved value.
 * @example
 *
 * var object = { 'a': [{ 'b': { 'c': 3 } }] };
 *
 * _.get(object, 'a[0].b.c');
 * // => 3
 *
 * _.get(object, ['a', '0', 'b', 'c']);
 * // => 3
 *
 * _.get(object, 'a.b.c', 'default');
 * // => 'default'
 */function get(object,path,defaultValue){var result=object==null?undefined:baseGet(object,path);return result===undefined?defaultValue:result}module.exports=get},{"./_baseGet":149}],299:[function(require,module,exports){var baseHas=require("./_baseHas"),hasPath=require("./_hasPath");
/**
 * Checks if `path` is a direct property of `object`.
 *
 * @static
 * @since 0.1.0
 * @memberOf _
 * @category Object
 * @param {Object} object The object to query.
 * @param {Array|string} path The path to check.
 * @returns {boolean} Returns `true` if `path` exists, else `false`.
 * @example
 *
 * var object = { 'a': { 'b': 2 } };
 * var other = _.create({ 'a': _.create({ 'b': 2 }) });
 *
 * _.has(object, 'a');
 * // => true
 *
 * _.has(object, 'a.b');
 * // => true
 *
 * _.has(object, ['a', 'b']);
 * // => true
 *
 * _.has(other, 'a');
 * // => false
 */function has(object,path){return object!=null&&hasPath(object,path,baseHas)}module.exports=has},{"./_baseHas":153,"./_hasPath":230}],300:[function(require,module,exports){var baseHasIn=require("./_baseHasIn"),hasPath=require("./_hasPath");
/**
 * Checks if `path` is a direct or inherited property of `object`.
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Object
 * @param {Object} object The object to query.
 * @param {Array|string} path The path to check.
 * @returns {boolean} Returns `true` if `path` exists, else `false`.
 * @example
 *
 * var object = _.create({ 'a': _.create({ 'b': 2 }) });
 *
 * _.hasIn(object, 'a');
 * // => true
 *
 * _.hasIn(object, 'a.b');
 * // => true
 *
 * _.hasIn(object, ['a', 'b']);
 * // => true
 *
 * _.hasIn(object, 'b');
 * // => false
 */function hasIn(object,path){return object!=null&&hasPath(object,path,baseHasIn)}module.exports=hasIn},{"./_baseHasIn":154,"./_hasPath":230}],301:[function(require,module,exports){
/**
 * This method returns the first argument it receives.
 *
 * @static
 * @since 0.1.0
 * @memberOf _
 * @category Util
 * @param {*} value Any value.
 * @returns {*} Returns `value`.
 * @example
 *
 * var object = { 'a': 1 };
 *
 * console.log(_.identity(object) === object);
 * // => true
 */
function identity(value){return value}module.exports=identity},{}],302:[function(require,module,exports){var baseIsArguments=require("./_baseIsArguments"),isObjectLike=require("./isObjectLike");
/** Used for built-in method references. */var objectProto=Object.prototype;
/** Used to check objects for own properties. */var hasOwnProperty=objectProto.hasOwnProperty;
/** Built-in value references. */var propertyIsEnumerable=objectProto.propertyIsEnumerable;
/**
 * Checks if `value` is likely an `arguments` object.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is an `arguments` object,
 *  else `false`.
 * @example
 *
 * _.isArguments(function() { return arguments; }());
 * // => true
 *
 * _.isArguments([1, 2, 3]);
 * // => false
 */var isArguments=baseIsArguments(function(){return arguments}())?baseIsArguments:function(value){return isObjectLike(value)&&hasOwnProperty.call(value,"callee")&&!propertyIsEnumerable.call(value,"callee")};module.exports=isArguments},{"./_baseIsArguments":156,"./isObjectLike":312}],303:[function(require,module,exports){
/**
 * Checks if `value` is classified as an `Array` object.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is an array, else `false`.
 * @example
 *
 * _.isArray([1, 2, 3]);
 * // => true
 *
 * _.isArray(document.body.children);
 * // => false
 *
 * _.isArray('abc');
 * // => false
 *
 * _.isArray(_.noop);
 * // => false
 */
var isArray=Array.isArray;module.exports=isArray},{}],304:[function(require,module,exports){var isFunction=require("./isFunction"),isLength=require("./isLength");
/**
 * Checks if `value` is array-like. A value is considered array-like if it's
 * not a function and has a `value.length` that's an integer greater than or
 * equal to `0` and less than or equal to `Number.MAX_SAFE_INTEGER`.
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is array-like, else `false`.
 * @example
 *
 * _.isArrayLike([1, 2, 3]);
 * // => true
 *
 * _.isArrayLike(document.body.children);
 * // => true
 *
 * _.isArrayLike('abc');
 * // => true
 *
 * _.isArrayLike(_.noop);
 * // => false
 */function isArrayLike(value){return value!=null&&isLength(value.length)&&!isFunction(value)}module.exports=isArrayLike},{"./isFunction":308,"./isLength":309}],305:[function(require,module,exports){var isArrayLike=require("./isArrayLike"),isObjectLike=require("./isObjectLike");
/**
 * This method is like `_.isArrayLike` except that it also checks if `value`
 * is an object.
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is an array-like object,
 *  else `false`.
 * @example
 *
 * _.isArrayLikeObject([1, 2, 3]);
 * // => true
 *
 * _.isArrayLikeObject(document.body.children);
 * // => true
 *
 * _.isArrayLikeObject('abc');
 * // => false
 *
 * _.isArrayLikeObject(_.noop);
 * // => false
 */function isArrayLikeObject(value){return isObjectLike(value)&&isArrayLike(value)}module.exports=isArrayLikeObject},{"./isArrayLike":304,"./isObjectLike":312}],306:[function(require,module,exports){var root=require("./_root"),stubFalse=require("./stubFalse");
/** Detect free variable `exports`. */var freeExports=typeof exports=="object"&&exports&&!exports.nodeType&&exports;
/** Detect free variable `module`. */var freeModule=freeExports&&typeof module=="object"&&module&&!module.nodeType&&module;
/** Detect the popular CommonJS extension `module.exports`. */var moduleExports=freeModule&&freeModule.exports===freeExports;
/** Built-in value references. */var Buffer=moduleExports?root.Buffer:undefined;
/* Built-in method references for those with the same name as other `lodash` methods. */var nativeIsBuffer=Buffer?Buffer.isBuffer:undefined;
/**
 * Checks if `value` is a buffer.
 *
 * @static
 * @memberOf _
 * @since 4.3.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a buffer, else `false`.
 * @example
 *
 * _.isBuffer(new Buffer(2));
 * // => true
 *
 * _.isBuffer(new Uint8Array(2));
 * // => false
 */var isBuffer=nativeIsBuffer||stubFalse;module.exports=isBuffer},{"./_root":268,"./stubFalse":338}],307:[function(require,module,exports){var baseKeys=require("./_baseKeys"),getTag=require("./_getTag"),isArguments=require("./isArguments"),isArray=require("./isArray"),isArrayLike=require("./isArrayLike"),isBuffer=require("./isBuffer"),isPrototype=require("./_isPrototype"),isTypedArray=require("./isTypedArray");
/** `Object#toString` result references. */var mapTag="[object Map]",setTag="[object Set]";
/** Used for built-in method references. */var objectProto=Object.prototype;
/** Used to check objects for own properties. */var hasOwnProperty=objectProto.hasOwnProperty;
/**
 * Checks if `value` is an empty object, collection, map, or set.
 *
 * Objects are considered empty if they have no own enumerable string keyed
 * properties.
 *
 * Array-like values such as `arguments` objects, arrays, buffers, strings, or
 * jQuery-like collections are considered empty if they have a `length` of `0`.
 * Similarly, maps and sets are considered empty if they have a `size` of `0`.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is empty, else `false`.
 * @example
 *
 * _.isEmpty(null);
 * // => true
 *
 * _.isEmpty(true);
 * // => true
 *
 * _.isEmpty(1);
 * // => true
 *
 * _.isEmpty([1, 2, 3]);
 * // => false
 *
 * _.isEmpty({ 'a': 1 });
 * // => false
 */function isEmpty(value){if(value==null){return true}if(isArrayLike(value)&&(isArray(value)||typeof value=="string"||typeof value.splice=="function"||isBuffer(value)||isTypedArray(value)||isArguments(value))){return!value.length}var tag=getTag(value);if(tag==mapTag||tag==setTag){return!value.size}if(isPrototype(value)){return!baseKeys(value).length}for(var key in value){if(hasOwnProperty.call(value,key)){return false}}return true}module.exports=isEmpty},{"./_baseKeys":166,"./_getTag":228,"./_isPrototype":246,"./isArguments":302,"./isArray":303,"./isArrayLike":304,"./isBuffer":306,"./isTypedArray":317}],308:[function(require,module,exports){var baseGetTag=require("./_baseGetTag"),isObject=require("./isObject");
/** `Object#toString` result references. */var asyncTag="[object AsyncFunction]",funcTag="[object Function]",genTag="[object GeneratorFunction]",proxyTag="[object Proxy]";
/**
 * Checks if `value` is classified as a `Function` object.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a function, else `false`.
 * @example
 *
 * _.isFunction(_);
 * // => true
 *
 * _.isFunction(/abc/);
 * // => false
 */function isFunction(value){if(!isObject(value)){return false}
// The use of `Object#toString` avoids issues with the `typeof` operator
// in Safari 9 which returns 'object' for typed arrays and other constructors.
var tag=baseGetTag(value);return tag==funcTag||tag==genTag||tag==asyncTag||tag==proxyTag}module.exports=isFunction},{"./_baseGetTag":151,"./isObject":311}],309:[function(require,module,exports){
/** Used as references for various `Number` constants. */
var MAX_SAFE_INTEGER=9007199254740991;
/**
 * Checks if `value` is a valid array-like length.
 *
 * **Note:** This method is loosely based on
 * [`ToLength`](http://ecma-international.org/ecma-262/7.0/#sec-tolength).
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a valid length, else `false`.
 * @example
 *
 * _.isLength(3);
 * // => true
 *
 * _.isLength(Number.MIN_VALUE);
 * // => false
 *
 * _.isLength(Infinity);
 * // => false
 *
 * _.isLength('3');
 * // => false
 */function isLength(value){return typeof value=="number"&&value>-1&&value%1==0&&value<=MAX_SAFE_INTEGER}module.exports=isLength},{}],310:[function(require,module,exports){var baseIsMap=require("./_baseIsMap"),baseUnary=require("./_baseUnary"),nodeUtil=require("./_nodeUtil");
/* Node.js helper references. */var nodeIsMap=nodeUtil&&nodeUtil.isMap;
/**
 * Checks if `value` is classified as a `Map` object.
 *
 * @static
 * @memberOf _
 * @since 4.3.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a map, else `false`.
 * @example
 *
 * _.isMap(new Map);
 * // => true
 *
 * _.isMap(new WeakMap);
 * // => false
 */var isMap=nodeIsMap?baseUnary(nodeIsMap):baseIsMap;module.exports=isMap},{"./_baseIsMap":159,"./_baseUnary":187,"./_nodeUtil":264}],311:[function(require,module,exports){
/**
 * Checks if `value` is the
 * [language type](http://www.ecma-international.org/ecma-262/7.0/#sec-ecmascript-language-types)
 * of `Object`. (e.g. arrays, functions, objects, regexes, `new Number(0)`, and `new String('')`)
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is an object, else `false`.
 * @example
 *
 * _.isObject({});
 * // => true
 *
 * _.isObject([1, 2, 3]);
 * // => true
 *
 * _.isObject(_.noop);
 * // => true
 *
 * _.isObject(null);
 * // => false
 */
function isObject(value){var type=typeof value;return value!=null&&(type=="object"||type=="function")}module.exports=isObject},{}],312:[function(require,module,exports){
/**
 * Checks if `value` is object-like. A value is object-like if it's not `null`
 * and has a `typeof` result of "object".
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is object-like, else `false`.
 * @example
 *
 * _.isObjectLike({});
 * // => true
 *
 * _.isObjectLike([1, 2, 3]);
 * // => true
 *
 * _.isObjectLike(_.noop);
 * // => false
 *
 * _.isObjectLike(null);
 * // => false
 */
function isObjectLike(value){return value!=null&&typeof value=="object"}module.exports=isObjectLike},{}],313:[function(require,module,exports){var baseGetTag=require("./_baseGetTag"),getPrototype=require("./_getPrototype"),isObjectLike=require("./isObjectLike");
/** `Object#toString` result references. */var objectTag="[object Object]";
/** Used for built-in method references. */var funcProto=Function.prototype,objectProto=Object.prototype;
/** Used to resolve the decompiled source of functions. */var funcToString=funcProto.toString;
/** Used to check objects for own properties. */var hasOwnProperty=objectProto.hasOwnProperty;
/** Used to infer the `Object` constructor. */var objectCtorString=funcToString.call(Object);
/**
 * Checks if `value` is a plain object, that is, an object created by the
 * `Object` constructor or one with a `[[Prototype]]` of `null`.
 *
 * @static
 * @memberOf _
 * @since 0.8.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a plain object, else `false`.
 * @example
 *
 * function Foo() {
 *   this.a = 1;
 * }
 *
 * _.isPlainObject(new Foo);
 * // => false
 *
 * _.isPlainObject([1, 2, 3]);
 * // => false
 *
 * _.isPlainObject({ 'x': 0, 'y': 0 });
 * // => true
 *
 * _.isPlainObject(Object.create(null));
 * // => true
 */function isPlainObject(value){if(!isObjectLike(value)||baseGetTag(value)!=objectTag){return false}var proto=getPrototype(value);if(proto===null){return true}var Ctor=hasOwnProperty.call(proto,"constructor")&&proto.constructor;return typeof Ctor=="function"&&Ctor instanceof Ctor&&funcToString.call(Ctor)==objectCtorString}module.exports=isPlainObject},{"./_baseGetTag":151,"./_getPrototype":224,"./isObjectLike":312}],314:[function(require,module,exports){var baseIsSet=require("./_baseIsSet"),baseUnary=require("./_baseUnary"),nodeUtil=require("./_nodeUtil");
/* Node.js helper references. */var nodeIsSet=nodeUtil&&nodeUtil.isSet;
/**
 * Checks if `value` is classified as a `Set` object.
 *
 * @static
 * @memberOf _
 * @since 4.3.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a set, else `false`.
 * @example
 *
 * _.isSet(new Set);
 * // => true
 *
 * _.isSet(new WeakSet);
 * // => false
 */var isSet=nodeIsSet?baseUnary(nodeIsSet):baseIsSet;module.exports=isSet},{"./_baseIsSet":163,"./_baseUnary":187,"./_nodeUtil":264}],315:[function(require,module,exports){var baseGetTag=require("./_baseGetTag"),isArray=require("./isArray"),isObjectLike=require("./isObjectLike");
/** `Object#toString` result references. */var stringTag="[object String]";
/**
 * Checks if `value` is classified as a `String` primitive or object.
 *
 * @static
 * @since 0.1.0
 * @memberOf _
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a string, else `false`.
 * @example
 *
 * _.isString('abc');
 * // => true
 *
 * _.isString(1);
 * // => false
 */function isString(value){return typeof value=="string"||!isArray(value)&&isObjectLike(value)&&baseGetTag(value)==stringTag}module.exports=isString},{"./_baseGetTag":151,"./isArray":303,"./isObjectLike":312}],316:[function(require,module,exports){var baseGetTag=require("./_baseGetTag"),isObjectLike=require("./isObjectLike");
/** `Object#toString` result references. */var symbolTag="[object Symbol]";
/**
 * Checks if `value` is classified as a `Symbol` primitive or object.
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a symbol, else `false`.
 * @example
 *
 * _.isSymbol(Symbol.iterator);
 * // => true
 *
 * _.isSymbol('abc');
 * // => false
 */function isSymbol(value){return typeof value=="symbol"||isObjectLike(value)&&baseGetTag(value)==symbolTag}module.exports=isSymbol},{"./_baseGetTag":151,"./isObjectLike":312}],317:[function(require,module,exports){var baseIsTypedArray=require("./_baseIsTypedArray"),baseUnary=require("./_baseUnary"),nodeUtil=require("./_nodeUtil");
/* Node.js helper references. */var nodeIsTypedArray=nodeUtil&&nodeUtil.isTypedArray;
/**
 * Checks if `value` is classified as a typed array.
 *
 * @static
 * @memberOf _
 * @since 3.0.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a typed array, else `false`.
 * @example
 *
 * _.isTypedArray(new Uint8Array);
 * // => true
 *
 * _.isTypedArray([]);
 * // => false
 */var isTypedArray=nodeIsTypedArray?baseUnary(nodeIsTypedArray):baseIsTypedArray;module.exports=isTypedArray},{"./_baseIsTypedArray":164,"./_baseUnary":187,"./_nodeUtil":264}],318:[function(require,module,exports){
/**
 * Checks if `value` is `undefined`.
 *
 * @static
 * @since 0.1.0
 * @memberOf _
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is `undefined`, else `false`.
 * @example
 *
 * _.isUndefined(void 0);
 * // => true
 *
 * _.isUndefined(null);
 * // => false
 */
function isUndefined(value){return value===undefined}module.exports=isUndefined},{}],319:[function(require,module,exports){var arrayLikeKeys=require("./_arrayLikeKeys"),baseKeys=require("./_baseKeys"),isArrayLike=require("./isArrayLike");
/**
 * Creates an array of the own enumerable property names of `object`.
 *
 * **Note:** Non-object values are coerced to objects. See the
 * [ES spec](http://ecma-international.org/ecma-262/7.0/#sec-object.keys)
 * for more details.
 *
 * @static
 * @since 0.1.0
 * @memberOf _
 * @category Object
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of property names.
 * @example
 *
 * function Foo() {
 *   this.a = 1;
 *   this.b = 2;
 * }
 *
 * Foo.prototype.c = 3;
 *
 * _.keys(new Foo);
 * // => ['a', 'b'] (iteration order is not guaranteed)
 *
 * _.keys('hi');
 * // => ['0', '1']
 */function keys(object){return isArrayLike(object)?arrayLikeKeys(object):baseKeys(object)}module.exports=keys},{"./_arrayLikeKeys":128,"./_baseKeys":166,"./isArrayLike":304}],320:[function(require,module,exports){var arrayLikeKeys=require("./_arrayLikeKeys"),baseKeysIn=require("./_baseKeysIn"),isArrayLike=require("./isArrayLike");
/**
 * Creates an array of the own and inherited enumerable property names of `object`.
 *
 * **Note:** Non-object values are coerced to objects.
 *
 * @static
 * @memberOf _
 * @since 3.0.0
 * @category Object
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of property names.
 * @example
 *
 * function Foo() {
 *   this.a = 1;
 *   this.b = 2;
 * }
 *
 * Foo.prototype.c = 3;
 *
 * _.keysIn(new Foo);
 * // => ['a', 'b', 'c'] (iteration order is not guaranteed)
 */function keysIn(object){return isArrayLike(object)?arrayLikeKeys(object,true):baseKeysIn(object)}module.exports=keysIn},{"./_arrayLikeKeys":128,"./_baseKeysIn":167,"./isArrayLike":304}],321:[function(require,module,exports){
/**
 * Gets the last element of `array`.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Array
 * @param {Array} array The array to query.
 * @returns {*} Returns the last element of `array`.
 * @example
 *
 * _.last([1, 2, 3]);
 * // => 3
 */
function last(array){var length=array==null?0:array.length;return length?array[length-1]:undefined}module.exports=last},{}],322:[function(require,module,exports){var arrayMap=require("./_arrayMap"),baseIteratee=require("./_baseIteratee"),baseMap=require("./_baseMap"),isArray=require("./isArray");
/**
 * Creates an array of values by running each element in `collection` thru
 * `iteratee`. The iteratee is invoked with three arguments:
 * (value, index|key, collection).
 *
 * Many lodash methods are guarded to work as iteratees for methods like
 * `_.every`, `_.filter`, `_.map`, `_.mapValues`, `_.reject`, and `_.some`.
 *
 * The guarded methods are:
 * `ary`, `chunk`, `curry`, `curryRight`, `drop`, `dropRight`, `every`,
 * `fill`, `invert`, `parseInt`, `random`, `range`, `rangeRight`, `repeat`,
 * `sampleSize`, `slice`, `some`, `sortBy`, `split`, `take`, `takeRight`,
 * `template`, `trim`, `trimEnd`, `trimStart`, and `words`
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Collection
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function} [iteratee=_.identity] The function invoked per iteration.
 * @returns {Array} Returns the new mapped array.
 * @example
 *
 * function square(n) {
 *   return n * n;
 * }
 *
 * _.map([4, 8], square);
 * // => [16, 64]
 *
 * _.map({ 'a': 4, 'b': 8 }, square);
 * // => [16, 64] (iteration order is not guaranteed)
 *
 * var users = [
 *   { 'user': 'barney' },
 *   { 'user': 'fred' }
 * ];
 *
 * // The `_.property` iteratee shorthand.
 * _.map(users, 'user');
 * // => ['barney', 'fred']
 */function map(collection,iteratee){var func=isArray(collection)?arrayMap:baseMap;return func(collection,baseIteratee(iteratee,3))}module.exports=map},{"./_arrayMap":129,"./_baseIteratee":165,"./_baseMap":169,"./isArray":303}],323:[function(require,module,exports){var baseAssignValue=require("./_baseAssignValue"),baseForOwn=require("./_baseForOwn"),baseIteratee=require("./_baseIteratee");
/**
 * Creates an object with the same keys as `object` and values generated
 * by running each own enumerable string keyed property of `object` thru
 * `iteratee`. The iteratee is invoked with three arguments:
 * (value, key, object).
 *
 * @static
 * @memberOf _
 * @since 2.4.0
 * @category Object
 * @param {Object} object The object to iterate over.
 * @param {Function} [iteratee=_.identity] The function invoked per iteration.
 * @returns {Object} Returns the new mapped object.
 * @see _.mapKeys
 * @example
 *
 * var users = {
 *   'fred':    { 'user': 'fred',    'age': 40 },
 *   'pebbles': { 'user': 'pebbles', 'age': 1 }
 * };
 *
 * _.mapValues(users, function(o) { return o.age; });
 * // => { 'fred': 40, 'pebbles': 1 } (iteration order is not guaranteed)
 *
 * // The `_.property` iteratee shorthand.
 * _.mapValues(users, 'age');
 * // => { 'fred': 40, 'pebbles': 1 } (iteration order is not guaranteed)
 */function mapValues(object,iteratee){var result={};iteratee=baseIteratee(iteratee,3);baseForOwn(object,function(value,key,object){baseAssignValue(result,key,iteratee(value,key,object))});return result}module.exports=mapValues},{"./_baseAssignValue":139,"./_baseForOwn":148,"./_baseIteratee":165}],324:[function(require,module,exports){var baseExtremum=require("./_baseExtremum"),baseGt=require("./_baseGt"),identity=require("./identity");
/**
 * Computes the maximum value of `array`. If `array` is empty or falsey,
 * `undefined` is returned.
 *
 * @static
 * @since 0.1.0
 * @memberOf _
 * @category Math
 * @param {Array} array The array to iterate over.
 * @returns {*} Returns the maximum value.
 * @example
 *
 * _.max([4, 2, 8, 6]);
 * // => 8
 *
 * _.max([]);
 * // => undefined
 */function max(array){return array&&array.length?baseExtremum(array,identity,baseGt):undefined}module.exports=max},{"./_baseExtremum":143,"./_baseGt":152,"./identity":301}],325:[function(require,module,exports){var MapCache=require("./_MapCache");
/** Error message constants. */var FUNC_ERROR_TEXT="Expected a function";
/**
 * Creates a function that memoizes the result of `func`. If `resolver` is
 * provided, it determines the cache key for storing the result based on the
 * arguments provided to the memoized function. By default, the first argument
 * provided to the memoized function is used as the map cache key. The `func`
 * is invoked with the `this` binding of the memoized function.
 *
 * **Note:** The cache is exposed as the `cache` property on the memoized
 * function. Its creation may be customized by replacing the `_.memoize.Cache`
 * constructor with one whose instances implement the
 * [`Map`](http://ecma-international.org/ecma-262/7.0/#sec-properties-of-the-map-prototype-object)
 * method interface of `clear`, `delete`, `get`, `has`, and `set`.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Function
 * @param {Function} func The function to have its output memoized.
 * @param {Function} [resolver] The function to resolve the cache key.
 * @returns {Function} Returns the new memoized function.
 * @example
 *
 * var object = { 'a': 1, 'b': 2 };
 * var other = { 'c': 3, 'd': 4 };
 *
 * var values = _.memoize(_.values);
 * values(object);
 * // => [1, 2]
 *
 * values(other);
 * // => [3, 4]
 *
 * object.a = 2;
 * values(object);
 * // => [1, 2]
 *
 * // Modify the result cache.
 * values.cache.set(object, ['a', 'b']);
 * values(object);
 * // => ['a', 'b']
 *
 * // Replace `_.memoize.Cache`.
 * _.memoize.Cache = WeakMap;
 */function memoize(func,resolver){if(typeof func!="function"||resolver!=null&&typeof resolver!="function"){throw new TypeError(FUNC_ERROR_TEXT)}var memoized=function(){var args=arguments,key=resolver?resolver.apply(this,args):args[0],cache=memoized.cache;if(cache.has(key)){return cache.get(key)}var result=func.apply(this,args);memoized.cache=cache.set(key,result)||cache;return result};memoized.cache=new(memoize.Cache||MapCache);return memoized}
// Expose `MapCache`.
memoize.Cache=MapCache;module.exports=memoize},{"./_MapCache":115}],326:[function(require,module,exports){var baseMerge=require("./_baseMerge"),createAssigner=require("./_createAssigner");
/**
 * This method is like `_.assign` except that it recursively merges own and
 * inherited enumerable string keyed properties of source objects into the
 * destination object. Source properties that resolve to `undefined` are
 * skipped if a destination value exists. Array and plain object properties
 * are merged recursively. Other objects and value types are overridden by
 * assignment. Source objects are applied from left to right. Subsequent
 * sources overwrite property assignments of previous sources.
 *
 * **Note:** This method mutates `object`.
 *
 * @static
 * @memberOf _
 * @since 0.5.0
 * @category Object
 * @param {Object} object The destination object.
 * @param {...Object} [sources] The source objects.
 * @returns {Object} Returns `object`.
 * @example
 *
 * var object = {
 *   'a': [{ 'b': 2 }, { 'd': 4 }]
 * };
 *
 * var other = {
 *   'a': [{ 'c': 3 }, { 'e': 5 }]
 * };
 *
 * _.merge(object, other);
 * // => { 'a': [{ 'b': 2, 'c': 3 }, { 'd': 4, 'e': 5 }] }
 */var merge=createAssigner(function(object,source,srcIndex){baseMerge(object,source,srcIndex)});module.exports=merge},{"./_baseMerge":172,"./_createAssigner":207}],327:[function(require,module,exports){var baseExtremum=require("./_baseExtremum"),baseLt=require("./_baseLt"),identity=require("./identity");
/**
 * Computes the minimum value of `array`. If `array` is empty or falsey,
 * `undefined` is returned.
 *
 * @static
 * @since 0.1.0
 * @memberOf _
 * @category Math
 * @param {Array} array The array to iterate over.
 * @returns {*} Returns the minimum value.
 * @example
 *
 * _.min([4, 2, 8, 6]);
 * // => 2
 *
 * _.min([]);
 * // => undefined
 */function min(array){return array&&array.length?baseExtremum(array,identity,baseLt):undefined}module.exports=min},{"./_baseExtremum":143,"./_baseLt":168,"./identity":301}],328:[function(require,module,exports){var baseExtremum=require("./_baseExtremum"),baseIteratee=require("./_baseIteratee"),baseLt=require("./_baseLt");
/**
 * This method is like `_.min` except that it accepts `iteratee` which is
 * invoked for each element in `array` to generate the criterion by which
 * the value is ranked. The iteratee is invoked with one argument: (value).
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Math
 * @param {Array} array The array to iterate over.
 * @param {Function} [iteratee=_.identity] The iteratee invoked per element.
 * @returns {*} Returns the minimum value.
 * @example
 *
 * var objects = [{ 'n': 1 }, { 'n': 2 }];
 *
 * _.minBy(objects, function(o) { return o.n; });
 * // => { 'n': 1 }
 *
 * // The `_.property` iteratee shorthand.
 * _.minBy(objects, 'n');
 * // => { 'n': 1 }
 */function minBy(array,iteratee){return array&&array.length?baseExtremum(array,baseIteratee(iteratee,2),baseLt):undefined}module.exports=minBy},{"./_baseExtremum":143,"./_baseIteratee":165,"./_baseLt":168}],329:[function(require,module,exports){
/**
 * This method returns `undefined`.
 *
 * @static
 * @memberOf _
 * @since 2.3.0
 * @category Util
 * @example
 *
 * _.times(2, _.noop);
 * // => [undefined, undefined]
 */
function noop(){
// No operation performed.
}module.exports=noop},{}],330:[function(require,module,exports){var root=require("./_root");
/**
 * Gets the timestamp of the number of milliseconds that have elapsed since
 * the Unix epoch (1 January 1970 00:00:00 UTC).
 *
 * @static
 * @memberOf _
 * @since 2.4.0
 * @category Date
 * @returns {number} Returns the timestamp.
 * @example
 *
 * _.defer(function(stamp) {
 *   console.log(_.now() - stamp);
 * }, _.now());
 * // => Logs the number of milliseconds it took for the deferred invocation.
 */var now=function(){return root.Date.now()};module.exports=now},{"./_root":268}],331:[function(require,module,exports){var basePick=require("./_basePick"),flatRest=require("./_flatRest");
/**
 * Creates an object composed of the picked `object` properties.
 *
 * @static
 * @since 0.1.0
 * @memberOf _
 * @category Object
 * @param {Object} object The source object.
 * @param {...(string|string[])} [paths] The property paths to pick.
 * @returns {Object} Returns the new object.
 * @example
 *
 * var object = { 'a': 1, 'b': '2', 'c': 3 };
 *
 * _.pick(object, ['a', 'c']);
 * // => { 'a': 1, 'c': 3 }
 */var pick=flatRest(function(object,paths){return object==null?{}:basePick(object,paths)});module.exports=pick},{"./_basePick":175,"./_flatRest":217}],332:[function(require,module,exports){var baseProperty=require("./_baseProperty"),basePropertyDeep=require("./_basePropertyDeep"),isKey=require("./_isKey"),toKey=require("./_toKey");
/**
 * Creates a function that returns the value at `path` of a given object.
 *
 * @static
 * @memberOf _
 * @since 2.4.0
 * @category Util
 * @param {Array|string} path The path of the property to get.
 * @returns {Function} Returns the new accessor function.
 * @example
 *
 * var objects = [
 *   { 'a': { 'b': 2 } },
 *   { 'a': { 'b': 1 } }
 * ];
 *
 * _.map(objects, _.property('a.b'));
 * // => [2, 1]
 *
 * _.map(_.sortBy(objects, _.property(['a', 'b'])), 'a.b');
 * // => [1, 2]
 */function property(path){return isKey(path)?baseProperty(toKey(path)):basePropertyDeep(path)}module.exports=property},{"./_baseProperty":177,"./_basePropertyDeep":178,"./_isKey":243,"./_toKey":283}],333:[function(require,module,exports){var createRange=require("./_createRange");
/**
 * Creates an array of numbers (positive and/or negative) progressing from
 * `start` up to, but not including, `end`. A step of `-1` is used if a negative
 * `start` is specified without an `end` or `step`. If `end` is not specified,
 * it's set to `start` with `start` then set to `0`.
 *
 * **Note:** JavaScript follows the IEEE-754 standard for resolving
 * floating-point values which can produce unexpected results.
 *
 * @static
 * @since 0.1.0
 * @memberOf _
 * @category Util
 * @param {number} [start=0] The start of the range.
 * @param {number} end The end of the range.
 * @param {number} [step=1] The value to increment or decrement by.
 * @returns {Array} Returns the range of numbers.
 * @see _.inRange, _.rangeRight
 * @example
 *
 * _.range(4);
 * // => [0, 1, 2, 3]
 *
 * _.range(-4);
 * // => [0, -1, -2, -3]
 *
 * _.range(1, 5);
 * // => [1, 2, 3, 4]
 *
 * _.range(0, 20, 5);
 * // => [0, 5, 10, 15]
 *
 * _.range(0, -4, -1);
 * // => [0, -1, -2, -3]
 *
 * _.range(1, 4, 0);
 * // => [1, 1, 1]
 *
 * _.range(0);
 * // => []
 */var range=createRange();module.exports=range},{"./_createRange":211}],334:[function(require,module,exports){var arrayReduce=require("./_arrayReduce"),baseEach=require("./_baseEach"),baseIteratee=require("./_baseIteratee"),baseReduce=require("./_baseReduce"),isArray=require("./isArray");
/**
 * Reduces `collection` to a value which is the accumulated result of running
 * each element in `collection` thru `iteratee`, where each successive
 * invocation is supplied the return value of the previous. If `accumulator`
 * is not given, the first element of `collection` is used as the initial
 * value. The iteratee is invoked with four arguments:
 * (accumulator, value, index|key, collection).
 *
 * Many lodash methods are guarded to work as iteratees for methods like
 * `_.reduce`, `_.reduceRight`, and `_.transform`.
 *
 * The guarded methods are:
 * `assign`, `defaults`, `defaultsDeep`, `includes`, `merge`, `orderBy`,
 * and `sortBy`
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Collection
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function} [iteratee=_.identity] The function invoked per iteration.
 * @param {*} [accumulator] The initial value.
 * @returns {*} Returns the accumulated value.
 * @see _.reduceRight
 * @example
 *
 * _.reduce([1, 2], function(sum, n) {
 *   return sum + n;
 * }, 0);
 * // => 3
 *
 * _.reduce({ 'a': 1, 'b': 2, 'c': 1 }, function(result, value, key) {
 *   (result[value] || (result[value] = [])).push(key);
 *   return result;
 * }, {});
 * // => { '1': ['a', 'c'], '2': ['b'] } (iteration order is not guaranteed)
 */function reduce(collection,iteratee,accumulator){var func=isArray(collection)?arrayReduce:baseReduce,initAccum=arguments.length<3;return func(collection,baseIteratee(iteratee,4),accumulator,initAccum,baseEach)}module.exports=reduce},{"./_arrayReduce":131,"./_baseEach":142,"./_baseIteratee":165,"./_baseReduce":180,"./isArray":303}],335:[function(require,module,exports){var baseKeys=require("./_baseKeys"),getTag=require("./_getTag"),isArrayLike=require("./isArrayLike"),isString=require("./isString"),stringSize=require("./_stringSize");
/** `Object#toString` result references. */var mapTag="[object Map]",setTag="[object Set]";
/**
 * Gets the size of `collection` by returning its length for array-like
 * values or the number of own enumerable string keyed properties for objects.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Collection
 * @param {Array|Object|string} collection The collection to inspect.
 * @returns {number} Returns the collection size.
 * @example
 *
 * _.size([1, 2, 3]);
 * // => 3
 *
 * _.size({ 'a': 1, 'b': 2 });
 * // => 2
 *
 * _.size('pebbles');
 * // => 7
 */function size(collection){if(collection==null){return 0}if(isArrayLike(collection)){return isString(collection)?stringSize(collection):collection.length}var tag=getTag(collection);if(tag==mapTag||tag==setTag){return collection.size}return baseKeys(collection).length}module.exports=size},{"./_baseKeys":166,"./_getTag":228,"./_stringSize":281,"./isArrayLike":304,"./isString":315}],336:[function(require,module,exports){var baseFlatten=require("./_baseFlatten"),baseOrderBy=require("./_baseOrderBy"),baseRest=require("./_baseRest"),isIterateeCall=require("./_isIterateeCall");
/**
 * Creates an array of elements, sorted in ascending order by the results of
 * running each element in a collection thru each iteratee. This method
 * performs a stable sort, that is, it preserves the original sort order of
 * equal elements. The iteratees are invoked with one argument: (value).
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Collection
 * @param {Array|Object} collection The collection to iterate over.
 * @param {...(Function|Function[])} [iteratees=[_.identity]]
 *  The iteratees to sort by.
 * @returns {Array} Returns the new sorted array.
 * @example
 *
 * var users = [
 *   { 'user': 'fred',   'age': 48 },
 *   { 'user': 'barney', 'age': 36 },
 *   { 'user': 'fred',   'age': 40 },
 *   { 'user': 'barney', 'age': 34 }
 * ];
 *
 * _.sortBy(users, [function(o) { return o.user; }]);
 * // => objects for [['barney', 36], ['barney', 34], ['fred', 48], ['fred', 40]]
 *
 * _.sortBy(users, ['user', 'age']);
 * // => objects for [['barney', 34], ['barney', 36], ['fred', 40], ['fred', 48]]
 */var sortBy=baseRest(function(collection,iteratees){if(collection==null){return[]}var length=iteratees.length;if(length>1&&isIterateeCall(collection,iteratees[0],iteratees[1])){iteratees=[]}else if(length>2&&isIterateeCall(iteratees[0],iteratees[1],iteratees[2])){iteratees=[iteratees[0]]}return baseOrderBy(collection,baseFlatten(iteratees,1),[])});module.exports=sortBy},{"./_baseFlatten":146,"./_baseOrderBy":174,"./_baseRest":181,"./_isIterateeCall":242}],337:[function(require,module,exports){
/**
 * This method returns a new empty array.
 *
 * @static
 * @memberOf _
 * @since 4.13.0
 * @category Util
 * @returns {Array} Returns the new empty array.
 * @example
 *
 * var arrays = _.times(2, _.stubArray);
 *
 * console.log(arrays);
 * // => [[], []]
 *
 * console.log(arrays[0] === arrays[1]);
 * // => false
 */
function stubArray(){return[]}module.exports=stubArray},{}],338:[function(require,module,exports){
/**
 * This method returns `false`.
 *
 * @static
 * @memberOf _
 * @since 4.13.0
 * @category Util
 * @returns {boolean} Returns `false`.
 * @example
 *
 * _.times(2, _.stubFalse);
 * // => [false, false]
 */
function stubFalse(){return false}module.exports=stubFalse},{}],339:[function(require,module,exports){var toNumber=require("./toNumber");
/** Used as references for various `Number` constants. */var INFINITY=1/0,MAX_INTEGER=17976931348623157e292;
/**
 * Converts `value` to a finite number.
 *
 * @static
 * @memberOf _
 * @since 4.12.0
 * @category Lang
 * @param {*} value The value to convert.
 * @returns {number} Returns the converted number.
 * @example
 *
 * _.toFinite(3.2);
 * // => 3.2
 *
 * _.toFinite(Number.MIN_VALUE);
 * // => 5e-324
 *
 * _.toFinite(Infinity);
 * // => 1.7976931348623157e+308
 *
 * _.toFinite('3.2');
 * // => 3.2
 */function toFinite(value){if(!value){return value===0?value:0}value=toNumber(value);if(value===INFINITY||value===-INFINITY){var sign=value<0?-1:1;return sign*MAX_INTEGER}return value===value?value:0}module.exports=toFinite},{"./toNumber":341}],340:[function(require,module,exports){var toFinite=require("./toFinite");
/**
 * Converts `value` to an integer.
 *
 * **Note:** This method is loosely based on
 * [`ToInteger`](http://www.ecma-international.org/ecma-262/7.0/#sec-tointeger).
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Lang
 * @param {*} value The value to convert.
 * @returns {number} Returns the converted integer.
 * @example
 *
 * _.toInteger(3.2);
 * // => 3
 *
 * _.toInteger(Number.MIN_VALUE);
 * // => 0
 *
 * _.toInteger(Infinity);
 * // => 1.7976931348623157e+308
 *
 * _.toInteger('3.2');
 * // => 3
 */function toInteger(value){var result=toFinite(value),remainder=result%1;return result===result?remainder?result-remainder:result:0}module.exports=toInteger},{"./toFinite":339}],341:[function(require,module,exports){var isObject=require("./isObject"),isSymbol=require("./isSymbol");
/** Used as references for various `Number` constants. */var NAN=0/0;
/** Used to match leading and trailing whitespace. */var reTrim=/^\s+|\s+$/g;
/** Used to detect bad signed hexadecimal string values. */var reIsBadHex=/^[-+]0x[0-9a-f]+$/i;
/** Used to detect binary string values. */var reIsBinary=/^0b[01]+$/i;
/** Used to detect octal string values. */var reIsOctal=/^0o[0-7]+$/i;
/** Built-in method references without a dependency on `root`. */var freeParseInt=parseInt;
/**
 * Converts `value` to a number.
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Lang
 * @param {*} value The value to process.
 * @returns {number} Returns the number.
 * @example
 *
 * _.toNumber(3.2);
 * // => 3.2
 *
 * _.toNumber(Number.MIN_VALUE);
 * // => 5e-324
 *
 * _.toNumber(Infinity);
 * // => Infinity
 *
 * _.toNumber('3.2');
 * // => 3.2
 */function toNumber(value){if(typeof value=="number"){return value}if(isSymbol(value)){return NAN}if(isObject(value)){var other=typeof value.valueOf=="function"?value.valueOf():value;value=isObject(other)?other+"":other}if(typeof value!="string"){return value===0?value:+value}value=value.replace(reTrim,"");var isBinary=reIsBinary.test(value);return isBinary||reIsOctal.test(value)?freeParseInt(value.slice(2),isBinary?2:8):reIsBadHex.test(value)?NAN:+value}module.exports=toNumber},{"./isObject":311,"./isSymbol":316}],342:[function(require,module,exports){var copyObject=require("./_copyObject"),keysIn=require("./keysIn");
/**
 * Converts `value` to a plain object flattening inherited enumerable string
 * keyed properties of `value` to own properties of the plain object.
 *
 * @static
 * @memberOf _
 * @since 3.0.0
 * @category Lang
 * @param {*} value The value to convert.
 * @returns {Object} Returns the converted plain object.
 * @example
 *
 * function Foo() {
 *   this.b = 2;
 * }
 *
 * Foo.prototype.c = 3;
 *
 * _.assign({ 'a': 1 }, new Foo);
 * // => { 'a': 1, 'b': 2 }
 *
 * _.assign({ 'a': 1 }, _.toPlainObject(new Foo));
 * // => { 'a': 1, 'b': 2, 'c': 3 }
 */function toPlainObject(value){return copyObject(value,keysIn(value))}module.exports=toPlainObject},{"./_copyObject":203,"./keysIn":320}],343:[function(require,module,exports){var baseToString=require("./_baseToString");
/**
 * Converts `value` to a string. An empty string is returned for `null`
 * and `undefined` values. The sign of `-0` is preserved.
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Lang
 * @param {*} value The value to convert.
 * @returns {string} Returns the converted string.
 * @example
 *
 * _.toString(null);
 * // => ''
 *
 * _.toString(-0);
 * // => '-0'
 *
 * _.toString([1, 2, 3]);
 * // => '1,2,3'
 */function toString(value){return value==null?"":baseToString(value)}module.exports=toString},{"./_baseToString":186}],344:[function(require,module,exports){var arrayEach=require("./_arrayEach"),baseCreate=require("./_baseCreate"),baseForOwn=require("./_baseForOwn"),baseIteratee=require("./_baseIteratee"),getPrototype=require("./_getPrototype"),isArray=require("./isArray"),isBuffer=require("./isBuffer"),isFunction=require("./isFunction"),isObject=require("./isObject"),isTypedArray=require("./isTypedArray");
/**
 * An alternative to `_.reduce`; this method transforms `object` to a new
 * `accumulator` object which is the result of running each of its own
 * enumerable string keyed properties thru `iteratee`, with each invocation
 * potentially mutating the `accumulator` object. If `accumulator` is not
 * provided, a new object with the same `[[Prototype]]` will be used. The
 * iteratee is invoked with four arguments: (accumulator, value, key, object).
 * Iteratee functions may exit iteration early by explicitly returning `false`.
 *
 * @static
 * @memberOf _
 * @since 1.3.0
 * @category Object
 * @param {Object} object The object to iterate over.
 * @param {Function} [iteratee=_.identity] The function invoked per iteration.
 * @param {*} [accumulator] The custom accumulator value.
 * @returns {*} Returns the accumulated value.
 * @example
 *
 * _.transform([2, 3, 4], function(result, n) {
 *   result.push(n *= n);
 *   return n % 2 == 0;
 * }, []);
 * // => [4, 9]
 *
 * _.transform({ 'a': 1, 'b': 2, 'c': 1 }, function(result, value, key) {
 *   (result[value] || (result[value] = [])).push(key);
 * }, {});
 * // => { '1': ['a', 'c'], '2': ['b'] }
 */function transform(object,iteratee,accumulator){var isArr=isArray(object),isArrLike=isArr||isBuffer(object)||isTypedArray(object);iteratee=baseIteratee(iteratee,4);if(accumulator==null){var Ctor=object&&object.constructor;if(isArrLike){accumulator=isArr?new Ctor:[]}else if(isObject(object)){accumulator=isFunction(Ctor)?baseCreate(getPrototype(object)):{}}else{accumulator={}}}(isArrLike?arrayEach:baseForOwn)(object,function(value,index,object){return iteratee(accumulator,value,index,object)});return accumulator}module.exports=transform},{"./_arrayEach":124,"./_baseCreate":141,"./_baseForOwn":148,"./_baseIteratee":165,"./_getPrototype":224,"./isArray":303,"./isBuffer":306,"./isFunction":308,"./isObject":311,"./isTypedArray":317}],345:[function(require,module,exports){var baseFlatten=require("./_baseFlatten"),baseRest=require("./_baseRest"),baseUniq=require("./_baseUniq"),isArrayLikeObject=require("./isArrayLikeObject");
/**
 * Creates an array of unique values, in order, from all given arrays using
 * [`SameValueZero`](http://ecma-international.org/ecma-262/7.0/#sec-samevaluezero)
 * for equality comparisons.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Array
 * @param {...Array} [arrays] The arrays to inspect.
 * @returns {Array} Returns the new array of combined values.
 * @example
 *
 * _.union([2], [1, 2]);
 * // => [2, 1]
 */var union=baseRest(function(arrays){return baseUniq(baseFlatten(arrays,1,isArrayLikeObject,true))});module.exports=union},{"./_baseFlatten":146,"./_baseRest":181,"./_baseUniq":188,"./isArrayLikeObject":305}],346:[function(require,module,exports){var toString=require("./toString");
/** Used to generate unique IDs. */var idCounter=0;
/**
 * Generates a unique ID. If `prefix` is given, the ID is appended to it.
 *
 * @static
 * @since 0.1.0
 * @memberOf _
 * @category Util
 * @param {string} [prefix=''] The value to prefix the ID with.
 * @returns {string} Returns the unique ID.
 * @example
 *
 * _.uniqueId('contact_');
 * // => 'contact_104'
 *
 * _.uniqueId();
 * // => '105'
 */function uniqueId(prefix){var id=++idCounter;return toString(prefix)+id}module.exports=uniqueId},{"./toString":343}],347:[function(require,module,exports){var baseValues=require("./_baseValues"),keys=require("./keys");
/**
 * Creates an array of the own enumerable string keyed property values of `object`.
 *
 * **Note:** Non-object values are coerced to objects.
 *
 * @static
 * @since 0.1.0
 * @memberOf _
 * @category Object
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of property values.
 * @example
 *
 * function Foo() {
 *   this.a = 1;
 *   this.b = 2;
 * }
 *
 * Foo.prototype.c = 3;
 *
 * _.values(new Foo);
 * // => [1, 2] (iteration order is not guaranteed)
 *
 * _.values('hi');
 * // => ['h', 'i']
 */function values(object){return object==null?[]:baseValues(object,keys(object))}module.exports=values},{"./_baseValues":189,"./keys":319}],348:[function(require,module,exports){var assignValue=require("./_assignValue"),baseZipObject=require("./_baseZipObject");
/**
 * This method is like `_.fromPairs` except that it accepts two arrays,
 * one of property identifiers and one of corresponding values.
 *
 * @static
 * @memberOf _
 * @since 0.4.0
 * @category Array
 * @param {Array} [props=[]] The property identifiers.
 * @param {Array} [values=[]] The property values.
 * @returns {Object} Returns the new object.
 * @example
 *
 * _.zipObject(['a', 'b'], [1, 2]);
 * // => { 'a': 1, 'b': 2 }
 */function zipObject(props,values){return baseZipObject(props||[],values||[],assignValue)}module.exports=zipObject},{"./_assignValue":135,"./_baseZipObject":190}]},{},[1])(1)});