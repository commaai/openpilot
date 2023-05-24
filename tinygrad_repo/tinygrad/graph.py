import os, atexit, itertools
try:
  import networkx as nx  # type: ignore
except ImportError:
  nx = None # graph won't work
from collections import defaultdict
from typing import Dict, List, Optional
from tinygrad.ops import DeviceBuffer, UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, FusedOps, Op, OpType, LazyOp, get_buffers, get_lazyops
from tinygrad.helpers import getenv, DEBUG

GRAPH, PRUNEGRAPH, GRAPHPATH = getenv("GRAPH", 0), getenv("PRUNEGRAPH", 0), getenv("GRAPHPATH", "/tmp/net")

# **** debugging and graphing ****

G = nx.DiGraph() if nx is not None else None
cnts : Dict[OpType, int] = defaultdict(int)
if GRAPH:
  def save_graph_exit():
    for k,v in cnts.items(): print(k, v)
    if PRUNEGRAPH: prune_graph()
    print("saving", G)
    nx.drawing.nx_pydot.write_dot(G, f'{GRAPHPATH}.dot')
    # -Gnslimit=100 can make it finish, but you won't like results
    os.system(f'dot -Tsvg {GRAPHPATH}.dot -o {GRAPHPATH}.svg')
  atexit.register(save_graph_exit)

node_count = 0
def nm(x):
  global node_count
  if not hasattr(x, 'node_id'):
    setattr(x, 'node_id', node_count)
    node_count += 1
  return x.node_id

def get_sop(op : List[Op]):
  if len(op) <= 2: return '.'.join([str(y).split(".")[1] for y in op][::-1])
  if len(op) <= 4: return '.'.join([str(y).split(".")[1][0:2] for y in op][::-1])
  return str(len(op))

def log_op(ret : DeviceBuffer, ast : LazyOp, show_graph : Optional[bool] = None):
  if show_graph is None: show_graph = GRAPH
  if not DEBUG and not show_graph: return
  op : List[Op] = [x.op for x in get_lazyops(ast)]
  inp : List[DeviceBuffer] = get_buffers(ast)
  if len(inp) == 1 and inp[0] == ret:
    if show_graph and nm(ret) in G.nodes: G.nodes[nm(ret)]['style'] += ', bold'
    return   # don't log self loops
  oporder = [LoadOps, FusedOps, ReduceOps, BinaryOps, UnaryOps, MovementOps]
  optype = type(sorted(op, key=lambda x: oporder.index(type(x)))[0])
  cnts[optype] += 1
  if DEBUG >= 4: print(f"{op} : {', '.join([f'{x.shape}-<{nm(x)}>' for x in inp])} -> {ret.shape}-<{nm(ret)}>")
  if show_graph:
    top_colors = {LoadOps: '#FFFF80', UnaryOps: "#c0c0c0", ReduceOps: "#8080ff", BinaryOps: "#c0c0c0", MovementOps: "#80ff80", FusedOps: "#ff8080"}
    dashed = (optype == LoadOps and hasattr(ret, "_backing")) or (hasattr(ret, "st") and not ret.st.contiguous)  # type: ignore

    for x in inp:
      G.add_edge(nm(x), nm(ret), label=get_sop(op))
      if 'label' not in G.nodes[nm(x)]:
        G.nodes[nm(x)]['label'] = str(x.shape)
    if nm(ret) not in G.nodes: G.add_node(nm(ret))

    G.nodes[nm(ret)]['label'] = str(set(x.shape for x in inp))+"\n"+str(ret.shape) if optype == ReduceOps else str(ret.shape)
    G.nodes[nm(ret)]['fillcolor'] = (top_colors[optype] + ('80' if dashed else str())) if optype in top_colors else "#ffffff"
    G.nodes[nm(ret)]['style'] = 'filled, dashed' if dashed else 'filled'
    G.nodes[nm(ret)]['prunable'] = optype in [LoadOps, MovementOps]

# prune movementops and loadops
def prune_graph():
  dead_nodes = []
  for n in G.nodes:
    if 'prunable' in G.nodes[n] and G.nodes[n]['prunable']:
      G.add_edges_from([(x, y) for (x,_),(_,y) in itertools.product(G.in_edges(n), G.out_edges(n))])
      dead_nodes.append(n)
  G.remove_nodes_from(dead_nodes)
