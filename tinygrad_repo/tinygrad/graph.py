import os
import atexit
import itertools
from collections import defaultdict
from typing import Dict, List
from tinygrad.ops import DeviceBuffer, DEBUG, UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps, LoadOps, Op, OpType

GRAPH = int(os.getenv("GRAPH", "0"))

# **** debugging and graphing ****

cnts : Dict[OpType, int] = defaultdict(int)
if GRAPH:
  import networkx as nx  # type: ignore
  G = nx.DiGraph()
  def save_graph_exit():
    for k,v in cnts.items():
      print(k, v)
    if int(os.getenv("PRUNEGRAPH", "0")):
      dead_nodes = []
      for n in G.nodes:
        # prune movementops and loadops
        if 'fillcolor' in G.nodes[n] and G.nodes[n]['fillcolor'] in ["#80ff8080", "#80ff80", "#FFFF8080", "#FFFF80"]:
          for (x,_),(_,y) in itertools.product(G.in_edges(n), G.out_edges(n)):
            G.add_edge(x, y)
          dead_nodes.append(n)
      for n in dead_nodes:
        G.remove_node(n)
    print("saving", G)
    nx.drawing.nx_pydot.write_dot(G, '/tmp/net.dot')
    # -Gnslimit=100 can make it finish, but you won't like results
    os.system('dot -Tsvg /tmp/net.dot -o /tmp/net.svg')
  atexit.register(save_graph_exit)

global_num_max = 0
def log_op(optype : OpType, op : List[Op], ret : DeviceBuffer, inp : List[DeviceBuffer]):
  cnts[optype] += 1
  if DEBUG >= 3:
    print(f"{op} : {', '.join([str(x.shape) for x in inp])} -> {ret.shape}")
  if GRAPH:
    def nm(x):
      global global_num_max
      if not hasattr(x, 'global_num'):
        setattr(x, 'global_num', global_num_max)
        global_num_max += 1
      return f"<<< {x.global_num} >>>"

    top_colors = {LoadOps: '#FFFF80', UnaryOps: "#c0c0c0", ReduceOps: "#8080ff", BinaryOps: "#c0c0c0", MovementOps: "#80ff80", ProcessingOps: "#ff8080"}
    dashed = (optype == LoadOps and hasattr(ret, "_backing")) or (hasattr(ret, "st") and not ret.st.contiguous)  # type: ignore

    for x in inp:
      if len(op) <= 2:
        sop = '.'.join([str(y).split(".")[1] for y in op][::-1])
      elif len(op) <= 4:
        sop = '.'.join([str(y).split(".")[1][0:2] for y in op][::-1])
      else:
        sop = str(len(op))
      G.add_edge(nm(x), nm(ret), label=sop)
      if 'label' not in G.nodes[nm(x)]:
        G.nodes[nm(x)]['label'] = str(x.shape)
    if nm(ret) not in G.nodes:
      G.add_node(nm(ret))

    if optype == ReduceOps:
      G.nodes[nm(ret)]['label'] = str(set(x.shape for x in inp))+"\n"+str(ret.shape)
    else:
      G.nodes[nm(ret)]['label'] = str(ret.shape)
    G.nodes[nm(ret)]['fillcolor'] = (top_colors[optype] + ('80' if dashed else '')) if optype in top_colors else "#ffffff"
    G.nodes[nm(ret)]['style'] = 'filled, dashed' if dashed else 'filled'
