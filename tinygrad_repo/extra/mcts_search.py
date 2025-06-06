from __future__ import annotations
from typing import List, Optional, Dict, cast
import numpy as np
np.set_printoptions(suppress=True)
import math, functools, time, random, statistics
from tinygrad.helpers import DEBUG, getenv, CACHELEVEL, diskcache_get, diskcache_put, colored, Profiling
from tinygrad.codegen.kernel import Kernel
from tinygrad.device import Buffer, Device, CompileError
from tinygrad.engine.search import _ensure_buffer_alloc, get_kernel_actions, _time_program

class MCTSNode:
  def __init__(self, kernel:Kernel, parent=None):
    self.kernel:Kernel = kernel
    self.t = math.inf
    self.n = 0
    self.tm = math.inf
    self.i = -1
    self.parents: List[MCTSNode] = [parent] if parent is not None else []
    self.children: Optional[List[MCTSNode]] = None
    self.removed_children: List[MCTSNode] = []

def expand_node(node:MCTSNode):
  assert node.children is None
  node.children = [MCTSNode(x, node) for x in get_kernel_actions(node.kernel, include_0=False).values()]

def remove_node(node:MCTSNode):
  for parent in node.parents:
    assert parent.children is not None
    parent.children.remove(node)
    parent.removed_children.append(node)

C = math.sqrt(2)
TEMP = 0.5
def _sample_tree(node:MCTSNode, best_tm:float) -> MCTSNode:
  if node.children is None or len(node.children) == 0: return node
  unexplored_children = []
  explored_children = []
  ucb_explored_children: List[float] = []
  for child in node.children:
    if child.n == 0: unexplored_children.append(child)
    else:
      ucb = -child.t/best_tm + C*math.sqrt(math.log(node.n)/child.n)
      if not math.isinf(ucb):
        explored_children.append(child)
        ucb_explored_children.append(ucb)
  if len(unexplored_children): return random.choice(unexplored_children)
  if not len(explored_children): return node
  # safe softmax
  ucb_exp = np.exp((np.array(ucb_explored_children)-max(ucb_explored_children))/TEMP)
  return _sample_tree(explored_children[np.random.choice(len(ucb_exp), p=ucb_exp/np.sum(ucb_exp))], best_tm)

# this will expand/remove sometimes
def sample_tree(root:MCTSNode, best_tm:float) -> Optional[MCTSNode]:
  if root.children is None: expand_node(root)
  while root.children:
    # tree traversal
    node = _sample_tree(root, best_tm)

    if node.children is not None and len(node.children) == 0:
      remove_node(node)
      continue

    # node expansion
    if node.n != 0:
      if node.children is None: expand_node(node)
      assert node.children is not None
      if len(node.children) == 0:
        remove_node(node)
        continue
      node = random.choice(node.children)
    return node
  return None

def backprop(bnode:MCTSNode, tm, strength=1.0):
  if bnode.t > tm: bnode.t = tm
  bnode.n += strength
  for parent in bnode.parents: backprop(parent, tm, strength/len(bnode.parents))

graph_mcts_cnt = 0
def mcts_search(lin:Kernel, rawbufs:List[Buffer], amt:int) -> Kernel:
  global graph_mcts_cnt
  # TODO: copied from BEAM
  key = {"ast": lin.ast.key, "amt": amt, "device": lin.opts.device, "suffix": lin.opts.suffix}
  if not getenv("IGNORE_MCTS_CACHE") and CACHELEVEL >= 1 and (val:=diskcache_get("mcts_search", key)) is not None:
    ret = lin.copy()
    for o in val[len(lin.applied_opts):]: ret.apply_opt(o)
    return ret

  rawbufs = _ensure_buffer_alloc(rawbufs)
  var_vals = {k:(k.vmax+k.vmin)//2 for k in lin.ast.variables()}
  dev = Device[lin.opts.device]
  root = MCTSNode(lin)

  st = time.perf_counter()
  best, best_idx, best_tm = lin, 0, math.inf
  seen_libs: Dict[bytes, MCTSNode] = {}
  seen_asts: Dict[bytes, MCTSNode] = {}
  compile_time, runtime_time = 0.0, 0.0
  for i in range(amt):
    node = sample_tree(root, best_tm)  # sample and expand
    if node is None: break  # finished the whole tree
    node.i = i  # when was node explored

    opt_ast = node.kernel.get_optimized_ast()
    if (sibling_node:=seen_asts.get(opt_ast.key, None)) is not None:
      # early check for same optimized AST hit
      remove_node(node)
      tm = sibling_node.t
    else:
      seen_asts[opt_ast.key] = node

      # lowering (50% of the time)
      p = node.kernel.to_program(name_override="test")

      # rollout
      tm1 = time.perf_counter()
      try:
        lib = dev.compiler.compile(p.src)
      except CompileError:
        # NOTE: many of these "compiler errors" are caused by bad code output from the lowerer
        lib = None
      tm2 = time.perf_counter()
      if lib is None:
        tm = math.inf
      else:
        if (sibling_node:=seen_libs.get(lib, None)) is not None:
          # NOTE: these should all be caught by the AST check, need to canonicalize
          # remove this node, it's a duplicate
          remove_node(node)
          tm = sibling_node.t
        else:
          seen_libs[lib] = node
          try: tm = statistics.median(_time_program(p, lib, var_vals, rawbufs, cnt=3, early_stop=best_tm*5/1e6))*1e6
          except RuntimeError: tm = math.inf
          node.tm = tm
      tm3 = time.perf_counter()
      compile_time += tm2-tm1
      runtime_time += tm3-tm2

      # mock rollout
      #node.tm = tm = random.random() + 0.1

    if tm < best_tm: best, best_idx, best_tm = node.kernel, i, tm
    et = time.perf_counter() - st
    if DEBUG>=2: print(f"\r{et:7.2f}s {colored(f'{compile_time*100/et:3.0f}%', 'cyan')} {colored(f'{runtime_time*100/et:3.0f}%', 'red')}: {tm:12.2f} us     best: {best_tm:12.2f} us @ {best_idx+1:4d}      {i+1:4d}/{amt:4d}  {int(round((i+1)/et)):4d}/s     {node.kernel.colored_shape()}\033[K", end="")  # noqa: E501

    # backprop
    backprop(node, tm)
  if DEBUG>=2: print()

  if getenv("MCTSGRAPH"):
    import networkx as nx
    import os
    GRAPHPATH = "/tmp/net"
    def save_graph(G, fn, opt=""):
      print("saving", G, f"to {fn}.svg")
      nx.drawing.nx_pydot.write_dot(G, f'{fn}.dot')
      os.system(f'dot {opt} -Tsvg {fn}.dot -o {fn}.svg')

    G = nx.DiGraph()
    def add_node(node:MCTSNode):
      if node.n == 0: return
      for parent in node.parents: G.add_edge(parent, node)
      gopts = node.kernel.applied_opts
      edge_lbl = f"{str(gopts[-1].op)[7:]} {gopts[-1].axis} {gopts[-1].arg}" if len(gopts) else "ROOT"
      G.add_node(node, label=f"{node.i+1}\n{node.tm:.2f} us\n{edge_lbl}\nt {node.t:.2f}\nn {node.n}",
                 fillcolor="#80ff8080" if node.tm == best_tm else "#ffff8080", style='filled' if node.t == best_tm else '')
      if node.children is not None:
        for child in node.children+node.removed_children: add_node(child)
    add_node(root)
    save_graph(G, f"{GRAPHPATH}.{graph_mcts_cnt}.mcts", '-Grankdir=LR')
    graph_mcts_cnt += 1

  if CACHELEVEL >= 1: diskcache_put("mcts_search", key, best.applied_opts)
  return best
