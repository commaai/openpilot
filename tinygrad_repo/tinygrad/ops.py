from __future__ import annotations
from enum import Enum
from typing import Optional, Tuple, NamedTuple, Union, Any, List, Dict, Type
from copy import copy
import os, sys, functools, itertools, operator, weakref
from tinygrad.helpers import ConvArgs, get_available_llops, prod
from tinygrad.shapetracker import ShapeTracker

# lazy can recurse a lot
sys.setrecursionlimit(10000)

# these are the llops your accelerator must implement, along with toCpu
UnaryOps = Enum("UnaryOps", ["NOOP", "NEG", "RELU", "EXP", "LOG", "SIGN", "RECIPROCAL"])
BinaryOps = Enum("BinaryOps", ["ADD", "SUB", "MUL", "DIV", "POW", "CMPEQ"])
ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
MovementOps = Enum("MovementOps", ["RESHAPE", "PERMUTE", "EXPAND", "FLIP", "STRIDED", "PAD", "SHRINK"])
ProcessingOps = Enum("ProcessingOps", ["CONV"])
LoadOps = Enum("LoadOps", ["FROMCPU", "CONTIGUOUS"])

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps, LoadOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[ProcessingOps], Type[LoadOps]]

DEBUG = int(os.getenv("DEBUG", "0"))
GRAPH = int(os.getenv("GRAPH", "0"))
OPT = int(os.getenv("OPT", "1"))
NOCONV = int(os.getenv("NOCONV", "0"))

# TODO: movement ops that only change shape are really nops. treat them as such
REMOVE_MOVEMENT_NOPS, MERGE_UNARY_OPS, MERGE_ELEMENTWISE_INTO_REDUCE, SHUFFLE_MOVEMENT_OPS = OPT>=1, OPT>=1, OPT>=1, OPT>=1
MERGE_ELEMENTWISE_OPS, MERGE_ONE_REDUCE_INTO_ELEMENTWISE = OPT>=2, OPT>=2
SHUFFLE_PAD_OPS = OPT>=3  # NOTE: 0/0 is NaN if you pad, so this can change the output

# **** enumerate supported devices ****

class Device:
  _buffers, DEFAULT = get_available_llops()
  for name in _buffers.keys():
    vars()[name] = name

# TODO: get device buffer types
DeviceBuffer = Any

# **** debugging and graphing ****

import atexit
from collections import defaultdict
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
      if getattr(x, 'global_num', None) is None:
        setattr(x, 'global_num', global_num_max)
        global_num_max += 1
      return f"<<< {x.global_num} >>>"

    top_colors = {LoadOps: '#FFFF80', UnaryOps: "#c0c0c0", ReduceOps: "#8080ff", BinaryOps: "#c0c0c0", MovementOps: "#80ff80", ProcessingOps: "#ff8080"}
    dashed = (optype == LoadOps and getattr(ret, "_backing", None) is not None) or (getattr(ret, "st", None) is not None and not ret.st.contiguous)

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


# **** realize helpers ****

def _ast(x: Union[LazyBuffer, LazyOp], buf_names: Dict[LazyBuffer, str], code_for_op: Dict[Op, str]) -> str:
  if isinstance(x, LazyBuffer):
    return buf_names[x]
  srcs_code = [_ast(src, buf_names, code_for_op) for src in x.src]
  code = code_for_op[x.op]
  if len(srcs_code) >= 1:
    code = code.replace("A", srcs_code[0])
  if len(srcs_code) >= 2:
    code = code.replace("B", srcs_code[1])
  return code

# **** realize functions ****

def _realize_loadops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], Optional[OpType]]:
  if self.op.op == LoadOps.FROMCPU:
    return Device._buffers[self.device].fromCPU(self.op.arg), [], LoadOps
  elif self.op.op == LoadOps.CONTIGUOUS:
    real_src = self.op.src[0].realize(self.device)
    ret = real_src.contiguous_op()
    return ret, [real_src], LoadOps if ret != real_src else None
  else:
    raise NotImplementedError(f"unknown LoadOp {self.op.op}")

def _realize_movementops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  real_src = self.op.src[0].realize(self.device)
  return real_src.movement_op(self.op.op, self.op.arg), [real_src], MovementOps

# TODO: unify _realize_reduceops, _realize_processingops, and _realize_binaryops
def _realize_reduceops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  # TODO: this can also corealize a binary op after the reduce, not just before
  src = self.op.src[0]
  if MERGE_ELEMENTWISE_INTO_REDUCE and getattr(self.dbuffer, "start_for_op", None) and src.realized is None and src.optype == BinaryOps and len(src.children) <= 1:
    # TODO: this code is (somewhat) repeated in _realize_binaryops
    real_srcs : Dict[LazyBuffer, DeviceBuffer] = {x:x.realize(self.device) for x in get_lazybuffers(src.op)}
    buf_names : Dict[LazyBuffer, str] = {x:f"arg_{i}" for i,x in enumerate(real_srcs.keys())}

    return self.dbuffer(self.shape)._processing_op([(buf_names[lb], db) for lb,db in real_srcs.items()],
      earlycode=_ast(LazyOp(self.op.op, (src.op,), self.op.arg), buf_names, self.dbuffer.code_for_op), earlybufs=buf_names.values(), op=self.op.op), \
      list(real_srcs.values()), ReduceOps
  else:
    real_src = src.realize(self.device)
    return real_src.reduce_op(self.op.op, self.op.arg), [real_src], ReduceOps

def _realize_processingops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  real_src_x, real_src_w = [x.realize(self.device) for x in self.op.src]
  return real_src_x.processing_op(self.op.op, real_src_w, self.op.arg), [real_src_x, real_src_w], ProcessingOps

def _realize_binaryops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  real_srcs : Dict[LazyBuffer, DeviceBuffer] = {x:None for x in get_lazybuffers(self.op)}
  if getattr(self.dbuffer, "_processing_op", None) is not None:
    buf_names : Dict[LazyBuffer, str] = {x:f"arg_{i}" for i,x in enumerate(real_srcs.keys())}
    reduce_shape = (list(real_srcs.keys())[0].shape, list(real_srcs.keys())[0].shape)
    earlycode = "acc"
    conv_args : Optional[ConvArgs] = None

    # if there's *one* processing or reduce op in here, we can corealize it. we can corealize binary op siblings as well
    # NOTE: if it references the same conv multiple times, they should already be merged by the dictionary
    #for k,x in zip(real_srcs.keys(), map(get_movementroot_contiguous, real_srcs.keys())):
    #  print(k,x, len(x.children), [x for x in x.children])
    psrcs : List[Tuple[LazyBuffer, LazyBuffer]] = [(k,x) for k,x in zip(real_srcs.keys(), map(get_movementroot_contiguous, real_srcs.keys())) if x.optype in [ProcessingOps,ReduceOps] and x.realized is None and len(x.children) <= 1 and len(k.children) <= 1]
    if len(psrcs) == 1 and MERGE_ONE_REDUCE_INTO_ELEMENTWISE and (self.device != "OPENCL" or self.shape[-1] == 4):
      if psrcs[0][1].optype == ProcessingOps:
        # TODO: do something similar to what i did with reduceop to use the ast engine?
        # it's hard because conv also has convargs
        conv_args = psrcs[0][1].op.arg
        real_srcs[psrcs[0][1].op.src[0]], real_srcs[psrcs[0][1].op.src[1]] = None, None
        buf_names[psrcs[0][1].op.src[0]], buf_names[psrcs[0][1].op.src[1]] = "input", "weight"   # NOTE: these will not be in the ast
      elif psrcs[0][1].optype == ReduceOps:
        src = psrcs[0][1].op.src[0]
        reduce_shape = (src.shape, psrcs[0][1].shape)

        if MERGE_ELEMENTWISE_INTO_REDUCE and getattr(self.dbuffer, "start_for_op", None) and src.realized is None and src.optype == BinaryOps and len(src.children) <= 1:
          src = src.op
        for i,x in enumerate(get_lazybuffers(src) if isinstance(src, LazyOp) else [src]):
          real_srcs[x] = None
          buf_names[x] = f"earlyarg_{i}"
        earlycode = _ast(LazyOp(psrcs[0][1].op.op, (src,), psrcs[0][1].op.arg), buf_names, self.dbuffer.code_for_op)

      del real_srcs[psrcs[0][0]]
      buf_names[psrcs[0][0]] = "acc"

    for x in real_srcs.keys():
      real_srcs[x] = x.realize(self.device)
    # fast path, no middle buffers
    return self.dbuffer(self.shape)._processing_op([(buf_names[lb], db) for lb,db in real_srcs.items()],
      _ast(self.op, buf_names, self.dbuffer.code_for_op), earlycode=earlycode, earlybufs=set(x for x in buf_names.values() if x.startswith("earlyarg_")),
      C=conv_args, reduce_shape=reduce_shape), \
      list(real_srcs.values()), ProcessingOps if conv_args is not None else (ReduceOps if reduce_shape[0] != reduce_shape[1] else BinaryOps)
  else:
    for x in real_srcs.keys():
      real_srcs[x] = x.realize(self.device)
    # slow path, creates middle buffers
    def ast_eval(x: Union[LazyBuffer, LazyOp]) -> DeviceBuffer:
      if isinstance(x, LazyBuffer):
        return real_srcs[x]
      if x.op in UnaryOps:
        return ast_eval(x.src[0]).unary_op(x.op)
      if x.op in BinaryOps:
        return ast_eval(x.src[0]).binary_op(x.op, ast_eval(x.src[1]))
    return ast_eval(self.op), list(real_srcs.values()), BinaryOps

_realize = {LoadOps:_realize_loadops, ReduceOps:_realize_reduceops, MovementOps:_realize_movementops, BinaryOps:_realize_binaryops, ProcessingOps:_realize_processingops}

# **** lazy operations ****

class LazyOp(NamedTuple):
  op: Op
  src: Tuple[Union[LazyOp, LazyBuffer], ...]  # type: ignore
  arg: Any = None
  # TODO: add dest to support multiple outputs

def get_lazybuffers(op:LazyOp) -> List[LazyBuffer]: return functools.reduce(operator.add, [get_lazybuffers(x) if isinstance(x, LazyOp) else [x] for x in op.src], [])
def get_lazyops(op:LazyOp) -> List[LazyOp]: return functools.reduce(operator.add, [get_lazyops(x) for x in op.src if isinstance(x, LazyOp)], [op])
def get_weakop(op:LazyOp) -> LazyOp: return LazyOp(op.op, tuple(get_weakop(x) if isinstance(x, LazyOp) else weakref.ref(x) for x in op.src), op.arg)
def get_movementroot(root:LazyBuffer) -> LazyBuffer: return get_movementroot(root.op.src[0]) if root.optype == MovementOps and root.realized is None else root
def get_movementroot_contiguous(x:LazyBuffer) -> LazyBuffer: return get_movementroot(x) if x.optype == MovementOps and x.st.contiguous else x

LAZY = int(os.getenv("LAZY", "1"))

class LazyBuffer:
  lazycache : weakref.WeakValueDictionary[LazyOp, LazyBuffer] = weakref.WeakValueDictionary()
  def __new__(cls, device, shape, optype, op):
    # loadops aren't cached
    if optype == LoadOps:
      return super().__new__(cls)
    wop = (device, optype, get_weakop(op))   # NOTE: shape should be deterministic. annoying to cache with the ShapeTracker
    # NOTE: we need "ret" to prevent the new buffer from being immediately deleted
    if wop not in LazyBuffer.lazycache:
      LazyBuffer.lazycache[wop] = ret = super().__new__(cls) # noqa: F841, pylint: disable=W0612
    return LazyBuffer.lazycache[wop]

  def __init__(self, device, shape:Union[ShapeTracker, Tuple[int, ...]], optype:OpType, op:LazyOp):
    if getattr(self, 'device', None) is not None:
      return  # cache hit, we return and don't reinit
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape, self.optype, self.op = self.st.shape, optype, op
    self.realized : Optional[DeviceBuffer] = None
    self.device, self.dbuffer = device, Device._buffers[device]
    self.children : weakref.WeakSet[LazyBuffer] = weakref.WeakSet()
    # NOTE: op should be read only after construction of LazyBuffer
    for x in get_lazybuffers(op):
      x.children.add(self)
    if not LAZY:
      self.realize()

  def __repr__(self): return f"<LB {self.shape} op:{self.op.op if self.realized is None else 'realized'}>"

  # this produces a device buffer
  def realize(self:LazyBuffer, required_device=None) -> DeviceBuffer:
    if required_device is not None:
      assert required_device == self.device
    if self.realized is None:
      # we haven't realized the Buffer yet
      self.realized, real_srcs, real_type = _realize[self.optype](self)
      # in lazy mode, we don't log until we realize
      if real_type is not None:
        log_op(real_type, [x.op for x in get_lazyops(self.op)], self.realized, real_srcs)
      # no need to keep the op after realization
      del self.op

    assert self.realized.shape == self.shape
    assert isinstance(self.realized, Device._buffers[self.device])
    return self.realized

  @staticmethod
  def fromCPU(x, device): return LazyBuffer(device, x.shape, LoadOps, LazyOp(LoadOps.FROMCPU, tuple(), x.copy()))
  def toCPU(self): return self.realize().toCPU()

  def unary_op(self:LazyBuffer, op:UnaryOps) -> LazyBuffer: return elementwise_op(op, self)
  def binary_op(self:LazyBuffer, op:BinaryOps, y:LazyBuffer) -> LazyBuffer: return elementwise_op(op, self, y)
  def contiguous_op(self:LazyBuffer) -> LazyBuffer: return LazyBuffer(self.device, self.shape, LoadOps, LazyOp(LoadOps.CONTIGUOUS, (self,)))

  def reduce_op(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[int, ...]) -> LazyBuffer:
    if self.shape == tuple(new_shape):
      return self
    reduce = list(enumerate(zip(self.shape, new_shape)))
    # move the reduce axes to the end
    x = self.movement_op(MovementOps.PERMUTE, [i for i,(s,n) in reduce if s == n] + [i for i,(s,n) in reduce if s != n])
    new_tmp_shape = tuple([n for _,(s,n) in reduce if s == n] + [n for _,(s,n) in reduce if s != n])
    # NOTE: this reshape can only move around 1s
    return LazyBuffer(x.device, new_tmp_shape, ReduceOps, LazyOp(op, (x,), new_tmp_shape)).movement_op(MovementOps.RESHAPE, new_shape)

  # syntactic sugar around PAD and SHRINK
  # TODO: turn RESHAPE into EXPAND and CONTRACT (current EXPAND should be REPEAT)
  def slice(self:LazyBuffer, arg):
    padding = [(max(0, -p[0]), max(0, p[1]-self.shape[i])) for i,p in enumerate(arg)]
    return self.movement_op(MovementOps.PAD, padding).movement_op(MovementOps.SHRINK, tuple((p[0] + padding[i][0], p[1] + padding[i][0]) for i,p in enumerate(arg)))

  def movement_op(self:LazyBuffer, op:MovementOps, arg) -> LazyBuffer:
    # TODO: look into why that copy is needed
    arg = tuple(copy(arg))

    # instant nops
    if op in [MovementOps.RESHAPE, MovementOps.EXPAND] and arg == self.shape:
      return self
    if op == MovementOps.PERMUTE and arg == tuple(range(len(self.shape))):
      return self
    if op == MovementOps.SHRINK and arg == tuple((0,i) for i in self.shape):
      return self
    if op == MovementOps.PAD and arg == tuple((0,0) for _ in self.shape):
      return self
    if op == MovementOps.FLIP and all(s == 1 or i not in arg for i,s in enumerate(self.shape)):
      return self

    # two ops in a row is one op
    if op in [MovementOps.RESHAPE, MovementOps.EXPAND, MovementOps.SHRINK] and self.realized is None and self.op.op == op:
      return self.op.src[0].movement_op(op, arg)
    if op == MovementOps.PERMUTE and self.realized is None and self.op.op == op:
      return self.op.src[0].movement_op(op, tuple(self.op.arg[i] for i in arg))
    if op == MovementOps.PAD and self.realized is None and self.op.op == op:
      return self.op.src[0].movement_op(op, tuple((b1+b2, e1+e2) for (b1,e1),(b2,e2) in zip(self.op.arg, arg)))

    # some permutes are actually just reshapes
    if op == MovementOps.PERMUTE and ShapeTracker(self.shape).movement_op(op, arg).contiguous:
      return self.movement_op(MovementOps.RESHAPE, tuple(self.shape[i] for i in arg))

    if SHUFFLE_MOVEMENT_OPS and self.optype == BinaryOps and self.realized is None and len(self.children) == 0 and (SHUFFLE_PAD_OPS or op != MovementOps.PAD) and op not in [MovementOps.EXPAND, MovementOps.STRIDED]:
      # if this MovementOp is being applied to a BinaryOp, apply the MovementOp to all the BinaryOp inputs instead
      def replace_with_movement_op(y:Union[LazyOp, LazyBuffer]) -> LazyBuffer:
        if isinstance(y, LazyBuffer):
          return y.movement_op(op, arg)
        assert y.op in BinaryOps or y.op in UnaryOps
        return elementwise_op(y.op, *[replace_with_movement_op(z) for z in y.src])   # type: ignore
      return replace_with_movement_op(self.op)

    # create the buffer
    ret = LazyBuffer(self.device, ShapeTracker(self.st).movement_op(op, arg), MovementOps, LazyOp(op, (self,), arg))

    # NOTE: if ret is in the cache, it can already be realized
    if REMOVE_MOVEMENT_NOPS and ret.realized is None and self.realized is None and ret.st.contiguous:
      # MovementOps aren't stacked any more, they each have one parent, find the root
      root = get_movementroot(self)
      if root.st.contiguous and root != self and prod(ret.st.shape) == prod(root.shape):
        return root.movement_op(MovementOps.RESHAPE, ret.st.shape) if ret.st.shape != root.shape else root

    return ret

  def processing_op(self:LazyBuffer, op:ProcessingOps, w:LazyBuffer, C:ConvArgs) -> LazyBuffer:
    x = self
    # TODO: fixup C?
    if NOCONV or not getattr(x.dbuffer, "SUPPORTS_PADDING", False):
      x = x.slice(((0, x.shape[0]), (0, x.shape[1]), (-C.py, x.shape[2]+C.py_), (-C.px, x.shape[3]+C.px_)))

    if NOCONV or not getattr(x.dbuffer, "processing_op", False):
      # universal conv, just mul and reduce
      # TODO: is there any way to replace strided with other movement ops?
      x = x.movement_op(MovementOps.STRIDED, (
        (C.bs, C.groups*C.cin*x.shape[2]*x.shape[3]), (C.groups, C.cin*x.shape[2]*x.shape[3]),
        (C.rcout, 0), (C.oy, C.sy*x.shape[3]), (C.ox, C.sx),
        (C.cin, x.shape[2]*x.shape[3]), (C.H, C.dy*x.shape[3]), (C.W, C.dx)))
      w = w.movement_op(MovementOps.RESHAPE, (1, C.groups, C.rcout, 1, 1, C.cin, C.H, C.W)) \
           .movement_op(MovementOps.EXPAND, (C.bs, C.groups, C.rcout, C.oy, C.ox, C.cin, C.H, C.W))
      #print(x.st.views, w.st.views)
      return x.binary_op(BinaryOps.MUL, w).reduce_op(ReduceOps.SUM, (C.bs, C.groups, C.rcout, C.oy, C.ox, 1, 1, 1)) \
                                          .movement_op(MovementOps.RESHAPE, (C.bs, C.cout, C.oy, C.ox))
    elif x.device == "OPENCL":
      # TODO: these can be properties on the device buffer
      from accel.opencl.preprocessing import preprocessing_op, postprocessing_op  # type: ignore
      x,w,Cn = preprocessing_op(x, w, C)
      ret = LazyBuffer(x.device, Cn.out_shape, ProcessingOps, LazyOp(op, (x, w), Cn))
      return postprocessing_op(ret, Cn, C)
    else:
      return LazyBuffer(x.device, C.out_shape, ProcessingOps, LazyOp(op, (x, w), C))

def elementwise_op(op:Union[UnaryOps, BinaryOps], *srcs:LazyBuffer) -> LazyBuffer:
  out_device, out_shape = srcs[0].device, srcs[0].shape

  if MERGE_ELEMENTWISE_OPS or (MERGE_UNARY_OPS and len(set(srcs)) == 1):
    # remove the buffers from any (childless) BinaryOps that feed into this
    srcs = tuple(x.op if x.optype == BinaryOps and len(x.children) == 0 and x.realized is None else x for x in srcs)  # type: ignore

  return LazyBuffer(out_device, out_shape, BinaryOps, LazyOp(op, srcs))
