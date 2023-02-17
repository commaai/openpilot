from __future__ import annotations
from typing import Optional, Tuple, Union, List, Dict
from copy import copy
import os, sys, weakref
from tinygrad.helpers import ConvArgs, get_available_llops, prod
from tinygrad.shape import ShapeTracker
from tinygrad.ops import DeviceBuffer, UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps, LoadOps, OpType, LazyOp, get_buffers, get_lazyops, DEBUG
from tinygrad.graph import log_op

# lazy can recurse a lot
sys.setrecursionlimit(10000)

OPT = int(os.getenv("OPT", "2"))
NOCONV = int(os.getenv("NOCONV", "0"))
IMAGE = int(os.getenv("IMAGE", "0"))

# TODO: movement ops that only change shape are really nops. treat them as such
REMOVE_MOVEMENT_NOPS, MERGE_UNARY_OPS, MERGE_ELEMENTWISE_INTO_REDUCE, SHUFFLE_MOVEMENT_OPS = OPT>=1, OPT>=1, OPT>=1, OPT>=1
MERGE_ELEMENTWISE_OPS, MERGE_ONE_REDUCE_INTO_ELEMENTWISE = OPT>=2, OPT>=2
SHUFFLE_PAD_OPS = OPT>=3  # NOTE: 0/0 is NaN if you pad, so this can change the output

# **** enumerate supported devices ****

class Device:
  _buffers, DEFAULT = get_available_llops()
  for name in _buffers.keys():
    vars()[name] = name

# **** realize helpers ****
def realize_buffers(real_srcs, x:LazyOp) -> LazyOp:
  if x in real_srcs:
    return realize_buffers(real_srcs, real_srcs[x]) if isinstance(real_srcs[x], LazyOp) else real_srcs[x]
  return LazyOp(x.op, tuple(realize_buffers(real_srcs, y) for y in x.src), x.arg)

# **** realize functions ****
# TODO: make all _realize functions return an AST, perhaps unrealized

def _realize_loadops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], Optional[OpType]]:
  if self.op.op == LoadOps.FROMCPU:
    return Device._buffers[self.device].fromCPU(self.op.arg), [], LoadOps
  elif self.op.op == LoadOps.CONTIGUOUS:
    real_src = self.op.src[0].realize(self.device)
    ret = real_src.contiguous()
    return ret, [real_src], LoadOps if id(ret) != id(real_src) else None
  else:
    raise NotImplementedError(f"unknown LoadOp {self.op.op}")

def _realize_movementops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  src = self.op.src[0]

  # fuse RESHAPE and ReduceOps
  if src.realized is None and src.optype == ReduceOps and self.op.op == MovementOps.RESHAPE and len(src.children) <= 1:
    return _realize_reduceops_w_shape(src, output_shape = self.op.arg)

  real_src = src.realize(self.device)
  return real_src.movement_op(self.op.op, self.op.arg), [real_src], MovementOps

def _realize_processingops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  real_src_x, real_src_w = [x.realize(self.device) for x in self.op.src]
  return real_src_x.processing_op(self.op.op, real_src_w, self.op.arg), [real_src_x, real_src_w], ProcessingOps

# this supports late merging an upstream Elementwise op
def _realize_reduceops_w_shape(self:LazyBuffer, output_shape=None) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  # TODO: this can also corealize a binary op after the reduce, not just before
  src = self.op.src[0]
  if MERGE_ELEMENTWISE_INTO_REDUCE and src.realized is None and src.optype == BinaryOps and len(src.children) <= 1:
    # this is the new version, deprecate _processing_op
    real_srcs : Dict[LazyBuffer, DeviceBuffer] = {x:x.realize(self.device) for x in get_buffers(src.op)}
    ast = LazyOp(self.op.op, (realize_buffers(real_srcs, src.op),), self.op.arg)
  else:
    real_src = src.realize(self.device)
    real_srcs = {src:real_src}
    ast = LazyOp(self.op.op, (real_src,), self.op.arg)
  if output_shape is not None: ast = LazyOp(MovementOps.RESHAPE, (ast, ), output_shape)
  return self.dbuffer.exec_ast(ast), list(real_srcs.values()), ReduceOps
def _realize_reduceops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]: return _realize_reduceops_w_shape(self)

# this supports late merging an upstream Reduce op and even an Elementwise op above that
def _realize_binaryops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  real_srcs : Dict[LazyBuffer, Union[None, LazyOp, DeviceBuffer]] = {x:None for x in get_buffers(self.op)}
  op_type : OpType = BinaryOps
  if DEBUG >= 3:
    for k,x in zip(real_srcs.keys(), map(get_movementroot_contiguous, real_srcs.keys())):
      if x.optype in [ProcessingOps,ReduceOps] and x.realized is None:
        print("\nHIT", k,x)
        for tk in k.children: print("k", tk)
        for tx in x.children: print("x", tx)
  # NOTE: contiguous does not always mean the same size with SHRINK. this is still mergeable but requires more thought how
  psrcs : List[Tuple[LazyBuffer, LazyBuffer]] = [(k,x) for k,x in zip(real_srcs.keys(), map(get_movementroot_contiguous, real_srcs.keys())) if x.optype in [ProcessingOps,ReduceOps] and x.realized is None and prod(k.shape) == prod(x.shape) and len(x.children) <= 1 and len(k.children) <= 1]
  intermediate_shape = self.shape
  if len(psrcs) == 1 and MERGE_ONE_REDUCE_INTO_ELEMENTWISE and (self.device != "OPENCL" or self.shape[-1] == 4):
    if psrcs[0][1].optype == ProcessingOps:
      real_srcs[psrcs[0][0]] = psrcs[0][1].op
      for x in psrcs[0][1].op.src:
        real_srcs[x] = x.realize(self.device)
      op_type = ProcessingOps
    elif psrcs[0][1].optype == ReduceOps:
      src = psrcs[0][1].op.src[0]
      if MERGE_ELEMENTWISE_INTO_REDUCE and src.realized is None and src.optype == BinaryOps and len(src.children) <= 1:
        src = src.op
      real_srcs[psrcs[0][0]] = LazyOp(psrcs[0][1].op.op, (src,), psrcs[0][1].op.arg)
      for x in get_buffers(real_srcs[psrcs[0][0]]):  # type: ignore
        # these are the early buffers
        real_srcs[x] = x.realize(self.device)
      op_type = ReduceOps
    # if the ReduceOp is followed by a reshape, we push this reshape before all the ElementwiseOp inputs
    if psrcs[0][0].shape != psrcs[0][1].shape:
      intermediate_shape = psrcs[0][1].shape
      assert psrcs[0][0].shape == self.shape, f"shape mismatch {psrcs[0][0].shape} != {self.shape}"

  # reshape all the late ops into the output shape
  # NOTE: these RESHAPEs will return self if they don't change the shape
  for x in real_srcs.keys():
    if real_srcs[x] is None:
      real_srcs[x] = x.movement_op(MovementOps.RESHAPE, intermediate_shape).realize(self.device)
  ast = LazyOp(MovementOps.RESHAPE, (realize_buffers(real_srcs, self.op), ), self.shape)
  ret = self.dbuffer.exec_ast(ast)
  return ret, [x for x in real_srcs.values() if not isinstance(x, LazyOp) and x is not None], op_type

_realize = {LoadOps:_realize_loadops, ReduceOps:_realize_reduceops, MovementOps:_realize_movementops, BinaryOps:_realize_binaryops, ProcessingOps:_realize_processingops}

# **** lazy operations ****

def get_weakop(op:LazyOp) -> LazyOp: return LazyOp(op.op, tuple(get_weakop(x) if isinstance(x, LazyOp) else weakref.ref(x) for x in op.src), op.arg)
def get_movementroot(root:LazyBuffer) -> LazyBuffer: return get_movementroot(root.op.src[0]) if root.realized is None and (root.optype == MovementOps or (root.op.op == LoadOps.CONTIGUOUS and root.op.src[0].st.contiguous)) else root
def get_movementroot_contiguous(x:LazyBuffer) -> LazyBuffer: return get_movementroot(x) if x.optype == MovementOps and x.st.contiguous else x

LAZY = int(os.getenv("LAZY", "1"))

class LazyBuffer:
  lazycache : weakref.WeakValueDictionary[LazyOp, LazyBuffer] = weakref.WeakValueDictionary()
  def __new__(cls, device, shape, optype, op):
    # fromcpu aren't cached
    if optype == LoadOps and op.op == LoadOps.FROMCPU:
      return super().__new__(cls)
    wop = (device, optype, get_weakop(op))   # NOTE: shape should be deterministic. annoying to cache with the ShapeTracker
    # NOTE: we need "ret" to prevent the new buffer from being immediately deleted
    if wop not in LazyBuffer.lazycache:
      LazyBuffer.lazycache[wop] = ret = super().__new__(cls) # noqa: F841, pylint: disable=W0612
    return LazyBuffer.lazycache[wop]

  def __init__(self, device, shape:Union[ShapeTracker, Tuple[int, ...]], optype:OpType, op:LazyOp):
    if hasattr(self, 'device'):
      return  # cache hit, we return and don't reinit
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape, self.optype, self.op = self.st.shape, optype, op
    self.realized : Optional[DeviceBuffer] = None
    self.device, self.dbuffer = device, Device._buffers[device]
    self.children : weakref.WeakSet[LazyBuffer] = weakref.WeakSet()
    # NOTE: op should be read only after construction of LazyBuffer
    for x in get_buffers(op):
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
  def contiguous(self:LazyBuffer) -> LazyBuffer: return LazyBuffer(self.device, self.shape, LoadOps, LazyOp(LoadOps.CONTIGUOUS, (self,)))

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
    local_st = ShapeTracker(self.shape).movement_op(op, arg)

    # instant nops
    if local_st.contiguous and self.shape == local_st.shape and op != MovementOps.STRIDED:
      return self

    # two ops in a row is one op. merge them if unresolved
    if self.realized is None and self.op.op == op:
      if op in [MovementOps.RESHAPE, MovementOps.EXPAND, MovementOps.SHRINK]:
        return self.op.src[0].movement_op(op, arg)
      if op == MovementOps.PERMUTE:
        return self.op.src[0].movement_op(op, tuple(self.op.arg[i] for i in arg))
      if op == MovementOps.PAD:
        return self.op.src[0].movement_op(op, tuple((b1+b2, e1+e2) for (b1,e1),(b2,e2) in zip(self.op.arg, arg)))
      # TODO: MovementOps.FLIP / MovementOps.STRIDED?

    # some permutes are actually just reshapes
    if op == MovementOps.PERMUTE and local_st.contiguous:
      return self.movement_op(MovementOps.RESHAPE, tuple(self.shape[i] for i in arg))

    # some strideds are actually just reshapes
    # NOTE: due to how strided works, we have to check the parent to be contiguous also
    if op == MovementOps.STRIDED and local_st.contiguous and self.st.contiguous:
      return self.movement_op(MovementOps.RESHAPE, tuple(i for i,_ in arg))

    # if this MovementOp is being applied to a BinaryOp, apply the MovementOp to all the BinaryOp inputs instead
    if SHUFFLE_MOVEMENT_OPS and self.optype == BinaryOps and self.realized is None and len(self.children) == 0 and (SHUFFLE_PAD_OPS or op != MovementOps.PAD) and op not in [MovementOps.EXPAND, MovementOps.STRIDED]:
      def replace_with_movement_op(y:Union[LazyOp, LazyBuffer]) -> LazyBuffer:
        if isinstance(y, LazyBuffer):
          return y.movement_op(op, arg)
        assert y.op in BinaryOps or y.op in UnaryOps
        return elementwise_op(y.op, *[replace_with_movement_op(z) for z in y.src])   # type: ignore
      return replace_with_movement_op(self.op)

    # create the buffer
    ret = LazyBuffer(self.device, ShapeTracker(self.st).movement_op(op, arg), MovementOps, LazyOp(op, (self,), arg))

    # if the ShapeTracker becomes contiguous, replace the whole thing with a reshape (or nothing if shapes match)
    # NOTE: if ret is in the cache, it can already be realized
    if REMOVE_MOVEMENT_NOPS and ret.realized is None and self.realized is None and ret.st.contiguous:
      # MovementOps aren't stacked any more, they each have one parent, find the root
      root = get_movementroot(self)
      if root.st.contiguous and root != self and prod(ret.st.shape) == prod(root.shape):
        return root.movement_op(MovementOps.RESHAPE, ret.st.shape) if ret.st.shape != root.shape else root

    return ret

  def processing_op(self:LazyBuffer, op:ProcessingOps, w:LazyBuffer, C:ConvArgs) -> LazyBuffer:
    x = self

    if IMAGE >= 1:
      from accel.opencl.preprocessing import preprocessing_op, postprocessing_op  # type: ignore
      Cold = C
      x,w,C = preprocessing_op(x, w, Cold, False)

      # set up the conv
      # (C.bs*C.iy, C.ix*C.groups*C.cin//4, 4)
      x = x.movement_op(MovementOps.RESHAPE, (C.bs, C.iy, C.ix, C.groups, C.cin))
      # padding (implicit is fine in image)
      x = x.slice(((0, x.shape[0]), (-C.py, x.shape[1]+C.py_), (-C.px, x.shape[2]+C.px_), (0, x.shape[3]), (0, x.shape[4])))

      x = x.movement_op(MovementOps.STRIDED, (
        (C.bs, x.shape[1]*x.shape[2]*C.groups*C.cin),
        (C.oy, C.sy*x.shape[2]*C.groups*C.cin), (C.ox, C.sx*C.groups*C.cin),
        (C.groups, C.cin), (1, 1), (1, 1),
        (C.H, C.dy*x.shape[2]*C.groups*C.cin), (C.W, C.dx*C.groups*C.cin), (C.cin//4 if C.cin >= 4 else 1, 4), (4 if C.cin >= 4 else 1, 1)
      ))
      x = x.movement_op(MovementOps.EXPAND, (C.bs, C.oy, C.ox, C.groups, C.rcout//4 if C.rcout >= 4 else 1, 4 if C.rcout >= 4 else 1, C.H, C.W, C.cin//4 if C.cin >= 4 else 1, 4 if C.cin >= 4 else 1))
      x = x.movement_op(MovementOps.RESHAPE, (C.bs, C.oy, C.ox, C.cout//4, 4, C.H, C.W, C.cin//4 if C.cin >= 4 else 1, 4 if C.cin >= 4 else 1))

      # set up the weights
      if C.cin == 1:
        # depthwise
        w = w.movement_op(MovementOps.RESHAPE, (C.cout//4, C.H, C.W, 4))
        w = w.movement_op(MovementOps.PERMUTE, (0,3,1,2))
        w = w.movement_op(MovementOps.RESHAPE, (1, 1, 1, C.cout//4, 4, C.H, C.W, 1, 1)) \
            .movement_op(MovementOps.EXPAND, (C.bs, C.oy, C.ox, C.cout//4, 4, C.H, C.W, 1, 1))
      else:
        w = w.movement_op(MovementOps.RESHAPE, (C.cout//4, C.H, C.cin//4, C.W, 4, 4))
        w = w.movement_op(MovementOps.PERMUTE, (0,4,1,3,2,5))
        w = w.movement_op(MovementOps.RESHAPE, (1, 1, 1, C.cout//4, 4, C.H, C.W, C.cin//4, 4)) \
            .movement_op(MovementOps.EXPAND, (C.bs, C.oy, C.ox, C.cout//4, 4, C.H, C.W, C.cin//4, 4))

      # now do the conv in this space
      ret = x.binary_op(BinaryOps.MUL, w).reduce_op(ReduceOps.SUM, (C.bs, C.oy, C.ox, C.cout//4, 4, 1, 1, 1, 1))
      ret = ret.movement_op(MovementOps.RESHAPE, (C.bs*C.oy, C.ox*C.cout//4, 4)).contiguous() #True)
      return postprocessing_op(ret, C, Cold)

    # TODO: fixup C?
    if NOCONV or not getattr(x.dbuffer, "SUPPORTS_PADDING", False):
      x = x.slice(((0, x.shape[0]), (0, x.shape[1]), (-C.py, x.shape[2]+C.py_), (-C.px, x.shape[3]+C.px_)))

    if NOCONV or not getattr(x.dbuffer, "processing_op", False):
      # universal conv, just mul and reduce
      # TODO: is there any way to replace strided with other movement ops? answer: not really
      if C.sy == 1 and C.sx == 1 and C.H == 1 and C.W == 1 and False:
        # TODO: this doesn't belong here, ShapeTracker or lazy should be able to infer this from STRIDED
        # TODO: this is disabled. it breaks fusion of ops without pushing PERMUTES. this is also a depthwise conv
        x = x.movement_op(MovementOps.RESHAPE, (C.bs, C.groups, C.cin, C.oy, C.ox, 1, C.H, C.W))
        x = x.movement_op(MovementOps.PERMUTE, (0,1,5,3,4,2,6,7))
      else:
        x = x.movement_op(MovementOps.STRIDED, (
          (C.bs, C.groups*C.cin*x.shape[2]*x.shape[3]), (C.groups, C.cin*x.shape[2]*x.shape[3]),
          (1, 1), (C.oy, C.sy*x.shape[3]), (C.ox, C.sx),
          (C.cin, x.shape[2]*x.shape[3]), (C.H, C.dy*x.shape[3]), (C.W, C.dx)))
      #if C.H <= 3 and C.W <= 3:  # max 9x the RAM overhead, this is im2col
      #  x = x.contiguous()
      x = x.movement_op(MovementOps.EXPAND, (C.bs, C.groups, C.rcout, C.oy, C.ox, C.cin, C.H, C.W))
      w = w.movement_op(MovementOps.RESHAPE, (1, C.groups, C.rcout, 1, 1, C.cin, C.H, C.W)) \
           .movement_op(MovementOps.EXPAND, (C.bs, C.groups, C.rcout, C.oy, C.ox, C.cin, C.H, C.W))
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
