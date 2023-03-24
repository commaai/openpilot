from __future__ import annotations
import functools, itertools, operator, random
import numpy as np
from enum import Enum, auto
from typing import Union, Type, NamedTuple, Tuple, Any, List, ClassVar, Optional, Callable, Dict, TypeVar, Set
from tinygrad.helpers import prod, DEBUG, getenv
from tinygrad.shape import ShapeTracker

# these are the llops your accelerator must implement, along with toCpu
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
class UnaryOps(Enum): NOOP = auto(); NEG = auto(); EXP = auto(); LOG = auto(); NOT = auto() # noqa: E702
class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); POW = auto(); CMPEQ = auto(); MAX = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); FLIP = auto(); PAD = auto(); SHRINK = auto() # noqa: E702
class FusedOps(Enum): MULACC = auto() # noqa: E702
class LoadOps(Enum): FROMCPU = auto(); CONTIGUOUS = auto() # noqa: E702

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, FusedOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[LoadOps], Type[FusedOps]]

class LazyOp(NamedTuple):
  op: Op
  # Any == Union[LazyOp, LazyBuffer, DeviceBuffer]
  src: Tuple[Any, ...]  # type: ignore
  arg: Any = None
  # TODO: add dest to support multiple outputs

# Any == Union[LazyBuffer, DeviceBuffer]
def get_buffers(op:LazyOp) -> List[Any]: return functools.reduce(operator.add, [get_buffers(x) if isinstance(x, LazyOp) else [x] for x in op.src], [])
def get_lazyops(op:LazyOp) -> List[LazyOp]: return functools.reduce(operator.add, [get_lazyops(x) for x in op.src if isinstance(x, LazyOp)], [op])
def map_buffers(real_srcs, x:LazyOp) -> LazyOp:
  if x in real_srcs: return map_buffers(real_srcs, real_srcs[x]) if isinstance(real_srcs[x], LazyOp) else real_srcs[x]
  return LazyOp(x.op, tuple((map_buffers(real_srcs, y) if isinstance(y, LazyOp) else real_srcs[y]) for y in x.src), x.arg)

_T = TypeVar("_T")
class RawBuffer:
  def __init__(self, size): raise NotImplementedError("must be implemented")
  @classmethod
  def fromCPU(cls:Type[_T], x:np.ndarray) -> _T: raise NotImplementedError("must be implemented")
  def toCPU(self:RawBuffer) -> np.ndarray: raise NotImplementedError("must be implemented")

class RawBufferCopyIn(RawBuffer):
  def copyin(self, x:np.ndarray) -> None: raise NotImplementedError("must be implemented")

  @classmethod
  def fromCPU(cls, x:np.ndarray):
    ret = cls(4*prod(x.shape))
    ret.copyin(x)
    return ret

class RawBufferCopyInOut(RawBufferCopyIn):
  size : int
  def copyout(self, x:np.ndarray) -> None: raise NotImplementedError("must be implemented")

  def toCPU(self) -> np.ndarray:
    x = np.empty((self.size//4), dtype=np.float32)
    self.copyout(x)
    return x

# a placeholder class to extend by the exec classes
class DeviceBuffer(RawBuffer):
  _buf: Any                # underlying buffer
  shape: Tuple[int, ...]
  @classmethod
  def exec_ast(cls, ast:LazyOp, output_buffer=None): raise NotImplementedError("must be implemented")

# this is a quick "buffer" class for flop tracking and getting the output shape
class GenericShape:
  def __init__(self, shape:Tuple[int, ...], flops:int=0): self.shape, self.flops = shape, flops
  def consume_flops(self):
    self.flops, ret = 0, self.flops
    return ret
shape_fxn_for_op : Dict[Op, Callable] = {
  **{op:lambda self: GenericShape(self.shape, self.consume_flops() + prod(self.shape)) for op in UnaryOps},
  **{op:lambda self,y: GenericShape(self.shape, self.consume_flops() + y.consume_flops() + prod(self.shape)) for op in BinaryOps},
  **{op:lambda self,new_shape: GenericShape(new_shape, self.consume_flops() + prod(self.shape)) for op in ReduceOps},
  **{op:functools.partial(lambda mop,self,arg: GenericShape(ShapeTracker(self.shape).movement_op(mop, arg).shape, self.consume_flops()), op) for op in MovementOps}}

# used in CPUBuffer and TorchBuffer
class InterpretedBuffer(DeviceBuffer):  # pylint: disable=abstract-method
  fxn_for_op : ClassVar = shape_fxn_for_op
  # TODO: use generic types here to remove __init__ in specialized classes
  def __init__(self, lbuf:Any): self._buf, self.shape = lbuf, tuple(lbuf.shape)
  def contiguous(self): return type(self).exec_ast(LazyOp(op=UnaryOps.NOOP, src=(self,)))
  def movement_op(self, op:MovementOps, arg=None): return type(self)(self.fxn_for_op[op](self._buf, arg)) if op in self.fxn_for_op else type(self)(getattr(self._buf, op.name.lower())(arg))
  @classmethod
  def exec_ast(cls, ast:LazyOp, output_buffer:Optional[InterpretedBuffer]=None, context=None):
    if FusedOps.MULACC in cls.fxn_for_op and ast.op == ReduceOps.SUM and isinstance(ast.src[0], LazyOp) and ast.src[0].op == BinaryOps.MUL:
      ast = LazyOp(FusedOps.MULACC, ast.src[0].src, ast.arg)
    if context is None: context = dict()
    if ast in context: return context[ast]
    srcs = [cls.exec_ast(x, context=context) if isinstance(x, LazyOp) else x for x in ast.src]
    if DEBUG >= 4: print("exec_ast", ast.op, [x.shape for x in srcs], ast.arg)
    if ast.op in BinaryOps: assert srcs[0].shape == srcs[1].shape, f"BinaryOps shape mismatch {srcs[0].shape} != {srcs[1].shape}"
    if ast.op in ReduceOps: assert all(r == n or n == 1 for r,n in zip(srcs[0].shape, ast.arg)), f"ReduceOps can't reduce {srcs[0].shape} -> {ast.arg}"
    if ast.op in MovementOps: ret = srcs[0].movement_op(ast.op, ast.arg)
    else: ret = cls(cls.fxn_for_op[ast.op](*([x._buf for x in srcs] + ([ast.arg] if ast.arg else []))))
    context[ast] = ret
    if output_buffer is not None:
      assert output_buffer.shape == ret.shape
      output_buffer._buf = ret._buf
      return output_buffer
    else:
      return ret
def get_lazyop_info(ast:LazyOp): return InterpretedBuffer.exec_ast(map_buffers({x:InterpretedBuffer(GenericShape(x.shape)) for x in get_buffers(ast)}, ast))._buf

class ASTRunner:
  def __init__(self, name, prg, bufs_to_delete:Optional[Set[int]]=None, global_size:Optional[List[int]]=None, local_size:Optional[List[int]]=None, op_estimate=0, mem_estimate=0):
    if DEBUG >= 4: print(prg)
    self.name, self.prg, self.global_size, self.local_size, self.bufs_to_delete, self.op_estimate, self.mem_estimate = name, prg, global_size, local_size, bufs_to_delete if bufs_to_delete else set(), op_estimate, mem_estimate
  def build(self, runtime):
    self.clprg = runtime(self.name, self.prg)
    return self
  def timeit(self, bufs, local_override=None) -> float:
    try: return self.clprg(self.global_size, local_override if local_override is not None else self.local_size, *bufs, wait=True)
    except Exception: return float('inf')
  def optimize_local_size(self, bufs) -> List[int]:
    assert self.global_size is not None, "needs a global size to optimize local size"
    MAX_WORKGROUP = self.clprg.max_work_group_size() if hasattr(self.clprg, 'max_work_group_size') else 1024
    local_dims = [[x for x in set([sz, 1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_WORKGROUP]) if x<=sz] for sz in self.global_size]
    local_sizes = [list(x) for x in itertools.product(*local_dims) if prod(x) <= MAX_WORKGROUP] * 2  # try each valid size twice
    return min([(self.timeit(bufs, local_size), local_size) for local_size in random.sample(local_sizes, len(local_sizes))])[1]
  def lower(self, bufs) -> List[RawBuffer]: return [x.raw() for i,x in enumerate(bufs) if x is not None and i not in self.bufs_to_delete]
  def __call__(self, bufs):
    if getenv("OPTLOCAL") and self.global_size is not None and self.local_size is None: self.local_size = self.optimize_local_size(bufs)
    if et := self.clprg(self.global_size, self.local_size, *bufs, wait=DEBUG>=2): GlobalCounters.time_sum_s += et
    if DEBUG >= 1:
      print(f"**** {GlobalCounters.kernel_count:4d} {self.name:20s} args {len(bufs):5d}  kernels {str(self.global_size):18s} {str(self.local_size):12s} OPs {self.op_estimate/1e6:7.1f}M/{GlobalCounters.global_ops/1e9:7.2f}G  mem {GlobalCounters.mem_used/1e9:5.2f} GB " +
            (str() if et is None else f"tm {et*1e6:9.2f}us/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({self.op_estimate/(et*1e9):8.2f} GFLOPS)"))
    GlobalCounters.log_kernel(self.op_estimate, self.mem_estimate)
    return et

# assumes you are using ShapeTracker
# used in GPUBuffer and LLVMBuffer
class CompiledBuffer(DeviceBuffer):  # pylint: disable=abstract-method
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf:Optional[CompiledBuffer]=None, backing:Optional[np.ndarray]=None, force_create=False):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape = self.st.shape
    self._base_shape : Tuple[int, ...] = hostbuf._base_shape if hostbuf is not None else self.shape
    self._buf = hostbuf._buf if hostbuf is not None else None
    self._backing : Optional[np.ndarray] = hostbuf._backing if hostbuf is not None else backing
    if (self._backing is not None and self._backing.shape != (1,)) or force_create: self.raw()

  # TODO: not GPUBuffer, get name of class
  def __repr__(self): return f"GPUBuffer(shape={self.st}, hostbuf=GPUBuffer(shape={self._base_shape}" + (f", backing=np.array({self._backing}, dtype=np.float32)))" if self._backing else ", force_create=True))")

  raw_buffer_type : Type[RawBuffer]
  @classmethod
  def create_raw_buffer(cls, shape, backing) -> RawBuffer:
    assert backing is None or prod(shape) == prod(backing.shape), "backing has the wrong shape"
    assert backing is None or GlobalCounters.cache is None, f"can't copy in {backing.shape} while caching"
    return cls.raw_buffer_type(4*prod(shape)) if backing is None else cls.raw_buffer_type.fromCPU(backing)
  def raw(self) -> RawBuffer:
    if self._buf is None:
      self._buf = self.create_raw_buffer(self._base_shape, self._backing)
      self._backing = None
    return self._buf

  @classmethod
  def fromCPU(cls, x:np.ndarray) -> CompiledBuffer: return cls(x.shape, backing=x.view(np.ndarray).astype(np.float32).ravel())
  def toCPU(self) -> np.ndarray:
    assert GlobalCounters.cache is None, f"can't copy out {self} while caching"
    return self.contiguous().raw().toCPU().reshape(self.shape)

  codegen_type : Any
  runtime_type : Type
  method_cache : Dict[str, ASTRunner] = {}
  @classmethod
  def exec_ast(cls, ast:LazyOp, output_buffer:Optional[CompiledBuffer]=None):
    k = cls.codegen_type(ast, output_buffer)
    if getenv("ENABLE_METHOD_CACHE"):   # TODO: this breaks the ops test!
      if k.key not in cls.method_cache: cls.method_cache[k.key] = k.codegen().build(cls.runtime_type)
      elif DEBUG >= 4: print(f"method cache hit : {k.key}")
      prg = cls.method_cache[k.key]
    else:
      prg = k.codegen().build(cls.runtime_type)
    if getenv("PRINT_AST", "") == prg.name:
      k.print()
      print(prg.prg)
    rawbufs = prg.lower(k.bufs)
    if GlobalCounters.cache is not None: GlobalCounters.cache.append((prg, rawbufs))
    prg(rawbufs)
    return k.ret

  # universal for shape tracked
  def contiguous(self): return self if self.st.contiguous and prod(self._base_shape) == prod(self.shape) else type(self).exec_ast(LazyOp(op=UnaryOps.NOOP, src=(self,)))
  def movement_op(self, op:MovementOps, arg): return type(self)(ShapeTracker(self.st).movement_op(op, arg), self)

class GlobalCounters:
  global_ops : ClassVar[int] = 0
  global_mem : ClassVar[int] = 0
  time_sum_s : ClassVar[float] = 0.0
  kernel_count : ClassVar[int] = 0
  mem_used : ClassVar[int] = 0   # NOTE: this is not reset
  cache : ClassVar[Optional[List[Tuple[Callable, Any]]]] = None
  @staticmethod
  def reset(): GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.time_sum_s, GlobalCounters.kernel_count, GlobalCounters.cache = 0,0,0.0,0,None
  @staticmethod
  def log_kernel(op_estimate:int, mem_estimate:int):
    GlobalCounters.kernel_count += 1
    GlobalCounters.global_ops += op_estimate
    GlobalCounters.global_mem += mem_estimate