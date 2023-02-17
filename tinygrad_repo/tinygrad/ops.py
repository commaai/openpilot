import os
from enum import Enum
from typing import Union, Type, NamedTuple, Tuple, Any, List
import functools, operator
from tinygrad.helpers import prod
from tinygrad.shape import ShapeTracker

DEBUG = int(os.getenv("DEBUG", "0"))

# these are the llops your accelerator must implement, along with toCpu
UnaryOps = Enum("UnaryOps", ["NOOP", "NEG", "RELU", "EXP", "LOG", "SIGN", "RECIPROCAL"])
BinaryOps = Enum("BinaryOps", ["ADD", "SUB", "MUL", "DIV", "POW", "CMPEQ"])
ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
MovementOps = Enum("MovementOps", ["RESHAPE", "PERMUTE", "EXPAND", "FLIP", "STRIDED", "PAD", "SHRINK"])
ProcessingOps = Enum("ProcessingOps", ["CONV"])
LoadOps = Enum("LoadOps", ["FROMCPU", "CONTIGUOUS"])

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps, LoadOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[ProcessingOps], Type[LoadOps]]

class LazyOp(NamedTuple):
  op: Op
  # Any == Union[LazyOp, LazyBuffer, DeviceBuffer]
  src: Tuple[Any, ...]  # type: ignore
  arg: Any = None
  # TODO: add dest to support multiple outputs

# Any == Union[LazyBuffer, DeviceBuffer]
def get_buffers(op:LazyOp) -> List[Any]: return functools.reduce(operator.add, [get_buffers(x) if isinstance(x, LazyOp) else [x] for x in op.src], [])
def get_lazyops(op:LazyOp) -> List[LazyOp]: return functools.reduce(operator.add, [get_lazyops(x) for x in op.src if isinstance(x, LazyOp)], [op])

# a placeholder class to extend by the exec classes
class DeviceBuffer:
  shape: Any   # should be Tuple[int, ...] but ndarray and torch.tensor have incompatible types

# extend this if you don't have an exec_ast function
# used in CPUBuffer and TorchBuffer
class GenericExecAST(DeviceBuffer):
  @classmethod
  def exec_ast(cls, ast:LazyOp, preprocess=lambda x: x):
    srcs = [cls.exec_ast(x, preprocess) if isinstance(x, LazyOp) else preprocess(x) for x in ast.src]
    if ast.op in UnaryOps:
      ret = srcs[0].unary_op(ast.op)
    elif ast.op in BinaryOps:
      assert srcs[0].shape == srcs[1].shape, f"BinaryOps shape mismatch {srcs[0].shape} != {srcs[1].shape}"
      ret = srcs[0].binary_op(ast.op, srcs[1])
    elif ast.op in ReduceOps:
      assert all(r == n or n == 1 for r,n in zip(srcs[0].shape, ast.arg)), f"ReduceOps can't reduce {srcs[0].shape} -> {ast.arg}"
      ret = srcs[0].reduce_op(ast.op, ast.arg)
    elif ast.op in MovementOps:
      ret = srcs[0].movement_op(ast.op, ast.arg)
    elif ast.op in ProcessingOps:
      ret = srcs[0].processing_op(ast.op, srcs[1], ast.arg)
    else:
      raise Exception("unknown op")
    return ret

class GlobalCounters:
  global_ops, global_mem = 0, 0

class GenericShape(GenericExecAST):
  def __init__(self, shape, flops=0): self.shape, self.flops = shape, flops
  def unary_op(self, op:UnaryOps): return GenericShape(self.shape, self.flops + prod(self.shape))
  def binary_op(self, op:BinaryOps, y): return GenericShape(self.shape, self.flops + y.flops + prod(self.shape))
  def reduce_op(self, op:ReduceOps, new_shape:Tuple[int, ...]): return GenericShape(new_shape, self.flops + prod(self.shape))
  def movement_op(self, op:MovementOps, arg): return GenericShape(ShapeTracker(self.shape).movement_op(op, arg).shape, self.flops)
  # https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
  def processing_op(self, op:ProcessingOps, w, C): return GenericShape(C.out_shape, 2 * (C.bs * C.cout * C.oy * C.ox) * (C.cin * C.H * C.W))
def get_lazyop_info(ast:LazyOp): return GenericShape.exec_ast(ast, lambda x: GenericShape(x.shape))

# assumes you are using ShapeTracker
# used in GPUBuffer, OpenCLBuffer, and LLVMBuffer
class ExplicitExecAST(DeviceBuffer):
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf=None):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape = self.st.shape

  @classmethod
  def exec_ast(cls, ast:LazyOp): raise NotImplementedError("must be implemented")

  # universal
  def unary_op(self, op:UnaryOps): return type(self)(self.shape).exec_ast(LazyOp(op=op, src=(self,)))
  def binary_op(self, op:BinaryOps, y): return type(self)(self.shape).exec_ast(LazyOp(op=op, src=(self, y)))
  def reduce_op(self, op:ReduceOps, new_shape:Tuple[int, ...]): return type(self)(new_shape).exec_ast(LazyOp(op=op, src=(self,), arg=new_shape))

  # universal for shape tracked
  def movement_op(self, op:MovementOps, arg): return type(self)(ShapeTracker(self.st).movement_op(op, arg), self)

  # TODO: creating a new object is making a copy, breaking the thneed compiler
  def contiguous(self): return self if self.st.contiguous else self.unary_op(UnaryOps.NOOP)
  #def contiguous(self): return type(self)(self.shape, hostbuf=self) if self.st.contiguous else self.unary_op(UnaryOps.NOOP)