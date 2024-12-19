"""This is where the forwards and backwards passes live."""
import math
from typing import Tuple, Optional
from tinygrad.helpers import argsort
from tinygrad.dtype import dtypes, DType, sum_acc_dtype
from tinygrad.ops import Ops, resolve, sint, UOp
from tinygrad.tensor import Function

class Contiguous(Function):
  def forward(self, x:UOp) -> UOp: return x.contiguous()
  def backward(self, grad_output:UOp) -> UOp: return grad_output

class ContiguousBackward(Function):
  def forward(self, x:UOp) -> UOp: return x
  def backward(self, grad_output:UOp) -> UOp: return grad_output.contiguous()

class Cast(Function):
  def forward(self, x:UOp, dtype:DType, bitcast:bool=False) -> UOp:
    self.input_dtype, self.bitcast = x.dtype, bitcast
    return x.bitcast(dtype) if self.bitcast else x.cast(dtype)

  def backward(self, grad_output:UOp) -> UOp:
    if self.bitcast: raise RuntimeError("bitcast cannot backward")
    return grad_output.cast(self.input_dtype)

# ************* unary ops *************

class Reciprocal(Function):
  def forward(self, x:UOp) -> UOp:
    self.ret = x.reciprocal()
    return self.ret

  def backward(self, grad_output:UOp) -> UOp: return -grad_output * self.ret * self.ret

class Sin(Function):
  def forward(self, x:UOp) -> UOp:
    self.x = x
    return x.sin()

  def backward(self, grad_output:UOp) -> UOp: return (math.pi/2 - self.x).sin() * grad_output

class Relu(Function):
  def forward(self, x:UOp) -> UOp:
    self.ret = (x>0).where(x, 0)
    return self.ret

  def backward(self, grad_output:UOp) -> UOp: return (self.ret>0).cast(grad_output.dtype) * grad_output

class Log(Function):
  def forward(self, x:UOp) -> UOp:
    self.x = x
    return x.log2() * math.log(2)

  def backward(self, grad_output:UOp) -> UOp: return grad_output / self.x

class Exp(Function):
  def forward(self, x:UOp) -> UOp:
    self.ret = (x * (1/math.log(2))).exp2()
    return self.ret

  def backward(self, grad_output:UOp) -> UOp: return self.ret * grad_output

class Sqrt(Function):
  def forward(self, x:UOp) -> UOp:
    self.ret = x.sqrt()
    return self.ret

  def backward(self, grad_output:UOp) -> UOp: return grad_output / (self.ret*2)

# NOTE: the implicit derivative of sigmoid is not stable
# https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
# TODO: have the backend automatically find this
class Sigmoid(Function):
  def forward(self, x:UOp) -> UOp:
    self.ret = (1 + (x * (-1/math.log(2))).exp2()).reciprocal()
    return self.ret

  def backward(self, grad_output:UOp) -> UOp:
    return (self.ret * (1 - self.ret)) * grad_output

class Sign(Function):
  # NOTE: the x*0 is to match torch behavior without function.py
  def forward(self, x:UOp) -> UOp: return x.ne(0).where((x<0).where(x.const_like(-1), x.const_like(1)), x.const_like(0)) + x*0
  # backward always return 0 to match torch
  def backward(self, grad_output:UOp) -> UOp: return grad_output.const_like(0)

# ************* binary ops *************

class Less(Function):
  def forward(self, x:UOp, y:UOp) -> UOp: return x<y
  def backward(self, grad_output:UOp) -> Tuple[Optional[UOp], Optional[UOp]]: return None, None

class Neq(Function):
  def forward(self, x:UOp, y:UOp) -> UOp: return x.ne(y)
  def backward(self, grad_output:UOp) -> Tuple[Optional[UOp], Optional[UOp]]: return None, None

class Xor(Function):
  def forward(self, x:UOp, y:UOp) -> UOp: return x^y

class BitwiseAnd(Function):
  def forward(self, x:UOp, y:UOp) -> UOp: return x&y

class BitwiseOr(Function):
  def forward(self, x:UOp, y:UOp) -> UOp: return x|y

class Threefry(Function):
  def forward(self, x:UOp, seed:UOp) -> UOp: return x.threefry(seed)

class Add(Function):
  def forward(self, x:UOp, y:UOp) -> UOp: return x+y

  def backward(self, grad_output:UOp) -> Tuple[Optional[UOp], Optional[UOp]]:
    return grad_output if self.needs_input_grad[0] else None, \
           grad_output if self.needs_input_grad[1] else None

class Mul(Function):
  def forward(self, x:UOp, y:UOp) -> UOp:
    self.x, self.y = x, y
    return x * y

  def backward(self, grad_output:UOp) -> Tuple[Optional[UOp], Optional[UOp]]:
    return (self.y * grad_output) if self.needs_input_grad[0] else None, \
           (self.x * grad_output) if self.needs_input_grad[1] else None

class IDiv(Function):
  def forward(self, x:UOp, y:UOp) -> UOp: return x // y

# ************* ternary ops *************

class Where(Function):
  def forward(self, x:UOp, y:UOp, z:UOp) -> UOp:
    self.x = x
    return self.x.where(y, z)

  def backward(self, grad_output:UOp) -> Tuple[None, Optional[UOp], Optional[UOp]]:
    return None, \
      self.x.where(grad_output, grad_output.const_like(0)) if self.needs_input_grad[1] else None, \
      self.x.where(grad_output.const_like(0), grad_output) if self.needs_input_grad[2] else None

# ************* reduce ops *************

class Sum(Function):
  def forward(self, x:UOp, axis:Tuple[int, ...]) -> UOp:
    self.input_shape = x.shape
    return x.r(Ops.ADD, axis)

  def backward(self, grad_output:UOp) -> UOp: return grad_output.expand(self.input_shape)

class Prod(Function):
  def forward(self, x:UOp, axis:Tuple[int, ...]) -> UOp:
    self.x, self.ret = x, x.r(Ops.MUL, axis)
    return self.ret

  def backward(self, grad_output:UOp) -> UOp:
    return (grad_output * self.ret).expand(self.x.shape) / self.x

class Max(Function):
  def forward(self, x:UOp, axis:Tuple[int, ...]) -> UOp:
    self.x, self.ret, self.axis = x, x.r(Ops.MAX, axis), axis
    return self.ret

  def backward(self, grad_output:UOp) -> UOp:
    # 1s in locations where the max was chosen (can be two locations)
    max_is_1s = self.x.ne(self.ret.expand(self.x.shape)).ne(self.x.const_like(1).cast(dtypes.bool)).cast(grad_output.dtype)
    div = max_is_1s.r(Ops.ADD, self.axis).expand(self.x.shape)
    return (max_is_1s/div) * grad_output.expand(self.x.shape)

# ************* movement ops *************

# NOTE: this is sum in reverse
class Expand(Function):
  def forward(self, x:UOp, shape:Tuple[int, ...]) -> UOp:
    self.expanded_axis = tuple(i for i, (si, so) in enumerate(zip(x.shape, shape)) if resolve(si != so))
    return x.expand(shape)

  def backward(self, grad_output:UOp) -> UOp:
    return grad_output.cast(sum_acc_dtype(grad_output.dtype)).r(Ops.ADD, self.expanded_axis).cast(grad_output.dtype)

class Reshape(Function):
  def forward(self, x:UOp, shape:Tuple[int, ...]) -> UOp:
    self.input_shape = x.shape
    return x.reshape(shape)

  def backward(self, grad_output:UOp) -> UOp: return grad_output.reshape(self.input_shape)

class Permute(Function):
  def forward(self, x:UOp, order:Tuple[int, ...]) -> UOp:
    self.input_order = order
    return x.permute(order)

  def backward(self, grad_output:UOp) -> UOp: return grad_output.permute(argsort(self.input_order))

class Pad(Function):
  def forward(self, x:UOp, arg:Tuple[Tuple[int, int], ...]) -> UOp:
    self.narg = tuple([(p[0], s+p[0]) for s,p in zip(x.shape, arg)])
    return x.pad(arg)

  def backward(self, grad_output:UOp) -> UOp: return grad_output.shrink(self.narg)

class Shrink(Function):
  def forward(self, x:UOp, arg:Tuple[Tuple[sint, sint], ...]) -> UOp:
    self.narg = tuple([(p[0], s-p[1]) for s,p in zip(x.shape, arg)])
    return x.shrink(arg)

  def backward(self, grad_output:UOp) -> UOp: return grad_output.pad(self.narg)

class Flip(Function):
  def forward(self, x:UOp, axis:Tuple[int, ...]) -> UOp:
    self.arg = tuple([-1 if i in axis else 1 for i in range(len(x.shape))])
    return x.stride(self.arg)

  def backward(self, grad_output:UOp) -> UOp: return grad_output.stride(self.arg)
