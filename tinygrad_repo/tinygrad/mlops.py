from tinygrad.helpers import argsort
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps
from tinygrad.tensor import Function

class Contiguous(Function):
  def forward(self, x): return x.contiguous()
  def backward(self, grad_output): return grad_output

# ************* unary ops *************

class Log(Function):
  def forward(self, x):
    self.x = x
    return x.unary_op(UnaryOps.LOG)

  def backward(self, grad_output):
    return grad_output.binary_op(BinaryOps.DIV, self.x)

class Exp(Function):
  def forward(self, x):
    self.ret = x.unary_op(UnaryOps.EXP)
    return self.ret

  def backward(self, grad_output):
    return self.ret.binary_op(BinaryOps.MUL, grad_output)

# ************* reduce ops *************

class Sum(Function):
  def forward(self, x, new_shape):
    self.input_shape = x.shape
    return x.reduce_op(ReduceOps.SUM, new_shape)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.EXPAND, self.input_shape)

class Max(Function):
  def forward(self, x, new_shape):
    self.x, self.ret = x, x.reduce_op(ReduceOps.MAX, new_shape)
    return self.ret

  def backward(self, grad_output):
    # 1s in locations where the max was chosen (can be two locations)
    max_is_1s = self.x.binary_op(BinaryOps.CMPEQ, self.ret.movement_op(MovementOps.EXPAND, self.x.shape))

    # sum of locations, averaged
    div = max_is_1s.reduce_op(ReduceOps.SUM, grad_output.shape).movement_op(MovementOps.EXPAND, self.x.shape)
    max_is_amount = max_is_1s.binary_op(BinaryOps.DIV, div)

    grad_output_expanded = grad_output.movement_op(MovementOps.EXPAND, self.x.shape)
    return max_is_amount.binary_op(BinaryOps.MUL, grad_output_expanded)

# ************* binary ops *************

class Equal(Function):
  def forward(self, x, y):
    return x.binary_op(BinaryOps.CMPEQ, y)

class Maximum(Function):
  def forward(self, x, y):
    self.y, self.ret = y, x.binary_op(BinaryOps.MAX, y)
    return self.ret

  def backward(self, grad_output):
    mask = self.y.binary_op(BinaryOps.CMPEQ, self.ret)
    # TODO: if they are equal, do they split the gradient?
    return grad_output.binary_op(BinaryOps.MUL, mask.unary_op(UnaryOps.NOT)) if self.needs_input_grad[0] else None, \
           grad_output.binary_op(BinaryOps.MUL, mask) if self.needs_input_grad[1] else None

class Add(Function):
  def forward(self, x, y):
    return x.binary_op(BinaryOps.ADD, y)

  def backward(self, grad_output):
    return grad_output if self.needs_input_grad[0] else None, \
           grad_output if self.needs_input_grad[1] else None

class Sub(Function):
  def forward(self, x, y):
    return x.binary_op(BinaryOps.SUB, y)

  def backward(self, grad_output):
    return grad_output if self.needs_input_grad[0] else None, \
           grad_output.unary_op(UnaryOps.NEG) if self.needs_input_grad[1] else None

class Mul(Function):
  def forward(self, x, y):
    self.x, self.y = x, y
    return x.binary_op(BinaryOps.MUL, y)

  def backward(self, grad_output):
    return self.y.binary_op(BinaryOps.MUL, grad_output) if self.needs_input_grad[0] else None, \
           self.x.binary_op(BinaryOps.MUL, grad_output) if self.needs_input_grad[1] else None

class Pow(Function):
  def forward(self, x, y):
    self.x, self.y, self.ret = x, y, x.binary_op(BinaryOps.POW, y)
    return self.ret

  def backward(self, grad_output):
    return grad_output.binary_op(BinaryOps.MUL, self.y.binary_op(BinaryOps.MUL, self.ret.binary_op(BinaryOps.DIV, self.x))) if self.needs_input_grad[0] else None, \
           grad_output.binary_op(BinaryOps.MUL, self.x.unary_op(UnaryOps.LOG).binary_op(BinaryOps.MUL, self.ret)) if self.needs_input_grad[1] else None

class Div(Function):
  def forward(self, x, y):
    self.x, self.y = x, y
    return x.binary_op(BinaryOps.DIV, y)

  def backward(self, grad_output):
    return grad_output.binary_op(BinaryOps.DIV, self.y) if self.needs_input_grad[0] else None, \
           grad_output.unary_op(UnaryOps.NEG).binary_op(BinaryOps.MUL, self.x).binary_op(BinaryOps.DIV, self.y.binary_op(BinaryOps.MUL, self.y)) if self.needs_input_grad[1] else None

# ************* movement ops *************

# NOTE: this is sum in reverse
class Expand(Function):
  def forward(self, x, shape):
    self.input_shape = x.shape
    return x.movement_op(MovementOps.EXPAND, shape)

  def backward(self, grad_output):
    return grad_output.reduce_op(ReduceOps.SUM, self.input_shape)

class Reshape(Function):
  def forward(self, x, shape):
    self.input_shape = x.shape
    return x.movement_op(MovementOps.RESHAPE, shape)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.RESHAPE, self.input_shape)

class Permute(Function):
  def forward(self, x, order=(1,0)):
    self.input_order = order
    return x.movement_op(MovementOps.PERMUTE, order)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.PERMUTE, tuple(argsort(self.input_order)))

class Pad(Function):
  def forward(self, x, arg):
    self.narg = tuple((p[0], s+p[0]) for s,p in zip(x.shape, arg))
    return x.movement_op(MovementOps.PAD, arg)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.SHRINK, self.narg)

class Shrink(Function):
  def forward(self, x, arg):
    self.narg = tuple((p[0], s-p[1]) for s,p in zip(x.shape, arg))
    return x.movement_op(MovementOps.SHRINK, arg)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.PAD, self.narg)

class Flip(Function):
  def forward(self, x, axis):
    self.axis = axis
    return x.movement_op(MovementOps.FLIP, axis)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.FLIP, self.axis)
