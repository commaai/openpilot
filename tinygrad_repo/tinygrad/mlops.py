from tinygrad.helpers import prod, argsort, reduce_shape, get_conv_args
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps
from tinygrad.tensor import Function

class Contiguous(Function):
  def forward(self, x): return x.contiguous()
  def backward(self, grad_output): return grad_output

# ************* unary ops *************

class ReLU(Function):
  def forward(self, x):
    ret = x.unary_op(UnaryOps.RELU)
    self.save_for_backward(ret)
    return ret

  def backward(self, grad_output):
    return self.saved_tensors[0].unary_op(UnaryOps.SIGN).binary_op(BinaryOps.MUL, grad_output)

class Log(Function):
  def forward(self, x):
    self.save_for_backward(x)
    return x.unary_op(UnaryOps.LOG)

  def backward(self, grad_output):
    return grad_output.binary_op(BinaryOps.DIV, self.saved_tensors[0])

class Exp(Function):
  def forward(self, x):
    ret = x.unary_op(UnaryOps.EXP)
    self.save_for_backward(ret)
    return ret

  def backward(self, grad_output):
    return self.saved_tensors[0].binary_op(BinaryOps.MUL, grad_output)

class Reciprocal(Function):
  def forward(self, x):
    ret = x.unary_op(UnaryOps.RECIPROCAL)
    self.save_for_backward(ret)
    return ret

  def backward(self, grad_output):
    return grad_output.unary_op(UnaryOps.NEG).binary_op(BinaryOps.MUL, self.saved_tensors[0]).binary_op(BinaryOps.MUL, self.saved_tensors[0])

# TODO: add Neg? confirm the optimizer on Sub good enough

# ************* reduce ops *************

class Sum(Function):
  def forward(self, x, axis=None):
    self.input_shape = x.shape
    return x.reduce_op(ReduceOps.SUM, reduce_shape(x.shape, axis))

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.EXPAND, self.input_shape)

class Max(Function):
  def forward(self, x, axis=None):
    ret = x.reduce_op(ReduceOps.MAX, reduce_shape(x.shape, axis))
    self.save_for_backward(x, ret)
    return ret

  def backward(self, grad_output):
    x, ret = self.saved_tensors

    # 1s in locations where the max was chosen (can be two locations)
    max_is_1s = x.binary_op(BinaryOps.CMPEQ, ret.movement_op(MovementOps.EXPAND, x.shape))

    # sum of locations, averaged
    div = max_is_1s.reduce_op(ReduceOps.SUM, grad_output.shape).movement_op(MovementOps.EXPAND, x.shape)
    max_is_amount = max_is_1s.binary_op(BinaryOps.DIV, div)

    grad_output_expanded = grad_output.movement_op(MovementOps.EXPAND, x.shape)
    return max_is_amount.binary_op(BinaryOps.MUL, grad_output_expanded)

# ************* binary ops *************

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
    self.save_for_backward(x, y)
    return x.binary_op(BinaryOps.MUL, y)

  def backward(self, grad_output):
    return self.saved_tensors[1].binary_op(BinaryOps.MUL, grad_output) if self.needs_input_grad[0] else None, \
           self.saved_tensors[0].binary_op(BinaryOps.MUL, grad_output) if self.needs_input_grad[1] else None

class Pow(Function):
  def forward(self, x, y):
    ret = x.binary_op(BinaryOps.POW, y)
    self.save_for_backward(x, y, ret)
    return ret

  def backward(self, grad_output):
    x,y,powxy = self.saved_tensors
    # grad_x = grad_output * y * (pow(x,y)/x)
    # grad_y = grad_output * log(x) * pow(x,y)
    return grad_output.binary_op(BinaryOps.MUL, y.binary_op(BinaryOps.MUL, powxy.binary_op(BinaryOps.DIV, x))) if self.needs_input_grad[0] else None, \
           grad_output.binary_op(BinaryOps.MUL, x.unary_op(UnaryOps.LOG).binary_op(BinaryOps.MUL, powxy)) if self.needs_input_grad[1] else None

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
    shape = tuple(-prod(x.shape) // prod(shape) if s == -1 else s for s in shape)
    return x.movement_op(MovementOps.RESHAPE, shape)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.RESHAPE, self.input_shape)

class Permute(Function):
  def forward(self, x, order=(1,0)):
    self.input_order = order
    return x.movement_op(MovementOps.PERMUTE, order)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.PERMUTE, tuple(argsort(self.input_order)))

# TODO: merge Slice and Flip into Stride with the 3 arguments
class Slice(Function):
  def forward(self, x, arg=None):
    self.narg = tuple((0-p[0], x.shape[i]-p[0]) for i,p in enumerate(arg))
    return x.slice(tuple(arg))

  def backward(self, grad_output):
    return grad_output.slice(self.narg)

class Flip(Function):
  def forward(self, x, axis):
    self.axis = axis
    return x.movement_op(MovementOps.FLIP, axis)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.FLIP, self.axis)

# ************* processing ops *************

class Conv2D(Function):
  def forward(self, x, w, stride=1, groups=1, dilation=1, padding=0):
    self.C = get_conv_args(x.shape, w.shape, stride, groups, dilation=dilation, padding=padding)
    self.save_for_backward(x,w)
    return x.processing_op(ProcessingOps.CONV, w, self.C)

  def backward(self, grad_output):
    x, w = self.saved_tensors
    C = self.C   # conv args from the context
    dx, dw = None, None

    if self.needs_input_grad[0]:    # compute derivative of inputs using ProcessingOps.CONV (this is a transposed conv)
      xt = grad_output
      if C.sx > 1 or C.sy > 1:   # unstride. NOTE: this is really memory intensive for big strides. (but only when we contiguous it)
        xt = xt.movement_op(MovementOps.RESHAPE, (grad_output.shape[0], grad_output.shape[1], grad_output.shape[2], 1, grad_output.shape[3], 1))
        xt = xt.movement_op(MovementOps.PAD, ((0,0), (0,0), (0,0), (0,C.sy-1), (0,0), (0,C.sx-1)))
        xt = xt.movement_op(MovementOps.RESHAPE, (xt.shape[0], xt.shape[1], xt.shape[2]*C.sy, xt.shape[4]*C.sx))
      wt = w.movement_op(MovementOps.RESHAPE, (C.groups, C.rcout, C.cin, C.H, C.W)).movement_op(MovementOps.PERMUTE, (0, 2, 1, 3, 4)).movement_op(MovementOps.FLIP, (3, 4))
      wt = wt.movement_op(MovementOps.RESHAPE, (C.groups*C.cin, C.rcout, C.H, C.W))
      py, px = (C.H-1)*C.dy - C.py, (C.W-1)*C.dx - C.px
      Cdx = get_conv_args(xt.shape, wt.shape, out_shape=x.shape, dilation=(C.dy, C.dx), padding=(py, px), groups=C.groups)
      dx = xt.processing_op(ProcessingOps.CONV, wt, Cdx)

    if self.needs_input_grad[1]:   # compute derivative of weights using ProcessingOps.CONV
      xdw = x.movement_op(MovementOps.RESHAPE, (C.bs, C.groups, C.cin, C.iy, C.ix)).movement_op(MovementOps.PERMUTE, (2, 1, 0, 3, 4))
      xdw = xdw.movement_op(MovementOps.RESHAPE, (C.cin, C.groups*C.bs, C.iy, C.ix))
      grad_output_dw = grad_output.movement_op(MovementOps.PERMUTE, (1,0,2,3))
      Cdw = get_conv_args(xdw.shape, grad_output_dw.shape, out_shape=(w.shape[1], w.shape[0], w.shape[2], w.shape[3]), padding=(C.py, C.px), stride=(C.dy, C.dx), dilation=(C.sy, C.sx), groups=C.groups)
      dw = xdw.processing_op(ProcessingOps.CONV, grad_output_dw, Cdw).movement_op(MovementOps.PERMUTE, (1,0,2,3))

    return dx, dw
