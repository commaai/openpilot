from functools import lru_cache
from .tensor import Device, Function, register

@lru_cache
def compile_wrapper(ane, dat):
  return ane.compile(dat)

def roundup(x, v):
  return x + (v-x)%v

@lru_cache
def compile_relu(ane, sz):
  dat = list(open("accel/ane/ops/relu.hwx", "rb").read())
  # TODO: make this all nice and once
  # number of engines? (max 0x100)
  l2_stride = max(0x100, roundup(sz*2, 0x10))
  # 0x1ec = L2.SourceChannelStride.Stride, 0x1f0 = L2.SourceRowStride.Stride
  # 0x1f4, 0x1f8?
  # 0x214 = L2.ResultBase.Addr
  dat = ane.fill(dat, [0x1ec, 0x1f0, 0x1f4, 0x1f8, 0x214], "I", l2_stride)
  stride = roundup(sz*2, 0x40)
  dat = ane.filln(dat, {
    "NeuronType": 0x11,   # 0x10 makes this a copy, 0x11 = ReLU, 0x12 = crash
    "InputWidth": sz, "OutputWidth": sz,
    "InputRowStride": stride, "InputPlaneStride": stride, "InputDepthStride": stride,
    "OutputRowStride": stride, "OutputPlaneStride": stride, "OutputDepthStride": stride,
    })
  return compile_wrapper(ane, bytes(dat))

class ReLU(Function):
  def forward(ctx, input):
    ret = ctx.ane.tensor(input.shape)
    ctx.ane.run(compile_relu(ctx.ane, input.sz), input, ret)
    return ret

  def backward(ctx, grad_output):
    return 0

register('relu', ReLU, device=Device.ANE)
