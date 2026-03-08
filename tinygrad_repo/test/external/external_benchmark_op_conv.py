# ruff: noqa: E501 E712 F401
from dataclasses import replace
from tinygrad import dtypes, Device
from tinygrad.uop.ops import UOp, AxisType, Ops, KernelInfo
from tinygrad.codegen.opt import Opt, OptOps # pylint: disable=unused-import
from tinygrad.engine.realize import CompiledRunner, get_program
from tinygrad.helpers import dedup, getenv
from tinygrad.device import Buffer
from tinygrad.dtype import ImageDType, Invalid

# PYTHONPATH="." DEV=QCOM FLOAT16=1 IMAGE=2 NOLOCALS=1 taskset -c 4-7 python3 examples/openpilot/compile3.py https://github.com/commaai/openpilot/raw/720392c9a5b986981fdbed1bb8c47a6c5573a50e/selfdrive/modeld/models/driving_vision.onnx

def vision_conv_143():
  c0 = UOp(Ops.PARAM, dtypes.imageh((16, 1024, 4)), (), 0)
  c2 = UOp.range(32, 3, AxisType.LOOP)
  c5 = UOp.range(128, 4, AxisType.LOOP)
  c8 = UOp.range(16, 2, AxisType.LOOP)
  c16 = UOp.range(7, 0, AxisType.REDUCE)
  c17 = c8*2+c16
  c24 = ((c17<3)!=True)&(c17<35)
  c26 = UOp.range(7, 1, AxisType.REDUCE)
  c27 = c2*2+c26
  c32 = ((c27<3)!=True)&(c27<67)
  c34 = UOp(Ops.PARAM, dtypes.imageh((32, 1024, 4)), (), 1)
  c38 = c5//2
  c45 = (c32&c24).where((c27*64+c38+c17*4096+-12480), UOp.const(dtypes.index, Invalid))
  c48 = (c24&c32).where(c34.index(c45), UOp.const(dtypes.float, 0.0))
  c49 = UOp(Ops.PARAM, dtypes.imageh((64, 49, 4)), (), 2)
  c61 = c48*c49.index((c26*4+c5%2+c16*28+c38*196))
  c63 = UOp(Ops.PARAM, dtypes.float.ptr(128), (), 3)
  c65 = c61.reduce(c16, c26, arg=Ops.ADD)+c63.index(c5)
  c67 = c0.index((c2*128+c5+c8*4096), ptr=True).store(c65).end(c8, c2, c5)

  opts = None
  # JITBEAM=2
  # (Opt(op=OptOps.UPCAST, axis=2, arg=4), Opt(op=OptOps.NOLOCALS, axis=None, arg=None), Opt(op=OptOps.UPCAST, axis=2, arg=2), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.SWAP, axis=1, arg=2))
  return c67.sink(arg=KernelInfo(name="conv", opts_to_apply=opts))

def vision_conv_153():
  c0 = UOp(Ops.PARAM, dtypes.imageh((8, 1024, 4)), (), 0)
  c2 = UOp.range(16, 3, AxisType.LOOP)
  c5 = UOp.range(256, 4, AxisType.LOOP)
  c8 = UOp.range(8, 2, AxisType.LOOP)
  c16 = UOp.range(7, 0, AxisType.REDUCE)
  c17 = c8*2+c16
  c24 = ((c17<3)!=True)&(c17<19)
  c26 = UOp.range(7, 1, AxisType.REDUCE)
  c27 = c2*2+c26
  c32 = ((c27<3)!=True)&(c27<35)
  c34 = UOp(Ops.PARAM, dtypes.imageh((16, 1024, 4)), (), 1)
  c38 = c5//2
  c45 = (c32&c24).where((c27*128+c38+c17*4096+-12672), UOp.const(dtypes.index, Invalid))
  c48 = (c24&c32).where(c34.index(c45), UOp.const(dtypes.float, 0.0))
  c49 = UOp(Ops.PARAM, dtypes.imageh((128, 49, 4)), (), 2)
  c61 = c48*c49.index((c26*4+c5%2+c16*28+c38*196))
  c63 = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 3)
  c65 = c61.reduce(c16, c26, arg=Ops.ADD)+c63.index(c5)
  c67 = c0.index((c2*256+c5+c8*4096), ptr=True).store(c65).end(c8, c2, c5)

  opts = None
  # JITBEAM=2
  # (Opt(op=OptOps.UPCAST, axis=2, arg=4), Opt(op=OptOps.NOLOCALS, axis=None, arg=None), Opt(op=OptOps.UPCAST, axis=2, arg=2), Opt(op=OptOps.SWAP, axis=1, arg=2))
  return c67.sink(arg=KernelInfo(name="conv", opts_to_apply=opts))

def dm_conv_172():
  c0 = UOp(Ops.PARAM, dtypes.imageh((1, 240, 4)), (), 0)
  c2 = UOp.range(960, 4, AxisType.LOOP)
  c5 = UOp(Ops.PARAM, dtypes.imageh((8, 384, 4)), (), 1)
  c7 = UOp.range(32, 0, AxisType.REDUCE)
  c10 = UOp.range(4, 1, AxisType.REDUCE)
  c13 = UOp.range(12, 3, AxisType.REDUCE)
  c18 = UOp.range(8, 2, AxisType.REDUCE)
  c23 = UOp(Ops.PARAM, dtypes.imageh((240, 128, 4)), (), 2)
  c35 = c5.index((c7*4+c10+c13*128+c18*1536))*c23.index((c10*4+c2%4+c7*16+c2//4*512))
  c37 = UOp(Ops.PARAM, dtypes.float.ptr(960), (), 3)
  c39 = c35.reduce(c7, c10, arg=Ops.ADD)+c37.index(c2)
  c50 = (1.0+((c39+0.044708251953125*(c39*(c39*c39)))*-2.3021129851685216).exp2()).reciprocal()*c39
  c53 = c50.reduce(c18, c13, arg=Ops.ADD)*0.010416666666666666
  c55 = c0.index(c2, ptr=True).store(c53).end(c2)

  opts = None
  # JITBEAM=2
  # (Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.GROUPTOP, axis=1, arg=32), Opt(op=OptOps.UNROLL, axis=1, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=8), Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.GROUP, axis=1, arg=0))
  return c55.sink(arg=KernelInfo(name="conv", opts_to_apply=opts))

ast = {143: vision_conv_143, 153: vision_conv_153, 172: dm_conv_172}[getenv("NUM", 143)]()

renderer = Device.default.renderer
allocator = Device.default.allocator

ps = get_program(ast, renderer)
cr = CompiledRunner(replace(ps, device=Device.DEFAULT))

gs = sorted(dedup([u for u in ast.toposort() if u.op is Ops.PARAM]), key=lambda u: u.arg)
# print(len(gs))
# print([g.dtype for g in gs])
bufs = [Buffer(ps.device, g.size, g.dtype if isinstance(g.dtype, ImageDType) else g.dtype._base).ensure_allocated() for g in gs]

t = cr(bufs, wait=True)
print(f"{t*1e6:.2f} us")