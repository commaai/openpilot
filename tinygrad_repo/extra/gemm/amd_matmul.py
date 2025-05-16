# kernel8_batched_gmem.s from https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html
import pathlib
import numpy as np
from dataclasses import replace
from tinygrad import Tensor, Device, Context
from tinygrad.helpers import getenv
from tinygrad.codegen.kernel import Kernel, Opt, OptOps
from tinygrad.engine.realize import CompiledRunner, ExecItem
from tinygrad.ops import graph_rewrite, PatternMatcher, UPat, Ops, UOp

# TODO: on METAL for `DEBUG=4 python3 extra/gemm/amd_matmul.py`
#  * fix load grouping (like float4). idk why it's not working, need new devectorizer (this is a Monday project)
#  * DONE - remove extra barrier
#  * DONE (moved Ops.ADD) - fix load order to be in order (the +0 one is last!)
#  * explore async (fast) global load -> local store
#  * why is TC=3 broken for 4096x4096?
#  * write syntactic sugar for these local additions + use it in tensor core kernel.py

N = 4096
LN = 16
run_count = 5

from tinygrad.shape.shapetracker import ShapeTracker, View
def transform_load(ctx:tuple[Kernel, set[UOp]], x:UOp):
  if x.src[0].op is not Ops.DEFINE_GLOBAL: return None
  if x in ctx[1]: return None
  print(ctx[0].colored_shape())
  ctx[1].add(x)
  input_st: ShapeTracker = x.src[1].arg
  #strides = input_st.real_strides()
  #strides = (0,0)+strides[2:]
  if input_st.real_strides()[2] == 0:
    perm = (0,1,5,3,4,2)
    strides = (0,0,LN*4,4,0,0,1,0)
  elif input_st.real_strides()[3] == 0:
    perm = (0,1,2,5,4,3)
    strides = (0,0,LN*4,4,0,0,0,1)
  else:
    return None
  if len(input_st.shape) == 8:
    local_st = ShapeTracker(views=(View.create((1,1,LN,LN,1,1,4,4), strides),))
    perm = perm + (6,7)
  else:
    local_st = ShapeTracker(views=(View.create((1,1,LN,LN,1,1)),))
  #local_st = ShapeTracker(views=(View.create((1,1,LN,LN,1,1)),))
  load_st = local_st.permute(perm)
  input_st = input_st.permute(perm)
  lcl = UOp(Ops.DEFINE_LOCAL, x.dtype.ptr(local_st.real_size(), local=True), (), f"temp{x.src[0].arg}")
  global_load = x.replace(src=(x.src[0], input_st.to_uop()))
  ret = UOp(Ops.STORE, src=(lcl, local_st.to_uop(), global_load))
  return UOp(Ops.LOAD, x.dtype, src=(lcl, load_st.to_uop(), ret))

local_loads_pm = PatternMatcher([
  (UPat(Ops.LOAD, name="x"), transform_load),
])

def ast_transform(k, ast):
  #return ast
  ast = graph_rewrite(ast, local_loads_pm, ctx=(k, set()))
  #ast = ast.replace(arg=replace(ast.arg, upcasted=0))
  print(ast)
  return ast

if __name__ == "__main__":
  rng = np.random.default_rng()
  a = Tensor(na:=rng.random((4096, 4096), dtype=np.float32)).realize()
  b = Tensor(nb:=rng.random((4096, 4096), dtype=np.float32)).realize()
  c = a @ b
  si = c.schedule()[-1]
  k = Kernel(si.ast, opts=Device[Device.DEFAULT].renderer)
  #opts = [Opt(op=OptOps.LOCAL, axis=1, arg=16),
  #        Opt(op=OptOps.LOCAL, axis=0, arg=8),
  #        Opt(op=OptOps.UPCAST, axis=2, arg=4),
  #        Opt(op=OptOps.UPCAST, axis=1, arg=4),
  #        Opt(op=OptOps.UPCAST, axis=0, arg=2)]
  #opts = [Opt(op=OptOps.UPCAST, axis=1, arg=4),
  #        Opt(op=OptOps.UPCAST, axis=0, arg=4),
  #        Opt(op=OptOps.LOCAL, axis=1, arg=8),
  #        Opt(op=OptOps.LOCAL, axis=0, arg=4)]
  opts = [Opt(op=OptOps.UNROLL, axis=0, arg=LN),
          #Opt(op=OptOps.UPCAST, axis=0, arg=4),
          #Opt(op=OptOps.UPCAST, axis=1, arg=4),
          Opt(op=OptOps.LOCAL, axis=1, arg=LN),
          Opt(op=OptOps.LOCAL, axis=0, arg=LN)]
  k.apply_opts(opts)
  prg = k.to_program(ast_transform=ast_transform)
  if getenv("FAST", 1) and Device.DEFAULT == "AMD":
    #src = (pathlib.Path(__file__).parent / "fp32_sgemm_amd" / "src" / "kernel8_batched_gmem.s").read_text()
    src = (pathlib.Path(__file__).parent / "kernel8_batched_gmem.s").read_text()
    prg = replace(prg, src=src, global_size=[N//128, N//128, 1], local_size=[128, 1, 1])
  print(prg.global_size, prg.local_size)
  ei = ExecItem(CompiledRunner(prg), [x.ensure_allocated() for x in si.bufs], si.metadata)
  with Context(DEBUG=2):
    for _ in range(run_count): ei.run(wait=True)
  nc = c.numpy()
  np.testing.assert_allclose(na@nb, nc, rtol=1e-5)
