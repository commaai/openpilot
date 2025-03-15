from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import getenv, DEBUG
from tinygrad.codegen.kernel import Kernel, Opt, OptOps
from tinygrad.engine.realize import CompiledRunner, ExecItem
from dataclasses import replace

N = 4096
if __name__ == "__main__":
  if getenv("GEMV"):
    A, B = Tensor.empty(1, N, dtype=dtypes.float), Tensor.empty(14336, N, dtype=dtypes.float16).T
  else:
    A, B = Tensor.empty(N, N, dtype=dtypes.float16), Tensor.empty(N, N, dtype=dtypes.float16)
  C = A.matmul(B)
  si = C.schedule()[-1]
  ast = si.ast
  k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
  if getenv("GEMV"):
    opts = [
      Opt(op=OptOps.UNROLL, axis=0, amt=8),
      Opt(op=OptOps.GROUP, axis=0, amt=32),
    ]
  else:
    opts = [
      Opt(op=OptOps.TC, axis=0, amt=0),
      Opt(op=OptOps.UPCAST, axis=0, amt=4),
      Opt(op=OptOps.UPCAST, axis=1, amt=8),
      Opt(op=OptOps.LOCAL, axis=0, amt=2),
      Opt(op=OptOps.LOCAL, axis=1, amt=2),
      Opt(op=OptOps.LOCAL, axis=0, amt=2),
    ]
  for opt in opts: k.apply_opt(opt)
  prg = k.to_program()
  new_src = prg.src
  # can mod source here
  prg = replace(prg, src=new_src)
  ei = ExecItem(CompiledRunner(prg), [x.ensure_allocated() for x in si.bufs], si.metadata)
  for i in range(5): ei.run(wait=True)
