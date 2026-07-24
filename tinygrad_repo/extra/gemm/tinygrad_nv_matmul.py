from tinygrad import Tensor, dtypes, Context
from tinygrad.helpers import getenv
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.engine.realize import run_linear
from dataclasses import replace

N = 4096
if __name__ == "__main__":
  if getenv("GEMV"):
    A, B = Tensor.empty(1, N, dtype=dtypes.float), Tensor.empty(14336, N, dtype=dtypes.float16).T
  else:
    A, B = Tensor.empty(N, N, dtype=dtypes.float16), Tensor.empty(N, N, dtype=dtypes.float16)
  C = A.matmul(B)
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
  linear = C.schedule_linear()
  call = linear.src[-1]
  new_ast = call.src[0].replace(arg=replace(call.src[0].arg, opts_to_apply=tuple(opts)))
  new_call = call.replace(src=(new_ast, *call.src[1:]))
  linear = linear.replace(src=tuple(new_call if c is call else c for c in linear.src))
  with Context(DEBUG=2):
    for i in range(5): run_linear(linear)
