# https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html#example-2-manually-optimizing-matrix-multiplication-with-te

M, N, K = 1024, 1024, 1024

try:
  import tvm
  from tvm import te
  #print(tvm.target.Target.list_kinds())

  # c, opencl
  target = tvm.target.Target(target="c")

  # TVM Matrix Multiplication using TE
  k = te.reduce_axis((0, K), "k")
  A = te.placeholder((M, K), name="A")
  B = te.placeholder((K, N), name="B")
  C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

  # Default schedule
  s = te.create_schedule(C.op)
  #print(tvm.lower(s, [A, B, C], simple_mode=True))

  # Output C code
  func = tvm.build(s, [A, B, C], target=target, name="mmult")
  print(func.get_source())
except ImportError:
  print("** please install TVM for TVM output")

# tinygrad version

import os
from tinygrad.tensor import Tensor

# define the compute
A = Tensor.rand(M, K, device="CPU")
B = Tensor.rand(K, N, device="CPU")
C = (A.reshape(M, 1, K) * B.permute(1,0).reshape(1, N, K)).sum(axis=2)

sched = C.schedule()
from tinygrad.codegen.kernel import Kernel
from tinygrad.device import CompilerOptions
lin = Kernel(sched[-1].ast, CompilerOptions(has_local=False, supports_float4=False))
lin.linearize()
from tinygrad.runtime.ops_cpu import renderer
src = renderer("mmult", lin.uops)
print(src)
