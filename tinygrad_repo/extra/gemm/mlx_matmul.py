import mlx.core as mx
from tinygrad.helpers import Timing
N = 4096
x = mx.random.normal((N,N))
w = mx.random.normal((N,N))

FLOPS = N*N*N*2
for i in range(10):
  with Timing("", lambda x: f"  {FLOPS/x:.2f} GFLOPS"):
    mx.eval(x@w)
