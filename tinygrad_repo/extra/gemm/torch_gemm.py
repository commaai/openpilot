import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import time
import torch
torch.set_num_threads(1)
from tinygrad.helpers import getenv
CUDA = getenv("CUDA", 1)

for dtype in [torch.float32, torch.float16]:
  for N in [256, 512, 1024, 2048, 4096]:
    FLOPS = N*N*N*2

    b = torch.rand((N,N), dtype=dtype)
    c = torch.rand((N,N), dtype=dtype)
    if CUDA: b,c = b.cuda(),c.cuda()

    def torch_prog(b, c):
      st = time.perf_counter()
      a = b@c
      if CUDA: torch.cuda.synchronize()
      return time.perf_counter() - st
    tm = min([torch_prog(b, c) for _ in range(20)])
    print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS {N:4d}x{N:4d}x{N:4d} matmul in {dtype}")
