import time
import torch

for dtype in [torch.float16, torch.float32]:
  for N in [256, 512, 1024, 2048, 4096]:
    FLOPS = N*N*N*2

    b = torch.rand((N,N), dtype=dtype).cuda()
    c = torch.rand((N,N), dtype=dtype).cuda()

    def torch_prog(b, c):
      st = time.perf_counter()
      a = b@c
      torch.cuda.synchronize()
      return time.perf_counter() - st
    tm = min([torch_prog(b, c) for _ in range(20)])
    print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS {N:4d}x{N:4d}x{N:4d} matmul in {dtype}")
