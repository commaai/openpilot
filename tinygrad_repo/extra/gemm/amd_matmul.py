# kernel8_batched_gmem.s from https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html
# sudo PATH=/opt/homebrew/Cellar/llvm/20.1.6/bin:$PATH AMD_LLVM=0 AMD=1 DEBUG=2 python3 extra/gemm/amd_matmul.py
import pathlib
from tinygrad import Tensor, Device, Context, GlobalCounters
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from tinygrad.engine.realize import run_linear

N = 4096
run_count = 5

def make_matmul_kernel(name:str, src:str, local_size:int):
  def fxn(a:UOp, b:UOp, c:UOp) -> UOp:
    threads = UOp.special(local_size, "lidx0")
    wg_x = UOp.special(N//128, "gidx0")
    wg_y = UOp.special(N//128, "gidx1")
    sink = UOp.sink(a.base, b.base, c.base, threads, wg_x, wg_y, arg=KernelInfo(name, estimates=Estimates(ops=2*N**3, mem=3*N*N*4)))
    lib = Device[Device.DEFAULT].compiler.compile_cached(src)
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                                 UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))
  return fxn

if __name__ == "__main__":
  if getenv("ASM") == 1:
    src = (pathlib.Path(__file__).parent / "amd_seb" / "kernel8_batched_gmem.s").read_text()
    name, local_size = "kernel", 128
  elif getenv("ASM") == -1:
    src = (pathlib.Path(__file__).parent / "amd_seb" / "kernel3_registers.cpp").read_text()
    name, local_size = "kernel3_registers", 256
  elif getenv("ASM") == -2:
    src = (pathlib.Path(__file__).parent / "amd_seb" / "kernel4_gmem_df.cpp").read_text()
    name, local_size = "kernel4_gmem_db", 256
  else:
    src = (pathlib.Path(__file__).parent / "amd_seb" / "kernel5_lds_optim.cpp").read_text()
    name, local_size = "kernel5_lds_optim", 128

  a = Tensor.randn(N, N).realize()
  b = Tensor.randn(N, N).realize()
  c = Tensor.zeros(N, N).contiguous().realize()

  GlobalCounters.reset()
  with Context(DEBUG=2):
    for _ in range(run_count): tc = (a@b).realize()

  linear = Tensor.custom_kernel(a, b, c, fxn=make_matmul_kernel(name, src, local_size))[2].schedule_linear()
  GlobalCounters.reset()
  with Context(DEBUG=2):
    for _ in range(run_count): run_linear(linear)
  print(f"custom  {(c-tc).square().mean().item()}")
