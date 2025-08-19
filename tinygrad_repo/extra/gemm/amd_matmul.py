# kernel8_batched_gmem.s from https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html
# sudo PATH=/opt/homebrew/Cellar/llvm/20.1.6/bin:$PATH AMD_LLVM=0 AMD=1 DEBUG=2 python3 extra/gemm/amd_matmul.py
import pathlib
from dataclasses import replace
from tinygrad import Tensor, Device, Context, GlobalCounters
from tinygrad.helpers import getenv
from tinygrad.engine.realize import CompiledRunner, ExecItem, get_program

N = 4096
run_count = 5

if __name__ == "__main__":
  ast = (Tensor.empty(N, N)@Tensor.empty(N, N)).schedule()[-1].ast
  prg = get_program(ast, Device.default.renderer)

  if getenv("ASM") == 1:
    src = (pathlib.Path(__file__).parent / "amd_seb" / "kernel8_batched_gmem.s").read_text()
    prgfast = replace(prg, name="kernel", src=src, global_size=[N//128, N//128, 1], local_size=[128, 1, 1])
  elif getenv("ASM") == -1:
    src = (pathlib.Path(__file__).parent / "amd_seb" / "kernel3_registers.cpp").read_text()
    prgfast = replace(prg, name="kernel3_registers", src=src, global_size=[N//128, N//128, 1], local_size=[256, 1, 1])
  elif getenv("ASM") == -2:
    src = (pathlib.Path(__file__).parent / "amd_seb" / "kernel4_gmem_df.cpp").read_text()
    prgfast = replace(prg, name="kernel4_gmem_db", src=src, global_size=[N//128, N//128, 1], local_size=[256, 1, 1])
  else:
    src = (pathlib.Path(__file__).parent / "amd_seb" / "kernel5_lds_optim.cpp").read_text()
    prgfast = replace(prg, name="kernel5_lds_optim", src=src, global_size=[N//128, N//128, 1], local_size=[128, 1, 1])
  runner = CompiledRunner(prgfast)

  a = Tensor.randn(N, N).realize()
  b = Tensor.randn(N, N).realize()
  c = Tensor.zeros(N, N).contiguous().realize()

  GlobalCounters.reset()
  with Context(DEBUG=2):
    for _ in range(run_count): tc = (a@b).realize()

  GlobalCounters.reset()
  ei = ExecItem(runner, [a.uop.buffer, b.uop.buffer, c.uop.buffer])
  with Context(DEBUG=2):
    for _ in range(run_count): ei.run(wait=True)
  print(f"custom  {(c-tc).square().mean().item()}")
