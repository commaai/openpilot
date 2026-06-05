import pathlib
from tinygrad import Device, Tensor
from tinygrad.helpers import Context, getenv
from tinygrad.runtime.support.compiler_cuda import pretty_ptx, NVCCCompiler

if __name__ == "__main__":
  if getenv("MATMUL2"):
    code = (pathlib.Path(__file__).parent / "matmul2.cu").read_text()
  else:
    code = (pathlib.Path(__file__).parent / "matmul.cu").read_text()

  device = Device["CUDA"]
  kitten_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "--expt-relaxed-constexpr"]
  lib = NVCCCompiler(device.compiler.arch, kitten_args).compile(code)
  kernel_name = lib.decode().split(".globl\t")[1].split("\n")[0]
  print("kernel name", kernel_name)
  print(pretty_ptx(lib.decode()))

  prg = device.runtime(kernel_name, lib)
  if getenv("MATMUL2"):
    prg.smem = 16384 * 2
  else:
    prg.smem = 10000

  N = 8192
  a = Tensor.randn(N, N, device='CUDA', dtype="bfloat16")
  b = Tensor.randn(N, N, device='CUDA', dtype="bfloat16")
  c = Tensor.empty(N, N, device='CUDA', dtype="bfloat16")
  Tensor.realize(a, b, c)

  WARP_THREADS = 32
  if getenv("MATMUL2"):
    SUPER_N = 2
    SUPER_M = 2
    NUM_WORKERS = SUPER_N * SUPER_M
    BLOCK_SIZE = 32
    gsz = (N // (BLOCK_SIZE * SUPER_N), N // (BLOCK_SIZE * SUPER_M), 1)
  else:
    NUM_WORKERS = 1
    BLOCK_SIZE = 32
    gsz = (N // (BLOCK_SIZE), N // (BLOCK_SIZE), 1)

  for _ in range(5):
    et = prg(c.uop.buffer.ensure_allocated()._buf, a.uop.buffer._buf, b.uop.buffer._buf,
             global_size=gsz, local_size=(NUM_WORKERS*WARP_THREADS,1,1), wait=True)
    print(f"{N*N*N*2/(et*1e9):2f} GFLOPS")

  # print(c.tolist())

  for _ in range(5):
    with Context(DEBUG=2):
      ref = (a@b).realize()

  ref, c = ref.float(), c.float()
  print((ref-c).mean().item(), (ref-c).max().item())
