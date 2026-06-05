import time
import triton
import triton.language as tl
from triton.compiler import AttrsDescriptor, ASTSource, compile as triton_compile
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.engine.realize import get_runtime
from tinygrad.codegen import to_program
from tinygrad.uop.ops import Ops, UOp, KernelInfo, ProgramInfo
from tinygrad.helpers import getenv
np.set_printoptions(suppress=True)

@triton.jit
def matmul_kernel(c_ptr, a_ptr, b_ptr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
  pid_m = tl.program_id(axis=0)
  pid_n = tl.program_id(axis=1)

  M, N, K = 4096, 4096, 4096
  stride_am = 4096
  stride_ak = 1
  stride_bk = 4096
  stride_bn = 1
  stride_cm = 4096
  stride_cn = 1

  offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
  offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
  offs_k = tl.arange(0, BLOCK_SIZE_K)

  a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
  b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

  accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
  for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk

  c = tl.cast(accumulator, tl.float16)
  offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
  c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
  tl.store(c_ptrs, c)

# CUDA=1 CUDA_PTX=1 python3 extra/gemm/triton_nv_matmul.py
if __name__ == "__main__":
  BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 64
  M, N, K = 4096, 4096, 4096

  # **** torch test ****

  if getenv("TORCH"):
    import torch
    c = torch.empty((M, N), device='cuda:0', dtype=torch.float16)
    a = torch.empty((M, K), device='cuda:0', dtype=torch.float16)
    b = torch.empty((K, N), device='cuda:0', dtype=torch.float16)

    for i in range(5):
      st = time.perf_counter()
      matmul_kernel[triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N)](
        c, a, b, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
      torch.cuda.synchronize()
      et = time.perf_counter() - st
      print(f"TFLOPS {2*M*N*K*1e-12/et:.2f}")

  # **** tinygrad test ****

  compiled = triton_compile(ASTSource(matmul_kernel, "*fp16,*fp16,*fp16",
                            attrs=AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=()),
                            constants={"BLOCK_SIZE_M": BLOCK_SIZE_M, "BLOCK_SIZE_N": BLOCK_SIZE_N, "BLOCK_SIZE_K": BLOCK_SIZE_K}))
  print(compiled.metadata)

  A, B = Tensor.normal(M, K, std=1e-1, dtype=dtypes.float16).realize(), Tensor.normal(K, N, std=1e-1, dtype=dtypes.float16).realize()
  C = A.matmul(B)
  from tinygrad.uop.ops import Ops
  linear, var_vals = C.linear_with_vars()
  last_call = linear.src[-1]
  ast = last_call.src[0]
  bufs = [s.buffer for s in last_call.src[1:] if s.op is not Ops.BIND]

  src = compiled.asm["ptx"]
  # specify the shared memory here so we don't need to do it dynamically
  src = src.replace(".extern .shared .align 16 .b8 global_smem[];", f".shared .align 16 .b8 global_smem[{compiled.metadata.shared}];")
  # useless comment spam
  src = src.replace("\t// begin inline asm\n", "")
  src = src.replace("\t// end inline asm\n", "")
  # remove debug sections
  src = src.split("\t.file")[0]
  assert '.extern .shared' not in src
  info = ProgramInfo(name="matmul_kernel",
                     global_size=(M//BLOCK_SIZE_M, N//BLOCK_SIZE_N, 1), local_size=(32*compiled.metadata.num_warps, 1, 1))
  sink = UOp.sink(arg=KernelInfo(name="matmul_kernel"))
  prg_uop = to_program(UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=src)), arg=info),
                       Device.default.renderer)
  rt = get_runtime(Device.DEFAULT, prg_uop)
  all_bufs = [x.ensure_allocated() for x in bufs]
  prg_bufs = [all_bufs[i] for i in info.globals]
  gsize, lsize = info.launch_dims({})
  tflops = []
  for i in range(5):
    tm = rt(*[b._buf for b in prg_bufs], global_size=gsize, local_size=lsize, vals=info.vals({}), wait=True)
    tflops.append((2*M*K*N/tm)*1e-12)
  print(f"TFLOPS: {max(tflops):.2f}")

  # check correctness
  if getenv("VERIFY"):
    from tinygrad.engine.realize import run_linear
    triton_buf = np.frombuffer(si.bufs[0].as_memoryview(), np.float16).reshape(M,N)
    print(triton_buf)
    run_linear(linear, var_vals)
    tinygrad_buf = np.frombuffer(si.bufs[0].as_memoryview(), np.float16).reshape(M,N)
    print(tinygrad_buf)
    np.testing.assert_allclose(triton_buf, tinygrad_buf)
    print("correct!")
