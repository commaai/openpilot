import numpy as np, os
from tinygrad.helpers import getenv, flat_mv
from tinygrad import dtypes
from typing import Optional, List, Tuple, cast, Dict, Final, DefaultDict, Self
from tinygrad.engine.realize import get_program

# for copied uops
from tinygrad.codegen.opt.kernel import Kernel, KernelOptError
from tinygrad.uop.ops import UOp, Ops, BinaryOps, UnaryOps, TernaryOps, KernelInfo
from tinygrad.codegen.opt.search import Opt, OptOps
from tinygrad import Device, dtypes, Tensor
from tinygrad.dtype import PtrDType, DType, DTYPES_DICT
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View

script_dir = os.path.dirname(os.path.abspath(__file__))

# problem variations
DTYPE_IN = DTYPES_DICT[getenv("DTYPE_IN", "half")]
DTYPE_OUT = DTYPES_DICT[getenv("DTYPE_OUT", "half")]
DTYPE_ACC = DTYPES_DICT[getenv("DTYPE_ACC", "float")]
N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)
CNT = getenv("CNT", 10)
ATOL = getenv("ATOL", 5e-3 if DTYPE_IN == dtypes.float else 1e-2)
RTOL = getenv("RTOL", 1e-4 if DTYPE_IN == dtypes.float else 1e-3)
FLOPS = M * N * K * 2
BW = 2 * ((M*K) + (K*N) + (M*N))

# algorithm variations
INPUT = getenv("INPUT", "RAND")
GEMM_VARIATION = getenv("GEMM_VARIATION", "nv_hcopt")

def randoms():
  if INPUT == "RAND":
    na = np.random.default_rng().normal(scale=1.0, size=(M,K)).astype(dtype=np.float32)
    nb = np.random.default_rng().normal(scale=1.0, size=(K,N)).astype(dtype=np.float32)
  elif INPUT == "IDENTITY" and M==N==K:
    na = np.identity(K, dtype=np.float32)
    nb = np.identity(K, dtype=np.float32)
  elif INPUT == "OUTPUTONES" and M==K:
    na = np.identity(K, dtype=np.float32)
    nb = np.ones((K,N), dtype=np.float32)
  else:
    na = np.ones((M,K), dtype=np.float32)
    nb = np.ones((K,N), dtype=np.float32)
  nc = np.zeros(M*N, np.float32)
  if DTYPE_IN != dtypes.float:
    na = na.astype(np.bfloat16 if DTYPE_IN == dtypes.bfloat16 else np.float16)
    nb = nb.astype(np.bfloat16 if DTYPE_IN == dtypes.bfloat16 else np.float16)
  if DTYPE_OUT != dtypes.float:
    nc = nc.astype(np.bfloat16 if DTYPE_IN == dtypes.bfloat16 else np.float16)
  return na, nb, nc

def ast_to_cuda_prog(compiler, ast, opts):
  k = Kernel(ast)
  k.apply_opts(opts)
  p = get_program(k.get_optimized_ast(), k.opts)
  return CUDAProgram(device, p.function_name, compiler.compile(p.src))

if __name__ == "__main__":
  print(f"gemm variation: {GEMM_VARIATION=} {M=} {N=} {K=} {DTYPE_IN=} {DTYPE_OUT=} {DTYPE_ACC=}")
  prog, global_size, local_size = None, None, None

  if getenv("CUDA") == 1:
    from tinygrad.runtime.ops_cuda import CUDAAllocator, CUDADevice, CUDAProgram, CUDACompiler
    device = CUDADevice("cuda:0")
    compiler = CUDACompiler(device.arch)
    cudaalloc = CUDAAllocator(device)

    a = cudaalloc.alloc(M*K*DTYPE_IN.itemsize)
    b = cudaalloc.alloc(K*N*DTYPE_IN.itemsize)
    c = cudaalloc.alloc(M*N*DTYPE_OUT.itemsize)

    if GEMM_VARIATION == "max" and (M%64)==0 and (N%128)==0 and (K%64)==0 and DTYPE_IN == dtypes.half and DTYPE_OUT == dtypes.float and DTYPE_ACC == dtypes.float:
      print("Using CUDA and triton-generated kernel")
      # See nv_triton_gemm.annotated.ptx for PTX code which was generated from `PYTHONPATH=. DEBUG=6 CUDA=1 PTX=1 python3 extra/gemm/triton_nv_matmul.py`
      # this kernel with M=N=K=4096 does 162TFLOPS, vs torch at 144TFLOPS and BEAM=8 tinygrad at 138TFLOPS.  theo max is 165TFLOPS.

      # WMMA element size is (M, N, K) = (16, 8, 16)
      # warpgroup size in WMMA tiles is (B_M, B_N, B_K) = (2, 8, 4) so 64 HMMA calls per threadgroup reduce iteration
      # thread block size is (T_M, T_N, T_K) = (2, 2, 1), i.e. macro blocks in M and N, so 256 HMMA calls per kernel reduce iteration
      # kernel reduce iteration size in elements = (64, 128, 64)
      # single iteration SMEM_A = (64 * 64) * (2 bytes / half) =  8192 bytes, SMEM_B = (128 * 64) * (2 bytes / half) = 16384 bytes
      # double-buffer smem = (8192 + 16384) * 2 = 49152 bytes
      # reduce for_loop size = [1, 1, (4096 // 16 // 4)==64]
       # NOTE: T_K > 0 would be group_for_reduce
      prog = CUDAProgram(device, "wmma_example", compiler.compile(open(os.path.join(script_dir, 'max_kernels/nv.fp16_fp32_fp32.max.cu')).read()))
      args = (c, a, b)
      kwargs = {
        'global_size': [M//64, N//128, 1],
        'local_size': [128, 1, 1], # 4 warpgroups == (T_M:=2) * (T_N:=2)
        'wait': True,
        'vals': (N, K),
      }
    elif GEMM_VARIATION == "2_stage_swizzled_smem_input" and (M%64)==0 and (N%128)==0 and (K%64)==0 and DTYPE_IN == dtypes.half and DTYPE_OUT == dtypes.float and DTYPE_ACC == dtypes.float:
      print("Using CUDA, 2-stage reduce pipeline, swizzled SMEM inputs")
      prog = CUDAProgram(device, "wmma_example", compiler.compile(open(os.path.join(script_dir, 'max_kernels/nv.fp16_fp32_fp32.2_stage_swizzled_smem_input.cu')).read()))
      args = (c, a, b)
      kwargs = {
        'global_size': [M//64, N//128, 1],
        'local_size': [128, 1, 1], # 4 warpgroups == (T_M:=2) * (T_N:=2)
        'wait': True,
        'vals': (N, K),
      }
    elif GEMM_VARIATION == "swizzled_smem_input" and (M%64)==0 and (N%128)==0 and (K%64)==0 and DTYPE_IN == dtypes.half and DTYPE_OUT == dtypes.float and DTYPE_ACC == dtypes.float:
      print("Using CUDA, swizzled SMEM inputs")
      prog = CUDAProgram(device, "wmma_example", compiler.compile(open(os.path.join(script_dir, 'max_kernels/nv.fp16_fp32_fp32.swizzled_smem_input.cu')).read()))
      args = (c, a, b)
      kwargs = {
        'global_size': [M//64, N//128, 1],
        'local_size': [128, 1, 1], # 4 warpgroups == (T_M:=2) * (T_N:=2)
        'wait': True,
        'vals': (N, K),
      }
    elif GEMM_VARIATION == "flat_smem_input" and (M%64)==0 and (N%128)==0 and (K%64)==0 and DTYPE_IN == dtypes.half and DTYPE_OUT == dtypes.float and DTYPE_ACC == dtypes.float:
      print("Using CUDA, flat SMEM inputs")
      prog = CUDAProgram(device, "wmma_example", compiler.compile(open(os.path.join(script_dir, 'max_kernels/nv.fp16_fp32_fp32.flat_smem_input.cu')).read()))
      args = (c, a, b)
      kwargs = {
        'global_size': [M//64, N//128, 1],
        'local_size': [128, 1, 1], # 4 warpgroups == (T_M:=2) * (T_N:=2)
        'wait': True,
        'vals': (N, K),
      }
    elif GEMM_VARIATION == "hcopt" and M == N == K == 4096 and DTYPE_IN == dtypes.half and DTYPE_OUT == dtypes.half and DTYPE_ACC == dtypes.float:
      print("Using CUDA and generated hcopt")
      # [Opt(op=OptOps.TC, axis=0, amt=0), Opt(op=OptOps.UPCAST, axis=0, amt=4), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.LOCAL, axis=1, amt=4)]
      prog = CUDAProgram(device, "wmma_example", compiler.compile(open(os.path.join(script_dir, 'max_kernels/nv.fp16_fp32_fp16.hcopt.cu')).read()))
      args = (c, a, b)
      kwargs = {
        'global_size': [32, 64, 1],
        'local_size': [16, 2, 4], # 16,2 are warp, 4 workgroups upcasted to axis=1
        'wait': True,
      }
    elif GEMM_VARIATION == "2_stage" and (M%64)== 0 and (N%128)==0 and (K%64)==0 and DTYPE_IN == dtypes.half and DTYPE_OUT == dtypes.half and DTYPE_ACC == dtypes.half:
      print("Using CUDA and un-optimized 2-stage, swizzled SMEM inputs and direct acc to output kernel")
      prog = CUDAProgram(device, "wmma_example", compiler.compile(open(os.path.join(script_dir, 'max_kernels/nv.fp16_fp16_fp16.2_stage.cu')).read()))
      args = (c, a, b)
      kwargs = {
        'global_size': [M//64, N//128, 1],
        'local_size': [128, 1, 1], # 4 warpgroups == (T_M:=2) * (T_N:=2)
        'wait': True,
        'vals': (N, K),
      }
    elif GEMM_VARIATION == "3_stage" and (M%256)== 0 and (N%128)==0 and (K%32)==0 and DTYPE_IN == dtypes.half and DTYPE_OUT == dtypes.half and DTYPE_ACC == dtypes.half:
      print("Using CUDA and 3-stage (interleave global copies and ldmatrix)")
      prog = CUDAProgram(device, "wmma_example", compiler.compile(open(os.path.join(script_dir, 'max_kernels/nv.fp16_fp16_fp16.3_stage.cu')).read()), 73728)
      args = (c, a, b)
      kwargs = {
        'global_size': [M//256, N//128, 1],
        'local_size': [32, 4, 2], # 8 warpgroups, WG_M=4 and WG_N=2
        'wait': True,
        'vals': (N, K),
      }
    elif GEMM_VARIATION == "3_stage_swizzled" and (M%256)== 0 and (N%128)==0 and (K%32)==0 and DTYPE_IN == dtypes.half and DTYPE_OUT == dtypes.half and DTYPE_ACC == dtypes.half:
      print("Using CUDA and 3-stage (interleave global copies and ldmatrix) and swizzled SMEM inputs")
      prog = CUDAProgram(device, "wmma_example", compiler.compile(open(os.path.join(script_dir, 'max_kernels/nv.fp16_fp16_fp16.3_stage_swizzled.cu')).read()), 73728)
      args = (c, a, b)
      kwargs = {
        'global_size': [M//256, N//128, 1],
        'local_size': [32, 4, 2], # 8 warpgroups, WG_M=4 and WG_N=2
        'wait': True,
        'vals': (N, K),
      }
    elif GEMM_VARIATION == "max" and (M%256)== 0 and (N%128)==0 and (K%32)==0 and DTYPE_IN == dtypes.half and DTYPE_OUT == dtypes.half and DTYPE_ACC == dtypes.half:
      print("Using CUDA and 3-stage (interleave global copies and ldmatrix), swizzled SMEM inputs and epilogue")
      prog = CUDAProgram(device, "wmma_example", compiler.compile(open(os.path.join(script_dir, 'max_kernels/nv.fp16_fp16_fp16.max.cu')).read()), 73728)
      args = (c, a, b)
      kwargs = {
        'global_size': [M//256, N//128, 1],
        'local_size': [32, 4, 2], # 8 warpgroups, WG_M=4 and WG_N=2
        'wait': True,
        'vals': (N, K),
      }
    elif GEMM_VARIATION == "no_xor" and (M%256)== 0 and (N%128)==0 and (K%32)==0 and DTYPE_IN == dtypes.half and DTYPE_OUT == dtypes.half and DTYPE_ACC == dtypes.half:
      print("Using CUDA and 3-stage (interleave global copies and ldmatrix), swizzled SMEM inputs and epilogue")
      prog = CUDAProgram(device, "wmma_example", compiler.compile(open(os.path.join(script_dir, 'max_kernels/nv.fp16_fp16_fp16.no_xor.cu')).read()), 73728)
      args = (c, a, b)
      kwargs = {
        'global_size': [M//256, N//128, 1],
        'local_size': [32, 4, 2], # 8 warpgroups, WG_M=4 and WG_N=2
        'wait': True,
        'vals': (N, K),
      }
    else:
      raise RuntimeError(f"invalid gemm variation: {GEMM_VARIATION=} {M=} {N=} {K=} {DTYPE_IN=} {DTYPE_OUT=} {DTYPE_ACC=}")

    tms = []
    na, nb, nc = randoms()
    cudaalloc.copyin(a, bytearray(na))
    cudaalloc.copyin(b, bytearray(nb))
    for i in range(CNT):
      tms.append(prog(*args, **kwargs))
    cudaalloc.copyout(flat_mv(nc.data), c)
    comp = na.astype(np.float32) @ nb.astype(np.float32)
    result = nc.reshape(M, N).astype(np.float32)

    print(f"{N*N:10d} {min(tms)*1e6:9.2f} us, would be {FLOPS*1e-9/min(tms):9.2f} GFLOPS matmul, {BW*1e-9/min(tms):.2f} GB/s")
    try:
      np.testing.assert_allclose(result, comp, atol=ATOL, rtol=RTOL)
    except AssertionError as e:
      if getenv("DEBUG_VALUES") > 0:
        indices = np.where(~np.isclose(result, comp, rtol=RTOL, atol=ATOL))
        non_matching_elements_result = result[indices]
        non_matching_elements_comp = comp[indices]
        print("valid       :", np.where(np.isclose(result, comp, rtol=RTOL, atol=ATOL)))
        print("invalid     :", indices)
        print("result      :", non_matching_elements_result)
        print("ground truth:", non_matching_elements_comp)
        print("result sum  :", np.sum(result))
        print("ground sum  :", np.sum(comp))
      raise e

    if getenv("DEBUG_VALUES") > 0:
      print(comp)
      print("ground sum  :", np.sum(comp))
      print(result)
      print("result sum  :", np.sum(result))

  elif getenv("AMD") == 1:
    # note: https://hipfft.readthedocs.io/en/rocm-6.1.2/how-to/fine-tuning-llms/optimizing-triton-kernel.html

    # also this is different than the rocblas/tensile approach to GEMM
    # see: https://github.com/ROCm/Tensile/blob/develop/Tensile/KernelWriterAssembly.py
    raise RuntimeError("invalid max_matmul device")

  else:
    raise RuntimeError("invalid max_matmul device")

