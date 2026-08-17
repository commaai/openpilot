import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
from extra.gemm.cdna_asm_gemm import quantize_mxfp8, _mx_block_scale, _mx_block_scale_3d

@functools.cache
def custom_hk_grouped_mxfp8_gemm(C:UOp, A:UOp, B:UOp, scale_A:UOp, scale_B:UOp, *extra:UOp, dname:str, n_experts:int) -> UOp:
  M, K = A.shape
  E, N, K2 = B.shape
  assert K == K2, f"{A.shape} {B.shape}"
  assert E == n_experts, f"{E} != {n_experts}"
  threads = UOp.special(64 * 8, "lidx0")
  workgroups = UOp.special((M // 256) * (N // 256), "gidx0")
  sink_inputs = (C.base, A.base, B.base, scale_A.base, scale_B.base, extra[0].base, extra[1].base, extra[2].base, threads, workgroups)
  sink = UOp.sink(*sink_inputs,
                  arg=KernelInfo(f"hk_grouped_mxfp8_gemm_{E}_{M}_{N}_{K}",
                                 estimates=Estimates(ops=2*M*N*K, mem=(M*K+E*N*K)*A.dtype.itemsize+M*N*C.dtype.itemsize)))
  kittens_path = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (kittens_path/"grouped_mxfp8_gemm.cpp").read_text()
  lib = HIPCCCompiler("gfx950", [f"-I{(kittens_path/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-ffast-math",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}",
                                 f"-DGEMM_E={E}"]).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=lib)))

@functools.cache
def custom_hk_grouped_mxfp8_wgrad(C:UOp, A:UOp, B:UOp, scale_A:UOp, scale_B:UOp, expert_off:UOp, *, dname:str, n_experts:int) -> UOp:
  N, M = A.shape
  K, M2 = B.shape
  assert M == M2, f"{A.shape} {B.shape}"
  E = n_experts
  threads = UOp.special(64 * 8, "lidx0")
  workgroups = UOp.special(E * (N // 256) * (K // 256), "gidx0")
  sink = UOp.sink(C.base, A.base, B.base, scale_A.base, scale_B.base, expert_off.base, threads, workgroups,
                  arg=KernelInfo(f"hk_grouped_mxfp8_wgrad_{E}_{M}_{N}_{K}",
                                 estimates=Estimates(ops=2*M*N*K, mem=(N*M+K*M)*A.dtype.itemsize+E*N*K*C.dtype.itemsize)))
  kittens_path = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (kittens_path/"grouped_mxfp8_wgrad.cpp").read_text()
  lib = HIPCCCompiler("gfx950", [f"-I{(kittens_path/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-ffast-math",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DWGRAD_M={M}", f"-DWGRAD_N={N}", f"-DWGRAD_K={K}",
                                 f"-DWGRAD_E={E}"]).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=lib)))

def grouped_mx_wgrad(g:Tensor, xg:Tensor, expert_off:Tensor, n_experts:int) -> Tensor:
  from extra.llama_kernels.transpose_quantize_mxfp8 import transpose_quantize_mxfp8
  M, N = g.shape
  M2, K = xg.shape
  assert M == M2, f"{g.shape} {xg.shape}"
  assert M % 128 == 0 and N % 256 == 0 and K % 256 == 0, f"wgrad needs M%128,N%256,K%256, got {g.shape} {xg.shape}"
  gT, _, g_si = transpose_quantize_mxfp8(g.contiguous())
  xT, _, x_si = transpose_quantize_mxfp8(xg.contiguous())
  dname = (g.device[0] if isinstance(g.device, tuple) else g.device).split(":")[0]
  is_multi = isinstance(g.device, tuple)
  inv = Tensor.invalids(1, n_experts * N, K, dtype=dtypes.bfloat16, device=g.device)
  out = Tensor(inv.uop.unshard(0), device=g.device) if is_multi else inv
  out = Tensor.custom_kernel(out, gT, xT, g_si, x_si, expert_off,
                             fxn=functools.partial(custom_hk_grouped_mxfp8_wgrad, dname=dname, n_experts=n_experts))[0]
  out = out.sum(0) if is_multi else out.squeeze(0)
  return out.reshape(n_experts, N, K)

def mx_pack_3d(e8:Tensor) -> Tensor:
  E, rows, scale_K = e8.shape
  return e8.reshape(E, rows, scale_K // 4, 4).bitcast(dtypes.uint32).reshape(E, rows, scale_K // 4).permute(0, 2, 1).contiguous()

@functools.cache
def custom_grouped_mx_gemm_bw(gradient:UOp, kernel:UOp, w_stored:bool=False) -> tuple:
  inputs = kernel.src[1:]
  aq = Tensor(inputs[1], device=inputs[1].device)
  bq = Tensor(inputs[2], device=inputs[2].device)
  ae8 = Tensor(inputs[5], device=inputs[5].device)
  be8 = Tensor(inputs[6], device=inputs[6].device)
  E, N = bq.shape[0], bq.shape[1]
  M, K = aq.shape
  g = Tensor(gradient, device=aq.device).reshape(M, N).cast(dtypes.bfloat16)
  x_phys = (aq.cast(dtypes.bfloat16) * _mx_block_scale(ae8).cast(dtypes.bfloat16))
  w_phys = (bq.cast(dtypes.bfloat16) * _mx_block_scale_3d(be8).cast(dtypes.bfloat16))
  expert_off = Tensor(inputs[7], device=inputs[7].device)
  grad_x = grouped_mx_gemm(g, w_phys.transpose(1, 2), expert_off)
  grad_w = grouped_mx_wgrad(g, x_phys, expert_off, E)
  grad_xq = grad_x * _mx_block_scale(ae8).cast(dtypes.bfloat16)
  grad_wq = grad_w.contiguous() if w_stored else (grad_w * _mx_block_scale_3d(be8).cast(dtypes.bfloat16)).contiguous()
  return (None, grad_xq.uop, grad_wq.uop) + tuple(None for _ in inputs[3:])

_grouped_bw_stored = functools.partial(custom_grouped_mx_gemm_bw, w_stored=True)

def grouped_mx_gemm(x:Tensor, w:Tensor|tuple[Tensor, Tensor], expert_off:Tensor) -> Tensor:
  if (pre_quantized := isinstance(w, tuple)):
    w_q, w_e8 = w
    E, N, K2 = w_q.shape
  else:
    E, N, K2 = w.shape
  M, K = x.shape
  assert K == K2, f"shape mismatch {x.shape} {w.shape}"
  assert M % 256 == 0 and N % 256 == 0 and K % 128 == 0, f"grouped mxfp8 needs M%256,N%256,K%128, got {x.shape} {w.shape}"
  dname = (x.device[0] if isinstance(x.device, tuple) else x.device).split(":")[0]
  x_q, x_e8, x_si = quantize_mxfp8(x)
  if not pre_quantized: w_q, w_e8, _ = quantize_mxfp8(w)
  w_si = mx_pack_3d(w_e8)
  xe_in, out_shape = x_e8.reshape(M, K // 32), (M, N)
  if isinstance(x.device, tuple) and (row_axis := x.uop.axis) is not None:
    ndev = len(x.device)
    out = Tensor(Tensor.invalids(*(s // ndev if i == row_axis else s for i, s in enumerate(out_shape)),
                                 dtype=dtypes.bfloat16, device=x.device).uop.unshard(row_axis), device=x.device)
  else:
    out = Tensor.invalids(*out_shape, dtype=dtypes.bfloat16, device=x.device)
  return Tensor.custom_kernel(out, x_q, w_q, x_si, w_si, xe_in, w_e8, expert_off,
                              fxn=functools.partial(custom_hk_grouped_mxfp8_gemm, dname=dname, n_experts=E),
                              grad_fxn=(_grouped_bw_stored if pre_quantized else custom_grouped_mx_gemm_bw))[0]
