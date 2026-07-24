import atexit, functools, pathlib
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.renderer import Estimates
from tinygrad.helpers import getenv, all_same, DEBUG
from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
from examples.mlperf.models.flat_llama import FP8_DTYPE, quantize_fp8

TILE_M, TILE_N, TILE_K = 256, 256, 64

# ** FP8 GEMM custom kernel

@functools.cache
def custom_hk_fp8_gemm(C:UOp, A:UOp, B:UOp, *args:UOp, dname:str, scale_mode:int=3) -> UOp:
  # scale_mode: 0=no scale, 1=x only, 2=w only, 3=both
  n_scales = (1 if scale_mode & 1 else 0) + (1 if scale_mode & 2 else 0) + (1 if scale_mode & 4 else 0)
  scales, extra = args[:n_scales], args[n_scales:]
  M, K = A.shape[0]*A.shape[1], A.shape[2]
  N, K2 = B.shape[(1 if B.ndim == 3 else 0):]
  assert K == K2, f"{A.shape} {B.shape}"
  block_size = 256
  threads = UOp.special(64 * 8, "lidx0")
  workgroups = UOp.special((M // block_size) * (N // block_size), "gidx0")
  sink_inputs = (C.base, A.base, B.base) + tuple(s.base for s in scales) + (threads, workgroups)
  sink = UOp.sink(*sink_inputs,
                  arg=KernelInfo(f"hk_fp8_gemm_{M}_{N}_{K}", estimates=Estimates(ops=2*M*N*K, mem=(M*K+N*K)*A.dtype.itemsize+M*N*C.dtype.itemsize)))
  kittens_path = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (kittens_path/"gemm_fp8.cpp").read_text()
  lib = HIPCCCompiler("gfx950", [f"-I{(kittens_path/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-ffast-math",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}",
                                 f"-DSCALE_MODE={scale_mode}"]).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=lib)))

# ** FP8 AtB GEMM custom kernel

@functools.cache
def custom_hk_fp8_atb_gemm(C:UOp, A:UOp, B:UOp, *args:UOp, dname:str, scale_mode:int=5) -> UOp:
  # C = A.T @ B, A and B are physically [K, M] and [K, N].
  n_scales = (1 if scale_mode & 1 else 0) + (1 if scale_mode & 2 else 0) + (1 if scale_mode & 4 else 0)
  scales = args[:n_scales]
  K, M = A.shape[0]*A.shape[1], A.shape[2]
  K2, N = B.shape[0]*B.shape[1], B.shape[2]
  assert K == K2, f"{A.shape} {B.shape}"
  block_m, block_n, block_k, num_warps = 256, 256, 128, 8
  assert M % block_m == 0 and N % block_n == 0 and K % block_k == 0, f"invalid fp8 atb tile {(block_m, block_n, block_k)} for {(M, N, K)}"
  threads = UOp.special(64 * num_warps, "lidx0")
  workgroups = UOp.special((M // block_m) * (N // block_n), "gidx0")
  sink_inputs = (C.base, A.base, B.base) + tuple(s.base for s in scales) + (threads, workgroups)
  sink = UOp.sink(*sink_inputs,
                  arg=KernelInfo(f"hk_fp8_atb_gemm_{M}_{N}_{K}", estimates=Estimates(ops=2*M*N*K, mem=(M*K+N*K)*A.dtype.itemsize+M*N*C.dtype.itemsize)))
  kittens_path = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (kittens_path/"gemm_fp8_atb.cpp").read_text()
  lib = HIPCCCompiler("gfx950", [f"-I{(kittens_path/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-ffast-math",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}",
                                 f"-DSCALE_MODE={scale_mode}"]).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=lib)))

def hk_fp8_atb_gemm(a:Tensor, b:Tensor, x_scale:Tensor|None=None, g_amax:Tensor|None=None) -> Tensor:
  assert a.dtype == b.dtype == FP8_DTYPE, f"expected fp8, got {a.dtype} {b.dtype}"
  assert a.ndim == b.ndim == 3 and a.shape[:2] == b.shape[:2], f"{a.shape} {b.shape}"
  batch, rows, M = a.shape
  N = b.shape[2]
  assert M % TILE_M == 0 and N % TILE_N == 0 and (batch * rows) % 128 == 0, \
    f"fp8 atb shape {a.shape} {b.shape} must produce (M,N,K) multiples of ({TILE_M},{TILE_N},128)"
  is_multi = isinstance(a.device, tuple)
  reduce_out = False
  if is_multi:
    ndev = len(a.device)
    if a.uop.axis in (0, 1) or b.uop.axis in (0, 1): inv, out_axis, reduce_out = Tensor.invalids(1, M, N, dtype=dtypes.bfloat16, device=a.device), 0, True
    elif b.uop.axis == 2: inv, out_axis = Tensor.invalids(1, M, N // ndev, dtype=dtypes.bfloat16, device=a.device), 2
    elif a.uop.axis == 2: inv, out_axis = Tensor.invalids(1, M // ndev, N, dtype=dtypes.bfloat16, device=a.device), 1
    else: inv, out_axis, reduce_out = Tensor.invalids(1, M, N, dtype=dtypes.bfloat16, device=a.device), 0, True
    out = Tensor(inv.uop.multi(out_axis), device=a.device)
    dname = a.device[0]
  else:
    out = Tensor.invalids(1, M, N, dtype=dtypes.bfloat16, device=a.device)
    dname = a.device
  dname = dname.split(":")[0]
  scales = tuple(s for s in (x_scale, g_amax) if s is not None)
  scale_mode = (1 if x_scale is not None else 0) | (4 if g_amax is not None else 0)
  out = Tensor.custom_kernel(out, a, b, *scales, fxn=functools.partial(custom_hk_fp8_atb_gemm, dname=dname, scale_mode=scale_mode))[0]
  if reduce_out: out = out.sum(0)
  return out.squeeze(0) if out.ndim == 3 else out

# ** MXFP8 GEMM custom kernel

@functools.cache
def custom_hk_mxfp8_gemm(C:UOp, A:UOp, B:UOp, scale_A:UOp, scale_B:UOp, *extra:UOp, dname:str) -> UOp:
  # mxfp8 block-scaled gemm: A(M,K) @ B(N,K).T, e8m0 1x32 microscales packed (k_iters,dim) uint32
  M, K = A.shape[0]*A.shape[1], A.shape[2]
  N, K2 = B.shape[(1 if B.ndim == 3 else 0):]
  assert K == K2, f"{A.shape} {B.shape}"
  block_size = 256
  threads = UOp.special(64 * 8, "lidx0")
  workgroups = UOp.special((M // block_size) * (N // block_size), "gidx0")
  e_a = extra[0].base if len(extra) >= 1 else scale_A.base
  e_b = extra[1].base if len(extra) >= 2 else scale_B.base
  sink_inputs = (C.base, A.base, B.base, scale_A.base, scale_B.base, e_a, e_b, threads, workgroups)
  sink = UOp.sink(*sink_inputs,
                  arg=KernelInfo(f"hk_mxfp8_gemm_{M}_{N}_{K}", estimates=Estimates(ops=2*M*N*K, mem=(M*K+N*K)*A.dtype.itemsize+M*N*C.dtype.itemsize)))
  kittens_path = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (kittens_path/"gemm_mxfp8.cpp").read_text()
  lib = HIPCCCompiler("gfx950", [f"-I{(kittens_path/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-ffast-math",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}"]).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=lib)))

def quantize_mxfp8(x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  # 1x32 block scaling along the last axis
  *batch, K = x.shape
  scale_K = K // 32
  amax = x.detach().float().reshape(*batch, scale_K, 32).abs().max(axis=-1)
  e8 = (amax.maximum(1e-38).log2().floor() + 127).clamp(0, 254).cast(dtypes.uint8)
  qscale = (127.0 - e8.cast(dtypes.float32)).exp2().reshape(*batch, scale_K, 1).expand(*batch, scale_K, 32).reshape(*batch, K)
  x_scaled = x.float() * qscale
  x_clamped = x_scaled + (x_scaled.detach().clamp(-448.0, 448.0) - x_scaled.detach())  # STE
  packed = mx_pack(e8) if len(batch) == 1 and scale_K % 4 == 0 else None
  return x_clamped.cast(FP8_DTYPE), e8, packed

def mx_pack(e8:Tensor) -> Tensor:
  rows, scale_K = e8.shape
  return e8.reshape(rows, scale_K // 4, 4).bitcast(dtypes.uint32).reshape(rows, scale_K // 4).permute(1, 0).contiguous()

def _mx_block_scale(e8:Tensor) -> Tensor:
  # dequant scale 2^(e8-127) broadcast back to element shape
  rows, scale_K = e8.shape
  return (e8.cast(dtypes.float32) - 127.0).exp2().reshape(rows, scale_K, 1).expand(rows, scale_K, 32).reshape(rows, scale_K*32)

def _mx_block_scale_3d(e8:Tensor) -> Tensor:
  # batched (E, rows, scale_K) dequant scale 2^(e8-127) broadcast to (E, rows, scale_K*32)
  E, rows, scale_K = e8.shape
  return (e8.cast(dtypes.float32) - 127.0).exp2().reshape(E, rows, scale_K, 1).expand(E, rows, scale_K, 32).reshape(E, rows, scale_K*32)

counters = {"used":0, "todos":[]}
def todo(msg:str) -> bool: counters["todos"].append(msg); return False
def _asm_gemm_report():
  print(f'asm_gemm: {counters["used"]} used, {len(counters["todos"])} not used')
  if DEBUG >= 2 and counters["todos"]:
    from collections import Counter
    for msg, cnt in Counter(counters["todos"]).most_common(): print(f'  {cnt:3d}x {msg}')
atexit.register(_asm_gemm_report)

def can_use_asm_gemm(a:Tensor, b:Tensor) -> bool:
  if a.dtype != b.dtype: return todo(f"dtypes must match {a.dtype} != {b.dtype}")
  if a.dtype not in {dtypes.bfloat16, dtypes.float16, FP8_DTYPE}: return todo(f"only bfloat16/float16/fp8, got {a.dtype}")
  batch, M, K = (1, *a.shape) if a.ndim == 2 else a.shape
  N = b.shape[1]
  if isinstance(a.device, tuple):
    if a.ndim == 2 and a.uop.axis == 0 and b.uop.axis is None: M //= len(a.device)
    elif a.ndim == 2 and a.uop.axis == 1 and b.uop.axis == 0: K //= len(a.device)
    elif a.ndim == 2 and a.uop.axis is None and b.uop.axis == 1: N //= len(a.device)
    elif a.ndim == 3 and a.uop.axis == 0 and b.uop.axis is None: batch //= len(a.device)
    elif a.ndim == 3 and a.uop.axis == 1 and b.uop.axis is None: M //= len(a.device)
    elif a.ndim == 3 and a.uop.axis is None and b.uop.axis == 1: N //= len(a.device)
    elif a.ndim == 3 and a.uop.axis == 2 and b.uop.axis == 0: K //= len(a.device)
    else: return todo(f"sharding mismatch a.ndim={a.ndim} a.uop.axis={a.uop.axis} b.uop.axis={b.uop.axis}")
    dname = a.device[0]
  else: dname = a.device
  arch = Device[dname].renderer.target.arch
  if batch not in {1, 2}: return todo(f"GEMM batch size {batch}")
  if (M % TILE_M != 0 or N % TILE_N != 0 or K % TILE_K != 0) and arch == "gfx950":
    return todo(f"GEMM shape ({M},{N},{K}) not a multiple of ({TILE_M},{TILE_N},{TILE_K})")
  return True

# ** UOp gemm to test Tensor.custom_kernel multi and backward correctness on non cdna4
# note: this can be removed after we have GEMM on mixins

def custom_uop_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  M, K = A.shape[0]*A.shape[1], A.shape[2]
  K2, N = B.shape[(1 if B.ndim == 3 else 0):]
  assert K == K2
  m = UOp.range(M, 1, AxisType.LOOP)
  n = UOp.range(N, 2, AxisType.LOOP)
  k = UOp.range(K, 0, AxisType.REDUCE)
  mul = (A.flatten().index((m*UOp.const(dtypes.weakint, K)+k))*
         B.flatten().index((k*UOp.const(dtypes.weakint, N)+n))).cast(dtypes.float32)
  red = mul.reduce(k, arg=Ops.ADD, dtype=dtypes.float32).cast(C.dtype)
  store = C.flatten().index((m*UOp.const(dtypes.weakint, N)+n)).store(red).end(m, n)
  return store.sink(arg=KernelInfo(name=f'uop_gemm_{M}_{N}_{K}'))

# ** bf16 A @ B.T kernel in C

@functools.cache
def custom_hk_bf16_gemm(C:UOp, A:UOp, B:UOp, *args:UOp, dname:str) -> UOp:
  M, K = A.shape[0]*A.shape[1], A.shape[2]
  N, K2 = B.shape[(1 if B.ndim == 3 else 0):]
  assert K == K2, f"{A.shape} {B.shape}"
  block_m, block_n, block_k, num_warps = 256, 256, 64, 8
  assert M % block_m == 0 and N % block_n == 0 and K % block_k == 0, f"invalid bf16 tile {(block_m, block_n, block_k)} for {(M, N, K)}"
  threads = UOp.special(64 * num_warps, "lidx0")
  workgroups = UOp.special((M // block_m) * (N // block_n), "gidx0")
  b_extra = args[0].base if len(args) >= 1 else B.base
  sink = UOp.sink(C.base, A.base, B.base, b_extra, threads, workgroups,
                  arg=KernelInfo(f"hk_bf16_gemm_{M}_{N}_{K}", estimates=Estimates(ops=2*M*N*K, mem=(M*K+N*K+M*N)*A.dtype.itemsize)))
  kittens_path = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (kittens_path/"gemm_bf16.cpp").read_text()
  lib = HIPCCCompiler("gfx950", [f"-I{(kittens_path/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-ffast-math",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}"]).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                                UOp(Ops.BINARY, arg=lib)))

@functools.cache
def custom_hk_bf16_atb_gemm(C:UOp, A:UOp, B:UOp, dname:str) -> UOp:
  K, M = A.shape[0]*A.shape[1], A.shape[2]
  K2, N = B.shape[0]*B.shape[1], B.shape[2]
  assert K == K2, f"{A.shape} {B.shape}"
  block_m, block_n, block_k, num_warps = 256, 256, 64, 8
  assert M % block_m == 0 and N % block_n == 0 and K % block_k == 0, f"invalid bf16 atb tile {(block_m, block_n, block_k)} for {(M, N, K)}"
  threads = UOp.special(64 * num_warps, "lidx0")
  workgroups = UOp.special((M // block_m) * (N // block_n), "gidx0")
  sink = UOp.sink(C.base, A.base, B.base, threads, workgroups,
                  arg=KernelInfo(f"hk_bf16_atb_gemm_{M}_{N}_{K}", estimates=Estimates(ops=2*M*N*K, mem=(M*K+N*K+M*N)*A.dtype.itemsize)))
  kittens_path = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (kittens_path/"gemm_bf16_atb.cpp").read_text()
  lib = HIPCCCompiler("gfx950", [f"-I{(kittens_path/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-ffast-math",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}"]).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                                UOp(Ops.BINARY, arg=lib)))

def hk_bf16_atb_gemm(a:Tensor, b:Tensor) -> Tensor:
  assert a.dtype == b.dtype == dtypes.bfloat16, f"expected bf16, got {a.dtype} {b.dtype}"
  assert a.ndim == b.ndim == 3 and a.shape[:2] == b.shape[:2], f"{a.shape} {b.shape}"
  batch, rows, M = a.shape
  N = b.shape[2]
  assert M % TILE_M == 0 and N % TILE_N == 0 and (batch * rows) % TILE_K == 0, \
    f"atb shape {a.shape} {b.shape} must produce (M,N,K) multiples of ({TILE_M},{TILE_N},{TILE_K})"
  is_multi = isinstance(a.device, tuple)
  reduce_out = False
  if is_multi:
    ndev = len(a.device)
    if a.uop.axis in (0, 1) or b.uop.axis in (0, 1): inv, out_axis, reduce_out = Tensor.invalids(1, M, N, dtype=a.dtype, device=a.device), 0, True
    elif b.uop.axis == 2: inv, out_axis = Tensor.invalids(1, M, N // ndev, dtype=a.dtype, device=a.device), 2
    elif a.uop.axis == 2: inv, out_axis = Tensor.invalids(1, M // ndev, N, dtype=a.dtype, device=a.device), 1
    else: inv, out_axis, reduce_out = Tensor.invalids(1, M, N, dtype=a.dtype, device=a.device), 0, True
    out = Tensor(inv.uop.multi(out_axis), device=a.device)
    dname = a.device[0]
  else:
    out = Tensor.invalids(1, M, N, dtype=a.dtype, device=a.device)
    dname = a.device
  dname = dname.split(":")[0]
  out = Tensor.custom_kernel(out, a, b, fxn=functools.partial(custom_hk_bf16_atb_gemm, dname=dname))[0]
  if reduce_out: out = out.sum(0)
  return out.squeeze(0) if out.ndim == 3 else out


# ** backward gemm, might use the asm gemm

def custom_gemm_bw(gradient:UOp, kernel:UOp, n_scales:int=2, has_grad_amax:bool=False, has_w_post:bool=False):
  inputs = kernel.src[1:]
  if inputs[1].dtype == FP8_DTYPE:
    out, a, b = inputs[:3]
    i = 3
    s_x = inputs[i]; i += 1
    has_w = n_scales >= 2
    s_w = inputs[i] if has_w else None; i += has_w
    s_g_amax = inputs[i] if n_scales == 3 else None; i += (n_scales == 3)
    grad_amax_state = inputs[i] if has_grad_amax else None; i += has_grad_amax
    next_grad_amax_state = inputs[i] if has_grad_amax else None; i += has_grad_amax
    w_post = inputs[i] if has_w_post else None
    a_t, b_t, g_t = Tensor(a, device=a.device), Tensor(b, device=a.device), Tensor(gradient, device=a.device)
    s_x_t = Tensor(s_x, device=a.device)
    s_w_t = Tensor(s_w, device=a.device) if has_w else None
    s_g_amax_t = Tensor(s_g_amax, device=a.device) if s_g_amax is not None else None
    w_post_t = Tensor(w_post, device=a.device) if has_w_post else None
    g_t = g_t[:a.shape[0]]
    from extra.llama_kernels.cast_amax import _grad_fp8_mailbox
    from extra.llama_kernels.quantize_fp8_delayed import quantize_fp8_delayed
    gbase = gradient.base if hasattr(gradient, "base") else gradient
    mailbox_entry = _grad_fp8_mailbox.pop(gbase, None) or _grad_fp8_mailbox.pop(gradient, None)
    if mailbox_entry is not None:
      g_fp8_u, grad_amax_u = mailbox_entry
      g_fp8 = Tensor(g_fp8_u, device=a.device)[:a.shape[0]]
      g_amax = Tensor(grad_amax_u, device=a.device)
    else:
      assert grad_amax_state is not None, "fp8 matmul bwd needs either a mailbox entry or a grad_amax_state"
      if getenv("CURRENT_GRAD_SCALE", 0):
        g_fp8, _, g_amax = quantize_fp8(g_t, amax_state=None)
      elif getenv("FUSED_GRAD_QUANTIZE", 0):
        grad_amax_t = Tensor(grad_amax_state, device=a.device)
        g_amax = grad_amax_t
        g_fp8, _ = quantize_fp8_delayed(g_t, g_amax, Tensor(next_grad_amax_state, device=a.device))
      else:
        grad_amax_t = Tensor(grad_amax_state, device=a.device)
        g_amax = grad_amax_t
        g_fp8, _, new_grad_amax = quantize_fp8(g_t, amax_state=g_amax)
        store_effect = next_grad_amax_state.store(new_grad_amax.uop)
        g_fp8 = Tensor(g_fp8.contiguous().uop.after(store_effect), device=a.device)
    # dgrad: applies grad/activation amax scales in the GEMM epilogue; w_scale is already inverse.
    assert s_g_amax_t is None, "fp8 GEMM bwd through g_amax scaling is unsupported"
    grad_a = asm_gemm(g_fp8, b_t, x_scale=s_x_t, w_scale=s_w_t, g_amax=g_amax) if has_w else asm_gemm(g_fp8, b_t, x_scale=s_x_t, g_amax=g_amax)
    # wgrad: no w_scale
    grad_b = hk_fp8_atb_gemm(g_fp8, a_t, x_scale=s_x_t, g_amax=g_amax)
    # wgrad: rescale if not scalar
    if w_post_t is not None:
      grad_b = grad_b / w_post_t.reshape(*w_post_t.shape, *([1]*(grad_b.ndim - w_post_t.ndim)))
    # one None per input: (out, a, b, x_scale[, w_scale][, grad_amax][, w_post_scale])
    ret = (None, grad_a.uop, grad_b.uop) + tuple(None for _ in inputs[3:])
    return ret
  else:
    hk_bf16 = len(inputs) == 4 and inputs[1].dtype == dtypes.bfloat16
    if hk_bf16:
      out, a, b_t, b = inputs
      assert all_same([gradient.device, a.device, b_t.device, b.device, out.device])
    else:
      assert len(inputs) == 3, f"regular gemm must have exactly 3 sources, got: {len(inputs)}"
      out, a, b = inputs
      assert all_same([gradient.device, a.device, b.device, out.device])
    a_t, b_t, g_t = Tensor(a, device=a.device), Tensor(b, device=a.device), Tensor(gradient, device=a.device)
    g_t = g_t[:a.shape[0]]
    if hk_bf16 and g_t.dtype != b_t.dtype: g_t = g_t.cast(b_t.dtype)
    if can_use_asm_gemm(g_t, b_t.T): grad_a = asm_gemm(g_t, b_t.T).uop
    else: grad_a = (g_t @ b_t.T).uop
    if hk_bf16:
      grad_b = hk_bf16_atb_gemm(a_t, g_t).uop
    else:
      a_t_flat, g_t_flat = a_t.permute(2, 0, 1).reshape(a_t.shape[2], -1), g_t.reshape(-1, g_t.shape[-1])
      if can_use_asm_gemm(a_t_flat, g_t_flat): grad_b = asm_gemm(a_t_flat, g_t_flat).uop
      else: grad_b = (a_t_flat @ g_t_flat).uop
    # hk_bf16 uses b.T, writes gradients only for a and b
    return (None, grad_a, None, grad_b) if hk_bf16 else (None, grad_a, grad_b)

# ** mxfp8 gemm backward

def custom_mx_gemm_bw(gradient:UOp, kernel:UOp, has_w_post:bool, w_stored:bool=False):
  inputs = kernel.src[1:]  # (out, a_q, b_q, a_si, b_si, a_e8, b_e8, [w_post])
  aq, bq = Tensor(inputs[1], device=inputs[1].device), Tensor(inputs[2], device=inputs[2].device)
  ae8, be8 = Tensor(inputs[5], device=inputs[5].device), Tensor(inputs[6], device=inputs[6].device)
  wp = Tensor(inputs[7], device=inputs[7].device) if has_w_post else None

  a_phys = (aq.reshape(-1, aq.shape[-1]).cast(dtypes.bfloat16) * _mx_block_scale(ae8)).cast(dtypes.bfloat16)
  b_phys = (bq.cast(dtypes.bfloat16) * _mx_block_scale(be8)).cast(dtypes.bfloat16)

  g = Tensor(gradient, device=aq.device)[:aq.shape[0]].reshape(aq.shape[0]*aq.shape[1], bq.shape[0]).cast(dtypes.bfloat16)
  grad_a = asm_gemm(g, b_phys, mx=True)
  grad_b = asm_gemm(g.T, a_phys, mx=True, a_pretranspose=g)

  grad_a = (grad_a * _mx_block_scale(ae8)).reshape(aq.shape)
  if not w_stored: grad_b = grad_b * _mx_block_scale(be8)
  if wp is not None: grad_b = grad_b / wp.reshape(-1, 1)
  return (None, grad_a.uop, grad_b.uop) + tuple(None for _ in inputs[3:])

# ** main gemm function

def asm_gemm(a:Tensor, b:Tensor, x_scale:Tensor|None=None, w_scale:Tensor|None=None, grad_amax_state:Tensor|None=None,
             next_grad_amax_state:Tensor|None=None,
             w_post_scale:Tensor|None=None, mx:bool=False, mx_scales:tuple|None=None, mx_w_stored:bool=False, g_amax:Tensor|None=None,
             a_pretranspose:Tensor|None=None) -> Tensor:
  assert can_use_asm_gemm(a, b), f"{counters['todos'][-1]}"
  counters["used"] += 1
  unfold_batch = a.ndim == 3 and isinstance(a.device, tuple) and a.uop.axis == 2 and b.uop.axis == 0
  if unfold_batch:
    orig_batch = a.shape[0]
    a = a.reshape(a.shape[0]*a.shape[1], a.shape[2])
  squeeze = a.ndim == 2
  if squeeze: a = a.unsqueeze(0)
  out_dtype = dtypes.bfloat16 if a.dtype == FP8_DTYPE else a.dtype

  batch, M, K = a.shape
  N = b.shape[1]
  is_multi = isinstance(a.device, tuple)
  if (k_sharded:=is_multi and a.uop.axis == 2): K //= len(a.device)
  if (m_sharded:=is_multi and a.uop.axis == 1): M //= len(a.device)
  n_sharded = is_multi and b.uop.axis == 1

  if is_multi:
    if n_sharded:
      out = Tensor(Tensor.invalids(batch, M, N//len(a.device), dtype=out_dtype, device=a.device).uop.multi(2), device=a.device)
    elif m_sharded:
      out = Tensor(Tensor.invalids(batch, M, N, dtype=out_dtype, device=a.device).uop.multi(1), device=a.device)
    else:
      out = Tensor(Tensor.invalids(batch//len(a.device) if a.uop.axis==0 else batch, M, N, dtype=out_dtype, device=a.device).uop.multi(0),
                   device=a.device)
  else:
    out = Tensor.invalids(batch, M, N, dtype=out_dtype, device=a.device)

  renderer = Device[dname:=(a.device[0] if is_multi else a.device)].renderer
  dname, arch = dname.split(":")[0], renderer.target.arch
  if arch.startswith("gfx950") and getenv("USE_ASM", 1):
    if mx:
      # mxfp8 1x32 block scaling
      if mx_scales is not None:
        a_si, a_e8, b_si, b_e8 = mx_scales
        a_q, b_q = a.reshape(-1, a.shape[-1]), b.T
      elif (a_pretranspose is not None and getenv("FUSED_GRAD_QUANTIZE", 0) and a_pretranspose.dtype == dtypes.bfloat16
            and a_pretranspose.shape[0] % 32 == 0 and a_pretranspose.shape[1] % 256 == 0):
        from extra.llama_kernels.transpose_quantize_mxfp8 import transpose_quantize_mxfp8
        a_q, a_e8, a_si = transpose_quantize_mxfp8(a_pretranspose)
        b_q, b_e8, b_si = quantize_mxfp8(b.T)
      else:
        a_q, a_e8, a_si = quantize_mxfp8(a.reshape(-1, a.shape[-1]))
        b_q, b_e8, b_si = quantize_mxfp8(b.T)
      has_w_post = w_post_scale is not None
      fxn = functools.partial(custom_hk_mxfp8_gemm, dname=dname)
      grad_fxn = functools.partial(custom_mx_gemm_bw, has_w_post=has_w_post, w_stored=mx_w_stored)
      extra = [w_post_scale] if w_post_scale is not None else []
      out = Tensor.custom_kernel(out, a_q.reshape(a.shape), b_q, a_si, b_si, a_e8, b_e8, *extra, fxn=fxn, grad_fxn=grad_fxn)[0]
    # fp8 gemm computes a@b.T, kernel multiplies output by x_scale * w_scale before bf16 store
    elif a.dtype == FP8_DTYPE:
      scales = tuple(s for s in (x_scale, w_scale, g_amax) if s is not None)
      scale_mode = (1 if x_scale is not None else 0) | (2 if w_scale is not None else 0) | (4 if g_amax is not None else 0)
      assert (grad_amax_state is None) == (next_grad_amax_state is None)
      extra = ([grad_amax_state, next_grad_amax_state] if grad_amax_state is not None else []) + ([w_post_scale] if w_post_scale is not None else [])
      fxn = functools.partial(custom_hk_fp8_gemm, dname=dname, scale_mode=scale_mode)
      bw = functools.partial(custom_gemm_bw, n_scales=len(scales), has_grad_amax=grad_amax_state is not None, has_w_post=w_post_scale is not None)
      out = Tensor.custom_kernel(out, a, b.T, *scales, *extra, fxn=fxn, grad_fxn=bw)[0]
    elif a.dtype == dtypes.bfloat16:
      out = Tensor.custom_kernel(out, a, b.T, b, fxn=functools.partial(custom_hk_bf16_gemm, dname=dname), grad_fxn=custom_gemm_bw)[0]
  else:
    out = Tensor.custom_kernel(out, a, b, fxn=custom_uop_gemm, grad_fxn=custom_gemm_bw)[0]
  if k_sharded: out = out.sum(0)
  out = out.squeeze(0) if squeeze else out
  if unfold_batch: out = out.reshape(orig_batch, -1, out.shape[-1])
  if w_post_scale is not None: out = (out * w_post_scale.reshape(*([1]*(out.ndim-1)), -1)).cast(out.dtype)
  return out
