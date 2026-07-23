import math, pathlib, functools, struct

from tinygrad import Device, Tensor
from tinygrad.dtype import DTypeLike, dtypes
from tinygrad.helpers import DEBUG
from tinygrad.renderer import Estimates
from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.uop.ops import UOp, Ops, KernelInfo

def _sharded_empty(shape:Tensor, ref:Tensor, axis:int|None, dtype:DTypeLike|None=None) -> Tensor:
  dtype = dtype or ref.dtype
  if not isinstance(ref.device, tuple): return Tensor.invalids(*shape, dtype=dtype, device=ref.device)
  shard_axis = ref.uop.axis if axis is None else axis
  shape = tuple(s // len(ref.device) if i == shard_axis else s for i, s in enumerate(shape))
  axis = ref.uop.axis if axis is None else axis
  return Tensor(Tensor.invalids(*shape, dtype=dtype, device=ref.device).uop.multi(axis), dtype=dtype, device=ref.device)

@functools.cache
def custom_fused_qkv_rope_forward(q:UOp, k:UOp, v:UOp, xqkv:UOp, freqs_cis:UOp,
                                  device:str, arch:str, B:int, N:int, H:int, H_KV:int, D:int):
  code = (pathlib.Path(__file__).parent / "fused_qkv_rope.cpp").read_text()
  threads = 256
  thread_idx = UOp.special(threads, "lidx0")
  block_idx_x, block_idx_y = UOp.special(B, "gidx0"), UOp.special(N, "gidx1")
  sink = UOp.sink(q.base, k.base, v.base, xqkv.base, freqs_cis.base, thread_idx, block_idx_x, block_idx_y,
                  arg=KernelInfo(name="fused_qkv_rope_forward"))
  compile_args = ["-std=c++20", "-ffast-math", f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}",
                  f"-DATTN_H_KV={H_KV}", f"-DATTN_D={D}", f"-DTHREADS_PER_BLOCK={threads}"]
  lib = HIPCCCompiler(arch, compile_args).compile_cached(code)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def custom_fused_qkv_rope_backward(dxqkv:UOp, dq:UOp, dk:UOp, dv:UOp, freqs_cis:UOp,
                                   device:str, arch:str, B:int, N:int, H:int, H_KV:int, D:int):
  assert (B, N, H, H_KV, D) == (2, 8192, 32, 8, 128)
  code = (pathlib.Path(__file__).parent / "fused_qkv_rope_bwd.cpp").read_text()
  threads = 256
  thread_idx = UOp.special(threads, "lidx0")
  gsz = (B, N // 64, H + 2 * H_KV)
  block_idx_x, block_idx_y, block_idx_z = (UOp.special(x, f"gidx{i}") for i, x in enumerate(gsz))
  sink = UOp.sink(dxqkv.base, dq.base, dk.base, dv.base, freqs_cis.base, thread_idx, block_idx_x, block_idx_y, block_idx_z,
                  arg=KernelInfo(name="fused_qkv_rope_backward"))
  compile_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math", f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}",
                  f"-DATTN_H_KV={H_KV}", f"-DATTN_D={D}", f"-DTHREADS_PER_BLOCK={threads}"]
  lib = HIPCCCompiler(arch, compile_args).compile_cached(code)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))

def _fa_native_grads(dq:UOp, dk:UOp, dv:UOp) -> tuple[UOp, UOp, UOp]|None:
  def unwrap_partial(x:UOp) -> UOp|None:
    expected = (Ops.CAST, Ops.REDUCE, Ops.PERMUTE, Ops.CAST, Ops.RESHAPE, Ops.AFTER)
    for op in expected:
      if x.op is not op: return None
      if op is not Ops.AFTER: x = x.src[0]
    return x
  dq_native, dk_partial, dv_partial = dq.base, unwrap_partial(dk), unwrap_partial(dv)
  if dq_native.op is not Ops.AFTER or dk_partial is None or dv_partial is None: return None
  B, N, H, D, H_KV = dq.shape[0], dq.shape[1], dq.shape[2], dq.shape[3], dk.shape[2]
  heads_per_wg = 2 if D == 128 and (H // H_KV) % 2 == 0 else 1
  partials = (H // H_KV) // heads_per_wg
  if dq_native.shape != (B, H, N, D) or dk_partial.shape != (B * partials, N, H_KV, D) or dv_partial.shape != dk_partial.shape: return None
  return dq_native, dk_partial, dv_partial

def _fused_qkv_rope_grad(dq_u:UOp, dk_u:UOp, dv_u:UOp, call:UOp) -> tuple[None, None, None, UOp, None]:
  dq, dk, dv = Tensor(dq_u, device=dq_u.device), Tensor(dk_u, device=dk_u.device), Tensor(dv_u, device=dv_u.device)
  xqkv_u, freqs_u = call.src[4], call.src[5]
  xqkv, freqs_cis = Tensor(xqkv_u, device=xqkv_u.device), Tensor(freqs_u, device=freqs_u.device)
  B, N, _ = xqkv.shape
  H, H_KV, D = dq.shape[2], dk.shape[2], dq.shape[3]
  num_devices = len(xqkv.device) if isinstance(xqkv.device, tuple) else 1
  is_dp, is_mp = xqkv.uop.axis == 0, xqkv.uop.axis == 2
  B_local = B // num_devices if is_dp else B
  H_local = H // num_devices if is_mp else H
  H_KV_local = H_KV // num_devices if is_mp else H_KV
  single_device = xqkv.device[0] if isinstance(xqkv.device, tuple) else xqkv.device
  arch = Device[single_device].renderer.target.arch
  fa_native = _fa_native_grads(dq_u, dk_u, dv_u)
  assert fa_native is not None, "fused QKV RoPE backward requires native Flash Attention gradients"
  dq, dk, dv = (Tensor(x, device=x.device) for x in fa_native)
  dxqkv = _sharded_empty_like(xqkv, axis=xqkv.uop.axis if isinstance(xqkv.device, tuple) else None)
  fxn = functools.partial(custom_fused_qkv_rope_backward, device=single_device, arch=arch,
                          B=B_local, N=N, H=H_local, H_KV=H_KV_local, D=D)
  dxqkv = Tensor.custom_kernel(dxqkv, dq, dk, dv, freqs_cis, fxn=fxn)[0]
  return None, None, None, dxqkv.uop, None

def fused_qkv_rope(xqkv:Tensor, freqs_cis:Tensor, n_heads:int, n_kv_heads:int, head_dim:int) -> tuple[Tensor, Tensor, Tensor]:
  B, N, packed_dim = xqkv.shape
  assert packed_dim == n_kv_heads * (n_heads // n_kv_heads + 2) * head_dim
  assert freqs_cis.dtype == dtypes.bfloat16, f"fused QKV RoPE requires bfloat16 frequencies, got {freqs_cis.dtype}"
  assert freqs_cis.shape == (1, freqs_cis.shape[1], 1, head_dim // 2, 2) and freqs_cis.shape[1] >= N, \
    f"invalid RoPE frequency shape {freqs_cis.shape} for sequence length {N} and head dimension {head_dim}"
  num_devices = len(xqkv.device) if isinstance(xqkv.device, tuple) else 1
  is_dp, is_mp = xqkv.uop.axis == 0, xqkv.uop.axis == 2
  B_local = B // num_devices if is_dp else B
  H_local = n_heads // num_devices if is_mp else n_heads
  H_KV_local = n_kv_heads // num_devices if is_mp else n_kv_heads
  assert H_local % H_KV_local == 0 and head_dim % 2 == 0 and head_dim <= 512
  single_device = xqkv.device[0] if isinstance(xqkv.device, tuple) else xqkv.device
  arch = Device[single_device].renderer.target.arch
  axis = 0 if is_dp else 2 if is_mp else None
  q = _sharded_empty((B, N, n_heads, head_dim), xqkv, axis=axis, dtype=dtypes.bfloat16)
  k = _sharded_empty((B, N, n_kv_heads, head_dim), xqkv, axis=axis, dtype=dtypes.bfloat16)
  v = _sharded_empty((B, N, n_kv_heads, head_dim), xqkv, axis=axis, dtype=dtypes.bfloat16)
  fxn = functools.partial(custom_fused_qkv_rope_forward, device=single_device, arch=arch,
                          B=B_local, N=N, H=H_local, H_KV=H_KV_local, D=head_dim)
  q, k, v, *_ = Tensor.custom_kernel(q, k, v, xqkv, freqs_cis, fxn=fxn, grad_fxn=_fused_qkv_rope_grad)
  return q, k, v

def _sharded_empty_like(ref:Tensor, axis:int|None=None) -> Tensor:
  return _sharded_empty(ref.shape, ref, axis)

@functools.cache
def _fa_grad_fxn(B, H, N, D, H_local, H_KV_local, H_KV, B_local, shard_axis, shard_axis_t, single_device, arch, has_sink):
  def grad(dou:UOp, ker:UOp) -> tuple:
    do = Tensor(dou, device=dou.device)
    attn = Tensor(ker.src[1].after(ker), device=ker.src[1].device)
    l_vec = Tensor(ker.src[2].after(ker), device=ker.src[2].device)
    xq = Tensor(ker.src[3], device=ker.src[3].device)
    xk = Tensor(ker.src[4], device=ker.src[4].device)
    xv = Tensor(ker.src[5], device=ker.src[5].device)

    dq = _sharded_empty((B, H, N, D), xq, axis=shard_axis_t)
    GROUP_SIZE = H_local // H_KV_local
    HEADS_PER_WG = 2 if D == 128 and GROUP_SIZE % 2 == 0 else 1
    dk_partial = _sharded_empty((B * GROUP_SIZE // HEADS_PER_WG, N, H_KV, D), xk, axis=shard_axis)
    dv_partial = _sharded_empty((B * GROUP_SIZE // HEADS_PER_WG, N, H_KV, D), xv, axis=shard_axis)

    # delta_vec = (do * attn).sum(-1, dtype=dtypes.float32).transpose(1, 2).unsqueeze(-2).detach()
    delta_vec = _sharded_empty((B, H, 1, N), xq, dtype=dtypes.float32, axis=shard_axis_t)
    delta_vec, dq = Tensor.custom_kernel(delta_vec, dq, attn, do, fxn=functools.partial(custom_fa_backward_pre, device=single_device, arch=arch, B=B_local, N=N, H=H_local, H_KV=H_KV_local, D=D))[:2]

    dq, dk_partial, dv_partial = Tensor.custom_kernel(dq, dk_partial, dv_partial, do, xq, xk, xv, l_vec, delta_vec, fxn=functools.partial(custom_fa_backward, device=single_device, arch=arch, B=B_local, N=N, H=H_local, H_KV=H_KV_local, D=D))[:3]

    if D == 64:
      dq = dq.reshape(B, H, N//16, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2).permute(0, 1, 2, 8, 9, 10, 11, 3, 4, 6, 7, 5, 12).reshape(B, H, N, D).transpose(1, 2)
    else:
      dq = dq.reshape(B, H, N//16, 4, 2, 2, D//32, 4, 4, 2).permute(0, 1, 2, 7, 8, 3, 4, 6, 5, 9).reshape(B, H, N, D).transpose(1, 2)

    # reduce partial dK/dV across GROUP_SIZE query heads
    dk = dk_partial.reshape(B, GROUP_SIZE // HEADS_PER_WG, N, H_KV, D).sum(1)
    dv = dv_partial.reshape(B, GROUP_SIZE // HEADS_PER_WG, N, H_KV, D).sum(1)

    if not has_sink: return None, None, dq.uop, dk.uop, dv.uop
    sinks = Tensor(ker.src[6], device=ker.src[6].device)
    p_sink = (sinks.reshape(1, H, 1, 1) - l_vec).exp()
    dsink = -(delta_vec.float() * p_sink).sum(axis=(0, 2, 3))

    return None, None, dq.uop, dk.uop, dv.uop, dsink.uop
  return grad

# TODO: remove write_flat once scheduler can remove reshapes between custom_kernel. TestCustomKernel.test_simple_reshape
def flash_attention(xq, xk, xv, attn_mask:Tensor|None=None, is_causal:bool=False, write_flat:bool=False, sinks:Tensor|None=None):
  assert attn_mask is None, "attn_mask not supported"
  assert is_causal, "only causal attention supported"

  B, N, H, D = xq.shape
  H_KV = xk.shape[2]
  assert D in (64, 128), "only D=64 or D=128 supported"
  has_sink = sinks is not None
  if has_sink: sinks = sinks.float()

  num_devices = len(xq.device) if isinstance(xq.device, tuple) else 1
  is_dp = xq.uop.axis == 0
  is_mp = xq.uop.axis == 2
  B_local = B // num_devices if is_dp else B
  H_local = H // num_devices if is_mp else H
  H_KV_local = H_KV // num_devices if is_mp else H_KV
  shard_axis = 0 if is_dp else 2 if is_mp else None
  shard_axis_t = 0 if is_dp else 1 if is_mp else None
  if DEBUG >= 2: print(f"Flash Attention {B=} {B_local=} {N=} {H=} {H_local=} {H_KV=} {H_KV_local=} {D=} on {num_devices} devices, {'DP' if is_dp else 'MP' if is_mp else 'no sharding'}")

  single_device = xq.device[0] if isinstance(xq.device, tuple) else xq.device
  arch = Device[single_device].renderer.target.arch

  attn = _sharded_empty_like(xq, axis=shard_axis)
  attn = _sharded_empty((B, N, H * D), xq, axis=shard_axis) if write_flat else _sharded_empty_like(xq, axis=shard_axis)
  l_vec = _sharded_empty((B, H, 1, N), xq, dtype=dtypes.float32, axis=shard_axis_t)

  grad = _fa_grad_fxn(B, H, N, D, H_local, H_KV_local, H_KV, B_local, shard_axis, shard_axis_t, single_device, arch, has_sink)

  fwd_inputs = (attn, l_vec, xq, xk, xv) + ((sinks,) if has_sink else ())
  attn, l_vec = Tensor.custom_kernel(*fwd_inputs, fxn=functools.partial(custom_fa_forward, device=single_device, arch=arch, B=B_local, N=N, H=H_local, H_KV=H_KV_local, D=D, has_sink=has_sink), grad_fxn=grad)[:2]

  return attn, attn, l_vec

@functools.cache
def custom_fa_forward(o:UOp, l_vec:UOp, q:UOp, k:UOp, v:UOp, sinks:UOp|None=None, *, device:str, arch:str, B:int, N:int, H:int, H_KV:int, D:int, has_sink:bool=True):
  code = (pathlib.Path(__file__).parent / "fa_fwd_causal.cpp").read_text()
  compile_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
                  f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}", f"-DATTN_H_KV={H_KV}", f"-DATTN_D={D}", f"-DATTN_SINK={int(has_sink)}"]

  Q_BLOCK_SIZE = 32
  NUM_WARPS = 8
  NUM_THREADS = 64 * NUM_WARPS
  gsz = (H, (math.ceil((N // Q_BLOCK_SIZE) / NUM_WARPS)), B)
  lsz = (NUM_THREADS, 1, 1)
  threadIdx_x = UOp.special(lsz[0], "lidx0")
  blockIdx_x, blockIdx_y, blockIdx_z = UOp.special(gsz[0], "gidx0"), UOp.special(gsz[1], "gidx1"), UOp.special(gsz[2], "gidx2")

  el = q.dtype.itemsize
  mem = (2*B*N*H*D + 2*B*N*H_KV*D) * el + B*H*N * l_vec.dtype.itemsize
  estimates = Estimates(ops=2*B*H*N*N*D, lds=mem, mem=mem)
  buf_inputs = (o.base, l_vec.base, q.base, k.base, v.base) + ((sinks.base,) if has_sink else ())
  sink = UOp.sink(*buf_inputs,
                  threadIdx_x, blockIdx_x, blockIdx_y, blockIdx_z,
                  arg=KernelInfo(name="custom_fa_forward", estimates=estimates))

  lib = HIPCCCompiler(arch, compile_args).compile_cached(code)
  lib = bytearray(lib)
  rodata_off = next(sh.header.sh_offset for sh in elf_loader(bytes(lib))[1] if sh.name == ".rodata")
  struct.pack_into('<I', lib, rodata_off, 160000)
  lib = bytes(lib)

  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def custom_fa_backward_pre(delta_vec:UOp, dq:UOp, o:UOp, do:UOp, device:str, arch:str, B:int, N:int, H:int, H_KV:int, D:int):
  code = (pathlib.Path(__file__).parent / "fa_bwd_pre.cpp").read_text()
  compile_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
                  f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}", f"-DATTN_D={D}"]

  DOT_SLICE_QO = 16
  NUM_WARPS = 4
  NUM_THREADS = 64 * NUM_WARPS
  gsz = (B, H, N // (DOT_SLICE_QO * NUM_WARPS))
  lsz = (NUM_THREADS, 1, 1)
  threadIdx_x = UOp.special(lsz[0], "lidx0")
  blockIdx_x, blockIdx_y, blockIdx_z = UOp.special(gsz[0], "gidx0"), UOp.special(gsz[1], "gidx1"), UOp.special(gsz[2], "gidx2")

  el = o.dtype.itemsize
  mem = 3*B*H*N*D * el + B*H*N * delta_vec.dtype.itemsize
  estimates = Estimates(ops=2*B*H*N*D, lds=mem, mem=mem)
  sink = UOp.sink(delta_vec.base, dq.base, o.base, do.base,
                  threadIdx_x, blockIdx_x, blockIdx_y, blockIdx_z,
                  arg=KernelInfo(name="custom_fa_backward_pre", estimates=estimates))

  lib = HIPCCCompiler(arch, compile_args).compile_cached(code)
  lib = bytearray(lib)
  rodata_off = next(sh.header.sh_offset for sh in elf_loader(bytes(lib))[1] if sh.name == ".rodata")
  struct.pack_into('<I', lib, rodata_off, 160000)
  lib = bytes(lib)

  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def custom_fa_backward(dq:UOp, dk:UOp, dv:UOp, do:UOp, q:UOp, k:UOp, v:UOp, l_vec:UOp, delta_vec:UOp, device:str, arch:str, B:int, N:int, H:int, H_KV:int, D:int):
  code = (pathlib.Path(__file__).parent / "fa_bwd_causal.cpp").read_text()
  compile_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
                  f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}", f"-DATTN_H_KV={H_KV}", f"-DATTN_D={D}"]

  BLOCK_SIZE_KV = 256
  GROUP_SIZE = H // H_KV
  HEADS_PER_WG = 2 if D == 128 and GROUP_SIZE % 2 == 0 else 1
  NUM_WARPS = 4
  NUM_THREADS = 64 * NUM_WARPS
  gsz = (H // HEADS_PER_WG, N // BLOCK_SIZE_KV, B)
  lsz = (NUM_THREADS, 1, 1)
  threadIdx_x = UOp.special(lsz[0], "lidx0")
  blockIdx_x, blockIdx_y, blockIdx_z = UOp.special(gsz[0], "gidx0"), UOp.special(gsz[1], "gidx1"), UOp.special(gsz[2], "gidx2")

  el = q.dtype.itemsize
  mem = (3*B*H*N*D + 4*B*H_KV*N*D) * el + 2*B*H*N * l_vec.dtype.itemsize
  estimates = Estimates(ops=5*B*H*N*N*D, lds=mem, mem=mem)
  sink = UOp.sink(dq.base, dk.base, dv.base, do.base, q.base, k.base, v.base, l_vec.base, delta_vec.base,
                  threadIdx_x, blockIdx_x, blockIdx_y, blockIdx_z,
                  arg=KernelInfo(name="custom_fa_backward", estimates=estimates))

  lib = HIPCCCompiler(arch, compile_args).compile_cached(code)
  lib = bytearray(lib)
  rodata_off = next(sh.header.sh_offset for sh in elf_loader(bytes(lib))[1] if sh.name == ".rodata")
  struct.pack_into('<I', lib, rodata_off, 160000)
  lib = bytes(lib)

  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def custom_fa_backward_post(dq_out:UOp, dq_in:UOp, device:str, arch:str, B:int, N:int, H:int, H_KV:int, D:int):
  code = (pathlib.Path(__file__).parent / "fa_bwd_post.cpp").read_text()
  compile_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
                  f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}", f"-DATTN_D={D}"]

  DOT_SLICE_QO = 16
  NUM_WARPS = 4
  NUM_THREADS = 64 * NUM_WARPS
  gsz = (B, H, N // (DOT_SLICE_QO * NUM_WARPS))
  lsz = (NUM_THREADS, 1, 1)
  threadIdx_x = UOp.special(lsz[0], "lidx0")
  blockIdx_x, blockIdx_y, blockIdx_z = UOp.special(gsz[0], "gidx0"), UOp.special(gsz[1], "gidx1"), UOp.special(gsz[2], "gidx2")

  el = dq_out.dtype.itemsize
  mem = 2*B*H*N*D * el
  estimates = Estimates(lds=mem, mem=mem)
  sink = UOp.sink(dq_out.base, dq_in.base,
                  threadIdx_x, blockIdx_x, blockIdx_y, blockIdx_z,
                  arg=KernelInfo(name="custom_fa_backward_post", estimates=estimates))

  lib = HIPCCCompiler(arch, compile_args).compile_cached(code)
  lib = bytearray(lib)
  rodata_off = next(sh.header.sh_offset for sh in elf_loader(bytes(lib))[1] if sh.name == ".rodata")
  struct.pack_into('<I', lib, rodata_off, 160000)
  lib = bytes(lib)

  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))
