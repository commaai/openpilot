from __future__ import annotations
import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops
from tinygrad.runtime.support.compiler_amd import HIPCCCompiler

FP8_MAX = 448.0
NUM_WG, THREADS_PER_WG = 1024, 256

# per-device abs max without allreduce
@functools.cache
def _local_abs_max_fxn(x_p, device):
  x = Tensor(x_p, device=device)
  inner = Tensor(x.uop.src[0]) if x.uop.op is Ops.MULTI else x
  return (inner.abs().max(),)

def local_abs_max(x:Tensor) -> Tensor:
  param = x.as_param(0)
  fxn = _local_abs_max_fxn(param.uop, x.device)
  return Tensor(fxn[0].uop.call(x.uop).gettuple(0))

def scalar_amax(amax_buf:Tensor) -> Tensor:
  if isinstance(amax_buf.device, tuple):
    return local_abs_max(amax_buf).detach()
  return amax_buf.max().detach()

def shard_shape(shape:tuple, axis:int, ndev:int) -> list:
  s = list(shape)
  s[axis] //= ndev
  return s

def dname_of(device) -> str:
  if isinstance(device, tuple): return device[0].split(":")[0]
  return device.split(":")[0] if isinstance(device, str) else device

def alloc_like(shape, dtype, device, axis=None) -> Tensor:
  if isinstance(device, tuple) and axis is not None:
    return Tensor(Tensor.invalids(*shard_shape(shape, axis, len(device)), dtype=dtype, device=device).uop.multi(axis), device=device)
  return Tensor.invalids(*shape, dtype=dtype, device=device)

def alloc_local(shape, dtype, device, axis=None) -> Tensor:
  if isinstance(device, tuple) and axis is not None:
    return Tensor(Tensor.invalids(*shape, dtype=dtype, device=device).uop.multi(0), device=device)
  return Tensor.invalids(*shape, dtype=dtype, device=device)

def compile_hip(src:str, defines:list[str]):
  return HIPCCCompiler("gfx950", ["-std=c++20", "-ffast-math", *defines]).compile_cached(src)

def compile_cpp(cpp_dir:pathlib.Path, cpp_name:str, n_elems:int, hidden:int):
  src = (cpp_dir/cpp_name).read_text()
  return src, compile_hip(src, [f"-DN_ELEMS={n_elems}", f"-DHIDDEN={hidden}", f"-DNUM_WG={NUM_WG}", f"-DTHREADS_PER_WG={THREADS_PER_WG}"])
