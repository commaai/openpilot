import time, struct
from typing import Any, Callable
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.uop.ops import UOp, Ops
from tinygrad.tensor import _to_np_dtype
from tinygrad.engine.realize import Runner
from tinygrad.dtype import DType
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import T, CI
from tinygrad.codegen import full_rewrite
from tinygrad.runtime.ops_python import PythonProgram, PythonRenderer, PythonCompiler

def derandomize_model(model):
  for p in get_parameters(model):
    p.replace(Tensor.empty(p.shape, device=p.device, dtype=p.dtype))
    p.realize()

def assert_jit_cache_len(fxn, expected_len):
  if not fxn.jit_cache:
    assert expected_len == 0, expected_len
    return
  # until we have a better way of typing the prg in ExecItem
  if issubclass(type(fxn.jit_cache[0].prg), Runner) and not type(fxn.jit_cache[0].prg).__name__.endswith('Graph'):
    assert len(fxn.jit_cache) == expected_len, f"expected {expected_len}, got {len(fxn.jit_cache)}"
  else:
    assert len(fxn.jit_cache) == 1, len(fxn.jit_cache)
    # until we have a better way of typing the prg in ExecItem
    assert type(fxn.jit_cache[0].prg).__name__.endswith('Graph')
    assert len(fxn.jit_cache[0].prg.jit_cache) == expected_len

def rand_for_dtype(dt:DType, size:int):
  if dtypes.is_unsigned(dt):
    return np.random.randint(0, 100, size=size, dtype=_to_np_dtype(dt))
  elif dtypes.is_int(dt):
    return np.random.randint(-100, 100, size=size, dtype=_to_np_dtype(dt))
  elif dt == dtypes.bool:
    return np.random.choice([True, False], size=size)
  return np.random.uniform(-10, 10, size=size).astype(_to_np_dtype(dt))

def timeit(fxn:Callable[..., T], *args, **kwargs) -> tuple[T, float]:
  st = time.perf_counter_ns()
  ret = fxn(*args, **kwargs)
  return ret, (time.perf_counter_ns()-st)*1e-6

def eval_uop(uop:UOp, inputs:list[tuple[DType, list[Any]]]|None=None):
  allocator = Device['PYTHON'].allocator
  bufs = []
  for buf_dt, data in inputs or []:
    bufs.append(buf:=allocator.alloc(len(data) * buf_dt.itemsize))
    allocator._copyin(buf, memoryview(struct.pack(str(len(data)) + buf_dt.fmt, *data)))
  g = UOp(Ops.DEFINE_GLOBAL, uop.dtype.ptr(), arg=0, src=())
  opts = PythonRenderer()
  lst = full_rewrite(UOp.store(g.index(UOp.const(dtypes.int, 0)), uop).sink(), opts)
  prog = PythonProgram("run", PythonCompiler().compile(opts.render(lst)))
  prog(out_buf:=allocator.alloc(uop.dtype.itemsize), *bufs)
  return out_buf.cast(uop.dtype.fmt).tolist()[0]

def not_support_multi_device():
  # GPU and CUDA don't support multi device if in CI
  return CI and REAL_DEV in ("GPU", "CUDA")

# NOTE: This will open REMOTE if it's the default device
REAL_DEV = (Device.DEFAULT if Device.DEFAULT != "REMOTE" else Device['REMOTE'].properties.real_device)
