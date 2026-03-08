import os, time, struct, functools, unittest
from typing import Any, Callable
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.tensor import _to_np_dtype
from tinygrad.engine.realize import Runner, get_program
from tinygrad.dtype import DType
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import T, CI
from tinygrad.renderer import Renderer
from tinygrad.codegen import full_rewrite_to_sink, line_rewrite, pm_linearize_cleanups
from tinygrad.codegen.late.linearizer import linearize

# decorator to skip slow tests by default, run with RUN_SLOW=1 to include them
slow = unittest.skipUnless(os.getenv("RUN_SLOW"), "slow test, set RUN_SLOW=1 to run")
from tinygrad.runtime.ops_python import PythonProgram, PythonRenderer, PythonCompiler

def get_uops(sink:UOp, ren:Renderer|None=None) -> list[UOp]:
  """Extract linearized UOps from a sink. Test helper that only does linearization (no render)."""
  if ren is None: ren = Renderer()
  if sink.arg is None: sink = sink.replace(arg=KernelInfo())
  full_sink = full_rewrite_to_sink(sink, ren, optimize=sink.tag is None)
  return line_rewrite(linearize(full_sink), pm_linearize_cleanups)

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
    allocator._copyin(buf, memoryview(struct.pack(str(len(data)) + (buf_dt.fmt or ""), *data)))
  g = UOp(Ops.PARAM, uop.dtype.ptr(), arg=0, src=())
  prg = get_program(UOp.store(g.index(UOp.const(dtypes.int, 0)), uop).sink(), PythonRenderer())
  prog = PythonProgram("run", PythonCompiler().compile(prg.src))
  prog(out_buf:=allocator.alloc(uop.dtype.itemsize), *bufs)
  return out_buf.cast(uop.dtype.fmt or "").tolist()[0]

def not_support_multi_device():
  # CL and CUDA don't support multi device if in CI
  return CI and Device.DEFAULT in ("CL", "CUDA")

def needs_second_gpu(fn):
  @functools.wraps(fn)
  def wrapper(self, *args, **kwargs):
    # check if there's a second GPU, if not, skip multi tests
    try: Tensor.zeros(10, device=f"{Device.DEFAULT}:1").contiguous().realize()
    except Exception as e: self.skipTest(f"second device not available: {e}")
    return fn(self, *args, **kwargs)
  return wrapper
