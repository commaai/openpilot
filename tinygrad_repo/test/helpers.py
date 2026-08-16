import os, time, struct, functools, unittest
from dataclasses import replace
from typing import Any, Callable
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.tensor import _to_np_dtype
from tinygrad.codegen import to_program
from tinygrad.dtype import DType, truncate
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import T, Target, DEV, DEBUG, Context, GlobalCounters
from tinygrad.renderer import Renderer
from tinygrad.codegen import full_rewrite_to_sink, line_rewrite, pm_linearize_cleanups
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.engine.realize import compile_linear

# decorator to skip slow tests by default, run with RUN_SLOW=1 to include them
slow = unittest.skipUnless(os.getenv("RUN_SLOW"), "slow test, set RUN_SLOW=1 to run")
from tinygrad.runtime.ops_python import PythonRenderer

def full_rewrite(sink:UOp, ren:Renderer|None=None) -> UOp:
  if ren is None: ren = Renderer(Target())
  if sink.arg is None: sink = sink.replace(arg=KernelInfo())
  return full_rewrite_to_sink(sink, ren, optimize=sink.tag is None)

def get_uops(sink:UOp, ren:Renderer|None=None) -> list[UOp]:
  """Extract linearized UOps from a sink. Test helper that only does linearization (no render)."""
  full_sink = full_rewrite(sink, ren)
  return line_rewrite(linearize(full_sink), pm_linearize_cleanups)

def replace_opts(ast:UOp, opts:list) -> UOp: return ast.replace(arg=replace(ast.arg, opts_to_apply=tuple(opts)))

def derandomize_model(model):
  for p in get_parameters(model):
    p.replace(Tensor.empty(p.shape, device=p.device, dtype=p.dtype))
    p.realize()

class KernelCountException(Exception):
  def __init__(self, expected:int, got:int):
    self.expected, self.got = expected, got
    super().__init__(f"expected {expected}, got {got}")

def check_schedule(t:Tensor|list[Tensor]|UOp, allowed:int, to_prerealize:list[Tensor]|None=None, filter_sink=True):
  if to_prerealize:
    with Context(DEBUG=0, TRACK_MATCH_STATS=0): Tensor.realize(*to_prerealize)
  if isinstance(t, Tensor): linear, var_vals = t.linear_with_vars()
  elif isinstance(t, list) and isinstance(t[0], Tensor): linear, var_vals = Tensor.linear_with_vars(*t)
  else:
    assert isinstance(t, UOp), f"can't schedule {t}"
    linear, var_vals = Tensor(t).linear_with_vars()
  kernel_cnt = sum((len(call.device) if isinstance(call.device, tuple) else 1)
                   for call in linear.src if call.src[0].op is Ops.SINK or not filter_sink)
  if kernel_cnt != allowed:
    print(f"SCHEDULE ISSUE, expecting {allowed} got {kernel_cnt}")
    if DEBUG >= 3:
      for i,call in enumerate(linear.src):
        print("kernel", i+1)
        print(call.src[0])
    raise KernelCountException(allowed, kernel_cnt)
  # test compiling the linear
  compile_linear(linear)
  return linear, var_vals

def assert_kernel_count(expected:int):
  got = GlobalCounters.kernel_count
  if got != expected: raise KernelCountException(expected, got)

def call_is_graph(call:UOp) -> bool:
  ast = call.src[0]
  return ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "graph"

def call_is_hcq(call:UOp) -> bool:
  ast = call.src[0]
  return ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "hcq"

def jit_cache_count(linear:UOp) -> int:
  n = 0
  for call in linear.src:
    ast = call.src[0]
    if ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "graph": n += jit_cache_count(ast.src[0])
    else: n += 1
  return n

def assert_jit_cache_len(fxn, expected_len):
  linear = fxn.captured.linear if fxn.captured is not None else None
  if linear is None or not linear.src:
    if expected_len != 0: raise KernelCountException(expected_len, 0)
    return
  if expected_len and all(call_is_hcq(call) for call in linear.src): expected_len = 3 # HCQ2: merged same-queue calls + finalizer + bumps
  if call_is_graph(linear.src[0]):
    if len(linear.src) != 1: raise KernelCountException(1, len(linear.src))
    inner = linear.src[0].src[0].src[0]  # LINEAR UOp inside CUSTOM_FUNCTION
    if len(inner.src) != expected_len: raise KernelCountException(expected_len, len(inner.src))
  else:
    if len(linear.src) != expected_len: raise KernelCountException(expected_len, len(linear.src))

def min_normal(dt:DType) -> float: return 2.0 ** (2 - (1 << (dtypes.finfo(dt)[0] - 1)))

def rand_for_dtype(dt:DType, size:int, allow_subnormal=True):
  if dtypes.is_unsigned(dt):
    return np.random.randint(0, 100, size=size, dtype=_to_np_dtype(dt))
  elif dtypes.is_int(dt):
    return np.random.randint(-100, 100, size=size, dtype=_to_np_dtype(dt))
  elif dt == dtypes.bool:
    return np.random.choice([True, False], size=size)
  ret = np.random.uniform(-10, 10, size=size).astype(_to_np_dtype(dt))
  if dt == dtypes.bfloat16 or dt in dtypes.fp8s: ret = np.array([truncate[dt](x) for x in ret], dtype=ret.dtype)
  if not allow_subnormal: ret = np.where(np.abs(ret) < min_normal(dt), 0, ret)
  return ret

def timeit(fxn:Callable[..., T], *args, **kwargs) -> tuple[T, float]:
  st = time.perf_counter_ns()
  ret = fxn(*args, **kwargs)
  return ret, (time.perf_counter_ns()-st)*1e-6

def eval_uop(uop:UOp, inputs:list[tuple[DType, list[Any]]]|None=None, vals:tuple[int, ...]=()):
  dev = Device['PYTHON']
  allocator = dev.allocator
  bufs = []
  for buf_dt, data in inputs or []:
    bufs.append(buf:=allocator.alloc(len(data) * buf_dt.itemsize))
    allocator._copyin(buf, memoryview(struct.pack(str(len(data)) + (buf_dt.fmt or ""), *data)))
  g = UOp.param(0, uop.dtype, (1,))
  prg = to_program(UOp.store(g.index(UOp.const(0)), uop).sink(arg=KernelInfo()), PythonRenderer(Target("PYTHON")))
  prog = dev.runtime(prg.to_elf())
  prog(out_buf:=allocator.alloc(uop.dtype.itemsize), *bufs, vals=vals)
  return out_buf.cast(uop.dtype.fmt or "").tolist()[0]

def to_uops_list(u:list[UOp], ren=None) -> list[UOp]:
  sink = UOp.group(*u)
  for r in sink.ranges: sink = sink.end(r)
  ret = get_uops(sink.sink(arg=KernelInfo(opts_to_apply=())), ren)
  assert ret[-1].op is Ops.SINK
  return ret

def not_support_multi_device():
  # CL and CUDA don't support multi device if in CI
  return (Device.DEFAULT == "CL" and Device[Device.DEFAULT].count() < 2) or (Device.DEFAULT == "CUDA" and DEV.interface.startswith("MOCK"))

def needs_second_gpu(fn):
  @functools.wraps(fn)
  def wrapper(self, *args, **kwargs):
    # check if there's a second GPU, if not, skip multi tests
    try: Tensor.zeros(10, device=f"{Device.DEFAULT}:1").contiguous().realize()
    except Exception as e: self.skipTest(f"second device not available: {e}")
    return fn(self, *args, **kwargs)
  return wrapper
