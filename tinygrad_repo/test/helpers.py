import os, sys, time, struct, functools, unittest
from dataclasses import replace
from typing import Any, Callable
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.device import Buffer
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.tensor import _to_np_dtype
from tinygrad.codegen import to_program
from tinygrad.dtype import DType, truncate, AddrSpace
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

def buffer_uops(ast:UOp, bufs:list[Buffer]) -> list[UOp]:
  params = {p.arg.slot:p.dtype for p in ast.toposort() if p.op is Ops.PARAM and p.addrspace is not AddrSpace.ALU}
  return [UOp.from_buffer(b, params[i]) for i, b in enumerate(bufs)]

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
  # test compiling the linear
  compile_linear(linear)
  kernel_cnt = sum((len(call.device) if isinstance(call.device, tuple) else 1)
                   for call in linear.src if call.src[0].op is Ops.SINK or not filter_sink)
  if kernel_cnt != allowed:
    print(f"SCHEDULE ISSUE, expecting {allowed} got {kernel_cnt}")
    if DEBUG >= 3:
      for i,call in enumerate(linear.src):
        print("kernel", i+1)
        print(call.src[0])
    raise KernelCountException(allowed, kernel_cnt)
  return linear, var_vals

def assert_kernel_count(expected:int):
  got = GlobalCounters.kernel_count
  if got != expected: raise KernelCountException(expected, got)

def is_hcq2_device() -> bool: # an hcq2 device stages every copy from the host through a pinned buffer: such a copy is two calls, not one
  from tinygrad.runtime.support.hcq2 import HCQ_DEVS
  return Device.DEFAULT.split(":")[0] in HCQ_DEVS

def call_is_hcq(call:UOp) -> bool: # an hcq2 batch: a compiled body whose aux lists the kernels it submits
  from tinygrad.runtime.support.hcq2 import HCQInfo
  return isinstance(getattr(call.without_after.arg, "aux", None), HCQInfo)

def jit_cache_count(linear:UOp) -> int:
  return sum(len(call.without_after.arg.aux.kernels) if call_is_hcq(call) else 1 for call in linear.src)

def assert_jit_cache_len(fxn, expected_len):
  linear = fxn.captured.linear if fxn.captured is not None else None
  if linear is None or not linear.src:
    if expected_len != 0: raise KernelCountException(expected_len, 0)
    return
  if expected_len and any(call_is_hcq(call) for call in linear.src): # HCQ2: kernels batch into submits, the finalizers carry the batch's kernels
    count = sum(len(call.without_after.arg.aux.kernels) if call_is_hcq(call) else 1 for call in linear.src)
    if count != expected_len: raise KernelCountException(expected_len, count)
    return
  if len(linear.src) != expected_len: raise KernelCountException(expected_len, len(linear.src))

def prepare_test_op(low, high, shps, vals, forward_only=False):
  import torch
  if shps is None:
    ts = [torch.tensor(x, requires_grad=(not forward_only)) for x in vals]
  else:
    np.random.seed(0)
    np_data = [np.random.uniform(low=low, high=high, size=size).astype(_to_np_dtype(dtypes.default_float)) for size in shps]
    ts = [torch.tensor(data, requires_grad=(not forward_only)) for data in np_data]
  for i in range(len(ts)):
    # NOTE: torch default int64 for python ints input
    if ts[i].dtype == torch.int64: ts[i] = ts[i].type(torch.int32)
  tst = [Tensor(x.detach().cpu().numpy()) for x in ts]
  return ts, tst

class TensorTestCase(unittest.TestCase):
  def helper_test_exception(self, shps, torch_fxn, tinygrad_fxn=None, expected=None, forward_only=False, exact=False, vals=None, low=-1.5, high=1.5):
    if DEV.interface.startswith("MOCK") and Device.DEFAULT == "NV": self.skipTest('helper_test_exception fails in CI CUDA')
    ts, tst = prepare_test_op(low, high, shps, vals, forward_only)
    if tinygrad_fxn is None:
      tinygrad_fxn = torch_fxn
    with self.assertRaises(expected) as torch_cm:
      torch_fxn(*ts)
    with self.assertRaises(expected) as tinygrad_cm:
      tinygrad_fxn(*tst)
    if exact: self.assertEqual(str(torch_cm.exception), str(tinygrad_cm.exception))
    if sys.stdout.isatty(): print("\ntesting %40r   torch/tinygrad exception: %s / %s" % (shps, torch_cm.exception, tinygrad_cm.exception), end="")

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
    allocator._copyin(buf.buf, memoryview(struct.pack(str(len(data)) + (buf_dt.fmt or ""), *data)))
  g = UOp.param(0, uop.dtype, 1)
  prg = to_program(UOp.store(g.index(UOp.const(0)), uop).sink(arg=KernelInfo()), PythonRenderer(Target("PYTHON")))
  prog = dev.runtime(prg.to_elf())
  out_buf = Buffer("PYTHON", uop.dtype.itemsize, preallocate=True)
  prog(out_buf._buf, *[b.buf for b in bufs], vals=vals)
  return out_buf.as_memoryview().cast(uop.dtype.fmt or "").tolist()[0]

def to_uops_list(u:list[UOp], ren=None) -> list[UOp]:
  sink = UOp.sink(*u)
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
