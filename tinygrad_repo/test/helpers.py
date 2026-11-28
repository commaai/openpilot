import sys, gc, time, struct, unittest, contextlib
from dataclasses import replace
from typing import Any, Callable, Iterator
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.device import Buffer
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.tensor import _to_np_dtype
from tinygrad.codegen import to_program
from tinygrad.dtype import DType, truncate, AddrSpace
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import T, Target, DEV, DEBUG, Context, GlobalCounters, VALIDATE_WITH_CPU
from tinygrad.renderer import Renderer, TinyELF
from tinygrad.codegen import full_rewrite_to_sink, linearize_cleanups
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.engine.realize import link_linear, run_linear
from tinygrad.codegen.compile import compile_linear
from tinygrad.engine.jit import bind_export
from tinygrad.runtime.ops_python import PythonRenderer

def full_rewrite(sink:UOp, ren:Renderer|None=None) -> UOp:
  if ren is None: ren = Renderer(Target())
  if sink.arg is None: sink = sink.replace(arg=KernelInfo())
  return full_rewrite_to_sink(sink, ren, optimize=sink.arg is None or not sink.arg.optimized)

def get_uops(sink:UOp, ren:Renderer|None=None) -> list[UOp]:
  """Extract linearized UOps from a sink. Test helper that only does linearization (no render)."""
  full_sink = full_rewrite(sink, ren)
  return linearize_cleanups(linearize(full_sink))

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

def bound_schedule(*ts:Tensor) -> UOp: return bind_export(*Tensor.linear_with_externs(*ts))

def run_schedule(linear:UOp, externs:tuple[UOp, ...]=(), var_vals:dict[str, int]|None=None, update_stats=True, wait=False) -> None:
  var_vals = {**{f"e{i}": x.arg.val for i,x in enumerate(externs) if x.is_bound_var}, **(var_vals or {})}
  linked = link_linear(compile_linear(linear, validate=VALIDATE_WITH_CPU), list(externs))
  run_linear(linked, var_vals, externs, update_stats=update_stats, wait=wait)

def check_schedule(t:Tensor|list[Tensor]|UOp, allowed:int, to_prerealize:list[Tensor]|None=None, filter_sink=True):
  if to_prerealize:
    with Context(DEBUG=0, TRACK_MATCH_STATS=0): Tensor.realize(*to_prerealize)
  if isinstance(t, Tensor): linear, externs = t.linear_with_externs()
  elif isinstance(t, list) and isinstance(t[0], Tensor): linear, externs = Tensor.linear_with_externs(*t)
  else:
    assert isinstance(t, UOp), f"can't schedule {t}"
    linear, externs = Tensor(t).linear_with_externs()
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
  return linear, externs

@contextlib.contextmanager
def measure_mem() -> Iterator[Callable[[], int]]:
  Device.default.synchronize()
  gc.collect()
  base = GlobalCounters.mem_used
  yield lambda: (gc.collect(), GlobalCounters.mem_used - base)[1]

def assert_kernel_count(expected:int):
  got = GlobalCounters.kernel_count
  if got != expected: raise KernelCountException(expected, got)

def is_hcq2_device() -> bool: # an hcq2 device stages every copy from the host through a pinned buffer: such a copy is two calls, not one
  from tinygrad.runtime.support.hcq2 import HCQ_DEVS
  return Device.DEFAULT.split(":")[0] in HCQ_DEVS

def jit_cache_count(linear:UOp) -> int:
  return sum(len(c.arg.hcq.kernels) if (c:=call.without_after).arg.hcq is not None else 1 for call in linear.src)

def assert_jit_cache_len(fxn, expected_len):
  linear = fxn.captured._linear if fxn.captured is not None else None
  if linear is None or not linear.src:
    if expected_len != 0: raise KernelCountException(expected_len, 0)
    return
  # HCQ2: kernels batch into submits, the finalizers carry the batch's kernels
  if expected_len and any(call.without_after.arg.hcq is not None for call in linear.src):
    if (count:=jit_cache_count(linear)) != expected_len: raise KernelCountException(expected_len, count)
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
  prog = dev.runtime(TinyELF.from_program(prg))
  out_buf = Buffer("PYTHON", uop.dtype.itemsize, preallocate=True)
  prog(out_buf._buf, *[b.buf for b in bufs], *vals)
  return out_buf.as_memoryview().cast(uop.dtype.fmt or "").tolist()[0]

def to_uops_list(u:list[UOp], ren=None) -> list[UOp]:
  sink = UOp.sink(*u)
  for r in sink.ranges: sink = sink.end(r)
  ret = get_uops(sink.sink(arg=KernelInfo(opts_to_apply=())), ren)
  assert ret[-1].op is Ops.SINK
  return ret

def needs_devices(n:int):
  # Only these backends use the device suffix to index a physical accelerator. CPU, Metal, etc. open logical devices.
  device = Device.DEFAULT
  return unittest.skipIf(device in ("AMD", "NV", "CUDA", "HIP", "CL") and Device[device].count() < n, f"needs {n} {device} devices")
