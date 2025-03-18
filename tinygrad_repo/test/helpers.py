import time, logging, difflib
from typing import Callable, Optional, Tuple
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.ops import UOp, Ops, sint
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.tensor import _to_np_dtype
from tinygrad.engine.realize import Runner
from tinygrad.dtype import ConstType, DType
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import T, getenv, colored
from tinygrad.codegen.linearize import linearize_uop
from tinygrad.codegen.uopgraph import full_graph_rewrite
from tinygrad.runtime.ops_python import PythonProgram, PythonRenderer, PythonCompiler, PythonAllocator

def derandomize_model(model):
  for p in get_parameters(model):
    p.lazydata = Tensor.empty(p.shape, device=p.device, dtype=p.dtype).lazydata
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

def print_diff(s0, s1, unified=getenv("UNIFIED_DIFF",1)):
  if not logging.getLogger().hasHandlers(): logging.basicConfig(level=logging.INFO, format="%(message)s")
  if unified:
    lines = list(difflib.unified_diff(str(s0).splitlines(), str(s1).splitlines()))
    diff = "\n".join(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None) for line in lines)
  else:
    import ocdiff
    diff = ocdiff.console_diff(str(s0), str(s1))
  logging.info(diff)

def ast_const(dtype:DType, val:ConstType, shape:Tuple[sint, ...]=(), st:Optional[ShapeTracker]=None, st_src:Optional[Tuple[UOp]]=None) -> UOp:
  if st_src is None:
    st_src = (st.to_uop() if st is not None else ShapeTracker.from_shape(()).reshape((1,)*len(shape)).expand(shape).to_uop(),)
  return UOp(Ops.VALID, dtypes.bool, st_src).where(UOp.const(dtype, val), UOp.const(dtype, 0))

def timeit(fxn:Callable[..., T], *args, **kwargs) -> Tuple[T, float]:
  st = time.perf_counter_ns()
  ret = fxn(*args, **kwargs)
  return ret, (time.perf_counter_ns()-st)*1e-6

def eval_uop(uop:UOp):
  g = UOp(Ops.DEFINE_GLOBAL, uop.dtype.ptr(), arg=0, src=())
  rw = full_graph_rewrite(UOp.store(g.index(UOp.const(dtypes.int, 0)), uop).sink(), PythonRenderer)
  prog = PythonProgram("run", PythonCompiler().compile(PythonRenderer().render("run", linearize_uop(rw))))
  buf = PythonAllocator().alloc(uop.dtype.itemsize)
  prog(buf)
  return buf.cast(uop.dtype.fmt).tolist()[0]
