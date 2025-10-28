import itertools
import numpy as np
from typing import DefaultDict, Dict, List, Set, Tuple, TypeVar, Union
from tinygrad.device import Buffer
from tinygrad.engine.realize import capturing, lower_schedule_item
from tinygrad.helpers import DEBUG, MULTIOUTPUT, colored, getenv
from tinygrad.engine.schedule import LBScheduleItem, _graph_schedule, ScheduleItem
from tinygrad.uop.ops import Ops, UOp
from tinygrad.tensor import Tensor, _to_np_dtype

ctx_vars = { MULTIOUTPUT: (0, 1) }
FUZZ_SCHEDULE_MAX_PATHS = getenv("FUZZ_SCHEDULE_MAX_PATHS", 10)

def fuzz_schedule(outs:List[UOp]):
  # find toposorts across all tunable params
  unique_ts: Dict[Tuple[LBScheduleItem, ...], Dict[str, int]] = {}
  for combination in itertools.product(*ctx_vars.values()):
    for var, val in zip(ctx_vars, combination): var.value = val
    ctx_var_values = dict(zip([v.key for v in ctx_vars], combination))
    graph, in_degree, _ = _graph_schedule(outs)
    for ts in find_all_toposorts(graph, in_degree): unique_ts[ts] = ctx_var_values
  toposorts = list(unique_ts.items())
  if DEBUG >= 1: print(colored(f"fuzzing {len(toposorts)} schedule permutations", "yellow"))

  # setup ground truth
  ground_truth: Dict[UOp, memoryview] = {}
  assign_targets: Dict[UOp, UOp] = {}
  # IMPORTANT: freeze prerealized bufs before ScheduleItem exec
  prerealized: Dict[UOp, memoryview] = {}
  seed = Tensor._seed
  ts,_ = toposorts[0]
  for lsi in ts:
    for out in lsi.outputs:
      # freeze assign state before exec
      if out.op is Ops.ASSIGN:
        prerealized[out] = out.buffer.as_buffer()
        assign_targets[out.srcs[1]] = out
    for x in lsi.inputs:
      if x not in ground_truth and x.device != "NPY": prerealized[x] = x.buffer.as_buffer()
    si = ScheduleItem(lsi.ast, tuple(x.buffer for x in lsi.outputs+lsi.inputs if x.size != 0), lsi.metadata)
    _exec_si(si, seed)
    for out in lsi.outputs:
      ground_truth[out] = out.buffer.as_buffer()
      del out.srcs # only schedule the LazyBuffer in this fuzz run

  # exec and validate each permutation with new Buffers
  for i, (ts, ctx) in enumerate(toposorts[1:]):
    if DEBUG >= 1: print(colored(f"testing permutation {i} {ctx}", "yellow"))
    rawbufs: Dict[UOp, Buffer] = {}
    for lsi in ts:
      for out in lsi.outputs:
        base = rawbufs[lsi.inputs[0]].base if out.op is Ops.BUFFER_VIEW else None
        rawbufs[out] = Buffer(out.buffer.device, out.buffer.size, out.buffer.dtype, base=base)
        if out.op is Ops.ASSIGN: rawbufs[out].ensure_allocated().copyin(prerealized[out])
      for x in lsi.inputs:
        if x not in rawbufs:
          # override the assign_target after ASSIGN
          if x in assign_targets and assign_targets[x] in rawbufs: rawbufs[x] = rawbufs[assign_targets[x]]
          elif x.device == "NPY": rawbufs[x] = x.buffer
          # copy the pre realized input
          else: rawbufs[x] = Buffer(x.buffer.device, x.buffer.size, x.buffer.dtype, initial_value=bytes(prerealized[x]))
      si = ScheduleItem(lsi.ast, tuple(rawbufs[x] for x in lsi.bufs if x.size != 0), lsi.metadata)
      _exec_si(si, seed)
      for out in lsi.outputs:
        outbuf = np.frombuffer(rawbufs[out].as_buffer(), _to_np_dtype(out.dtype))
        try: np.testing.assert_allclose(outbuf, np.frombuffer(ground_truth[out], _to_np_dtype(out.dtype)), atol=1e-2, rtol=1e-2)
        except Exception as e:
          print(f"FAILED FOR {out}")
          raise e

def _exec_si(si:ScheduleItem, seed:int):
  ei = lower_schedule_item(si)
  if len(capturing): capturing[0].add(ei)
  ei.run()

T = TypeVar("T")
def find_all_toposorts(graph:DefaultDict[T, List[T]], in_degree:Union[DefaultDict[T, int], Dict[T, int]]) -> List[Tuple[T, ...]]:
  visited: Set[T] = set()
  ret: List[Tuple[T, ...]] = []
  path: List[T] = []

  def recurse_paths(path:List[T]):
    for v, d in in_degree.items():
      if d != 0 or v in visited: continue
      for u in graph[v]: in_degree[u] -= 1
      path.append(v)
      visited.add(v)
      recurse_paths(path)
      if len(ret) >= FUZZ_SCHEDULE_MAX_PATHS: return
      # backtrack
      for u in graph[v]: in_degree[u] += 1
      path.pop()
      visited.remove(v)
    if len(path) == len(in_degree): ret.append(tuple(path))
  recurse_paths(path)

  if len(ret) == 0: raise RuntimeError("detected cycle in the graph")
  # verify all paths are unique
  assert len(ret) == len(set(ret))
  return ret
