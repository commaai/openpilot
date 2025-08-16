from typing import cast
from dataclasses import dataclass, field
from collections import deque, defaultdict
from tinygrad.uop.ops import UOp, Variable, Ops, buffers
from tinygrad.device import Device, Buffer, MultiBuffer
from tinygrad.helpers import Metadata, all_same

# **** ScheduleItem return type

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...] = ()
  fixedvars: dict[Variable, int] = field(default_factory=dict)

# **** schedule linearizer

def create_schedule_with_vars(sched_sink:UOp) -> tuple[list[ScheduleItem], dict[Variable, int]]:
  # construct the KERNEL children graph based on assigns
  children: defaultdict[UOp, list[UOp]] = defaultdict(list)
  in_degree: dict[UOp, int] = {}
  var_vals: dict[Variable, int] = {}
  for u in sched_sink.toposort():
    if u.op is not Ops.ASSIGN: continue  # anything that's not an ASSIGN doesn't write a kernel, so we can skip
    k = u.src[1]
    in_degree.setdefault(k, 0)
    for s in k.src:
      if s.op is Ops.ASSIGN:
        children[s.src[1]].append(k)
        in_degree[k] += 1
      elif s.op in {Ops.MSELECT, Ops.MSTACK}:
        for ss in s.src:
          if ss.op is Ops.MSELECT: ss = ss.src[0]
          if ss.op is not Ops.BUFFER:
            assert ss.op is Ops.ASSIGN
            children[ss.src[1]].append(k)
            in_degree[k] += 1
      elif s.op is Ops.BUFFER:
        pass  # a BUFFER is already realized, nothing to do here
      elif s.op is Ops.BIND:
        var, val = s.unbind()
        assert var not in var_vals or var_vals[var] == val, f"bind mismatch on {var}, {var_vals[var]} != {val}"
        var_vals[var] = val
      else:
        raise RuntimeError(f"input to kernel must be ASSIGN or BUFFER, not {s.op}")

  # linearize KERNEL UOps into ScheduleItems in BFS order

  def _heuristic(k: UOp):
    if k.arg.ast.op is Ops.COPY and not all_same([Device[cast(Buffer, s.buf_uop.buffer).device].group_id for s in k.src]): return 1000
    return 0

  last_heuristic: int = 0
  queues: defaultdict[int, deque[UOp]] = defaultdict(deque)
  last_queue: deque[UOp] = deque()
  for k,v in in_degree.items():
    if v == 0: queues[_heuristic(k)].append(k)

  schedule: list[ScheduleItem] = []
  while last_queue or any(queues.values()):
    if not last_queue: last_heuristic, last_queue = min((it for it in queues.items() if it[1]), key=lambda x: abs(x[0]-last_heuristic))
    k = last_queue.popleft()
    ast = k.arg.ast
    # create subbuffers if needed
    if ast.op is Ops.BUFFER_VIEW:
      base = k.src[1].buf_uop.buffer
      assert isinstance(base, Buffer), "base can't be MultiBuffer"
      buffers[k.src[0]] = base.view(k.size, ast.dtype, ast.arg[1]*base.dtype.itemsize)
    ubufs = tuple(s.buf_uop.buffer for s in k.src if s.op is not Ops.BIND)
    if any(isinstance(x, MultiBuffer) for x in ubufs):
      assert all(isinstance(x, MultiBuffer) for x in ubufs), "kernel must all be multibuffer"
      dnums = [x for x in ast.variables() if x.arg[0] == '_device_num']
      for i,bufs in enumerate(zip(*[x.bufs for x in cast(tuple[MultiBuffer, ...], ubufs)])):
        schedule.append(ScheduleItem(ast, bufs, k.arg.metadata, {dnums[0]:i} if len(dnums) else {}))
    else:
      # ONE -> ONE
      schedule.append(ScheduleItem(ast, cast(tuple[Buffer, ...], ubufs), k.arg.metadata))
    for x in children[k]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queues[_heuristic(x)].append(x)

  return schedule, var_vals
