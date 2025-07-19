from typing import cast
from collections import defaultdict
from tinygrad.engine.schedule import ScheduleItem
from tinygrad.device import Device, Buffer
from tinygrad.helpers import NO_MEMORY_PLANNER, dedup, DEBUG, round_up
from tinygrad.uop.ops import Ops
from tinygrad.dtype import dtypes, ImageDType
from tinygrad.runtime.support.memory import TLSFAllocator

# **************** memory planning ****************

def _internal_memory_planner(buffers:list[list[Buffer]], noopt_buffers=None, ignore_checks=False, debug_prefix="") -> dict[Buffer, Buffer]:
  if NO_MEMORY_PLANNER: return {}
  first_appearance, last_appearance, buf_to_opt = {}, {}, set()
  for i,u in enumerate(buffers):
    for buf in u:
      should_skip = buf.is_allocated() or buf.base.is_allocated() or buf.uop_refcount > 0 or (noopt_buffers is not None and buf.base in noopt_buffers)
      if not ignore_checks and should_skip: continue
      if buf.base not in first_appearance: first_appearance[buf.base] = i
      last_appearance[buf.base] = i
      buf_to_opt.add(buf)

  # Sort buffer operations in timeline order. Two events: buffer is allocated or buffer is freed.
  buffer_requests = sorted([((first_appearance[buf], True), buf) for buf in first_appearance.keys()] + \
                           [((last_appearance[buf] + 1, False), buf) for buf in first_appearance.keys()], key=lambda x: x[0])

  # Try to suballocate from a shared buffer managed by global_planner using TLSFAllocator.
  # Also track buffer replacements for buffers that do not support suballocation.
  buffer_replace:dict[Buffer, tuple[Buffer|None, int|None]] = {}
  reuse_buffers:dict[tuple, list[Buffer]] = defaultdict(list)
  global_planner:dict[str, tuple[int, TLSFAllocator]] = defaultdict(lambda: (0, TLSFAllocator(1 << 44, block_size=0x1000, lv2_cnt=32)))
  for (_, is_open_ev), buf in buffer_requests:
    # Check if suballocation is possible for the given buffer and device.
    if hasattr(Device[buf.device].allocator, "_offset") and not isinstance(buf.dtype, ImageDType):
      if is_open_ev: buffer_replace[buf] = (None, global_planner[buf.device][1].alloc(round_up(buf.nbytes, 0x1000)))
      else: global_planner[buf.device][1].free(cast(int, buffer_replace[buf][1]))
      global_planner[buf.device] = (max(global_planner[buf.device][0], buffer_replace[buf][1] + buf.nbytes), global_planner[buf.device][1])
    else:
      key = (buf.device, buf.dtype, buf.options, buf.nbytes)
      if is_open_ev: buffer_replace[buf] = (reuse_buffers[key].pop(), None) if key in reuse_buffers and len(reuse_buffers[key]) > 0 else (buf, None)
      else: reuse_buffers[key].append(cast(Buffer, buffer_replace[buf][0]))

  # Allocate global buffers based on the memory planner.
  global_buffers = {dev: Buffer(dev, round_up(sz, 0x1000), dtypes.int8) for dev, (sz, _) in global_planner.items()}
  buffer_resolve:dict[Buffer, tuple[Buffer, int|None]] = {buf: (base or global_buffers[buf.device], off) for buf,(base,off) in buffer_replace.items()}

  # Assign buffers. First, assign full buffers (not sub-buffers).
  assigned:dict[Buffer, Buffer] = {}
  for buf, (base, off) in buffer_resolve.items():
    if buf != base:
      assigned[buf] = base if off is None else Buffer(buf.device, buf.size, buf.dtype, base=base, offset=off)

  # Now assign sub-buffers.
  for buf in buf_to_opt:
    if buf._base is not None:
      assigned[buf] = Buffer(buf.device, buf.size, buf.dtype, base=(pbuf:=assigned.get(buf.base, buf.base)).base, offset=pbuf.offset+buf.offset)

  if DEBUG >= 1:
    ak, av = dedup(x for x in assigned.keys() if x._base is None),dedup(x for x in assigned.values() if x._base is None)+list(global_buffers.values())
    omem, nmem = sum([x.nbytes for x in ak])/1e6, sum([x.nbytes for x in av])/1e6
    if omem != nmem: print(f"{debug_prefix}memory reduced from {omem:.2f} MB -> {nmem:.2f} MB,", f"{len(ak)} -> {len(av)} bufs")

  return assigned

def memory_planner(schedule:list[ScheduleItem]) -> list[ScheduleItem]:
  # Exclude buffers involved in load ops (e.g transfers) to preserve parallelism in graphs.
  assigned = _internal_memory_planner([list(si.bufs) for si in schedule],
                                      noopt_buffers={b for si in schedule if si.ast.op is not Ops.SINK for b in si.bufs})
  return [ScheduleItem(si.ast, tuple(assigned.get(x, x) for x in si.bufs), si.metadata, si.fixedvars) for si in schedule]
