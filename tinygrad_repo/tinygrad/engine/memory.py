from typing import List, Union, Tuple, Dict
from collections import defaultdict
from tinygrad.engine.schedule import ScheduleItem
from tinygrad.device import Device, Buffer
from tinygrad.helpers import NO_MEMORY_PLANNER, dedup, DEBUG
from tinygrad.ops import Ops

# **************** memory planning ****************

def _internal_memory_planner(buffers:List[Union[List[Buffer], Tuple[Buffer, ...]]], noopt_buffers=None, debug_prefix="") -> Dict[Buffer, Buffer]:
  if NO_MEMORY_PLANNER: return {}
  first_appearance, last_appearance = {}, {}
  for i,u in enumerate(buffers):
    for buf in u:
      if buf.is_allocated() or buf.lb_refcount > 0 or (noopt_buffers is not None and buf.base in noopt_buffers): continue
      if buf.base not in first_appearance: first_appearance[buf.base] = i
      last_appearance[buf.base] = i

  # Sort buffers by size in descending order, prioritizing largest buffers for allocation first.
  # Track free segments, each containing (start, stop, and buffer that could be reused on this segment).
  free_segs: Dict[Tuple, List[Tuple[int, int, Buffer]]] = defaultdict(list) # Dict[buffer key, Tuple[start, end, buffer to reuse on the seg]]
  def find_replace_buffer(buf, st, en):
    key = (buf.device, buf.dtype, buf.options) + ((buf.nbytes,) if not hasattr(Device[buf.device].allocator, "offset") else tuple())

    default_buf = (0, len(buffers) - 1, buf) # will return the buffer itself if the replace one is not found.
    seg_st, seg_en, seg_buf = next((free_segs[key].pop(i) for i,(sst,sen,_) in enumerate(free_segs[key]) if sst <= st and en <= sen), default_buf)

    free_segs[key] += [(seg_st, st - 1, seg_buf)] if st - 1 >= seg_st else []
    free_segs[key] += [(en + 1, seg_en, seg_buf)] if seg_en >= en + 1 else []

    return seg_buf if seg_buf.nbytes == buf.nbytes else Buffer(buf.device, buf.size, buf.dtype, base=seg_buf)

  buffer_requests = sorted([(first_appearance[buf], last_appearance[buf], buf) for buf in first_appearance.keys()], key=lambda x: -x[2].nbytes)
  assigned = {buf:find_replace_buffer(buf, st, en) for st, en, buf in buffer_requests}

  for i,u in enumerate(buffers):
    for buf in u:
      if buf.is_allocated() or buf.lb_refcount > 0 or (noopt_buffers is not None and buf.base in noopt_buffers): continue
      if buf._base is not None: assigned[buf] = Buffer(buf.device, buf.size, buf.dtype, base=assigned.get(buf.base, buf.base).base, offset=buf.offset)
      else: assigned[buf] = assigned.get(buf, buf)

  if DEBUG >= 1 and len(ak:=dedup(x for x in assigned.keys() if x._base is None)) != len(av:=dedup(x for x in assigned.values() if x._base is None)):
    print(debug_prefix+f"memory reduced from {sum([x.nbytes for x in ak])/1e6:.2f} MB -> {sum([x.nbytes for x in av])/1e6:.2f} MB,",
          f"{len(ak)} -> {len(av)} bufs")
  return assigned

def memory_planner(schedule:List[ScheduleItem]) -> List[ScheduleItem]:
  # Exclude buffers involved in load ops (e.g transfers) to preserve parallelism in graphs.
  assigned = _internal_memory_planner([si.bufs for si in schedule],
                                      noopt_buffers={b for si in schedule if si.ast.op is not Ops.SINK for b in si.bufs})
  return [ScheduleItem(si.ast, tuple(assigned.get(x, x) for x in si.bufs), si.metadata, si.assign_preloads) for si in schedule]
