import time
from typing import cast
from collections import deque
from tinygrad.uop.ops import UOp, Ops, buffers, UOpMetaClass, track_rewrites, PatternMatcher, UPat, graph_rewrite, graph_rewrite_map, Kernel
from tinygrad.uop.spec import type_verify, tensor_spec
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.helpers import DEBUG, cpu_profile, TracingKey, SPEC, flatten, pluralize, SCACHE, Metadata
from tinygrad.engine.realize import ExecItem

# **** schedule linearizer

# ScheduleItem = tuple[AST, buffer UOps, metadata, fixedvars, bound_ranges]
ScheduleItem = tuple[UOp, tuple[UOp, ...], tuple[Metadata, ...], dict[str, int], tuple[UOp, ...]]

# unwrap VIEW/CAST/etc to find the actual data source (kernel output, buffer, or multi-device op)
def _unwrap_src(s: UOp) -> UOp:
  while len(s.src) and s.op not in {Ops.AFTER, Ops.BUFFER, Ops.MSELECT, Ops.MSTACK, Ops.BIND}: s = s.src[0]
  return s

def create_schedule(sched_sink:UOp) -> tuple[list[ExecItem], UOp]:
  with cpu_profile(TracingKey("toposort sched_sink")):
    # build kernel dependency graph: edges from producer kernel to consumer kernels
    children: dict[UOp, list[UOp]] = {}
    in_degree: dict[UOp, int] = {}
    for u in sched_sink.toposort():
      if u.op is Ops.RANGE:
        in_degree.setdefault(u, 0)
        continue
      if u.op is not Ops.AFTER or u.src[1].op is Ops.RANGE: continue
      k = u.src[1]
      in_degree.setdefault(k, 0)
      for s in k.src[0].src if k.op is Ops.END else k.src:
        match (s := _unwrap_src(s)).op:
          case Ops.AFTER:
            children.setdefault(s.src[1], []).append(k)
            in_degree[k] += 1
          case Ops.MSELECT | Ops.MSTACK:
            for ss in s.src:
              if ss.op is Ops.MSELECT: ss = ss.src[0]
              if ss.op is not Ops.BUFFER:
                assert ss.op is Ops.AFTER, f"ss.op is not AFTER, it's {ss.op}"
                children.setdefault(ss.src[1], []).append(k)
                in_degree[k] += 1
          case Ops.BUFFER | Ops.BIND:
            pass  # BUFFER is already realized, BIND is outer range (handled via bound_ranges below)
          case _:
            raise RuntimeError(f"input to kernel must be AFTER, BUFFER, MSELECT, MSTACK, or BIND, not {s.op}")

  with cpu_profile(TracingKey("linearize schedule")):
    queue: deque[UOp] = deque()
    for k,v in in_degree.items():
      if v == 0: queue.append(k)

    schedule: list[ScheduleItem|UOp] = []  # ScheduleItem for kernels, UOp for RANGE/END
    while len(queue):
      k = rk = queue.popleft()
      if k.op is Ops.END: k = k.src[0]
      assert k.op in {Ops.RANGE, Ops.KERNEL}, f"unexpected op in queue: {k.op}"
      if k.op is Ops.RANGE: schedule.append(k)
      elif k.op is Ops.KERNEL:
        ast = (kernel:=cast(Kernel, k.arg)).ast
        buf_uops = tuple(_unwrap_src(s).buf_uop for s in k.src if s.op is not Ops.BIND)
        bound_ranges = tuple(s for s in k.src if s.op is Ops.BIND and len(s.src) > 1 and s.src[1].op is Ops.RANGE)
        schedule.append((ast, buf_uops, kernel.metadata, {}, bound_ranges))
        if rk.op is Ops.END: schedule.append(rk)
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0: queue.append(x)

  with cpu_profile(TracingKey("unroll outer ranges")):
    pre_schedule, buf_uops_list = unroll_outer_ranges(schedule)
  return pre_schedule, UOp.sink(*buf_uops_list)

def unroll_outer_ranges(schedule:list[ScheduleItem|UOp]) -> tuple[list[ExecItem], list[UOp]]:
  pre_schedule: list[ExecItem] = []
  buf_uops_list: list[UOp] = []
  sched_ptr, in_ranges, range_ptrs = 0, dict[UOp, int](), dict[UOp, int]()
  while sched_ptr < len(schedule):
    if isinstance(si := schedule[sched_ptr], UOp):
      if si.op is Ops.RANGE:
        in_ranges[si] = 0
        range_ptrs[si] = sched_ptr + 1
      elif si.op is Ops.END:
        if in_ranges[si.src[1]] < si.src[1].vmax:
          in_ranges[si.src[1]] += 1
          sched_ptr = range_ptrs[si.src[1]]
          continue
    else:
      ast, buf_uops, metadata, _, bound_ranges = si
      fixedvars = {s.src[0].arg[0]:in_ranges[s.src[1]] for s in bound_ranges}
      pre_schedule.append(ExecItem(ast, [], metadata, fixedvars))
      buf_uops_list.append(UOp.sink(*buf_uops))
    sched_ptr += 1
  return pre_schedule, buf_uops_list

from tinygrad.engine.memory import memory_planner
from tinygrad.schedule.rangeify import get_rangeify_map
from tinygrad.schedule.multi import get_multi_map

def replace_input_buffer(ctx:tuple[dict[UOp, UOp], dict[str, int]], b:UOp):
  if (ret:=ctx[0].get(b, None)) is None:
    # both BUFFER and CONST have src=(UNIQUE, DEVICE), replace UNIQUE with LUNIQUE
    ctx[0][b] = ret = b.replace(src=(UOp(Ops.LUNIQUE, arg=len(ctx[0])), b.src[1]))
  return ret

def strip_bind(ctx:tuple[dict[UOp, UOp], dict[str, int]], b:UOp):
  var, val = b.src[0], b.src[1].arg
  assert var.expr not in ctx[1] or ctx[1][var.expr] == val, f"bind mismatch on {var}, {ctx[1][var.expr]} != {val}"
  ctx[1][var.expr] = val
  return ctx[0].setdefault(b, b.replace(src=(b.src[0],)))

pm_pre_sched_cache = PatternMatcher([
  # replace UNIQUE with LUNIQUE for cache key normalization
  (UPat((Ops.BUFFER, Ops.CONST), src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE)), name="b"), replace_input_buffer),
  # strip value from BIND for cache key normalization, so different values hit same cache
  (UPat(Ops.BIND, src=(UPat(Ops.DEFINE_VAR), UPat(Ops.CONST)), name="b"), strip_bind),
])

def replace_input_buffer_back(ctx:dict[UOp, UOp], b:UOp):
  if (ret:=ctx.get(b, None)) is None:
    assert b.op is Ops.BUFFER
    # if it's not in the cache, create a new buffer
    ctx[b] = ret = UOp.new_buffer(b.device, b.arg, b.dtype)
  return ret

pm_post_sched_cache = PatternMatcher([
  # restore LUNIQUE back to UNIQUE
  (UPat((Ops.BUFFER, Ops.CONST), src=(UPat(Ops.LUNIQUE), UPat(Ops.DEVICE)), name="b"), replace_input_buffer_back),
  # restore BIND value stripped in pm_pre_sched_cache
  (UPat(Ops.BIND, src=(UPat(Ops.DEFINE_VAR),), name="b"), lambda ctx,b: ctx.get(b)),
])

schedule_cache: dict[bytes, tuple[list[ExecItem], UOp]] = {}
@track_rewrites(lambda _,ret: f"Schedule {pluralize('Kernel', len(ret[1]))}")
def complete_create_schedule_with_vars(big_sink:UOp) -> tuple[dict[UOp, UOp], list[ExecItem], dict[str, int]]:
  # big_sink srcs are all the Tensors
  st = time.perf_counter()

  # replace all UNIQUE buffers with LUNIQUE, strip BIND values for cache key, extract var_vals
  input_buffers: dict[UOp, UOp] = {}
  var_vals: dict[str, int] = {}
  big_sink_cache = graph_rewrite(big_sink, pm_pre_sched_cache, ctx=(input_buffers, var_vals), name="rewrite for sched cache")
  sched_cache_key = big_sink_cache.key

  if not SCACHE or (sc_ret:=schedule_cache.get(sched_cache_key, None)) is None:
    # verify Tensors match the spec (on big_sink, we only need to do this if cache misses)
    if SPEC: type_verify(big_sink, tensor_spec)

    # hack to preserve metadata
    graph_rewrite_map(big_sink, pm_pre_sched_cache, ctx=({}, {}), name="preserve metadata")

    # tensor map is what we return
    tensor_map: dict[UOp, UOp] = {}

    if any(isinstance(x._device, tuple) for x in big_sink_cache.toposort()):
      tensor_map |= get_multi_map(big_sink_cache)
      big_sink_cache = big_sink_cache.substitute(tensor_map, name="Apply Multi Map")
      big_sink_cache = UOp.sink(*flatten([x.src if x.op is Ops.MULTI else [x] for x in big_sink_cache.src]))

    tensor_map |= get_rangeify_map(big_sink_cache)
    big_sink = big_sink_cache.substitute(tensor_map, name="Apply Kernelize Map")

    pre_schedule, buf_uops_sink = create_schedule(big_sink)

    # save in schedule cache (include AFTERs in tensor_map so we don't need big_sink)
    after_map = [(u, u.buf_uop) for u in big_sink.toposort() if u.op is Ops.AFTER]
    tensor_map_sink = UOp.sink(*flatten([(k,v) for k,v in tensor_map.items()]), *flatten(after_map))
    combined_sink = UOp.sink(tensor_map_sink, buf_uops_sink)
    if SCACHE: schedule_cache[sched_cache_key] = (pre_schedule, combined_sink)
  else:
    # schedule cache hit
    del big_sink_cache
    pre_schedule, combined_sink = sc_ret

  # replace all the LUNIQUEs with UNIQUEs (single graph_rewrite for everything)
  input_buffers_inverse = {v:k for k,v in input_buffers.items()}
  combined = graph_rewrite(combined_sink, pm_post_sched_cache, ctx=input_buffers_inverse, name="unrewrite combined")
  tensor_map_sink, buf_uops_sink = combined.src
  tm_src = tensor_map_sink.src
  tensor_map = {tm_src[i]:tm_src[i+1] for i in range(0, len(tm_src), 2)}

  # add bufs to pre_schedule
  schedule: list[ExecItem] = []
  for i, si in enumerate(pre_schedule):
    buf_uops = buf_uops_sink.src[i].src
    # create subbuffers if needed
    if si.ast.op is Ops.BUFFER_VIEW:
      base = buf_uops[1].buffer
      assert isinstance(base, Buffer), "base can't be MultiBuffer"
      buffers[buf_uops[0]] = base.view(buf_uops[0].arg, si.ast.dtype, si.ast.arg[1]*base.dtype.itemsize)
    ubufs = tuple(b.buffer for b in buf_uops)
    if any(isinstance(x, MultiBuffer) for x in ubufs):
      assert all(isinstance(x, MultiBuffer) for x in ubufs), "kernel must all be multibuffer"
      dnums = [x for x in si.ast.variables() if x.arg[0] == '_device_num']
      for j, bufs in enumerate(zip(*[x.bufs for x in cast(tuple[MultiBuffer, ...], ubufs)])):
        schedule.append(ExecItem(si.ast, list(bufs), si.metadata, si.fixedvars | ({dnums[0].expr:j} if len(dnums) else {})))
    else:
      # ONE -> ONE
      schedule.append(ExecItem(si.ast, list(ubufs), si.metadata, si.fixedvars))
  with cpu_profile(TracingKey("memory planner")): schedule = memory_planner(schedule)

  if (DEBUG >= 1 and len(schedule) > 1) or DEBUG >= 3:
    print(f"scheduled {len(schedule):4d} kernels in {(time.perf_counter()-st)*1000:8.2f} ms"+\
          f" | {' cache hit' if SCACHE and sc_ret is not None else 'CACHE MISS'} {sched_cache_key.hex()[:8]}"+\
          f" | {len(UOpMetaClass.ucache)} uops in cache")

  used_vars = set().union(*[{v.arg[0] for v in si.ast.variables()} for si in schedule])
  return tensor_map, schedule, {k:v for k,v in var_vals.items() if k in used_vars}
