import time, inspect
from collections import deque
from dataclasses import dataclass, field, replace
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import GroupOp, remove_all_tags, UOp, Ops, UOpMetaClass, graph_rewrite, gate_kernel_sink, KernelInfo
from tinygrad.uop.spec import type_verify, spec_tensor
from tinygrad.helpers import DEBUG, cpu_profile, TracingKey, SPEC, SCACHE, BASEDIR, partition, dedup, all_int, VIZ
from tinygrad.helpers import diskcache_get, diskcache_put, colored

# **** schedule linearizer

# unwrap VIEW/CAST/etc to find the actual data source (kernel output, buffer, or multi-device op)
def _unwrap_src(s: UOp) -> UOp:
  while len(s.src) and s.op not in {Ops.AFTER, Ops.BUFFER, Ops.ALLOC, Ops.PARAM, Ops.MSELECT, Ops.MSTACK}: s = s.src[0]
  return s

# a buffer state is AFTER | BUFFER | ALLOC | PARAM. MSELECT/MSTACK join per-device states
def _states(s: UOp) -> list[UOp]:
  s = _unwrap_src(s)
  if s.op in {Ops.MSELECT, Ops.MSTACK}: return [st for ss in s.src for st in _states(ss)]
  assert s.op in {Ops.AFTER, Ops.BUFFER, Ops.ALLOC, Ops.PARAM}, f"input to kernel must resolve to a buffer state, not {s.op}"
  return [s]

def _split_after(after: UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]:
  kernels, remaining = partition(after.src[1:], lambda s: s.op in {Ops.CALL, Ops.END})
  deps, remaining = partition(remaining, lambda s: s.op is Ops.AFTER)
  if invalid := [s for s in remaining if s.op is not Ops.STORE]:
    raise AssertionError(f"AFTER source should be CALL, END, STORE, or AFTER, not {invalid[0].op}")
  return tuple(kernels), tuple(deps)

def create_schedule(sched_sink:UOp) -> UOp:
  with cpu_profile(TracingKey("toposort sched_sink")):
    # build kernel dependency graph: edges from producer kernel to consumer kernels
    children: dict[UOp, list[UOp]] = {}
    in_degree: dict[UOp, int] = {}
    writes: dict[UOp, list[tuple[UOp, tuple[UOp, ...]]]] = {}  # superseded state -> (AFTER, new kernels)
    reads: list[tuple[UOp, UOp, UOp]] = []  # (reader AFTER, reader kernel, buffer state read)
    for u in sched_sink.toposort(gate_kernel_sink):
      if u.op is not Ops.AFTER: continue
      kernels, after_deps = _split_after(u)
      prev_state = _unwrap_src(u.src[0])
      prev_kernels = set(_split_after(prev_state)[0]) if prev_state.op is Ops.AFTER else set()
      writes.setdefault(prev_state, []).append((u, tuple(k for k in kernels if k not in prev_kernels)))
      for k in kernels:
        in_degree.setdefault(k, 0)
        if k.op is Ops.END: assert k.src[0].op is Ops.CALL, f"END src[0] should be KERNEL, not {k.src[0].op}"
        kernel_deps = k.src[0].src[1:] if k.op is Ops.END else k.src[1:]
        read_states = [st for s in kernel_deps for st in _states(s)]
        reads += [(u, k, st) for st in read_states]
        # RAW deps: a kernel runs after the kernels that produced the states it reads or joins
        for st in read_states + [st for s in after_deps for st in _states(s)]:
          if st.op is Ops.AFTER:
            for t in _split_after(st)[0]:
              children.setdefault(t, []).append(k)
              in_degree[k] += 1
    # WAR deps: a kernel reading buffer state S must run before another write that supersedes S. an AFTER only
    # supersedes its immediate prior state; join members already present in that prior state are ordering deps, not writes
    for u, k, s in reads:
      for a, write_kernels in writes.get(s, []):
        if a is u: continue
        for t in write_kernels:
          if t is not k and t not in k.backward_slice:
            children.setdefault(k, []).append(t)
            in_degree[t] += 1

  with cpu_profile(TracingKey("linearize schedule")):
    queue: deque[UOp] = deque(k for k,v in in_degree.items() if v == 0)
    linearized: list[UOp] = []
    while len(queue):
      rk = queue.popleft()
      k = rk.src[0] if rk.op is Ops.END else rk
      assert k.op is Ops.CALL, f"unexpected op in queue: {k.op}"
      buf_uops = tuple(_unwrap_src(s).buf_uop for s in k.src[1:] if not s.is_bound_var)
      linearized.append(k.replace(src=(k.body, *buf_uops)))
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0: queue.append(x)
    if any(in_degree.values()): raise RuntimeError("cycle detected in assign graph")
  return UOp(Ops.LINEAR, src=tuple(linearized))

from tinygrad.schedule.memory import memory_plan_rewrite
from tinygrad.engine.realize import capturing, pm_flatten_linear
from tinygrad.schedule.prepare import prepare_rangeify
from tinygrad.schedule.multi import multi_pm
from tinygrad.schedule.rangeify import get_kernel_graph
from tinygrad.helpers import CAPTURING
from tinygrad.uop.ops import PatternMatcher, UPat

def create_new_buffer(ctx:tuple[dict[UOp, UOp], tuple[UOp, ...]], b:UOp):
  if (ret:=ctx[0].get(b, None)) is None:
    device = b.device if b.device is not None else next(a.device for a in ctx[1] if a.device is not None)
    ctx[0][b] = ret = UOp.new_buffer(device, b.max_numel(), b.dtype)
  return ret

pm_post_sched_cache = PatternMatcher([
  # Resolve positional arguments outside kernel bodies; free Variables have slot -1.
  (UPat(Ops.PARAM, name="x"), lambda ctx,x: ctx[1][x.arg.slot] if x.arg.slot >= 0 else None),
  # bind ALLOCs to fresh BUFFERs for this invocation
  (UPat(Ops.ALLOC, name="b"), create_new_buffer),
])

def resolve_linear_call(linear_call:UOp, outer_binds:dict[int, UOp]|None=None):
  linear = graph_rewrite(linear_call.body, pm_post_sched_cache, ctx=({}, linear_call.src[1:]), walk=True, name="params to buffers")
  # nested LINEAR calls are lexical scopes: their positional params shadow the enclosing scope, while calls without
  # scalar args (e.g. precompiled allreduce) inherit it
  binds = {**(outer_binds or {}),
           **{i:x.unbound() if x.is_variable else x for i,x in enumerate(linear_call.src[1:])
              if x.op is Ops.PARAM and x.addrspace is AddrSpace.ALU}}
  def apply_binds(si:UOp) -> UOp:
    if si.op is Ops.CALL and si.body.op is Ops.LINEAR: return resolve_linear_call(si, binds)
    if si.op is Ops.CALL and si.body.op is Ops.PROGRAM: return si  # compiled parameters already have ABI slots
    subs = {v:binds[v.arg.slot] for s in si.src for v in s.variables() if v.arg.slot in binds}
    return si.replace(src=tuple(s.substitute(subs, name="resolve scalar params") for s in si.src))
  return linear.replace(src=tuple(apply_binds(si) for si in linear.src))

pm_resolve_linear_call = PatternMatcher([
  # call LINEAR is resolved here
  (UPat(Ops.CALL, src=(UPat(Ops.LINEAR),), name="linear_call", allow_any_len=True), resolve_linear_call),
])+pm_flatten_linear

schedule_cache: dict[bytes, UOp] = {}
# ctx is just for DEBUG on inner
def lower_sink_to_linear(call:UOp) -> UOp|None:
  function = call.body
  if function.op is not Ops.SINK or isinstance(function.arg, KernelInfo) or not call.arg.precompile: return None
  st = time.perf_counter()
  cache_key = function.key
  # SCACHE >= 2 also persists the cache to disk
  sc_ret, disk_hit = schedule_cache.get(cache_key, None) if SCACHE else None, False
  if sc_ret is None and SCACHE >= 2: disk_hit = (sc_ret:=diskcache_get("schedule_cache", {"key": cache_key})) is not None
  if sc_ret is None:
    if SPEC: type_verify(function, spec_tensor)
    # support recursive CALLs
    linear = create_schedule(get_kernel_graph(prepare_rangeify(function)))
    if SCACHE: schedule_cache[cache_key] = linear
    if SCACHE >= 2: diskcache_put("schedule_cache", {"key": cache_key}, linear)
  else:
    # schedule cache hit (memory or disk)
    linear = schedule_cache[cache_key] = sc_ret
  if (DEBUG >= 1 and len(linear.src) > 1) or DEBUG >= 3:
    for frm in inspect.stack():
      if frm.filename == "<string>": continue
      if not frm.filename.startswith(str(BASEDIR)) and not frm.filename.endswith("/contextlib.py"): break
    else:
      frm = None
    print(f"scheduled {len(linear.src):5d} kernels in {(time.perf_counter()-st)*1000:8.2f} ms"+\
          f" | {colored(' cache hit', 'yellow') if disk_hit else (' cache hit' if sc_ret is not None else 'CACHE MISS')} {cache_key.hex()[:8]}"+\
          f" | {len(UOpMetaClass.ucache):7d} uops in cache"+("" if frm is None else f" | {frm.filename}:{frm.lineno}"))
  return call.replace(src=(linear,)+call.src[1:])

pm_schedule = PatternMatcher([
  (UPat(Ops.CALL, name="call"), lower_sink_to_linear),
])

def assert_all_same_devices(ast:UOp):
  devices = dedup([x.device for x in ast.toposort() if x.op is Ops.PARAM and x.device is not None])
  if len(devices) >= 2: raise RuntimeError(f"all buffers must be on the same device: {devices}")

def copy_kernel_to_store(call:UOp, dst:UOp, src:UOp, r:UOp|None=None):
  if dst.device == src.device and not dst.on_disk(): return None
  return call.replace(src=(dst.store(src),) + call.src[1:])

def simplify_copy_kernel(call:UOp, ast:UOp, dst:UOp, src:UOp):
  # NOTE: this is a codegen for SDMA devices
  if dst.device == src.device and not dst.on_disk(): return None
  from tinygrad.codegen.simplify import pm_flatten_range, pm_simplify_ranges
  from tinygrad.schedule.prepare import pm_mops
  from tinygrad.uop.symbolic import sym
  sink = graph_rewrite(ast, sym+pm_mops+pm_flatten_range+pm_simplify_ranges, ctx={}, name="simplify ranges in copy")
  return call.replace(src=(sink,) + call.src[1:])

pm_copy_from_store = PatternMatcher([
  # simplify copy kernels
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="ast"), UPat.var("dst"), UPat.var("src")), name="call"), simplify_copy_kernel),

  # lower copy kernels to bulk STOREs
  (UPat(Ops.CALL, src=(UPat(Ops.PARAM, name="dst").index(UPat(Ops.CONST, arg=0))
                .store(UPat(Ops.PARAM, name="src").index(UPat(Ops.CONST, arg=0))).sink(),),
                name="call", allow_any_len=True), copy_kernel_to_store),
  (UPat(Ops.CALL, src=(UPat(Ops.PARAM, name="dst").index(UPat(Ops.RANGE, name="r"))
                .store(UPat(Ops.PARAM, name="src").index(UPat(Ops.RANGE, name="r"))).end(UPat(Ops.RANGE, name="r")).sink(),),
                name="call", allow_any_len=True), copy_kernel_to_store),

  # if it wasn't copy, it currently can't be cross device
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="ast"),), allow_any_len=True), assert_all_same_devices),
])

# **** callify: transform a tensor graph into a CALL UOp such that all state is properly scoped

@dataclass
class CallifyCtx:
  replacements: list[UOp] = field(default_factory=list)
  allocs: dict[UOp, UOp] = field(default_factory=dict)
  views: set[UOp] = field(default_factory=set)
  stores: list[UOp] = field(default_factory=list)

def contiguous_mops_to_view(ctx:CallifyCtx|None, c:UOp, src:UOp):
  """MOPS(BUFFER) → SHRINK when movement ops collapse to a contiguous range."""
  if not all_int(c.shape): return None
  buf = src.base
  while buf.op is Ops.BITCAST: buf = buf.src[0].base
  if buf.op is Ops.UNSHARD:
    if isinstance(c.device, str): return None
    if (unshard := graph_rewrite(src, multi_pm, name="multi_buffer_view")).op is not Ops.UNSHARD: return None
    view = contiguous_mops_to_view(ctx, unshard.src[0], unshard.src[0])
    return None if view is None else view.unshard(unshard.arg, unshard.src[1:])

  if buf.op is not Ops.BUFFER or (cv := src.contiguous_view()) is None or cv[0].op is not Ops.BUFFER: return None
  buf, offset = cv
  view = buf[offset:offset + src.max_numel() * src.element_size() // buf.element_size()].bitcast(src.dtype)
  if ctx is not None: ctx.views.add(view)
  view = view.reshape(c.shape)
  return c.replace(src=(view,)+c.src[1:]) if c.op in {Ops.COPY, Ops.STORE} else view

def is_store_after(u:UOp) -> bool:
  return u.op is Ops.AFTER and (u.src[0].unsharded_base.op is not Ops.ALLOC or u.src[1].op is Ops.STORE)

def collect_stores(ctx:CallifyCtx, u:UOp):
  if is_store_after(u): ctx.stores.append(u)

# NOTE: scheduling rewrites belong in prepare; only storage/interface normalization belongs here.
pm_callify_ctx_collect = PatternMatcher([
  # fold MOPS+BITCAST over BUFFER into SHRINK when movement ops collapse to contiguous range
  (UPat((Ops.COPY, Ops.STAGE), src=(UPat(GroupOp.Movement|{Ops.BITCAST}, name="src"),), allow_any_len=True, name="c"), contiguous_mops_to_view),
  (UPat(Ops.STORE, src=(UPat(Ops.BITCAST, name="src"), UPat()), name="c", allow_any_len=True), contiguous_mops_to_view),

  # Collect effects after their sources have been rewritten, without entering call bodies.
  (UPat(Ops.AFTER, name="u"), collect_stores),
])

# ALLOCs get canonical scope-local id slots here so structurally identical calls hash identically for the
# schedule cache (fresh slots are all positive from the global counter; negative slots are already canonical)
def canonicalize_alloc(ctx:CallifyCtx, b:UOp):
  if b.arg.slot >= 0 and b not in ctx.allocs: ctx.allocs[b] = b.replace(arg=replace(b.arg, slot=-1-len(ctx.allocs)))
  return ctx.allocs.get(b)

def canonicalize_call_body(c:UOp):
  return c.replace(src=(graph_rewrite(c.body, pm_canonicalize_alloc, ctx=CallifyCtx(), bottom_up=True),)+c.src[1:])

pm_canonicalize_alloc = PatternMatcher([
  # NOTE: lambda for late binding, canonicalize_call_body references pm_canonicalize_alloc
  (UPat(Ops.CALL, name="c"), lambda c: canonicalize_call_body(c)),
  (UPat(Ops.ALLOC, name="b"), canonicalize_alloc),
])

def replace_input_buffer(ctx:CallifyCtx, b:UOp):
  ctx.replacements.append(b)
  return b.param_like(len(ctx.replacements)-1)

pm_replace_buf = PatternMatcher([
  # replace GLOBAL BUFFERs with PARAMs for cache key normalization (Variables are ALU PARAMs, they don't match this)
  (UPat(Ops.BUFFER, name="b"), lambda ctx,b: replace_input_buffer(ctx, b) if b.addrspace is AddrSpace.GLOBAL else None),
  # replace buffer views (SHRINK/BITCAST) with PARAM (only the views created by contiguous_mops_to_view)
  (UPat((Ops.SHRINK, Ops.BITCAST), name="b"), lambda ctx,b: replace_input_buffer(ctx, b) if b in ctx.views else None),
  # replace bound Variables with renamed value-stripped PARAMs for cache key normalization, so different values hit same cache
  (UPat(Ops.PARAM, name="b"), lambda ctx,b: replace_input_buffer(ctx, b) if b.is_bound_var else None),
])

def transform_to_call(big_sink:UOp) -> UOp:
  if VIZ: graph_rewrite(big_sink, PatternMatcher([]), name="View Graph")
  if SPEC: type_verify(big_sink, spec_tensor)

  # The tensor replacement map is collected before these rewrites change node identities.
  graph_rewrite(big_sink, pm_callify_ctx_collect, ctx=(ctx:=CallifyCtx()), name="early transform tensor graph")
  ret = graph_rewrite(UOp.sink(*ctx.stores), pm_canonicalize_alloc+pm_replace_buf+remove_all_tags, ctx=ctx, bottom_up=True, name="replace bufs")
  ret = ret.call(*ctx.replacements, precompile=True)
  if VIZ: graph_rewrite(ret, PatternMatcher([]), name="View Call")
  return ret

def create_linear_with_vars(big_sink:UOp) -> tuple[UOp, dict[str, int]]:
  big_sink = transform_to_call(big_sink)
  # big_sink srcs are all the Tensors
  linear_call = graph_rewrite(big_sink, pm_schedule, name="schedule to linear", enter_calls=True)

  # this recursively resolves the linear_call and allocates buffers
  linear = graph_rewrite(linear_call, pm_resolve_linear_call, name="resolve linear call")

  # create copies
  linear = graph_rewrite(linear, pm_copy_from_store, name="lower copy kernels to STORE calls")

  # vars used in the schedule
  used_vars = set().union(*[{v.expr for v in si.src[0].variables()} for si in linear.src])
  # get var_vals from the bound Variables in the call args
  var_vals: dict[str, int] = {}
  for b in big_sink.src[1:]:
    if b.is_bound_var:
      nm, val = b.expr, b.arg.val
      if nm not in used_vars: continue
      if var_vals.get(nm, val) != val: raise RuntimeError(f"bind mismatch on {nm}, {var_vals[nm]} != {val}")
      var_vals[nm] = val

  # jit captures this schedule, no need to execute.
  if len(capturing) and CAPTURING:
    capturing[0].add_linear(linear)
    return UOp(Ops.LINEAR, src=()), var_vals

  held_bufs = {b for b in linear_call.src[1:] if b.op is Ops.BUFFER}
  return memory_plan_rewrite(linear, held_bufs), var_vals
