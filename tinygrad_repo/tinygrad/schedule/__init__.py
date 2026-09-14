import time, inspect
from collections import deque
from tinygrad.uop.ops import UOp, Ops, UOpMetaClass, rewrite_group, graph_rewrite, gate_kernel_sink, KernelInfo
from tinygrad.uop.spec import type_verify, spec_tensor
from tinygrad.helpers import DEBUG, cpu_profile, TracingKey, SPEC, pluralize, SCACHE, BASEDIR, partition, dedup

# **** schedule linearizer

# unwrap VIEW/CAST/etc to find the actual data source (kernel output, buffer, or multi-device op)
def _unwrap_src(s: UOp) -> UOp:
  while len(s.src) and s.op not in {Ops.AFTER, Ops.BUFFER, Ops.PARAM, Ops.MSELECT, Ops.MSTACK}: s = s.src[0]
  return s

# a buffer state is AFTER | BUFFER | PARAM. MSELECT/MSTACK join per-device states
def _states(s: UOp) -> list[UOp]:
  s = _unwrap_src(s)
  if s.op in {Ops.MSELECT, Ops.MSTACK}: return [st for ss in s.src for st in _states(ss)]
  assert s.op in {Ops.AFTER, Ops.BUFFER, Ops.PARAM}, f"input to kernel must resolve to a buffer state, not {s.op}"
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
    writes: dict[UOp, list[tuple[UOp, UOp, tuple[UOp, ...]]]] = {}  # buffer -> (AFTER, prior state, new kernels)
    reads: list[tuple[UOp, UOp, UOp]] = []  # (reader AFTER, reader kernel, buffer state read)
    for u in sched_sink.toposort(gate_kernel_sink):
      if u.op is not Ops.AFTER: continue
      kernels, after_deps = _split_after(u)
      prev_state = _unwrap_src(u.src[0])
      prev_kernels = set(_split_after(prev_state)[0]) if prev_state.op is Ops.AFTER else set()
      writes.setdefault(u.buf_uop, []).append((u, prev_state, tuple(k for k in kernels if k not in prev_kernels)))
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
      for a, prev_state, write_kernels in writes.get(s.buf_uop, []):
        if a is u or prev_state is not s: continue
        for t in write_kernels:
          if t is not k and t not in k.backward_slice:
            children.setdefault(k, []).append(t)
            in_degree[t] += 1

  with cpu_profile(TracingKey("linearize schedule")):
    queue: deque[UOp] = deque(k for k,v in in_degree.items() if v == 0)
    linearized: list[UOp] = []
    while len(queue):
      rk = queue.popleft()
      if rk.op is Ops.LINEAR:
        linearized.extend(rk.src)
      else:
        k = rk.src[0] if rk.op is Ops.END else rk
        assert k.op is Ops.CALL, f"unexpected op in queue: {k.op}"
        buf_uops = tuple(_unwrap_src(s).buf_uop for s in k.src[1:] if not s.is_bound_var)
        linearized.append(k.src[0].call(*buf_uops))
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0: queue.append(x)
    if any(in_degree.values()): raise RuntimeError("cycle detected in assign graph")
  return UOp(Ops.LINEAR, src=tuple(linearized))

from tinygrad.schedule.memory import memory_plan_rewrite
from tinygrad.engine.realize import capturing, pm_flatten_linear
from tinygrad.schedule.prepare import prepare_rangeify
from tinygrad.schedule.rangeify import get_kernel_graph
from tinygrad.helpers import CAPTURING
from tinygrad.uop.ops import PatternMatcher, UPat, ParamArg
from tinygrad.dtype import AddrSpace

def create_new_buffer(ctx:tuple[dict[UOp, UOp], tuple[UOp, ...]], b:UOp):
  if (ret:=ctx[0].get(b, None)) is None: ctx[0][b] = ret = UOp.new_buffer(b.device, b.max_numel(), b.dtype)
  return ret

pm_post_sched_cache = PatternMatcher([
  # only resolve buffer PARAMs (slot>=0); ALU/shape vars use slot=-1 and must not be swapped for call args
  (UPat(Ops.PARAM, name="x"), lambda ctx,x: ctx[1][x.arg.slot] if x.arg.slot >= 0 else None),
  # create new BUFFERs
  (UPat(Ops.BUFFER, src=(), name="b"), lambda ctx,b:
   create_new_buffer(ctx, b) if isinstance(b.arg, ParamArg) and b.addrspace is AddrSpace.GLOBAL else None),
])

def resolve_linear_call(linear_call:UOp, outer_binds:dict[str, UOp]|None=None):
  linear = graph_rewrite(linear_call.src[0], pm_post_sched_cache, ctx=({}, linear_call.src[1:]), walk=True, name="params to buffers")
  # nested LINEAR calls are lexical scopes: their positional params shadow the enclosing scope, while calls without
  # scalar args (e.g. precompiled allreduce) inherit it
  binds = {**(outer_binds or {}),
           **{f"p{i}":x.src[0].replace(op=Ops.PARAM) for i,x in enumerate(linear_call.src[1:]) if x.is_bound_var}}
  def apply_binds(si:UOp) -> UOp:
    if si.op is Ops.CALL and si.src[0].op is Ops.LINEAR: return resolve_linear_call(si, binds)
    subs = {v:binds[v.expr] for v in si.variables() if v.expr in binds}
    return si.replace(src=tuple(s.substitute(subs, name="resolve scalar params") for s in si.src))
  return linear.replace(src=tuple(apply_binds(si) for si in linear.src))

pm_resolve_linear_call = PatternMatcher([
  # call LINEAR is resolved here
  (UPat(Ops.CALL, src=(UPat(Ops.LINEAR),), name="linear_call", allow_any_len=True), resolve_linear_call),
])+pm_flatten_linear

schedule_cache: dict[bytes, UOp] = {}
# ctx is just for DEBUG on inner
def lower_sink_to_linear(function:UOp) -> UOp|None:
  st = time.perf_counter()
  if isinstance(function.arg, KernelInfo): return None
  cache_key = function.key
  if not SCACHE or (sc_ret:=schedule_cache.get(cache_key, None)) is None:
    if SPEC: type_verify(function, spec_tensor)
    # support recursive CALLs
    linear = create_schedule(get_kernel_graph(prepare_rangeify(function)))
    if SCACHE: schedule_cache[cache_key] = linear
  else:
    # schedule cache hit
    linear = sc_ret
  if (DEBUG >= 1 and len(linear.src) > 1) or DEBUG >= 3:
    for frm in inspect.stack():
      if frm.filename == "<string>": continue
      if frm.filename.startswith(str(BASEDIR / "apps")): break
      if not frm.filename.startswith(str(BASEDIR)) and not frm.filename.endswith("/contextlib.py"): break
    else:
      frm = None
    print(f"scheduled {len(linear.src):5d} kernels in {(time.perf_counter()-st)*1000:8.2f} ms"+\
          f" | {' cache hit' if SCACHE and sc_ret is not None else 'CACHE MISS'} {cache_key.hex()[:8]}"+\
          f" | {len(UOpMetaClass.ucache):7d} uops in cache"+("" if frm is None else f" | {frm.filename}:{frm.lineno}"))
  return linear

pm_schedule = PatternMatcher([
  (UPat(Ops.SINK, name="function"), lower_sink_to_linear),
])

def assert_all_same_devices(ast:UOp):
  devices = dedup([x.device for x in ast.toposort() if x.op is Ops.PARAM and x.device is not None])
  if len(devices) >= 2: raise RuntimeError(f"all buffers must be on the same device: {devices}")

def copy_kernel_to_copy_uop(call:UOp, dst:UOp, src:UOp, r:UOp|None=None):
  if dst.device == src.device and not (isinstance(dst.device, str) and dst.device.startswith("DISK")): return None
  return call.replace(src=(UOp(Ops.COPY, src=(src,), arg=dst.device),) + call.src[1:])

def simplify_copy_kernel(call:UOp, ast:UOp, dst:UOp, src:UOp):
  # NOTE: this is a codegen for SDMA devices
  if dst.device == src.device and not (isinstance(dst.device, str) and dst.device.startswith("DISK")): return None
  from tinygrad.codegen.simplify import pm_flatten_range, pm_simplify_ranges
  from tinygrad.schedule.prepare import pm_mops
  from tinygrad.uop.symbolic import sym
  sink = graph_rewrite(ast, sym+pm_mops+pm_flatten_range+pm_simplify_ranges, ctx={}, name="simplify ranges in copy")
  return call.replace(src=(sink,) + call.src[1:])

pm_copy_from_store = PatternMatcher([
  # simplify copy kernels
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="ast"), UPat.var("dst"), UPat.var("src")), name="call"), simplify_copy_kernel),

  # replace this with a copy if it's a copy
  (UPat(Ops.CALL, src=(UPat(Ops.PARAM, name="dst").index(UPat(Ops.CONST, arg=0))
                .store(UPat(Ops.PARAM, name="src").index(UPat(Ops.CONST, arg=0))).sink(),),
                name="call", allow_any_len=True), copy_kernel_to_copy_uop),
  (UPat(Ops.CALL, src=(UPat(Ops.PARAM, name="dst").index(UPat(Ops.RANGE, name="r"))
                .store(UPat(Ops.PARAM, name="src").index(UPat(Ops.RANGE, name="r"))).end(UPat(Ops.RANGE, name="r")).sink(),),
                name="call", allow_any_len=True), copy_kernel_to_copy_uop),

  # if it wasn't copy, it currently can't be cross device
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="ast"),), allow_any_len=True), assert_all_same_devices),
])

@rewrite_group(lambda _,ret: f"Schedule {pluralize('Kernel', len(ret[0].src))}")
def create_linear_with_vars(big_sink:UOp) -> tuple[UOp, dict[str, int]]:
  # big_sink srcs are all the Tensors
  linear_call = graph_rewrite(big_sink, pm_schedule, name="schedule to linear", enter_calls=True)

  # this recursively resolves the linear_call and allocates buffers
  linear = graph_rewrite(linear_call, pm_resolve_linear_call, name="resolve linear call")

  # create copies
  linear = graph_rewrite(linear, pm_copy_from_store, name="create COPY kernels for SDMA")

  # vars used in the schedule
  used_vars = set().union(*[{v.expr for v in si.src[0].variables()} for si in linear.src])
  # get var_vals from the bound Variables in the call args
  var_vals: dict[str, int] = {}
  for b in big_sink.src[1:]:
    if b.is_bound_var:
      v, val = b.unbind()
      nm = v.expr
      if nm not in used_vars: continue
      if var_vals.get(nm, val) != val: raise RuntimeError(f"bind mismatch on {nm}, {var_vals[nm]} != {val}")
      var_vals[nm] = val

  # jit captures this schedule, no need to execute.
  if len(capturing) and CAPTURING:
    capturing[0].add_linear(linear, var_vals)
    return UOp(Ops.LINEAR, src=()), var_vals

  held_bufs = ({b for b in linear_call.src[1:] if b.op is Ops.BUFFER} if linear_call.op is Ops.CALL else set())
  return memory_plan_rewrite(linear, held_bufs), var_vals
