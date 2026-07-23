from typing import TypeVar, Generic, Callable, Any
import functools, collections
from tinygrad.tensor import Tensor, all_tensors
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, BEAM, getenv, JIT, JIT_BATCH_SIZE, dedup, pluralize, VIZ, disable_gc
from tinygrad.device import Buffer, Compiled, Device, MultiBuffer
from tinygrad.dtype import DType
from tinygrad.uop.ops import UOp, PatternMatcher, Variable, sym_infer, Ops, buffers, track_rewrites, graph_rewrite
from tinygrad.renderer import Estimates
from tinygrad.engine.realize import capturing, compile_linear, link_linear, run_linear, graph_cache, estimate_uop, get_runtime
from tinygrad.engine.realize import unwrap_multi, resolve_params, get_call_arg_uops, get_call_outs_ins
from tinygrad.schedule.memory import memory_plan_rewrite, _collect_bufs
from tinygrad.nn.state import get_parameters
from tinygrad.uop.movement import mop_cleanup
from dataclasses import dataclass

def prune_linear(linear:UOp, needed:set[UOp]) -> tuple[UOp, UOp]:
  kept, onetime = [], []
  for si in linear.src:
    si_bufs = {b for src in si.src[1:] for b in _collect_bufs(src)}
    if not si_bufs.isdisjoint(needed):
      kept.append(si)
      needed |= si_bufs
    else: onetime.append(si)
  return linear.replace(src=tuple(kept)), linear.replace(src=tuple(onetime))

def create_graph_call(batch:list[UOp]) -> UOp:
  # all external inputs are PARAMs
  input_list = dedup(u for si in batch for b in si.src[1:] for u in b.toposort() if u.op is Ops.PARAM)
  cf = UOp(Ops.CUSTOM_FUNCTION, src=(UOp(Ops.LINEAR, src=tuple(batch)),), arg="graph")
  return cf.call(*input_list)

def graph_split_rewrite(linear:UOp, max_batch_size:int=0) -> UOp:
  new_src: list[UOp] = []
  current_batch: list[UOp] = []
  current_batch_devs: list[Compiled] = []

  def flush_batch():
    nonlocal current_batch, current_batch_devs, max_batch_size, new_src
    if len(current_batch) <= 1 and not getenv("GRAPH_ONE_KERNEL"): new_src.extend(current_batch)
    else:
      new_src.append(create_graph_call(current_batch))
      max_batch_size *= 2
      if DEBUG >= 2: print(f"JIT GRAPHing batch with {len(current_batch)} kernels")
    current_batch, current_batch_devs = [], []

  for si in linear.src:
    if si.src[0].op is Ops.SLICE: continue

    devs = dedup([Device[x] for b in si.src[1:] if b.op is not Ops.BIND for x in (b.device if isinstance(b.device, tuple) else (b.device,))])
    graph_t = graph_class(devs[0]) if devs[0].graph is not None else None

    can_graph = graph_t is not None and graph_t.supports_uop(devs, si)
    can_extend = can_graph and graph_t is not None and (not current_batch_devs or graph_t.supports_uop(current_batch_devs, si)) \
      and (max_batch_size == 0 or len(current_batch) < max_batch_size)
    if not can_extend and current_batch: flush_batch()

    # append this si and update devs
    (current_batch if can_graph else new_src).append(si)
    current_batch_devs = dedup(current_batch_devs + devs) if can_graph else []
  if current_batch: flush_batch()
  return linear.replace(src=tuple(new_src))

def _copy_input(u:UOp) -> UOp:
  run_linear(UOp(Ops.LINEAR, src=(u.copy_to_device(u.device).call(new:=UOp.new_buffer(u.device, u.max_numel(), u.dtype), u),)))
  return new

@track_rewrites(lambda linear,held_bufs,input_uops,ret=(): f"JIT {pluralize('call', len(linear.src))}")
def jit_lower(linear:UOp, held_bufs:set[UOp], input_uops:list[UOp]) -> UOp:
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View captured linear")

  # parametrize input buffers: map each input buffer UOp to a PARAM with the correct slot index
  linear = linear.substitute({u: UOp.param(i, u.dtype, u.shape, u.device) for i,u in enumerate(input_uops)}, walk=True)
  linear = memory_plan_rewrite(linear, held_bufs)
  linear = compile_linear(linear, beam=getenv("JITBEAM", BEAM.value), jit=True)
  if JIT < 2: linear = graph_split_rewrite(linear, max_batch_size=JIT_BATCH_SIZE.value)
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View graphed linear")
  return linear

class GraphException(Exception): pass
class JitError(Exception): pass

def _check_no_non_tensor_return(ret):
  if ret is None or isinstance(ret, Tensor): return
  if isinstance(ret, (tuple, list, dict)):
    for item in (ret.values() if isinstance(ret, dict) else ret): _check_no_non_tensor_return(item)
    return
  raise JitError(f"JIT return contains non-Tensor value of type {type(ret).__name__}")

def graph_class(dev): return dev.graph.func if isinstance(dev.graph, functools.partial) else dev.graph

class DepsTracker:
  def __init__(self):
    # tracks (offset, end, dep) ranges per base buffer id to handle suballocated buffers correctly.
    self.w_dependency_map: dict[int, list[tuple[int, int, Any]]] = collections.defaultdict(list)
    self.r_dependency_map: dict[int, list[tuple[int, int, Any]]] = collections.defaultdict(list)

  @staticmethod
  def _key(buf:Any) -> tuple[Any, int, int]: return id(buf.base), buf.offset, buf.offset + buf.nbytes

  def access_resources(self, bufs:list[Any], write:list[int], new_dependency:Any):
    wait_nodes = []
    for i,buf in enumerate(bufs):
      key, s, e = self._key(buf)
      wait_nodes += [dep for st,en,dep in self.w_dependency_map[key] if st < e and s < en]
      if i in write: wait_nodes += [dep for st,en,dep in self.r_dependency_map[key] if st < e and s < en]
    for i,buf in enumerate(bufs):
      key, s, e = self._key(buf)
      if i in write:
        for dmap in [self.w_dependency_map, self.r_dependency_map]:
          kept = []
          for st,en,dep in dmap[key]:
            if st < min(s, en): kept.append((st, min(s, en), dep))
            if max(e, st) < en: kept.append((max(e, st), en, dep))
          dmap[key] = kept
        self.w_dependency_map[key].append((s, e, new_dependency))
      else: self.r_dependency_map[key].append((s, e, new_dependency))
    return list({id(x):x for x in wait_nodes}.values())

class GraphRunner:
  def __init__(self, linear:UOp, input_uops:tuple[UOp, ...]=()):
    self.linear = linear.src[0]
    self.calls: list[tuple[int, UOp, list[Buffer], dict[str, int]]] = []
    self.runtimes: list[Any|None] = []
    self.uop_replace: list[list[tuple[int, int]]] = []
    for call in self.linear.src:
      replace = [(p, b.arg.slot) for p, b in enumerate(get_call_arg_uops(call)) if b.op is Ops.PARAM]
      for dev_idx, (bufs, device_vars) in enumerate(unwrap_multi(call, resolve_params(call, input_uops))):
        self.calls.append((dev_idx, call.src[0], [b.ensure_allocated() for b in bufs], device_vars))
        self.runtimes.append(get_runtime(bufs[0].device, call.src[0]) if call.src[0].op is Ops.PROGRAM else None)
        self.uop_replace.append(replace)

    self.var_vals_replace:dict[int, list[tuple[int, int]]] = {}
    self.launch_dims_replace:dict[int, tuple[int|None, int|None]] = {}
    self.launch_dims_base:dict[int, tuple[tuple[int|float, ...], tuple[int, ...]]] = {}

    def is_sym_dim(dim) -> bool: return not all(isinstance(d, (int, float)) for d in dim)

    crs = [(j, self.calls[j][1].arg, self.calls[j][3]) for j in range(len(self.calls)) if self.calls[j][1].op is Ops.PROGRAM]
    self.vars = sorted({v.expr for _,p,dv in crs for v in p.vars if v.expr not in dv | p.runtimevars})
    self.symbolic_dims = dedup(tuple(d) for _,p,_ in crs for d in (p.local_size, p.global_size) if d and is_sym_dim(d))

    def find_symbolic_dim(dim): return self.symbolic_dims.index(tuple(dim)) if dim is not None and tuple(dim) in self.symbolic_dims else None

    for j,p,dv in crs:
      if (replace:=[(i, self.vars.index(v.expr)) for i, v in enumerate(p.vars) if v.expr not in dv | p.runtimevars]):
        self.var_vals_replace[j] = replace
      global_dim_idx, local_dim_idx = find_symbolic_dim(p.global_size), find_symbolic_dim(p.local_size)
      if global_dim_idx is not None or local_dim_idx is not None:
        self.launch_dims_replace[j] = (global_dim_idx, local_dim_idx)
        assert p.local_size is not None
        self.launch_dims_base[j] = (tuple(p.global_size), tuple(p.local_size))

    estimates = sum((estimate_uop(call) for call in self.linear.src), Estimates())

    # used in MultiGraphRunner
    self.deps = DepsTracker()

    self.device, self.estimates = self.calls[0][2][0].device.split(":")[0], estimates.simplify()

  def __call__(self, input_uops:tuple[UOp, ...], var_vals:dict[str, int], wait=False) -> float|None: raise NotImplementedError("override this")

  def updated_vars(self, var_vals: dict[str, int]):
    vals = [var_vals[v] for v in self.vars]
    for j, vidxs in self.var_vals_replace.items():
      for i, v in vidxs: yield j, i, vals[v]

  def updated_launch_dims(self, var_vals: dict[str, int]):
    dims = [tuple(sym_infer(s, var_vals) for s in dim) for dim in self.symbolic_dims]
    for j, (gl, lc) in self.launch_dims_replace.items():
      yield j, (dims[gl] if gl is not None else self.launch_dims_base[j][0]), (dims[lc] if lc is not None else self.launch_dims_base[j][1])

  def _access_resources(self, bufs:list[Buffer], write:list[int], new_dependency:Any):
    return self.deps.access_resources(bufs, write, new_dependency)

  @staticmethod
  def _all_devs(batch_devs:list[Compiled], new_call:UOp) -> list[Compiled]:
    return dedup(batch_devs + [Device[x] for b in get_call_arg_uops(new_call)
                 for x in (b.device if isinstance(b.device, tuple) else (b.device,))])

  @staticmethod
  def supports_uop(batch_devs:list[Compiled], new_call:UOp) -> bool:
    return new_call.src[0].op is Ops.PROGRAM and len(GraphRunner._all_devs(batch_devs, new_call)) == 1

# a marker for your graph supporting multiple devices of the same type
class MultiGraphRunner(GraphRunner):
  @staticmethod
  def supports_uop(batch_devs:list[Compiled], new_call:UOp) -> bool:
    # Devices must be the same type
    return new_call.src[0].op in (Ops.PROGRAM, Ops.COPY) and len(dedup([type(d) for d in GraphRunner._all_devs(batch_devs, new_call)])) == 1

ReturnType = TypeVar('ReturnType')
@dataclass
class CapturedJit(Generic[ReturnType]):
  ret: Any  # includes the Tensors or any other returned object
  _linear: UOp
  expected_names: list[int|str]
  expected_input_info: list[tuple[UOp, tuple[Variable, ...], DType, str]]  # (view, variables, dtype, device) per input

  @functools.cached_property
  def linear(self) -> UOp: return link_linear(self._linear, jit=True)

  def __reduce__(self): return self.__class__, (self.ret, self._linear, self.expected_names, self.expected_input_info)

  @functools.cached_property
  def _written_uops(self) -> set[UOp]:
    out: set[UOp] = set()
    for call in self.linear.toposort():
      if call.op is not Ops.CALL: continue
      arg_uops = get_call_arg_uops(call)
      outs, ins = get_call_outs_ins(call)
      out |= {arg_uops[k] for k in set(outs) - set(ins) if arg_uops[k].op in (Ops.BUFFER, Ops.SLICE)}
    return out

  def __call__(self, input_uops:list[UOp], var_vals:dict[str, int]) -> ReturnType:
    concrete = tuple(_copy_input(u) if u in self._written_uops else u for u in input_uops)
    if DEBUG >= 1 and len(self.linear.src) >= 10: print(f"jit execs {len(self.linear.src)} calls")
    run_linear(self.linear, var_vals, input_uops=concrete, jit=True)
    return self.ret

  def free_intermediates(self):
    # drop graph runners
    for call in self.linear.src:
      if call.src[0].op is Ops.CUSTOM_FUNCTION and call.src[0].arg == "graph": graph_cache.pop(call.src[0], None)
    for u in self._written_uops:
      if (buf:=buffers.get(u)) is None: continue
      for b in (buf.bufs if isinstance(buf, MultiBuffer) else (buf,)):
        if b.is_initialized(): b.deallocate()
        if (base:=b._base) is not None and base.allocated_views == 0 and base.is_allocated(): base.deallocate()

def _prepare_jit_inputs(args, kwargs):
  input_tensors: list[tuple[int|str, Tensor]] = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
  names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
  # extract tensors from containers (shallow, not recursive to avoid grabbing model weights)
  for x in args + tuple(kwargs.values()):
    it = x if isinstance(x, (tuple,list)) else x.values() if isinstance(x, dict) else []
    tensors += [t for t in it if t.__class__ is Tensor and not any(t is y for y in tensors)]
  def get_input_uops() -> list[UOp]: return flatten([t.uop.src if t.uop.op is Ops.MULTI else [t.uop] for t in tensors])
  if any(u.device is None for u in get_input_uops()): raise JitError("JIT inputs must be real buffers; use .clone()")
  if len(unrealized_tensors := [x for x in tensors if not x.uop.is_realized]): Tensor.realize(*unrealized_tensors)
  input_uops = get_input_uops()
  # collect buffer UOps (including MultiBuffer)
  input_buf_uops: list[UOp] = [u.base for u in input_uops if u.base.realized is not None]
  if len(set(input_buf_uops)) != len(input_buf_uops): raise JitError("duplicate inputs to JIT")
  inputs = [(*(u.substitute({u.base:UOp(Ops.NOOP, u.base.dtype)}, extra_pm=mop_cleanup).unbind_all()), u.dtype, u.device) for u in input_uops]
  _var_vals = merge_dicts([x[1] for x in inputs] + [dict(v.unbind() for v in (args + tuple(kwargs.values())) if isinstance(v, UOp))])
  var_vals = {k.expr:v for k,v in _var_vals.items()}
  expected_input_info = [(x[0], tuple(sorted(x[1].keys(), key=lambda v: v.expr)), x[2], x[3]) for x in inputs]
  return input_buf_uops, var_vals, names, expected_input_info

class TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]|None, captured:CapturedJit|None=None, prune=False):
    assert fxn or captured, "need either a function or a CapturedJit"
    self.fxn = fxn
    self.captured: CapturedJit|None = captured
    self.cnt: int = 2 if self.fxn is None else 0
    self.prune = prune

  def add_linear(self, linear:UOp, var_vals:dict[str, int]): self._linears.append(linear)

  def reset(self):
    assert self.fxn is not None, "can't reset without function"
    self.cnt = 0
    self.captured = None

  def __reduce__(self):
    assert self.captured is not None, "can't pickle an uncaptured JIT"
    return self.__class__, (None, self.captured)

  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj) # add support for instance methods

  @disable_gc()
  def __call__(self, *args, **kwargs) -> ReturnType:
    input_buf_uops, var_vals, names, expected_input_info = _prepare_jit_inputs(args, kwargs)
    if not JIT or self.cnt == 0:
      # jit ignore
      assert self.fxn is not None
      with Context(BEAM=0 if getenv("IGNORE_JIT_FIRST_BEAM") else BEAM.value):
        ret = self.fxn(*args, **kwargs)
        if len(params:=get_parameters(ret)): Tensor.realize(*params)
    elif self.cnt == 1:
      # jit capture
      assert self.fxn is not None
      if capturing: raise RuntimeError(f"having TinyJit inside another TinyJit is not supported {len(capturing)=} {capturing=}")
      self._linears: list[UOp] = []
      capturing.append(self)
      try:
        ret = self.fxn(*args, **kwargs)
        if len(params:=get_parameters(ret)): Tensor.realize(*params)
      finally: capturing.clear()
      if not len(self._linears): raise JitError("didn't JIT anything!")
      _check_no_non_tensor_return(ret)
      if DEBUG >= 1: print(f"JIT captured {len(self._linears)} linears with {len(input_buf_uops)} inputs")

      # combine all captured linears into one, memory plan, and graph split
      big_linear = UOp(Ops.LINEAR, src=tuple(flatten([l.src for l in self._linears])))
      del self._linears

      if self.prune:
        big_linear, onetime_linear = prune_linear(big_linear, set(input_buf_uops))
        if DEBUG >= 1: print(f"pruned from {len(big_linear.src) + len(onetime_linear.src)} -> {len(big_linear.src)} kernels")
        run_linear(onetime_linear, var_vals)

      # hold all buffers reachable from live Tensors (e.g. lazy .grad created during capture), the memory planner can't suballocate those
      held_bufs = set(buffers) | {u for tref in list(all_tensors) if (t:=tref()) is not None for u in t.uop.toposort() if u.op is Ops.BUFFER}
      linear = jit_lower(big_linear, held_bufs, input_buf_uops)
      self.captured = CapturedJit(ret, linear, names, expected_input_info)
      ret = self.captured(input_buf_uops, var_vals)
    elif self.cnt >= 2:
      # jit exec
      assert self.captured is not None
      if self.captured.expected_names != names: raise JitError(f"args mismatch in JIT: {self.captured.expected_names=} != {names}")
      if self.captured.expected_input_info != expected_input_info:
        raise JitError(f"args mismatch in JIT: {self.captured.expected_input_info=} != {expected_input_info=}")
      ret = self.captured(input_buf_uops, var_vals)

    self.cnt += 1
    return ret
