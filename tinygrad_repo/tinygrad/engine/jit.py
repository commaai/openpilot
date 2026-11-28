from typing import TypeVar, Generic, Callable, Any, overload
import functools
from tinygrad.tensor import Tensor, all_tensors
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, BEAM, getenv, JIT, pluralize, VIZ, disable_gc
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.dtype import DType, AddrSpace
from tinygrad.uop.ops import UOp, PatternMatcher, Variable, Ops, rewrite_group, graph_rewrite
from tinygrad.engine.realize import capturing, link_linear, run_linear
from tinygrad.codegen.compile import compile_linear
from tinygrad.codegen import create_linear
from tinygrad.codegen.memory import memory_plan_rewrite, _collect_bufs
from tinygrad.nn.state import get_parameters
from tinygrad.uop.movement import mop_cleanup
from dataclasses import dataclass, field

def prune_linear(linear:UOp, needed:set[UOp]) -> tuple[UOp, UOp]:
  kept, onetime = [], []
  for si in linear.src:
    si_bufs = {b for src in si.src[1:] for b in _collect_bufs(src)}
    if not si_bufs.isdisjoint(needed):
      kept.append(si)
      needed |= si_bufs
    else: onetime.append(si)
  return linear.replace(src=tuple(kept)), linear.replace(src=tuple(onetime))

# a captured export is scheduled on its storage: the EXTERNs are its externs, the ALLOCs get fresh storage
def bind_export(linear:UOp, externs:tuple[UOp, ...]) -> UOp:
  storage = {x.as_extern(i): x.unbound() if x.is_variable else x for i,x in enumerate(externs)}
  allocs = {b: UOp.new_buffer(b.device, b.max_numel(), b.dtype) for b in linear.toposort() if b.op is Ops.ALLOC and b.addrspace is AddrSpace.GLOBAL}
  return linear.substitute(storage | allocs, walk=True)

@rewrite_group(lambda linear,held_bufs,input_uops,ret=(): f"JIT {pluralize('call', len(linear.src))}")
def jit_lower(linear:UOp, held_bufs:set[UOp], input_uops:list[UOp]) -> tuple[UOp, tuple[UOp, ...]]:
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View captured linear")

  # parametrize input buffers: map each input buffer UOp to a PARAM with the correct slot index
  linear = linear.substitute({u: UOp.param(i, u.dtype, u.max_numel(), u.device) for i,u in enumerate(input_uops)}, walk=True)
  # storage no one outside the jit holds is scratch: it's planned and bound at link
  scratch = {u: UOp.alloc((u.max_numel(),), u.dtype, device=u.device) for u in linear.toposort() if u.op is Ops.BUFFER and u not in held_bufs}
  linear = memory_plan_rewrite(linear.substitute(scratch, walk=True))
  # the storage it holds is linked after the inputs: the compiled linear has no buffers
  held = tuple(u for u in linear.toposort() if u.op is Ops.BUFFER)
  linear = linear.substitute({u: u.as_extern(len(input_uops)+i) for i,u in enumerate(held)}, walk=True)
  linear = compile_linear(linear, beam=getenv("JITBEAM", BEAM.value))
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View compiled linear")
  return linear, held

class JitError(Exception): pass

def _check_no_non_tensor_return(ret):
  if ret is None or isinstance(ret, Tensor): return
  if isinstance(ret, (tuple, list, dict)):
    for item in (ret.values() if isinstance(ret, dict) else ret): _check_no_non_tensor_return(item)
    return
  raise JitError(f"JIT return contains non-Tensor value of type {type(ret).__name__}")

ReturnType = TypeVar('ReturnType')
@dataclass
class CapturedJit(Generic[ReturnType]):
  ret: Any  # includes the Tensors or any other returned object
  _linear: UOp
  _externs: tuple[UOp, ...]  # the storage the jit holds, linked after the inputs
  expected_names: list[int|str]
  expected_input_info: list[tuple[UOp, tuple[Variable, ...], DType, str]]  # (view, variables, dtype, device) per input
  var_vals: dict[str, int]  # the values of the Variables at capture, inputs are replaced every call
  _linears: dict[tuple[UOp, ...], tuple[UOp, dict[UOp, UOp]]] = field(default_factory=dict)  # aliased inputs -> (linked linear with copies, staging)

  def __reduce__(self): return self.__class__, (self.ret, self._linear, self._externs, self.expected_names, self.expected_input_info, self.var_vals)

  @functools.cached_property
  def _jit_bufs(self) -> set[UOp]: return set(self._externs)

  @functools.cached_property
  def _symbolic_ret(self) -> list[tuple[Tensor, UOp, dict[Variable, int]]]:
    return [(t, *ub) for t in get_parameters(self.ret) if (ub:=t.uop.unbind_all())[1]]

  def __call__(self, input_uops:list[UOp], var_vals:dict[str, int]) -> ReturnType:
    # aliased inputs
    if (aliased:=tuple(u for u in input_uops if u in self._jit_bufs)) not in self._linears:
      if any(u.on_disk() for u in aliased): raise JitError("cannot make an independent copy of a written DISK input")
      staging = {u: UOp.new_buffer(u.device, u.max_numel(), u.dtype) for u in aliased}
      copies = compile_linear(UOp(Ops.LINEAR, src=tuple(s.store_call(u) for u,s in staging.items())))
      self._linears[aliased] = (link_linear(self._linear.replace(src=copies.src+self._linear.src), [*input_uops, *self._externs]), staging)

    linear, staging = self._linears[aliased]
    if DEBUG >= 1 and len(linear.src) >= 10: print(f"jit execs {len(linear.src)} calls")
    run_linear(linear, {**self.var_vals, **var_vals}, input_uops=tuple(staging.get(u, u) for u in input_uops)+self._externs, jit=True)
    for t,u,vals in self._symbolic_ret: t.uop = u.substitute({v:v.bind(var_vals.get(v.expr, i)) for v,i in vals.items()}, walk=True)
    return self.ret

  def free_intermediates(self): self._linears.clear()

def _prepare_jit_inputs(args, kwargs):
  input_tensors: list[tuple[int|str, Tensor]] = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
  names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
  # extract tensors from containers (shallow, not recursive to avoid grabbing model weights)
  for x in args + tuple(kwargs.values()):
    it = x if isinstance(x, (tuple,list)) else x.values() if isinstance(x, dict) else []
    tensors += [t for t in it if t.__class__ is Tensor and not any(t is y for y in tensors)]
  def get_input_uops() -> list[UOp]: return [t.uop.src[0] if t.uop.op is Ops.UNSHARD else t.uop for t in tensors]
  if any(u.is_virtual for u in get_input_uops()): raise JitError("JIT inputs must be real buffers; use .clone()")
  if len(unrealized_tensors := [x for x in tensors if not x.uop.is_realized]): Tensor.realize(*unrealized_tensors)
  input_uops = get_input_uops()
  # collect buffer UOps (including MultiBuffer)
  input_buf_uops: list[UOp] = [u.base for u in input_uops if u.base.realized is not None]
  if len(set(input_buf_uops)) != len(input_buf_uops): raise JitError("duplicate inputs to JIT")
  inputs = [(*(u.substitute({u.base:UOp(Ops.NOOP)}, extra_pm=mop_cleanup).unbind_all()), u.dtype, u.device) for u in input_uops]
  _var_vals = merge_dicts([x[1] for x in inputs] + [dict(v.unbind() for v in (args + tuple(kwargs.values())) if isinstance(v, UOp))])
  var_vals = {k.expr:v for k,v in _var_vals.items()}
  expected_input_info = [(x[0], tuple(sorted(x[1].keys(), key=lambda v: v.expr)), x[2], x[3]) for x in inputs]
  return input_buf_uops, var_vals, names, expected_input_info

class _TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]|None, captured:CapturedJit|None=None, prune=False):
    assert fxn or captured, "need either a function or a CapturedJit"
    self.fxn = fxn
    self.captured: CapturedJit|None = captured
    self.cnt: int = 2 if self.fxn is None else 0
    self.prune = prune

  def add_export(self, fn:UOp, externs:tuple[UOp, ...]): self._exports.append((fn, externs))

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
      self._exports: list[tuple[UOp, tuple[UOp, ...]]] = []
      capturing.append(self)
      try:
        ret = self.fxn(*args, **kwargs)
        if len(params:=get_parameters(ret)): Tensor.realize(*params)
      finally: capturing.clear()
      if not len(self._exports): raise JitError("didn't JIT anything!")
      _check_no_non_tensor_return(ret)
      if DEBUG >= 1: print(f"JIT captured {len(self._exports)} exports with {len(input_buf_uops)} inputs")

      # schedule the captured exports on their storage. the prune needs unplanned storage, jit_lower plans the memory of all of them together
      with Context(NO_MEMORY_PLANNER=1): linears = [bind_export(create_linear(fn), externs) for fn,externs in self._exports]
      # combine all the linears into one, memory plan, and compile
      big_linear = UOp(Ops.LINEAR, src=tuple(flatten([l.src for l in linears])))
      # the values of the Variables at capture, inputs are replaced every call
      capture_var_vals = merge_dicts([{x.expr: x.arg.val for x in externs if x.is_bound_var} for _,externs in self._exports])
      del linears, self._exports

      if self.prune:
        big_linear, onetime_linear = prune_linear(big_linear, set(input_buf_uops))
        if DEBUG >= 1: print(f"pruned from {len(big_linear.src) + len(onetime_linear.src)} -> {len(big_linear.src)} kernels")
        run_linear(link_linear(compile_linear(onetime_linear)), capture_var_vals)
        del onetime_linear

      # hold all buffers with real storage reachable from live Tensors (e.g. lazy .grad created during capture) and all buffers with
      # allocated storage in the captured linear (e.g. constants baked in by copies): the memory planner can't suballocate those
      def _buf_or_none(u:UOp) -> Buffer|MultiBuffer|None: return u.arg.buffer if u.op is Ops.BUFFER else None
      held_bufs = {u for tref in list(all_tensors) if (t:=tref()) is not None for u in t.uop.toposort() if _buf_or_none(u) is not None}
      held_bufs |= {u for u in big_linear.toposort() if (b:=_buf_or_none(u)) is not None and b.is_allocated()}
      linear, held = jit_lower(big_linear, held_bufs, input_buf_uops)
      # drop the pre-planning graph: it keeps the whole capture-time working set allocated (big_linear) or referenced (held_bufs).
      # the planned linear only uses the arena/held buffers, so the intermediates must be freed before linking and first exec
      del big_linear, held_bufs
      self.captured = CapturedJit(ret, linear, held, names, expected_input_info, capture_var_vals)
      ret = self.captured(input_buf_uops, var_vals)
    elif self.cnt >= 2:
      # jit exec
      assert self.captured is not None
      if self.captured.expected_names != names: raise JitError(f"args mismatch in JIT: {self.captured.expected_names=} != {names}")
      if self.captured.expected_input_info != expected_input_info:
        raise JitError(f"args mismatch in JIT: {self.captured.expected_input_info=} != {expected_input_info=}")
      ret = self.captured(input_buf_uops, var_vals)

    if JIT: self.cnt += 1
    return ret

# overload signatures support both @TinyJit and @TinyJit(prune=True) syntax
@overload
def TinyJit(fxn:Callable[..., ReturnType], *, prune:bool=False) -> _TinyJit[ReturnType]: ...
@overload
def TinyJit(fxn:None=None, *, prune:bool=False) -> Callable[[Callable[..., ReturnType]], _TinyJit[ReturnType]]: ...
def TinyJit(fxn=None, **kwargs): return (lambda f: _TinyJit(f, **kwargs)) if fxn is None else _TinyJit(fxn, **kwargs)
