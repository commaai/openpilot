from typing import TypeVar, Generic, Callable, Any, overload
import functools
from tinygrad.tensor import Tensor, all_tensors
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, BEAM, getenv, JIT, pluralize, VIZ, disable_gc
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.dtype import DType
from tinygrad.uop.ops import UOp, PatternMatcher, Variable, Ops, rewrite_group, graph_rewrite
from tinygrad.engine.realize import capturing, compile_linear, link_linear, run_linear
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

def _copy_input(u:UOp) -> UOp:
  if u.on_disk(): raise JitError("cannot make an independent copy of a written DISK input")
  run_linear(UOp(Ops.LINEAR, src=((new:=UOp.new_buffer(u.device, u.max_numel(), u.dtype)).store_call(u),)))
  return new

@rewrite_group(lambda linear,held_bufs,input_uops,ret=(): f"JIT {pluralize('call', len(linear.src))}")
def jit_lower(linear:UOp, held_bufs:set[UOp], input_uops:list[UOp]) -> UOp:
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View captured linear")

  # parametrize input buffers: map each input buffer UOp to a PARAM with the correct slot index
  linear = linear.substitute({u: UOp.param(i, u.dtype, u.max_numel(), u.device) for i,u in enumerate(input_uops)}, walk=True)
  linear = memory_plan_rewrite(linear, held_bufs)
  linear = linear.substitute({u: u.rtag("scratch") for u in linear.toposort() if u.op is Ops.BUFFER and u not in held_bufs}, walk=True)
  linear = compile_linear(linear, beam=getenv("JITBEAM", BEAM.value), input_uops=input_uops, cache=False)
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View compiled linear")
  return linear

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
  expected_names: list[int|str]
  expected_input_info: list[tuple[UOp, tuple[Variable, ...], DType, str]]  # (view, variables, dtype, device) per input

  @functools.cached_property
  def linear(self) -> UOp: return link_linear(self._linear, allow_cache=False) # do not cache jit

  def __reduce__(self): return self.__class__, (self.ret, self._linear, self.expected_names, self.expected_input_info)

  @functools.cached_property
  def _jit_bufs(self) -> set[UOp]: return {u for u in self._linear.toposort() if u.op is Ops.BUFFER}

  @functools.cached_property
  def _symbolic_ret(self) -> list[tuple[Tensor, UOp, dict[Variable, int]]]:
    return [(t, *ub) for t in get_parameters(self.ret) if (ub:=t.uop.unbind_all())[1]]

  def __call__(self, input_uops:list[UOp], var_vals:dict[str, int]) -> ReturnType:
    concrete = tuple(_copy_input(u) if u in self._jit_bufs else u for u in input_uops)
    if DEBUG >= 1 and len(self.linear.src) >= 10: print(f"jit execs {len(self.linear.src)} calls")
    run_linear(self.linear, var_vals, input_uops=concrete, jit=True)
    for t,u,vals in self._symbolic_ret: t.uop = u.substitute({v:v.bind(var_vals.get(v.expr, i)) for v,i in vals.items()}, walk=True)
    return self.ret

  def free_intermediates(self):
    bufs = [u.arg.buffer for u in self._jit_bufs if u.tag == "scratch"]
    if not (freed:=[b for b in flatten(b.bufs if isinstance(b, MultiBuffer) else [b] for b in bufs) if b.is_allocated()]): return
    for b in freed: b.deallocate()
    self.__dict__.pop("linear", None) # should be relinked

def _prepare_jit_inputs(args, kwargs):
  input_tensors: list[tuple[int|str, Tensor]] = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
  names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
  # extract tensors from containers (shallow, not recursive to avoid grabbing model weights)
  for x in args + tuple(kwargs.values()):
    it = x if isinstance(x, (tuple,list)) else x.values() if isinstance(x, dict) else []
    tensors += [t for t in it if t.__class__ is Tensor and not any(t is y for y in tensors)]
  def get_input_uops() -> list[UOp]: return flatten([[t.uop.src[0]] if t.uop.op is Ops.UNSHARD else [t.uop] for t in tensors])
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

  def add_linear(self, linear:UOp): self._linears.append(linear)

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

      # combine all captured linears into one, memory plan, and compile
      big_linear = UOp(Ops.LINEAR, src=tuple(flatten([l.src for l in self._linears])))
      del self._linears

      if self.prune:
        big_linear, onetime_linear = prune_linear(big_linear, set(input_buf_uops))
        if DEBUG >= 1: print(f"pruned from {len(big_linear.src) + len(onetime_linear.src)} -> {len(big_linear.src)} kernels")
        run_linear(onetime_linear, var_vals)
        del onetime_linear

      # hold all buffers with real storage reachable from live Tensors (e.g. lazy .grad created during capture) and all buffers with
      # allocated storage in the captured linear (e.g. constants baked in by copies): the memory planner can't suballocate those
      def _buf_or_none(u:UOp) -> Buffer|MultiBuffer|None: return u.arg.buffer if u.op is Ops.BUFFER else None
      held_bufs = {u for tref in list(all_tensors) if (t:=tref()) is not None for u in t.uop.toposort() if _buf_or_none(u) is not None}
      held_bufs |= {u for u in big_linear.toposort() if (b:=_buf_or_none(u)) is not None and b.is_allocated()}
      linear = jit_lower(big_linear, held_bufs, input_buf_uops)
      # drop the pre-planning graph: it keeps the whole capture-time working set allocated (big_linear) or referenced (held_bufs).
      # the planned linear only uses the arena/held buffers, so the intermediates must be freed before linking and first exec
      del big_linear, held_bufs
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

# overload signatures support both @TinyJit and @TinyJit(prune=True) syntax
@overload
def TinyJit(fxn:Callable[..., ReturnType], *, prune:bool=False) -> _TinyJit[ReturnType]: ...
@overload
def TinyJit(fxn:None=None, *, prune:bool=False) -> Callable[[Callable[..., ReturnType]], _TinyJit[ReturnType]]: ...
def TinyJit(fxn=None, **kwargs): return (lambda f: _TinyJit(f, **kwargs)) if fxn is None else _TinyJit(fxn, **kwargs)
