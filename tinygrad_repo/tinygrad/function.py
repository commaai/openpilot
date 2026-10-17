import functools, time
from dataclasses import replace
from typing import Generic, TypeVar, Callable, cast, overload
from tinygrad.helpers import Context, dedup, getenv, DEBUG
from tinygrad.uop.ops import UOp, Ops, GroupOp, graph_rewrite, PatternMatcher, UPat
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_state_dict

def add_to_ctx(ctx, x:UOp):
  if x.buf_uop in ctx[1]: return None
  ret = x.param_like(len(ctx[0]))
  ctx[0].append(x)
  return ret

def is_implicit_storage(ctx, x:UOp) -> bool:
  # Variables are caller-provided values (slot -1); positional scalar params (slot >= 0) are already captured
  return (x.is_variable and x.arg.slot == -1) or x.op is Ops.BUFFER or (x.op is Ops.ALLOC and x.arg.bind_on_realize and x.arg.slot < ctx[2])

pm_ctx = PatternMatcher([
  # Capture caller-owned storage, not allocations created while tracing this function.
  (UPat(GroupOp.Defines, name="x"), lambda ctx,x: add_to_ctx(ctx,x) if is_implicit_storage(ctx, x) else None),
  (UPat((Ops.AFTER, Ops.STAGE), name="x"), lambda ctx,x: add_to_ctx(ctx,x) if
   not any(p.op is Ops.PARAM and p.arg.slot >= 0 for p in x.backward_slice) and
   any(is_implicit_storage(ctx, b) for b in x.toposort()) else None),
])

def invalid_outputs(uret:UOp) -> set[UOp]:
  # invalids() returns fresh write-only scratch: a clone storing CONST(Invalid)
  # don't capture it as an input; only skip fresh buffers, not realized ones
  return {u.src[0].buf_uop for u in uret.backward_slice_with_self
          if u.op is Ops.STORE and u.src[1].base.is_invalid and not u.src[0].buf_uop.is_realized}

def renumber_invalid_outputs(uret:UOp) -> UOp:
  invalid = invalid_outputs(uret)
  return uret.substitute({b:b.replace(op=Ops.ALLOC, arg=replace(b.arg, slot=i, buffer=None, bind_on_realize=False))
                          for i,b in enumerate(x for x in uret.toposort() if x in invalid)})

ReturnType = TypeVar('ReturnType')
class _function(Generic[ReturnType]):
  depth = 0
  def __init__(self, fxn:Callable[..., ReturnType], *, precompile:bool, precompile_backward:bool, allow_implicit:bool, grad_fxn:Callable|None):
    self.fxn = fxn
    self.precompile = precompile
    self.precompile_backward = precompile_backward
    self.allow_implicit = allow_implicit
    self.grad_fxn = grad_fxn

  def __get__(self, obj, objtype=None): return functools.partial(self.__call__, obj) if obj is not None else self

  def __call__(self, *args, **kwargs) -> ReturnType:
    st = time.perf_counter()

    params = get_state_dict((args, kwargs), tensor_type=(Tensor, UOp)).values()

    # deduplicate input_uops, keeping the first occurrence index for each unique uop
    call_uops: list[UOp] = dedup([u for t in params if (u:=t._uop).device is not None])

    # disable realize/schedule while this is running
    # run it and do surgery later
    alloc_start = next(UOp.unique_num)
    with Context(ALLOW_DEVICE_USAGE=getenv("DEVICE_IN_FUNCTION_BUG", 0)):
      _function.depth += 1
      try:
        ret = self.fxn(*args, **kwargs)
      finally:
        _function.depth -= 1
    if isinstance(ret, Tensor):
      uret = ret.uop
    elif isinstance(ret, tuple) and all(isinstance(x, Tensor) for x in ret):
      uret = UOp.sink(*[x.uop for x in ret])
    else:
      raise RuntimeError(f"function return type {type(ret)} not supported")

    # replace the known inputs with params (using deduplicated slots)
    subs = {x:x.param_like(i) for i,x in enumerate(call_uops)}
    uret = uret.substitute(subs)

    # caller-owned storage left in the graph becomes implicit inputs
    num_explicit = len(call_uops)
    uret = graph_rewrite(uret, pm_ctx, (call_uops, invalid_outputs(uret), alloc_start), bottom_up=True, name="get_implicit_inputs")
    uret = renumber_invalid_outputs(uret)
    name = getattr(self.fxn, '__qualname__', None) or type(self.fxn).__qualname__
    if not self.allow_implicit:
      implicit_buffers = [x for x in call_uops[num_explicit:] if x.op in {Ops.BUFFER, Ops.ALLOC}]
      if implicit_buffers:
        buf_strs = '\n  '.join(f"{i}: dtype={b.dtype}, size={b.max_numel()}, device={b.device}" for i,b in enumerate(implicit_buffers))
        raise RuntimeError(f"function {name} has {len(implicit_buffers)} implicit buffer(s), but allow_implicit=False\n  {buf_strs}")

    outs = UOp.call_with_outputs(uret.src if isinstance(ret, tuple) else (uret,), *call_uops, grad_fxn=self.grad_fxn, name=name,
                                 precompile=self.precompile, precompile_backward=self.precompile_backward)

    if DEBUG >= 2:
      print("  "*_function.depth+f"function {uret.key.hex()[:8]} in {(time.perf_counter()-st)*1000:8.2f} ms: {name}")

    if isinstance(ret, tuple):
      return cast(ReturnType, tuple(Tensor(o) for o in outs))
    else:
      return cast(ReturnType, Tensor(outs[0]))

# overload signatures support both @function and @function(precompile=True) syntax
@overload
def function(fxn:Callable[..., ReturnType], *, precompile:bool=False, precompile_backward:bool=False,
             allow_implicit:bool=False, grad_fxn:Callable|None=None) -> _function[ReturnType]: ...
@overload
def function(fxn:None=None, *, precompile:bool=False, precompile_backward:bool=False,
             allow_implicit:bool=False, grad_fxn:Callable|None=None) -> Callable[[Callable[..., ReturnType]], _function[ReturnType]]: ...
def function(fxn=None, *, precompile:bool=False, precompile_backward:bool=False,
             allow_implicit:bool=False, grad_fxn:Callable|None=None):
  if fxn is None:
    return lambda f: _function(f, precompile=precompile, precompile_backward=precompile_backward,
                               allow_implicit=allow_implicit, grad_fxn=grad_fxn)
  return _function(fxn, precompile=precompile, precompile_backward=precompile_backward,
                   allow_implicit=allow_implicit, grad_fxn=grad_fxn)
