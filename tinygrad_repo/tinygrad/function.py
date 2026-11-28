import functools, time
from typing import TypeVar, Callable, overload
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
   any(is_implicit_storage(ctx, b) for b in x.backward_slice) else None),
])

def invalid_outputs(uret:UOp) -> set[UOp]:
  # invalids() returns fresh write-only scratch: a clone storing CONST(Invalid)
  # don't capture it as an input; only skip fresh buffers, not realized ones
  return {u.src[0].buf_uop for u in uret.backward_slice_with_self
          if u.op is Ops.STORE and u.src[1].base.is_invalid and not u.src[0].buf_uop.is_realized}

F = TypeVar('F', bound=Callable)
@overload
def function(fxn:F, *, precompile:bool=False, precompile_backward:bool=False, allow_implicit:bool=False, grad_fxn:Callable|None=None) -> F: ...
@overload
def function(fxn:None=None, *, precompile:bool=False, precompile_backward:bool=False,
             allow_implicit:bool=False, grad_fxn:Callable|None=None) -> Callable[[F], F]: ...
def function(fxn=None, *, precompile:bool=False, precompile_backward:bool=False, allow_implicit:bool=False, grad_fxn:Callable|None=None):
  if fxn is None:
    return functools.partial(function, precompile=precompile, precompile_backward=precompile_backward,
                             allow_implicit=allow_implicit, grad_fxn=grad_fxn)

  name = getattr(fxn, '__qualname__', None) or type(fxn).__qualname__
  @functools.wraps(fxn)
  def call(*args, **kwargs):
    st = time.perf_counter()

    params = get_state_dict((args, kwargs), tensor_type=(Tensor, UOp)).values()

    # deduplicate input_uops, keeping the first occurrence index for each unique uop
    call_uops: list[UOp] = dedup([u for t in params if (u:=t._uop).device is not None])

    alloc_start = next(UOp.unique_num)
    with Context(ALLOW_DEVICE_USAGE=getenv("DEVICE_IN_FUNCTION_BUG", 0)):
      ret = fxn(*args, **kwargs)
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
    if not allow_implicit:
      implicit_buffers = [x for x in call_uops[num_explicit:] if x.op in {Ops.BUFFER, Ops.ALLOC}]
      if implicit_buffers:
        buf_strs = '\n  '.join(f"{i}: dtype={b.dtype}, size={b.max_numel()}, device={b.device}" for i,b in enumerate(implicit_buffers))
        raise RuntimeError(f"function {name} has {len(implicit_buffers)} implicit buffer(s), but allow_implicit=False\n  {buf_strs}")

    outs = UOp.call_with_outputs(uret.src if isinstance(ret, tuple) else (uret,), *call_uops, grad_fxn=grad_fxn,
                                 precompile=precompile, precompile_backward=precompile_backward)

    if DEBUG >= 2:
      print(f"function {uret.key.hex()[:8]} in {(time.perf_counter()-st)*1000:8.2f} ms: {name}")

    return tuple(Tensor(o) for o in outs) if isinstance(ret, tuple) else Tensor(outs[0])
  return call
