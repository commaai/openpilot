import functools, itertools, time
from typing import Generic, TypeVar, Callable, cast, overload
from tinygrad.helpers import Context, dedup, getenv, DEBUG
from tinygrad.uop.ops import UOp, Ops, graph_rewrite, PatternMatcher, UPat
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_state_dict

def add_to_ctx(ctx, x:UOp):
  ret = x.param_like(len(ctx[0]))
  ctx[0].append(x)
  return ret

pm_transform_unique_const = PatternMatcher([
  # transform unique consts to LUNIQUE
  (UPat(Ops.CONST, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE)), name="x"),
   lambda ctx,x: x.replace(src=(UOp(Ops.LUNIQUE, arg=next(ctx[1])), x.src[1]))),
])

pm_ctx = PatternMatcher([
  (UPat((Ops.BUFFER, Ops.BIND), name="x"), add_to_ctx),
  (UPat((Ops.AFTER, Ops.CONTIGUOUS), name="x"),
   lambda ctx,x: add_to_ctx(ctx,x) if not x.op_in_backward_slice_with_self(Ops.PARAM) and x.op_in_backward_slice_with_self(Ops.BUFFER) else None),
])+pm_transform_unique_const

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
    call_uops: list[UOp] = dedup([u for t in params if not ((u:=(t.uop if isinstance(t, Tensor) else t)).base.op is Ops.CONST and u.device is None)])

    # disable realize/schedule while this is running
    # run it and do surgery later
    with Context(ALLOW_DEVICE_USAGE=getenv("DEVICE_IN_FUNCTION_BUG", 0)):
      _function.depth += 1
      ret = self.fxn(*args, **kwargs)
      _function.depth -= 1
    if isinstance(ret, Tensor):
      uret = ret.uop
    elif isinstance(ret, tuple) and all(isinstance(x, Tensor) for x in ret):
      uret = UOp.maketuple(*[x.uop for x in ret])
    else:
      raise RuntimeError(f"function return type {type(ret)} not supported")

    # replace the known inputs with params (using deduplicated slots)
    subs = {}
    for i,x in enumerate(call_uops): subs[x] = x.param_like(i)
    uret = uret.substitute(subs)

    # add contiguous to call_uops
    #call_uops = [x.contiguous() for x in call_uops]

    # the BUFFERs that are left are the implicit inputs
    num_explicit = len(call_uops)
    uret = graph_rewrite(uret, pm_ctx, (call_uops, itertools.count(0)), bottom_up=True, name="get_implicit_inputs")
    name = getattr(self.fxn, '__qualname__', None) or type(self.fxn).__qualname__
    if not self.allow_implicit:
      implicit_buffers = [x for x in call_uops[num_explicit:] if x.op is Ops.BUFFER]
      if implicit_buffers:
        buf_strs = '\n  '.join(f"{i}: dtype={b.dtype}, size={b.arg}, device={b.device}" for i,b in enumerate(implicit_buffers))
        raise RuntimeError(f"function {name} has {len(implicit_buffers)} implicit buffer(s), but allow_implicit=False\n  {buf_strs}")

    # assign output
    #pbuffer = uret.param_like(len(call_uops))
    #assigned = pbuffer.assign(uret).sink()
    #buffer = UOp.new_buffer(pbuffer.device, pbuffer.size, pbuffer.dtype).reshape(uret.shape)
    #call = assigned.call(*call_uops, buffer, name=name)
    #ret = buffer.after(call)

    fret = uret.call(*call_uops, grad_fxn=self.grad_fxn, name=name, precompile=self.precompile,
                     precompile_backward=self.precompile_backward)

    if DEBUG >= 2:
      #signature = [(x._shape, x.dtype, x.device) for x in call_uops]
      print("  "*_function.depth+f"function {uret.key.hex()[:8]} in {(time.perf_counter()-st)*1000:8.2f} ms: {name}") # with sig {signature}")

    if isinstance(ret, tuple):
      return cast(ReturnType, tuple(Tensor(fret.gettuple(i)) for i in range(len(ret))))
    else:
      return cast(ReturnType, Tensor(fret.gettuple(0)))

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
