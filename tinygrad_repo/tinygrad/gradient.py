from typing import cast, Iterator
import math, functools, dataclasses
from tinygrad.dtype import dtypes, sum_acc_dtype
from tinygrad.ops import UOp, PatternMatcher, UPat, Ops, all_metadata
from tinygrad.helpers import argsort

def reduce_gradient(ctx:UOp, ret:UOp):
  if ret.arg[0] == Ops.ADD: return (ctx.expand(ret.src[0].shape),)
  if ret.arg[0] == Ops.MAX:
    max_is_1s = ret.src[0].ne(ret.expand(ret.src[0].shape)).ne(ret.src[0].const_like(1).cast(dtypes.bool)).cast(ctx.dtype)
    div = max_is_1s.r(Ops.ADD, ret.arg[1]).expand(ret.src[0].shape)
    return ((max_is_1s/div) * ctx.expand(ret.src[0].shape),)
  if ret.arg[0] == Ops.MUL: return ((ctx * ret).expand(ret.src[0].shape) / ret.src[0],)

# ctx is grad_output
pm_gradient = PatternMatcher([
  (UPat(Ops.CAST, name="ret"), lambda ctx, ret: (ctx.cast(ret.src[0].dtype),)),
  (UPat(Ops.RECIP, name="ret"), lambda ctx, ret: (-ctx * ret * ret,)),
  (UPat(Ops.SIN, name="ret"), lambda ctx, ret: ((math.pi/2 - ret.src[0]).sin() * ctx,)),
  (UPat(Ops.LOG2, name="ret"), lambda ctx, ret: (ctx / (ret.src[0] * math.log(2)),)),
  (UPat(Ops.EXP2, name="ret"), lambda ctx, ret: (ret * ctx * math.log(2),)),
  (UPat(Ops.SQRT, name="ret"), lambda ctx, ret: (ctx / (ret*2),)),
  (UPat((Ops.CMPLT, Ops.CMPNE)), lambda: (None, None)),
  (UPat(Ops.ADD), lambda ctx: (ctx, ctx)),
  (UPat(Ops.POW, name="ret"), lambda ctx, ret:
   (ret.src[0].eq(0).where(ret.src[1].eq(0).where(ret.src[1], ret.src[1]*math.inf), ctx*ret*ret.src[1]/ret.src[0]),
    ret.src[0].eq(0).where((ret.src[1]<0).where(ret.const_like(-math.inf), ret.const_like(0)), ctx*ret*ret.src[0].log2()*math.log(2.0)))),
  (UPat(Ops.MAX, name="ret"), lambda ctx, ret: ((ret.src[0]>ret.src[1]).where(ctx, (ret.src[0]!=ret.src[1]).where(ctx.const_like(0), ctx * 0.5)),
                                                (ret.src[0]<ret.src[1]).where(ctx, (ret.src[0]!=ret.src[1]).where(ctx.const_like(0), ctx * 0.5)))),
  (UPat(Ops.MUL, name="ret"), lambda ctx, ret: (ret.src[1]*ctx, ret.src[0]*ctx)),
  (UPat(Ops.WHERE, name="ret"), lambda ctx, ret: (None, ret.src[0].where(ctx, ctx.const_like(0)), ret.src[0].where(ctx.const_like(0), ctx))),
  (UPat(Ops.REDUCE_AXIS, name="ret"), reduce_gradient),
  (UPat(Ops.CONTIGUOUS), lambda ctx: (ctx,)),
  (UPat(Ops.CONTIGUOUS_BACKWARD), lambda ctx: (ctx.contiguous(),)),
  (UPat(Ops.RESHAPE, name="ret"), lambda ctx, ret: (ctx.reshape(ret.src[0].shape),)),
  (UPat(Ops.PERMUTE, name="ret"), lambda ctx, ret: (ctx.permute(argsort(ret.arg)),)),
  (UPat(Ops.PAD, name="ret"), lambda ctx, ret: (ctx.shrink(tuple([(p[0], s+p[0]) for s,p in zip(ret.src[0].shape, ret.arg)])),)),
  (UPat(Ops.SHRINK, name="ret"), lambda ctx, ret: (ctx.pad(tuple([(p[0], s-p[1]) for s,p in zip(ret.src[0].shape, ret.arg)])),)),
  (UPat(Ops.FLIP, name="ret"), lambda ctx, ret: (ctx.flip(ret.arg),)),
  # TODO: this cast can be removed by putting the casts around the EXPAND
  (UPat(Ops.EXPAND, name="ret"), lambda ctx, ret:
    (ctx.cast(sum_acc_dtype(ctx.dtype)).r(Ops.ADD, tuple(i for i,(si,so) in enumerate(zip(ret.src[0].shape, ret.arg)) if si!=so)).cast(ctx.dtype),)),
  (UPat(Ops.MULTI, name="ret"), lambda ctx, ret: ctx.shard(ret.device, ret.axis).src),
  # there's no gradient for bitcast
  (UPat(Ops.BITCAST), lambda ctx: (None,)),
])

# copied from tensor.py, get relevant toposort of gradients
def _deepwalk(root:UOp, targets:set[UOp]) -> list[UOp]:
  @functools.lru_cache(None)
  def is_in_target_path(x:UOp) -> bool: return any(u in targets or is_in_target_path(u) for u in x.src)
  def _walk(node:UOp, visited:set[UOp]) -> Iterator[UOp]:
    visited.add(node)
    if node.op is Ops.DETACH: return
    if is_in_target_path(node):
      for i in node.src:
        if i not in visited: yield from _walk(i, visited)
      yield node
  return list(_walk(root, set()))

def compute_gradient(root:UOp, root_grad:UOp, targets:set[UOp]) -> dict[UOp, UOp]:
  grads = {root: root_grad}
  for t0 in reversed(_deepwalk(root, targets)):
    if t0 not in grads: continue
    lgrads: tuple[UOp|None, ...]|None = cast(tuple[UOp, ...]|None, pm_gradient.rewrite(t0, ctx=grads[t0]))
    if lgrads is None: raise RuntimeError(f"failed to compute gradient for {t0.op}\n\nin {str(t0)[0:1000]}...")
    assert len(lgrads) == len(t0.src), f"got {len(lgrads)} gradient, expected {len(t0.src)}"
    for k,v in zip(t0.src, lgrads):
      if v is None: continue
      if k in grads: grads[k] = grads[k] + v
      else: grads[k] = v
      if (forward_metadata:=all_metadata.get(t0)) is not None: all_metadata[v] = dataclasses.replace(forward_metadata, backward=True)
  return grads
