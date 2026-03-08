from typing import cast
import math, dataclasses
from tinygrad.uop.ops import UOp, PatternMatcher, UPat, Ops, all_metadata
from tinygrad.helpers import argsort

def reduce_gradient(ctx:UOp, ret:UOp, op:Ops):
  def broadcast_to_input(x): return x.reshape(x.shape+(1,)*(len(ret.src[0].shape)-len(x.shape))).expand(ret.src[0].shape)
  if op == Ops.ADD: return (broadcast_to_input(ctx),)
  if op == Ops.MAX:
    assert ret.op is Ops.REDUCE_AXIS, "only works on REDUCE_AXIS"
    mask = ret.src[0].eq(broadcast_to_input(ret)).cast(ctx.dtype)
    count = mask.r(Ops.ADD, ret.arg[1])
    return ((mask/broadcast_to_input(count)) * broadcast_to_input(ctx),)
  if op == Ops.MUL: return (broadcast_to_input(ctx * ret) / ret.src[0],)

def call_gradient(ctx:UOp, k:UOp):
  if k.arg is not None: return (None,) + k.arg(ctx, k)
  # auto-differentiate the function
  fxn, args = k.src[0], k.src[1:]
  params = sorted([x for x in fxn.toposort() if x.op == Ops.PARAM], key=lambda x: x.arg)
  grads = compute_gradient(fxn, ctx, set(params))
  subst = dict(zip(params, args))
  return (None,) + tuple(grads[p].substitute(subst) if p in grads else None for p in params)

# ctx is grad_output
pm_gradient = PatternMatcher([
  (UPat(Ops.CAST, name="ret"), lambda ctx, ret: (ctx.cast(ret.src[0].dtype),)),
  (UPat(Ops.RECIPROCAL, name="ret"), lambda ctx, ret: (-ctx * ret * ret,)),
  (UPat(Ops.SIN, name="ret"), lambda ctx, ret: ((math.pi/2 - ret.src[0]).sin() * ctx,)),
  (UPat(Ops.LOG2, name="ret"), lambda ctx, ret: (ctx / (ret.src[0] * math.log(2)),)),
  (UPat(Ops.EXP2, name="ret"), lambda ctx, ret: (ret * ctx * math.log(2),)),
  (UPat(Ops.SQRT, name="ret"), lambda ctx, ret: (ctx / (ret*2),)),
  (UPat((Ops.CMPLT, Ops.CMPNE)), lambda: (None, None)),
  (UPat(Ops.ADD), lambda ctx: (ctx, ctx)),
  (UPat(Ops.POW, name="ret", src=(UPat.var("b"), UPat.var("e"))), lambda ctx, ret, b, e:
    (ctx * (b.eq(0)&e.eq(0)).where(e, e*b.pow(e-1)), ctx * b.eq(0).where((e<0).where(ret.const_like(-math.inf), 0), ret*b.log2()*math.log(2.0)))),
  (UPat(Ops.MAX, src=(UPat.var("x"), UPat.var("y"))), lambda ctx, x, y:
    ((x>y).where(ctx, (x.eq(y)).where(ctx * 0.5, 0)), (x<y).where(ctx, (x.eq(y)).where(ctx * 0.5, 0)))),
  (UPat(Ops.MUL, name="ret"), lambda ctx, ret: (ret.src[1]*ctx, ret.src[0]*ctx)),
  (UPat(Ops.WHERE, name="ret"), lambda ctx, ret: (None, ret.src[0].where(ctx, ctx.const_like(0)), ret.src[0].where(ctx.const_like(0), ctx))),
  (UPat(Ops.REDUCE_AXIS, name="ret"), lambda ctx, ret: reduce_gradient(ctx, ret, ret.arg[0])),
  (UPat(Ops.REDUCE, name="ret"), lambda ctx, ret: reduce_gradient(ctx, ret, ret.arg) + (None,)*(len(ret.src)-1)),
  (UPat(Ops.CONTIGUOUS), lambda ctx: (ctx,)),
  (UPat(Ops.CONTIGUOUS_BACKWARD), lambda ctx: (ctx.contiguous(),)),
  (UPat(Ops.RESHAPE, name="ret"), lambda ctx, ret: (ctx.reshape(ret.src[0].shape), None)),
  (UPat(Ops.EXPAND, name="ret"), lambda ctx, ret: (ctx.r(Ops.ADD,tuple(i for i,(s,n) in enumerate(zip(ret.src[0].shape, ret.shape)) if s!=n)), None)),
  (UPat(Ops.PAD, name="ret"), lambda ctx, ret: (ctx.shrink(tuple([(p[0], s+p[0]) for s,p in zip(ret.src[0].shape, ret.marg)])), None, None)),
  (UPat(Ops.SHRINK, name="ret"), lambda ctx, ret: (ctx.pad(tuple([(p[0], s-p[1]) for s,p in zip(ret.src[0].shape, ret.marg)])), None, None)),
  (UPat(Ops.PERMUTE, name="ret"), lambda ctx, ret: (ctx.permute(argsort(ret.marg)),)),
  (UPat(Ops.FLIP, name="ret"), lambda ctx, ret: (ctx.flip([i for i,x in enumerate(ret.marg) if x]),)),
  (UPat(Ops.COPY, name="ret"), lambda ctx, ret: (ctx.copy_to_device(ret.src[0].device), None)),
  (UPat(Ops.MULTI, name="ret"), lambda ctx, ret: ctx.shard(ret.device, ret.axis).src),
  # NOTE: this is only correct when the KERNEL has a single output
  (UPat(Ops.AFTER), lambda ctx: (ctx, ctx)),
  (UPat(Ops.CUSTOM_KERNEL, name="k"), lambda ctx, k: k.arg.grad_fxn(ctx, k)),
  # gradient on CALL: use provided grad_fxn or auto-differentiate
  (UPat(Ops.CALL, name="k"), call_gradient),
  # there's no gradient for bitcast
  (UPat(Ops.BITCAST), lambda: (None,)),
])

def _deepwalk(root:UOp, targets:set[UOp]) -> list[UOp]:
  # compute the target path (top down)
  in_target_path: dict[UOp, bool] = {}
  for u in root.toposort(): in_target_path[u] = any(x in targets or in_target_path[x] for x in u.src)
  # don't flow through DETACH/ASSIGN or anything not in target path
  return list(root.toposort(lambda node: node.op not in {Ops.DETACH, Ops.ASSIGN} and in_target_path[node]))

def compute_gradient(root:UOp, root_grad:UOp, targets:set[UOp]) -> dict[UOp, UOp]:
  grads = {root: root_grad}
  for t0 in reversed(_deepwalk(root, targets)):
    if t0 not in grads: continue
    lgrads: tuple[UOp|None, ...]|None = cast(tuple[UOp|None, ...]|None, pm_gradient.rewrite(t0, ctx=grads[t0]))
    if lgrads is None: raise RuntimeError(f"failed to compute gradient for {t0.op}\n\nin {str(t0)[0:1000]}...")
    assert len(lgrads) == len(t0.src), f"got {len(lgrads)} gradient, expected {len(t0.src)}"
    for k,v in zip(t0.src, lgrads):
      if v is None: continue
      if k in grads: grads[k] = grads[k] + v
      else: grads[k] = v
      if len(forward_metadata:=all_metadata.get(t0, ())):
        backward_metadata = tuple(dataclasses.replace(x, backward=True) for x in forward_metadata)
        # we add the backward metadata to everything new in the graph
        for bw_uop in v.toposort(lambda x: x not in (t0, *t0.src, grads[t0])):
          all_metadata[bw_uop] = all_metadata.get(bw_uop, ())+backward_metadata
  # end any ranges on grads with a reduce sum
  for k,v in grads.items():
    if len(v.ranges):
      grads[k] = v.reduce(*v.ranges, arg=Ops.ADD)
  return grads
