from typing import cast
import math, dataclasses
from tinygrad.uop.ops import UOp, PatternMatcher, UPat, Ops, all_metadata, broadcast_axes
from tinygrad.helpers import argsort
from tinygrad.dtype import dtypes, sum_acc_dtype
from tinygrad.function import renumber_invalid_outputs

def reduce_gradient(ctx:UOp, ret:UOp, op:Ops):
  if op == Ops.ADD: return (ctx._broadcast_to(ret.src[0].shape),)
  if op == Ops.MAX:
    # count the ties in the acc dtype, the count can overflow the gradient dtype
    mask = ret.src[0].eq(ret).cast(sum_acc_dtype(ctx.dtype))
    return ((mask/mask._rop(Ops.ADD, tuple(range(ret.arg[1])))).cast(ctx.dtype) * ctx,)
  if op == Ops.MUL:
    # d(prod x)/dx_j = prod_{i!=j} x_i: ret/x_j whenever x_j != 0 (any zero makes ret 0), else the product of the others
    safe_x, axes = (is_zero:=(x:=ret.src[0]).eq(0)).where(1, x), tuple(range(ret.arg[1]))
    zero_count = is_zero.cast(sum_acc_dtype(is_zero.dtype))._rop(Ops.ADD, axes)
    return (ctx * is_zero.where(zero_count.eq(1).where(safe_x._rop(Ops.MUL, axes), 0), ret/safe_x),)

def call_gradient(ctx:UOp, k:UOp, needed:set[int]) -> tuple[UOp|None, ...]:
  fxn, args = k.body, k.src[1:]
  outputs = {st.src[0].unsharded_base.arg.slot:st for st in fxn.src
             if st.op is Ops.STORE and st.src[0].unsharded_base.op is Ops.PARAM} if fxn.op is Ops.SINK and fxn.arg is None else {}
  if k.arg.grad_fxn is not None:
    real = [g.clone(device=args[i].device) if g.device is None else g
            for i,g in enumerate(ctx.src if ctx.op is Ops.SINK else (ctx,)) if g.op is not Ops.NOOP]
    git = iter(k.arg.grad_fxn(*real, call=k) if len(real) > 1 else k.arg.grad_fxn(real[0], k))
    return (None,) + tuple(None if i in outputs else next(git) for i in range(len(args)))
  assert outputs, f"expected a CALL with output STOREs or a grad_fxn, got {fxn.op}"
  params = {p.arg.slot:p for p in fxn.toposort(enter_calls=False) if p.op is Ops.PARAM and p.arg.slot >= 0}
  grad_args = tuple(ctx.src[i] for i in outputs)
  root_grad = UOp.sink(*[g if g.device is None else g.param_like(len(args)+i) for i,g in enumerate(grad_args)])
  grads = compute_gradient(UOp.sink(*[st.src[1] for st in outputs.values()]), root_grad, set(params.values()))
  grad_bodies = {i:grads[p].view_as(args[i].shard_shape, args[i].axis)
                 for i in needed - outputs.keys() if (p:=params.get(i)) is not None and p in grads}
  bwd_body = UOp.sink(*grad_bodies.values())
  # Reuse the output PARAMs for saved forward values instead of recomputing them.
  if k.arg.precompile:
    bwd_body = bwd_body.substitute({st.src[1]:st.src[0] for st in outputs.values()}, walk=True)
    args = tuple(a.after(k) if i in outputs else a for i,a in enumerate(args))
  args += grad_args
  bwd_body = renumber_invalid_outputs(bwd_body)
  # Compact this scope's PARAMs, capturing free Variables as inputs.
  used = sorted((p for p in bwd_body.toposort(enter_calls=False) if p.op is Ops.PARAM), key=lambda p:p.arg.slot)
  bwd_args = [p if p.arg.slot < 0 else args[p.arg.slot] for p in used]
  bwd_body = bwd_body.substitute({p:p.replace(arg=dataclasses.replace(p.arg, slot=i, name=None, val=None))
                                  for i,p in enumerate(used)}, walk=True)
  bwd_outs = dict(zip(grad_bodies, UOp.call_with_outputs(bwd_body.src, *bwd_args,
                                                       name=(k.arg.name or "")+"_backward", precompile=k.arg.precompile_backward)))
  return (None,) + tuple(bwd_outs.get(i) for i in range(len(k.src)-1))

def partial_store_gradient(ctx:UOp, dest:UOp, view:UOp):
  # A write through a non-overlapping view replaces only that region of the returned state.
  path, base = [], view
  while base is not dest and base.op in {Ops.RESHAPE, Ops.SHRINK, Ops.PERMUTE, Ops.FLIP, Ops.PAD}:
    path.append(base)
    base = base.src[0]
  if base is not dest: return None
  grad = ctx
  for mop in reversed(path): grad = mop.replace(src=(grad,)+mop.src[1:])
  mask = grad.const_like(1)
  for mop in path: mask = pm_gradient.rewrite(mop, ctx=mask)[0]
  return mask.cast(dtypes.bool).where(0, ctx), grad

# ctx is grad_output
pm_gradient = PatternMatcher([
  (UPat(Ops.CAST, name="ret"), lambda ctx, ret: (ctx.cast(ret.src[0].dtype),)),
  (UPat(Ops.RECIPROCAL, name="ret"), lambda ctx, ret: (-ctx * ret * ret,)),
  (UPat(Ops.SIN, name="ret"), lambda ctx, ret: ((math.pi/2 - ret.src[0]).sin() * ctx,)),
  (UPat(Ops.LOG2, name="ret"), lambda ctx, ret: (ctx / (ret.src[0] * math.log(2)),)),
  (UPat(Ops.EXP2, name="ret"), lambda ctx, ret: (ret * ctx * math.log(2),)),
  (UPat(Ops.SQRT, name="ret"), lambda ctx, ret: (ctx / (ret*2),)),
  (UPat(Ops.TRUNC), lambda ctx: (ctx.const_like(0),)),
  (UPat((Ops.CMPLT, Ops.CMPNE)), lambda: (None, None)),
  (UPat(Ops.ADD), lambda ctx: (ctx, ctx)),
  (UPat(Ops.POW, name="ret", src=(UPat.var("b"), UPat.var("e"))), lambda ctx, ret, b, e:
    (ctx * e.eq(0).where(e, e*b.pow(e-1)), ctx * b.eq(0).where((e<0).where(ret.const_like(-math.inf), 0), ret*b.log2()*math.log(2.0)))),
  (UPat(Ops.MAX, src=(UPat.var("x"), UPat.var("y"))), lambda ctx, x, y:
    ((x>y).where(ctx, (x.eq(y)).where(ctx * 0.5, 0)), (x<y).where(ctx, (x.eq(y)).where(ctx * 0.5, 0)))),
  (UPat(Ops.MUL, name="ret"), lambda ctx, ret: (ret.src[1]*ctx, ret.src[0]*ctx)),
  (UPat(Ops.WHERE, name="ret"), lambda ctx, ret: (None, ret.src[0].where(ctx, ctx.const_like(0)), ret.src[0].where(ctx.const_like(0), ctx))),
  (UPat(Ops.REDUCE, name="ret"), lambda ctx, ret: reduce_gradient(ctx, ret, ret.arg[0])),
  (UPat(Ops.CONTIGUOUS_BACKWARD), lambda ctx: (ctx.contiguous(),)),
  (UPat(Ops.STAGE), lambda ctx: (ctx,)),
  (UPat(Ops.RESHAPE, name="ret"), lambda ctx, ret: (ctx.reshape(ret.src[0].shape), None)),
  (UPat(Ops.EXPAND), lambda ctx: (ctx, None)),
  (UPat(Ops.PAD, name="ret"), lambda ctx, ret: (ctx.shrink(tuple([(p[0], s+p[0]) for s,p in zip(ret.src[0].shape, ret.marg)])), None, None)),
  (UPat(Ops.SHRINK, name="ret"), lambda ctx, ret: (ctx.pad(tuple([(p[0], s-p[0]-p[1]) for s,p in zip(ret.src[0].shape, ret.marg)])), None, None)),
  (UPat(Ops.PERMUTE, name="ret"), lambda ctx, ret: (ctx.permute(argsort(ret.marg)),)),
  (UPat(Ops.FLIP, name="ret"), lambda ctx, ret: (ctx.flip([i for i,x in enumerate(ret.marg) if x]),)),
  (UPat(Ops.STACK, name="ret"), lambda ctx, ret: tuple(ctx[i] for i in range(len(ret.src)))),
  (UPat(Ops.COPY, name="ret"), lambda ctx, ret: (ctx.copy_to_device(ret.src[0].device),) + (None,) * (len(ret.src)-1)),
  (UPat(Ops.UNSHARD, name="ret"), lambda ctx, ret: ctx.shard(ret.device, ret.axis).src),
  (UPat(Ops.SINK), lambda ctx: ctx.src),
  (UPat(Ops.AFTER, src=(UPat.var("d"), UPat(Ops.CALL, name="k"))), lambda ctx, d, k:
    (ctx, UOp.sink(*([ctx if i == k.src.index(d)-1 else UOp(Ops.NOOP) for i in range(len(k.src)-1)])))),
  # ordering-only AFTER: store target is a different buffer, gradient flows straight through to dest
  (UPat(Ops.AFTER, src=(UPat(name="dest"), UPat(Ops.STORE, src=(UPat(name="t"), UPat())))),
   lambda ctx, dest, t: (ctx, None) if t.buf_uop is not dest.buf_uop else None),
  # clone/assign gradient passes through to val
  (UPat(Ops.AFTER, src=(UPat(name="dest"), UPat(Ops.STORE, src=(UPat(name="dest"), UPat())))), lambda ctx,dest: (None, ctx)),
  (UPat(Ops.AFTER, src=(UPat(name="dest"), UPat(Ops.STORE, src=(UPat(name="view"), UPat())))),
   lambda ctx, dest, view: partial_store_gradient(ctx, dest, view)),
  (UPat(Ops.STORE, src=(UPat(), UPat())), lambda ctx: (None, ctx)),
  # there's no gradient for bitcast
  (UPat(Ops.BITCAST), lambda: (None,)),
])

def _deepwalk(root:UOp, targets:set[UOp]) -> tuple[list[UOp], dict[UOp, bool]]:
  # compute the target path (top down)
  in_target_path: dict[UOp, bool] = {}
  root.topovisit(lambda u: any(in_target_path[x] or x in targets for x in u.src), in_target_path)
  # don't flow through DETACH or anything not in target path
  return [node for node in in_target_path if node.op is not Ops.DETACH and in_target_path[node]], in_target_path

def compute_gradient(root:UOp, root_grad:UOp, targets:set[UOp]) -> dict[UOp, UOp]:
  walk, in_target_path = _deepwalk(root, targets)
  grads: dict[UOp, UOp] = {root: root_grad}
  for t0 in reversed(walk):
    if t0 not in grads or grads[t0].op is Ops.NOOP: continue
    # CALL: pass needed param set so backward only computes required gradients
    if t0.op is Ops.CALL:
      needed = {i for i, arg in enumerate(t0.src[1:]) if arg in targets or in_target_path.get(arg, False)}
      lgrads:tuple[UOp|None, ...]|None = call_gradient(grads[t0], t0, needed)
    else:
      lgrads = cast(tuple[UOp|None, ...]|None, pm_gradient.rewrite(t0, ctx=grads[t0]))
    if lgrads is None: raise RuntimeError(f"failed to compute gradient for {t0.op}\n\nin {str(t0)[0:1000]}...")
    assert len(lgrads) == len(t0.src), f"got {len(lgrads)} gradient, expected {len(t0.src)}"
    for k,v in zip(t0.src, lgrads):
      if v is None: continue
      # a shaped edge's gradient is summed to its source's shape
      if k._shape is not None and v._shape is not None and k._shape != v._shape:
        v = v.cast(sum_acc_dtype(v.dtype))._rop(Ops.ADD, broadcast_axes(k.shape, v.shape)).reshape(k.shape).cast(v.dtype)
      if k in grads and grads[k].op is not Ops.NOOP:
        if v.op is Ops.SINK and grads[k].op is Ops.SINK:
          grads[k] = UOp.sink(*[p + n if (p.op is not Ops.NOOP and n.op is not Ops.NOOP) else
                                 n if p.op is Ops.NOOP else p for p, n in zip(grads[k].src, v.src)])
        else: grads[k] = grads[k] + v
      else: grads[k] = v
      if len(forward_metadata:=all_metadata.get(t0, ())):
        backward_metadata = tuple(dataclasses.replace(x, backward=True) for x in forward_metadata)
        # we add the backward metadata to everything new in the graph
        for bw_uop in v.toposort(lambda x: x not in (t0, *t0.src, grads[t0])):
          all_metadata[bw_uop] = all_metadata.get(bw_uop, ())+backward_metadata
  return grads
