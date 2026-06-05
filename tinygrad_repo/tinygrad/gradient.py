from typing import cast
import math, dataclasses, itertools
from tinygrad.uop.ops import UOp, PatternMatcher, UPat, Ops, all_metadata, graph_rewrite
from tinygrad.helpers import argsort
from tinygrad.dtype import sum_acc_dtype

def reduce_gradient(ctx:UOp, ret:UOp, op:Ops):
  def broadcast_to_input(x): return x.reshape(x.shape+(1,)*(len(ret.src[0].shape)-len(x.shape))).expand(ret.src[0].shape)
  if op == Ops.ADD: return (broadcast_to_input(ctx),)
  if op == Ops.MAX:
    assert ret.op is Ops.REDUCE, "only works on REDUCE"
    mask = ret.src[0].eq(broadcast_to_input(ret)).cast(ctx.dtype)
    count = mask._rop(Ops.ADD, ret.arg[1])
    return ((mask/broadcast_to_input(count)) * broadcast_to_input(ctx),)
  if op == Ops.MUL: return (broadcast_to_input(ctx * ret) / ret.src[0],)

def _compact_params(body:UOp, all_args:tuple[UOp, ...]) -> tuple[UOp, tuple[UOp, ...]]:
  """Remove unused PARAMs from body and return compacted (body, args)."""
  used = sorted({p.arg: p for p in body.toposort() if p.op is Ops.PARAM}.items())
  return body.substitute({p: p.replace(arg=j) for j,(_, p) in enumerate(used)}, walk=True), tuple(all_args[i] for i,_ in used)

def call_gradient(ctx:UOp, k:UOp, needed:set[int]) -> tuple[UOp|None, ...]:
  fxn, args = k.src[0], k.src[1:]
  if k.arg.grad_fxn is not None:
    # put const on a device, also TODO why do we still have NOOP...
    def on_dev(g, i): return g.clone(device=args[i].device if k.op is Ops.CALL else k.device) if g.device is None else g
    if ctx.op is Ops.TUPLE:
      real = [on_dev(g, i) for i,g in enumerate(ctx.src) if g.op is not Ops.NOOP]
      return (None,) + (k.arg.grad_fxn(*real, call=k) if len(real) > 1 else k.arg.grad_fxn(real[0], k))
    return (None,) + k.arg.grad_fxn(on_dev(ctx, 0), k)
  assert fxn.op is Ops.TUPLE, f"expected TUPLE body for gradient, got {fxn.op}"
  params = {x.arg:x for x in fxn.toposort(enter_calls=False) if x.op == Ops.PARAM}
  grad_args = ctx.src
  root_grad = UOp(Ops.TUPLE, src=tuple(UOp(Ops.NOOP) if g.op is Ops.NOOP else
    g if g.base.op is Ops.CONST and g.device is None else g.param_like(len(args)+i) for i,g in enumerate(grad_args)))
  grads = compute_gradient(fxn, root_grad, set(params.values()))
  # for precompiled calls, substitute forward outputs with params so intermediates aren't recomputed
  fwd_subs = {src: src.param_like(len(args)+len(grad_args)+i) for i, src in enumerate(fxn.src)} if k.arg.precompile else {}
  fwd_outs = tuple(k.gettuple(i) for i in range(len(fxn.src))) if k.arg.precompile else ()
  # collect needed gradient bodies, compact unused params, create a single backward CALL
  grad_bodies = [(i, grads[p]) for i in needed if (p:=params.get(i)) is not None and p in grads]
  bwd_body = UOp.maketuple(*(gb for _, gb in grad_bodies)).substitute(fwd_subs, walk=True)
  bwd_body, compact_args = _compact_params(bwd_body, (*args, *grad_args, *fwd_outs))
  # TODO: is this okay here?
  from tinygrad.function import pm_transform_unique_const
  bwd_body = graph_rewrite(bwd_body, pm_transform_unique_const, ctx=(None, itertools.count(0)))
  bwd_call = bwd_body.call(*compact_args, name=(k.arg.name or "")+"_backward", precompile=k.arg.precompile_backward)
  gb_map = {i: idx for idx, (i, _) in enumerate(grad_bodies)}
  return (None,) + tuple(bwd_call.gettuple(gb_map[i]) if i in gb_map else None for i in range(len(args)))

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
    (ctx * (b.eq(0)&e.eq(0)).where(e, e*b.pow(e-1)), ctx * b.eq(0).where((e<0).where(ret.const_like(-math.inf), 0), ret*b.log2()*math.log(2.0)))),
  (UPat(Ops.MAX, src=(UPat.var("x"), UPat.var("y"))), lambda ctx, x, y:
    ((x>y).where(ctx, (x.eq(y)).where(ctx * 0.5, 0)), (x<y).where(ctx, (x.eq(y)).where(ctx * 0.5, 0)))),
  (UPat(Ops.MUL, name="ret"), lambda ctx, ret: (ret.src[1]*ctx, ret.src[0]*ctx)),
  (UPat(Ops.WHERE, name="ret"), lambda ctx, ret: (None, ret.src[0].where(ctx, ctx.const_like(0)), ret.src[0].where(ctx.const_like(0), ctx))),
  (UPat(Ops.REDUCE, name="ret"), lambda ctx, ret: reduce_gradient(ctx, ret, ret.arg[0])),
  (UPat(Ops.CONTIGUOUS), lambda ctx: (ctx,)),
  (UPat(Ops.CONTIGUOUS_BACKWARD), lambda ctx: (ctx.contiguous(),)),
  (UPat(Ops.RESHAPE, name="ret"), lambda ctx, ret: (ctx.reshape(ret.src[0].shape), None)),
  (UPat(Ops.EXPAND, name="ret"), lambda ctx, ret:
    (ctx.cast(sum_acc_dtype(ctx.dtype))._rop(Ops.ADD, tuple(i for i,(s,n) in enumerate(zip(ret.src[0].shape, ret.shape)) if s!=n))
     .cast(ctx.dtype), None)),
  (UPat(Ops.PAD, name="ret"), lambda ctx, ret: (ctx.shrink(tuple([(p[0], s+p[0]) for s,p in zip(ret.src[0].shape, ret.marg)])), None, None)),
  (UPat(Ops.SHRINK, name="ret"), lambda ctx, ret: (ctx.pad(tuple([(p[0], s-p[1]) for s,p in zip(ret.src[0].shape, ret.marg)])), None, None)),
  (UPat(Ops.PERMUTE, name="ret"), lambda ctx, ret: (ctx.permute(argsort(ret.marg)),)),
  (UPat(Ops.FLIP, name="ret"), lambda ctx, ret: (ctx.flip([i for i,x in enumerate(ret.marg) if x]),)),
  (UPat(Ops.COPY, name="ret"), lambda ctx, ret: (ctx.copy_to_device(ret.src[0].device), None)),
  (UPat(Ops.MULTI, name="ret"), lambda ctx, ret: ctx.shard(ret.device, ret.axis).src),
  (UPat(Ops.TUPLE), lambda ctx: ctx.src),
  (UPat(Ops.AFTER, src=(UPat.var("d"), UPat(Ops.CALL, name="k"))), lambda ctx, d, k:
    (ctx, UOp.maketuple(*(ctx if i == k.src.index(d)-1 else UOp(Ops.NOOP) for i in range(len(k.src)-1))))),
  # clone/assign gradient passes through to val
  (UPat(Ops.AFTER, src=(UPat(), UPat(Ops.STORE))), lambda ctx: (None, ctx)),
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
    # GETTUPLE: accumulate gradient into a TUPLE UOp on the FUNCTION, process when we hit the FUNCTION
    if t0.op is Ops.GETTUPLE:
      k = t0.src[0]  # the FUNCTION
      assert k.op is Ops.FUNCTION and k.src[0].op is Ops.TUPLE
      n_outputs = len(k.src[0].src)
      prev = grads[k].src if k in grads else tuple(UOp(Ops.NOOP) for _ in range(n_outputs))
      grads[k] = UOp.maketuple(*(prev[i] + grads[t0] if i == t0.arg and prev[i].op is not Ops.NOOP else
                                 grads[t0] if i == t0.arg else prev[i] for i in range(n_outputs)))
      continue
    # FUNCTION/CALL: pass needed param set so backward only computes required gradients
    # (FUNCTION uses implicit TUPLE gradient or grad_fxn; CALL requires an explicit grad_fxn)
    if t0.op in {Ops.FUNCTION, Ops.CALL}:
      needed = {i for i, arg in enumerate(t0.src[1:]) if arg in targets or in_target_path.get(arg, False)}
      lgrads:tuple[UOp|None, ...]|None = call_gradient(grads[t0], t0, needed)
    else:
      lgrads = cast(tuple[UOp|None, ...]|None, pm_gradient.rewrite(t0, ctx=grads[t0]))
    if lgrads is None: raise RuntimeError(f"failed to compute gradient for {t0.op}\n\nin {str(t0)[0:1000]}...")
    assert len(lgrads) == len(t0.src), f"got {len(lgrads)} gradient, expected {len(t0.src)}"
    for k,v in zip(t0.src, lgrads):
      if v is None: continue
      if k in grads and grads[k].op is not Ops.NOOP:
        if v.op is Ops.TUPLE and grads[k].op is Ops.TUPLE:
          grads[k] = UOp.maketuple(*(p + n if (p.op is not Ops.NOOP and n.op is not Ops.NOOP) else
                                     n if p.op is Ops.NOOP else p for p, n in zip(grads[k].src, v.src)))
        else: grads[k] = grads[k] + v
      else: grads[k] = v
      if len(forward_metadata:=all_metadata.get(t0, ())):
        backward_metadata = tuple(dataclasses.replace(x, backward=True) for x in forward_metadata)
        # we add the backward metadata to everything new in the graph
        for bw_uop in v.toposort(lambda x: x not in (t0, *t0.src, grads[t0])):
          all_metadata[bw_uop] = all_metadata.get(bw_uop, ())+backward_metadata
  return grads
