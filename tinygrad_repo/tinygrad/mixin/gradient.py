from typing import cast
import math, dataclasses
from tinygrad.uop.ops import UOp, PatternMatcher, UPat, Ops, all_metadata, broadcast_axes
from tinygrad.helpers import argsort
from tinygrad.dtype import sum_acc_dtype
from tinygrad.function import renumber_invalid_outputs

def reduce_gradient(ctx:UOp, ret:UOp, op:Ops):
  if op == Ops.ADD: return (ctx._broadcast_to(ret.src[0].shape),)
  if op == Ops.MAX: return (((mask:=ret.src[0].eq(ret).cast(ctx.dtype))/mask._rop(Ops.ADD, tuple(range(ret.arg[1])))) * ctx,)
  if op == Ops.MUL:
    # d(prod x)/dx_j = prod_{i!=j} x_i: ret/x_j whenever x_j != 0 (any zero makes ret 0), else the product of the others
    safe_x, axes = (is_zero:=(x:=ret.src[0]).eq(0)).where(1, x), tuple(range(ret.arg[1]))
    zero_count = is_zero.cast(sum_acc_dtype(is_zero.dtype))._rop(Ops.ADD, axes)
    return (ctx * is_zero.where(zero_count.eq(1).where(safe_x._rop(Ops.MUL, axes), 0), ret/safe_x),)

def _compact_params(body:UOp, all_args:tuple[UOp, ...]) -> tuple[UOp, tuple[UOp, ...]]:
  """Remove unused PARAMs from body and return compacted (body, args)."""
  # NOTE: don't enter nested calls, their PARAMs are lexical params of the subprogram
  used = sorted({p.arg.slot: p for p in body.toposort(enter_calls=False) if p.op is Ops.PARAM}.items())
  body = body.substitute({p: p.replace(arg=dataclasses.replace(p.arg, slot=j)) for j,(_, p) in enumerate(used)}, walk=True)
  return body, tuple(all_args[i] for i,_ in used)

def call_gradient(ctx:UOp, k:UOp, needed:set[int]) -> tuple[UOp|None, ...]:
  fxn, args = k.src[0], k.src[1:]
  if k.arg.grad_fxn is not None:
    # put const on a device, also TODO why do we still have NOOP...
    def on_dev(g, i): return g.clone(device=args[i].device) if g.device is None else g
    # grads align with the call's src positions (None for the body and for RETURNED outputs, wherever they are)
    def arg_grads(g):
      git = iter(g)
      return (None,) + tuple(next(git) if a.unsharded_base.op is not Ops.RETURNED else None for a in k.src[1:])
    if ctx.op is Ops.SINK:
      real = [on_dev(g, i) for i,g in enumerate(ctx.src) if g.op is not Ops.NOOP]
      return arg_grads(k.arg.grad_fxn(*real, call=k) if len(real) > 1 else k.arg.grad_fxn(real[0], k))
    return arg_grads(k.arg.grad_fxn(on_dev(ctx, 0), k))
  # the RETURNED inputs are the call outputs: their positions in the args get the output gradients from the AFTER rule
  assert fxn.op is Ops.SINK and k.num_returned, f"expected a CALL with RETURNED inputs or a grad_fxn, got {fxn.op}"
  ret_pos = [i for i, a in enumerate(args) if a.unsharded_base.op is Ops.RETURNED]
  # the body stores the outputs into output PARAMs: the values are the stored values in slot order
  values = UOp.sink(*[st.src[1] for st in fxn.src if st.op is Ops.STORE])
  params = {x.arg.slot:x for x in fxn.toposort(enter_calls=False) if x.op == Ops.PARAM}
  # grads are collected at the flat param storage: reshape to each arg's view (max view shrunk to symbolic)
  def shaped_grad(grad:UOp, i:int) -> UOp:
    a = args[i]
    return grad.view_as(a.shard_shape, a.axis) if a.axis is not None and isinstance(a.device, tuple) else grad.view_as(a._shape)
  grad_args = tuple(ctx.src[i] for i in ret_pos)
  root_grad = UOp.sink(*[UOp(Ops.NOOP) if g.op is Ops.NOOP else
    g if g.device is None else g.param_like(len(args)+i) for i,g in enumerate(grad_args)])
  grads = compute_gradient(values, root_grad, set(params.values()))
  # for precompiled calls, substitute forward outputs with params so intermediates aren't recomputed
  fwd_subs = {src: src.param_like(len(args)+len(grad_args)+i) for i, src in enumerate(values.src)} if k.arg.precompile else {}
  fwd_outs = k.returned_outputs if k.arg.precompile else ()
  # collect needed gradient bodies, compact unused params, create a single backward CALL
  grad_bodies = [(i, shaped_grad(grads[p], i)) for i in needed if (p:=params.get(i)) is not None and p in grads]
  bwd_body = UOp.sink(*[gb for _, gb in grad_bodies]).substitute(fwd_subs, walk=True)
  bwd_body = renumber_invalid_outputs(bwd_body)
  # NOTE: args includes the RETURNED inputs so the param slots above line up; they are unused and compacted away
  bwd_body, compact_args = _compact_params(bwd_body, (*args, *grad_args, *fwd_outs))
  bwd_outs = UOp.call_outputs(bwd_body.src, *compact_args, name=(k.arg.name or "")+"_backward",
                              precompile=k.arg.precompile_backward).returned_outputs
  gb_map = {i: idx for idx, (i, _) in enumerate(grad_bodies)}
  # align gradients with the original source positions: None at RETURNED positions, gradients elsewhere
  ret_set = set(ret_pos)
  return (None,) + tuple(None if i in ret_set else (bwd_outs[gb_map[i]] if i in gb_map else None) for i in range(len(args)))

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
  (UPat(Ops.CONTIGUOUS), lambda ctx: (ctx,)),
  (UPat(Ops.CONTIGUOUS_BACKWARD), lambda ctx: (ctx.contiguous(),)),
  (UPat(Ops.RESHAPE, name="ret"), lambda ctx, ret: (ctx.reshape(ret.src[0].shape), None)),
  (UPat(Ops.EXPAND), lambda ctx: (ctx, None)),
  (UPat(Ops.PAD, name="ret"), lambda ctx, ret: (ctx.shrink(tuple([(p[0], s+p[0]) for s,p in zip(ret.src[0].shape, ret.marg)])), None, None)),
  (UPat(Ops.SHRINK, name="ret"), lambda ctx, ret: (ctx.pad(tuple([(p[0], s-p[0]-p[1]) for s,p in zip(ret.src[0].shape, ret.marg)])), None, None)),
  (UPat(Ops.PERMUTE, name="ret"), lambda ctx, ret: (ctx.permute(argsort(ret.marg)),)),
  (UPat(Ops.FLIP, name="ret"), lambda ctx, ret: (ctx.flip([i for i,x in enumerate(ret.marg) if x]),)),
  (UPat(Ops.STACK, name="ret"), lambda ctx, ret: tuple(ctx[i] for i in range(len(ret.src)))),
  (UPat(Ops.COPY, name="ret"), lambda ctx, ret: (ctx.copy_to_device(ret.src[0].device),)),
  (UPat(Ops.UNSHARD, name="ret"), lambda ctx, ret: ctx.shard(ret.device, ret.axis).src),
  (UPat(Ops.SINK), lambda ctx: ctx.src),
  (UPat(Ops.AFTER, src=(UPat.var("d"), UPat(Ops.CALL, name="k"))), lambda ctx, d, k:
    (ctx, UOp.sink(*([ctx if i == k.src.index(d)-1 else UOp(Ops.NOOP) for i in range(len(k.src)-1)])))),
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
    # CALL: pass needed param set so backward only computes required gradients
    # (calls with RETURNED inputs use the implicit body gradient or grad_fxn; opaque CALLs require an explicit grad_fxn)
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
