# the job of the lowerer is to do indexing
from dataclasses import dataclass
from tinygrad.uop.ops import KernelInfo, UOp, Ops, PatternMatcher, UPat, sint_to_uop, AxisType, graph_rewrite, resolve

# ***** indexing *****

@dataclass
class IndexContext:
  axis_types: tuple[AxisType, ...]
  idxs: list[UOp]
  start: int = 0

def shape_to_idx(s, axis_types, start=0):
  return [UOp.range(sint_to_uop(s), start+i, at) for i, (s, at) in enumerate(zip(s, axis_types))]

def get_index(ast:UOp) -> IndexContext:
  axis_types = ast.arg.axis_types if isinstance(ast.arg, KernelInfo) else ()
  if len(ast.full_shape) != len(axis_types) and ast.st is not None:
    axis_types = tuple([AxisType.REDUCE if resolve(s != fs) else AxisType.LOOP for s,fs in zip(ast.shape, ast.full_shape)])
  return IndexContext(axis_types, [], 0)

# ***** lowering (given index) *****

def subblock(ctx: IndexContext, full_new_idx: list[UOp], src: UOp):
  lc = IndexContext(ctx.axis_types, full_new_idx, ctx.start+1000)
  ctx.start = lc.start
  return graph_rewrite(src, pm_lowerer, lc, name="subblock", bottom_up=True)

def lower_reduce_axis(ctx: IndexContext, x: UOp):
  new_idxs = shape_to_idx(x.src[0].shape, ctx.axis_types, ctx.start)
  full_new_idx = list(ctx.idxs)
  for a in x.axis_arg: full_new_idx[a] = new_idxs[a]
  ret = subblock(ctx, full_new_idx, x.src[0])
  return UOp(Ops.REDUCE, x.dtype, (ret,)+tuple([full_new_idx[i] for i in x.axis_arg]), x.arg[0])

def lower_store(ctx: IndexContext, x: UOp, buf: UOp):
  # TODO: reenable after REDUCE_AXIS is fixed
  #assert x.src[1].shape == x.src[0].shape, f"shape mismatch on store {x.src[1].shape} != {x.src[0].shape}"

  new_idxs = shape_to_idx(x.src[0].shape, ctx.axis_types, ctx.start)
  idx = x.st_arg.to_valid_uop(new_idxs)
  used_idxs = [x for x in idx.toposort() if x in new_idxs]
  real_new_idxs = []
  for i in range(len(x.src[0].shape)):
    if new_idxs[i] in used_idxs or len(ctx.idxs) <= i: real_new_idxs.append(new_idxs[i])
    else: real_new_idxs.append(ctx.idxs[i])

  stored = subblock(ctx, real_new_idxs, x.src[1])
  used_ranges = [x for x in used_idxs if x.op is Ops.RANGE]
  return buf.index(idx).store(stored, *used_ranges)

def fixup_wmma(ctx:IndexContext, x:UOp):
  if x.tag is not None: return None
  new_idxs = shape_to_idx(x.src[0].shape, ctx.axis_types, ctx.start)
  full_new_idx = list(ctx.idxs)
  for a in x.arg[-1]: full_new_idx[a] = new_idxs[a]

  srcs = subblock(ctx, full_new_idx, UOp.sink(*x.src)).src

  # NOTE: this assumes these are expanded. which now shouldn't change anything
  new_x_arg_m2 = tuple([tuple([(full_new_idx[a].arg[0], sz) for a,sz in v]) for v in x.arg[-2]])
  new_x_arg_m1 = tuple([full_new_idx[a].arg[0] for a in x.arg[-1]])
  return x.replace(src=srcs, arg=x.arg[:-2]+(new_x_arg_m2, new_x_arg_m1), tag=1)

pm_lowerer = PatternMatcher([
  # TODO: remove these hacks
  # hack for old style CONST(VIEW) (now it's just VIEW(CONST))
  (UPat((Ops.DEFINE_VAR, Ops.CONST), src=(UPat(Ops.VIEW, name="v"),), name="c"), lambda c,v: c.replace(src=()).view(v.arg)),
  # hack for old style VALID (now it's just VIEW(CONST))
  (UPat(Ops.VALID, src=(UPat(Ops.VIEW, name="v"),)).where(UPat.cvar("c"), UPat(Ops.CONST, arg=0)), lambda c,v: c.replace(src=()).view(v.arg)),

  # consts and loads
  (UPat(Ops.VIEW, src=(UPat((Ops.CONST, Ops.DEFINE_VAR), name="c"),), name="view"),
   lambda ctx,view,c: c if all(x.mask is None for x in view.arg.views) else view.arg.to_valid_uop(ctx.idxs).get_valid().where(c, c.const_like(0))),
  (UPat(Ops.LOAD, src=(UPat.var("buf").view(),), allow_any_len=True, name="x"),
   lambda ctx,buf,x: UOp(Ops.LOAD, x.dtype, (buf.index(x.st_arg.to_valid_uop(ctx.idxs)),)+x.src[1:])),

  # reduce/view_const
  (UPat(Ops.REDUCE_AXIS, name="x"), lower_reduce_axis),
  (UPat(Ops.STORE, src=(UPat.var("buf").view(),), allow_any_len=True, name="x"), lower_store),
  (UPat(Ops.WMMA, name="x"), fixup_wmma),

  # axis fixups for WMMA
  (UPat((Ops.CONTRACT, Ops.UNROLL), name="x"),
   lambda ctx,x: x.replace(tag=1, arg=tuple([(ctx.idxs[a].arg[0], sz) for a,sz in x.arg])) if x.tag is None else None),
])
