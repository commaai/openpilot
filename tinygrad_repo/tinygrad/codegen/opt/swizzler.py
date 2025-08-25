from tinygrad.uop.ops import UOp, Ops, GroupOp, PatternMatcher, UPat, graph_rewrite, resolve, sint
from tinygrad.helpers import all_same, prod, unwrap, colored
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, strides_for_shape, get_contraction_with_reduce
from tinygrad.schedule.grouper import ALWAYS_CONTIGUOUS
from tinygrad.dtype import ImageDType, dtypes

merge_views = PatternMatcher([
  # merge adjacent views
  (UPat(Ops.VIEW, src=(UPat(Ops.VIEW, name="v1"),), name="v2"), lambda v1,v2: v1.replace(arg=v1.arg+v2.arg)),
  # replace MovementOps with VIEW
  (UPat(GroupOp.Movement, src=(UPat.var("x"),), name="mop"), lambda mop,x: x.base.view(mop.st)),
  # remove NOOP views
  (UPat.var("x").view(name="view"),
   lambda x,view: x if x.st is not None and x.op not in GroupOp.Defines and view.st.contiguous and view.shape == x.shape else None),
  (UPat(GroupOp.All-{Ops.DEFINE_GLOBAL}).view(name="view"),
   lambda view: view.const_like(0) if (mask:=view.st.views[-1].mask) is not None and any((x[1]-x[0]) == 0 for x in mask) else None),
  # only unmaksed VIEW on CONST replaces the ShapeTracker
  (UPat(Ops.VIEW, src=(UPat((Ops.CONST, Ops.DEFINE_VAR), name="x"),), name="view"),
   lambda x,view: x.replace(src=(x.src[0].replace(arg=x.st+view.st),)) if all(v.mask is None for v in (x.st+view.st).views) else None),
])

def reduce_push_add_ones(src:UOp, r:UOp, view:UOp):
  # contiguous, expand, and the same with ones removed
  if unwrap(view.st).contiguous and len(r.shape) < len(view.shape) and \
      tuple(x for x in r.shape if resolve(x != 1)) == tuple(x for x in view.shape if resolve(x != 1)):
    new_shape: list[sint] = []
    new_reduce_axis = []
    if (contraction:=get_contraction_with_reduce(view.shape, r.shape, r.arg[1])) is None: return None
    for i,pairs in enumerate(contraction):
      new_shape_chunk = [view.shape[p] for p in pairs]
      if i in r.arg[1]:
        # if this is a reduce axis, we need a 1 in the view here to put it
        assert len(new_shape_chunk) > 0
        new_shape += [1]*(len(pairs)-1) + [src.shape[i]]
        new_reduce_axis.append(len(new_shape)-1)
      else:
        # otherwise, pass through the new_shape_chunk
        new_shape += new_shape_chunk
    ret = r.replace(src=(src.reshape(tuple(new_shape)),), arg=(r.arg[0], tuple(new_reduce_axis))+r.arg[2:])
    assert ret.shape == view.shape, f"shape mismatch on reduce_push_add_ones, {ret.shape} != {view.shape}"
    return ret
  return None

view_left = merge_views+PatternMatcher([
  # view before elementwise and buffer ops
  (UPat(Ops.VIEW, src=(UPat({*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.BIND, Ops.STORE, Ops.VALID, Ops.SINK}, name="e"),), name="view"),
   lambda e,view: e.replace(src=tuple(s.view(view.st) for s in e.src))),
  # if there's ones added after reduce, put this before the reduce
  (UPat(Ops.VIEW, src=(UPat(Ops.REDUCE_AXIS, src=(UPat.var("src"),), name="r"),), name="view"), reduce_push_add_ones),
])

view_left_through_load = PatternMatcher([
  # view before load
  (UPat(Ops.VIEW, src=(UPat(Ops.LOAD, name="e"),), name="view"),
   lambda e,view: e.replace(src=tuple(s.view(view.st) for s in e.src))),
])

def apply_swizzle(u:UOp) -> UOp: return graph_rewrite(u, view_left, name="Sub View Left")

# change reduceop axes and input ShapeTrackers, view gets replaced with a reshape.
def swizzle_reduceop(r:UOp, src:UOp, view:UOp, fuse=False):
  # contiguous and same size can push to children
  # if there's a reduce child, shapes match with ones removed
  if unwrap(view.st).contiguous and view.size == r.size and \
      (not (len(r.arg) == 3 and r.arg[2]) or # arg[2] = True is fuse marker
       tuple((i,x) for i,x in enumerate(r.shape) if resolve(x != 1)) == tuple((i,x) for i,x in enumerate(view.shape) if resolve(x != 1))):
    return None
  # swizzle the input
  input_st = ShapeTracker.from_shape(src.shape)
  tmp = input_st.permute(tuple(i for i in range(len(input_st.shape)) if i not in r.axis_arg)+r.axis_arg)
  prshape = prod(rshape:=tmp.shape[-len(r.axis_arg):])
  strides = strides_for_shape(rshape)
  nv = [View.create(v.shape+rshape, tuple(x*prshape for x in v.strides)+strides,
                    v.offset*prshape, v.mask+tuple((0,s) for s in rshape) if v.mask is not None else None) for v in unwrap(view.st).views]
  new_view = tmp + ShapeTracker(tuple(nv))
  swizzled_input = apply_swizzle(src.view(new_view))
  # create a new reduceop
  new_axis = tuple(range(len(view.shape), len(view.shape) + len(r.axis_arg)))
  if fuse: red = UOp(Ops.REDUCE_AXIS, r.dtype, (swizzled_input.fuse(),), (r.arg[0], new_axis, True))
  else: red = UOp(Ops.REDUCE_AXIS, r.dtype, (swizzled_input,), (r.arg[0], new_axis))
  return red.reshape(view.shape)

def reduceop_view_right(src:UOp, v:UOp, r:UOp):
  assert unwrap(v.st).contiguous and v.size == src.size, f"can't compute new axis for {src.shape} -> {r.shape}"
  new_axis = [i for i,(s,u) in enumerate(zip(src.shape, r.shape)) if s != u]
  return src.r(r.arg[0], tuple(new_axis)).reshape(r.shape)

def elementwise_view_right(root:UOp):
  if not (swizzles:=[x for x in root.src if x.op is Ops.VIEW and x.base.op not in ALWAYS_CONTIGUOUS]): return None
  assert all_same([x.base.size for x in swizzles]), f"swizzle inputs must have the same size {swizzles}"
  # place view after applying the elementwise op
  new_st = ShapeTracker.from_shape(swizzles[0].base.shape)
  new_src = [x.base if x.base.shape==new_st.shape else apply_swizzle(x.view(new_st)) for x in root.src]
  # reshape to match downstream shapes
  return root.replace(src=tuple(new_src)).reshape(root.shape)

# push VIEW to children
view_right = merge_views+PatternMatcher([
  # push a non contiguous ShapeTracker through reduceop
  (UPat(Ops.VIEW, src=(UPat(Ops.REDUCE_AXIS, src=(UPat.var("src"),), name="r"),), name="view"), swizzle_reduceop),
  # apply view after reduceops
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.VIEW, src=(UPat(GroupOp.All-ALWAYS_CONTIGUOUS, name="src"),), name="v"),), name="r"), reduceop_view_right),
  # apply view after elementwise ops
  (UPat(GroupOp.All-{Ops.SINK, Ops.REDUCE_AXIS}, name="root"), elementwise_view_right),
  # merge axes for double reduce (invert of SPLIT_REDUCEOP=1)
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.REDUCE_AXIS, name="r1"),), name="r2"),
   lambda r1,r2: r1.replace(arg=(r1.arg[0], r2.arg[1]+r1.arg[1])) if r1.arg[0] is r2.arg[0] else None),
  # remove view from sink
  (UPat(Ops.VIEW, name="v").sink(name="sink"), lambda v,sink: v.src[0].sink(arg=sink.arg)),
])

def check_load_st(glbl:UOp, view:UOp):
  if glbl.arg != 0 or (st:=unwrap(view.st)).contiguous: return
  # if it has a single view and it becomes contiguous when you shrink expanded axes, it's fine
  if len(st.views) == 1 and st.shrink(tuple((0,1) if st == 0 else (0,s) for s,st in zip(st.shape, st.views[0].strides))).contiguous: return
  # if it has a single view and it's equal when you shrink a contig, it's fine
  if len(st.views) == 1 and (mask:=st.views[0].mask) is not None and ShapeTracker.from_shape(st.shape).shrink(mask) == st.shrink(mask): return
  # otherwise, it's not fine
  raise RuntimeError("self operand of augmented assign must be contiguous.\nhelp: consider using .contiguous():\n"
                     +colored("   - a += a.T\n", "red")+colored("   + a += a.T.contiguous()", "green"))

fix_kernel_ops = view_left_through_load+PatternMatcher([
  # add view to LOAD and STORE
  (UPat(Ops.DEFINE_GLOBAL, name="g").load(), lambda g: g.view(g.st).load()),
  (UPat(Ops.DEFINE_GLOBAL, name="g").store(UPat.var('x')), lambda g,x: g.view(g.st).store(x)),
  # VALID
  (UPat(Ops.VIEW, src=(UPat.cvar(),), name="self"),
   lambda self: UOp.where(UOp(Ops.VALID, dtypes.bool, (UOp(Ops.VIEW, arg=self.st),)), self.const_like(self.base.arg), 0)),
  # no ImageDType after index
  (UPat(GroupOp.All-{Ops.DEFINE_GLOBAL, Ops.VIEW}, name="x"), lambda x: x.replace(dtype=x.dtype.base) if isinstance(x.dtype, ImageDType) else None),
  # if this kernel also assigns to the loaded buffer, ensure we can index it correctly
  (UPat(Ops.LOAD, src=(UPat.var("glbl").view(name="view"),)), check_load_st),
])
