from collections import defaultdict, deque
from dataclasses import dataclass
from tinygrad.uop.ops import UOp, Ops, GroupOp, PatternMatcher, UPat, graph_rewrite, graph_rewrite_map, identity_element, resolve
from tinygrad.uop.ops import can_pad, sint, track_rewrites, _substitute
from tinygrad.codegen.lowerer import get_contraction_with_reduce, get_contraction
from tinygrad.codegen.symbolic import symbolic_simple
from tinygrad.helpers import Metadata, all_int, all_same, colored, prod, dedup, unwrap, getenv, pluralize, ContextVar, Context, diskcache_put, flatten
from tinygrad.helpers import FUSE_CONV_BW, FUSE_ARANGE, DEBUG, DONT_REALIZE_EXPAND, DONT_GROUP_REDUCES, SPLIT_REDUCEOP, CAPTURE_PROCESS_REPLAY
from tinygrad.dtype import ImageDType
from tinygrad.engine.multi import multi_pm, replace_allreduce
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.uop.spec import type_verify, sched_spec

# creation can recurse a lot
import sys
sys.setrecursionlimit(10000)

# **** UOp merge views

merge_views = PatternMatcher([
  # merge adjacent views
  (UPat(Ops.VIEW, src=(UPat(Ops.VIEW, name="v1"),), name="v2"), lambda v1,v2: v1.replace(arg=v1.arg+v2.arg)),
  # merge view on load/store/valid
  (UPat(Ops.VIEW, src=(UPat((Ops.LOAD, Ops.STORE, Ops.VALID), name="x"),), name="view"),
   lambda x,view: x.replace(src=tuple((s.st+view.st).to_uop() if s.op is Ops.VIEW else s for s in x.src))),
  # merge view on const if it's not masked
  (UPat((Ops.CONST, Ops.DEFINE_VAR), name="x").view(name="view"),
   lambda x,view: x.replace(src=(x.src[0].replace(arg=x.st+view.st),)) if all(v.mask is None for v in (x.st+view.st).views) else None),
  # replace view with base if it's contiguous and the shapes match
  (UPat(GroupOp.All-{Ops.DEVICE}, name="x").view(name="view"), lambda x,view: x if view.st.contiguous and x.shape == view.shape else None),
  # replace masked view with zero if it can collapse
  (UPat(Ops.VIEW, src=(UPat(),), name="view"),
   lambda view: view.const_like(0) if (mask:=view.st.views[-1].mask) is not None and any((x[1]-x[0]) == 0 for x in mask) else None),
  # movement ops apply a new view on the base
  (UPat(GroupOp.Movement, src=(UPat.var("x"),), name="mop"), lambda mop,x: x.view(mop.st)),
])

# **** schedule simplifier

def simplify_stride0_reduce(reduce:UOp, x:UOp):
  # must be unmasked (NOTE: can be relaxed if not masked on stride 0 axis)
  if any(v.mask is not None for v in unwrap(x.st).views): return None
  # must have all stride 0 in the relevant axis (NOTE: can do partial)
  if not all(unwrap(x.st).views[-1].strides[axis] == 0 for axis in reduce.arg[1]) or not all_int(x.shape): return None
  prshape = prod(x.shape[i] for i in reduce.arg[1])
  ret = x.shrink(tuple((0,s) if i not in reduce.arg[1] else (0,1) for i,s in enumerate(x.shape)))
  match reduce.arg[0]:
    case Ops.ADD: return ret*prshape
    case Ops.MUL: return ret.pow(prshape)
    case Ops.MAX: return ret # NOTE: Ops.MAX is passthrough

def split_reduceop(reduce:UOp, x:UOp):
  if not SPLIT_REDUCEOP or not all_int(x.shape) or (prod(x.shape)//prod(reduce.shape))<getenv("REDUCEOP_SPLIT_THRESHOLD", 32768): return None
  # if there are few globals, make some reduces into globals by splitting into two kernels
  # cap output buffer to 2**22: heuristic number of global outputs to achieve max occupancy with enough locals+upcasts for gemm
  #   ~2**10 should be enough if GROUP is used
  # 256 split maximum should be "negligible reduce" for low prod(reduce.shape), 8 split minimum.
  # split is moved to the end to provide maximum locality for the second phase reduce.
  real_strides = unwrap(x.st).real_strides(ignore_valid=True)
  if not (split_candidates:=[(i,d) for i in reduce.arg[1] for d in range(min(256,2**getenv("REDUCEOP_SPLIT_SIZE",22)//prod(reduce.shape)),8-1,-1)
                             if x.shape[i]%d==0 and real_strides[i]!=0]): return None
  dim_to_split, divisor = split_candidates[0]
  splitted_shape = x.shape[:dim_to_split]+(divisor,)+(x.shape[dim_to_split]//divisor,)+x.shape[dim_to_split+1:]
  splitted = x.reshape(splitted_shape).permute(tuple([d for d in range(len(splitted_shape)) if d!=dim_to_split]+[dim_to_split]))
  if DEBUG >= 3: print(f"split {divisor}: {x.shape} -> {splitted.shape} -> {reduce.shape}")
  # reduce original axes, then split
  return splitted.r(*reduce.arg).r(reduce.arg[0], (len(reduce.shape),)).reshape(reduce.shape)

ALWAYS_CONTIGUOUS = {Ops.CONTIGUOUS, Ops.ASSIGN, Ops.COPY, Ops.BUFFER, Ops.BUFFER_VIEW, Ops.CONST, Ops.BIND, Ops.DEVICE}

sym = symbolic_simple+PatternMatcher([
  # UOp with size 0 is zero
  (UPat(GroupOp.All-{Ops.SINK}, name="root"), lambda root: root.const_like(0) if root.base.st is not None and root.size == 0 \
    and not (root.base.op is Ops.CONST and root.base.arg == 0) else None),
  # DETACH and CONTIGUOUS_BACKWARD are NOOPs here
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), name="x"), lambda x: x.src[0]),
  # MULTI in SINK just flattens srcs
  (UPat(Ops.SINK, name="x"),
   lambda x: UOp.sink(*new_src) if (new_src:=tuple(flatten([s.src if s.op is Ops.MULTI else [s] for s in x.src]))) != x.src else None),
  # reduce of size 0 is the identity element
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)),
   lambda reduce,x: reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) if x.size == 0 and reduce.size != 0 else None),
  # reduce on stride 0 is collapsed
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)), simplify_stride0_reduce),
  # split_reduceop
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)), split_reduceop),
  # COPY(CONST) creates a new CONST on the destination device
  (UPat(Ops.COPY, name="root", src=(UPat.cvar("x"), UPat(Ops.DEVICE))), lambda root,x: root.const_like(x.arg)),
  # store a shrink before COPY, otherwise view after the COPY
  (UPat(Ops.COPY, src=(UPat(Ops.VIEW, name="v"), UPat(Ops.DEVICE)), name="copy"), lambda copy,v:
    v.contiguous().copy_to_device(copy.device, arg=copy.arg) if prod(v.shape) < prod(v.base.shape) else \
    v.base.copy_to_device(copy.device, arg=copy.arg).view(v.st)),
  # remove cast to image when it's already a contiguous image
  (UPat(Ops.CAST, name="cast", src=(UPat(Ops.VIEW, name="vm", src=(UPat(Ops.CONTIGUOUS, name="base"),)),)),
   lambda cast,base,vm: base.view(vm.st) if isinstance(cast.dtype, ImageDType) and isinstance(base.dtype, ImageDType) else None),
  # make things that can't be images not images
  (UPat(GroupOp.All-{Ops.BUFFER, Ops.VIEW, Ops.CONST, Ops.DEVICE}, name="u"), lambda u: u.replace(dtype=dt.base) if isinstance(dt:=u.dtype,ImageDType)
   and (prod(u.shape) != prod(dt.shape) or not any(u.shape[x]%4 == 0 for x in u.st.unit_stride_axes())) else None),
  # remove contiguous if we can just view the buffer
  (UPat(Ops.CONTIGUOUS, name="root", src=(UPat(Ops.VIEW, name="view", src=(UPat(Ops.BUFFER, name="buf"),)),)),
   lambda root,view,buf: view if view.st.contiguous and view.size == buf.size else None),
  # contiguous/buffer/copy/assign is already contiguous
  (UPat(Ops.CONTIGUOUS, name="root", src=(UPat((Ops.CONTIGUOUS, Ops.BUFFER, Ops.COPY, Ops.ASSIGN)),)), lambda root: root.src[0]),
  # substitute BITCAST/CONTIGUOUS with BUFFER_VIEW on DISK
  (UPat((Ops.BITCAST, Ops.CONTIGUOUS), src=(UPat.var("x"),), name="t"), lambda x,t: UOp(Ops.BUFFER_VIEW, t.dtype, (x.base,),
    (t.size, x.st.views[0].offset)).reshape(t.shape) if isinstance(x.device, str) and x.device.startswith("DISK") else None),
  # double ASSIGN to same target is one ASSIGN
  (UPat(Ops.ASSIGN, src=(UPat.var("t"), UPat(Ops.ASSIGN, src=(UPat.var("t"), UPat.var("x"))))), lambda x,t: t.assign(x.contiguous())),
  # ASSIGN to unrealized replaces the UOp
  (UPat(Ops.ASSIGN, src=(UPat.var("t"), UPat.var("x"))), lambda x,t: x.contiguous() if t.base.op not in {Ops.BUFFER, Ops.BUFFER_VIEW} else None),
  # put CAST to smaller dtype before EXPAND
  (UPat(Ops.CAST, name="cast", src=(UPat(Ops.VIEW, name="vm"),)), lambda cast,vm: vm.base.cast(cast.dtype).view(vm.st)
    if cast.dtype.itemsize <= vm.dtype.itemsize and resolve(prod(vm.shape) > vm.st.real_size()) else None),
  # put UnaryOps before EXPANDs, if it can fuse with the input
  (UPat(GroupOp.Unary, src=(UPat(Ops.VIEW, src=(UPat(GroupOp.All-ALWAYS_CONTIGUOUS, name="inp"),), name="v"),), name="alu"),
   lambda inp,v,alu: inp.alu(alu.op).view(v.st) if resolve(prod(alu.shape) > v.st.real_size()) else None),
])

# support for using a contiguous permuted view instead of the parent view if one exists

def found_contiguous(ctx:dict[UOp, UOp], contig:UOp, src:UOp):
  if (sti:=unwrap(src.st).invert(src.base.shape)) is not None: ctx[src.base] = contig.view(sti)

replace_contiguous = PatternMatcher([
  (UPat(Ops.CONTIGUOUS, src=(UPat(Ops.VIEW, name="src"),), name="contig"), found_contiguous),
  (UPat(GroupOp.ALU, name="alu"), lambda ctx,alu: alu.replace(src=new_src) if (new_src:=tuple(ctx.get(s, s) for s in alu.src)) != alu.src else None),
])

# **** Grouper decides which of the UOps realize

def realize(ctx:dict[UOp, None], tr:UOp) -> None: ctx[tr] = None

def realize_before_view(ctx:dict[UOp, None], view:UOp, tr:UOp) -> None:
  st = unwrap(view.st)
  # always realize unsafe pad ops before masked view
  if any(v.mask is not None for v in st.views) and not can_pad(tr, ctx): return realize(ctx, tr)
  # fold simple pads
  if len(st.views) == 1 and (m:=st.views[-1].mask) is not None and all_int(tr.shape) and resolve(prod(tr.shape) >= prod([y-x for x,y in m])): return
  # realize before expand
  if resolve(prod(tr.shape) < prod(st.shape)) and not DONT_REALIZE_EXPAND: return realize(ctx, tr)

do_realize = PatternMatcher([
  # always realize SINK parents
  (UPat(Ops.SINK, name="s"), lambda ctx,s: ctx.update((x.base, None) for x in s.src if x.base.op not in ALWAYS_CONTIGUOUS)),
  # always realize ASSIGN/CONTIGUOUS/GroupOp.Meta
  (UPat({Ops.ASSIGN, Ops.CONTIGUOUS, *GroupOp.Meta}, name="tr"), realize),
  # realize before expand or unsafe pad ops
  (UPat(Ops.VIEW, src=(UPat(GroupOp.All-ALWAYS_CONTIGUOUS, name="tr"),), name="view"), realize_before_view),
  # realize before COPY
  (UPat(Ops.COPY, src=(UPat(GroupOp.All-ALWAYS_CONTIGUOUS, name="tr"),), allow_any_len=True), realize),
])

def recursive_group(tr:UOp, st:ShapeTracker, r:UOp, children:defaultdict[UOp, dict[UOp, None]], realizes:dict[UOp, None],
                    reduce_for_op:dict[UOp, UOp], group:dict[UOp, None], cache:dict[tuple[UOp, ShapeTracker], None]) -> None:
  if (tr, st) in cache: return
  cache.setdefault((tr, st))
  rsize = unwrap(r.st).size
  if tr in realizes and tr is not r:
    # can only fuse contiguous
    # max one reduceop per kernel
    if not st.contiguous or st.size != rsize or tr in reduce_for_op: group.setdefault(r)
    return group.setdefault(tr)
  for tr_next in children[tr]:
    # max one reduceop per kernel
    if tr_next.op is Ops.REDUCE_AXIS: return group.setdefault(r)
    # can only fuse contiguous
    if len(st_childs:=dedup(unwrap(x.st) for x in tr_next.src if x.base == tr)) > 1: return group.setdefault(r)
    recursive_group(tr_next, st+st_childs[0], r, children, realizes, reduce_for_op, group, cache)

def group_realizes(sink:UOp) -> dict[UOp, None]:
  # start by adding uops that always realize
  realizes: dict[UOp, None] = {}
  sink = graph_rewrite(sink, do_realize, ctx=realizes, name="do_realize")
  if DONT_GROUP_REDUCES: return realizes

  # construct children graph (only for bases)
  children: defaultdict[UOp, dict[UOp, None]] = defaultdict(dict)
  assigns: dict[UOp, None] = {}
  for u in (toposort:=sink.toposort()):
    if u.op in {Ops.VIEW, Ops.SINK}: continue
    if u.op is Ops.ASSIGN: assigns[u.buf_uop] = None
    for s in u.src: children[s.base][u] = None

  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: dict[UOp, UOp] = {}
  double_reduces: list[UOp] = []
  for r in toposort:
    if r.op is not Ops.REDUCE_AXIS: continue
    if len(r.arg) == 3 and r.arg[2] is True: continue
    if FUSE_CONV_BW and r.src[0].base.op is Ops.REDUCE_AXIS and r.src[0] is not r.src[0].base: double_reduces.append(r)
    if r in realizes: continue
    group: dict[UOp, None] = {}
    recursive_group(r, unwrap(r.st), r, children, realizes, reduce_for_op, group, cache={})
    # max one reduceop per kernel
    can_chase = all(tr not in reduce_for_op for tr in group)
    # TODO: forced_realize exists because the scheduler is incapable of checking for self-contained DAGs
    forced_realize = r in group
    # can only have one output
    if not forced_realize and len(group) > 1: forced_realize = True
    # can only fuse assign if no other assign_target is used in the kernel
    if not forced_realize and (assign_targets:={x.buf_uop for x in group if x.op is Ops.ASSIGN}):
      parents = deque((r, *group))
      while parents and not forced_realize:
        p = parents.pop().base
        if p.op is Ops.BUFFER and p in assigns and p not in assign_targets: forced_realize, can_chase = True, False
        if p in realizes: continue
        parents.extend(p.src)
    if forced_realize or not group:
      tr = r
      if can_chase:
        # can chase this down to contiguous children
        st = unwrap(tr.st)
        while len(children[tr]) == 1:
          tr_next = next(iter(children[tr]))
          st_childs = dedup(unwrap(s.st) for s in tr_next.src if s.base is tr)
          if len(st_childs) > 1: break
          if st.size != st_childs[0].size: break
          st = st + st_childs[0]
          if not st.contiguous or tr_next.op is Ops.REDUCE_AXIS: break
          tr = tr_next
        # don't cast to higher size before store (tr cannot be realized if forced_realize)
        if tr.op is Ops.CAST and tr.dtype.itemsize > tr.src[0].dtype.itemsize:
          tr = tr.src[0].base
      group = {tr: None}
      realizes[tr] = None
    reduce_for_op.update((tr, r) for tr in group)
  # fuse double reduces with no other child
  for reduceop in double_reduces:
    top_reduce = reduceop.src[0].base
    if len(children[top_reduce]) == 1: del realizes[top_reduce]
  return realizes

# **** create kernels

@dataclass(frozen=True)
class Kernel:
  ast: UOp
  metadata: tuple[Metadata, ...] = ()
  def __repr__(self):
    ast_rep = f"SINK{tuple(s.op for s in self.ast.src)}" if self.ast.op is Ops.SINK else repr(self.ast.op)
    return f"<Kernel {len(list(self.ast.toposort()))} {ast_rep} {self.metadata}>"

def create_kernel(x:UOp, b:UOp|None=None):
  if b is None: b = UOp.new_buffer(x.device, x.size, x.dtype)
  kernel = UOp(Ops.KERNEL, src=(b,)+x.src, arg=Kernel(x.sink(), (m,) if (m:=x.metadata) else ()))
  buffer = b.base if b.size == b.base.size else UOp(Ops.BUFFER_VIEW, b.dtype, (b.base,), (b.size, b.arg.views[0].offset))
  return buffer.assign(kernel).reshape(x.shape)

DONT_PLACE_IN_KERNEL = {Ops.KERNEL, Ops.ASSIGN, Ops.BUFFER}
def append_to_kernel(ctx:dict[UOp, None], x:UOp):
  new_srcs: list[UOp] = []
  metadata = dict.fromkeys(x.arg.metadata)
  for s in x.src:
    if s.op in DONT_PLACE_IN_KERNEL or s in ctx: new_srcs.append(s)
    else:
      new_srcs.extend(s.src)
      if s.base.op not in {Ops.CONST, Ops.DEVICE} and (m:=s.metadata): metadata[m] = None
  if (new_src:=tuple(dedup(new_srcs))) != x.src: return x.replace(src=new_src, arg=Kernel(x.arg.ast, tuple(metadata)))

create_kernels = merge_views+PatternMatcher([
  # always give assign/contiguous a kernel
  (UPat.assign(UPat.var("b"), UPat(GroupOp.All-{Ops.KERNEL}), name="x"), create_kernel),
  (UPat(Ops.CONTIGUOUS, name="x"), create_kernel),
  # create a buffer for COPY on the new device
  (UPat(Ops.COPY, src=(UPat(), UPat(Ops.DEVICE)), name="x"), create_kernel),
  # otherwise check the context if we're realizing this UOp
  (UPat(GroupOp.All-DONT_PLACE_IN_KERNEL, name="x"), lambda ctx,x: create_kernel(x) if x in ctx else None),
  # walk back the local graph until we reach a realized source
  (UPat(Ops.KERNEL, name="x"), append_to_kernel),
  # remove extra views and constants from SINK
  (UPat(Ops.SINK, name="x"),
   lambda x: x.replace(src=new_src) if (new_src:=tuple(dedup(s.base for s in x.src if s.base.op not in {Ops.CONST, Ops.BIND}))) != x.src else None),
])

# **** swizzler

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
  # view before elementwise ops
  (UPat(Ops.VIEW, src=(UPat({*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.BIND}, name="e"),), name="view"),
   lambda e,view: e.replace(src=tuple(s.view(s.st+view.st) if s.op is Ops.VIEW else s.view(view.st) for s in e.src))),
  # if there's ones added after reduce, put this before the reduce
  (UPat(Ops.VIEW, src=(UPat(Ops.REDUCE_AXIS, src=(UPat.var("src"),), name="r"),), name="view"), reduce_push_add_ones),
])

def apply_swizzle(u:UOp) -> UOp: return graph_rewrite(u, view_left, name="Sub View Left")

def swizzle_reduceop(r:UOp, src:UOp, view:UOp, fuse=False):
  if (st:=unwrap(view.st)).contiguous and st.size == r.size: return None
  input_st = ShapeTracker.from_shape(src.shape)
  tmp = input_st.permute(tuple(i for i in range(len(input_st.shape)) if i not in r.axis_arg)+r.axis_arg)
  prshape = prod(rshape:=tmp.shape[-len(r.axis_arg):])
  strides = strides_for_shape(rshape)
  nv = [View.create(v.shape+rshape, tuple(x*prshape for x in v.strides)+strides,
                    v.offset*prshape, v.mask+tuple((0,s) for s in rshape) if v.mask is not None else None) for v in st.views]
  # create a new reduceop for the swizzled input
  new_input_st = tmp + ShapeTracker(tuple(nv))
  new_axis = tuple(range(len(st.shape), len(st.shape) + len(r.axis_arg)))
  swizzled_src = apply_swizzle(src.view(src.arg+new_input_st if src.op is Ops.VIEW else new_input_st))
  if fuse: red = UOp(Ops.REDUCE_AXIS, r.dtype, (swizzled_src.fuse(),), (r.arg[0], new_axis, True))
  else: red = UOp(Ops.REDUCE_AXIS, r.dtype, (swizzled_src,), (r.arg[0], new_axis))
  return red.view(ShapeTracker.from_shape(st.shape))

def reduceop_view_right(src:UOp, v:UOp, r:UOp):
  assert unwrap(v.st).contiguous and v.size == src.size, f"can't compute new axis for {src.shape} -> {r.shape}"
  if (contraction:=get_contraction(v.shape, src.shape)) is None: return None
  new_axis: list[int] = []
  for i,pairs in enumerate(contraction):
    if any(x in r.axis_arg for x in pairs): new_axis.append(i)
  return src.r(r.arg[0], tuple(new_axis)).reshape(r.shape)

def elementwise_view_right(root:UOp):
  if not (swizzles:=[x for x in root.src if x.op is Ops.VIEW and x.base.op not in ALWAYS_CONTIGUOUS]): return None
  assert all_same([x.base.size for x in swizzles]), f"swizzle inputs must have the same size {swizzles}"
  # place view after applying the elementwise op
  new_st = ShapeTracker.from_shape(swizzles[0].base.shape)
  new_src = [x.base if x.base.shape==new_st.shape else apply_swizzle(x.view(x.arg+new_st) if x.op is Ops.VIEW else x.view(new_st)) for x in root.src]
  # reshape to match downstream shapes
  return root.replace(src=tuple(new_src)).reshape(root.shape)

# push VIEW to children
view_right = merge_views+PatternMatcher([
  # push a non contiguous ShapeTracker through reduceop
  (UPat(Ops.VIEW, src=(UPat(Ops.REDUCE_AXIS, src=(UPat.var("src"),), name="r"),), name="view"), swizzle_reduceop),
  # apply view after reduceops
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.VIEW, src=(UPat(GroupOp.All-ALWAYS_CONTIGUOUS, name="src"),), name="v"),), name="r"), reduceop_view_right),
  # apply view after elementwise ops
  (UPat(GroupOp.All-{Ops.SINK}, name="root"), elementwise_view_right),
  # merge axes for double reduce (invert of SPLIT_REDUCEOP=1)
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.REDUCE_AXIS, name="r1"),), name="r2"),
   lambda r1,r2: r1.replace(arg=(r1.arg[0], r2.arg[1]+r1.arg[1])) if r1.arg[0] is r2.arg[0] else None),
])

# **** fix kernel AST

add_buffer_ops = PatternMatcher([
  # LOAD
  (UPat(Ops.BUFFER, name="x"), lambda ctx,x: UOp.load(UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(x.size), (), ctx.index(x)), x.st.to_uop())),
  # STORE (except for meta ops)
  (UPat(Ops.SINK, src=(UPat(GroupOp.Meta, name="x"),)), lambda x:x),
  (UPat(Ops.SINK, src=UPat(GroupOp.All-{Ops.STORE}), name="sink"), lambda ctx,sink:
   UOp.sink(*[UOp.store(UOp(Ops.DEFINE_GLOBAL, (s:=x.base).dtype.ptr(ctx[i].size), (), i), s.st.to_uop(), s) for i,x in enumerate(sink.src)])),
  # passthrough ASSIGN
  (UPat(Ops.ASSIGN, name="x"), lambda x: x.src[1]),
  # VALID
  (UPat(Ops.VIEW, src=(UPat((Ops.CONST, Ops.DEFINE_VAR), name="x"),), name="view"), lambda x,view: x.valid(view.arg)),
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

fix_kernel_ops = PatternMatcher([
  # remove CONTIGUOUS/DEVICE from kernel AST
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("x"),)), lambda x: x),
  (UPat(Ops.VIEW, src=(UPat(Ops.DEVICE),), name="view"), lambda view: view.replace(src=())),
  # no ImageDType after load
  (UPat(GroupOp.All-{Ops.DEFINE_GLOBAL}, name="x"), lambda x: x.replace(dtype=x.dtype.base) if isinstance(x.dtype, ImageDType) else None),
  # if this kernel also assigns to the loaded buffer, ensure we can index it correctly
  (UPat(Ops.LOAD, src=(UPat.var("glbl"), UPat.var("view"))), check_load_st),
])

def fix_kernel_ast(k:UOp) -> UOp|None:
  if k.arg.ast.op in GroupOp.Meta or all(s.op is Ops.STORE for s in k.arg.ast.src): return None
  # replace assign sources with a view of the target buffer
  parents_rep: dict[UOp, UOp] = {}
  for s in k.src:
    if s.op is Ops.ASSIGN:
      for out in s.src[1].arg.ast.src: parents_rep[out] = s.buf_uop.view(unwrap(out.st))
      parents_rep[s] = s.buf_uop
  ast = k.arg.ast.substitute(parents_rep, name="replace realized")
  # push views to edges
  ast = graph_rewrite(graph_rewrite(ast, view_left, name="Main View Left"), view_right, name="Main View Right")
  # replace buffer with define_global + add load/store last
  ast = graph_rewrite(ast, merge_views+add_buffer_ops+fix_kernel_ops, bufs:=tuple(s.buf_uop for s in k.src), bottom_up=True, name="replace buffer")
  if ast.op is Ops.SINK and not all_same([x.device for x in bufs]):
    raise RuntimeError(f"all buffers must be on the same device: {tuple(b.buffer for b in bufs)}")
  return k.replace(arg=Kernel(ast, k.arg.metadata))

create_ast = PatternMatcher([(UPat(Ops.KERNEL, name="k"), fix_kernel_ast),])

pm_fuse = PatternMatcher([
  # FUSE on CONTIGUOUS removes FUSE
  (UPat(Ops.CONTIGUOUS, name="c").fuse(), lambda c: c),

  # FUSE triggers swizzle on reduceop
  (UPat(Ops.VIEW, src=(UPat(Ops.REDUCE_AXIS, src=(UPat.var("src"),), name="r").or_casted(),), name="view").fuse(),
   lambda r,src,view: ret.cast(view.dtype) if (ret:=swizzle_reduceop(r, src, view, fuse=True)) is not None else None),

  # FUSE on reduce (without view) adds fuse marker to grouper
  (UPat(Ops.REDUCE_AXIS, name="r").fuse(),
   lambda r: r.replace(src=(r.src[0].fuse(),), arg=r.arg+(True,)) if len(r.arg) == 2 else None),

  # remove FUSE and insert CONTIGUOUS if it's an unsafe pad
  (UPat(Ops.VIEW, src=(UPat(GroupOp.UnsafePad, name="alu"),), name="view").fuse(),
   lambda alu, view: alu.contiguous().view(view.st) if any(v.mask is not None for v in view.st.views) else None),

  # FUSE elementwise.
  (UPat(Ops.VIEW, src=(UPat({*GroupOp.ALU, Ops.CAST}, name="alu"),), name="view").fuse(),
   lambda alu, view: alu.replace(src=tuple(x.view(x.arg+view.arg if x.op is Ops.VIEW else view.arg).fuse() for x in alu.src))),

  # push FUSE through to srcs
  (UPat(Ops.FUSE, name="x"), lambda x: x.src[0].replace(src=tuple(y.fuse() for y in x.src[0].src))),
])

def do_fusion(x:UOp):
  found_contiguous = {}
  def gate_contiguous(x):
    if is_contiguous:=(x.op is Ops.CONTIGUOUS): found_contiguous[x] = x.replace(src=(UOp(Ops.VIEW, arg=x.st),))
    return not is_contiguous
  x.toposort(gate=gate_contiguous)
  del gate_contiguous
  return graph_rewrite(x.substitute(found_contiguous), pm_fuse, name="local fusion").substitute({v:k for k,v in found_contiguous.items()})

def fuse_arange(root:UOp):
  # skip if root is arange
  if not FUSE_ARANGE or root.src[0].base.op is Ops.CONST: return None
  # gather all local aranges (including any fused ones)
  local_arange: list[UOp] = []
  def gate_reduce(u):
    if u.op is Ops.REDUCE_AXIS and u.src[0].base.op is Ops.CONST: local_arange.append(u)
    return u.op not in {*ALWAYS_CONTIGUOUS, Ops.REDUCE_AXIS} or u is root
  toposort = root.toposort(gate=gate_reduce)
  if not local_arange: return None
  # fuse the nearest expand child of arange
  local_children: dict[UOp, list[UOp]] = {}
  for u in toposort:
    for s in u.src: local_children.setdefault(s, []).append(u)
  fuse_rep: dict[UOp, UOp] = {}
  # skip if root depends on aranges with different ndims. This can be improved
  if any(len(set(dims)) > 1 for dims in zip(*[r.src[0].shape for r in local_arange])): return
  for r in local_arange:
    # skip if already fused
    if len(r.arg) > 2: continue
    q = list(local_children[r])
    while q:
      u = q.pop()
      if not (curr_children:=local_children.get(u, [])): continue
      for child in curr_children:
        other_paths = {s for s in child.toposort() if s.op in {Ops.REDUCE_AXIS, Ops.BUFFER} and s not in {root, r}}
        fuse_rep[child] = child.replace(src=tuple(s.fuse() if s is u else s for s in child.src))
        if other_paths: break
      else: q.extend(curr_children)
  return root.substitute(fuse_rep, name="fuse_arange") if fuse_rep else None


do_fuse = PatternMatcher([
  (UPat(Ops.FUSE, name="x"), do_fusion),
  (UPat(Ops.REDUCE_AXIS, name="root"), fuse_arange),
])

PROCESS_REPLAY_CAPTURE:dict[int,bytes] = {}
if CAPTURE_PROCESS_REPLAY:
  import atexit
  @atexit.register
  def save_process_replay():
    for k,v in PROCESS_REPLAY_CAPTURE.items(): diskcache_put("schedule_process_replay", k, v, prepickled=True)

def get_name(becomes_map:dict[UOp, UOp]) -> str:
  assigned_kernels = {u.base.buf_uop:u.base.src[1] for u in becomes_map.values() if u.base.op is Ops.ASSIGN}.values()
  return f"Schedule {pluralize('Kernel', len(set(assigned_kernels)))}"

@track_rewrites(name_fxn=get_name)
def get_becomes_map(big_sink:UOp) -> dict[UOp, UOp]:
  # multi + merge_views + simplify
  tensor_map = graph_rewrite_map(big_sink, multi_pm+replace_allreduce+do_fuse+merge_views+sym+replace_contiguous, ctx={}, name="merge_views")

  # display the cleaned up tensor graph
  if getenv("VIZ"): graph_rewrite(tensor_map[big_sink], PatternMatcher([]), name="View Tensor Graph")

  # group into kernels
  realize_map = group_realizes(tensor_map[big_sink])
  tensor_map = graph_rewrite_map(tensor_map[big_sink], create_kernels, ctx=realize_map, bottom_up=True, input_map=tensor_map, name="create_kernels")

  # if a kernel depends on a buffer, and that buffer is later assigned to, make the assign depend on the kernel's assign
  kernel_assign: dict[UOp, UOp] = {}
  assign_rep: dict[UOp, UOp] = {}
  for u in tensor_map[big_sink].toposort():
    if u.op is not Ops.ASSIGN: continue
    kernel_assign[u.buf_uop] = u
    for s in u.src[1].src:
      if s.op is not Ops.BUFFER or s is u.buf_uop or (a:=kernel_assign.get(s)) is None: continue
      if any(x.op is Ops.ASSIGN and x.buf_uop is s for x in u.toposort()):
        raise RuntimeError(f"cycle detected in graph, kernel for {u.buf_uop} must either depend on ASSIGN or BUFFER")
      assign_rep[a] = kernel_assign[s] = a.replace(src=a.src+(u,))
  if assign_rep:
    tensor_map = graph_rewrite_map(tensor_map[big_sink], _substitute, ctx=assign_rep, bottom_up=True, input_map=tensor_map, name="fix_assign")

  # finally, create the AST for kernels
  tensor_map = graph_rewrite_map(tensor_map[big_sink], create_ast, bottom_up=True, input_map=tensor_map, name="create_ast")

  # display the final graph
  sched_sink = tensor_map[big_sink]
  if getenv("VIZ"): graph_rewrite(sched_sink, PatternMatcher([]), name="View Kernel Graph")
  if getenv("VIZ"): graph_rewrite(sched_sink, PatternMatcher([]), name="View Memory Graph")

  # verify Kernels match the spec
  type_verify(list(toposort:=sched_sink.toposort()), sched_spec)

  # capture process replay
  if CAPTURE_PROCESS_REPLAY:
    with Context(PICKLE_BUFFERS=0):
      import pickle
      PROCESS_REPLAY_CAPTURE[id(big_sink)] = pickle.dumps((big_sink, ContextVar._cache, [u.arg.ast for u in toposort if u.op is Ops.KERNEL]))

  # map tensors to buffer/assign/const
  becomes_map: dict[UOp, UOp] = {}
  for k,v in tensor_map.items():
    if k is v: continue
    op = v.base.op
    if op in {Ops.BUFFER, Ops.ASSIGN}: becomes_map[k] = v
    if op is Ops.CONST and all_int(v.shape): becomes_map[k] = v
    if op is Ops.MULTI and all(x.base in becomes_map for x in v.base.src): becomes_map[k] = v

  return becomes_map
