import sys, atexit, pickle
from collections import defaultdict, deque
from dataclasses import dataclass
from tinygrad.ops import UOp, Variable, Ops, GroupOp, PatternMatcher, UPat, graph_rewrite, graph_rewrite_map, track_rewrites, buffers
from tinygrad.ops import can_pad, identity_element, resolve, view_left, merge_views
from tinygrad.codegen.symbolic import symbolic_simple
from tinygrad.helpers import Context, ContextVar, Metadata, all_int, all_same, colored, diskcache_put, prod, dedup, unwrap, flatten, getenv, pluralize
from tinygrad.helpers import FUSE_CONV_BW, FUSE_ARANGE, DEBUG, CAPTURE_PROCESS_REPLAY, DONT_REALIZE_EXPAND, SPLIT_REDUCEOP
from tinygrad.dtype import ImageDType
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.device import Buffer
from tinygrad.spec import type_verify, kernel_spec

# creation can recurse a lot
sys.setrecursionlimit(10000)

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

def found_contiguous(ctx:dict[UOp, UOp], contig:UOp, src:UOp):
  if (sti:=unwrap(src.st).invert(src.base.shape)) is not None: ctx[src.base] = contig.view(sti)
def replace_contiguous(ctx:dict[UOp, UOp], alu:UOp):
  new_src = list(alu.src)
  for i,s in enumerate(alu.src):
    if (replace_src:=ctx.get(s, None)) is not None: new_src[i] = replace_src
  if tuple(new_src) != alu.src: return alu.replace(src=tuple(new_src))

def create_buffer_view(tr:UOp, x:UOp):
  assert isinstance(tr.device, str), "device must be string"
  if not tr.device.startswith("DISK"): return None
  return UOp(Ops.BUFFER_VIEW, tr.dtype, (x.base,), (tr.size, unwrap(x.st).views[0].offset)).reshape(tr.shape)

sym = symbolic_simple+PatternMatcher([
  # UOp with size 0 is zero
  (UPat(GroupOp.All-{Ops.SINK}, name="root"), lambda root: root.const_like(0) if root.base.st is not None and root.size == 0 \
   and not (root.base.op is Ops.CONST and root.base.arg == 0) else None),
  # DETACH and CONTIGUOUS_BACKWARD are NOOPs here
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), name="x"), lambda x: x.src[0]),
  # reduce of size 0 is the identity element
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)),
   lambda reduce,x: reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) if x.size == 0 and reduce.size != 0 else None),
  # reduce on stride 0 is collapsed
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)), simplify_stride0_reduce),
  # split_reduceop
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)), split_reduceop),
  # COPY(CONST) creates a new CONST on the destination device
  (UPat(Ops.COPY, name="root", src=(UPat(), UPat.cvar("x"),)), lambda root,x: root.const_like(x.arg)),
  # no COPY to same device, except clone (arg is True)
  (UPat(Ops.COPY, src=(UPat(), UPat.var("copyin")), name="copy"),
   lambda copyin,copy: copyin if copyin.device == copy.device and copy.arg is not True else None),
  # copyin must be base
  (UPat(Ops.COPY, src=(UPat(), UPat(Ops.VIEW, name="v")), name="copy"), lambda copy,v: v.contiguous().copy_to_device(copy.device) \
    if prod(v.shape) < prod(v.base.shape) else v.base.copy_to_device(copy.device, clone=copy.arg).view(v.st)),
  # remove cast to image when it's already a contiguous image
  (UPat(Ops.CAST, name="cast", src=(UPat(Ops.VIEW, name="vm", src=(UPat(Ops.CONTIGUOUS, name="base"))),)),
   lambda cast,base,vm: base.view(vm.st) if isinstance(cast.dtype, ImageDType) and isinstance(base.dtype, ImageDType) else None),
  # put CAST to smaller dtype before EXPAND
  (UPat(Ops.CAST, name="cast", src=(UPat(Ops.VIEW, name="vm"),)), lambda cast,vm: vm.base.cast(cast.dtype).view(vm.st) \
    if cast.dtype.itemsize <= vm.dtype.itemsize and resolve(prod(vm.shape) > vm.st.real_size()) else None),
  # make things that can't be images not images
  (UPat(GroupOp.All-{Ops.BUFFER, Ops.VIEW, Ops.CONST, Ops.DEVICE}, name="u"), lambda u: u.replace(dtype=dt.base) if isinstance(dt:=u.dtype,ImageDType)
   and (prod(u.shape) != prod(dt.shape) or not any(u.shape[x]%4 == 0 for x in u.st.unit_stride_axes())) else None),
  # remove contiguous if we can just view the buffer
  (UPat(Ops.CONTIGUOUS, name="root", src=(UPat(Ops.VIEW, name="view", src=(UPat(Ops.BUFFER, name="buf"),)),)),
   lambda root,view,buf: view if view.st.contiguous and view.size == buf.size else None),
  # contiguous/buffer/copy is already contiguous
  (UPat(Ops.CONTIGUOUS, name="root", src=(UPat((Ops.CONTIGUOUS, Ops.BUFFER, Ops.COPY)),)), lambda root: root.src[0]),
  # support for using a contiguous permuted view instead of the parent view if one exists
  (UPat(Ops.CONTIGUOUS, name="contig", src=(UPat(Ops.VIEW, name="src"),)), found_contiguous),
  (UPat(GroupOp.ALU, name="alu"), replace_contiguous),
  # substitute BITCAST/CONTIGUOUS with BUFFER_VIEW on DISK
  (UPat((Ops.BITCAST, Ops.CONTIGUOUS), src=(UPat.var("x"),), name="tr"), create_buffer_view),
  # put UnaryOps before EXPANDs
  (UPat(GroupOp.Unary, src=UPat(Ops.VIEW, src=(UPat.var("inp"),), name="v"), name="alu"),
   lambda inp,v,alu: inp.alu(alu.op).view(v.st) if resolve(prod(alu.shape) > v.st.real_size()) else None),
])

remove_movement_ops = merge_views+PatternMatcher([
  # NOTE: movement ops are always applied to base
  (UPat(GroupOp.Movement, name="mov", src=(UPat.var("x"),)), lambda x,mov: x.view(unwrap(mov.st))),
  # some masked views can collapse to 0, VIEW(x) -> CONST(VIEW)
  (UPat(Ops.VIEW, name="view"),
   lambda view: view.const_like(0) if (vm:=view.st.views[-1].mask) is not None and any((x[1]-x[0]) == 0 for x in vm) else None),
])

# **** UOp realization

@dataclass(frozen=True)
class GrouperContext:
  assigns: dict[UOp, UOp]                     # maps realized buffers to assigns
  realizes: dict[UOp, None]                   # all the simplified tensor uops we realize
  children: defaultdict[UOp, dict[UOp, None]] # children graph of tensor uops

def realize(ctx:GrouperContext, tr:UOp) -> None: ctx.realizes[tr] = None

def realize_before_view(ctx:GrouperContext, view:UOp, src:UOp) -> None:
  st = unwrap(view.st)
  # fold simple pads
  if len(st.views) == 1 and (m:=st.views[-1].mask) is not None and all_int(src.shape) and resolve(prod(src.shape) >= prod([y-x for x,y in m])):
    return None if can_pad(src, ctx.realizes, cache=dict()) else realize(ctx, src)
  # early realize before expand
  if resolve(prod(src.shape) < prod(st.shape)) and not DONT_REALIZE_EXPAND: return realize(ctx, src)
  # otherwise safety check pads
  return None if (all(v.mask is None for v in st.views) or can_pad(src, ctx.realizes, cache=dict())) else realize(ctx, src)

do_realize = PatternMatcher([
  # always realize SINK parents
  (UPat(Ops.SINK, name="s"), lambda ctx,s: ctx.realizes.update((x.base, None) for x in s.src if x.base.op not in {Ops.CONST, Ops.BIND, Ops.BUFFER})),
  # always realize ASSIGN/CONTIGUOUS/COPY/BUFFER_VIEW
  (UPat({Ops.ASSIGN, Ops.CONTIGUOUS, Ops.COPY, Ops.BUFFER_VIEW}, name="tr"), realize),
  # realize before expand or unsafe pad ops
  (UPat(Ops.VIEW, name="view", src=(UPat(GroupOp.All-{Ops.BUFFER, Ops.CONST, Ops.DEVICE}, name="src"),)), realize_before_view),
  # realize before COPY
  (UPat(Ops.COPY, src=(UPat(), UPat(GroupOp.All-{Ops.BUFFER, Ops.VIEW}, name="tr"))), realize),
])

def recursive_group(tr:UOp, st:ShapeTracker, r:UOp, children:defaultdict[UOp, dict[UOp, None]], realizes:dict[UOp, None],
                    reduce_for_op:dict[UOp, UOp], group:dict[UOp, None], cache:dict[tuple[UOp, ShapeTracker], None]) -> None:
  """recursively search the uop for groupable children, realize the UOp if a child can't group"""
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

def append_uop(ctx:GrouperContext, u:UOp) -> None:
  if u.op is Ops.ASSIGN: ctx.assigns[u.buf_uop] = u
  for s in u.src: ctx.children[s.base][u] = None
create_ctx = PatternMatcher([(UPat(GroupOp.All-{Ops.SINK, Ops.VIEW}, name="u"), append_uop)])

def group_realizes(sink:UOp) -> dict[UOp, None]:
  # start by adding uops that always realize
  sink = graph_rewrite(sink, do_realize+create_ctx, ctx:=GrouperContext({}, {}, defaultdict(dict)))
  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: dict[UOp, UOp] = {}
  double_reduces: list[UOp] = []
  for r in sink.toposort:
    if r.op is not Ops.REDUCE_AXIS: continue
    if FUSE_CONV_BW and r.src[0].base.op is Ops.REDUCE_AXIS and r.src[0] is not r.src[0].base: double_reduces.append(r)
    if r in ctx.realizes: continue
    group: dict[UOp, None] = {}
    recursive_group(r, unwrap(r.st), r, ctx.children, ctx.realizes, reduce_for_op, group, cache={})
    # max one reduceop per kernel
    can_chase = all(tr not in reduce_for_op for tr in group)
    # TODO: forced_realize exists because the scheduler is incapable of checking for self-contained DAGs
    forced_realize = r in group
    # can only have one output
    if not forced_realize and len(group) > 1: forced_realize = True
    # can only fuse assign if no other assign_target is used in the kernel
    if not forced_realize and any(x.op is Ops.ASSIGN for x in group):
      parents = deque((r, *group))
      while parents and not forced_realize:
        p = parents.pop().base
        if (assign:=ctx.assigns.get(p)) is not None and assign not in group: forced_realize, can_chase = True, False
        if p in ctx.realizes: continue
        parents.extend(p.src)
    if forced_realize or not group:
      tr = r
      if can_chase:
        # can chase this down to contiguous children
        st = unwrap(tr.st)
        while len(ctx.children[tr]) == 1:
          tr_next = next(iter(ctx.children[tr]))
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
      ctx.realizes[tr] = None
    reduce_for_op.update((tr, r) for tr in group)
    if FUSE_ARANGE and r.arg[0] is Ops.ADD and r.src[0].base.op is Ops.CONST:
      # maybe fuse arange with its children
      if len(flatten(ctx.children[tr] for tr in group)) != 0:
        for tr in group: del ctx.realizes[tr]
  # fuse double reduces with no other child
  for reduceop in double_reduces:
    top_reduce = reduceop.src[0].base
    if len(ctx.children[top_reduce]) == 1: del ctx.realizes[top_reduce]
  return ctx.realizes

# break the SINK into kernels

@dataclass(frozen=True)
class Kernel:
  ast: UOp
  metadata: tuple[Metadata, ...] = ()
  def __repr__(self): return f"<Kernel {len(list(self.ast.toposort))} {self.ast.op} {self.metadata}>"

@dataclass(frozen=True)
class KernelContext:
  realizes: dict[UOp, None]
  ops_metadata: dict[UOp, Metadata]

DONT_PLACE_IN_KERNEL = {Ops.KERNEL, Ops.ASSIGN, Ops.BUFFER}
def append_to_kernel(ctx:KernelContext, x:UOp):
  new_srcs: list[UOp] = []
  metadata = dict.fromkeys(x.arg.metadata)
  if (m:=ctx.ops_metadata.get(x.arg.ast)) is not None: metadata[m] = None
  for s in x.src:
    if s.op in DONT_PLACE_IN_KERNEL or s in ctx.realizes: new_srcs.append(s)
    else:
      new_srcs.extend(s.src)
      if (m:=ctx.ops_metadata.get(s)) is not None: metadata[m] = None
  if (new_src:=tuple(dedup(new_srcs))) != x.src: return x.replace(src=new_src, arg=Kernel(x.arg.ast, tuple(metadata)))
  return x.replace(arg=Kernel(x.arg.ast, new_metadata)) if (new_metadata:=tuple(metadata)) != x.arg.metadata else None

create_kernels = merge_views+PatternMatcher([
  # always give assign a kernel
  (UPat.assign(UPat.var("b"), UPat(GroupOp.All-{Ops.KERNEL}), name="x"), lambda x,b: b.assign(UOp(Ops.KERNEL, src=(b,)+x.src, arg=Kernel(x)))),
  # otherwise check if need to assign this UOp to a new buffer
  (UPat(GroupOp.All-DONT_PLACE_IN_KERNEL, name="x"), lambda ctx,x: UOp(Ops.ASSIGN, x.dtype, (b:=UOp.new_buffer(x.device, x.size, x.dtype).view(x.st),\
    UOp(Ops.KERNEL, src=(b,)+x.src, arg=Kernel(x)))) if x in ctx.realizes else None),
  # walk back the local graph until we reach a buffer/assign parent
  (UPat(Ops.KERNEL, name="x"), append_to_kernel),
  # remove CONST/BIND from SINK
  (UPat(Ops.SINK, name="x"), lambda x: x.replace(src=new_src)
    if (new_src:=tuple(dedup(s.base for s in x.src if s.op not in {Ops.CONST,Ops.BIND}))) != x.src else None),
])

# **** fix kernel AST

# ** create buffer ops + enumerate buffers

def load_buf(ctx:list[UOp], x:UOp):
  if x not in ctx: ctx.append(x)
  return UOp(Ops.LOAD, x.dtype, (UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(x.size), (), ctx.index(x)), unwrap(x.st).to_uop()))

add_buffer_ops = PatternMatcher([
  # LOAD
  (UPat(Ops.BUFFER, name="x"), load_buf),
  # STORE (except for COPY/BUFFER_VIEW)
  (UPat(Ops.SINK, src=(UPat((Ops.COPY, Ops.BUFFER_VIEW), name="x"),)), lambda x:x),
  (UPat(Ops.SINK, src=(UPat(GroupOp.All-{Ops.STORE}, name="x"),)),
   lambda x: UOp.store(UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(x.size), (), 0), ShapeTracker.from_shape(x.shape).to_uop(), x).sink()),
])

# ** push views to buffer ops

def apply_swizzle(u:UOp) -> UOp:
  with Context(TRACK_MATCH_STATS=0): return graph_rewrite(u, view_left)

def swizzle_r(r:UOp, src:UOp, st:ShapeTracker) -> UOp:
  input_st = ShapeTracker.from_shape(unwrap(src.st).shape)
  tmp = input_st.permute(tuple(i for i in range(len(input_st.shape)) if i not in r.axis_arg)+r.axis_arg)
  prshape = prod(rshape:=tmp.shape[-len(r.axis_arg):])
  strides = strides_for_shape(rshape)
  nv = [View.create(v.shape+rshape, tuple(x*prshape for x in v.strides)+strides,
                    v.offset*prshape, v.mask+tuple((0,s) for s in rshape) if v.mask is not None else None) for v in st.views]
  # update input_st and axis
  new_input_st = tmp + ShapeTracker(tuple(nv))
  new_axis = tuple(range(len(st.shape), len(st.shape) + len(r.axis_arg)))
  return apply_swizzle(src.view(new_input_st)).r(r.arg[0], new_axis).view(ShapeTracker.from_shape(st.shape))

def reduceop_view_right(r:UOp, v:UOp, src:UOp) -> UOp:
  if not (swizzle_st:=unwrap(v.st)).contiguous or v.size != src.size: raise AssertionError(f"can't push {v} down through {src}")
  output_shape = swizzle_st.reduce(r.axis_arg)
  return src.r(r.arg[0], tuple(i for i,(s,u) in enumerate(zip(src.shape, output_shape)) if s != u)).view(ShapeTracker.from_shape(output_shape))

def elementwise_view_right(root:UOp) -> UOp|None:
  if len(swizzles:=[x for x in root.src if x.base is not x]) == 0: return None
  assert all(x.base.st is not None for x in swizzles), f"found shapeless VIEW src in {root}"
  assert all_same([x.base.size for x in swizzles]), f"swizzle inputs must have the same size {swizzles}"
  # push the swizzle from src to root
  output_swizzle = swizzles[0]
  new_input_st = ShapeTracker.from_shape(output_swizzle.base.shape)
  ret = root.replace(src=tuple(x if x.st is None else x.base if x in swizzles else apply_swizzle(x.view(new_input_st)) for x in root.src))
  return ret.view(ShapeTracker.from_shape(output_swizzle.shape))

def merge_double_reduce(root:UOp, first_reduce:UOp) -> UOp:
  assert root.arg[0] == first_reduce.arg[0], "can't merge reduceops with different alu"
  assert not any(x.op is Ops.REDUCE_AXIS for x in first_reduce.src[0].toposort), "can't merge more than two reduceops at a time"
  return first_reduce.replace(arg=(first_reduce.arg[0], root.axis_arg+first_reduce.axis_arg))

# push VIEW to children
view_right = merge_views+PatternMatcher([
  # STORE(.., ASSIGN(VIEW(BUFFER), new_val)) -> VIEW(STORE(.., new_val))
  (UPat(Ops.STORE, src=(UPat.var("b"), UPat.var("st"), UPat.assign(UPat.var("target"), UPat.var("val")))),
   lambda b,target,st,val: apply_swizzle(UOp.store(b, st, val).view(target.st))),
  # STORE is the last child, so we just merge the ShapeTrackers and store the base
  (UPat(Ops.STORE, src=(UPat.var("b"), UPat.var("st"), UPat(Ops.VIEW, src=(UPat.var("val"),)))), lambda b,st,val: UOp.store(b, st.view(val.st), val)),
  # REDUCE(src.view(contiguous=False)) -> REDUCE(src.view(contiguous=True)).view()
  (UPat(Ops.REDUCE_AXIS, src=(UPat.var("src"),), name="r").view(name="v"), lambda v,r,src: None if v.st.contiguous else swizzle_r(r, src, v.st)),
  # REDUCE(src.view()) -> REDUCE(src).view()
  (UPat(Ops.REDUCE_AXIS, src=(UPat.var("src").view(name="v"),), name="r"), reduceop_view_right),
  # ALU(src.view()) -> ALU(src).view()
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN, Ops.CONTIGUOUS, Ops.STORE), name="root"), elementwise_view_right),
  # double reduce op collapses to a single reduce op
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.REDUCE_AXIS, name="first_reduce"),), name="root"), merge_double_reduce),
])

# ** unbind variables

def unbind_shapetracker(ctx:dict[Variable, int], x:UOp) -> UOp|None:
  st = unwrap(x.st).simplify()
  if any(x.op is Ops.BIND for x in st.vars()):
    st, var_vals = st.unbind()
    ctx.update(var_vals)
  return st.to_uop() if st != x.st else None

def unbind_variable(ctx:dict[Variable, int], bind:UOp, var:UOp, val:UOp):
  ctx[var.replace(src=())] = val.arg
  return var
unbind_vars = PatternMatcher([(UPat(Ops.BIND, name="bind", src=(UPat.var("var"), UPat.cvar("val"))), unbind_variable),])

# ** fix_kernel_ops

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
  # BIND in shapetracker becomes DEFINE_VAR
  (UPat(Ops.VIEW, name="x"), unbind_shapetracker),
  # remove CONTIGUOUS/DEVICE
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("x"),)), lambda x: x),
  (UPat(Ops.VIEW, name="view", src=(UPat(Ops.DEVICE),)), lambda view: view.replace(src=())),
  # no ImageDType after load
  (UPat(GroupOp.All-{Ops.DEFINE_GLOBAL}, name="x"), lambda x: x.replace(dtype=x.dtype.base) if isinstance(x.dtype, ImageDType) else None),
  # if this kernel also assigns to the loaded buffer, ensure we can index it correctly
  (UPat(Ops.LOAD, src=(UPat.var("glbl"), UPat.var("view"))), check_load_st),
])

# TODO: replace this with the KERNEL UOp
@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...]

def schedule_uop(sink:UOp, var_vals:dict[Variable, int]) -> ScheduleItem:
  assert sink.op is Ops.ASSIGN and sink.src[1].op is Ops.KERNEL, f"{sink} must be ASSIGN"
  # substitute kernel sources for the target buffer
  ast = sink.src[1].arg.ast.substitute({s.src[1].arg.ast:s.src[0] for s in sink.src[1].src if s.op is Ops.ASSIGN}).sink()
  # add buffer ops
  ast = graph_rewrite(ast, add_buffer_ops, bufs:=[sink.buf_uop], bottom_up=True)
  if ast.op is Ops.SINK and not all_same(dev:=[x.device for x in bufs]): raise RuntimeError(f"all buffers must be on the same device: {dev}")
  # unbind_vars + push views to edges
  ast = graph_rewrite(graph_rewrite(ast, unbind_vars+view_left, ctx=var_vals), view_right)
  # fix_kernel_ops
  ast = graph_rewrite(ast, fix_kernel_ops, var_vals)
  # create subbuffer
  if ast.op is Ops.BUFFER_VIEW: buffers[bufs[0]] = (base:=bufs[1].buffer).view(ast.size, ast.dtype, ast.arg[1]*base.dtype.itemsize)
  return ScheduleItem(ast, tuple(dedup([x.buffer for x in bufs])), sink.src[1].arg.metadata)

PROCESS_REPLAY_CAPTURE:dict[str, bytes] = {}
if CAPTURE_PROCESS_REPLAY:
  @atexit.register
  def save_process_replay():
    for k,v in PROCESS_REPLAY_CAPTURE.items(): diskcache_put("schedule_process_replay", k, v, prepickled=True)

# **** schedule creation and toposort

@track_rewrites(name_fxn=lambda r: f"Schedule {pluralize('Kernel', len(r[0]))}"+(f" (with_{pluralize('Var', len(r[1]))})" if len(r[1]) != 0 else ""))
def create_schedule_with_vars(big_sink:UOp) -> tuple[list[ScheduleItem], dict[Variable, int], dict[UOp, UOp]]:
  # remove_movement_ops + sym
  tensor_map = graph_rewrite_map(big_sink, remove_movement_ops+sym, ctx={})

  # display the cleaned up tensor graph
  if getenv("VIZ"): graph_rewrite(tensor_map[big_sink], PatternMatcher([]), name="View Tensor Graph")

  # get realizes
  sink = tensor_map[big_sink]
  realize_map = group_realizes(sink)
  # map tensor metadata to simplified ops
  ops_metadata = {v:k.metadata for k,v in tensor_map.items() if k.base.op not in {Ops.CONST, Ops.DEVICE} and isinstance(k.metadata, Metadata)}
  # create kernels
  kernel_map = graph_rewrite_map(sink, create_kernels, ctx=KernelContext(realize_map, ops_metadata), bottom_up=True)
  sched_sink = kernel_map[sink]
  type_verify(list(sched_sink.toposort), kernel_spec)

  # map tensors to buffer/const, optionally apply a VIEW on top
  becomes_map: dict[UOp, UOp] = {}
  for k,v in tensor_map.items():
    # if we created a KERNEL for this tensor, map it to the assigned buffer
    if (a:=kernel_map.get(v.base)) is not None and a.op is Ops.ASSIGN:
      becomes_map[k] = a.src[0] if v is v.base else a.src[0].view(unwrap(v.st))
    # tensors can also simplify to an existing buffer/const
    else:
      if k is v: continue
      if v.base.op is Ops.BUFFER: becomes_map[k] = v
      if v.base.op is Ops.CONST and all_int(v.shape): becomes_map[k] = v

  # if a kernel depends on a buffer, and that buffer is later assigned to, make the assign depend on the kernel's assign
  kernel_assign: dict[UOp, UOp] = {}
  assign_rep: dict[UOp, UOp] = {}
  for u in sched_sink.toposort:
    if u.op is not Ops.ASSIGN: continue
    kernel_assign[u.buf_uop] = u
    for s in u.src[1].src:
      if s.op is not Ops.BUFFER or s is u.buf_uop or (a:=kernel_assign.get(s)) is None: continue
      if any(x.op is Ops.ASSIGN and x.buf_uop is s for x in u.toposort):
        raise RuntimeError(f"cycle detected in graph, kernel for {u.buf_uop} must either depend on ASSIGN or BUFFER")
      assign_rep[a] = kernel_assign[s] = a.replace(src=a.src+(u,))
  if assign_rep:
    sched_sink = sched_sink.substitute(assign_rep)
    type_verify(list(sched_sink.toposort), kernel_spec)

  # display the final graph
  if getenv("VIZ"): graph_rewrite(sched_sink, PatternMatcher([]), name="View Kernel Graph")

  # final toposort (bfs)
  children: dict[UOp, list[UOp]] = {}
  in_degree: dict[UOp, int] = {}
  for u in sched_sink.toposort:
    if u.op is not Ops.ASSIGN: continue
    in_degree[u] = 0
    for s in u.src[1].src:
      if s.op is not Ops.ASSIGN: continue
      children.setdefault(s, []).append(u)
      in_degree[u] += 1

  queue = deque(k for k,v in in_degree.items() if v == 0)
  schedule: list[ScheduleItem] = []
  var_vals: dict[Variable, int] = {}
  while queue:
    u = queue.popleft()
    schedule.append(schedule_uop(u, var_vals))
    # increment the refcount of the target buf (this is required by the JIT and memory planner)
    u.buf_uop.buffer.ref(1)
    for x in children.get(u, []):
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)

  # confirm everything was scheduled correctly
  if len(schedule) != (kc:=len(in_degree)): raise RuntimeError(f"cycle detected in graph, created {kc} kernels but only scheduled {len(schedule)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")
  # capture process replay
  if CAPTURE_PROCESS_REPLAY:
    with Context(PICKLE_BUFFERS=0): PROCESS_REPLAY_CAPTURE[str(big_sink.key)] = pickle.dumps((big_sink, ContextVar._cache, [x.ast for x in schedule]))
  return schedule, var_vals, becomes_map
