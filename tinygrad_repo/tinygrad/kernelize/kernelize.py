from dataclasses import dataclass
from tinygrad.uop.ops import UOp, Ops, GroupOp, PatternMatcher, UPat, graph_rewrite, graph_rewrite_map, identity_element, resolve, sint
from tinygrad.uop.ops import track_rewrites, _substitute
from tinygrad.uop.spec import type_verify, tensor_uop_spec
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.helpers import Metadata, all_int, all_same, colored, prod, dedup, unwrap, getenv, pluralize, FUSE_ARANGE, DEBUG, SPLIT_REDUCEOP
from tinygrad.dtype import ImageDType
from tinygrad.kernelize.multi import multi_pm
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, strides_for_shape, get_contraction_with_reduce
from tinygrad.kernelize.grouper import group_realizes, ALWAYS_CONTIGUOUS

# creation can recurse a lot
import sys
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

def copy_reorder_view(copy:UOp, view:UOp, base:UOp):
  if prod(view.shape) < prod(base.shape): return view.contiguous().copy_to_device(copy.device)
  return base.copy_to_device(copy.device).view(view.arg)

sym = symbolic_simple+PatternMatcher([
  # UOp with size 0 is zero
  (UPat(GroupOp.All-{Ops.SINK}, name="root"), lambda root: root.const_like(0) if root.base.st is not None and root.size == 0 else None),
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
  (UPat(Ops.COPY, name="root", src=(UPat.cvar("x"), UPat(Ops.DEVICE))), lambda root,x: root.const_like(x.arg)),
  # non device changing COPY is a NOOP
  (UPat(Ops.COPY, name="c", src=(UPat.var("x"), UPat(Ops.DEVICE))), lambda c,x: x if c.device == x.device else None),
  # store a shrink before COPY, otherwise view after the COPY
  (UPat(Ops.COPY, src=(UPat(Ops.VIEW, src=(UPat.var("base"),), name="view"), UPat(Ops.DEVICE)), name="copy"), copy_reorder_view),
  # remove cast to image when it's already a contiguous image
  (UPat(Ops.CAST, name="cast", src=(UPat(Ops.VIEW, name="vm", src=(UPat(Ops.CONTIGUOUS, name="base"),)),)),
   lambda cast,base,vm: base.view(vm.st) if isinstance(cast.dtype, ImageDType) and isinstance(base.dtype, ImageDType) else None),
  # CAST before masking constants
  (UPat.cvar("x").view().cast(name="c"), lambda x,c: x.cast(c.dtype).view(c.src[0].arg)),
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
  (UPat(Ops.ASSIGN, src=(UPat.var("t"), UPat.var("x"))), lambda x,t: x.contiguous() if t.base.op not in {Ops.BUFFER, Ops.BUFFER_VIEW} and
   not (t.base.op is Ops.MSTACK and all(x.op is Ops.BUFFER for x in t.base.src)) else None),
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
  kernel = UOp(Ops.KERNEL, src=(b,)+x.src, arg=Kernel(x.sink(), m if (m:=x.metadata) else ()))
  buffer = b.base if b.size == b.base.size else UOp(Ops.BUFFER_VIEW, b.dtype, (b.base,), (b.size, b.arg.views[0].offset))
  return buffer.assign(kernel).reshape(x.shape)

DONT_PLACE_IN_KERNEL = {Ops.KERNEL, Ops.ASSIGN, Ops.BUFFER, Ops.MSELECT, Ops.MSTACK, Ops.MULTI}
def append_to_kernel(x:UOp):
  new_srcs: list[UOp] = []
  metadata = x.arg.metadata
  for s in x.src:
    if s.op in DONT_PLACE_IN_KERNEL: new_srcs.append(s)
    else:
      new_srcs.extend(s.src)
      # NOTE: because const and device are shared UOps they don't change metadata
      # NOTE: if it's a reshape after ASSIGN we're not fusing that parent kernel
      if s.base.op not in {Ops.CONST, Ops.DEVICE} and (not (s.op is Ops.RESHAPE and s.base.op is Ops.ASSIGN)) and (m:=s.metadata): metadata += m
  if (new_src:=tuple(dedup(new_srcs))) != x.src: return x.replace(src=new_src, arg=Kernel(x.arg.ast, tuple(dedup(metadata))))

create_kernels = PatternMatcher([
  # always give assign/gbarrier a kernel
  (UPat.assign(UPat.var("b"), UPat(GroupOp.All-{Ops.KERNEL}), name="x"), create_kernel),
  (UPat(Ops.GBARRIER, src=(UPat.var("x"),)), create_kernel),
  # walk back the local graph until we reach a realized source
  (UPat(Ops.KERNEL, name="x"), append_to_kernel),
  # push RESHAPE through MSELECT
  (UPat(Ops.MSELECT, src=(UPat(Ops.RESHAPE, name="r"),), name="ms"), lambda ms,r: r.src[0].mselect(ms.arg).reshape(r.arg)),
  # push RESHAPE through MSTACK
  (UPat(Ops.MSTACK, src=UPat(Ops.RESHAPE), name="ms"),
   lambda ms: UOp(Ops.MSTACK, ms.dtype, tuple(x.src[0] for x in ms.src)).reshape(ms.src[0].arg)),
])

# **** swizzler

merge_views = PatternMatcher([
  # merge adjacent views
  (UPat(Ops.VIEW, src=(UPat(Ops.VIEW, name="v1"),), name="v2"), lambda v1,v2: v1.replace(arg=v1.arg+v2.arg)),
  # replace MovementOps with VIEW
  (UPat(GroupOp.Movement, src=(UPat.var("x"),), name="mop"), lambda mop,x: x.base.view(mop.st)),
  # remove NOOP views
  (UPat.var("x").view(name="view"), lambda x,view: x if x.st is not None and view.st.contiguous and view.shape == x.shape else None),
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
  (UPat(Ops.VIEW, src=(UPat({*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.BIND, Ops.LOAD, Ops.STORE, Ops.VALID}, name="e"),), name="view"),
   lambda e,view: e.replace(src=tuple(s.view(view.st) for s in e.src))),
  # if there's ones added after reduce, put this before the reduce
  (UPat(Ops.VIEW, src=(UPat(Ops.REDUCE_AXIS, src=(UPat.var("src"),), name="r"),), name="view"), reduce_push_add_ones),
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
  (UPat(GroupOp.All-{Ops.SINK, Ops.GBARRIER, Ops.REDUCE_AXIS}, name="root"), elementwise_view_right),
  # merge axes for double reduce (invert of SPLIT_REDUCEOP=1)
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.REDUCE_AXIS, name="r1"),), name="r2"),
   lambda r1,r2: r1.replace(arg=(r1.arg[0], r2.arg[1]+r1.arg[1])) if r1.arg[0] is r2.arg[0] else None),
])

# **** fix kernel AST

add_buffer_ops = PatternMatcher([
  # LOAD
  (UPat(Ops.BUFFER, name="x"), lambda ctx,x: UOp.load(UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(x.size), (), ctx.index(x)).view(x.st),)),
  # STORE (except for meta ops)
  (UPat(Ops.SINK, src=(UPat(GroupOp.Meta, name="x"),)), lambda x:x),
  (UPat(Ops.SINK, src=UPat(GroupOp.All-{Ops.STORE}), name="sink"), lambda ctx,sink:
   UOp.sink(*[UOp.store(UOp(Ops.DEFINE_GLOBAL, (s:=x.base).dtype.ptr(ctx[i].size), (), i).view(s.st), s) for i,x in enumerate(sink.src)])),
  # passthrough ASSIGN
  (UPat(Ops.ASSIGN, name="x"), lambda x: x.src[1]),
  # VALID
  (UPat(Ops.VIEW, src=(UPat.cvar(),), name="self"), UOp.valid),
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
  (UPat((Ops.CONTIGUOUS, Ops.MSELECT), src=(UPat.var("x"),)), lambda x: x),
  (UPat(Ops.VIEW, src=(UPat(Ops.DEVICE),), name="view"), lambda view: view.replace(src=())),
  # no ImageDType after index
  (UPat(GroupOp.All-{Ops.DEFINE_GLOBAL, Ops.VIEW}, name="x"), lambda x: x.replace(dtype=x.dtype.base) if isinstance(x.dtype, ImageDType) else None),
  # if this kernel also assigns to the loaded buffer, ensure we can index it correctly
  (UPat(Ops.LOAD, src=(UPat.var("glbl").view(name="view"),)), check_load_st),
])

replace_globals = PatternMatcher([
  # replace ASSIGN with the target BUFFER
  (UPat(Ops.ASSIGN, src=(UPat(Ops.BUFFER), UPat(Ops.KERNEL)), name="assign", allow_any_len=True), lambda assign: assign.src[0]),
  # HACK: select the 0 branch of MSTACK (the device is wrong after this, is that okay?)
  (UPat(Ops.MSTACK, name="x"), lambda x: x.src[0]),
])

def fix_kernel_ast(k:UOp) -> UOp|None:
  if k.arg.ast.op in GroupOp.Meta or all(s.op is Ops.STORE for s in k.arg.ast.src): return None
  # replace global memory ops with the BUFFER they write to
  ast = graph_rewrite(k.arg.ast, replace_globals, bottom_up=True, name="replace globals")
  # push views to edges
  ast = graph_rewrite(graph_rewrite(ast, view_left, name="Main View Left"), view_right, name="Main View Right")
  # replace buffer with define_global + add load/store last
  bufs = []
  for s in k.src:
    s = s.buf_uop
    # traverse back through MSELECT and MSTACK. HACK: 0 branch of MSTACK only
    while s.op in {Ops.MSELECT, Ops.MSTACK}: s = s.src[0]
    bufs.append(s)
  ast = graph_rewrite(ast, view_left+add_buffer_ops+fix_kernel_ops, bufs, bottom_up=True, name="replace buffer")
  if ast.op is Ops.SINK and not all_same([x.device for x in k.src]):
    raise RuntimeError(f"all buffers must be on the same device: {tuple(b.buf_uop.buffer for b in k.src)}")
  return k.replace(arg=Kernel(ast, k.arg.metadata))

create_ast = PatternMatcher([(UPat(Ops.KERNEL, name="k"), fix_kernel_ast),])

# ** add metadata of KERNEL outputs

def append_metadata(root:UOp, k:UOp):
  if not root.metadata or (new_metadata:=tuple(dedup(k.arg.metadata+root.metadata))) == k.arg.metadata: return None
  return root.replace(src=(root.src[0], k.replace(arg=Kernel(k.arg.ast, new_metadata)))+root.src[2:])

replace_metadata = PatternMatcher([(UPat(Ops.ASSIGN, src=(UPat(), UPat(Ops.KERNEL, name="k")), name="root", allow_any_len=True), append_metadata),])

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
   lambda alu, view: alu.replace(src=tuple(apply_swizzle(x.view(view.arg)).fuse() for x in alu.src))),

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

add_gbarrier = PatternMatcher([(UPat(GroupOp.All-{Ops.GBARRIER, Ops.ASSIGN}, name="x"),
                                lambda ctx,x: x.replace(tag=1).gbarrier() if x in ctx and x.tag is None else None)])

# TODO: get this from the device through GrouperOpts
DEVICE_MAX_BUFS = {"METAL":32, "WEBGPU":8}

def limit_bufs(root:UOp):
  # check if backend has a buffer limit
  device = root.device if isinstance(root.device, str) else root.device[0].split(":")[0]
  if not (MAX_BUFS:=getenv("MAX_KERNEL_BUFFERS", DEVICE_MAX_BUFS.get(device, 0))): return None
  # count number of unique buffers flowing into this op
  bufs: set[UOp] = set()
  def gate_input(u:UOp):
    if (is_load:=(u.op in {Ops.BUFFER, Ops.GBARRIER, Ops.ASSIGN, Ops.MSTACK})): bufs.add(u)
    return not is_load
  root.toposort(gate=gate_input)
  # NOTE: this -1 is for the output buffer
  if len(bufs)>=MAX_BUFS-1:
    return root.replace(src=tuple(s if s.base in bufs else s.replace(tag=1).gbarrier() for s in root.src))

finalize_gbarrier = PatternMatcher([
  # if an op takes more than one input, check combined LOADs don't exceed device limits
  (UPat(set.union(GroupOp.Binary, GroupOp.Ternary), name="root"), limit_bufs),
  # merge gbarrier
  (UPat((Ops.GBARRIER, Ops.CONTIGUOUS), src=(UPat(Ops.GBARRIER),), name="x"), lambda x: x.src[0]),
  # add contiguous to VIEW before GBARRIER
  (UPat(Ops.GBARRIER, src=(UPat(Ops.VIEW,),), name="x"), lambda x: x.src[0].contiguous().gbarrier()),
  # remove gbarrier on constants without a contiguous
  (UPat(Ops.GBARRIER, src=(UPat(Ops.CONST),), name="x"), lambda x: x.src[0]),
  # simplify views
  (UPat(Ops.VIEW, src=(UPat.var('x')), name="v"), lambda x,v: x.view(new_st) if (new_st:=v.arg.simplify()) != v.arg else None),
])

remove_tags = PatternMatcher([(UPat(GroupOp.All, name="x"), lambda x: x.replace(tag=None) if x.tag is not None else None)])

@track_rewrites(name=lambda sink,ret: f"Schedule {pluralize('Kernel',len([u for u in ret[sink].toposort() if u.op is Ops.KERNEL]))}")
def get_kernelize_map(sink:UOp) -> dict[UOp, UOp]:
  """
  Function to transform the Tensor UOp graph into a version with Ops.KERNEL

  Args:
    sink: The Ops.SINK rooting the Tensor graph.

  Returns:
    Map transforming each UOp in the sink to the Ops.KERNEL graph.
  """

  # multi + merge_views + simplify
  tensor_map = graph_rewrite_map(sink, multi_pm+do_fuse+merge_views+sym+replace_contiguous, ctx={}, name="merge_views")

  # display the cleaned up tensor graph
  if getenv("VIZ"): graph_rewrite(tensor_map[sink], PatternMatcher([]), name="View Tensor Graph")

  # insert gbarriers in places determined by the realize map
  realize_map = group_realizes(tensor_map[sink])
  tensor_map = graph_rewrite_map(tensor_map[sink], add_gbarrier, ctx=realize_map, bottom_up=True, input_map=tensor_map, name="insert_gbarrier")
  tensor_map = graph_rewrite_map(tensor_map[sink], finalize_gbarrier+remove_tags, input_map=tensor_map, name="finalize_gbarrier")

  # TODO: move view_left/view_right here

  # group into kernels (this is context-free)
  tensor_map = graph_rewrite_map(tensor_map[sink], create_kernels, input_map=tensor_map, name="create_kernels")

  # if a kernel depends on a buffer, and that buffer is later assigned to, make the assign depend on the kernel's assign
  kernel_assign: dict[UOp, UOp] = {}
  assign_rep: dict[UOp, UOp] = {}
  for u in tensor_map[sink].toposort():
    if u.op is not Ops.ASSIGN: continue
    kernel_assign[u.buf_uop] = u
    for s in u.src[1].src:
      # TODO: this is probably broken for MSELECT/MSTACK
      if s.op is not Ops.BUFFER or s is u.buf_uop or (a:=kernel_assign.get(s)) is None: continue
      if any(x.op is Ops.ASSIGN and x.buf_uop is s for x in u.toposort()):
        raise RuntimeError(f"cycle detected in graph, kernel for {u.buf_uop} must either depend on ASSIGN or BUFFER")
      assign_rep[a] = kernel_assign[s] = a.replace(src=a.src+(u,))
  if assign_rep:
    tensor_map = graph_rewrite_map(tensor_map[sink], _substitute, ctx=assign_rep, bottom_up=True, input_map=tensor_map, name="fix_assign")

  # finally, create the AST for kernels
  tensor_map = graph_rewrite_map(tensor_map[sink], create_ast+replace_metadata, bottom_up=True, input_map=tensor_map, name="create_ast")

  # display the final graph
  sched_sink = tensor_map[sink]
  if getenv("VIZ"): graph_rewrite(sched_sink, PatternMatcher([]), name="View Kernel Graph")

  # verify Kernels match the spec
  if __debug__: type_verify(list(sched_sink.toposort()), tensor_uop_spec)

  return tensor_map
