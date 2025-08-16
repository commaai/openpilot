from dataclasses import dataclass
from tinygrad.uop.ops import UOp, Ops, GroupOp, PatternMatcher, UPat, graph_rewrite, graph_rewrite_map, identity_element, resolve
from tinygrad.uop.ops import track_rewrites, _substitute
from tinygrad.uop.spec import type_verify, tensor_uop_spec
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.helpers import Metadata, all_int, all_same, prod, dedup, unwrap, getenv, pluralize, FUSE_ARANGE, DEBUG, SPLIT_REDUCEOP
from tinygrad.dtype import ImageDType
from tinygrad.schedule.multi import multi_pm
from tinygrad.schedule.grouper import group_realizes, ALWAYS_CONTIGUOUS
from tinygrad.codegen.opt.swizzler import merge_views, apply_swizzle, swizzle_reduceop

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

DONT_PLACE_IN_KERNEL = {Ops.KERNEL, Ops.ASSIGN, Ops.BUFFER, Ops.MSELECT, Ops.MSTACK, Ops.MULTI, Ops.BIND}
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
  # always give assign/contiguous a kernel
  (UPat.assign(UPat.var("b"), UPat(GroupOp.All-{Ops.KERNEL}), name="x"), create_kernel),
  (UPat(Ops.CONTIGUOUS, name="x"), create_kernel),
  # walk back the local graph until we reach a realized source
  (UPat(Ops.KERNEL, name="x"), append_to_kernel),
  # push RESHAPE through MSELECT
  (UPat(Ops.MSELECT, src=(UPat(Ops.RESHAPE, name="r"),), name="ms"), lambda ms,r: r.src[0].mselect(ms.arg).reshape(r.arg)),
  # push RESHAPE through MSTACK
  (UPat(Ops.MSTACK, src=UPat(Ops.RESHAPE), name="ms"),
   lambda ms: UOp(Ops.MSTACK, ms.dtype, tuple(x.src[0] for x in ms.src)).reshape(ms.src[0].arg)),
])

# **** fix kernel AST

def unbind_view(x:UOp):
  if any(x.op is Ops.BIND for x in x.arg.vars()): return x.replace(arg=x.arg.unbind()[0])
  return None

replace_buffers = PatternMatcher([
  # replace ASSIGN with the target BUFFER
  (UPat(Ops.ASSIGN, src=(UPat((Ops.BUFFER, Ops.LOAD)), UPat(Ops.KERNEL)), name="assign", allow_any_len=True), lambda assign: assign.src[0]),
  # HACK: select the 0 branch of MSTACK (the device is wrong after this, is that okay?)
  (UPat(Ops.MSTACK, name="x"), lambda x: x.src[0]),
  # LOAD
  (UPat(Ops.BUFFER, name="x"), lambda ctx,x: UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(x.size), (), ctx.index(x)).load()),
  # no SINK for meta ops
  (UPat(Ops.SINK, src=(UPat(Ops.CONTIGUOUS, src=(UPat(GroupOp.Meta, name="x"),),))), lambda x:x),
  # STORE (except for meta ops)
  (UPat(Ops.SINK, src=UPat(GroupOp.All-{Ops.STORE}), name="sink"), lambda ctx,sink:
   UOp.sink(*[UOp.store(UOp(Ops.DEFINE_GLOBAL, (s:=x.base).dtype.ptr(ctx[i].size), (), i).view(s.st), s) for i,x in enumerate(sink.src)],
            arg=sink.arg)),
  # remove CONTIGUOUS/DEVICE from kernel AST
  (UPat((Ops.CONTIGUOUS, Ops.MSELECT), src=(UPat.var("x"),)), lambda x: x),
  (UPat(Ops.VIEW, src=(UPat(Ops.DEVICE),), name="view"), lambda view: view.replace(src=())),
  # passthrough ASSIGN (but let MSTACK process first)
  (UPat(Ops.ASSIGN, src=(UPat(GroupOp.All-{Ops.MSTACK}), UPat()), name="x"), lambda x: x.src[1]),
  # remove any BINDs from VIEWS
  (UPat(Ops.VIEW, src=(UPat(), UPat((Ops.BIND, Ops.DEFINE_VAR))), allow_any_len=True, name="x"), lambda x: x.replace(src=x.src[0:1])),
  # remove any BINDs from DEFINE_VARs
  (UPat(Ops.BIND, name="x"), lambda x: x.src[0]),
  # remove BINDs from ShapeTrackers
  (UPat(Ops.VIEW, name="x"), unbind_view),
])

def fix_kernel_ast(k:UOp) -> UOp|None:
  if k.arg.ast.op in GroupOp.Meta or all(s.op is Ops.STORE for s in k.arg.ast.src): return None
  # replace buffer with define_global + add load/store last
  bufs = []
  for s in k.src:
    if s.op is Ops.BIND: continue
    s = s.buf_uop
    # traverse back through MSELECT and MSTACK. HACK: 0 branch of MSTACK only
    while s.op in {Ops.MSELECT, Ops.MSTACK}: s = s.src[0]
    bufs.append(s)
  # replace global memory ops with the BUFFER they write to
  # NOTE: merge_views is needed to unbind the reshapes
  ast = graph_rewrite(k.arg.ast, merge_views+replace_buffers, bufs, bottom_up=True, name="replace buffers")
  if ast.op is Ops.SINK and not all_same([x.device for x in k.src if x.op is not Ops.BIND]):
    raise RuntimeError(f"all buffers must be on the same device: {tuple(b.buf_uop.buffer for b in k.src)}")
  return k.replace(arg=Kernel(ast, k.arg.metadata))

create_ast = PatternMatcher([
  (UPat(Ops.KERNEL, name="k"), fix_kernel_ast),
  (UPat(Ops.DEFINE_VAR, src=(UPat(),), allow_any_len=True, name="x"), lambda x: x.replace(src=())),
])

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
    if is_contiguous:=(x.op is Ops.CONTIGUOUS): found_contiguous[x] = x.replace(src=(UOp(Ops.VIEW, arg=x.st), UOp.unique()))
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

add_contiguous = PatternMatcher([(UPat(GroupOp.All-{Ops.CONTIGUOUS, Ops.ASSIGN}, name="x"),
                                lambda ctx,x: x.replace(tag=1).contiguous() if x in ctx and x.tag is None else None)])

# TODO: get this from the device through GrouperOpts
DEVICE_MAX_BUFS = {"METAL":32, "WEBGPU":8}

def limit_bufs(root:UOp):
  # check if backend has a buffer limit
  device = root.device if isinstance(root.device, str) else root.device[0].split(":")[0]
  if not (MAX_BUFS:=getenv("MAX_KERNEL_BUFFERS", DEVICE_MAX_BUFS.get(device, 0))): return None
  # count number of unique buffers flowing into this op
  bufs: set[UOp] = set()
  def gate_input(u:UOp):
    if (is_load:=(u.op in {Ops.BUFFER, Ops.CONTIGUOUS, Ops.ASSIGN, Ops.MSTACK})): bufs.add(u)
    return not is_load
  root.toposort(gate=gate_input)
  # NOTE: this -1 is for the output buffer
  if len(bufs)>=MAX_BUFS-1:
    return root.replace(src=tuple(s if s.base in bufs else s.replace(tag=1).contiguous() for s in root.src))

def view_add_srcs(x:UOp):
  if len(avars:=x.arg.vars()) and len(x.src) == 1:
    return x.replace(src=x.src+tuple(avars))
  return None

finalize_contiguous = PatternMatcher([
  # if an op takes more than one input, check combined LOADs don't exceed device limits
  (UPat(set.union(GroupOp.Binary, GroupOp.Ternary), name="root"), limit_bufs),
  # merge contiguous
  (UPat(Ops.CONTIGUOUS, src=(UPat(Ops.CONTIGUOUS),), name="x"), lambda x: x.src[0]),
  # simplify views
  (UPat(Ops.VIEW, src=(UPat.var('x')), name="v"), lambda x,v: x.view(new_st) if (new_st:=v.arg.simplify()) != v.arg else None),
  # vars to views srcs
  (UPat(Ops.VIEW, name="x"), view_add_srcs),
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

  # insert contiguous in places determined by the realize map
  realize_map = group_realizes(tensor_map[sink])
  tensor_map = graph_rewrite_map(tensor_map[sink], add_contiguous, ctx=realize_map, bottom_up=True, input_map=tensor_map, name="add_contiguous")
  tensor_map = graph_rewrite_map(tensor_map[sink], finalize_contiguous+remove_tags, input_map=tensor_map, name="finalize_contiguous")

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
