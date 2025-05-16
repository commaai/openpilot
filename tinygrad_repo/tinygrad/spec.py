from typing import cast, Callable
from tinygrad.ops import PatternMatcher, UPat, GroupOp, Ops, UOp, print_uops, python_alu, graph_rewrite, resolve
from tinygrad.dtype import DType, ImageDType, dtypes, PtrDType
from tinygrad.helpers import all_same, prod, DEBUG, IGNORE_OOB, Context
try:
  import z3

  # IDIV is truncated division but z3 does floored division; mod by power of two sometimes uses Ops.AND
  def z3_cdiv(a,b): return z3.If(a<0, (a+(b-1))/b, a/b)
  z3_alu: dict[Ops, Callable] = python_alu | {Ops.MOD: lambda a,b: a-z3_cdiv(a,b)*b, Ops.IDIV: z3_cdiv, Ops.SHR: lambda a,b: a/(2**b.as_long()),
    Ops.SHL: lambda a,b: a*(2**b.as_long()), Ops.AND: lambda a,b: a%(b+1) if isinstance(b, z3.ArithRef) else a&b}
  def create_bounded(name:str, vmin, vmax, solver:z3.Solver) -> z3.ArithRef:
    s = z3.Int(name, ctx=solver.ctx)
    solver.add(vmin <= s, s <= vmax)
    return s

  # ctx is (solver, load_number_dict)
  z3_renderer = PatternMatcher([
    # Ops.SPECIAL can have symbolic arg but it wont be in the toposort beacuse its not a src, we need to add it manually
    (UPat(Ops.SPECIAL, src=(), name="x"), lambda x: UOp(Ops.SPECIAL, arg=x.arg[0], src=(x.ufix(x.arg[1]),))),
    (UPat(Ops.SPECIAL, src=UPat(Ops.NOOP), name="x"), lambda x,ctx: UOp(Ops.NOOP, arg=create_bounded(x.arg, 0, x.src[0].arg-1, ctx[0]))),
    (UPat(Ops.DEFINE_VAR, name="x"), lambda x,ctx: UOp(Ops.NOOP, arg=create_bounded(x.arg[0], x.arg[1], x.arg[2], ctx[0]))),
    (UPat(Ops.RANGE, name="x"), lambda x,ctx: UOp(Ops.NOOP, arg=create_bounded(f"ridx{x.arg}", 0, x.src[0].arg-1, ctx[0]))),
    (UPat(Ops.LOAD, name="x"), lambda x,ctx: UOp(Ops.NOOP, arg=create_bounded(f"load{ctx[1].setdefault(x, len(ctx[1]))}", x.vmin, x.vmax, ctx[0]))),
    (UPat(Ops.CONST, name="x"), lambda x,ctx: UOp(Ops.NOOP, arg=(z3.BoolVal if dtypes.is_bool(x.dtype) else z3.IntVal)(x.arg, ctx=ctx[0].ctx))),
    (UPat(Ops.CAST, name="x"), lambda x: x.src[0]),
    (UPat(GroupOp.ALU, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=z3_alu[x.op](*(s.arg for s in x.src)))),
  ])

  z3_imported = True
except (ImportError, AttributeError): z3_imported = False

buffer_spec = PatternMatcher([
  (UPat(Ops.UNIQUE, dtypes.void, ()), lambda: True),
  (UPat(Ops.DEVICE, dtypes.void, (), name="device"), lambda device: isinstance(device.arg, str)),
  (UPat(Ops.BUFFER, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE)), allow_any_len=True, name="buf"),
   lambda buf: isinstance(buf.arg, int) and isinstance(buf.dtype, (DType, ImageDType))),
  (UPat(Ops.BUFFER_VIEW, src=(UPat(Ops.BUFFER),), name="buf_view"),
   lambda buf_view: isinstance(buf_view.arg, tuple) and len(buf_view.arg) == 2 and all(isinstance(arg, (int, UOp)) for arg in buf_view.arg)),
])

def validate_kernel(k:UOp):
  assert k.arg.ast.op in {Ops.COPY, Ops.BUFFER_VIEW, Ops.SINK}, f"must end with SINK/COPY/BUFFER_VIEW {k.arg}"
  if k.arg.ast.op is Ops.SINK: assert all(s.op is Ops.STORE for s in k.arg.ast.src), f"SINK must end with STORE {k.arg.ast}"
  return True

assign_spec = PatternMatcher([
  # KERNEL can attach to an ASSIGN to describe the compute required to realize a BUFFER
  (UPat(Ops.KERNEL, src=UPat((Ops.BUFFER, Ops.BUFFER_VIEW, Ops.ASSIGN)), name="k"), validate_kernel),

  # ASSIGN has a target and a value. It can also optionally depend on other assigns
  (UPat(Ops.ASSIGN, name="x"), lambda x: len(x.src) >= 2 and all(s.op is Ops.ASSIGN for s in x.src[2:])),
])

# *** this is the spec of a Tensor in UOp ***

tensor_uop_spec = buffer_spec+assign_spec+PatternMatcher([
  (UPat(GroupOp.Movement, name="mv", src=(UPat.var("x"),)),
   # naturally correct
   lambda mv,x: (isinstance(mv.arg, tuple) and mv.dtype == x.dtype) or
   # "make things that can't be images not images" can change the buffer dtype
   # this is fine as long as it's a realized buffer and base dtypes match.
   ((isinstance(mv.dtype, ImageDType) or isinstance(x.dtype, ImageDType)) and x.dtype.base == mv.dtype.base and x.base.op is Ops.BUFFER)),
  (UPat(Ops.VIEW, src=(UPat.var("x"),)), lambda x: x.base.op in {Ops.BUFFER, Ops.BUFFER_VIEW, Ops.ASSIGN, Ops.CONST, Ops.DEVICE}),

  # Tensor variable bindings
  (UPat(Ops.BIND, dtypes.int, (UPat(Ops.DEFINE_VAR), UPat.cvar(dtype=dtypes.int)), arg=None), lambda: True),

  # Tensor const has a device and an unmasked ShapeTracker of stride 0
  (UPat(Ops.CONST, src=(UPat(Ops.VIEW, name="st", src=(UPat(Ops.DEVICE),)),)),
   lambda st: st.st.views[0].mask is None and len(st.st.views) == 1 and all(s == 0 for s in st.st.views[0].strides)),

  # DETACH and CONTIGUOUS change how we interpret the source UOp
  # CONTIGUOUS ensures the source UOp realizes
  (UPat((Ops.DETACH, Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD, Ops.FUSE), name="root", src=(UPat.var("x"),), arg=None),
   lambda root,x: root.dtype == x.dtype),

  # COPY
  (UPat(Ops.COPY, name="copy", src=(UPat.var("x"), UPat(Ops.DEVICE))), lambda copy,x: copy.dtype == x.dtype),
])

# ***** uop type spec *****

def validate_index(idx:UOp, mask:UOp|None=None):
  if IGNORE_OOB or isinstance(idx.dtype, ImageDType) or (sz := cast(PtrDType, idx.src[0].dtype).size) == -1: return True
  # We can use UOp min/max to do a faster check, but it can give false positive since its not an exact bound and doesn't consider the mask
  if 0<=idx.src[1].vmin and idx.src[1].vmax<sz: return True

  # WEBGPU has a BITCAST in the index. TODO: fix
  if any(x.op is Ops.BITCAST for x in idx.toposort()): return True

  if not z3_imported: raise ImportError("z3 is required for bounds checking, try IGNORE_OOB=0 or \"pip install z3-solver\"")
  solver = z3.Solver(ctx=z3.Context())
  z3_sink = graph_rewrite(idx.src[1].sink(mask), z3_renderer, ctx=(solver, {}))
  z3_idx = z3_sink.src[0].arg
  if mask is not None: solver.add(z3_sink.src[1].arg)
  if solver.check((z3_idx<0)|(sz<=z3_idx)) == z3.sat:
    print(f"idx={idx.src[1].render(simplify=False)}")
    if mask is not None: print(f"mask={mask.render(simplify=False)}")
    print(f"# OUT OF BOUNDS ACCESS: at {solver.model()} INDEX not in 0 - {sz}\nconstraints = {solver}")
    return False
  return True

# this is the matcher for the final rendered UOps
# matcher functions returns True or False (or None to not match)
spec = PatternMatcher([
  (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda x: isinstance(x.dtype, (PtrDType, ImageDType)) and not x.dtype.local),
  (UPat(Ops.DEFINE_LOCAL, name="x"), lambda x: isinstance(x.dtype, PtrDType) and x.dtype.local),
  (UPat(Ops.DEFINE_ACC, src=(UPat.var("c"),), name="x", allow_any_len=True),
   lambda x,c: all(y.op is Ops.RANGE for y in x.src[1:]) and c.dtype == x.dtype),
  (UPat(Ops.DEFINE_VAR, name="x"), lambda x: isinstance(x.arg[1], int) and isinstance(x.arg[2], int)),

  (UPat(Ops.RANGE, src=(UPat.var("x"),), name="rng"), lambda rng,x: rng.dtype == x.dtype and isinstance(rng.arg, int)),
  (UPat(Ops.SPECIAL, src=()), lambda: True),

  # TODO: confirm the args of both of these are shapetrackers
  (UPat(Ops.VIEW, dtypes.void, src=()), lambda: True),
  (UPat(Ops.VIEW, src=(UPat.var("src"),), name="x"), lambda x,src: src.op is not Ops.STORE and x.dtype.base == src.dtype.base),

  (UPat(Ops.VALID, dtypes.bool, (UPat(Ops.VIEW),)), lambda: True),
  (UPat(Ops.CONST, name="x"), lambda x: type(x.arg) is type(dtypes.as_const(x.arg, x.dtype))),

  # early LOAD has a <buf, shapetracker, store?>
  (UPat(Ops.LOAD, src=(UPat((Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL)), UPat(Ops.VIEW))), lambda: True),
  (UPat(Ops.LOAD, src=(UPat((Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL)), UPat(Ops.VIEW), UPat(Ops.STORE))), lambda: True),

  # early STORE has a <buf, shapetracker, val>
  (UPat(Ops.STORE, src=(UPat((Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL)), UPat(Ops.VIEW), UPat())), lambda: True),

  # **** new style load/store ****

  # INDEX is used in new style load/store
  # INDEX takes a <buf, alu, gate?>
  (UPat(Ops.INDEX, src=(UPat((Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL)), UPat()), name="idx"), validate_index),
  (UPat(Ops.INDEX, src=(UPat((Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL)), UPat(), UPat(dtype=dtypes.bool, name="mask")), name="idx"), validate_index),

  # LOAD takes a <bufidx, alt?, barrier?>
  (UPat(Ops.LOAD, src=(UPat((Ops.INDEX, Ops.CAST)),)), lambda: True),
  (UPat(Ops.LOAD, src=(UPat((Ops.INDEX, Ops.CAST)), UPat((Ops.IF, Ops.BARRIER)))), lambda: True),
  (UPat(Ops.LOAD, src=(UPat((Ops.INDEX, Ops.CAST)), UPat.var("alt")), name="ld"), lambda ld,alt: ld.dtype == alt.dtype),

  # STORE takes a <bufidx, val, gate?>
  (UPat(Ops.STORE, dtype=dtypes.void, src=(UPat((Ops.INDEX, Ops.CAST)), UPat())), lambda: True),
  (UPat(Ops.STORE, dtype=dtypes.void, src=(UPat((Ops.INDEX, Ops.CAST)), UPat(), UPat(dtype=dtypes.bool))), lambda: True),
  (UPat(Ops.STORE, dtype=dtypes.void, src=(UPat((Ops.INDEX, Ops.CAST)), UPat(), UPat(Ops.IF))), lambda: True),

  # most ALUs have all matching dtypes, except CMPLT, CMPNE, and WHERE
  (UPat(Ops.WHERE, name="w", src=(UPat(dtype=dtypes.bool), UPat.var("x"), UPat.var("y"))), lambda w,x,y: w.dtype == x.dtype == y.dtype),
  (UPat((Ops.CMPLT, Ops.CMPNE), dtype=dtypes.bool, src=(UPat.var("x"), UPat.var("y"))), lambda x,y: x.dtype.base == y.dtype.base),
  # and SHL/SHR, the shift distance can be an int
  (UPat((Ops.SHL, Ops.SHR), src=(UPat.var("x"), UPat.var("y")), name="a"), lambda a,x,y: a.dtype == x.dtype and y.dtype in (x.dtype, dtypes.uint)),
  (UPat((Ops.IDIV, Ops.MOD), name="x"), lambda x: None if dtypes.is_int(x.dtype) else False),
  (UPat(GroupOp.ALU, name="x"), lambda x: all(x.dtype.base == y.dtype.base for y in x.src)),

  (UPat(Ops.ASSIGN, src=(UPat((Ops.DEFINE_ACC, Ops.DEFINE_GLOBAL)), UPat())), lambda: True),
  (UPat(Ops.ENDRANGE, dtype=dtypes.void, src=(UPat(Ops.RANGE),)), lambda: True),

  # WMMA has a <a, b, acc>
  (UPat(Ops.WMMA, src=(UPat(), UPat(), UPat()), name="x"), lambda x: isinstance(x.arg, tuple) and len(x.arg) == 8),
  (UPat(Ops.CONTRACT, name="x"), lambda x: x.dtype.count == prod(y[1] for y in x.arg)),
  (UPat(Ops.UNROLL, name="x"), lambda x: x.src[0].dtype.count == prod(y[1] for y in x.arg)),

  # if has a <gate, barrier?>
  (UPat(Ops.IF, dtype=dtypes.void, src=(UPat(),)), lambda: True),
  (UPat(Ops.IF, dtype=dtypes.void, src=(UPat(), UPat(Ops.BARRIER))), lambda: True),
  (UPat(Ops.ENDIF, dtype=dtypes.void, src=(UPat(Ops.IF),)), lambda: True),

  (UPat(Ops.REDUCE_AXIS, name="x"), lambda x: isinstance(x.arg, tuple) and len(x.arg) >= 2 and x.arg[0] in {Ops.ADD, Ops.MUL, Ops.MAX}),
  (UPat(Ops.GEP, src=(UPat.var("src"),), name="gep"), lambda gep,src: gep.dtype == src.dtype.scalar()),
  (UPat(Ops.VECTORIZE, name="x"), lambda x: len(x.src)>1 and len(x.src) == x.dtype.count and all(x.dtype == y.dtype.vec(len(x.src)) for y in x.src)),
  (UPat((Ops.BITCAST, Ops.CAST), src=(UPat(),), name="x"), lambda x: x.arg is None),
  (UPat(Ops.BARRIER, dtypes.void, src=UPat(Ops.STORE, allow_any_len=True)), lambda: True), # NOTE: all pointers must be local
  (UPat(Ops.BARRIER, dtypes.void), lambda: True), # BARRIERs can also happen at the end of loops

  # NOTE: for testing, we let sinks be anything
  #(UPat(Ops.SINK, src=UPat(Ops.STORE)), lambda: True),
  (UPat(Ops.SINK, dtypes.void), lambda: True),
  (UPat((Ops.NOOP, Ops.CUSTOMI, Ops.CUSTOM)), lambda: True),

  # PTX LOAD/STORE
  (UPat((Ops.LOAD, Ops.STORE), src=(UPat(dtype=dtypes.int64),), allow_any_len=True), lambda: True),
])

# *** schedule spec only allows buffers, assigns and kernels in the graph ***

sched_spec = buffer_spec+assign_spec+PatternMatcher([
  (UPat(GroupOp.All-{Ops.SINK}), lambda: False),
])

# *** this is the UOp shape spec ***

def verify_sink_dims(sink:UOp):
  if not all_same([s.shape for s in sink.src]): return False
  for dims in zip(*[x.shape for x in sink.toposort() if x.st is not None]):
    if len(n_dims:={s for s in dims if resolve(s!=1)}) > 1:
      print(f"# INVALID KERNEL DIMS: can only have 1 or n in each dimension: {n_dims}")
      return False

shape_spec = PatternMatcher([
  # shapes must have either 1 or n in each dimension
  (UPat(Ops.SINK, src=UPat(Ops.STORE), name="sink"), verify_sink_dims),
  # all parent UOps must have the same shape
  (UPat(GroupOp.All-{Ops.SINK}, name="root"), lambda root: all_same([x.shape for x in root.src if x.st is not None])),
])

# ***** uop helpers *****

def type_verify(uops:list[UOp], extra_spec:PatternMatcher|None=None):
  check_spec = (extra_spec+spec) if extra_spec is not None else spec
  for i,u in enumerate(uops):
    with Context(TRACK_MATCH_STATS=0): ret = check_spec.rewrite(u)
    if cast(bool|None, ret) is not True:
      if DEBUG >= 3: print_uops(uops)
      raise RuntimeError(f"UOp verification failed at {i} on {u.op} {u.dtype} {len(u.src)} {[x.op for x in u.src]} {u.arg}")
