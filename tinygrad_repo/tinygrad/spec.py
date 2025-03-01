from typing import cast
from tinygrad.ops import PatternMatcher, UPat, GroupOp, Ops, UOp, print_uops
from tinygrad.dtype import DType, ImageDType, dtypes, PtrDType
from tinygrad.helpers import all_same, all_int, dedup, prod

buffer_spec = PatternMatcher([
  (UPat(Ops.UNIQUE, dtypes.void, ()), lambda: True),
  (UPat(Ops.DEVICE, dtypes.void, (), name="device"), lambda device: isinstance(device.arg, str)),
  (UPat(Ops.BUFFER, src=(UPat(Ops.DEVICE), UPat(Ops.UNIQUE)), name="buf"),
   lambda buf: isinstance(buf.arg, int) and isinstance(buf.dtype, (DType, ImageDType))),
  (UPat(Ops.BUFFER_VIEW, src=(UPat(Ops.BUFFER),), name="buf_view"),
   lambda buf_view: isinstance(buf_view.arg, tuple) and len(buf_view.arg) == 2 and all_int(buf_view.arg)),
])

# *** this is the spec of a Tensor in UOp ***

tensor_uop_spec = buffer_spec+PatternMatcher([
  (UPat(GroupOp.Movement, name="mv", src=(UPat.var("x"),)),
   # naturally correct
   lambda mv,x: (isinstance(mv.arg, tuple) and mv.dtype == x.dtype) or
   # "make things that can't be images not images" can change the buffer dtype
   # this is fine as long as it's a realized buffer and base dtypes match.
   ((isinstance(mv.dtype, ImageDType) or isinstance(x.dtype, ImageDType)) and x.dtype.base == mv.dtype.base and x.base.op is Ops.BUFFER)),
  (UPat(Ops.VIEW, src=(UPat(GroupOp.All-{Ops.BUFFER, Ops.CONST, Ops.DEVICE}),)), lambda: False),

  # Tensor variable bindings
  (UPat(Ops.BIND, dtypes.int, (UPat(Ops.DEFINE_VAR), UPat.cvar(dtype=dtypes.int)), arg=None), lambda: True),

  # Tensor const has a device and an unmasked ShapeTracker of stride 0
  (UPat(Ops.CONST, src=(UPat(Ops.VIEW, name="st", src=(UPat(Ops.DEVICE),)),)),
   lambda st: st.st.views[0].mask is None and len(st.st.views) == 1 and all(s == 0 for s in st.st.views[0].strides)),

  # DETACH and CONTIGUOUS change how we interpret the source UOp
  # CONTIGUOUS ensures the source UOp realizes
  (UPat((Ops.DETACH, Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD), name="root", src=(UPat.var("x"),), arg=None), lambda root,x: root.dtype == x.dtype),

  # COPY
  # NOTE: the arg here specifies clone=True, which prevents folding same device copy
  (UPat(Ops.COPY, name="copy", src=(UPat(Ops.DEVICE), UPat.var("x"))), lambda copy,x: isinstance(copy.arg, bool) and copy.dtype == x.dtype),

  # ASSIGN changes the value of a buffer
  (UPat(Ops.ASSIGN, name="assign", src=(UPat.var("target"), UPat.var("new_val"))),
   lambda assign,target,new_val: target.base.op is Ops.BUFFER and (assign.dtype == target.dtype == new_val.dtype)),
])

# ***** uop type spec *****

# this is the matcher for the final rendered UOps
# matcher functions returns True or False (or None to not match)
spec = PatternMatcher([
  (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda x: isinstance(x.dtype, (PtrDType, ImageDType)) and not x.dtype.local),
  (UPat(Ops.DEFINE_LOCAL, name="x"), lambda x: isinstance(x.dtype, PtrDType) and x.dtype.local),
  (UPat(Ops.DEFINE_ACC, src=(UPat.var("c"),), name="x", allow_any_len=True),
   lambda x,c: all(y.op is Ops.RANGE for y in x.src[1:]) and c.dtype == x.dtype),
  (UPat(Ops.DEFINE_VAR, name="x"), lambda x: isinstance(x.arg[1], int) and isinstance(x.arg[2], int)),

  (UPat(Ops.RANGE, src=(UPat.var("x"), UPat.var("y")), name="rng"), lambda rng,x,y: rng.dtype == x.dtype == y.dtype and isinstance(rng.arg, int)),
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
  (UPat(Ops.INDEX, src=(UPat((Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL)), UPat())), lambda: True),

  # LOAD takes a <bufidx, alt?, gate?, barrier?>
  (UPat(Ops.LOAD, src=(UPat((Ops.INDEX, Ops.CAST)),)), lambda: True),
  (UPat(Ops.LOAD, src=(UPat((Ops.INDEX, Ops.CAST)), UPat((Ops.IF, Ops.BARRIER)))), lambda: True),
  (UPat(Ops.LOAD, src=(UPat((Ops.INDEX, Ops.CAST)), UPat.var("alt"), UPat(dtype=dtypes.bool)), name="ld"), lambda ld,alt: ld.dtype == alt.dtype),

  # STORE takes a <bufidx, val, gate?>
  (UPat(Ops.STORE, dtype=dtypes.void, src=(UPat((Ops.INDEX, Ops.CAST)), UPat())), lambda: True),
  (UPat(Ops.STORE, dtype=dtypes.void, src=(UPat((Ops.INDEX, Ops.CAST)), UPat(), UPat(dtype=dtypes.bool))), lambda: True),
  (UPat(Ops.STORE, dtype=dtypes.void, src=(UPat((Ops.INDEX, Ops.CAST)), UPat(), UPat(Ops.IF))), lambda: True),

  # most ALUs have all matching dtypes, except CMPLT, CMPNE, and WHERE
  (UPat(Ops.WHERE, name="w", src=(UPat(dtype=dtypes.bool), UPat.var("x"), UPat.var("y"))), lambda w,x,y: w.dtype == x.dtype == y.dtype),
  (UPat((Ops.CMPLT, Ops.CMPNE), dtype=dtypes.bool, src=(UPat.var("x"), UPat.var("y"))), lambda x,y: x.dtype.base == y.dtype.base),
  # and SHL/SHR, the shift distance can be an int
  (UPat((Ops.SHL, Ops.SHR), src=(UPat.var("x"), UPat.var("y")), name="a"), lambda a,x,y: a.dtype == x.dtype and y.dtype in (x.dtype, dtypes.uint)),
  (UPat(Ops.IDIV, name="x"), lambda x: None if dtypes.is_int(x.dtype) else False),
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

  (UPat(Ops.REDUCE_AXIS, name="x"), lambda x: isinstance(x.arg, tuple) and len(x.arg) == 2 and x.arg[0] in {Ops.ADD, Ops.MUL, Ops.MAX}),
  (UPat(Ops.GEP, src=(UPat.var("src"),), name="gep"), lambda gep,src: gep.dtype == src.dtype.scalar()),
  (UPat(Ops.VECTORIZE, name="x"), lambda x: len(x.src)>1 and len(x.src) == x.dtype.count and all(x.dtype == y.dtype.vec(len(x.src)) for y in x.src)),
  (UPat((Ops.BITCAST, Ops.CAST), src=(UPat(),), name="x"), lambda x: x.arg is None),
  (UPat(Ops.BARRIER, dtypes.void, src=UPat(Ops.STORE, allow_any_len=True)), lambda: True), # NOTE: all pointers must be local

  # NOTE: for testing, we let sinks be anything
  #(UPat(Ops.SINK, src=UPat(Ops.STORE)), lambda: True),
  (UPat((Ops.NAME, Ops.SINK), dtypes.void), lambda: True),
  (UPat((Ops.NOOP, Ops.CUSTOM)), lambda: True),

  # PTX LOAD/STORE
  (UPat((Ops.LOAD, Ops.STORE), src=(UPat(dtype=dtypes.int64),), allow_any_len=True), lambda: True),
])

# *** this is the spec of a Kernel in UOp ***

kernel_spec = buffer_spec+PatternMatcher([
  (UPat(Ops.KERNEL, src=UPat((Ops.BUFFER, Ops.ASSIGN))), lambda: True),
  # assign has a buffer view and kernel source, it can optionally depend on other assigns
  (UPat(Ops.ASSIGN, src=UPat((Ops.BUFFER, Ops.VIEW, Ops.KERNEL, Ops.ASSIGN))), lambda: True),
  (UPat(GroupOp.All-{Ops.SINK, Ops.VIEW}), lambda: False),
])

# *** this is the UOp shape spec ***

def verify_sink_dims(sink:UOp):
  shape_dims = [sorted(dedup(dims)) for dims in zip(*[x.shape for x in sink.toposort if x.op is not Ops.SINK and x.st is not None])]
  return all_same([x.st_arg.size for x in sink.src]) and all(len(x) == 1 or (len(x) == 2 and x[0] == 1) for x in shape_dims)

shape_spec = PatternMatcher([
  # shapes must have either 1 or n in each dimension
  (UPat(Ops.SINK, src=UPat(Ops.STORE), allow_any_len=True, name="sink"), verify_sink_dims),
  # all parent UOps must have the same shape
  (UPat(GroupOp.All-{Ops.SINK}, name="root"), lambda root: all_same([x.shape for x in root.src if x.st is not None])),
])

# ***** uop helpers *****

def type_verify(uops:list[UOp], *extra_specs:PatternMatcher):
  specs = [spec, *extra_specs]
  for i,u in enumerate(uops):
    spec_ret = [cast(bool|None, s.rewrite(u)) for s in specs]
    if any(ret is False for ret in spec_ret) or all(ret is None for ret in spec_ret):
      print_uops(uops)
      raise RuntimeError(f"UOp verification failed at {i} on {u.op} {u.dtype} {len(u.src)} {[x.op for x in u.src]} {u.arg}")
