from tinygrad.uop.ops import PatternMatcher, UPat, GroupOp, Ops, UOp, AxisType, ParamArg, CallInfo, OPAQUE_CALL_BODIES, \
  CustomFunction
from tinygrad.uop.render import render_uir
from tinygrad.dtype import DType, dtypes, AddrSpace, Invalid
from tinygrad.helpers import DEBUG, Context, CHECK_OOB, all_same, is_image_shape
from tinygrad.device import is_disk_device

# ***** uop helpers *****

def validate_index(uidx:UOp, gate:UOp|None=None):
  if len(uidx.src) != 2: return True  # skip for non final index. TODO: check more complex index with shape
  buf,idx = uidx.src
  if idx.is_invalid: return True
  if gate is None: gate = UOp.const(True)
  # TODO: check for overflow
  if not CHECK_OOB or is_image_shape(buf._shape): return True

  # buffer size
  sz = buf.max_numel()

  # We can use UOp min/max to do a faster check, but it can give false positive since its not an exact bound and doesn't consider the mask
  if 0<=idx.vmin and idx.vmax<sz: return True

  # if all is good and CHECK_OOB=1, validate with z3
  from tinygrad.uop.validate import validate_index_with_z3
  return validate_index_with_z3(sz, idx, gate)

def valid_device_range(device:str|tuple[str, ...]|None, src:tuple[UOp, ...]) -> bool:
  if not isinstance(device, tuple): return len(src) == 0
  if len(src) != 1: return False
  rng = src[0]
  return rng.op is Ops.RANGE and rng.axis_type is AxisType.DEVICE and int(rng.vmax)+1 == len(device)

def type_verify(ast:UOp|list[UOp], check_spec:PatternMatcher, enter_calls=True):
  lst = list(ast.toposort(enter_calls=enter_calls)) if isinstance(ast, UOp) else ast

  with Context(TRACK_MATCH_STATS=0):
    for i,u in enumerate(lst):
      ret: bool|None = check_spec.rewrite(u)
      if ret is not True:
        if DEBUG >= 3: print(render_uir(lst))
        raise RuntimeError(f"UOp verification failed at {i} on {u.op} {u.dtype} {len(u.src)} {[(x.op, x.dtype, x.arg) for x in u.src]} {u.arg}")

# ***** new specs *****
def matches_dtype(x:UOp, dtype:DType) -> bool: return x.dtype == dtype or x.base.is_invalid  # Invalid matches any dtype
p_size = UPat.any(UPat.cvar(dtype=dtypes.weakint), UPat(Ops.STACK, src=()))
# these ops can be used in the tensor graph and programs
spec_shared = PatternMatcher([
  # NOTE: for testing, we let sinks be anything
  (UPat(Ops.SINK, dtypes.void), lambda: True),

  # NOOP. TODO: remove this
  (UPat(Ops.NOOP), lambda: True),

  # CONST is everywhere; Invalid is a bool const
  (UPat(Ops.CONST, src=(), name="x"), lambda x: x.is_invalid or type(x.val) is type(x.dtype.const(x.val))),

  # STACK is everywhere too
  (UPat(Ops.STACK, dtype=dtypes.void, src=()), lambda: True),
  (UPat(Ops.STACK, src=(UPat(),), allow_any_len=True, name="s"),
   lambda s: all_same([x.shape for x in s.src]) and all(matches_dtype(x, s.dtype) or x.dtype in dtypes.weaks for x in s.src)),

  # ALUs: operands match the result dtype, except comparisons/WHERE; renderer-lowered shifts may use a uint32 count
  # a weak dtype matches any dtype until lowering commits its operand
  (UPat(Ops.WHERE, name="w", src=(UPat(dtype=dtypes.bool), UPat(), UPat())),
   lambda w: all(matches_dtype(s, w.dtype) or s.dtype in dtypes.weaks for s in w.src[1:])),
  (UPat(GroupOp.Comparison, dtype=dtypes.bool, src=(UPat.var("x"), UPat.var("y"))),
   lambda x,y: matches_dtype(x, y.dtype) or matches_dtype(y, x.dtype) or x.dtype in dtypes.weaks or y.dtype in dtypes.weaks),
  (UPat((Ops.AND, Ops.OR, Ops.XOR, Ops.SHL, Ops.SHR), name="x"), lambda x: False if any(dtypes.is_float(s.dtype) for s in x.src) else None),
  (UPat((Ops.SHL, Ops.SHR), src=(UPat.var("x"), UPat.var("c")), name="a"), lambda a,x,c:
   matches_dtype(c, a.dtype) or c.dtype in (dtypes.uint, dtypes.weakint) or x.base.is_invalid),
  (UPat((Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD), name="x"),
   lambda x: None if dtypes.is_int(x.dtype) or any(s.base.is_invalid for s in x.src) else False),
  (UPat(GroupOp.ALU, name="x"), lambda x: all(matches_dtype(y, x.dtype) or y.dtype in dtypes.weaks for y in x.src)),

  # CAST
  (UPat((Ops.BITCAST, Ops.CAST), src=(UPat(),), name="x"), lambda x: isinstance(x.arg, DType)),

  # RANGE can be in the big graph now. a void RANGE is a bound-less loop header, the arg is an axis id like RANGE
  # a RANGE has exactly one src, the bound. ordering deps wrap the bound in AFTER: RANGE(AFTER(CONST, other_range))
  (UPat(Ops.RANGE, src=(UPat(),), name="rng"), lambda rng: isinstance(rng.arg, tuple) and len(rng.arg) >= 2 and \
      isinstance(rng.arg[0], AxisType) and all(isinstance(ra, int) for ra in rng.arg[1:])),
  (UPat(Ops.INDEX, name="x"), lambda x: len(x.src)>0 and all(dtypes.is_int(y.dtype) or y.base.is_invalid for y in x.src[1:]) or None),
  # END closes bounded RANGEs around a void effect; it does not discard a value. Conditional loops use BACKEDGE.
  (UPat(Ops.END, src=(UPat(dtype=dtypes.void),), allow_any_len=True, name="x"),
   lambda x: x.arg is None and all(u.op is Ops.RANGE and dtypes.is_int(u.dtype) for u in x.src[1:])),
  # Execute body (discarding its value), then repeat the unbounded loop while the scalar condition is true.
  (UPat(Ops.BACKEDGE, dtypes.void, src=(UPat(), UPat(Ops.RANGE, dtypes.void), UPat(dtype=dtypes.bool)), name="x"),
   lambda x: x.arg is None and x.src[2].shape == () and not x.src[2].base.is_invalid),

  # PARAM/BUFFER carry a size CONST or an empty STACK for scalars
  (UPat(Ops.PARAM, src=(p_size,), name="x"), lambda x: isinstance(x.arg, ParamArg)),
  (UPat(Ops.BUFFER, src=(p_size,), name="x"), lambda x: isinstance(x.arg, ParamArg) and x.addrspace in (AddrSpace.REG, AddrSpace.LOCAL)),

  (UPat(Ops.BINARY, dtypes.uint8, src=(), name="x"), lambda x: isinstance(x.arg, bytes)),

  # AFTER on Movement Op, PARAM, BUFFER, ALLOC, STAGE, or another AFTER
  # CONST/CAST/NOOP are range bounds: RANGE(AFTER(CONST, other_range)) orders a loop after a sibling
  (UPat(Ops.AFTER, src=(UPat(GroupOp.Movement.union({Ops.PARAM, Ops.BUFFER, Ops.ALLOC, Ops.STAGE, Ops.INDEX,
                                                     Ops.AFTER, Ops.UNSHARD, Ops.BITCAST, Ops.INS,
                                                     Ops.CONST, Ops.CAST, Ops.NOOP, Ops.STACK})),),
        allow_any_len=True), lambda: True),
  # an AFTER can wrap a scalar ALU (e.g. a computed RANGE bound) to order it after effect ops
  (UPat(Ops.AFTER, src=(UPat(GroupOp.ALU),), allow_any_len=True, name="x"), lambda x: x.src[0].shape == ()),

  # CUSTOM (inline and non inline): the arg is the source string and the dtype it produces, void for a bare statement
  (UPat((Ops.CUSTOMI, Ops.CUSTOM), name="x"),
   lambda x: isinstance(x.arg, tuple) and len(x.arg) == 2 and isinstance(x.arg[0], str) and isinstance(x.arg[1], DType)),

  # a CUSTOM_FUNCTION names an external function: the arg is a CustomFunction stating the return dtype
  (UPat(Ops.CUSTOM_FUNCTION, name="x", allow_any_len=True), lambda x: isinstance(x.arg, CustomFunction)),
  # CALL: the body is always an opaque body stating the dtype, the arg is a CallInfo
  (UPat(Ops.CALL, src=(UPat(tuple(OPAQUE_CALL_BODIES)),), allow_any_len=True, name="x"), lambda x: isinstance(x.arg, CallInfo)),

  # pattern compiler IR ops (not in tensor/program graphs, but spec-compliant)
  (UPat(Ops.PYLITERAL), lambda: True),

  # BARRIER (on any length). TODO: this should only be in spec_program
  (UPat(Ops.BARRIER, dtypes.void), lambda: True),

  # assembly instruction
  (UPat(Ops.INS, name="x"), lambda x: isinstance(x.arg, tuple) and len(x.arg) == 2 and isinstance(x.arg[1], DType)),

  # LOAD(idx) / STORE(idx, val) with gates on the LOAD/STORE
  (UPat((Ops.INDEX, Ops.SHRINK), name="uidx").or_casted().load(), validate_index),
  (UPat((Ops.INDEX, Ops.SHRINK), name="uidx").or_casted().load(UPat.var("alt"), UPat.var("gate", dtype=dtypes.bool), name="load"),
   lambda uidx,gate,alt,load: validate_index(uidx, gate) if matches_dtype(alt, load.dtype) else False),
  (UPat((Ops.INDEX, Ops.SHRINK), name="uidx").or_casted().store(UPat()), validate_index),
  (UPat((Ops.INDEX, Ops.SHRINK), name="uidx").or_casted().store(UPat(), UPat.var("gate", dtype=dtypes.bool)), validate_index),

  # STORE: the target must be storage or a STAGE realization point (or an AFTER/BITCAST/view of one);
  # STAGE targets are written into the buffer the STAGE creates. INDEX stores are checked above
  (UPat(Ops.STORE, dtypes.void, (UPat(name="x"), UPat())), lambda x:
   True if (b:=x.storage_base).op in {Ops.BUFFER, Ops.ALLOC, Ops.PARAM, Ops.STAGE} else None if b.op is Ops.INDEX else False),

  # WMMA has a <a, b, acc>
  (UPat(Ops.WMMA, src=(UPat(), UPat(), UPat()), name="x"), lambda x: isinstance(x.arg, tuple) and len(x.arg) == 4),
])

def is_device(d): return isinstance(d, str) or (isinstance(d, tuple) and all(isinstance(s, str) for s in d))

# these ops can exist in tensor but not programs. example: movement
spec_tensor = PatternMatcher([
  (UPat((Ops.SIN, Ops.LOG2, Ops.EXP2, Ops.SQRT, Ops.RECIPROCAL), src=(UPat(),), name="u"),
   lambda u: dtypes.is_float(u.dtype) or u.src[0].base.is_invalid),

  # BUFFER has bound storage; ALLOC declares storage without a runtime buffer
  (UPat(Ops.BUFFER, src=(p_size,), allow_any_len=True, name="buf"), lambda buf:
   isinstance(buf.dtype, DType) and is_device(buf.arg.device) and buf.arg.buffer is not None
   and valid_device_range(buf.arg.device, buf.src[1:]) if isinstance(buf.arg, ParamArg) and buf.addrspace is AddrSpace.GLOBAL else None),
  (UPat(Ops.ALLOC, src=(p_size,), allow_any_len=True, name="buf"), lambda buf: isinstance(buf.arg, ParamArg)
   and buf.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL, AddrSpace.REG)
   and buf.arg.buffer is None
   and (buf.arg.device is None or (buf.addrspace is AddrSpace.GLOBAL and is_device(buf.arg.device)))
   and valid_device_range(buf.arg.device, buf.src[1:])),

  # a Variable is a scalar ALU PARAM with a value range and no device
  (UPat(Ops.PARAM, src=(UPat(Ops.STACK, src=()),), name="buf"), lambda buf: buf.arg.device is None if buf.is_variable else None),

  # SPECIAL is index before index lowering. custom_kernel currently has this
  (UPat(Ops.SPECIAL, src=(UPat(dtype=dtypes.weakint),), name="s"), lambda s: isinstance(s.arg, str)),

  # movement ops
  (UPat((Ops.RESHAPE, Ops.EXPAND), src=(UPat(), UPat())), lambda: True),
  (UPat((Ops.PAD, Ops.SHRINK), src=(UPat(), UPat(), UPat()), name="x"), lambda x: x.src[1].shape == x.src[2].shape),
  (UPat((Ops.PERMUTE, Ops.FLIP), name="mv", src=(UPat(),)), lambda mv: isinstance(mv.arg, tuple)),

  # REDUCE has arg=(op, num_axes), src[1:] are ranges after lowering
  (UPat(Ops.REDUCE, src=(UPat(),), allow_any_len=True, name="x"),
   lambda x: isinstance(x.arg, tuple) and len(x.arg) == 2 and x.arg[0] in GroupOp.Reduce
   and isinstance(x.arg[1], int) and all(y.dtype in (dtypes.weakint, dtypes.int) for y in x.src[1:])),

  # COPY carries the DEVICE range as src[1] when the target is multi-device
  (UPat(Ops.COPY, name="copy", src=(UPat(),), allow_any_len=True), lambda copy:
   is_device(copy.arg) and not is_disk_device(copy.arg) and valid_device_range(copy.arg, copy.src[1:])),
  (UPat(Ops.ALLREDUCE, name="red", src=(UPat(),)),
   lambda red: isinstance(red.arg, tuple) and len(red.arg) == 2 and red.arg[0] in GroupOp.Reduce and is_device(red.arg[1])),

  # UNSHARD/MSELECT/MSTACK
  # an UNSHARD carries the value and one sharding range per sharded axis (usually a DEVICE RANGE, but can be a derived expression)
  (UPat(Ops.UNSHARD, name="multi"), lambda multi: len(multi.src) == 1+len(multi.arg)
    and all(isinstance(a, int) for a in multi.arg) and all(r.dtype in dtypes.weaks for r in multi.src[1:])),
  (UPat(Ops.MSELECT, name="x"), lambda x: isinstance(x.src[0].device, tuple) and x.arg < len(x.src[0].device)),
  (UPat(Ops.MSTACK, name="x"), lambda x: all(isinstance(s.device, str) for s in x.src) or (all_same(x.src) and x.src[0].device is None)),

  # CONTIGUOUS ensures the source UOp realizes
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), src=(UPat(),), arg=None), lambda: True),

  # TODO: this should not be here. STAGE is transformed to BUFFER later
  (UPat(Ops.STAGE, src=(UPat(),), allow_any_len=True), lambda: True),

  # codegen: PROGRAM with progressive sources through the pipeline (SINK, LINEAR?, SOURCE?, BINARY?)
  (UPat(Ops.LINEAR, dtypes.void), lambda: True),
  (UPat(Ops.SOURCE, dtypes.void, src=()), lambda: True),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat(Ops.SINK),)), lambda: True),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat(Ops.SINK), UPat(Ops.LINEAR))), lambda: True),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat(Ops.SINK), UPat(Ops.LINEAR), UPat(Ops.SOURCE))), lambda: True),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat(Ops.SINK), UPat(Ops.LINEAR), UPat(Ops.SOURCE), UPat(Ops.BINARY))), lambda: True),
])+spec_shared

# these ops can exist in programs but not the tensor spec. example: LOAD
spec_program = PatternMatcher([
  # a bare CONST only appears under its width CAST or as a PARAM/BUFFER/ALLOC size
  (UPat(GroupOp.All-GroupOp.Defines, name="x"), lambda x: False if x.op is not Ops.CAST and any(s.op is Ops.CONST for s in x.src) else None),
  (UPat(GroupOp.All-{Ops.CONST}, dtypes.weaks), lambda: False),

  # allow special SHRINK of a buffer or its bitcast
  (UPat(Ops.SHRINK, src=(UPat((Ops.PARAM, Ops.BUFFER, Ops.ALLOC, Ops.AFTER)).or_bitcasted(), UPat(), UPat.cvar().or_casted())), lambda: True),

  # movement ops are not allowed in programs
  (UPat(GroupOp.Movement), lambda: False),

  # REG/LOCAL buffer
  (UPat((Ops.BUFFER, Ops.ALLOC), src=(p_size,), name="x"), lambda x: isinstance(x.arg, ParamArg) and x.addrspace in (AddrSpace.REG, AddrSpace.LOCAL)),

  # Invalid is not allowed in program
  (UPat(Ops.CONST, arg=Invalid), lambda: False),

  # if has a <gate, index_for_dedup>
  (UPat(Ops.IF, dtype=dtypes.void, src=(UPat(dtype=dtypes.bool), UPat((Ops.CAST, Ops.INDEX, Ops.SHRINK)))), lambda: True),
  (UPat(Ops.ENDIF, dtype=dtypes.void, src=(UPat(Ops.IF),)), lambda: True),

  # SPECIAL is int32 after index lowering
  (UPat(Ops.SPECIAL, src=(UPat(dtype=dtypes.int32),), name="s"), lambda s: isinstance(s.arg, str)),
])+spec_shared

spec_hcq = PatternMatcher([
  (UPat(Ops.GETADDR, dtypes.uint64, name="x",
        src=(UPat((Ops.BUFFER, Ops.ALLOC, Ops.PARAM, Ops.SHRINK, Ops.BITCAST, Ops.MSTACK, Ops.MSELECT, Ops.LINEAR)).or_after(),)),
   lambda x: is_device(x.arg)),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat((Ops.BUFFER, Ops.PARAM)).or_after(),)), lambda: True),
])+spec_shared

# these are intermediate ops. everything should be deleted from here
spec_full = PatternMatcher([
  (UPat(Ops.REWRITE_ERROR, dtypes.void, name="x"), lambda x: isinstance(x.arg, str)),

  # codegen may end ranges after gpudims has replaced RANGE with SPECIAL.
  (UPat(Ops.END, src=(UPat(dtype=dtypes.void), UPat()), allow_any_len=True, name="x"),
   lambda x: x.arg is None and all(dtypes.is_int(u.dtype) for u in x.src[1:])),

  # allow any AFTER
  (UPat(Ops.AFTER, src=(UPat(),), allow_any_len=True), lambda: True),

  # all loads/stores
  (UPat((Ops.LOAD, Ops.STORE)), lambda: True),
])+spec_tensor+spec_program+spec_hcq

# ***** kernel graph spec *****

spec_kernel_graph = PatternMatcher([
  # sink
  (UPat(Ops.SINK, dtypes.void), lambda: True),
  # const + stack to make vconsts and shape args. a 0-size/bound reduce keeps its const casted
  (UPat(Ops.CONST, src=()), lambda: True),
  (UPat(Ops.CAST, src=(UPat(Ops.CONST, src=()),)), lambda: True),
  (UPat(Ops.STACK, name="s"), lambda s: all(x.op in (Ops.CONST, Ops.PARAM) for x in s.src) or None),
  # linear for more kernels (TODO: we should enter non sink calls)
  #(UPat(Ops.LINEAR), lambda: True),
  # PARAM is caller-provided storage (or a Variable in the ALU addrspace), ALLOC is call-local storage
  (UPat(Ops.PARAM, src=(p_size,), name="x"), lambda x: isinstance(x.arg, ParamArg)),
  (UPat(Ops.BUFFER, src=(p_size,), allow_any_len=True, name="x"), lambda x:
   isinstance(x.arg, ParamArg) and valid_device_range(x.arg.device, x.src[1:]) and
   (x.arg.buffer is not None if x.addrspace is AddrSpace.GLOBAL else x.addrspace in (AddrSpace.LOCAL, AddrSpace.REG))),
  (UPat(Ops.ALLOC, src=(p_size,), allow_any_len=True, name="x"), lambda x:
   isinstance(x.arg, ParamArg) and x.addrspace is AddrSpace.GLOBAL and x.arg.buffer is None and valid_device_range(x.arg.device, x.src[1:])),
  (UPat(Ops.BITCAST), lambda: True),
  # mstack/mselect
  (UPat(Ops.MSTACK, name="x"), lambda x: all(isinstance(s.device, str) for s in x.src) or (all_same(x.src) and x.src[0].device is None)),
  (UPat(Ops.MSELECT, name="x"), lambda x: isinstance(x.src[0].device, tuple) and x.arg < len(x.src[0].device)),
  # open DEVICE ranges are bound per device at launch (e.g. the range on a multi-device BUFFER/ALLOC)
  (UPat(Ops.RANGE, name="r"), lambda r: r.axis_type is AxisType.DEVICE),
  # all calls are on opaque bodies
  (UPat(Ops.CALL, src=(UPat(tuple(OPAQUE_CALL_BODIES)),), allow_any_len=True), lambda: True),
  # after on PARAM or AFTER
  (UPat(Ops.AFTER, src=(UPat(GroupOp.Movement.union({Ops.PARAM, Ops.AFTER, Ops.BUFFER, Ops.ALLOC,
                                                  Ops.MSTACK, Ops.MSELECT, Ops.BITCAST, Ops.RESHAPE})),), allow_any_len=True), lambda: True),
])
