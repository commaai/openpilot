import math, functools
from typing import Any
from tinygrad.uop.ops import PatternMatcher, UPat, GroupOp, Ops, UOp, AxisType, KernelInfo, ParamArg
from tinygrad.uop.render import print_uops, pyrender
from tinygrad.dtype import DType, dtypes, AddrSpace, Invalid, ConstFloat
from tinygrad.helpers import DEBUG, Context, SPEC, Metadata, panic, CHECK_OOB, all_same, is_image_shape

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

  # TODO: validate these
  # WEBGPU has a BITCAST in the index, PTX casts pointer to long
  # VECTORIZE can't be properly modeled in z3 since it doesn't support vectors
  # don't descend into PARAM shape metadata; only the PARAM value participates in index arithmetic
  for x in idx.toposort(gate=lambda x: x.op is not Ops.PARAM) | gate.toposort(gate=lambda x: x.op is not Ops.PARAM):
    if x.op in {Ops.BITCAST, Ops.STACK}: return True

  # if all is good and CHECK_OOB=1, validate with z3
  from tinygrad.uop.validate import validate_index_with_z3
  return validate_index_with_z3(sz, idx, gate)

def type_verify(ast:UOp|list[UOp], check_spec:PatternMatcher, enter_calls=True):
  lst = list(ast.toposort(enter_calls=enter_calls)) if isinstance(ast, UOp) else ast
  if SPEC > 1: test_pyrender(lst[-1])  # assume this is the sink

  with Context(TRACK_MATCH_STATS=0):
    for i,u in enumerate(lst):
      ret: bool|None = check_spec.rewrite(u)
      if ret is not True:
        if DEBUG >= 3: print_uops(lst)
        raise RuntimeError(f"UOp verification failed at {i} on {u.op} {u.dtype} {len(u.src)} {[(x.op, x.dtype, x.arg) for x in u.src]} {u.arg}")

# ***** new specs *****
def matches_dtype(x:UOp, dtype:DType) -> bool: return x.dtype == dtype or x.base.is_invalid  # Invalid matches any dtype
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
  (UPat(Ops.RANGE, src=(UPat(),), allow_any_len=True, name="rng"), lambda rng: isinstance(rng.arg, tuple) and len(rng.arg) >= 2 and \
      all(isinstance(ra, int) for ra in rng.arg[0:-1]) and isinstance(rng.arg[-1], AxisType)),
  (UPat(Ops.INDEX, name="x"), lambda x: len(x.src)>0 and all(dtypes.is_int(y.dtype) or y.base.is_invalid for y in x.src[1:]) or None),
  # END closes RANGEs
  (UPat(Ops.END, src=(UPat(),), allow_any_len=True, name="x"), lambda x: all(u.op is Ops.RANGE for u in x.src[1:]) or None),
  # a loop-ended END requires a trailing bool condition for the backedge (loop again while true)
  (UPat(Ops.END, src=(UPat(), UPat(Ops.RANGE, dtypes.void), UPat(dtype=dtypes.bool))), lambda: True),

  # PARAM/BUFFER have a size in the arg, no shape input
  (UPat(Ops.PARAM, src=(), name="x"), lambda x: isinstance(x.arg, ParamArg)),
  (UPat(Ops.BUFFER, src=(), name="x"), lambda x: isinstance(x.arg, ParamArg) and x.addrspace in (AddrSpace.REG, AddrSpace.LOCAL)),

  # GROUP of stores (or groups, or NOOPs)
  (UPat(Ops.GROUP, dtypes.void, src=UPat((Ops.GROUP, Ops.STORE, Ops.NOOP, Ops.INS, Ops.END))), lambda: True),

  # AFTER on Movement Op, PARAM, BUFFER, CONTIGUOUS, RETURNED, or another AFTER
  (UPat(Ops.AFTER, src=(UPat(GroupOp.Movement.union({Ops.PARAM, Ops.BUFFER, Ops.CONTIGUOUS, Ops.INDEX,
                                                     Ops.AFTER, Ops.UNSHARD, Ops.BITCAST, Ops.INS, Ops.RETURNED})),),
        allow_any_len=True), lambda: True),

  # CUSTOM (inline and non inline): the arg is the source string and the dtype it produces, void for a bare statement
  (UPat((Ops.CUSTOMI, Ops.CUSTOM), name="x"),
   lambda x: isinstance(x.arg, tuple) and len(x.arg) == 2 and isinstance(x.arg[0], str) and isinstance(x.arg[1], DType)),

  # CALL of an external function
  (UPat(Ops.CALL, src=(UPat(),), allow_any_len=True, name="x"),
   lambda x: matches_dtype(x.src[0], dtypes.uint64) and isinstance(x.arg, DType) if x.src[0].dtype is not dtypes.void else None),

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

  # STORE: the target must be storage or a CONTIGUOUS realization point (or an AFTER/BITCAST/view of one);
  # CONTIGUOUS targets are written into the buffer the CONTIGUOUS creates. INDEX stores are checked above
  (UPat(Ops.STORE, dtypes.void, (UPat(name="x"), UPat())), lambda x:
   True if (b:=x.storage_base).op in {Ops.BUFFER, Ops.PARAM, Ops.RETURNED, Ops.CONTIGUOUS} else None if b.op is Ops.INDEX else False),

  # WMMA has a <a, b, acc>
  (UPat(Ops.WMMA, src=(UPat(), UPat(), UPat()), name="x"), lambda x: isinstance(x.arg, tuple) and len(x.arg) == 5),
])

def is_device(d): return isinstance(d, str) or (isinstance(d, tuple) and all(isinstance(s, str) for s in d))

# these ops can exist in tensor but not programs. example: movement
spec_tensor = PatternMatcher([
  (UPat((Ops.SIN, Ops.LOG2, Ops.EXP2, Ops.SQRT, Ops.RECIPROCAL), src=(UPat(),), name="u"),
   lambda u: dtypes.is_float(u.dtype) or u.src[0].base.is_invalid),

  # BUFFER
  (UPat(Ops.BUFFER, src=(), name="buf"), lambda buf:
   (isinstance(buf.dtype, DType) and isinstance(buf.arg.size, int) and is_device(buf.arg.device))
   if isinstance(buf.arg, ParamArg) and buf.addrspace is AddrSpace.GLOBAL else None),

  # a Variable is a 0-d ALU BUFFER with a value range and no device
  (UPat(Ops.BUFFER, src=(), name="buf"), lambda buf: buf.arg.device is None if buf.is_variable else None),

  # custom function
  (UPat(Ops.CUSTOM_FUNCTION, name="x"), lambda x: isinstance(x.arg, str)),

  # CALL
  (UPat(Ops.CALL, dtypes.void, src=(UPat((Ops.SINK, Ops.LINEAR, Ops.PROGRAM, Ops.COPY, Ops.CUSTOM_FUNCTION)),), allow_any_len=True), lambda: True),

  # RETURNED is a placeholder for a buffer a call writes and returns: it has a size in the arg, no shape input
  (UPat(Ops.RETURNED, src=(), name="x"), lambda x: isinstance(x.arg, ParamArg)),

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

  # COPY
  (UPat(Ops.COPY, name="copy", src=(UPat(),)), lambda copy: is_device(copy.arg)),
  (UPat(Ops.ALLREDUCE, name="red", src=(UPat(),)),
   lambda red: isinstance(red.arg, tuple) and len(red.arg) == 2 and red.arg[0] in GroupOp.Reduce and is_device(red.arg[1])),

  # UNSHARD/MSELECT/MSTACK
  # an UNSHARD carries the value and one sharding range per sharded axis (usually a DEVICE RANGE, but can be a derived expression)
  (UPat(Ops.UNSHARD, name="multi"), lambda multi: len(multi.src) == 1+len(multi.arg)
    and all(isinstance(a, int) for a in multi.arg) and all(r.dtype in dtypes.weaks for r in multi.src[1:])),
  (UPat(Ops.MSELECT, name="x"), lambda x: isinstance(x.src[0].device, tuple) and x.arg < len(x.src[0].device)),
  (UPat(Ops.MSTACK, name="x"), lambda x: all(isinstance(s.device, str) for s in x.src) or (all_same(x.src) and x.src[0].device is None)),

  # CONTIGUOUS ensures the source UOp realizes
  (UPat((Ops.DETACH, Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD), src=(UPat(),), arg=None), lambda: True),

  # TODO: this should not be here. STAGE is transformed to BUFFER later
  (UPat(Ops.STAGE, src=(UPat(),), allow_any_len=True), lambda: True),

  # codegen: PROGRAM with progressive sources through the pipeline (SINK, LINEAR?, SOURCE?, BINARY?)
  (UPat(Ops.LINEAR, dtypes.void), lambda: True),
  (UPat(Ops.SOURCE, dtypes.void, src=()), lambda: True),
  (UPat(Ops.BINARY, dtypes.uint8, src=(), name="x"), lambda x: isinstance(x.arg, bytes)),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat(Ops.SINK),)), lambda: True),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat(Ops.SINK), UPat(Ops.LINEAR))), lambda: True),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat(Ops.SINK), UPat(Ops.LINEAR), UPat(Ops.SOURCE))), lambda: True),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat(Ops.SINK), UPat(Ops.LINEAR), UPat(Ops.SOURCE), UPat(Ops.BINARY))), lambda: True),
])+spec_shared

# these ops can exist in programs but not the tensor spec. example: LOAD
spec_program = PatternMatcher([
  # every width in a program is stated: a CONST appears only under the CAST stating its width, and is the only weak node
  (UPat(GroupOp.All, name="x"), lambda x: False if x.op is not Ops.CAST and any(s.op is Ops.CONST for s in x.src) else None),
  (UPat(GroupOp.All-{Ops.CONST}, dtypes.weaks), lambda: False),

  # allow special SHRINK
  (UPat(Ops.SHRINK, src=(UPat((Ops.PARAM, Ops.BUFFER, Ops.AFTER)), UPat(), UPat(Ops.CONST).or_casted())), lambda: True),

  # movement ops are not allowed in programs
  (UPat(GroupOp.Movement), lambda: False),

  # REG/LOCAL buffer
  (UPat(Ops.BUFFER, name="x"), lambda x: isinstance(x.arg, ParamArg) and x.addrspace in (AddrSpace.REG, AddrSpace.LOCAL)),

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
        src=(UPat((Ops.BUFFER, Ops.PARAM, Ops.SHRINK, Ops.BITCAST, Ops.MSTACK, Ops.MSELECT, Ops.LINEAR)).or_after(),)),
   lambda x: is_device(x.arg)),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat((Ops.BUFFER, Ops.PARAM)).or_after(),)), lambda: True),
])+spec_shared

# these are intermediate ops. everything should be deleted from here
spec_full = PatternMatcher([
  (UPat(Ops.REWRITE_ERROR, dtypes.void, name="x"), lambda x: isinstance(x.arg, str)),

  # codegen may end ranges after gpudims has replaced RANGE with SPECIAL.
  (UPat(Ops.END, src=(UPat(), UPat()), allow_any_len=True), lambda: True),

  # allow any AFTER
  (UPat(Ops.AFTER, src=(UPat(),), allow_any_len=True), lambda: True),

  # all loads/stores
  (UPat((Ops.LOAD, Ops.STORE)), lambda: True),
])+spec_tensor+spec_program+spec_hcq

# ***** kernel graph spec *****

spec_kernel_graph = PatternMatcher([
  # sink
  (UPat(Ops.SINK, dtypes.void), lambda: True),
  # the store of a bound Variable binds it: AFTER(BUFFER, STORE(BUFFER, CONST)) in call args
  (UPat(Ops.STORE, dtypes.void, (UPat(Ops.BUFFER, name="b"), UPat(Ops.CONST))), lambda b: b.is_variable),
  # const + stack to make vconsts and shape args. a 0-size/bound reduce keeps its const casted
  (UPat(Ops.CONST, src=()), lambda: True),
  (UPat(Ops.CAST, src=(UPat(Ops.CONST, src=()),)), lambda: True),
  (UPat(Ops.STACK, name="s"), lambda s: all(x.op in (Ops.CONST, Ops.PARAM) or x.is_variable or x.is_bound_var for x in s.src) or None),
  # linear for more kernels (TODO: we should enter non sink calls)
  #(UPat(Ops.LINEAR), lambda: True),
  # param is outside buffer, buffer is local buffer. params have a size in the arg, no shape input
  (UPat(Ops.PARAM, src=(), name="x"), lambda x: isinstance(x.arg, ParamArg)),
  (UPat(Ops.BUFFER, name="x"), lambda x: isinstance(x.arg, ParamArg) and x.addrspace in (AddrSpace.GLOBAL, AddrSpace.ALU)),
  (UPat(Ops.BITCAST), lambda: True),
  # mstack/mselect
  (UPat(Ops.MSTACK, name="x"), lambda x: all(isinstance(s.device, str) for s in x.src) or (all_same(x.src) and x.src[0].device is None)),
  (UPat(Ops.MSELECT, name="x"), lambda x: isinstance(x.src[0].device, tuple) and x.arg < len(x.src[0].device)),
  # all calls are on various sinks
  (UPat(Ops.CALL, src=(UPat((Ops.SINK, Ops.LINEAR, Ops.PROGRAM, Ops.CUSTOM_FUNCTION)),), allow_any_len=True), lambda: True),
  # after on PARAM or AFTER
  (UPat(Ops.AFTER, src=(UPat(GroupOp.Movement.union({Ops.PARAM, Ops.AFTER, Ops.BUFFER, Ops.MSTACK, Ops.MSELECT, Ops.BITCAST, Ops.RESHAPE})),),
        allow_any_len=True), lambda: True),
])

# **** pyrender (move this) ****

# circular-import-safe eval globals for pyrender round-tripping (lazy: codegen/schedule/renderer are heavy)
@functools.cache
def pyrender_globals() -> dict[str, Any]:
  from tinygrad.codegen.opt import Opt, OptOps
  from tinygrad.schedule.rangeify import BufferizeOpts
  from tinygrad.renderer import Estimates
  return {"inf": math.inf, "nan": math.nan, "KernelInfo": KernelInfo, "Metadata": Metadata,
          "UOp": UOp, "dtypes": dtypes, "Ops": Ops, "AxisType": AxisType, "Invalid": Invalid,
          "Opt": Opt, "OptOps": OptOps, "BufferizeOpts": BufferizeOpts, "AddrSpace": AddrSpace, "panic": panic,
          "ConstFloat": ConstFloat, "ParamArg": ParamArg, "Estimates": Estimates}
def eval_pyrender(code:str) -> UOp:
  lcls:dict[str, Any] = {}
  exec(code, pyrender_globals(), lcls)
  return lcls['ast']

def test_pyrender(test_ast:UOp, assert_parents=True):
  try: code = pyrender(test_ast)
  except NotImplementedError: return None  # this is okay, not all ops can be pyrendered
  ast:UOp = eval_pyrender(code)
  if ast is not test_ast:
    if assert_parents:
      for u in test_ast.toposort(): test_pyrender(u, assert_parents=False)
    raise RuntimeError(f"PYRENDER ISSUE:\nSTR MATCH: {str(test_ast) == str(ast)}\nUOP:\n{test_ast}\nPRODUCED:\n{ast}\nCODE:\n{code}")
  return code
