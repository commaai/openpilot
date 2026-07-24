import math
from typing import Any
from tinygrad.uop.ops import PatternMatcher, UPat, GroupOp, Ops, UOp, AxisType, KernelInfo, ParamArg
from tinygrad.uop.render import print_uops, pyrender
from tinygrad.dtype import DType, dtypes, AddrSpace, Invalid, ConstFloat
from tinygrad.helpers import DEBUG, Context, SPEC, Metadata, panic, CHECK_OOB, all_same, is_image_shape

# ***** uop helpers *****

def validate_index(uidx:UOp, gate:UOp|None=None):
  if len(uidx.src) != 2: return True  # skip for non final index. TODO: check more complex index with shape
  buf,idx = uidx.src
  if idx.op is Ops.CONST and idx.arg is Invalid: return True
  if gate is None: gate = UOp.const(dtypes.bool, True)
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

def type_verify(ast:UOp|list[UOp], check_spec:PatternMatcher):
  lst = list(ast.toposort()) if isinstance(ast, UOp) else ast
  if SPEC > 1: test_pyrender(lst[-1])  # assume this is the sink

  with Context(TRACK_MATCH_STATS=0):
    for i,u in enumerate(lst):
      ret: bool|None = check_spec.rewrite(u)
      if ret is not True:
        if DEBUG >= 3: print_uops(lst)
        raise RuntimeError(f"UOp verification failed at {i} on {u.op} {u.dtype} {len(u.src)} {[(x.op, x.dtype, x.arg) for x in u.src]} {u.arg}")

# ***** new specs *****

# these ops can be used in the tensor graph and programs
spec_shared = PatternMatcher([
  # NOTE: for testing, we let sinks be anything
  (UPat(Ops.SINK, dtypes.void), lambda: True),

  # NOOP. TODO: remove this
  (UPat(Ops.NOOP), lambda: True),

  # CONST is everywhere
  (UPat(Ops.CONST, src=(), name="x"), lambda x: type(x.arg) is type(x.dtype.const(x.arg))),

  # STACK is everywhere too
  (UPat(Ops.STACK, dtype=dtypes.void, src=()), lambda: True),
  (UPat(Ops.STACK, src=(UPat(),), allow_any_len=True, name="s"),
   lambda s: all_same([x.shape for x in s.src]) and all(x.dtype == s.dtype for x in s.src)),

  # ALUs: operands match the result dtype, except comparisons/WHERE; renderer-lowered shifts may use a uint32 count
  # a weak dtype matches any dtype (TODO: make python scalars weak consts)
  (UPat(Ops.WHERE, name="w", src=(UPat(dtype=dtypes.bool), UPat(), UPat())),
   lambda w: all(s.dtype == w.dtype or s.dtype in dtypes.weaks for s in w.src[1:])),
  (UPat(GroupOp.Comparison, dtype=dtypes.bool, src=(UPat.var("x"), UPat.var("y"))),
   lambda x,y: x.dtype == y.dtype or x.dtype in dtypes.weaks or y.dtype in dtypes.weaks),
  (UPat((Ops.SHL, Ops.SHR), src=(UPat.var("x"), UPat(dtype=dtypes.uint)), name="a"), lambda a,x: a.dtype == x.dtype or None),
  (UPat((Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD), name="x"), lambda x: None if dtypes.is_int(x.dtype) else False),
  (UPat(GroupOp.ALU, name="x"), lambda x: all(y.dtype == x.dtype or y.dtype in dtypes.weaks for y in x.src)),

  # CAST
  (UPat((Ops.BITCAST, Ops.CAST), src=(UPat(),), name="x"), lambda x: isinstance(x.arg, DType)),

  # RANGE can be in the big graph now
  (UPat(Ops.RANGE, src=(UPat.var("x"),), allow_any_len=True, name="rng"), lambda rng,x:
    rng.dtype == x.dtype and isinstance(rng.arg, tuple) and len(rng.arg) >= 2 and \
      all(isinstance(ra, int) for ra in rng.arg[0:-1]) and isinstance(rng.arg[-1], AxisType)),
  (UPat(Ops.INDEX, src=(UPat(),), allow_any_len=True, name="x"), lambda x: all(dtypes.is_int(y.dtype) for y in x.src[1:]) or None),
  # LOOP is a bound-less loop header, the arg is an axis id like RANGE but without an AxisType
  (UPat(Ops.LOOP, dtypes.void, name="l"), lambda l: isinstance(l.arg, tuple) and all(isinstance(ra, int) for ra in l.arg)),
  # END closes RANGEs
  (UPat(Ops.END, src=(UPat(),), allow_any_len=True, name="x"), lambda x: all(u.op is Ops.RANGE for u in x.src[1:]) or None),
  # a LOOP-ended END requires a trailing bool condition for the backedge (loop again while true)
  (UPat(Ops.END, src=(UPat(), UPat(Ops.LOOP), UPat(dtype=dtypes.bool))), lambda: True),

  # PARAM
  (UPat(Ops.PARAM, name="x"), lambda x: isinstance(x.arg, ParamArg)),
  (UPat(Ops.BUFFER, src=(UPat(),), name="x"), lambda x:
   isinstance(x.arg, ParamArg) and x.addrspace in (AddrSpace.REG, AddrSpace.LOCAL)),

  # GROUP of stores (or groups, or NOOPs)
  (UPat(Ops.GROUP, dtypes.void, src=UPat((Ops.GROUP, Ops.STORE, Ops.NOOP, Ops.INS, Ops.END))), lambda: True),

  # AFTER on Movement Op, PARAM, BUFFER, CONTIGUOUS, or another AFTER
  (UPat(Ops.AFTER, src=(UPat(GroupOp.Movement.union({Ops.PARAM, Ops.BUFFER, Ops.CONTIGUOUS, Ops.INDEX,
                                                     Ops.AFTER, Ops.MULTI, Ops.BITCAST, Ops.INS})),),
        allow_any_len=True), lambda: True),

  # CUSTOM (inline and non inline)
  (UPat((Ops.CUSTOMI, Ops.CUSTOM)), lambda: True),

  # pattern compiler IR ops (not in tensor/program graphs, but spec-compliant)
  (UPat(Ops.PYLITERAL), lambda: True),

  # BARRIER (on any length). TODO: this should only be in spec_program
  (UPat(Ops.BARRIER, dtypes.void), lambda: True),

  # WAIT until a condition evaluates to true.
  (UPat(Ops.WAIT, dtypes.void, src=(UPat(dtype=dtypes.bool),)), lambda: True),

  # assembly instruction
  (UPat(Ops.INS), lambda: True),

  # LOAD(idx) / STORE(idx, val) with gates on the LOAD/STORE
  (UPat((Ops.INDEX, Ops.SHRINK), name="uidx").or_casted().load(), validate_index),
  (UPat((Ops.INDEX, Ops.SHRINK), name="uidx").or_casted().load(UPat.var("alt"), UPat.var("gate", dtype=dtypes.bool), name="load"),
   lambda uidx,gate,alt,load: validate_index(uidx, gate) if alt.dtype == load.dtype else False),
  (UPat((Ops.INDEX, Ops.SHRINK), name="uidx").or_casted().store(UPat()), validate_index),
  (UPat((Ops.INDEX, Ops.SHRINK), name="uidx").or_casted().store(UPat(), UPat.var("gate", dtype=dtypes.bool)), validate_index),

  # STORE in tensor graph: store a value into a target
  (UPat(Ops.STORE, dtypes.void, (UPat(name="x"), UPat())), lambda x: True),

  # WMMA has a <a, b, acc>
  (UPat(Ops.WMMA, src=(UPat(), UPat(), UPat()), name="x"), lambda x: isinstance(x.arg, tuple) and len(x.arg) == 5),
])

def is_device(d): return isinstance(d, str) or (isinstance(d, tuple) and all(isinstance(s, str) for s in d))

def valid_gettuple(g:UOp, t:UOp):
  return isinstance(g.arg, int) and 0 <= g.arg < len(t.src) and g.dtype == t.src[g.arg].dtype

# these ops can exist in tensor but not programs. example: movement
spec_tensor = PatternMatcher([
  (UPat((Ops.SIN, Ops.LOG2, Ops.EXP2, Ops.SQRT, Ops.RECIPROCAL), src=(UPat(),), name="u"), lambda u: dtypes.is_float(u.dtype)),

  # BUFFER
  (UPat(Ops.BUFFER, src=(UPat(),), name="buf"), lambda buf:
   (isinstance(buf.dtype, DType) and buf.src[0].dtype == dtypes.weakint and is_device(buf.arg.device))
   if isinstance(buf.arg, ParamArg) and buf.addrspace is AddrSpace.GLOBAL else None),

  # Tensor variable bindings
  (UPat(Ops.BIND, (dtypes.int, dtypes.weakint,), (UPat(Ops.PARAM), UPat.cvar(dtype=(dtypes.int,dtypes.weakint,))), arg=None), lambda: True),

  # custom function
  (UPat(Ops.CUSTOM_FUNCTION, name="x"), lambda x: isinstance(x.arg, str)),

  # CALL
  (UPat(Ops.CALL, dtypes.void, src=(UPat((Ops.SINK, Ops.LINEAR, Ops.PROGRAM, Ops.COPY, Ops.CUSTOM_FUNCTION)),), allow_any_len=True), lambda: True),

  # FUNCTION + TUPLE must have void dtype, GETTUPLE can only appear on FUNCTION or TUPLE
  (UPat(Ops.FUNCTION, dtypes.void, src=(UPat(Ops.TUPLE),), allow_any_len=True), lambda: True),
  (UPat(Ops.TUPLE, dtypes.void), lambda: True),
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.FUNCTION, src=(UPat(Ops.TUPLE, name="t"),), allow_any_len=True),), name="g"), valid_gettuple),
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.TUPLE, name="t"),), name="g"), valid_gettuple),

  # SPECIAL is index before index lowering. custom_kernel currently has this
  (UPat(Ops.SPECIAL, src=(UPat.var("x", dtypes.weakint),), name="s"), lambda s,x: s.dtype == x.dtype and isinstance(s.arg, str)),

  # inputs to movement ops
  (UPat({Ops.ADD, Ops.MUL, Ops.CDIV, Ops.FLOORDIV}, dtype=dtypes.weakint), lambda: True),

  # movement ops
  (UPat((Ops.RESHAPE, Ops.EXPAND), src=(UPat(), UPat())), lambda: True),
  (UPat((Ops.PAD, Ops.SHRINK), src=(UPat(), UPat(), UPat()), name="x"), lambda x: x.src[1].shape == x.src[2].shape),
  (UPat((Ops.PERMUTE, Ops.FLIP), name="mv", src=(UPat(),)), lambda mv: isinstance(mv.arg, tuple)),

  # REDUCE has arg=(op, num_axes), src[1:] are ranges after lowering
  (UPat(Ops.REDUCE, src=(UPat(),), allow_any_len=True, name="x"),
   lambda x: isinstance(x.arg, tuple) and len(x.arg) == 2 and x.arg[0] in GroupOp.Reduce
   and isinstance(x.arg[1], int) and all(y.dtype in (dtypes.weakint, dtypes.int) for y in x.src[1:])),

  # COPY. TODO: this should not have allow_any_len, but something is adding ranges
  (UPat(Ops.COPY, name="copy", src=(UPat.var("x"),), allow_any_len=True), lambda copy,x: copy.dtype == x.dtype and is_device(copy.arg)),
  (UPat(Ops.ALLREDUCE, name="red", src=(UPat.var("x"),)), lambda red,x: red.dtype == x.dtype and isinstance(red.arg, tuple) and
   len(red.arg) == 2 and red.arg[0] in GroupOp.Reduce and is_device(red.arg[1])),

  # MULTI/MSELECT/MSTACK
  (UPat(Ops.MULTI, name="multi"), lambda multi: all(x.dtype == multi.dtype for x in multi.src) and isinstance(multi.arg, int)),
  (UPat(Ops.MSELECT, name="x"), lambda x: isinstance(x.src[0].device, tuple) and x.arg < len(x.src[0].device)),
  (UPat(Ops.MSTACK, name="x"), lambda x: all(isinstance(s.device, str) for s in x.src) or (all_same(x.src) and x.src[0].device is None)),

  # CONTIGUOUS ensures the source UOp realizes
  (UPat((Ops.DETACH, Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD), name="root", src=(UPat.var("x"),), arg=None),
   lambda root,x: root.dtype == x.dtype),

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
  # index and weak dtypes are not allowed in programs
  (UPat(GroupOp.All, (dtypes.weakint, dtypes.weakfloat)), lambda: False),

  # allow special SHRINK
  (UPat(Ops.SHRINK, src=(UPat((Ops.PARAM, Ops.BUFFER, Ops.AFTER)), UPat(), UPat(Ops.CONST))), lambda: True),

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
  (UPat(Ops.SPECIAL, src=(UPat.var("x", dtypes.int32),), name="s"), lambda s,x: s.dtype == x.dtype and isinstance(s.arg, str)),
])+spec_shared

spec_hcq = PatternMatcher([
  (UPat(Ops.GETADDR, dtypes.uint64, src=(UPat((Ops.BUFFER, Ops.PARAM)).or_after(),), name="x"), lambda x: is_device(x.arg)),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat((Ops.BUFFER, Ops.PARAM)).or_after(),)), lambda: True),
])+spec_shared

# these are intermediate ops. everything should be deleted from here
spec_full = PatternMatcher([
  (UPat(Ops.REWRITE_ERROR, dtypes.void, name="x"), lambda x: isinstance(x.arg, str)),

  # SLICE on BUFFER is allowed if BUFFER is
  (UPat(Ops.SLICE, src=(UPat(GroupOp.Movement.union({Ops.BUFFER, Ops.PARAM, Ops.STAGE, Ops.AFTER})),
                        UPat(Ops.CONST, dtype=dtypes.weakint)), allow_any_len=True, name="bv"),
   lambda bv: isinstance(bv.arg, int)),

  (UPat(Ops.CALL, dtypes.void, src=(UPat((Ops.SLICE,)),), allow_any_len=True), lambda: True),

  # codegen may end ranges after gpudims has replaced RANGE with SPECIAL.
  (UPat(Ops.END, src=(UPat(), UPat()), allow_any_len=True), lambda: True),

  # allow any AFTER
  (UPat(Ops.AFTER, src=(UPat(),), allow_any_len=True), lambda: True),

  # all loads/stores
  (UPat((Ops.LOAD, Ops.STORE)), lambda: True),

  # while BIND is being casted
  (UPat(Ops.BIND, (dtypes.int, dtypes.weakint), (UPat(), UPat()), arg=None), lambda: True),
])+spec_tensor+spec_program+spec_hcq

# **** pyrender (move this) ****

# late imports to avoid circular import
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.schedule.rangeify import BufferizeOpts
glbls:dict[str, Any] = {"inf": math.inf, "nan": math.nan, "KernelInfo": KernelInfo, "Metadata": Metadata,
                        "UOp": UOp, "dtypes": dtypes, "Ops": Ops, "AxisType": AxisType, "Invalid": Invalid,
                        "Opt": Opt, "OptOps": OptOps, "BufferizeOpts": BufferizeOpts, "AddrSpace": AddrSpace, "panic": panic,
                        "ConstFloat": ConstFloat, "ParamArg": ParamArg}
def eval_pyrender(code:str) -> UOp:
  lcls:dict[str, Any] = {}
  exec(code, glbls, lcls)
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
