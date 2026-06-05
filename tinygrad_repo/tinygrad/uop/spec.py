import math
from typing import cast, Any
from tinygrad.uop.ops import PatternMatcher, UPat, GroupOp, Ops, UOp, AxisType, KernelInfo
from tinygrad.uop.render import print_uops, pyrender
from tinygrad.dtype import DType, ImageDType, dtypes, PtrDType, AddrSpace, Invalid, ConstFloat
from tinygrad.helpers import DEBUG, Context, prod, SPEC, Metadata, panic, CHECK_OOB

# ***** uop helpers *****

def validate_index(uidx:UOp, gate:UOp|None=None):
  if len(uidx.src) != 2: return True  # skip for non final index. TODO: check more complex index with shape
  buf,idx = uidx.src
  if idx.op is Ops.CONST and idx.arg is Invalid: return True
  if gate is None: gate = UOp.const(dtypes.bool, True)
  # TODO: check for overflow
  if not CHECK_OOB or isinstance(buf.dtype, ImageDType) or (sz := buf.ptrdtype.size) == -1: return True

  # We can use UOp min/max to do a faster check, but it can give false positive since its not an exact bound and doesn't consider the mask
  if 0<=idx.vmin and idx.vmax<sz: return True

  # TODO: validate these
  # WEBGPU has a BITCAST in the index, PTX casts pointer to long
  # VECTORIZE/GEP can't be properly modeled in z3 since it doesn't support vectors
  for x in idx.toposort() | gate.toposort():
    if x.op in {Ops.BITCAST, Ops.STACK, Ops.GEP} or (x.op is Ops.CAST and isinstance(x.src[0].dtype, PtrDType)): return True

  # if all is good and CHECK_OOB=1, validate with z3
  from tinygrad.uop.validate import validate_index_with_z3
  return validate_index_with_z3(sz, idx, gate)

def type_verify(ast:UOp|list[UOp], check_spec:PatternMatcher):
  lst = list(ast.toposort()) if isinstance(ast, UOp) else ast
  if SPEC > 1: test_pyrender(lst[-1])  # assume this is the sink

  with Context(TRACK_MATCH_STATS=0):
    for i,u in enumerate(lst):
      ret = check_spec.rewrite(u)
      if cast(bool|None, ret) is not True:
        if DEBUG >= 3: print_uops(lst)
        raise RuntimeError(f"UOp verification failed at {i} on {u.op} {u.dtype} {len(u.src)} {[(x.op, x.dtype, x.arg) for x in u.src]} {u.arg}")

# ***** new specs *****

# these ops can be used in the tensor graph and programs
spec_shared = PatternMatcher([
  (UPat(Ops.SINK, dtypes.void), lambda: True), # NOTE: for testing, we let sinks be anything

  # NOOP. TODO: remove this
  (UPat(Ops.NOOP), lambda: True),

  # CONST/DEFINE_VAR are everywhere
  (UPat(Ops.CONST, src=(), name="x"), lambda x: type(x.arg) is type(x.dtype.const(x.arg))),
  (UPat(Ops.DEFINE_VAR, name="x"), lambda x: len(x.arg) == 3 and isinstance(x.arg[0], str)),

  # ALUs: most ALUs have all matching dtypes, except CMPLT, CMPNE, and WHERE
  (UPat(Ops.WHERE, name="w", src=(UPat(dtype=dtypes.bool), UPat.var("x"), UPat.var("y"))), lambda w,x,y: w.dtype == x.dtype == y.dtype),
  (UPat((Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ), dtype=dtypes.bool, src=(UPat.var("x"), UPat.var("y"))), lambda x,y: x.dtype.base == y.dtype.base),
  # and SHL/SHR, the shift distance can be an int
  (UPat((Ops.SHL, Ops.SHR), src=(UPat.var("x"), UPat.var("y")), name="a"), lambda a,x,y: a.dtype == x.dtype and y.dtype in (x.dtype, dtypes.uint)),
  (UPat((Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD), name="x"), lambda x: None if dtypes.is_int(x.dtype) else False),
  (UPat(GroupOp.ALU, name="x"), lambda x: all(x.dtype.base == y.dtype.base for y in x.src)),

  # CAST
  (UPat((Ops.BITCAST, Ops.CAST), src=(UPat(),), name="x"), lambda x: x.arg is None),

  # RANGE can be in the big graph now
  (UPat(Ops.RANGE, src=(UPat.var("x"),), allow_any_len=True, name="rng"), lambda rng,x:
    rng.dtype == x.dtype and isinstance(rng.arg, tuple) and len(rng.arg) >= 2 and \
      all(isinstance(ra, int) for ra in rng.arg[0:-1]) and isinstance(rng.arg[-1], AxisType)),
  (UPat(Ops.INDEX, src=(UPat(),), allow_any_len=True, name="x"), lambda x: all(dtypes.is_int(y.dtype) for y in x.src[1:]) or None),
  (UPat(Ops.END, src=(UPat(),), allow_any_len=True, name="x"), lambda x: all(u.op is Ops.RANGE for u in x.src[1:])),

  # PARAM (that's really a DEFINE_GLOBAL)
  (UPat(Ops.PARAM, name="x"), lambda x: isinstance(x.dtype, (PtrDType, ImageDType)) and x.dtype.addrspace == AddrSpace.GLOBAL),

  # GROUP of stores (or groups, or NOOPs)
  # TODO: remove UNROLL here, it's for SPEC=2
  (UPat(Ops.GROUP, dtypes.void, src=UPat((Ops.GROUP, Ops.STORE, Ops.NOOP, Ops.UNROLL, Ops.INS))), lambda: True),

  # TOOD: these should be buffer with different addrspace
  (UPat(Ops.DEFINE_LOCAL, name="x"), lambda x: isinstance(x.dtype, PtrDType) and x.dtype.addrspace == AddrSpace.LOCAL),
  (UPat(Ops.DEFINE_REG, src=()), lambda: True),

  # AFTER on Movement Op, PARAM, BUFFER, CONTIGUOUS, or another AFTER
  (UPat(Ops.AFTER, src=(UPat(GroupOp.Movement.union({Ops.PARAM, Ops.BUFFER, Ops.CONTIGUOUS, Ops.DEFINE_REG, Ops.DEFINE_LOCAL, Ops.AFTER, Ops.MULTI,
                                                     Ops.BITCAST, Ops.INS})),),
        allow_any_len=True), lambda: True),

  # CUSTOM (inline and non inline)
  (UPat((Ops.CUSTOMI, Ops.CUSTOM)), lambda: True),

  # BARRIER (on any length). TODO: this should only be in spec_program
  (UPat(Ops.BARRIER, dtypes.void), lambda: True),

  # SPECIAL. TODO: this should only be in spec_program
  (UPat(Ops.SPECIAL, src=(UPat.var("x", (dtypes.weakint, dtypes.int32)),), name="s"), lambda s,x: s.dtype == x.dtype and isinstance(s.arg, str)),

  # assembly instruction
  (UPat(Ops.INS), lambda: True),

  # LOAD(idx) / STORE(idx, val) with gates on the LOAD/STORE
  (UPat(Ops.INDEX, name="uidx").or_casted().load(), validate_index),
  (UPat(Ops.INDEX, name="uidx").or_casted().load(UPat.var("alt"), UPat.var("gate", dtype=dtypes.bool), name="load"),
   lambda uidx,gate,alt,load: validate_index(uidx, gate) if alt.dtype == load.dtype else False),
  (UPat(Ops.INDEX, name="uidx").or_casted().store(UPat()), validate_index),
  (UPat(Ops.INDEX, name="uidx").or_casted().store(UPat(), UPat.var("gate", dtype=dtypes.bool)), validate_index),

  # STORE in tensor graph: store a value into a target
  (UPat(Ops.STORE, dtypes.void, (UPat(name="x"), UPat())), lambda x: True),

  # WMMA has a <a, b, acc>
  (UPat(Ops.WMMA, src=(UPat(), UPat(), UPat()), name="x"), lambda x: isinstance(x.arg, tuple) and len(x.arg) == 8),
])

# these ops can exist in tensor but not programs. example: movement
spec_tensor = PatternMatcher([
  # DEVICE
  (UPat(Ops.DEVICE, dtypes.void, (), name="d"), lambda d:
   isinstance(d.arg, str) or (isinstance(d.arg, tuple) and all(isinstance(s, str) for s in d.arg))),

  # UNIQUE
  (UPat(Ops.UNIQUE, dtypes.void, ()), lambda: True),
  (UPat(Ops.LUNIQUE, dtypes.void, ()), lambda: True),

  # CONST with a UNIQUE or DEVICE
  (UPat(Ops.CONST, src=(UPat(Ops.DEVICE),)), lambda: True),
  (UPat(Ops.CONST, src=(UPat((Ops.UNIQUE, Ops.LUNIQUE)), UPat(Ops.DEVICE))), lambda: True),

  # BUFFER
  (UPat(Ops.BUFFER, src=(UPat((Ops.UNIQUE, Ops.LUNIQUE)), UPat(Ops.DEVICE)), name="buf"),
   lambda buf: isinstance(buf.arg, int) and isinstance(buf.dtype, DType)),

  # PARAM (that's really a variable)
  (UPat(Ops.PARAM, src=(UPat(), UPat(), UPat(), UPat(), UPat()), name="x"), lambda x: True),

  # Tensor variable bindings
  (UPat(Ops.BIND, (dtypes.int, dtypes.weakint,), (UPat(Ops.DEFINE_VAR), UPat.cvar(dtype=(dtypes.int,dtypes.weakint,))), arg=None), lambda: True),

  # custom function
  (UPat(Ops.CUSTOM_FUNCTION, name="x"), lambda x: isinstance(x.arg, str)),

  # CALL
  (UPat(Ops.CALL, src=(UPat((Ops.SINK, Ops.LINEAR, Ops.PROGRAM, Ops.COPY, Ops.CUSTOM_FUNCTION)),), allow_any_len=True), lambda: True),

  # FUNCTION + TUPLE must have void dtype, GETTUPLE can only appear on FUNCTION or TUPLE
  (UPat(Ops.FUNCTION, dtypes.void, src=(UPat(Ops.TUPLE),), allow_any_len=True), lambda: True),
  (UPat(Ops.TUPLE, dtypes.void), lambda: True),
  (UPat(Ops.GETTUPLE, src=(UPat((Ops.FUNCTION, Ops.TUPLE)),), name="g"), lambda g: isinstance(g.arg, int)),

  # PARAM
  (UPat(Ops.PARAM, src=(UPat(), UPat(Ops.NOOP)), name="x"), lambda x: True),  # TODO: why does this have NOOP?
  (UPat(Ops.PARAM, src=(UPat(), UPat(Ops.DEVICE)), name="x"), lambda x: True),
  (UPat(Ops.PARAM, src=(UPat(), UPat(Ops.DEVICE), UPat(Ops.MULTI)), name="x"), lambda x: True),

  # inputs to movement ops
  (UPat(Ops.STACK), lambda: True),
  (UPat({Ops.ADD, Ops.MUL, Ops.CDIV, Ops.FLOORDIV}, dtype=dtypes.weakint), lambda: True),

  # movement ops
  (UPat((Ops.RESHAPE, Ops.EXPAND), src=(UPat(), UPat(dtype=dtypes.weakint))), lambda: True),
  (UPat((Ops.PAD, Ops.SHRINK), src=(UPat(), UPat(dtype=dtypes.weakint), UPat(dtype=dtypes.weakint))), lambda: True),
  (UPat((Ops.PERMUTE, Ops.FLIP), name="mv", src=(UPat(),)), lambda mv: isinstance(mv.arg, tuple)),

  # REDUCE has arg=(op, axis_tuple), src[1:] are ranges after lowering
  (UPat(Ops.REDUCE, src=(UPat(),), allow_any_len=True, name="x"),
   lambda x: isinstance(x.arg, tuple) and len(x.arg) == 2 and x.arg[0] in {Ops.ADD, Ops.MUL, Ops.MAX}
   and isinstance(x.arg[1], tuple) and all(y.dtype in (dtypes.weakint, dtypes.int) for y in x.src[1:])),

  # COPY. TODO: this should not have allow_any_len, but something is adding ranges
  (UPat(Ops.COPY, name="copy", src=(UPat.var("x"), UPat(Ops.DEVICE)), allow_any_len=True, arg=None), lambda copy,x: copy.dtype == x.dtype),
  (UPat(Ops.ALLREDUCE, name="red", src=(UPat.var("x"), UPat(Ops.DEVICE))), lambda red,x: red.dtype == x.dtype and isinstance(red.arg, Ops)),

  # MULTI/MSELECT/MSTACK
  (UPat(Ops.MULTI, name="multi"), lambda multi: all(x.dtype == multi.dtype for x in multi.src) and isinstance(multi.arg, int)),
  (UPat(Ops.MSELECT, name="x"), lambda x: isinstance(x.src[0].device, tuple) and x.arg < len(x.src[0].device)),
  (UPat(Ops.MSTACK, name="x"), lambda x: all(isinstance(x.device, str) for x in x.src)),

  # CONTIGUOUS ensures the source UOp realizes
  (UPat((Ops.DETACH, Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD), name="root", src=(UPat.var("x"),), arg=None),
   lambda root,x: root.dtype == x.dtype),

  # TODO: this should not be here. STAGE is transformed to DEFINE_LOCAL later
  (UPat(Ops.STAGE, src=(UPat(),), allow_any_len=True), lambda: True),

  # codegen: PROGRAM with progressive sources through the pipeline (SINK, DEVICE, LINEAR?, SOURCE?, BINARY?)
  (UPat(Ops.LINEAR, dtypes.void), lambda: True),
  (UPat(Ops.SOURCE, dtypes.void, src=()), lambda: True),
  (UPat(Ops.BINARY, dtypes.void, src=()), lambda: True),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat(Ops.SINK), UPat(Ops.DEVICE))), lambda: True),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat(Ops.SINK), UPat(Ops.DEVICE), UPat(Ops.LINEAR))), lambda: True),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat(Ops.SINK), UPat(Ops.DEVICE), UPat(Ops.LINEAR), UPat(Ops.SOURCE))), lambda: True),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat(Ops.SINK), UPat(Ops.DEVICE), UPat(Ops.LINEAR), UPat(Ops.SOURCE), UPat(Ops.BINARY))), lambda: True),

  # UNROLL/CONTRACT is used here for WMMA
  (UPat(Ops.CONTRACT, name="x"), lambda x: x.dtype.count == prod(y[1] for y in x.arg)),
  (UPat(Ops.UNROLL, name="x"), lambda x: x.src[0].dtype.count == prod(y[1] for y in x.arg)),
])+spec_shared

# these ops can exist in programs but not the tensor spec. example: LOAD
spec_program = PatternMatcher([
  # weakint is not allowed in programs
  (UPat(GroupOp.All, dtypes.weakint), lambda: False),

  # movement ops are not allowed in programs
  (UPat(GroupOp.Movement), lambda: False),

  # Invalid is not allowed in program
  (UPat(Ops.CONST, arg=Invalid), lambda: False),

  # shape of uop must match dtype.count in program
  (UPat(GroupOp.All-{Ops.INS, Ops.NOOP}, name="x"),
   lambda x: False if x.dtype.count > 1 and (x.dtype.count,) != x.shape else None),

  # STACK/GEP in program. TODO: this should match Tensor
  (UPat(Ops.STACK, name="x"), lambda x: len(x.src)>1 and len(x.src) == x.dtype.vcount and all(x.dtype == y.dtype.vec(len(x.src)) for y in x.src)),
  (UPat(Ops.GEP, src=(UPat.var("src"),), name="gep"), lambda gep,src: gep.dtype == src.dtype.scalar()),

  # if has a <gate, index_for_dedup>
  (UPat(Ops.IF, dtype=dtypes.void, src=(UPat(dtype=dtypes.bool), UPat((Ops.CAST, Ops.INDEX)))), lambda: True),
  (UPat(Ops.ENDIF, dtype=dtypes.void, src=(UPat(Ops.IF),)), lambda: True),
])+spec_shared

# these are intermediate ops. everything should be deleted from here
spec_full = PatternMatcher([
  # BUFFER_VIEW on BUFFER is allowed if BUFFER is
  (UPat(Ops.BUFFER_VIEW, src=(UPat((Ops.BUFFER, Ops.PARAM)),)), lambda: True),

  # TODO: BUFFER_VIEW shouldn't go on INDEX. why is this allowed? remove these both
  (UPat(Ops.BUFFER_VIEW, src=(UPat((Ops.INDEX,)),), allow_any_len=True), lambda: True),
  (UPat(Ops.CALL, src=(UPat((Ops.BUFFER_VIEW,)),), allow_any_len=True), lambda: True),

  # codegen may end ranges after gpudims has replaced RANGE with SPECIAL.
  (UPat(Ops.END, src=(UPat(), UPat()), allow_any_len=True), lambda: True),

  # allow any AFTER
  (UPat(Ops.AFTER, src=(UPat(),), allow_any_len=True), lambda: True),

  # expander: unroll/contract/gep/ptrcat/cat
  (UPat((Ops.UNROLL, Ops.CONTRACT), src=(UPat(),)), lambda: True),

  # GEP multi is supported here
  (UPat(Ops.GEP, name="gep"), lambda gep: gep.dtype is dtypes.void or gep.dtype.vcount == len(gep.arg)),

  # all loads/stores
  (UPat((Ops.LOAD, Ops.STORE)), lambda: True),

  # while BIND is being casted
  (UPat(Ops.BIND, (dtypes.int, dtypes.weakint), (UPat(), UPat()), arg=None), lambda: True),

  # TODO: PTRCAT and VCAT need to be deleted

  # PTRCAT is like VECTORIZE, but it functions on ptrs
  (UPat(Ops.PTRCAT, name="x"), lambda x: x.dtype.vcount == sum([y.dtype.base.count for y in x.src])),
  # VCAT is like VECTORIZE, but the srcs can be vectors
  (UPat(Ops.VCAT, name="x"), lambda x: x.dtype.vcount == sum([y.dtype.vcount for y in x.src])),
])+spec_tensor+spec_program

# **** pyrender (move this) ****

# late imports to avoid circular import
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.schedule.rangeify import BufferizeOpts
glbls:dict[str, Any] = {"inf": math.inf, "nan": math.nan, "KernelInfo": KernelInfo, "Metadata": Metadata,
                        "UOp": UOp, "dtypes": dtypes, "Ops": Ops, "AxisType": AxisType, "Invalid": Invalid,
                        "Opt": Opt, "OptOps": OptOps, "BufferizeOpts": BufferizeOpts, "AddrSpace": AddrSpace, "panic": panic,
                        "ConstFloat": ConstFloat}
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
