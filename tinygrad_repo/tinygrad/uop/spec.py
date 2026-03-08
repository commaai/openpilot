import math
from typing import cast, Any
from tinygrad.uop.ops import PatternMatcher, UPat, GroupOp, Ops, UOp, print_uops, AxisType, KernelInfo, pyrender, Kernel, CustomKernel
from tinygrad.dtype import DType, ImageDType, dtypes, PtrDType, AddrSpace, Invalid, ConstFloat
from tinygrad.helpers import DEBUG, Context, prod, SPEC, Metadata, panic, CHECK_OOB

def validate_index(buf:UOp, idx:UOp, gate:UOp|None=None):
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
    if x.op in {Ops.BITCAST, Ops.VECTORIZE, Ops.GEP} or (x.op is Ops.CAST and isinstance(x.src[0].dtype, PtrDType)): return True

  # if all is good and CHECK_OOB=1, validate with z3
  from tinygrad.uop.validate import validate_index_with_z3
  return validate_index_with_z3(sz, idx, gate)

# four specs:
#   shared_spec  -- usable anywhere
#   tensor_spec  -- usable in tensor graph
#   kernel_spec  -- usable in kernel passed into codegen
#   program_spec -- usable in linearized program
#   full_spec    -- all uops ever created

# *** these uops work anywhere ***

shared_spec = PatternMatcher([
  (UPat(Ops.SINK, dtypes.void), lambda: True), # NOTE: for testing, we let sinks be anything

  # CONST/DEFINE_VAR are everywhere
  (UPat(Ops.CONST, src=(), name="x"), lambda x: type(x.arg) is type(dtypes.as_const(x.arg, x.dtype))),
  (UPat(Ops.DEFINE_VAR, name="x"), lambda x: isinstance(x.arg[1], int) and isinstance(x.arg[2], int)),

  # ALUs: most ALUs have all matching dtypes, except CMPLT, CMPNE, and WHERE
  (UPat(Ops.WHERE, name="w", src=(UPat(dtype=dtypes.bool), UPat.var("x"), UPat.var("y"))), lambda w,x,y: w.dtype == x.dtype == y.dtype),
  (UPat((Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ), dtype=dtypes.bool, src=(UPat.var("x"), UPat.var("y"))), lambda x,y: x.dtype.base == y.dtype.base),
  # and SHL/SHR, the shift distance can be an int
  (UPat((Ops.SHL, Ops.SHR), src=(UPat.var("x"), UPat.var("y")), name="a"), lambda a,x,y: a.dtype == x.dtype and y.dtype in (x.dtype, dtypes.uint)),
  (UPat((Ops.IDIV, Ops.MOD), name="x"), lambda x: None if dtypes.is_int(x.dtype) else False),
  (UPat(GroupOp.ALU, name="x"), lambda x: all(x.dtype.base == y.dtype.base for y in x.src)),

  # CAST
  (UPat((Ops.BITCAST, Ops.CAST), src=(UPat(),), name="x"), lambda x: x.arg is None),

  # RANGE can be in the big graph now
  (UPat(Ops.RANGE, src=(UPat.var("x"),), allow_any_len=True, name="rng"), lambda rng,x:
    rng.dtype == x.dtype and isinstance(rng.arg, tuple) and len(rng.arg) >= 2 and \
      all(isinstance(ra, int) for ra in rng.arg[0:-1]) and isinstance(rng.arg[-1], AxisType)),
  (UPat(Ops.INDEX, src=(UPat(),), allow_any_len=True, name="x"), lambda x: all(y.dtype == dtypes.index for y in x.src[1:]) or None),

  # RANGE/SPECIAL define loops, END closes them
  (UPat(Ops.END, src=(UPat(), UPat(Ops.RANGE))), lambda: True),
])

# ***** UOp spec in the Tensor graph *****

movement_ops = PatternMatcher([
  (UPat((Ops.RESHAPE, Ops.EXPAND), name="mv", src=(UPat.var("x"), UPat(dtype=dtypes.index))), lambda mv,x: True),
  (UPat((Ops.PAD, Ops.SHRINK), name="mv", src=(UPat.var("x"), UPat(dtype=dtypes.index), UPat(dtype=dtypes.index))), lambda mv,x: True),
  (UPat((Ops.PERMUTE, Ops.FLIP), name="mv", src=(UPat.var("x"),)), lambda mv,x: isinstance(mv.arg, tuple)),

  # inputs to movement ops
  (UPat((Ops.VECTORIZE, Ops.VCONST), dtype=dtypes.index), lambda: True),
  (UPat({Ops.ADD, Ops.MUL, Ops.IDIV}, dtype=dtypes.index), lambda: True),

  # AFTER on Movement Op
  (UPat(Ops.AFTER, src=(UPat(GroupOp.Movement.union({Ops.MULTI, Ops.CONTIGUOUS})),), allow_any_len=True), lambda: True),

  # custom kernels allowed here
  (UPat(Ops.CUSTOM_KERNEL), lambda: True),
])

_tensor_spec = PatternMatcher([
  # buffer spec
  (UPat(Ops.UNIQUE, dtypes.void, ()), lambda: True),
  (UPat(Ops.LUNIQUE, dtypes.void, ()), lambda: True),
  (UPat(Ops.DEVICE, dtypes.void, (), name="d"), lambda d:
   isinstance(d.arg, str) or (isinstance(d.arg, tuple) and all(isinstance(s, str) for s in d.arg))),
  (UPat(Ops.BUFFER, src=(UPat((Ops.LUNIQUE, Ops.UNIQUE)), UPat(Ops.DEVICE)), name="buf"),
   lambda buf: isinstance(buf.arg, int) and isinstance(buf.dtype, (DType, ImageDType))),

  # KERNEL can attach to an AFTER to describe the compute required to realize a BUFFER
  (UPat(Ops.KERNEL, src=UPat((Ops.BUFFER, Ops.AFTER, Ops.MSELECT, Ops.MSTACK, Ops.BIND))), lambda: True),

  # ASSIGN has a target and a value. It can also optionally depend on other assigns
  (UPat(Ops.ASSIGN, name="x"), lambda x: len(x.src) >= 2 and all(s.op is Ops.ASSIGN for s in x.src[2:])),

  # MSELECT chooses one of the multi buffers
  (UPat(Ops.MSELECT, name="x"), lambda x: isinstance(x.src[0].device, tuple) and x.arg < len(x.src[0].device)),

  # MSTACK combines buffers into multi
  (UPat(Ops.MSTACK, name="x"), lambda x: all(isinstance(x.device, str) for x in x.src)),

  # Tensor variable bindings
  (UPat(Ops.BIND, (dtypes.int,dtypes.index,), (UPat(Ops.DEFINE_VAR), UPat.cvar(dtype=(dtypes.int,dtypes.index,))), arg=None), lambda: True),
  # single-src BIND used for schedule cache key normalization
  (UPat(Ops.BIND, (dtypes.int,dtypes.index,), (UPat(Ops.DEFINE_VAR),), arg=None), lambda: True),

  # device or unique
  (UPat(Ops.CONST, src=(UPat(Ops.DEVICE),)), lambda: True),
  (UPat(Ops.CONST, src=(UPat((Ops.LUNIQUE, Ops.UNIQUE)), UPat(Ops.DEVICE))), lambda: True),

  # DETACH and CONTIGUOUS change how we interpret the source UOp
  # CONTIGUOUS ensures the source UOp realizes
  (UPat((Ops.DETACH, Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD), name="root", src=(UPat.var("x"),), arg=None),
   lambda root,x: root.dtype == x.dtype),

  # CONTIGUOUS with a range
  (UPat(Ops.CONTIGUOUS, name="root", src=(UPat.var("x"),), allow_any_len=True, arg=None),
   lambda root,x: root.dtype == x.dtype and all(u.op is Ops.RANGE for u in root.src[1:])),

  # COPY/ALLREDUCE/MULTI/ENCDEC
  (UPat(Ops.COPY, name="copy", src=(UPat.var("x"), UPat(Ops.DEVICE)), arg=None), lambda copy,x: copy.dtype == x.dtype),
  (UPat(Ops.ALLREDUCE, name="red", src=(UPat.var("x"), UPat(Ops.DEVICE))), lambda red,x: red.dtype == x.dtype and isinstance(red.arg, Ops)),
  (UPat(Ops.MULTI, name="multi"), lambda multi: all(x.dtype == multi.dtype for x in multi.src) and isinstance(multi.arg, int)),
  (UPat(Ops.ENCDEC, name="x"), lambda x: len(x.src) >= 2), # state + inbuffer

  # REDUCE_AXIS is the reduce in the tensor graph
  (UPat(Ops.REDUCE_AXIS, name="x"), lambda x: isinstance(x.arg, tuple) and len(x.arg) >= 2 and x.arg[0] in {Ops.ADD, Ops.MUL, Ops.MAX}),

  # REDUCE with an outerworld range
  (UPat(Ops.REDUCE, src=(UPat(),), allow_any_len=True, name="x"), lambda x: all(y.dtype == dtypes.index for y in x.src[1:])),

  # AFTER if things were kernelized
  (UPat(Ops.AFTER, src=(UPat((Ops.BUFFER, Ops.AFTER)),), allow_any_len=True), lambda: True),

  # Tensor range bind / store
  (UPat(Ops.BIND, (dtypes.int,dtypes.index,), (UPat(Ops.DEFINE_VAR), UPat(Ops.RANGE)), arg=None), lambda: True),
  (UPat(Ops.STORE, src=(UPat(), UPat())), lambda: True),

  # allow CALL/PARAM
  (UPat(Ops.CALL, src=(UPat(name="f"),), name="c", allow_any_len=True), lambda c,f: c.dtype == f.dtype),
  (UPat(Ops.PARAM), lambda: True),
])+movement_ops+shared_spec

tensor_spec = PatternMatcher([
  # no tags allowed in tensor graph
  (UPat(GroupOp.All, name="x"), lambda x: None if x.tag is None else False),
])+_tensor_spec

# ***** UOp spec in codegen shared between kernel and program *****

shared_codegen_spec = PatternMatcher([
  # DEFINEs
  (UPat(Ops.PARAM, name="x"), lambda x: isinstance(x.dtype, (PtrDType, ImageDType)) and x.dtype.addrspace == AddrSpace.GLOBAL),
  (UPat(Ops.DEFINE_LOCAL, name="x"), lambda x: isinstance(x.dtype, PtrDType) and x.dtype.addrspace == AddrSpace.LOCAL),
  (UPat(Ops.DEFINE_REG, src=(), name="x"), lambda x: isinstance(x.arg, int)),

  # allow AFTER on buffers, GROUP anywhere
  (UPat(Ops.AFTER, src=(UPat(GroupOp.Defines|{Ops.AFTER}),), allow_any_len=True), lambda: True),
  (UPat(Ops.GROUP, dtypes.void), lambda: True),

  # WMMA has a <a, b, acc>
  (UPat(Ops.WMMA, src=(UPat(), UPat(), UPat()), name="x"), lambda x: isinstance(x.arg, tuple) and len(x.arg) == 8),

  # VECTORIZE/GEP
  (UPat(Ops.VECTORIZE, name="x"), lambda x: len(x.src)>1 and len(x.src) == x.dtype.vcount and all(x.dtype == y.dtype.vec(len(x.src)) for y in x.src)),
  (UPat(Ops.GEP, src=(UPat.var("src"),), name="gep"), lambda gep,src: gep.dtype == src.dtype.scalar()),

  # LOAD(idx) / STORE(idx, val)
  (UPat().index(UPat()).or_casted().load(), lambda: True),
  (UPat().index(UPat(), UPat(dtype=dtypes.bool)).or_casted().load(), lambda: True),  # gated load (alt added in program_spec)
  (UPat(Ops.INDEX).or_casted().store(UPat()), lambda: True),

  # CUSTOM (inline and non inline)
  (UPat((Ops.CUSTOMI, Ops.CUSTOM)), lambda: True),

  # INDEX (2-arg and 3-arg with bool gate)
  (UPat(GroupOp.Defines|{Ops.AFTER}, name="buf").index(UPat.var("idx")), validate_index),
  (UPat(Ops.INDEX, src=(UPat(GroupOp.Defines|{Ops.AFTER}, name="buf"), UPat.var("idx"), UPat.var("gate", dtype=dtypes.bool))), validate_index),

  # SPECIAL
  (UPat(Ops.SPECIAL, src=(UPat.var("x", (dtypes.index, dtypes.int32)),), name="s"), lambda s,x: s.dtype == x.dtype and isinstance(s.arg, str)),

  # BARRIER (on any length)
  (UPat(Ops.BARRIER, dtypes.void), lambda: True),
])

# ***** UOp spec in kernel graph *****

kernel_spec = PatternMatcher([
  # index is allowed here
  (UPat(GroupOp.Elementwise|{Ops.CONST, Ops.RANGE, Ops.DEFINE_VAR}, dtype=dtypes.index), lambda: True),

  # UNROLL/CONTRACT is used here for WMMA
  (UPat(Ops.CONTRACT, name="x"), lambda x: x.dtype.count == prod(y[1] for y in x.arg)),
  (UPat(Ops.UNROLL, name="x"), lambda x: x.src[0].dtype.count == prod(y[1] for y in x.arg)),

  # END can end multiple axes here
  (UPat(Ops.END, src=(UPat(), UPat()), allow_any_len=True), lambda: True),

  # bufferize can be on anything
  (UPat(Ops.BUFFERIZE, src=(UPat(),), allow_any_len=True, name="x"), lambda x: True),

  # reduce must be on ranges
  (UPat(Ops.REDUCE, src=(UPat(),), allow_any_len=True, name="x"), lambda x: all(y.dtype in (dtypes.index, dtypes.int) for y in x.src[1:])),
])+movement_ops+shared_codegen_spec+shared_spec

# ***** UOp spec in linearized programs *****

program_spec = PatternMatcher([
  # LOAD (idx, alt_value), LOAD can have an alt value, but only if the index has a gate
  (UPat().index(UPat(), UPat(dtype=dtypes.bool)).or_casted().load(UPat()), lambda: True),

  # END closes ranges
  (UPat(Ops.END, src=(UPat(), UPat(Ops.RANGE)), dtype=dtypes.void), lambda: True),

  # make sure all index dtypes have been lowered
  (UPat(GroupOp.All, dtype=dtypes.index), lambda: False),
  (UPat(Ops.CONST, arg=Invalid), lambda: False),
  (UPat(Ops.VCONST, name="x"), lambda x: all(v is not Invalid for v in x.arg) and len(x.arg)==x.dtype.vcount>1 and
    type(x.arg) is type(dtypes.as_const(x.arg, x.dtype))),

  # if has a <gate, index_for_dedup>
  (UPat(Ops.IF, dtype=dtypes.void, src=(UPat(dtype=dtypes.bool), UPat((Ops.CAST, Ops.INDEX)))), lambda: True),
  (UPat(Ops.ENDIF, dtype=dtypes.void, src=(UPat(Ops.IF),)), lambda: True),
])+shared_codegen_spec+shared_spec

# *** this spec should match all UOps ever created ***

full_spec = PatternMatcher([
  # NOOP in the full spec
  (UPat(Ops.NOOP), lambda: True),

  # all rewrite error are okay
  (UPat(Ops.REWRITE_ERROR), lambda: True),

  # rangeify: buffer view with index or load is okay
  (UPat(Ops.BUFFER_VIEW, src=(UPat((Ops.INDEX, Ops.LOAD)),)), lambda: True),
  # assign on index. the third op is the shape
  (UPat(Ops.ASSIGN, src=(UPat(), UPat(), UPat())), lambda: True),

  # expander: unroll/contract/gep/ptrcat/cat
  (UPat((Ops.UNROLL, Ops.CONTRACT), src=(UPat(),)), lambda: True),
  # GEP multi is supported here
  (UPat(Ops.GEP, name="gep"), lambda gep: gep.dtype is dtypes.void or gep.dtype.vcount == len(gep.arg)),
  # PTRCAT is like VECTORIZE, but it functions on ptrs
  (UPat(Ops.PTRCAT, name="x"), lambda x: x.dtype.vcount == sum([y.dtype.base.count for y in x.src])),
  # CAT is like VECTORIZE, but the srcs can be vectors
  (UPat(Ops.CAT, name="x"), lambda x: x.dtype.vcount == sum([y.dtype.vcount for y in x.src])),
  # vectorized index
  (UPat(Ops.INDEX, src=(UPat((Ops.VECTORIZE, Ops.CAST)), UPat())), lambda: True),

  # linearizer: outputs + intermediate KERNELs
  (UPat(Ops.KERNEL, dtype=dtypes.void), lambda: True),

  # Invalid must have type Index
  (UPat(Ops.CONST, arg=Invalid, name="x"), lambda x: x.dtype.scalar() == dtypes.index),
  # where on index in rhs position is fine
  (UPat(Ops.WHERE, dtype=dtypes.index, src=(UPat(dtype=dtypes.bool), UPat(), UPat(dtype=dtypes.index))), lambda: True),
  # allow index dtype on a restricted set of UOps
  (UPat((Ops.ADD, Ops.MUL, Ops.MOD, Ops.IDIV, Ops.MAX,
         Ops.SPECIAL, Ops.CAST, Ops.RANGE, Ops.VCONST, Ops.VECTORIZE), dtype=dtypes.index), lambda: True),

  # while BIND is being casted
  (UPat(Ops.BIND, (dtypes.int, dtypes.index), (UPat(), UPat()), arg=None), lambda: True),

  # in progress MSTACK may lose device
  (UPat((Ops.MSELECT, Ops.MSTACK), name="x"), lambda x: True),

  # codegen: PROGRAM with progressive sources through the pipeline (SINK, DEVICE, LINEAR?, SOURCE?, BINARY?)
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat(Ops.SINK), UPat(Ops.DEVICE))), lambda: True),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat(Ops.SINK), UPat(Ops.DEVICE), UPat(Ops.LINEAR))), lambda: True),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat(Ops.SINK), UPat(Ops.DEVICE), UPat(Ops.LINEAR), UPat(Ops.SOURCE))), lambda: True),
  (UPat(Ops.PROGRAM, dtypes.void, src=(UPat(Ops.SINK), UPat(Ops.DEVICE), UPat(Ops.LINEAR), UPat(Ops.SOURCE), UPat(Ops.BINARY))), lambda: True),
  # codegen: standalone LINEAR/SOURCE/BINARY
  (UPat(Ops.LINEAR, dtypes.void), lambda: True),
  (UPat(Ops.SOURCE, dtypes.void, src=()), lambda: True),
  (UPat(Ops.BINARY, dtypes.void, src=()), lambda: True),

  # temp VECTORIZE/INDEX during rewrite have the wrong dtype
  (UPat(Ops.VECTORIZE), lambda: True),
  (UPat(Ops.INDEX), lambda: True),

  # all loads/stores
  (UPat((Ops.LOAD, Ops.STORE)), lambda: True),
  # DEFINE_VAR to deal with the floats used in reduce collapse
  (UPat(Ops.DEFINE_VAR, dtype=dtypes.floats), lambda: True),
  # allow any AFTER
  (UPat(Ops.AFTER, src=(UPat(),), allow_any_len=True), lambda: True),
])+_tensor_spec+kernel_spec+program_spec+shared_spec

# ***** uop helpers *****

def type_verify(ast:UOp|list[UOp], check_spec:PatternMatcher):
  lst = list(ast.toposort()) if isinstance(ast, UOp) else ast
  if SPEC > 1: test_pyrender(lst[-1])  # assume this is the sink

  for i,u in enumerate(lst):
    with Context(TRACK_MATCH_STATS=0): ret = check_spec.rewrite(u)
    if cast(bool|None, ret) is not True:
      if DEBUG >= 3: print_uops(lst)
      raise RuntimeError(f"UOp verification failed at {i} on {u.op} {u.dtype} {len(u.src)} {[(x.op, x.dtype, x.arg) for x in u.src]} {u.arg}")

# late imports to avoid circular import
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.schedule.rangeify import BufferizeOpts
glbls:dict[str, Any] = {"inf": math.inf, "nan": math.nan, "KernelInfo": KernelInfo, "Kernel": Kernel, "Metadata": Metadata,
                        "UOp": UOp, "dtypes": dtypes, "Ops": Ops, "AxisType": AxisType, "Invalid": Invalid, "CustomKernel": CustomKernel,
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
