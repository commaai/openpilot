# flake8: noqa: E702
# allow semicolons to put multiple ops on one line
from enum import auto, IntEnum, Enum

# wrapper around IntEnum that preserves Enum.__str__ and makes auto() unique across all FastEnum subclasses
class FastEnum(IntEnum):
  def __str__(self): return Enum.__str__(self)
  def __repr__(x): return str(x)
  @staticmethod
  def _generate_next_value_(_, __, ___, last_values): return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])

# the order of these Ops controls the order of the toposort
class Ops(FastEnum):
  # ** 1 -- defines/special **

  # this is a RANGE for GPU dimensions, similar to symbolic shapes but not exactly
  SPECIAL = auto()

  # BUFFER references bound storage; ALLOC declares unbound global storage; EXTERN is storage from outside an exported function
  BUFFER = auto(); ALLOC = auto(); EXTERN = auto()

  # ** 2 -- non op uops **

  # uops that aren't rendered
  NOOP = auto(); REWRITE_ERROR = auto()
  # CALL is a kernel invocation; value-producing calls write to ALLOC arguments
  PARAM = auto(); REFPARAM = auto(); CALL = auto()

  # renderer
  # LINEAR is a list of UOps, SOURCE has a str arg that's human readable, BINARY has bytes arg that's compiled
  PROGRAM = auto(); LINEAR = auto(); SOURCE = auto(); BINARY = auto()

  # AFTER passes src[0] through and promises in the toposort that any consumers of the AFTER run after src[1:]
  SINK = auto(); AFTER = auto()

  # vector creation / item selection
  STACK = auto()

  # hcq specific
  GETADDR = auto()

  # ** 3 -- load/store **

  # INDEX is a BinaryOp similar to ADD, but it operates on pointers
  INDEX = auto(); SHRINK = auto()

  # load/store before math
  LOAD = auto(); STORE = auto()

  # ** 4 -- math **

  # tensor core math op, not elementwise
  WMMA = auto()

  # UnaryOps
  CAST = auto(); BITCAST = auto(); EXP2 = auto(); LOG2 = auto(); SIN = auto()
  SQRT = auto(); RECIP = auto(); NEG = auto(); TRUNC = auto()

  # BinaryOps
  ADD = auto(); MUL = auto(); SHL = auto(); SHR = auto(); TRUNCDIV = auto(); MAX = auto(); TRUNCMOD = auto()
  CMPLT = auto(); CMPNE = auto(); CMPEQ = auto()
  XOR = auto(); OR = auto(); AND = auto()
  THREEFRY = auto(); SUB = auto(); FDIV = auto(); POW = auto()
  DIV = auto(); MOD = auto()

  # TernaryOps
  WHERE = auto(); MULACC = auto()

  # ** 5 -- control flow / consts / custom **

  # control flow ops
  BARRIER = auto(); RANGE = auto(); IF = auto(); END = auto(); ENDIF = auto()
  BACKEDGE = auto()  # (body, unbounded RANGE, condition): repeat while condition is true

  # const.
  CONST = auto()

  # ** 6 -- ops that don't exist in programs **

  # ops that adjust the behavior of the scheduler
  CONTIGUOUS_BACKWARD = auto(); DETACH = auto()

  # buffer ops
  STAGE = auto(); COPY = auto(); MSELECT = auto(); MSTACK = auto(); NATIVE = auto()

  # the core 5 movement ops! these only exist in the tensor graph
  RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto()
  UNSHARD = auto()  # UNSHARD is really a movement op

  # reduce
  REDUCE = auto(); ALLREDUCE = auto()

  # ** 7 -- pattern compiler IR (used in upat.py) **
  # PYLITERAL carries a Python literal as an arg for custom predicates
  PYLITERAL = auto()

class GroupOp:
  Unary = {Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIP, Ops.NEG, Ops.TRUNC}
  Binary = {Ops.ADD, Ops.MUL, Ops.TRUNCDIV, Ops.MAX, Ops.TRUNCMOD, Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ,
            Ops.XOR, Ops.SHL, Ops.SHR, Ops.OR, Ops.AND, Ops.THREEFRY, Ops.SUB, Ops.FDIV, Ops.POW, Ops.DIV, Ops.MOD}
  Ternary = {Ops.WHERE, Ops.MULACC}
  ALU = set.union(Unary, Binary, Ternary)
  Broadcastable = set.union(Binary, Ternary)

  # TODO: is BITCAST always Elementwise if it's shape changing?
  Elementwise = set.union(ALU, {Ops.CAST, Ops.BITCAST})
  UnaryElementwise = Unary | {Ops.CAST}

  Defines = {Ops.PARAM, Ops.REFPARAM, Ops.BUFFER, Ops.ALLOC, Ops.EXTERN, Ops.BINARY}

  Irreducible = {Ops.CONST, Ops.SPECIAL, Ops.RANGE, Ops.PARAM, Ops.GETADDR}
  Movement = {Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.PAD, Ops.SHRINK}
  # ops with a shape from the shape of src[0] and the arg
  ShapeChanging = Movement | {Ops.UNSHARD, Ops.REDUCE}

  # ops with the dtype of their first src, and ops with the promoted dtype of their srcs. the other ops are in dtype_from_uop
  PassthroughDType = {Ops.LOAD, Ops.UNSHARD, Ops.REDUCE, Ops.AFTER, Ops.RANGE, Ops.CONTIGUOUS_BACKWARD, Ops.COPY, Ops.STAGE, Ops.DETACH,
                      Ops.MSTACK, Ops.MSELECT, Ops.ALLREDUCE, Ops.SPECIAL, Ops.END, Ops.NEG, Ops.TRUNC} | Movement

  # ops that may have a side effect each time they run
  Effect = {Ops.STORE, Ops.CALL, Ops.BARRIER}

  # BinaryOps that can be flipped
  Commutative = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE, Ops.CMPEQ, Ops.XOR, Ops.AND, Ops.OR}

  # BinaryOps where f(f(a,b),c) = f(a,f(b,c))
  Associative = {Ops.ADD, Ops.MUL, Ops.AND, Ops.OR, Ops.MAX, Ops.XOR}

  # BinaryOps that satisfy f(x,x)=x see https://en.wikipedia.org/wiki/Idempotence
  Idempotent = {Ops.OR, Ops.AND, Ops.MAX}

  # ALU ops valid as the reduce op in REDUCE/ALLREDUCE arg
  Reduce = {Ops.ADD, Ops.MUL, Ops.MAX}

  # These can change the dtype to bool
  Comparison = {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ}
  PromoteDType = Broadcastable - Comparison - {Ops.WHERE, Ops.THREEFRY, Ops.FDIV, Ops.SHL, Ops.SHR}

  All = set(Ops)
