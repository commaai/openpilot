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

  # TODO: unify these ops into the levels of the memory hierarchy
  DEFINE_GLOBAL = auto(); DEFINE_LOCAL = auto(); DEFINE_REG = auto()

  # this is for symbolic shapes
  DEFINE_VAR = auto(); BIND = auto()

  # this is a RANGE for GPU dimensions, similar to symbolic shapes but not exactly
  SPECIAL = auto()

  # ** 2 -- non op uops **

  # uops that aren't rendered
  NOOP = auto(); SINK = auto(); PRECAST = auto()

  # AFTER passes src[0] through and promises in the toposort that any consumers of the AFTER run after src[1:]
  AFTER = auto()

  # GROUP is a NOOP that just merges things together
  GROUP = auto()

  # vector creation / item selection
  GEP = auto(); VECTORIZE = auto()

  # ** 3 -- load/store **

  # INDEX is a BinaryOp similar to ADD, but it operates on pointers
  INDEX = auto()

  # load/store before math
  LOAD = auto(); STORE = auto()

  # ** 4 -- math **

  # tensor core math op, not elementwise
  WMMA = auto()

  # UnaryOps
  CAST = auto(); BITCAST = auto(); EXP2 = auto(); LOG2 = auto(); SIN = auto()
  SQRT = auto(); RECIPROCAL = auto(); NEG = auto(); TRUNC = auto()

  # BinaryOps
  ADD = auto(); MUL = auto(); SHL = auto(); SHR = auto(); IDIV = auto(); MAX = auto(); MOD = auto()
  CMPLT = auto(); CMPNE = auto(); CMPEQ = auto()
  XOR = auto(); OR = auto(); AND = auto()
  THREEFRY = auto(); SUB = auto(); FDIV = auto(); POW = auto()

  # TernaryOps
  WHERE = auto(); MULACC = auto()

  # ** 5 -- control flow / consts / custom **

  # control flow ops
  BARRIER = auto(); RANGE = auto(); IF = auto(); END = auto(); ENDIF = auto()

  # consts. VCONST is a vectorized const
  VCONST = auto(); CONST = auto()

  # CUSTOM/CUSTOMI are used to output strings into codegen. the I makes the string inline
  CUSTOM = auto(); CUSTOMI = auto()

  # ** 6 -- ops that don't exist in programs **

  # tensor graph ops
  UNIQUE = auto(); DEVICE = auto(); KERNEL = auto()
  ASSIGN = auto()

  # buffer ops
  BUFFERIZE = auto(); COPY = auto(); BUFFER = auto(); BUFFER_VIEW = auto(); MSELECT = auto(); MSTACK = auto()

  # ops that adjust the behavior of the scheduler
  CONTIGUOUS = auto(); CONTIGUOUS_BACKWARD = auto(); DETACH = auto()

  # movement ops! these only exist in the tensor graph
  RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); FLIP = auto()
  MULTI = auto()  # MULTI is really a movement op

  # reduce
  REDUCE_AXIS = auto(); REDUCE = auto(); ALLREDUCE = auto()

  # errors/placeholders
  REWRITE_ERROR = auto(); SENTINEL = auto()

  # expander ops
  UNROLL = auto(); CONTRACT = auto(); CAT = auto(); PTRCAT = auto()

class GroupOp:
  Unary = {Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIPROCAL, Ops.NEG, Ops.TRUNC}
  Binary = {Ops.ADD, Ops.MUL, Ops.IDIV, Ops.MAX, Ops.MOD, Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ,
            Ops.XOR, Ops.SHL, Ops.SHR, Ops.OR, Ops.AND, Ops.THREEFRY, Ops.SUB, Ops.FDIV, Ops.POW}
  Ternary = {Ops.WHERE, Ops.MULACC}
  ALU = set.union(Unary, Binary, Ternary)

  # TODO: is BITCAST always Elementwise if it's shape changing?
  Elementwise = set.union(ALU, {Ops.CAST, Ops.BITCAST})

  Defines = {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}

  Irreducible = {Ops.CONST, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.RANGE}
  Movement = {Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.PAD, Ops.SHRINK, Ops.FLIP}

  Buffer = {Ops.LOAD, Ops.STORE, Ops.CONST, Ops.DEFINE_VAR}

  # BinaryOps that can be flipped
  Commutative = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE, Ops.CMPEQ, Ops.XOR, Ops.AND, Ops.OR}

  # BinaryOps where f(f(a,b),c) = f(a,f(b,c))
  Associative = {Ops.ADD, Ops.MUL, Ops.AND, Ops.OR, Ops.MAX}

  # BinaryOps that satisfy f(x,x)=x see https://en.wikipedia.org/wiki/Idempotence
  Idempotent = {Ops.OR, Ops.AND, Ops.MAX}

  # These can change the dtype to bool
  Comparison = {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ}

  # do not preserve f(0) = 0
  UnsafePad = {Ops.RECIPROCAL, Ops.LOG2, Ops.EXP2, Ops.IDIV, Ops.POW}

  All = set(Ops)
