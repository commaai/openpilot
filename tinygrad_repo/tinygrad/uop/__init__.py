from enum import auto, IntEnum, Enum

# wrapper around IntEnum that preserves Enum.__str__ and makes auto() unique across all FastEnum subclasses
class FastEnum(IntEnum):
  def __str__(self): return Enum.__str__(self)
  @staticmethod
  def _generate_next_value_(_, __, ___, last_values): return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])

# the order of these Ops controls the order of the toposort
class Ops(FastEnum):
  # uops that aren't rendered
  SINK = auto(); CONTIGUOUS = auto(); CONTIGUOUS_BACKWARD = auto(); DETACH = auto(); KERNEL = auto(); UNIQUE = auto() # noqa: E702

  # MetaOps
  COPY = auto(); BUFFER_VIEW = auto(); MSELECT = auto(); MSTACK = auto() # noqa: E702

  # blocks in linearizer
  BLOCK = auto(); BLOCKSTART = auto(); BLOCKEND = auto(); BLOCKFINAL = auto() # noqa: E702

  # movement ops!
  RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); FLIP = auto() # noqa: E702

  # misc ops
  UNROLL = auto(); CONTRACT = auto() # noqa: E702
  VIEW = auto(); DEFINE_GLOBAL = auto(); BUFFER = auto() # noqa: E702
  DEFINE_VAR = auto(); DEFINE_LOCAL = auto(); DEFINE_ACC = auto() # noqa: E702
  VALID = auto(); SPECIAL = auto(); NOOP = auto() # noqa: E702

  # reduce
  REDUCE_AXIS = auto(); REDUCE = auto(); ALLREDUCE = auto() # noqa: E702

  # helper ops
  GEP = auto(); VECTORIZE = auto(); CAT = auto(); PTRCAT = auto() # noqa: E702

  # UnaryOps
  CAST = auto(); BITCAST = auto(); EXP2 = auto(); LOG2 = auto(); SIN = auto(); SQRT = auto(); RECIP = auto(); NEG = auto() # noqa: E702

  # load/store before math
  LOAD = auto(); STORE = auto() # noqa: E702

  # early INDEX
  INDEX = auto()

  # math ops
  WMMA = auto()

  # BinaryOps
  ADD = auto(); MUL = auto(); SHL = auto(); SHR = auto(); IDIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto(); CMPNE = auto() # noqa: E702
  XOR = auto(); OR = auto(); AND = auto(); THREEFRY = auto(); SUB = auto(); FDIV = auto(); POW = auto() # noqa: E702

  # TernaryOps
  WHERE = auto(); MULACC = auto() # noqa: E702

  # assignment ops
  ASSIGN = auto()
  BIND = auto()

  # control flow ops
  BARRIER = auto(); RANGE = auto(); IF = auto(); ENDRANGE = auto(); ENDIF = auto(); GBARRIER = auto() # noqa: E702

  # consts last!
  VCONST = auto(); CONST = auto() # noqa: E702

  # device
  DEVICE = auto()
  MULTI = auto()

  # CUSTOMI is inline
  CUSTOM = auto(); CUSTOMI = auto() # noqa: E702
  IGNORE = auto(); FUSE = auto() # noqa: E702

class GroupOp:
  Unary = {Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIP, Ops.NEG}
  Binary = {Ops.ADD, Ops.MUL, Ops.IDIV, Ops.MAX, Ops.MOD, Ops.CMPLT, Ops.CMPNE, Ops.XOR, Ops.SHL, Ops.SHR, Ops.OR, Ops.AND, Ops.THREEFRY,
            Ops.SUB, Ops.FDIV, Ops.POW}
  Ternary = {Ops.WHERE, Ops.MULACC}
  ALU = set.union(Unary, Binary, Ternary)

  Irreducible = {Ops.CONST, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.RANGE}
  Movement = {Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.PAD, Ops.SHRINK, Ops.FLIP}

  Buffer = {Ops.LOAD, Ops.STORE, Ops.VALID, Ops.CONST, Ops.DEFINE_VAR}
  Block = {Ops.BLOCK, Ops.BLOCKEND, Ops.BLOCKSTART}

  # BinaryOps that can be flipped
  Commutative = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE, Ops.XOR, Ops.AND, Ops.OR}

  # BinaryOps where f(f(a,b),c) = f(a,f(b,c))
  Associative = {Ops.ADD, Ops.MUL, Ops.AND, Ops.OR, Ops.MAX}

  # BinaryOps that satisfy f(x,x)=x see https://en.wikipedia.org/wiki/Idempotence
  Idempotent = {Ops.OR, Ops.AND, Ops.MAX}

  # do not preserve f(0) = 0
  UnsafePad = {Ops.RECIP, Ops.LOG2, Ops.EXP2, Ops.IDIV, Ops.POW}

  Meta = {Ops.COPY, Ops.BUFFER_VIEW}

  All = set(Ops)
