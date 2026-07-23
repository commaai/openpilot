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

  # BIND pairs a symbolic PARAM with a concrete value
  BIND = auto()

  # this is a RANGE for GPU dimensions, similar to symbolic shapes but not exactly
  SPECIAL = auto()

  # BUFFER allocates global/local/register storage depending on its addrspace
  BUFFER = auto()

  # ** 2 -- non op uops **

  # uops that aren't rendered
  NOOP = auto(); REWRITE_ERROR = auto()
  # FUNCTION has a TUPLE body and is gradient-able; CALL is an opaque kernel invocation
  PARAM = auto(); FUNCTION = auto(); CALL = auto()

  # renderer
  # LINEAR is a list of UOps, SOURCE has a str arg that's human readable, BINARY has bytes arg that's compiled
  PROGRAM = auto(); LINEAR = auto(); SOURCE = auto(); BINARY = auto()

  # AFTER passes src[0] through and promises in the toposort that any consumers of the AFTER run after src[1:]
  # GROUP is a NOOP that just merges things together
  SINK = auto(); AFTER = auto(); GROUP = auto()

  # vector creation / item selection
  STACK = auto()

  # tuple/gettuple for function with multiple returns
  TUPLE = auto(); GETTUPLE = auto()

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
  SQRT = auto(); RECIPROCAL = auto(); NEG = auto(); TRUNC = auto()

  # BinaryOps
  ADD = auto(); MUL = auto(); SHL = auto(); SHR = auto(); CDIV = auto(); MAX = auto(); CMOD = auto()
  CMPLT = auto(); CMPNE = auto(); CMPEQ = auto()
  XOR = auto(); OR = auto(); AND = auto()
  THREEFRY = auto(); SUB = auto(); FDIV = auto(); POW = auto()
  FLOORDIV = auto(); FLOORMOD = auto()

  # TernaryOps
  WHERE = auto(); MULACC = auto()

  # ** 5 -- control flow / consts / custom **

  # control flow ops
  BARRIER = auto(); RANGE = auto(); LOOP = auto(); IF = auto(); END = auto(); ENDIF = auto(); WAIT = auto()

  # const.
  CONST = auto()

  # CUSTOM/CUSTOMI are used to output strings into codegen. the I makes the string inline
  CUSTOM = auto(); CUSTOMI = auto()

  # INS is a machine instruction
  INS = auto()

  # ** 6 -- ops that don't exist in programs **

  # ops that adjust the behavior of the scheduler
  CONTIGUOUS = auto(); CONTIGUOUS_BACKWARD = auto(); DETACH = auto()

  # buffer ops
  STAGE = auto(); COPY = auto(); SLICE = auto(); MSELECT = auto(); MSTACK = auto(); CUSTOM_FUNCTION = auto()

  # the core 6 movement ops! these only exist in the tensor graph
  RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); FLIP = auto()
  MULTI = auto()  # MULTI is really a movement op

  # reduce
  REDUCE = auto(); ALLREDUCE = auto()

  # ** 7 -- pattern compiler IR (used in upat.py) **
  # PYLITERAL carries a Python literal as an arg for CUSTOM predicates
  PYLITERAL = auto()

class GroupOp:
  Unary = {Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIPROCAL, Ops.NEG, Ops.TRUNC}
  Binary = {Ops.ADD, Ops.MUL, Ops.CDIV, Ops.MAX, Ops.CMOD, Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ,
            Ops.XOR, Ops.SHL, Ops.SHR, Ops.OR, Ops.AND, Ops.THREEFRY, Ops.SUB, Ops.FDIV, Ops.POW, Ops.FLOORDIV, Ops.FLOORMOD}
  Ternary = {Ops.WHERE, Ops.MULACC}
  ALU = set.union(Unary, Binary, Ternary)
  Broadcastable = set.union(Binary, Ternary)

  # TODO: is BITCAST always Elementwise if it's shape changing?
  Elementwise = set.union(ALU, {Ops.CAST, Ops.BITCAST})

  Defines = {Ops.PARAM, Ops.BUFFER}

  Irreducible = {Ops.CONST, Ops.SPECIAL, Ops.RANGE, Ops.PARAM, Ops.GETADDR}
  Movement = {Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.PAD, Ops.SHRINK, Ops.FLIP}

  # BinaryOps that can be flipped
  Commutative = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE, Ops.CMPEQ, Ops.XOR, Ops.AND, Ops.OR}

  # BinaryOps where f(f(a,b),c) = f(a,f(b,c))
  Associative = {Ops.ADD, Ops.MUL, Ops.AND, Ops.OR, Ops.MAX}

  # BinaryOps that satisfy f(x,x)=x see https://en.wikipedia.org/wiki/Idempotence
  Idempotent = {Ops.OR, Ops.AND, Ops.MAX}

  # ALU ops valid as the reduce op in REDUCE/ALLREDUCE arg
  Reduce = {Ops.ADD, Ops.MUL, Ops.MAX}

  # These can change the dtype to bool
  Comparison = {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ}

  All = set(Ops)
