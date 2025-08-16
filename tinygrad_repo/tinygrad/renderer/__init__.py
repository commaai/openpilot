from __future__ import annotations
from typing import Callable, cast, TYPE_CHECKING
import functools, itertools
from dataclasses import dataclass, field, replace
from tinygrad.helpers import to_function_name, dedup, prod
from tinygrad.uop.ops import Ops, UOp, sym_infer, sint, Variable, ssimplify, GroupOp, PatternMatcher
from tinygrad.dtype import AddrSpace, PtrDType
if TYPE_CHECKING:
  from tinygrad.codegen.opt.tc import TensorCore
  from tinygrad.codegen.opt.kernel import Opt

@dataclass(frozen=True)
class Estimates:
  # number of FLOPS used in the Kernel
  ops:sint = 0
  # bytes accessed in loads and stores
  lds:sint = 0
  # total bytes accessed, counting only once for bytes that are accessed multiple times
  mem:sint = 0
  def __add__(self, o:Estimates): return Estimates(self.ops + o.ops, self.lds + o.lds, self.mem + o.mem)
  def simplify(self): return Estimates(ssimplify(self.ops), ssimplify(self.lds), ssimplify(self.mem))
  @staticmethod
  def from_uops(uops:list[UOp], ignore_indexing=False) -> Estimates:
    flops: sint = 0
    lds: sint = 0
    mults: sint = 1
    mult_stack: list[sint] = []
    dont_count: set[UOp] = set()
    if ignore_indexing:
      for u in uops:
        if u.op in {Ops.LOAD, Ops.STORE} and (not isinstance(u.src[0].dtype, PtrDType) or u.src[0].dtype.addrspace != AddrSpace.REG):
          dont_count = dont_count.union(u.src[0].toposort())
          if len(u.src) > 2: dont_count = dont_count.union(u.src[2].toposort())
        elif u.op is Ops.IF:
          dont_count = dont_count.union(u.src[0].toposort())
    for u in uops:
      if u.op is Ops.RANGE:
        mult_stack.append(mults)
        mults *= cast(sint, u.src[0].ssimplify())
        # SPECIAL are already counted in mults
        mults = mults.substitute({x:x.const_like(0) for x in mults.toposort() if x.op is Ops.SPECIAL}) if isinstance(mults, UOp) else mults
      elif u.op is Ops.ENDRANGE: mults = mult_stack.pop(-1)
      elif u.op is Ops.SPECIAL: mults *= u.arg[1] # NOTE: we don't push to the mult_stack here, you can't end these
      elif u.op is Ops.LOAD and (not isinstance(u.src[0].dtype, PtrDType) or u.src[0].dtype.addrspace != AddrSpace.REG):
        lds += u.dtype.itemsize * mults
      elif u.op is Ops.STORE and (not isinstance(u.src[0].dtype, PtrDType) or u.src[0].dtype.addrspace != AddrSpace.REG):
        lds += u.src[1].dtype.itemsize * mults
      elif u.op in GroupOp.ALU and u not in dont_count: flops += (mults * (2 if u.op is Ops.MULACC else 1)) * u.dtype.count
      elif u.op is Ops.WMMA and u not in dont_count: flops += 2 * prod(u.arg[1]) // u.arg[5] * mults
    return Estimates(flops, lds, lds) # TODO: properly track memory, lds is always a high estimate

@dataclass
class ProgramSpec:
  name:str
  src:str
  device:str
  ast:UOp  # save the base ast (this is method cache key)
  uops:list[UOp]|None=None

  # filled in from uops (if we have uops)
  global_size:list[int]|None=None
  local_size:list[int]|None=None
  vars:list[Variable]=field(default_factory=list)
  globals:list[int]=field(default_factory=list)
  outs:list[int]=field(default_factory=list)
  ins:list[int]=field(default_factory=list)
  _ran_post_init:bool=False  # NOTE: this is needed if you call replace on the Program

  def __post_init__(self):
    if not self._ran_post_init and self.uops is not None:
      # single pass through the uops
      for u in self.uops:
        if u.op is Ops.DEFINE_VAR: self.vars.append(u)
        if u.op is Ops.DEFINE_GLOBAL: self.globals.append(u.arg)
        if u.op is Ops.STORE: self.outs.extend([x.arg for x in u.src[0].toposort() if x.op is Ops.DEFINE_GLOBAL])
        if u.op is Ops.LOAD: self.ins.extend([x.arg for x in u.src[0].toposort() if x.op is Ops.DEFINE_GLOBAL])
        if u.op is Ops.SPECIAL:
          # NOTE: you have to set local_size and global_size to the base [1,1,1] outside this
          if u.arg[0][0] == 'i': self.local_size = None
          special_size = self.local_size if u.arg[0][0] == 'l' else self.global_size
          if special_size is not None: special_size[int(u.arg[0][-1])] = u.arg[1]
      self.vars = sorted(self.vars, key=lambda v: v.arg)
      self.outs = sorted(dedup(self.outs))
      self.ins = sorted(dedup(self.ins))
      self._ran_post_init = True

  @functools.cached_property
  def mem_estimate(self) -> sint:
    # group non-local bufs by the op type (LOAD or STORE) and the buffer arg. take the max access of that buffer in bytes
    # TODO: these max and min don't work on symbolic, and results are very wrong.
    return sum(max(x.src[0].dtype.nbytes() for x in group)
      for _, group in itertools.groupby([x for x in self.ast.toposort() if x.op in {Ops.LOAD, Ops.STORE} and x.src[0].base.op is Ops.DEFINE_GLOBAL],
                        key=lambda x: (x.op, x.src[0].base.arg)))

  @functools.cached_property
  def estimates(self) -> Estimates:
    return replace(Estimates() if self.uops is None else Estimates.from_uops(self.uops, ignore_indexing=True), mem=self.mem_estimate)

  @functools.cached_property
  def function_name(self) -> str: return to_function_name(self.name)

  @property
  def applied_opts(self) -> tuple[Opt, ...]|None: return self.uops[-1].arg.applied_opts if \
    self.uops is not None and self.uops[-1].op is Ops.SINK and self.uops[-1].arg is not None else None

  def launch_dims(self, var_vals:dict[Variable, int]):
    global_size = [sym_infer(sz, var_vals) for sz in self.global_size] if self.global_size is not None else None
    local_size = [sym_infer(sz, var_vals) for sz in self.local_size] if self.local_size is not None else None
    return global_size, local_size

class Renderer:
  device: str = ""
  suffix: str = ""
  # TODO: make this generic with a list of supported types
  supports_float4: bool = True
  has_local: bool = True
  has_shared: bool = True
  # NOTE: these two should be in (x,y,z) order to match the max_sizes argument in get_grouped_dims
  global_max: tuple[int, ...]|None = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  local_max: tuple[int, ...]|None = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  shared_max: int = 32768
  tensor_cores: list[TensorCore] = []
  pre_matcher: PatternMatcher|None = None
  extra_matcher: PatternMatcher|None = None
  code_for_op: dict[Ops, Callable] = {}

  def __reduce__(self): return self.__class__, ()
  def render(self, uops:list[UOp]) -> str: raise NotImplementedError("needs a renderer")
