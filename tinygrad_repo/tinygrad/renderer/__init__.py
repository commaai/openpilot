from __future__ import annotations
from typing import Callable, cast
from dataclasses import dataclass
from tinygrad.helpers import prod, Target, EMULATED_DTYPES
from tinygrad.uop.ops import Ops, UOp, sint, ssimplify, smin, GroupOp, PatternMatcher
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.codegen.opt.tc import TensorCore
from tinygrad.device import Compiler

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
  def from_uops(uops:tuple[UOp, ...], ignore_indexing=False) -> Estimates:
    flops: sint = 0
    lds: sint = 0
    mem: dict[tuple[UOp, Ops], sint] = {}
    mults: sint = 1
    mult_stack: list[sint] = []
    excluded: set[UOp] = set()
    if ignore_indexing:
      for u in uops:
        if u.op in {Ops.INDEX, Ops.SHRINK}:
          excluded = excluded.union(set(UOp.sink(*u.src[1:]).toposort(lambda x: x.op is not Ops.END)))
    for u in uops:
      if u.op in {Ops.LOAD, Ops.STORE}:
        buf = u
        while len(buf.src) and buf.op is not Ops.PARAM: buf = buf.src[0]
        if buf.op is Ops.PARAM:
          # u.src[0] is INDEX, cap at buffer size for re-reads (e.g. matmul)
          accessed = mem.get((buf, u.op), 0) + u.src[0].max_numel() * u.src[0].dtype.scalar().itemsize * mults
          mem[(buf, u.op)] = smin(accessed, buf.max_numel() * buf.dtype.scalar().itemsize)
      if u.op is Ops.RANGE:
        mult_stack.append(mults)
        mults *= cast(sint, u.src[0].ssimplify())
        # SPECIAL are already counted in mults
        mults = mults.substitute({x:x.const_like(0) for x in mults.toposort() if x.op is Ops.SPECIAL}) if isinstance(mults, UOp) else mults
      elif u.op is Ops.END: mults = mult_stack.pop(-1)
      elif u.op is Ops.LOOP: mult_stack.append(mults)  # unbounded loop, unknown trip count
      elif u.op is Ops.SPECIAL: mults *= cast(sint, u.src[0].ssimplify()) # NOTE: we don't push to the mult_stack here, you can't end these
      elif u.op is Ops.PARAM and u.arg.addrspace == AddrSpace.ALU and u.expr == 'core_id': mults *= int(u.vmax) + 1
      elif u.op is Ops.LOAD and u.src[0].addrspace != AddrSpace.REG:
        lds += u.max_numel() * u.dtype.scalar().itemsize * mults
      elif u.op is Ops.STORE and u.src[0].addrspace != AddrSpace.REG:
        lds += u.max_numel() * u.src[1].dtype.scalar().itemsize * mults
      elif u.op in GroupOp.ALU and u not in excluded:
        flops += (mults * (2 if u.op is Ops.MULACC else 1)) * u.max_numel()
      elif u.op is Ops.WMMA and u not in excluded:
        flops += 2 * prod(u.arg[0]) // u.arg[3] * mults
    return Estimates(flops, lds, sum(mem.values()))

class Renderer:
  target: Target
  suffix: str = ""
  # TODO: make this generic with a list of supported types
  supports_float4: bool = True
  has_local: bool = True
  has_threads: bool = False
  has_shared: bool = True
  has_aux: bool = False # additional program info, eg. image shapes
  # NOTE: these two should be in (x,y,z) order to match the max_sizes argument in get_grouped_dims
  global_max: tuple[int, ...]|None = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  local_max: tuple[int, ...]|None = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  global_prod_max: tuple[int, ...]|None = None
  shared_max: int = 32768
  tensor_cores: list[TensorCore] = []
  extra_matcher: PatternMatcher|None = None
  code_for_op: dict[Ops, Callable] = {}

  compiler: Compiler = Compiler()

  def __init__(self, target:Target): self.target = target
  def __reduce__(self): return self.__class__, (self.target,)
  def render(self, uops:list[UOp]) -> str: raise NotImplementedError("needs a renderer")
  def asm(self, prg:UOp, lin:UOp) -> bytes: raise NotImplementedError("needs an assembler")
  def aux(self, uops:list[UOp]) -> dict: raise NotImplementedError("needs aux")
  def supported_dtypes(self) -> set[DType]:
    # double can't be bitcast to anything without long support
    return set(dtypes.all) - ({dtypes.double} if dtypes.long in EMULATED_DTYPES.tolist(dtypes) else set())
