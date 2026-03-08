from __future__ import annotations
from typing import Callable, cast, TYPE_CHECKING
import functools
from dataclasses import dataclass, field
from tinygrad.helpers import to_function_name, dedup, prod, DEBUG
from tinygrad.uop.ops import Ops, UOp, sym_infer, sint, Variable, ssimplify, GroupOp, PatternMatcher, print_uops, KernelInfo
from tinygrad.dtype import AddrSpace, PtrDType
from tinygrad.codegen.opt.tc import TensorCore
from tinygrad.codegen.opt import Opt
if TYPE_CHECKING: from tinygrad.device import Compiler

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
    mem: dict[tuple[UOp, Ops], sint] = {}
    mults: sint = 1
    mult_stack: list[sint] = []
    dont_count: set[UOp] = set()
    if ignore_indexing:
      def range_gate(x): return x.op is not Ops.RANGE
      for u in uops:
        if u.op in {Ops.LOAD, Ops.STORE}:
          # if u.src[0] is INDEX, we have to include the buffer since it might be an AFTER
          dont_count = dont_count.union((UOp.sink(*u.src[0].src[1:]) if u.src[0].op is Ops.INDEX else u.src[0]).toposort(range_gate))
          # TODO: is this correct? this all needs to be cleaned up
          if len(u.src) > 2: dont_count = dont_count.union(u.src[2].toposort())
        elif u.op is Ops.IF:
          dont_count = dont_count.union(u.src[0].toposort())
    for u in uops:
      if u.op is Ops.SINK and isinstance(u.arg, KernelInfo) and u.arg.estimates is not None: return u.arg.estimates
      if u.op in {Ops.LOAD, Ops.STORE}:
        buf = u
        while len(buf.src): buf = buf.src[0]
        if buf.op is Ops.PARAM: # assume all DEFINE_GLOBAL memory is accessed
          mem[(buf, u.op)] = buf.ptrdtype.size * buf.dtype.itemsize
      if u.op is Ops.RANGE:
        mult_stack.append(mults)
        mults *= cast(sint, u.src[0].ssimplify())
        # SPECIAL are already counted in mults
        mults = mults.substitute({x:x.const_like(0) for x in mults.toposort() if x.op is Ops.SPECIAL}) if isinstance(mults, UOp) else mults
      elif u.op is Ops.END: mults = mult_stack.pop(-1)
      elif u.op is Ops.SPECIAL: mults *= cast(sint, u.src[0].ssimplify()) # NOTE: we don't push to the mult_stack here, you can't end these
      elif u.op is Ops.DEFINE_VAR and u.arg[0] == 'core_id': mults *= u.arg[2] + 1
      elif u.op is Ops.LOAD and (not isinstance(u.src[0].dtype, PtrDType) or u.src[0].dtype.addrspace != AddrSpace.REG):
        lds += u.dtype.itemsize * mults
      elif u.op is Ops.STORE and (not isinstance(u.src[0].dtype, PtrDType) or u.src[0].dtype.addrspace != AddrSpace.REG):
        lds += u.src[1].dtype.itemsize * mults
      elif u.op in GroupOp.ALU and u not in dont_count: flops += (mults * (2 if u.op is Ops.MULACC else 1)) * u.dtype.count
      elif u.op is Ops.WMMA and u not in dont_count: flops += 2 * prod(u.arg[1]) // u.arg[5] * mults
    return Estimates(flops, lds, sum(mem.values()))

@dataclass
class ProgramSpec:
  name:str
  src:str
  device:str
  ast:UOp  # save the base ast (this is method cache key)
  uops:list[UOp]|None=None
  lib:bytes|None=None
  aux:list=field(default_factory=list)

  # filled in from uops (via from_uop)
  global_size:list[int]=field(default_factory=lambda: [1,1,1])
  local_size:list[int]|None=None
  vars:list[Variable]=field(default_factory=list)
  globals:list[int]=field(default_factory=list)
  outs:list[int]=field(default_factory=list)
  ins:list[int]=field(default_factory=list)

  @functools.cached_property
  def estimates(self) -> Estimates:
    return Estimates() if self.uops is None else Estimates.from_uops(self.uops, ignore_indexing=True)

  @functools.cached_property
  def function_name(self) -> str: return to_function_name(self.name)

  @functools.cached_property
  def runtimevars(self) -> dict[str, int]: return {v.arg[0]: i for i, v in enumerate(self.vars) if v.arg[0] == 'core_id'}

  @property
  def applied_opts(self) -> tuple[Opt, ...]|None:
    if self.uops is None: return None
    assert self.uops[-1].op is Ops.SINK, self.uops[-1].op
    return self.uops[-1].arg.applied_opts

  def launch_dims(self, var_vals:dict[str, int]):
    global_size = [sym_infer(sz, var_vals) for sz in self.global_size]
    local_size = [sym_infer(sz, var_vals) for sz in self.local_size] if self.local_size is not None else None
    return global_size, local_size

  @staticmethod
  def from_uop(prg:UOp) -> ProgramSpec:
    """Construct ProgramSpec from a PROGRAM UOp."""
    assert prg.op is Ops.PROGRAM, f"expected PROGRAM, got {prg.op}"
    # SINK/DEVICE/LINEAR/SOURCE/BINARY?
    sink, device, linear, source = prg.src[:4]
    lib = prg.src[4].arg if len(prg.src) > 4 else None
    uops = list(linear.src)
    if DEBUG >= 6: print_uops(uops)  # LINEAR is src[2]

    # single pass through the uops to extract metadata
    _vars: list[Variable] = []
    _globals: list[int] = []
    outs: list[int] = []
    ins: list[int] = []
    global_size: list[int] = [1, 1, 1]
    local_size: list[int]|None = [1, 1, 1]
    for u in uops:
      if u.op is Ops.DEFINE_VAR: _vars.append(u)
      if u.op is Ops.PARAM: _globals.append(u.arg)
      if u.op in (Ops.STORE, Ops.LOAD):
        if (idx:=u.src[0]).op is Ops.INDEX or (u.src[0].op is Ops.CAST and (idx:=u.src[0].src[0]).op is Ops.INDEX):
          if (buf:=idx.src[0]).op is Ops.PARAM: (outs if u.op is Ops.STORE else ins).append(buf.arg)
        # TODO: can else happen?
      if u.op is Ops.SPECIAL:
        if u.arg[0] == 'i': local_size = None
        special_size = local_size if u.arg[0] == 'l' else global_size
        # TODO: this cast is wrong, u.src[0].ssimplify() can be sint
        if special_size is not None: special_size[int(u.arg[-1])] = cast(int, u.src[0].ssimplify())
      if u.op is Ops.DEFINE_VAR and u.arg[0] == 'core_id': global_size[0] = u.arg[2] + 1

    return ProgramSpec(sink.arg.name, source.arg, device.arg, sink, uops, lib, list(prg.arg) if prg.arg else [], global_size, local_size,
                       sorted(_vars, key=lambda v: v.arg), sorted(dedup(_globals)), sorted(dedup(outs)), sorted(dedup(ins)))

class Renderer:
  device: str = ""
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
  shared_max: int = 32768
  tensor_cores: list[TensorCore] = []
  pre_matcher: PatternMatcher|None = None
  extra_matcher: PatternMatcher|None = None
  code_for_op: dict[Ops, Callable] = {}
  compiler: Compiler|None = None

  def __reduce__(self): return self.__class__, ()
  def render(self, uops:list[UOp]) -> str: raise NotImplementedError("needs a renderer")
  def aux(self, uops:list[UOp]) -> dict: raise NotImplementedError("needs aux")
