from __future__ import annotations
from typing import Optional, Callable, cast
import functools, math
from enum import Enum, auto
from dataclasses import dataclass, field, replace
from tinygrad.helpers import to_function_name, dedup, prod
from tinygrad.ops import Ops, UOp, sym_infer, sint, Variable, ssimplify, GroupOp, PatternMatcher
from tinygrad.dtype import DType

class OptOps(Enum):
  TC = auto(); UPCAST = auto(); UNROLL = auto(); LOCAL = auto() # noqa: E702
  GROUP = auto(); GROUPTOP = auto(); NOLOCALS = auto(); PADTO = auto(); SWAP = auto() # noqa: E702
  def __lt__(self, x:OptOps): return self.value < x.value

@dataclass(frozen=True, order=True)
class Opt:
  op: OptOps
  axis: Optional[int] = None
  arg: Optional[int | tuple] = None
  def __repr__(self): return f"Opt(op={self.op}, axis={self.axis}, arg={self.arg})"

@dataclass(frozen=True)
class TensorCore: # D = A * B + C, A is (M x K), B is (K x N), C and D are (M x N)
  dims: tuple[int,int,int] # N, M, K
  threads: int # number of threads that construct the warp
  elements_per_thread: tuple[int, int, int] # elements per-thread to load/store from A/B/C
  dtype_in: DType # dtype for A and B
  dtype_out: DType # dtype for C and D
  opts: tuple[str, ...] # ordered tuple of "ux" or "lx" specifing kernel opts to perform. "ux" upcasts dim x and "lx" localizes dim x
  swizzle: tuple[Optional[tuple[tuple[int, ...], tuple[int, ...]]], Optional[tuple[tuple[int, ...], tuple[int, ...]]]] = (None, None)
  def get_reduce_axes(self): return [(i, 2) for i in range(int(math.log2(self.dims[2])))]
  def get_upcast_axes(self): return [opt for opt in self.opts if opt[0] == "u"]
  def get_local_axes(self): return [opt for opt in self.opts if opt[0] == "l"]
  def __str__(self): return "_".join(["WMMA"] + list(map(str, self.dims)) + [self.dtype_in.name, self.dtype_out.name])
  def __post_init__(self):
    local_axes, upcast_axes, reduce_axes = len(self.get_local_axes()), len(self.get_upcast_axes()), len(self.get_reduce_axes())
    assert self.dims[0] * self.dims[1] == 2**(local_axes + upcast_axes), (
      f"N({self.dims[0]}) x M({self.dims[1]}) != local({2**local_axes}) x upcast({2**upcast_axes}) with opts({self.opts})")
    assert 2**local_axes == self.threads, f"{self.threads} threads construct the warp but found {2**local_axes} in {self.opts}"
    assert 2**upcast_axes == self.elements_per_thread[2], (
      f"{self.elements_per_thread[2]} elements from C are processed per thread but found {2**upcast_axes} in {self.opts}")
    assert all(len(perm[0]) == local_axes and len(perm[1]) == reduce_axes + upcast_axes for perm in self.swizzle if perm), (
      f"swizzle perm should be of len (({local_axes})({reduce_axes + upcast_axes}))")

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
        if u.op in {Ops.LOAD, Ops.STORE}:
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
      elif u.op is Ops.LOAD: lds += u.dtype.itemsize * mults
      elif u.op is Ops.STORE: lds += u.src[1].dtype.itemsize * mults
      elif u.op in GroupOp.ALU and u not in dont_count: flops += (mults * (2 if u.op is Ops.MULACC else 1)) * u.dtype.count
      elif u.op is Ops.WMMA and u not in dont_count: flops += 2 * prod(u.arg[1]) // u.arg[5] * mults
    return Estimates(flops, lds, lds) # TODO: properly track memory, lds is always a high estimate

@dataclass
class ProgramSpec:
  name:str
  src:str
  device:str
  ast:UOp  # save the base ast (this is method cache key)
  uops:Optional[list[UOp]]=None
  applied_opts:Optional[list[Opt]]=None
  mem_estimate:sint=0  # TODO: get this from the load/store uops once min/max are good

  # filled in from uops (if we have uops)
  global_size:Optional[list[int]]=None
  local_size:Optional[list[int]]=None
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
  def estimates(self) -> Estimates:
    return replace(Estimates() if self.uops is None else Estimates.from_uops(self.uops, ignore_indexing=True), mem=self.mem_estimate)

  @functools.cached_property
  def function_name(self) -> str: return to_function_name(self.name)

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
  global_max: Optional[tuple[int, ...]] = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  local_max: Optional[tuple[int, ...]] = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  shared_max: int = 32768
  tensor_cores: list[TensorCore] = []
  pre_matcher: Optional[PatternMatcher] = None
  extra_matcher: Optional[PatternMatcher] = None
  code_for_op: dict[Ops, Callable] = {}

  def __reduce__(self): return self.__class__, ()
  def render(self, uops:list[UOp]) -> str: raise NotImplementedError("needs a renderer")
