from typing import Optional, List, Tuple, Dict, Callable, Any
import functools
from dataclasses import dataclass, field
from tinygrad.helpers import to_function_name, dedup, prod
from tinygrad.ops import Ops, UOp, flops_mem, sym_infer, sint, Variable
from tinygrad.dtype import DType

@dataclass(frozen=True)
class TensorCore: # D = A * B + C, A is (M x K), B is (K x N), C and D are (M x N)
  dims: Tuple[int,int,int] # N, M, K
  dtype_in: DType # dtype for A and B
  dtype_out: DType # dtype for C and D
  threads: List[Tuple[int,int]] # list of (TC dim,amt) that construct the warp thread structure
  reduce_axes: List[Tuple[int,int]] # list of (TC dim,amt) that constructs the shape of the reduce dim
  @property
  def early_upcast_axes(self) -> List[Tuple[int,int]]: # list of (TC dim,amt) that upcasts the threads remainders of dims [0,1]
    return [(d,self.dims[d]//sz) for d,sz in [(dim,prod(sz for d,sz in self.threads if d==dim)) for dim in range(2)] if self.dims[d]>sz]
  upcast_axes: Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]]] # list of (TC dim,amt) that upcast A, B and C
  st1_pattern: Optional[Tuple[Tuple[Tuple[int,int], ...], Tuple[Tuple[int,int], ...]]] = None # pattern to fix shapetracker for A
  st2_pattern: Optional[Tuple[Tuple[Tuple[int,int], ...], Tuple[Tuple[int,int], ...]]] = None # pattern to fix shapetracker for B
  expanded_shape: Optional[Tuple[int, ...]] = None
  opts_seq: Tuple[str,str] = ("UP","LC") # upcast input, local the thread pattern
  def __str__(self): return "_".join(["WMMA"] + list(map(str, self.dims)) + [self.dtype_in.name, self.dtype_out.name])

@dataclass
class ProgramSpec:
  name:str
  src:str
  device:str
  uops:Optional[List[UOp]]=None
  mem_estimate:sint=0  # TODO: get this from the load/store uops once min/max are good

  # filled in from uops (if we have uops)
  global_size:Optional[List[int]]=None
  local_size:Optional[List[int]]=None
  vars:List[Variable]=field(default_factory=list)
  globals:List[int]=field(default_factory=list)
  outs:List[int]=field(default_factory=list)
  _ran_post_init:bool=False  # NOTE: this is needed if you call replace on the Program

  def __post_init__(self):
    if not self._ran_post_init and self.uops is not None:
      # single pass through the uops
      for u in self.uops:
        if u.op is Ops.DEFINE_VAR: self.vars.append(u)
        if u.op is Ops.DEFINE_GLOBAL: self.globals.append(u.arg)
        if u.op is Ops.STORE: self.outs.extend([x.arg for x in u.src[0].toposort if x.op is Ops.DEFINE_GLOBAL])
        if u.op is Ops.SPECIAL:
          # NOTE: you have to set local_size and global_size to the base [1,1,1] outside this
          if u.arg[0][0] == 'i': self.local_size = None
          special_size = self.local_size if u.arg[0][0] == 'l' else self.global_size
          assert special_size is not None
          special_size[int(u.arg[0][-1])] = u.arg[1]
      self.vars = sorted(self.vars, key=lambda v: v.arg)
      self.outs = sorted(dedup(self.outs))
      self._ran_post_init = True

  @property
  def op_estimate(self) -> sint: return self._ops_lds[0]
  @property
  def lds_estimate(self) -> sint: return self._ops_lds[1]
  @functools.cached_property
  def _ops_lds(self) -> Tuple[sint, sint]: return (0,0) if self.uops is None else flops_mem(self.uops, ignore_indexing=True)

  @functools.cached_property
  def function_name(self) -> str: return to_function_name(self.name)

  def launch_dims(self, var_vals:Dict[Variable, int]):
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
  global_max: Optional[Tuple[int, ...]] = (0x8FFFFFFF,) * (3) # TODO: UOps.SPECIAL int32 indexes right now
  local_max: Optional[Tuple[int, ...]] = (0x8FFFFFFF,) * (3) # TODO: UOps.SPECIAL int32 indexes right now
  shared_max: int = 32768
  tensor_cores: List[TensorCore] = []
  extra_matcher: Any = None
  code_for_op: Dict[Ops, Callable] = {}

  def __reduce__(self): return self.__class__, ()
  def render(self, name:str, uops:List[UOp]) -> str: raise NotImplementedError("needs a renderer")
