from __future__ import annotations
from typing import Callable, Generator, cast
import subprocess, pathlib, struct
from dataclasses import dataclass, replace
from tinygrad.helpers import prod, Target, EMULATED_DTYPES, CCACHE, getenv, diskcache_get, diskcache_put, unwrap, round_up
from tinygrad.uop.ops import Ops, UOp, sint, ssimplify, smin, GroupOp, PatternMatcher
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.renderer.tc import TensorCore

class CompileError(Exception): pass

class Compiler:
  def __init__(self, cachekey:str|None=None): self.cachekey = cachekey if CCACHE else None
  def compile(self, src:str) -> bytes: return src.encode()   # NOTE: empty compiler is the default
  def compile_cached(self, src:str) -> bytes:
    if self.cachekey is None or (lib := diskcache_get(self.cachekey, src)) is None:
      assert not getenv("ASSERT_COMPILE"), f"tried to compile with ASSERT_COMPILE set\n{src}"
      lib = self.compile(src)
      if self.cachekey is not None: diskcache_put(self.cachekey, src, lib)
    return lib
  def disassemble(self, lib:bytes): pass
  def server(self, cmd:str, arch:str, *args) -> subprocess.Popen:
    argv = f"{cmd} {pathlib.Path(__file__).parent.parent}/runtime/support/compileserver.py {type(self).__module__}:{type(self).__name__} {arch}"
    return subprocess.Popen(argv.split() + [str(a) for a in args], stdout=subprocess.PIPE, stdin=subprocess.PIPE, bufsize=0)
  def compile_server(self, src:str, proc:subprocess.Popen) -> bytes:
    unwrap(proc.stdin).write(struct.pack("I", len(src.encode())) + src.encode())
    if (lib:=unwrap(proc.stdout).read(struct.unpack("I", unwrap(proc.stdout).read(4))[0])): return lib
    raise CompileError("Compilation Error")


@dataclass
class TinyELF:
  lib: bytes
  name: str
  target: Target
  # tuple of (name, addrspace, dtype, shape), one per arg in the kernel's order. an ALU arg is a value, the rest are buffers
  signature: tuple[tuple[str|None, AddrSpace, DType, tuple], ...]
  profile_key: bytes|None = None

  @staticmethod
  def from_program(prg:UOp) -> TinyELF:
    sig = tuple((p.arg.name, p.arg.addrspace, p.dtype, p._shape) for p in prg.kernel_params)
    return TinyELF(prg.src[3].arg, prg.src[0].arg.function_name, prg.arg.target, sig, prg.key)

  @staticmethod
  def iter_sig(signature:tuple[tuple[str|None, AddrSpace, DType, tuple], ...], offset:int=0) -> Generator[tuple[int, DType], None, None]:
    for _,a,dt,_ in signature:
      if a is not AddrSpace.ALU: dt = dtypes.uint64 # a buffer is passed by its address
      yield (offset:=round_up(offset, dt.itemsize)), dt
      offset += dt.itemsize

# an access takes its dtype from the buffer it indexes, so accessing at another dtype restates the storage on the buffer that owns it
def with_storage(x:UOp, dt:DType) -> UOp:
  if x.op in GroupOp.Defines: return x.replace(arg=replace(x.arg, dtype=dt))
  return x.replace(src=(with_storage(x.src[0], dt),)+x.src[1:])

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
    # ALUs used for indexing don't count as flops. the toposort stops at END/BACKEDGE so loop bodies still count
    idxs = [s for u in uops if u.op in {Ops.INDEX, Ops.SHRINK} for s in u.src[1:]] if ignore_indexing else []
    excluded = UOp.sink(*idxs).toposort(lambda x: x.op not in {Ops.END, Ops.BACKEDGE})
    for u in uops:
      if u.op in {Ops.LOAD, Ops.STORE}:
        buf = u
        while len(buf.src) and buf.op not in GroupOp.Defines: buf = buf.src[0]
        if buf.op is Ops.PARAM:
          # u.src[0] is INDEX, cap at buffer size for re-reads (e.g. matmul)
          accessed = mem.get((buf, u.op), 0) + u.src[0].max_numel() * u.src[0].dtype.itemsize * mults
          mem[(buf, u.op)] = smin(accessed, buf.max_numel() * buf.dtype.itemsize)
      if u.op is Ops.RANGE:
        mult_stack.append(mults)
        if u.dtype is not dtypes.void:  # unbounded loop, unknown trip count
          bound = u.src[0].without_after.simplify()
          mults *= bound.vmax if bound.ranges else bound.ssimplify()
          # SPECIAL are already counted in mults
          mults = mults.substitute({x:x.const_like(0) for x in mults.toposort() if x.op is Ops.SPECIAL}) if isinstance(mults, UOp) else mults
      elif u.op in {Ops.END, Ops.BACKEDGE}: mults = mult_stack.pop(-1)
      elif u.op is Ops.SPECIAL: mults *= cast(sint, u.src[0].ssimplify()) # NOTE: we don't push to the mult_stack here, you can't end these
      elif u.op is Ops.LOAD and u.src[0].addrspace != AddrSpace.REG:
        lds += u.max_numel() * u.dtype.itemsize * mults
      elif u.op is Ops.STORE and u.src[0].addrspace != AddrSpace.REG:
        lds += u.max_numel() * u.src[1].dtype.itemsize * mults
      elif u.op in GroupOp.ALU and u not in excluded:
        flops += (mults * (2 if u.op is Ops.MULACC else 1)) * u.max_numel()
      elif u.op is Ops.WMMA and u not in excluded:
        flops += 2 * prod(u.arg[0]) // u.arg[2] * mults
    return Estimates(ssimplify(flops), lds, sum(mem.values()))

class Renderer:
  target: Target
  suffix: str = ""
  inline_prefix: str|None = None
  # TODO: make this generic with a list of supported types
  supports_float4: bool = True
  has_local: bool = True
  has_shared: bool = True
  # NOTE: these two should be in (x,y,z) order to match the max_sizes argument in get_grouped_dims
  global_max: tuple[int, ...]|None = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  local_max: tuple[int, ...]|None = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  global_prod_max: tuple[int, ...]|None = None
  shared_max: int = 32768
  tensor_cores: list[TensorCore] = []
  extra_matcher: PatternMatcher|None = None
  packed_args: bool = False
  code_for_op: dict[Ops, Callable] = {}

  compiler: Compiler = Compiler()

  def __init__(self, target:Target): self.target = target
  def __reduce__(self): return self.__class__, (self.target,)
  def render(self, uops:list[UOp]) -> str: raise NotImplementedError("needs a renderer")
  def asm(self, prg:UOp, lin:UOp) -> bytes: raise NotImplementedError("needs an assembler")
  def supported_dtypes(self) -> set[DType]:
    # double can't be bitcast to anything without long support
    return set(dtypes.all) - ({dtypes.double} if dtypes.long in EMULATED_DTYPES.tolist(dtypes) else set())
