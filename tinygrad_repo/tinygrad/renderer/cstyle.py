from typing import Literal, Callable
import math, sys, struct
from collections import defaultdict, Counter
from tinygrad.codegen.opt import tc
from tinygrad.uop.ops import GroupOp, Ops, UOp, PatternMatcher, UPat, range_str, axis_letters
from tinygrad.helpers import strip_parens, getenv, prod, dedup, Target, CPU_COUNT, IMAGE, FLOAT16, is_image_shape
from tinygrad.dtype import dtypes, DType, AddrSpace, truncate, float_to_bf16
from tinygrad.renderer import Renderer


base_rewrite = PatternMatcher([
  # local/reg buffers
  (UPat(Ops.BUFFER, name="x"), lambda ctx,x: ctx.render_buffer(x)),

  # range/loop/if/endif
  (UPat(Ops.RANGE, name="x"),
   lambda ctx,x: f"for ({ctx.render_dtype(x.dtype)} {ctx[x]} = 0; {ctx[x]} < {ctx[x.src[0]]}; {ctx[x]}++) {{"),
  (UPat(Ops.LOOP, name="x"), lambda ctx,x: "for (;;) {"),
  (UPat(Ops.END, src=(UPat(), UPat(Ops.LOOP), UPat(name="c", dtype=dtypes.bool))), lambda ctx,c: f"  if (!({ctx[c]})) {{ break; }}\n}}"),
  (UPat(Ops.IF, name="x"), lambda ctx,x: f"if ({ctx[x.src[0]]}) {{"),
  (UPat((Ops.ENDIF, Ops.END)), lambda ctx: "}"),

  # casting
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"__builtin_convertvector({ctx[x.src[0]]}, {ctx.render_type(x)})" \
    if x.max_numel() > 1 and x.addrspace is AddrSpace.REG else None),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"({ctx.render_cast(x, ctx[x.src[0]])})"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: ctx[x.src[0]] if x.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL) else None),
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"__builtin_bit_cast({ctx.render_type(x)}, ({ctx.render_type(x.src[0])})({ctx[x.src[0]]}))"),

  # GPU stuff
  (UPat(Ops.BARRIER), lambda ctx: ctx.barrier),
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: f"{ctx.code_for_workitem[x.arg[0]](x.arg[-1])}; /* {(x.src[0]).render()} */"),

  # const
  (UPat(Ops.CONST, arg=math.inf, name="x"), lambda ctx, x: f"({ctx.render_cast(x, ctx.infinity)})"),
  (UPat(Ops.CONST, arg=-math.inf, name="x"), lambda ctx, x: f"({ctx.render_cast(x, f'-{ctx.infinity}')})"),
  (UPat(Ops.CONST, dtype=dtypes.floats, name="x"), lambda ctx,x: f"({ctx.render_cast(x, ctx.nan)})" if math.isnan(x.arg) else None),
  (UPat(Ops.CONST, dtype=dtypes.float, name="x"), lambda ctx,x: f"{x.arg}f"),
  (UPat(Ops.CONST, dtype=dtypes.int64, name="x"), lambda ctx,x: f"{x.arg}l"),
  (UPat(Ops.CONST, dtype=dtypes.uint64, name="x"), lambda ctx,x: f"{truncate[x.dtype](x.arg)}ul"),
  (UPat(Ops.CONST, dtype=dtypes.uint32, name="x"), lambda ctx,x: f"{truncate[x.dtype](x.arg)}u"),
  (UPat(Ops.CONST, dtype=dtypes.bool, name="x"), lambda ctx,x: "1" if x.arg else "0"),
  # consts are rendered to larger type and casted
  (UPat(Ops.CONST, (*dtypes.fp8s, dtypes.bfloat16, dtypes.half), name="x"), lambda ctx,x: f"({ctx.render_cast(x, f'{x.arg}f')})"),
  (UPat(Ops.CONST, (dtypes.uint8, dtypes.uint16), name="x"), lambda ctx,x: f"({ctx.render_cast(x, f'{x.arg}u')})"),
  (UPat(Ops.CONST, (dtypes.int8, dtypes.int16), name="x"), lambda ctx,x: f"({ctx.render_cast(x, str(x.arg))})"),
  # default const render
  (UPat(Ops.CONST, name="x"), lambda ctx,x: str(x.arg)),

  # SHRINK/INDEX
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var('idx')), name="x"), lambda ctx,**kwargs: ctx.render_index(**kwargs)),
  (UPat(Ops.SHRINK, src=(UPat.var("buf"), UPat.var('idx'), UPat.cvar()), name="x"), lambda ctx,**kwargs: ctx.render_index(**kwargs)),
  (UPat(Ops.STACK, name="x"),
   lambda ctx,x: f"{ctx.float4.replace('float4', ctx.render_type(x))}" + \
                 f"{ctx.float4_style[0]}{','.join([ctx[y] for y in x.src])}{ctx.float4_style[1]}"),

  # load/store
  (UPat(Ops.LOAD, src=(UPat.var('bidx'),)), lambda ctx,bidx: f"({ctx.render_access(bidx)})"),
  (UPat(Ops.LOAD, src=(UPat.var("bidx"), UPat.var("var"), UPat.var("gate"))),
   lambda ctx,bidx,var,gate: f"({ctx[gate]}?{ctx.render_access(bidx)}:{ctx[var]})"),
  (UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var"))), lambda ctx,bidx,var: f"{ctx.render_access(bidx)} = {ctx[var]};"),

  # alu/gep
  (UPat(Ops.WMMA, name="x"), lambda ctx,x: f"__{_wmma_name(x)}({ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]})"),
  (UPat(GroupOp.ALU, name="x"), lambda ctx,x: ctx.code_for_op[x.op](
    *([strip_parens(ctx[v]) if v.op == x.op and x.op in {Ops.ADD, Ops.MUL, Ops.XOR, Ops.OR, Ops.AND} else ctx[v] for v in x.src]), x.dtype)),

  # custom passes through with format
  (UPat((Ops.CUSTOM, Ops.CUSTOMI), name="x"), lambda ctx,x: x.arg.format(*[ctx[y] for y in x.src])),
])

def create_non_native_float_pats(dts:tuple[DType, ...], casting:bool=True):
  patterns = PatternMatcher([
    (UPat(Ops.WHERE, src=(UPat.var("b"), UPat.var("x", dtype=dts), UPat.var("y", dtype=dts))),
     lambda b,x,y: UOp(Ops.WHERE, src=(b,x.cast(dtypes.float),y.cast(dtypes.float))).cast(x.dtype)),
    (UPat(GroupOp.ALU, dtype=dts, name="x"),
     lambda x: UOp(x.op, src=tuple(vv.cast(dtypes.float) for vv in x.src), arg=x.arg).cast(x.dtype)),
    (UPat(GroupOp.ALU, dtypes.bool, name="alu", src=(UPat.var("x", dtype=dts), UPat.var("y", dtype=dts))),
     lambda alu,x,y: UOp(alu.op, src=(x.cast(dtypes.float), y.cast(dtypes.float)), arg=alu.arg))])
  if casting:
    # add float intermediate casting
    patterns += PatternMatcher([
      (UPat(Ops.CAST, dts, (UPat.var("x"),), name="y"), lambda x,y: x.cast(dtypes.float).cast(y.dtype) if x.dtype!=dtypes.float else None),
      (UPat(Ops.CAST, name="x", src=(UPat.var("y", dts),)), lambda x,y: y.cast(dtypes.float).cast(x.dtype) if x.dtype!=dtypes.float else None)])
  return patterns

def cast_float_to_bf16(x: UOp) -> UOp:
  assert x.dtype == dtypes.float, "cast float -> bf16 must start with float"
  x = x.bitcast(dtypes.uint)
  x = (-x & 0x7f800000).ne(0).where(x + ((x >> 16) & 1) + 0x7fff, (x & 0xffff).ne(0).where((x | 0x10000), x))
  return (x >> 16).cast(dtypes.ushort).bitcast(dtypes.bfloat16)

# manual bfloat16 casting patterns (shared between LLVM, Clang, and AMD renderers to avoid compiler intrinsics)
pm_manual_bf16_cast = PatternMatcher([
  (UPat(Ops.CAST, dtypes.float, (UPat.var("x", dtypes.bfloat16),)),
   lambda x: (x.bitcast(dtypes.ushort).cast(dtypes.uint)<<16).bitcast(dtypes.float)),
  (UPat(Ops.CAST, dtype=dtypes.bfloat16, src=(UPat.var("x", dtype=dtypes.float),)), cast_float_to_bf16),
])

def uops_to_dtypes(uops:list[UOp]) -> list[tuple[DType, int]]:
  return dedup((u.dtype, u.max_numel()) for u in uops if u.addrspace in (AddrSpace.ALU, None) and u.dtype != dtypes.void and u._shape is not None)

def _wmma_name(u:UOp) -> str:
  # sanitize spaces in DType.name (int8 = "signed char")
  return f"WMMA_{'_'.join(map(str, u.arg[0]))}_{u.arg[1].name}_{u.dtype.scalar().name}".replace(" ", "_")

# (name, dims, dtype_in, dtype_out, device, threads, upcast_sizes)
def wmma_args(uops:list[UOp]):
  return dedup((_wmma_name(uop), uop.arg[0], uop.arg[1], uop.dtype.scalar(), *(uop.arg[2:4]),
               tuple(uop.src[i].shape[-1] for i in range(3)))
              for uop in uops if uop.op is Ops.WMMA)

class CStyleLanguage(Renderer):
  kernel_typedef: str = "void"
  buffer_prefix: str = ""
  buffer_suffix: str = ""
  smem_align: str = ""
  smem_prefix: str = ""
  smem_prefix_for_cast: bool = True
  arg_int_prefix: str = "const int"
  barrier: str = ""
  code_for_workitem: dict[Literal["g", "l", "i"], Callable] = {}
  extra_args: list[str] = []
  float4: str|None = None
  float4_style: tuple[str, str] = ('(', ')')
  gep_arr_threshold: int = 4
  type_map: dict[DType, str] = {}
  infinity: str = "INFINITY"
  nan: str = "NAN"
  code_for_op: dict = {
    Ops.SQRT: lambda x,dtype: f"sqrt({x})", Ops.RECIPROCAL: lambda x,dtype: f"(1/{x})", Ops.NEG: lambda x,dtype: f"-{x}",
    Ops.EXP2: lambda x,dtype: f"exp2({x})", Ops.LOG2: lambda x,dtype: f"log2({x})", Ops.SIN: lambda x,dtype: f"sin({x})",
    Ops.TRUNC: lambda x,dtype: f"trunc({x})",
    Ops.AND: lambda a,b,dtype: f"({a}&{b})", Ops.XOR: lambda a,b,dtype: f"({a}^{b})", Ops.OR: lambda a,b,dtype: f"({a}|{b})",
    Ops.ADD: lambda a,b,dtype: f"({a}+{b})", Ops.SUB: lambda a,b,dtype: f"({a}-{b})", Ops.MUL: lambda a,b,dtype: f"({a}*{b})",
    Ops.CMOD: lambda a,b,dtype: f"({a}%{b})", Ops.CDIV: lambda a,b,dtype: f"({a}/{b})", Ops.CMPNE: lambda a,b,dtype: f"({a}!={b})",
    Ops.SHR: lambda a,b,dtype: f"({a}>>{b})", Ops.SHL: lambda a,b,dtype: f"({a}<<{b})", Ops.CMPLT: lambda a,b,dtype: f"({a}<{b})",
    Ops.WHERE: lambda a,b,c,dtype: f"({a}?{b}:{c})", Ops.CMPEQ: lambda a,b,dtype: f"({a}=={b})"}

  string_rewrite = base_rewrite

  def render_kernel(self, function_name:str, kernel:list[str], bufs:list[tuple[str,tuple[UOp,bool]]], uops:list[UOp], prefix=None) -> str:
    tmp = ""
    if any(is_image_shape(u._shape) for _,(u,_) in bufs):
      tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
    buftypes = [(name, self._render_dtype(u.dtype, sz=1, addrspace=u.addrspace, mutable=mutable, shape=u._shape)+self.buffer_suffix \
                 if u.addrspace == AddrSpace.GLOBAL else self.arg_int_prefix if u.dtype == dtypes.int else None) for name,(u,mutable) in bufs]
    local_dims = [u.src[0] for u in uops if u.op is Ops.SPECIAL and u.arg[0] == "l"]
    launch_bounds = prod([d.vmax for d in local_dims])
    prg = ''.join([f"{self.kernel_typedef.format(launch_bounds=launch_bounds)} {function_name}(",] +
    [', '.join([f'{t} {name}' for name,t in buftypes] + self.extra_args)] +
    [") {\n" + tmp] + ['\n'.join(kernel), "\n}"])
    return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"

  def render_index(self, x:UOp, buf:UOp, idx:UOp):
    if buf.addrspace == AddrSpace.ALU:
      # this is lane access in C
      if idx.op is not Ops.CONST: return f"({self[buf]})[{self[idx]}]"
      return self[buf]+(f"[{idx.arg}]" if buf.max_numel() > self.gep_arr_threshold else f".{'xyzwabcd'[idx.arg]}")
    return f"({self[buf]}+{strip_parens(self[idx]) if idx.arg == Ops.ADD else self[idx]})"

  def render_buffer(self, x:UOp):
    lanes = 1
    prefix = f"{self.smem_align}{self.smem_prefix}" if x.addrspace == AddrSpace.LOCAL else ""
    suffix = f"[{x.max_numel()}]"
    return f"{prefix}{self._render_dtype(x.dtype, sz=lanes)} {self[x]}{suffix};"

  def _render_dtype(self, dtype:DType, sz:int=1, addrspace=AddrSpace.ALU, mutable=True, override_ptr=False, shape=None):
    if is_image_shape(shape): return f"{'write_only' if mutable else 'read_only'} image2d_t"
    prefix, suffix = "", ""
    if addrspace in (AddrSpace.LOCAL, AddrSpace.GLOBAL):
      if addrspace == AddrSpace.LOCAL and self.smem_prefix_for_cast: prefix = self.smem_prefix
      if addrspace == AddrSpace.GLOBAL: prefix = self.buffer_prefix
    if addrspace in (AddrSpace.LOCAL, AddrSpace.GLOBAL) or override_ptr:
      suffix = "*"
    if sz > 1:
      return prefix + self.type_map.get(scalar:=dtype.scalar(), scalar.name).replace(" ", "_") + str(sz) + suffix
    return prefix + self.type_map.get(scalar:=dtype.scalar(), scalar.name) + suffix

  def render_type(self, u:UOp): return self._render_dtype(u.dtype, u.max_numel(), u.addrspace, shape=u._shape)
  def render_access(self, u:UOp):
    if u.max_numel() > 1 or u.dtype != u.src[0].dtype:
      return f"*(({self._render_dtype(u.dtype, u.max_numel(), u.addrspace, override_ptr=True, shape=u._shape)})({self[u]}))"
    else: return f"*{self[u]}"
  def render_cast(self, u:UOp, val:str) -> str: return f"({self.render_type(u)})({val})"

  # LEGACY
  def render_dtype(self, dt:DType, mutable=True) -> str:
    return self._render_dtype(dt, 1, AddrSpace.REG)

  def __getitem__(self, key): return self.r[key]  # hacky helper
  def _render(self, uops:list[UOp]) -> tuple[str, list[str], list[tuple[str,tuple[UOp,bool]]]]:
    r: dict[UOp, str] = {}
    self.r = r

    child_count = Counter(v for ru in uops for v in ru.src)
    # find which PARAMs are stored to with a single toposort
    writable_params = {u for u in UOp.sink(*[u.src[0] for u in uops if u.op is Ops.STORE]).toposort(lambda u: u.op != Ops.END) if u.op is Ops.PARAM}
    bufs: dict[UOp, tuple[str, tuple[UOp, bool]]] = {}
    kernel = []
    depth = 1
    c: defaultdict[str, int] = defaultdict(int)
    name = "test"
    for u in uops:
      if u.op in {Ops.NOOP, Ops.GROUP}: continue
      if u.op == Ops.STACK and len(u.src) == 0: continue
      if u.op is Ops.AFTER:
        r[u] = r[u.src[0]]
        continue
      if u.op is Ops.SINK:
        if u.arg is not None: name = u.arg.function_name
        continue
      if u.op is Ops.PARAM:
        r[u] = f"data{u.arg.slot}_" + '_'.join([str(x) for x in u.shape])
        bufs[u] = (r[u], (u, u in writable_params))
        continue

      # naming
      prefix = None
      if u.op is Ops.SPECIAL: r[u] = u.arg
      elif u.op is Ops.RANGE: r[u] = f"{axis_letters[u.arg[-1]]}idx"+range_str(u)
      else:
        prefix = {Ops.WMMA: "wmma", Ops.CONST: "const", Ops.BUFFER: "buf", Ops.CAST: "cast", Ops.BITCAST: "cast", Ops.STACK: "cast",
                  Ops.INDEX: "bidx", Ops.LOAD: "val"}.get(u.op, "alu")
        r[u] = f"{prefix}{c[prefix]}"

      l: str|None = self.string_rewrite.rewrite(u, ctx=self)
      assert l is not None, f"failed to render {u.op} {u.dtype} {[(x.op,x.dtype) for x in u.src]} {u.arg}"

      if u.op in {Ops.ENDIF, Ops.END}: depth -= 1
      if (u.op is not Ops.CAST or u.max_numel() == 1) and (u.op in {Ops.CONST, Ops.INDEX, Ops.SHRINK, Ops.CUSTOMI} or \
        (u.op is Ops.LOAD and u.src[0].addrspace == AddrSpace.REG and child_count[u] == 1) or \
        (u.op is Ops.CAST and u.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL)) or \
        (u.op in {Ops.STACK, *(GroupOp.ALU-{Ops.WHERE}), Ops.CAST, Ops.BITCAST} and child_count[u] == 1 and not getenv("EXPAND_SSA"))):
        r[u] = l
      else:
        if u.op not in {Ops.RANGE, Ops.STORE, Ops.BUFFER} and u.dtype != dtypes.void:
          l = f"{self.render_type(u)} {r[u]} = {l}" + (";" if u.op is not Ops.SPECIAL else "")
        kernel.append("\n".join("  "*depth + line for line in l.split("\n")))
        if prefix: c[prefix] += 1  # if it was used, increment
      if u.op in {Ops.IF, Ops.RANGE, Ops.LOOP}: depth += 1
    del self.r

    # NOTE: this relies on bufs dict preserving order
    return (name, kernel, list(bufs.values()))
  def render(self, uops:list[UOp]) -> str: return self.render_kernel(*self._render(uops), uops)

class ClangRenderer(CStyleLanguage):
  float4 = "(float4)"
  float4_style = ('{', '}')
  gep_arr_threshold = 0
  has_local = False
  has_threads = bool(getenv("THREADS", 1))
  global_max = (CPU_COUNT.value, 0, 0)
  infinity = "__builtin_inff()"
  nan = '__builtin_nanf("")'

  # language options
  buffer_suffix = " restrict"
  type_map = {dtypes.bool:"_Bool", dtypes.half:"__fp16"}
  code_for_op = {**({k:v for k,v in CStyleLanguage.code_for_op.items() if k not in [Ops.EXP2, Ops.SIN, Ops.LOG2, Ops.TRUNC, Ops.RECIPROCAL]}),
                 Ops.SQRT: lambda x,dtype: f"__builtin_sqrt({x})" if dtype == dtypes.float64 else f"__builtin_sqrtf({x})",
                 Ops.TRUNC: lambda x,dtype: f"__builtin_trunc({x})" if dtype == dtypes.float64 else f"__builtin_truncf({x})",
                 Ops.FDIV: lambda a,b,dtype: f"({a}/{b})"}

  # LLVM legalizes double => half/bf16 cast on systems that don't support it natively (like x86 cpus without AVX512-FP16) into a compiler-rt libcall.
  # there is also no native bfl16 <-> fp16 conversion on those CPUs
  extra_matcher = PatternMatcher([(UPat.var("x", dtypes.float64).cast(dtypes.float16), lambda x: x.cast(dtypes.float32).cast(dtypes.float16)),
                                 (UPat.var("x", dtypes.float64).cast(dtypes.bfloat16), lambda x: x.cast(dtypes.float32).cast(dtypes.bfloat16)),
                                 (UPat.var("x", dtypes.bfloat16).cast(dtypes.float16), lambda x: x.cast(dtypes.float32).cast(dtypes.float16))]) \
    + create_non_native_float_pats((dtypes.bfloat16,)) + pm_manual_bf16_cast

  if sys.platform == 'win32':
    kernel_typedef = "__attribute__((ms_abi)) void"
  def render_vector_prefix(self, dt:DType, count:int) -> str:
    # round (down) to power of two (this is actually the default clang behavior)
    alignment = 2**int(math.log2(dt.itemsize * count)) if getenv("ALIGNED", 1) and not dtypes.is_bool(dt) else 1
    vec = self._render_dtype(dt, count, AddrSpace.REG)
    return f"typedef {self.render_dtype(dt)} {vec} __attribute__((aligned({alignment}),ext_vector_type({count})));"

  def _render_defines(self, uops) -> list[str]: return [self.render_vector_prefix(dt, count) for dt, count in uops_to_dtypes(uops) if count > 1]
  def _render_body(self, function_name, kernel, bufs, uops, pref=None) -> str: return super().render_kernel(function_name, kernel, bufs, uops, pref)
  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[UOp,bool]]]) -> str: return ""

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    defines = '\n'.join(self._render_defines(uops))
    return defines + "\n" + self._render_body(function_name, kernel, bufs, uops, prefix) + "\n" + self._render_entry(function_name, bufs)

  def supported_dtypes(self):
    return {d for d in super().supported_dtypes() if (d != dtypes.bfloat16 or self.target.arch.startswith(("x86", "arm"))) and d not in dtypes.fp8s}

  def __init__(self, target:Target):
    super().__init__(target)
    from tinygrad.runtime.support.compiler_cpu import ClangCompiler
    self.compiler = ClangCompiler(target.arch.split(","))

class OpenCLRenderer(CStyleLanguage):
  has_aux = True

  # language options
  kernel_typedef = "__kernel void"
  buffer_prefix = "__global "
  smem_align = "__attribute__ ((aligned (16))) "
  smem_prefix = "__local "
  barrier = "barrier(CLK_LOCAL_MEM_FENCE);"
  float4 = "(float4)"
  code_for_workitem = {"g": lambda x: f"get_group_id({x})", "l": lambda x: f"get_local_id({x})", "i": lambda x: f"get_global_id({x})"}
  type_map = { dtypes.int8: "char", dtypes.uint8: "uchar", dtypes.uint32: "uint", dtypes.uint16: "ushort", dtypes.uint64: "ulong",
              dtypes.bfloat16: "ushort" }
  extra_matcher = create_non_native_float_pats((dtypes.bfloat16,)) + pm_manual_bf16_cast

  string_rewrite = PatternMatcher([
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"as_{ctx.render_dtype(x.dtype)}(({ctx.render_dtype(x.src[0].dtype)})({ctx[x.src[0]]}))"),
    # bfloat16 constants need to be rendered as their bit pattern since bf16 is stored as ushort
    (UPat(Ops.CONST, dtypes.bfloat16, name="x"),
      lambda ctx,x: f"{(struct.unpack('I', struct.pack('f', float_to_bf16(x.arg)))[0] >> 16)}u"),
    # load/store image (OpenCL)
    (UPat.var('buf').index(UPat.var('idx_y'), UPat.var('idx_x')), lambda ctx,buf,idx_y,idx_x: f"IMAGE<{ctx[buf]}, {ctx[idx_y]}, {ctx[idx_x]}>"),
    (UPat(Ops.LOAD, dtype=dtypes.float, src=(UPat.var('buf').index(UPat.var('idx_y'), UPat.var('idx_x')), UPat.var("var"), UPat.var("gate"))),
      lambda ctx,buf,idx_y,idx_x,var,gate: f"({ctx[gate]}?read_imagef({ctx[buf]}, smp, (int2)({ctx[idx_x]},{ctx[idx_y]})):{ctx[var]})"),
    (UPat(Ops.LOAD, dtype=dtypes.float, src=(UPat.var('buf').index(UPat.var('idx_y'), UPat.var('idx_x')),)),
      lambda ctx,buf,idx_y,idx_x: f"read_imagef({ctx[buf]}, smp, (int2)({ctx[idx_x]},{ctx[idx_y]}))"),
    (UPat(Ops.STORE, src=(UPat.var('buf').index(UPat.var('idx_y'), UPat.var('idx_x')), UPat.var("var", dtypes.float))),
      lambda ctx,buf,idx_y,idx_x,var: f"write_imagef({ctx[buf]}, (int2)({ctx[idx_x]},{ctx[idx_y]}), {ctx[var]});"),
  ]) + base_rewrite

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    if any(uop.dtype == dtypes.half for uop in uops): prefix = (["#pragma OPENCL EXTENSION cl_khr_fp16 : enable"] + (prefix or []))
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

  def aux(self, uops:list[UOp]):
    arg_dtypes:list[list[tuple[int, DType, tuple|None]]] = []
    for i,u in enumerate(u for u in uops if u.op is Ops.PARAM):
      while len(arg_dtypes) <= u.arg.slot: arg_dtypes.append([])
      arg_dtypes[u.arg.slot].append((i, u.dtype, u._shape))
    return tuple(tuple(a) for a in arg_dtypes),

  def supported_dtypes(self): return {d for d in super().supported_dtypes()
                                      if (d != dtypes.half or "cl_khr_fp16" in self.target.arch) and
                                      (d != dtypes.double or "cl_khr_fp64" in self.target.arch) and d not in dtypes.fp8s}

class MetalRenderer(CStyleLanguage):
  shared_max = 32768
  def __init__(self, target:Target):
    super().__init__(target)
    from tinygrad.runtime.ops_metal import MetalCompiler
    self.compiler, self.tensor_cores = MetalCompiler(), tc.metal if target.arch.startswith("Apple") and int(target.arch[5:]) >= 7 else []

  # language options
  kernel_typedef = "kernel void"
  buffer_prefix = "device "
  smem_prefix = "threadgroup __attribute__((aligned(16))) "
  arg_int_prefix = "constant int&"
  barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"
  float4 = "float4"
  code_for_workitem = {"g": lambda x: f"gid.{chr(120+int(x))}", "l": lambda x: f"lid.{chr(120+int(x))}"}
  # uint3 used for gid/lid - TODO: this should probably be `ushort3 lid [[thread_position_in_threadgroup]]`
  extra_args = ['uint3 gid [[threadgroup_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]']
  type_map = {dtypes.bfloat16: "bfloat"}

  # precise::sin
  code_for_op = {**CStyleLanguage.code_for_op, Ops.SIN: lambda x,dtype: f"precise::sin({x})"}

  # upcast to float32 all the ops that don't support bfloat16
  extra_matcher = PatternMatcher([
    # NOTE: this is copied from PTX
    (UPat((Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN), dtype=dtypes.bfloat16, name="x"),
      lambda x: (UOp(x.op, src=tuple(vv.cast(dtypes.float) for vv in x.src), arg=x.arg).cast(dtypes.bfloat16))),
  ]) + pm_manual_bf16_cast

  string_rewrite = PatternMatcher([
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"as_type<{ctx.render_dtype(x.dtype)}>(({ctx.render_dtype(x.src[0].dtype)})({ctx[x.src[0]]}))"),
  ]) + base_rewrite

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    prefix = ["#include <metal_stdlib>","using namespace metal;"]
    deduped_wmma_args = dedup([(name, dtype_in, dtype_out) for name, _, dtype_in, dtype_out, _, _, _ in wmma_args(uops)])
    for name, dtype_in, dtype_out in deduped_wmma_args:
      dstr_out, dstr_in = self._render_dtype(dtype_out, 2, AddrSpace.REG), self._render_dtype(dtype_in, 2, AddrSpace.REG)
      prefix.append(
f"""{dstr_out} __{name}({dstr_in} a, {dstr_in} b, {dstr_out} c){{
  simdgroup_{self.render_dtype(dtype_in)}8x8 mat_a, mat_b; simdgroup_{self.render_dtype(dtype_out)}8x8 mat_c;
  mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];
  mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];
  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);\n  return {dstr_out}(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);\n}}""")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

  def supported_dtypes(self):
    return {d for d in super().supported_dtypes() if (d != dtypes.bfloat16 or ((arch:=self.target.arch).startswith("Apple") and int(arch[5:]) >= 6))
            and d not in dtypes.fp8s+(dtypes.double,)}

_nms = list("xyzwabcdefghijkl") + [f'v{i}' for i in range(16, 32)]

class CUDARenderer(CStyleLanguage):
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 64)
  shared_max = 49152

  def __init__(self, target:Target, use_nvcc=False):
    super().__init__(target)
    from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler, NVCCCompiler
    iface, dev, arch = target.interface, target.device, target.arch
    self.compiler = (NVCCCompiler if use_nvcc else NVRTCCompiler)(arch, ptx=iface.startswith("MOCK") or dev == "CUDA", cache_key=dev.lower())
    self.tensor_cores = tc.get_cuda(arch)

  # language options
  # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
  kernel_typedef = "extern \"C\" __global__ void __launch_bounds__({launch_bounds})"
  smem_prefix = "__shared__ __align__(16) "
  smem_prefix_for_cast = False
  barrier = "__syncthreads();"
  float4 = "make_float4"
  gep_arr_threshold = 8
  code_for_workitem = {"g": lambda x: f"blockIdx.{chr(120+int(x))}", "l": lambda x: f"threadIdx.{chr(120+int(x))}",
                       "i": lambda x: f"(blockIdx.{chr(120+int(x))}*blockDim.{chr(120+int(x))}+threadIdx.{chr(120+int(x))})"}
  code_for_op = { **CStyleLanguage.code_for_op,
    Ops.TRUNC: lambda x,dtype: f"htrunc({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"trunc({x})",
    Ops.SIN: lambda x,dtype: f"hsin({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sin({x})",
    Ops.LOG2: lambda x,dtype: f"hlog2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"log2({x})",
    Ops.EXP2: lambda x,dtype: f"hexp2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"exp2({x})",
    Ops.SQRT: lambda x,dtype: f"hsqrt({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sqrt({x})",
    Ops.RECIPROCAL: lambda x,dtype: f"hrcp({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"(1/{x})" }
  type_map = {dtypes.bfloat16: "nv_bfloat16", dtypes.fp8e4m3: "__nv_fp8_e4m3", dtypes.fp8e5m2: "__nv_fp8_e5m2"}
  extra_matcher = create_non_native_float_pats(dtypes.fp8s, casting=False) + PatternMatcher([
    (UPat(Ops.CAST, dtypes.fp8s, UPat.var("x", dtypes.fp8s), name='y'), lambda x,y: x.cast(dtypes.float).cast(y.dtype) if x.dtype!=y.dtype else None),
  ])
  string_rewrite = PatternMatcher([
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"tg_bitcast<{ctx.render_dtype(x.dtype)}>(({ctx.render_dtype(x.src[0].dtype)})({ctx[x.src[0]]}))"),
  ]) + base_rewrite

  def render_vector_prefix(self, dt:DType, count:int) -> str:
    vec, scal = self._render_dtype(dt, count, AddrSpace.REG), self.render_dtype(dt)
    elems, header = ', '.join(_nms[:count]), ', '.join([f"{scal} {x}" for x in _nms[:count]])
    return f"struct __align__({dt.itemsize * count}) {vec} {{ {scal} {elems}; }}; " \
           f"__device__ {vec} make_{vec}({header}) {{ {vec} r={{{elems}}}; return r; }}"

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    # TODO: why is dtypes.bfloat16.name == "__bf16"? would be easier not override dtypes.name
    prefix = ["#define INFINITY (__int_as_float(0x7f800000))", "#define NAN (__int_as_float(0x7fffffff))",
              "template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }"]
    used_dtypes = uops_to_dtypes(uops)
    if any(dt in dtypes.fp8s for dt, _ in used_dtypes): prefix.append("#include <cuda_fp8.h>")
    if any(dt == dtypes.half for dt, _ in used_dtypes): prefix.append("#include <cuda_fp16.h>")
    if any(dt == dtypes.bfloat16 for dt, _ in used_dtypes): prefix.append("#include <cuda_bf16.h>")
    prefix += [self.render_vector_prefix(dt, count) for dt, count in used_dtypes if (count in (4,8) and dt in {dtypes.half, dtypes.bfloat16})
      or (count in (2,4,8,16) and dt in dtypes.fp8s)]
    dt_map_in = { dtypes.float: "tf32", dtypes.half: "f16", dtypes.bfloat16: "bf16", dtypes.fp8e4m3: "e4m3", dtypes.fp8e5m2: "e5m2" }
    dt_map_out = { dtypes.float: "f32", dtypes.half: "f16" }
    for name, (N, M, K), dtype_in, dtype_out, _, _, upcast_sizes in wmma_args(uops):
      wmma_dtypes = [self._render_dtype(dtype, size, AddrSpace.REG) for dtype, size in zip([dtype_in, dtype_in, dtype_out], upcast_sizes)]
      n_operands = [size*dtype.itemsize//4 for dtype, size in zip([dtype_in, dtype_in, dtype_out], upcast_sizes)] # 4 => CUDA reg size in bytes
      operands = [f"%{i}" for i in range(sum(n_operands))]

      # mma operands => {c}, {a}, {b}, {c}
      prefix.append(f"""__device__ {wmma_dtypes[2]} __{name}({wmma_dtypes[0]} a, {wmma_dtypes[1]} b, {wmma_dtypes[2]} c){{
  int *a_pk = (int *)(&a), *b_pk = (int *)(&b), *c_pk = (int *)(&c);
  asm("mma.sync.aligned.m{M}n{N}k{K}.row.col.{dt_map_out[dtype_out]}.{dt_map_in[dtype_in]}.{dt_map_in[dtype_in]}.{dt_map_out[dtype_out]}"
      "{{{", ".join(operands[:n_operands[2]])}}}, {{{", ".join(operands[n_operands[2]:n_operands[2]+n_operands[0]])}}},"
      "{{{", ".join(operands[-n_operands[1]:])}}}, {{{", ".join(operands[:n_operands[2]])}}};"
    : {", ".join([f'"+r"(c_pk[{i}])' for i in range(n_operands[2])])}
    : {", ".join([f'"r"(a_pk[{i}])' for i in range(n_operands[0])])}, {", ".join([f'"r"(b_pk[{i}])' for i in range(n_operands[1])])});
  return c;\n}}""")

    return super().render_kernel(function_name, kernel, bufs, uops, prefix=prefix)

  def supported_dtypes(self):
    ver = int(self.target.arch[3:])
    return {d for d in super().supported_dtypes() if (d != dtypes.half or ver >= 53) and (d != dtypes.bfloat16 or ver >= 80)
            and (d not in dtypes.fp8_ocp or ver >= 89) and d not in dtypes.fp8_fnuz}

class NVCCRenderer(CUDARenderer):
  def __init__(self, target:Target): super().__init__(target, use_nvcc=True)

def fp8_index(dtype: DType): return (dtypes.fp8e4m3, dtypes.fp8e5m2).index(dtype.scalar())
def _ocml(op): return lambda x,dtype: f"__ocml_{op}_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})"

class HIPRenderer(CStyleLanguage):
  shared_max = 65536
  # NOTE: this is only really needed on gfx12, even though gfx11 reports the same limitation
  global_max = (2147483647, 65535, 65535)
  global_prod_max = (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)

  @staticmethod
  def is_cdna(arch): return arch.split(":")[0] in {"gfx942", "gfx950"}
  @staticmethod
  def is_cdna4(arch): return arch.split(":")[0] == "gfx950"
  def __init__(self, target:Target, use_hipcc=False): # gfx942 => MI300, gfx1100 => RX 7900, gfx1201 => RX 9700
    super().__init__(target)
    from tinygrad.runtime.support.compiler_amd import HIPCompiler, HIPCCCompiler
    self.compiler, self.tensor_cores = (HIPCCCompiler if use_hipcc else HIPCompiler)(target.arch), tc.get_amd(target.arch)
    if not self.is_cdna4(target.arch): self.extra_matcher += pm_manual_bf16_cast
    if self.is_cdna(target.arch):
      self.string_rewrite = PatternMatcher([
        (UPat(Ops.WMMA, name="x"), lambda ctx,x: f"__{_wmma_name(x)}({ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]},"
          f" {fp8_index(x.src[0].dtype)}, {fp8_index(x.src[0].dtype)}, 0, 0, 0, 0)" if x.arg[0][2] == 128 else None),
        (UPat(Ops.WMMA, name="x"), lambda ctx,x: f"__{_wmma_name(x)}({ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]}, 0, 0, 0)"),
        (UPat(Ops.CONST, dtypes.fp8s, name="x"), lambda ctx,x: f"f32_to_fp8({ctx.nan}, {fp8_index(x.dtype)})" if math.isnan(x.arg) else None),
        (UPat(Ops.CONST, dtypes.fp8s, arg=math.inf, name="x"), lambda ctx,x: f"f32_to_fp8({ctx.infinity}, {fp8_index(x.dtype)})"),
        (UPat(Ops.CONST, dtypes.fp8s, arg=-math.inf, name="x"), lambda ctx,x: f"f32_to_fp8(-{ctx.infinity}, {fp8_index(x.dtype)})"),
        (UPat(Ops.CONST, dtypes.fp8s, name="x"), lambda ctx,x: f"f32_to_fp8({x.arg}f, {fp8_index(x.dtype)})"),
        (UPat(Ops.CAST, dtypes.fp8s, (UPat(dtype=dtypes.float),), name="x",),
          lambda ctx,x: f"f32_to_fp8({ctx[x.src[0]]}, {fp8_index(x.dtype)})"),
        (UPat(Ops.CAST, dtypes.float, (UPat.var("y", dtypes.fp8s),), name="x",),
          lambda ctx,x,y: f"__builtin_amdgcn_cvt_f32_{('fp8', 'bf8')[fp8_index(y.dtype)]}((unsigned int){ctx[x.src[0]]}, 0)"),
      ]) + base_rewrite

  # https://clang.llvm.org/docs/AttributeReference.html#amdgpu-flat-work-group-size
  # NOTE: this makes hlb_cifar10 twice as fast, there may be more gains in tweaking these parameters
  kernel_typedef = 'extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, {launch_bounds})))'
  code_for_workitem = {"g": lambda x: f"__ockl_get_group_id({x})", "l": lambda x: f"__ockl_get_local_id({x})",
                       "i": lambda x: f"(__ockl_get_group_id({x})*__ockl_get_local_size({x})+__ockl_get_local_id({x}))"}
  code_for_op = {**CStyleLanguage.code_for_op, Ops.TRUNC: _ocml("trunc"), Ops.SIN: _ocml("sin"),
                 Ops.LOG2: _ocml("log2"), Ops.EXP2: _ocml("exp2"), Ops.SQRT: _ocml("sqrt")}
  smem_prefix = "__attribute__((shared, aligned(16)))"
  smem_prefix_for_cast: bool = False
  barrier = '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");' + '__builtin_amdgcn_s_barrier();' + \
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");'
  float4 = "make_float4"
  type_map = {dtypes.bfloat16: "hip_bfloat16", dtypes.fp8e4m3: "hip_fp8", dtypes.fp8e5m2: "hip_bf8"}
  extra_matcher = create_non_native_float_pats((dtypes.bfloat16, *dtypes.fp8s)) + PatternMatcher([
    (UPat(Ops.WMMA, name="x", dtype=dtypes.float),
      lambda x: x.replace(src=(x.src[0].bitcast(dtypes.uint64), x.src[1].bitcast(dtypes.uint64), x.src[2]))
      if x.src[0].max_numel() == 8 and x.src[0].dtype in dtypes.fp8_ocp else None),
    # bfloat16 constant casting
    (UPat.cvar('x', dtypes.bfloat16), lambda x: cast_float_to_bf16(UOp.const(dtypes.float, x.arg))),
  ])

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    from tinygrad.renderer.amd.elf import assemble_linear
    return assemble_linear(prg, lin, self.target.arch)

  def render_vector_prefix(self, dtype:DType, count:int) -> str:
    vec, scal = self._render_dtype(dtype, count, AddrSpace.REG), self.render_dtype(dtype)
    return f"typedef {scal} {vec} __attribute__((ext_vector_type({count})));\nstatic inline __attribute__((device)) "+ \
           f"{vec} make_{vec}({', '.join([f'{scal} {x}' for x in _nms[:count]])}) {{ return {{ {', '.join(_nms[:count])} }}; }}"

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    prefix, ockl = [], []
    type_map = { dtypes.bfloat16: "bf16", dtypes.float: "f32", dtypes.half: "f16", dtypes.fp8e4m3: "_fp8_fp8", dtypes.fp8e5m2: "_bf8_bf8" }
    used_dtypes = uops_to_dtypes(uops)
    if any(u.op is Ops.CONST and not math.isfinite(u.arg) for u in uops):
      prefix += ["#define INFINITY (__builtin_inff())", "#define NAN (__builtin_nanf(\"\"))"]
    if any(u.op is Ops.SPECIAL for u in uops):
      prefix.append("typedef long unsigned int size_t;")
      ockl = [(f"__ockl_get_{name}", "unsigned int", "size_t", "const") for name in ["local_id", "group_id", "local_size"]]
    ocml_ops = {Ops.EXP2: ("exp2", "pure"), Ops.LOG2: ("log2", "pure"), Ops.SQRT: ("sqrt", "const"), Ops.SIN: ("sin", ""), Ops.TRUNC: ("trunc", "")}
    ocml = [(f"__ocml_{ocml_ops[op][0]}_f{dt.bitsize}", dt.name, dt.name, ocml_ops[op][1])
      for op, dt in dedup((u.op, u.dtype.scalar()) for u in uops) if op in ocml_ops and dt in (dtypes.half, dtypes.float, dtypes.double)]
    if any(dt == dtypes.bfloat16 for dt, _ in used_dtypes):
      prefix.append(f"typedef {'__bf16' if self.is_cdna4(self.target.arch) else 'unsigned short'} hip_bfloat16;")
    if any(dt == dtypes.half for dt, _ in used_dtypes): prefix.append("#define half _Float16")
    if any(dt in dtypes.fp8s for dt, _ in used_dtypes):
      prefix += ["typedef unsigned char hip_bf8;", "typedef unsigned char hip_fp8;"]
    if any((u.op is Ops.CAST and u.dtype in dtypes.fp8s and u.src[0].dtype == dtypes.float) or
           (u.op is Ops.CONST and u.dtype in dtypes.fp8s) for u in uops):
      prefix.append("""static inline __attribute__((device)) unsigned char f32_to_fp8(float v, int is_bf8) {
  v = (((*(unsigned*)&v)&0x7F800000)!=0x7F800000)?__builtin_amdgcn_fmed3f(v,is_bf8?57344.0f:448.0f,is_bf8?-57344.0f:-448.0f) : v;
  return (unsigned char)(is_bf8?__builtin_amdgcn_cvt_pk_bf8_f32(v,v,0,false):__builtin_amdgcn_cvt_pk_fp8_f32(v,v,0,false));\n}""")
    prefix += [f'extern "C" __attribute__((device{f", {atr}" if atr else ""})) {dto} {meth}({dti});' for meth,dti,dto,atr in ockl+ocml]
    prefix += [self.render_vector_prefix(dt, count) for dt, count in used_dtypes if count > 1]

    for name, (N, M, K), dtype_in, dtype_out, _, _, _ in wmma_args(uops): # TODO: handle TCs f32_bf16 and bf16_bf16 w/ wrapper
      if self.is_cdna(self.target.arch):
        if (N, M, K) == (16, 16, 16): type_map[dtypes.bfloat16] = 'bf16_1k'
        elif (N, M, K) == (16, 16, 32): type_map = {**type_map, dtypes.bfloat16: "_bf16", dtypes.half: "_f16"}
        elif (N, M, K) == (16, 16, 128): type_map = {**type_map, dtypes.fp8e4m3: "_f8f6f4", dtypes.fp8e5m2: "_f8f6f4"}
        prefix.append(f"#define __{name} __builtin_amdgcn_mfma_{'scale_' if K == 128 else ''}f32_{N}x{M}x{K}{type_map[dtype_in]}")
      # #define __WMMA_16_16_16_half_half __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12
      elif self.tensor_cores == tc.amd_rdna4:
        prefix.append(f"#define __{name} __builtin_amdgcn_wmma_{type_map[dtype_out]}_16x16x16_{type_map[dtype_in]}_w32_gfx12")
      elif dtype_out == dtypes.int32:
        prefix.append("typedef int wmma_int4 __attribute__((ext_vector_type(4)));\n"+
          f"static inline __attribute__((device)) int8 __{name}"+"""(signed_char16 a, signed_char16 b, int8 c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, __builtin_bit_cast(wmma_int4, a),
    true, __builtin_bit_cast(wmma_int4, b), c, false);\n}""")
      elif dtype_out == dtypes.float:
        prefix.append(f"#define __{name} __builtin_amdgcn_wmma_f32_16x16x16_{'f16' if dtype_in == dtypes.half else 'bf16'}_w32")
      else: prefix.append(f"static inline __attribute__((device)) half8 __{name}"+"""(half16 a, half16 b, half8 c) {
  half16 c_frag = {}; half8 d; for (int n = 0; n < 8; n++) { c_frag[n*2] = c[n]; }
  c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c_frag, false);
  for (int n = 0; n < 8; n++) { d[n] = c_frag[n*2]; } return d;\n}""")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

  def supported_dtypes(self): return {d for d in super().supported_dtypes()
                                      if (d not in dtypes.fp8_ocp or self.target.arch == "gfx950") and d not in dtypes.fp8_fnuz}

class HIPCCRenderer(HIPRenderer):
  def __init__(self, target:Target): super().__init__(target, use_hipcc=True)

class QCOMCLRenderer(OpenCLRenderer):
  def __init__(self, target:Target):
    super().__init__(target)
    from tinygrad.runtime.support.compiler_qcom import QCOMCompiler
    self.compiler = QCOMCompiler(target.arch)

  # QCOM compiler is flaky with half
  def supported_dtypes(self):
    return {d for d in Renderer.supported_dtypes(self)
            if (d != dtypes.float16 or (bool(IMAGE) and bool(FLOAT16))) and d not in dtypes.fp8s+(dtypes.bfloat16,dtypes.double)}
