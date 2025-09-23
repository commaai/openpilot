from typing import Literal, Callable, cast
import os, math, sys
from collections import defaultdict, Counter
from tinygrad.codegen.opt import tc
from tinygrad.uop.ops import GroupOp, Ops, UOp, PatternMatcher, UPat
from tinygrad.helpers import strip_parens, getenv, prod, dedup, AMX
from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType, AddrSpace, truncate
from tinygrad.renderer import Renderer
from tinygrad.codegen.devectorizer import no_vectorized_alu

base_rewrite = PatternMatcher([
  (UPat(Ops.DEFINE_REG, name="x"), lambda ctx,x: f"{ctx.render_dtype(x.dtype.base)} {ctx[x]}[{x.dtype.size}];"),
  (UPat(Ops.IF, name="x"), lambda ctx,x: f"if ({ctx[x.src[0]]}) {{"),
  (UPat((Ops.ENDIF, Ops.ENDRANGE)), lambda ctx: "}"),
  (UPat(Ops.WMMA, name="x"), lambda ctx,x: f"__{x.arg[0]}({ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]})"),
  # r method accesses
  (UPat(Ops.RANGE, name="x"),
   lambda ctx,x: f"for ({ctx.render_dtype(x.dtype)} {ctx[x]} = 0; {ctx[x]} < {ctx[x.src[0]]}; {ctx[x]}++) {{"),
  (UPat(Ops.VECTORIZE, name="x"),
   lambda ctx,x: f"{ctx.float4.replace('float4', ctx.render_dtype(x.dtype))}" + \
    f"{ctx.float4_style[0]}{','.join([ctx[y] for y in x.src])}{ctx.float4_style[1]}"),
  (UPat(Ops.CAST, name="x"), lambda ctx,x:
    f"__builtin_convertvector({ctx[x.src[0]]}, {ctx.render_dtype(x.dtype)})" if x.dtype.count > 1 and not isinstance(x.dtype, PtrDType) else None),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, ctx[x.src[0]])})"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"(*(({ctx.buffer_prefix}{ctx.render_dtype(x.dtype)}*)&{ctx[x.src[0]]}))"),
  (UPat(Ops.DEFINE_LOCAL, name="x"), lambda ctx,x: f"{ctx.smem_align}{ctx.smem_prefix}{ctx.render_dtype(x.dtype.base)} {ctx[x]}[{x.dtype.size}];"),
  (UPat(Ops.BARRIER), lambda ctx: ctx.barrier),
  (UPat(Ops.PRECAST, name="x"), lambda ctx,x: ctx[x.src[0]]),
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: f"{ctx.code_for_workitem[x.arg[0][0]](x.arg[0][-1])}; /* {x.arg[1]} */"),
  # const
  (UPat(Ops.CONST, arg=math.inf, name="x"), lambda ctx, x: f"({ctx.render_cast(x.dtype, ctx.infinity)})"),
  (UPat(Ops.CONST, arg=-math.inf, name="x"), lambda ctx, x: f"({ctx.render_cast(x.dtype, f'-{ctx.infinity}')})"),
  (UPat(Ops.CONST, dtype=dtypes.floats, name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, ctx.nan)})" if math.isnan(x.arg) else None),
  (UPat(Ops.CONST, dtype=dtypes.float, name="x"), lambda ctx,x: f"{x.arg}f"),
  (UPat(Ops.CONST, dtype=dtypes.int64, name="x"), lambda ctx,x: f"{x.arg}ll"),
  (UPat(Ops.CONST, dtype=dtypes.uint64, name="x"), lambda ctx,x: f"{truncate[x.dtype](x.arg)}ull"),
  (UPat(Ops.CONST, dtype=dtypes.uint32, name="x"), lambda ctx,x: f"{truncate[x.dtype](x.arg)}u"),
  (UPat(Ops.CONST, dtype=dtypes.bool, name="x"), lambda ctx,x: "1" if x.arg else "0"),
  # consts are rendered to larger type and casted
  (UPat(Ops.CONST, (dtypes.bfloat16, dtypes.half), name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, f'{x.arg}f')})"),
  (UPat(Ops.CONST, (dtypes.uint8, dtypes.uint16), name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, f'{x.arg}u')})"),
  (UPat(Ops.CONST, (dtypes.int8, dtypes.int16), name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, str(x.arg))})"),
  # default const render
  (UPat(Ops.CONST, name="x"), lambda ctx,x: str(x.arg)),
  # new load/store
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var('idx')), allow_any_len=True),
   lambda ctx,buf,idx: f"({ctx[buf]}+{strip_parens(ctx[idx]) if idx.arg == Ops.ADD else ctx[idx]})"),
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(), UPat(), UPat.var("gate"))).or_casted("bidx"), UPat.var("var")), allow_any_len=True),
   lambda ctx,bidx,var,gate: f"({ctx[gate]}?*{ctx[bidx]}:{ctx[var]})"),
  (UPat(Ops.LOAD, src=(UPat.var('bidx'),), allow_any_len=True), lambda ctx,bidx: f"(*{ctx[bidx]})"),
  (UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var")), allow_any_len=True), lambda ctx,bidx,var: f"*{ctx[bidx]} = {ctx[var]};"),
  # alu/gep
  # TODO: look for left-associative
  (UPat(GroupOp.ALU, name="x"), lambda ctx,x: ctx.code_for_op[x.op](
    *([strip_parens(ctx[v]) if v.op == x.op and x.op in {Ops.ADD, Ops.MUL, Ops.XOR, Ops.OR, Ops.AND} else ctx[v] for v in x.src]), x.dtype)),
  (UPat(Ops.GEP, name="x"), lambda ctx,x: ctx[x.src[0]] + \
    (f"[{x.arg[0]}]" if x.src[0].dtype.count > ctx.gep_arr_threshold else f".{'xyzwabcd'[x.arg[0]]}")),
  # custom passes through with format
  (UPat((Ops.CUSTOM, Ops.CUSTOMI), name="x"), lambda ctx,x: x.arg.format(*[ctx[y] for y in x.src])),
])

extra_pm = PatternMatcher([
  # insert a PRECAST before BITCAST to force it to be rendered. not needed on all backends?
  (UPat(Ops.BITCAST, name="x"), lambda x: UOp(Ops.BITCAST, x.dtype, (UOp(Ops.PRECAST, x.src[0].dtype, x.src),))
   if x.src[0].op not in {Ops.PRECAST, Ops.LOAD, Ops.CUSTOM} else None),
  # rewrite MAX to CMPLT + WHERE (max function is annoying on many cstyle backends)
  (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
  # devectorize any bools
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.INDEX), dtype=dtypes.bool, name="alu"), no_vectorized_alu),
  # CAST (from bool) can't be vectorized
  (UPat(Ops.CAST, src=(UPat(dtype=dtypes.bool),), name="alu"), no_vectorized_alu),
  # WHERE can't be vectorized
  (UPat(Ops.WHERE, name="alu"), no_vectorized_alu),
])

def uops_to_dtypes(uops:list[UOp]) -> list[DType]: return dedup(u.dtype for u in uops if not isinstance(u.dtype, (ImageDType, PtrDType)))

# (name, dims, dtype_in, dtype_out, device, threads, upcast_axes, reduce_axes)
def wmma_args(uops:list[UOp]):
  return dedup((uop.arg[0], uop.arg[1], uop.src[0].dtype.scalar(), uop.dtype.scalar(), *(uop.arg[4:8])) for uop in uops if uop.op is Ops.WMMA)

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
    Ops.SQRT: lambda x,dtype: f"sqrt({x})", Ops.RECIP: lambda x,dtype: f"(1/{x})", Ops.NEG: lambda x,dtype: f"-{x}",
    Ops.EXP2: lambda x,dtype: f"exp2({x})", Ops.LOG2: lambda x,dtype: f"log2({x})", Ops.SIN: lambda x,dtype: f"sin({x})",
    Ops.TRUNC: lambda x,dtype: f"trunc({x})",
    Ops.AND: lambda a,b,dtype: f"({a}&{b})", Ops.XOR: lambda a,b,dtype: f"({a}^{b})", Ops.OR: lambda a,b,dtype: f"({a}|{b})",
    Ops.ADD: lambda a,b,dtype: f"({a}+{b})", Ops.SUB: lambda a,b,dtype: f"({a}-{b})", Ops.MUL: lambda a,b,dtype: f"({a}*{b})",
    Ops.MOD: lambda a,b,dtype: f"({a}%{b})", Ops.IDIV: lambda a,b,dtype: f"({a}/{b})", Ops.CMPNE: lambda a,b,dtype: f"({a}!={b})",
    Ops.SHR: lambda a,b,dtype: f"({a}>>{b})", Ops.SHL: lambda a,b,dtype: f"({a}<<{b})", Ops.CMPLT: lambda a,b,dtype: f"({a}<{b})",
    Ops.WHERE: lambda a,b,c,dtype: f"({a}?{b}:{c})", Ops.CMPEQ: lambda a,b,dtype: f"({a}=={b})"}

  string_rewrite = base_rewrite
  extra_matcher = extra_pm

  def render_kernel(self, function_name:str, kernel:list[str], bufs:list[tuple[str,tuple[DType,bool]]], uops:list[UOp], prefix=None) -> str:
    tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" if any(isinstance(dtype, ImageDType) for _,(dtype,_) in bufs) else ""  # noqa: E501
    buftypes = [(name, self.render_dtype(dtype, mutable)+self.buffer_suffix if isinstance(dtype, (ImageDType, PtrDType)) else
                self.arg_int_prefix if dtype == dtypes.int else None) for name,(dtype,mutable) in bufs]
    launch_bounds = prod(u.arg[1] for u in uops if u.op is Ops.SPECIAL and u.arg[0][0] == "l")
    prg = ''.join([f"{self.kernel_typedef.format(launch_bounds=launch_bounds)} {function_name}(",] +
    [', '.join([f'{t} {name}' for name,t in buftypes] + self.extra_args)] +
    [") {\n" + tmp] + ['\n'.join(kernel), "\n}"])
    return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"

  def render_cast(self, dt:DType, val: str) -> str: return f"({self.render_dtype(dt)})({val})"
  def render_dtype(self, dt:DType, mutable=True) -> str:
    if isinstance(dt, ImageDType): return f"{'write_only' if mutable else 'read_only'} image2d_t"
    if isinstance(dt, PtrDType):
      prefix = ""
      if dt.addrspace == AddrSpace.LOCAL and self.smem_prefix_for_cast: prefix = self.smem_prefix
      if dt.addrspace == AddrSpace.GLOBAL: prefix = self.buffer_prefix
      return prefix + self.render_dtype(dt.base) + "*"
    if dt.count > 1: return self.type_map.get(scalar:=dt.scalar(), scalar.name).replace(" ", "_") + str(dt.count)
    return self.type_map.get(scalar:=dt.scalar(), scalar.name)

  def __getitem__(self, key): return self.r[key]  # hacky helper
  def _render(self, uops:list[UOp]) -> tuple[str, list[str], list[tuple[str,tuple[DType,bool]]]]:
    r: dict[UOp, str] = {}
    self.r = r

    child_count = Counter(v for ru in uops for v in ru.src)
    bufs: dict[UOp, tuple[str, tuple[DType, bool]]] = {}
    kernel = []
    depth = 1
    c: defaultdict[str, int] = defaultdict(int)
    name = "test"
    for u in uops:
      if u.op is Ops.NOOP: continue
      if u.op is Ops.SINK:
        if u.arg is not None: name = u.arg.function_name
        continue
      if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
        r[u] = (f"data{u.arg}_{sz}" if (sz:=cast(PtrDType, u.dtype).size) > 0 else f"data{u.arg}") if u.op is Ops.DEFINE_GLOBAL else u.arg[0]
        bufs[u] = (r[u], (u.dtype, False))
        continue

      # mark buffers that we store to writable
      if u.op is Ops.STORE:
        for up in u.src[0].toposort():
          if up.op is Ops.DEFINE_GLOBAL: bufs[up] = (bufs[up][0], (bufs[up][1][0], True))

      # naming
      prefix = None
      if u.op is Ops.SPECIAL: r[u] = u.arg[0]
      elif u.op is Ops.RANGE: r[u] = f"ridx{u.arg}"
      else:
        prefix = {Ops.WMMA: "wmma", Ops.DEFINE_LOCAL: "temp", Ops.CONST: "const",
                  Ops.CAST: "cast", Ops.BITCAST: "cast", Ops.GEP: "gep", Ops.VECTORIZE: "cast", Ops.PRECAST: "precast",
                  Ops.INDEX: "bidx", Ops.DEFINE_REG: "acc", Ops.LOAD: "val"}.get(u.op, "alu")
        r[u] = f"{prefix}{c[prefix]}"

      l = cast(str, self.string_rewrite.rewrite(u, ctx=self))
      assert l is not None, f"failed to render {u.op} {u.dtype} {[(x.op,x.dtype) for x in u.src]} {u.arg}"

      if u.op in {Ops.ENDIF, Ops.ENDRANGE}: depth -= 1
      if (u.op is not Ops.CAST or u.dtype.vcount == 1) and (u.op in {Ops.CONST, Ops.GEP, Ops.INDEX, Ops.CUSTOMI} or \
        (u.op is Ops.LOAD and cast(PtrDType, u.src[0].dtype).addrspace == AddrSpace.REG) or \
        (u.op is Ops.CAST and isinstance(u.dtype, PtrDType)) or \
        (u.op in {Ops.VECTORIZE, *(GroupOp.ALU-{Ops.WHERE}), Ops.CAST, Ops.BITCAST} and child_count[u] == 1 and not getenv("EXPAND_SSA"))):
        r[u] = l
      else:
        if u.op in {Ops.RANGE, Ops.DEFINE_LOCAL, Ops.STORE, Ops.DEFINE_REG} or u.dtype == dtypes.void: pass
        else: l = f"{self.render_dtype(u.dtype)} {r[u]} = {l}" + (";" if u.op is not Ops.SPECIAL else "")
        kernel.append("  "*depth + l)
        if prefix: c[prefix] += 1  # if it was used, increment
      if u.op in {Ops.IF, Ops.RANGE}: depth += 1
    del self.r

    # NOTE: this relies on bufs dict preserving order
    return (name, kernel, list(bufs.values()))
  def render(self, uops:list[UOp]) -> str: return self.render_kernel(*self._render(uops), uops)

class ClangRenderer(CStyleLanguage):
  device = "CPU"
  float4 = "(float4)"
  float4_style = ('{', '}')
  gep_arr_threshold = 0
  has_local = False
  global_max = None
  infinity = "__builtin_inff()"
  nan = '__builtin_nanf("")'
  if AMX: tensor_cores = tc.amx

  # language options
  buffer_suffix = " restrict"
  type_map = {dtypes.bool:"_Bool", dtypes.half:"__fp16"}
  code_for_op = {**({k:v for k,v in CStyleLanguage.code_for_op.items() if k not in [Ops.EXP2, Ops.SIN, Ops.LOG2, Ops.TRUNC]}),
                 Ops.SQRT: lambda x,dtype: f"__builtin_sqrt({x})" if dtype == dtypes.float64 else f"__builtin_sqrtf({x})",
                 Ops.TRUNC: lambda x,dtype: f"__builtin_trunc({x})" if dtype == dtypes.float64 else f"__builtin_truncf({x})"}
  # LLVM legalizes double => half cast on systems that don't support it natively (like x86 cpus without AVX512-FP16) into a compiler-rt libcall.
  extra_matcher = PatternMatcher([(UPat.var("x", dtypes.float64).cast(dtypes.float16), lambda x: x.cast(dtypes.float32).cast(dtypes.float16)),
    (UPat((Ops.SQRT, Ops.TRUNC), name="alu"), no_vectorized_alu),]) + CStyleLanguage.extra_matcher

  if sys.platform == 'win32':
    kernel_typedef = "__attribute__((ms_abi)) void"
  def render_vector_prefix(self, dt:DType) -> str:
    # round (down) to power of two (this is actually the default clang behavior)
    alignment = 2**int(math.log2(dt.itemsize)) if getenv("ALIGNED", 1) else 1
    return f"typedef {self.render_dtype(dt.scalar())} {self.render_dtype(dt)} __attribute__((aligned({alignment}),vector_size({dt.itemsize})));"

  def _render_defines(self, uops) -> list[str]:
    prefix = [self.render_vector_prefix(dt) for dt in uops_to_dtypes(uops) if dt.count > 1]
    # https://github.com/corsix/amx
    for name, (N, M, _), dtype_in, _, _, _, _, _ in wmma_args(uops):
      prefix += [
        '#define AMX_SET(imm5) __asm("nop\\nnop\\nnop\\n.word (0x201000+(%0<<5)+%1)" : : "i"(17), "i"(imm5) : "memory")',
        '#define AMX(op, gpr, btf) __asm(".word (0x201000+(%0 << 5)+0%1-((0%1>>4)*6))" : : "i"(op), "r"((unsigned long long)(gpr)+(btf)) : "memory")',
      ]
      # 'static' in C roughly means that function symbol isn't exported. LLVM puts those symbols at the end of object file which allows Clang JIT
      # to just jump at the start of a shellcode without having to deal with symbols or trampolines at all. This is better than having to inline
      # wmma function every time it is called or wasting complexity on a symbol parsing and a memory page on trampoline.
      prefix += [f"""static {(out := self.render_dtype(dtype_in.vec(N*N)))} __{name}({self.render_dtype(dtype_in.vec(N))} data1, {self.render_dtype(dtype_in.vec(M))} data2, {out} data0){{
  AMX_SET(0);\n  for(int ridx0 = 0; ridx0 < 16; ridx0++){{ AMX(4, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }}
  AMX(0, (int *)(&data2), 0ull<<62); AMX(1, (int *)(&data1), 0ull<<62); AMX(12, 0, 0ull);
  for(int ridx0 = 0; ridx0 < 16; ridx0++){{ AMX(5, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }}\n  AMX_SET(1);\n  return data0;\n}}"""] # noqa: E501
    return prefix
  def _render_body(self, function_name, kernel, bufs, uops, pref=None) -> str: return super().render_kernel(function_name, kernel, bufs, uops, pref)
  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[DType,bool]]]) -> str: return ""

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    defines = '\n'.join(self._render_defines(uops))
    return defines + "\n" + self._render_body(function_name, kernel, bufs, uops, prefix) + "\n" + self._render_entry(function_name, bufs)

class OpenCLRenderer(CStyleLanguage):
  device = "GPU"

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

  string_rewrite = PatternMatcher([
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"as_{ctx.render_dtype(x.dtype)}({ctx[x.src[0]]})"),
    # load/store image (OpenCL)
    (UPat(Ops.LOAD, dtype=dtypes.float.vec(4), src=(UPat.var('buf').index(UPat.var('idx', dtypes.int.vec(2)), UPat.var("gate")), UPat.var("var"))),
      lambda ctx,buf,idx,var,gate: f"({ctx[gate]}?read_imagef({ctx[buf]}, smp, {ctx[idx]}):{ctx[var]})"),
    (UPat(Ops.LOAD, dtype=dtypes.float.vec(4), src=(UPat.var('buf').index(UPat.var('idx', dtypes.int.vec(2))),)),
      lambda ctx,buf,idx: f"read_imagef({ctx[buf]}, smp, {ctx[idx]})"),
    (UPat(Ops.STORE, src=(UPat.var('buf').index(UPat.var('idx', dtypes.int.vec(2))), UPat.var("var", dtypes.float.vec(4))), allow_any_len=True),
      lambda ctx,buf,idx,var: f"write_imagef({ctx[buf]}, {ctx[idx]}, {ctx[var]});"),
  ]) + base_rewrite

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    if any(uop.dtype.base == dtypes.half for uop in uops): prefix = (["#pragma OPENCL EXTENSION cl_khr_fp16 : enable"] + (prefix or []))
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

class IntelRenderer(OpenCLRenderer):
  device, suffix, kernel_typedef = "GPU", "INTEL", "__attribute__((intel_reqd_sub_group_size(8)))\n" + "__kernel void"
  tensor_cores = tc.intel

  string_rewrite = PatternMatcher([
    (UPat(Ops.CAST, dtype=dtypes.bfloat16, src=(UPat.var('x', dtype=dtypes.float),)), lambda ctx,x: f"intel_convert_bfloat16_as_ushort({ctx[x]})"),
    (UPat(Ops.CAST, dtype=dtypes.float, src=(UPat.var('x', dtype=dtypes.bfloat16),)), lambda ctx,x: f"intel_convert_as_bfloat16_float({ctx[x]})"),
  ]) + OpenCLRenderer.string_rewrite

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    prefix = []
    for name, _, dtype_in, dtype_out, _, _, _, _ in wmma_args(uops):
      dt_in = ("ushort", "bf16") if dtype_in == dtypes.bfloat16 else (dtype_in.name, "f16")
      prefix.append(f"""{dtype_out.name}8 __{name}({dt_in[0]}16 a, {dt_in[0]}16 b, {dtype_out.name}8 c) {{
    return intel_sub_group_{dt_in[1]}_{dt_in[1]}_matrix_mad_k16(as_int8(a), as_int8(b), c);\n}}""")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix or None)

class MetalRenderer(CStyleLanguage):
  device = "METAL"
  shared_max = 32768
  def __init__(self): self.tensor_cores = tc.metal if hasattr(os, 'uname') and os.uname().machine == "arm64" else []

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
      lambda x: (UOp(x.op, dtypes.float, tuple(vv.cast(dtypes.float) for vv in x.src), x.arg).cast(dtypes.bfloat16))),
  ]) + extra_pm

  string_rewrite = PatternMatcher([
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"as_type<{ctx.render_dtype(x.dtype)}>({ctx[x.src[0]]})"),
  ]) + base_rewrite

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    prefix = ["#include <metal_stdlib>","using namespace metal;"]
    for name, _, dtype_in, dtype_out, _, _, _, _ in wmma_args(uops): prefix.append(
  f"""{(dstr_out:=self.render_dtype(dtype_out.vec(2)))} __{name}({(dstr_in:=self.render_dtype(dtype_in.vec(2)))} a, {dstr_in} b, {dstr_out} c){{
  simdgroup_{self.render_dtype(dtype_in)}8x8 mat_a, mat_b; simdgroup_{self.render_dtype(dtype_out)}8x8 mat_c;
  mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];
  mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];
  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);\n  return {dstr_out}(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);\n}}""")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

_nms = "xyzwabcdefghijkl"

class CUDARenderer(CStyleLanguage):
  device = "CUDA"
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 64)
  shared_max = 49152

  def __init__(self, arch:str):
    self.tensor_cores, self.arch = tc.cuda_sm80 if int(arch[3:]) >= 80 else tc.cuda_sm75 if int(arch[3:]) >= 75 else [], arch
  def __reduce__(self): return self.__class__, (self.arch,)

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
    Ops.SIN: lambda x,dtype: f"hsin({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sin({x})",
    Ops.LOG2: lambda x,dtype: f"hlog2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"log2({x})",
    Ops.EXP2: lambda x,dtype: f"hexp2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"exp2({x})",
    Ops.SQRT: lambda x,dtype: f"hsqrt({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sqrt({x})",
    Ops.RECIP: lambda x,dtype: f"hrcp({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"(1/{x})" }
  type_map = {dtypes.bfloat16: "nv_bfloat16"}

  def render_vector_prefix(self, dt:DType) -> str:
    vec, scal = self.render_dtype(dt), self.render_dtype(dt.scalar()),
    elems, header = ', '.join(_nms[:dt.count]), ', '.join([f"{scal} {x}" for x in _nms[:dt.count]])
    return f"struct __align__({dt.itemsize}) {vec} {{ {scal} {elems}; }}; __device__ {vec} make_{vec}({header}) {{ {vec} r={{{elems}}}; return r; }}"

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    # TODO: why is dtypes.bfloat16.name == "__bf16"? would be easier not override dtypes.name
    prefix = ["#define INFINITY (__int_as_float(0x7f800000))","#define NAN (__int_as_float(0x7fffffff))"]

    used_dtypes = uops_to_dtypes(uops)
    if any(dt.scalar() == dtypes.half for dt in used_dtypes): prefix.append("#include <cuda_fp16.h>")
    if any(dt.scalar() == dtypes.bfloat16 for dt in used_dtypes): prefix.append("#include <cuda_bf16.h>")
    prefix += [self.render_vector_prefix(dt) for dt in used_dtypes if dt.count in (4,8) and dt.scalar() in {dtypes.half, dtypes.bfloat16}]

    dt_map_in = { dtypes.float: "tf32", dtypes.half: "f16", dtypes.bfloat16: "bf16" }
    dt_map_out = { dtypes.float: "f32", dtypes.half: "f16" }
    for name, (N, M, K), dtype_in, dtype_out, _, _, upcast_axes, _ in wmma_args(uops):
      upcast_sizes = [prod(size for _, size in upcast) for upcast in upcast_axes]
      wmma_dtypes = [self.render_dtype(dtype.vec(size)) for dtype, size in zip([dtype_in, dtype_in, dtype_out], upcast_sizes)]
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

def cast_float_to_bf16(x: UOp) -> UOp:
  assert x.dtype == dtypes.float, "cast float -> bf16 must start with float"
  x = x.bitcast(dtypes.uint)
  x = (-x & 0x7f800000).where(x + ((x >> 16) & 1) + 0x7fff, (x & 0xffff).where((x | 0x10000), x))
  return (x >> 16).cast(dtypes.ushort).bitcast(dtypes.bfloat16)

class AMDRenderer(CStyleLanguage):
  device = "AMD"
  shared_max = 65536
  # NOTE: this is only really needed on gfx12, even though gfx11 reports the same limitation
  global_max = (2147483647, 65535, 65535)

  @staticmethod
  def get_tensor_cores(arch):
    return {"gfx942": tc.amd_cdna, "gfx950": tc.amd_cdna, "gfx1200": tc.amd_rdna4, "gfx1201": tc.amd_rdna4}.get(arch.split(":")[0], tc.amd_rdna3)
  def __init__(self, arch:str): # gfx942 => MI300, gfx1100 => RX 7900, gfx1201 => RX 9700
    self.arch = arch
    self.tensor_cores = self.get_tensor_cores(arch)
    if self.tensor_cores == tc.amd_cdna:
      self.string_rewrite = PatternMatcher([
        (UPat(Ops.WMMA, name="x"), lambda ctx,x: f"__{x.arg[0]}({ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]}, 0, 0, 0)")]) + base_rewrite
  def __reduce__(self): return self.__class__, (self.arch,)

  # language options
  ockl = [(f"__ockl_get_{name}", "unsigned int", "size_t", "const") for name in ["local_id", "group_id", "local_size"]]
  ocml = [(f"__ocml_{name}_f{n}", f"{dt}, {dt}" if "fmax" == name else dt, dt, atr)
            for dt, n in [(dtype.name, dtype.itemsize * 8) for dtype in [dtypes.float, dtypes.double, dtypes.half]]
            for name, atr in [("fmax", "const"), ("exp2", "pure"), ("log2", "pure"), ("sqrt", "const"), ("sin", ""), ("trunc", "")]]

  kernel_typedef = "\n".join(f'extern "C" __attribute__((device{f", {atr}" if atr else ""})) {dto} {meth}({dti});' for meth,dti,dto,atr in ockl+ocml)
  # https://clang.llvm.org/docs/AttributeReference.html#amdgpu-flat-work-group-size
  # NOTE: this makes hlb_cifar10 twice as fast, there may be more gains in tweaking these parameters
  kernel_typedef += '\nextern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, {launch_bounds})))'
  code_for_workitem = {"g": lambda x: f"__ockl_get_group_id({x})", "l": lambda x: f"__ockl_get_local_id({x})",
                       "i": lambda x: f"(__ockl_get_group_id({x})*__ockl_get_local_size({x})+__ockl_get_local_id({x}))"}
  code_for_op = { **CStyleLanguage.code_for_op,
    Ops.TRUNC: lambda x,dtype: f"__ocml_trunc_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
    Ops.SIN: lambda x,dtype: f"__ocml_sin_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
    Ops.LOG2: lambda x,dtype: f"__ocml_log2_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
    Ops.EXP2: lambda x,dtype: f"__ocml_exp2_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
    Ops.SQRT: lambda x,dtype: f"__ocml_sqrt_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})" }
  smem_prefix = "__attribute__((shared, aligned(16)))"
  smem_prefix_for_cast: bool = False
  barrier = '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");' + '__builtin_amdgcn_s_barrier();' + \
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");'
  float4 = "make_float4"
  type_map = {dtypes.bfloat16: "hip_bfloat16"}
  extra_matcher = PatternMatcher([
    # cast bfloat16 alus to float
    (UPat(Ops.WHERE, src=(UPat.var("b"), UPat.var("x", dtype=dtypes.bfloat16), UPat.var("y", dtype=dtypes.bfloat16))),
      lambda b,x,y: UOp(Ops.WHERE, dtype=dtypes.float, src=(b,x.cast(dtypes.float),y.cast(dtypes.float))).cast(dtypes.bfloat16)),
    (UPat(GroupOp.ALU, dtype=dtypes.bfloat16, name="x"),
      lambda x: UOp(x.op, dtypes.float, tuple(vv.cast(dtypes.float) for vv in x.src), x.arg).cast(dtypes.bfloat16)),
    (UPat(GroupOp.ALU, dtypes.bool, name="alu", src=(UPat.var("x", dtype=dtypes.bfloat16), UPat.var("y", dtype=dtypes.bfloat16))),
      lambda alu,x,y: UOp(alu.op, dtypes.bool, (x.cast(dtypes.float), y.cast(dtypes.float)), alu.arg)),
    # add float intermediate casting for bfloat16
    (UPat(Ops.CAST, name="x", src=(UPat.var("y", dtypes.bfloat16),)),
      lambda x,y: y.cast(dtypes.float).cast(x.dtype) if x.dtype!=dtypes.float else None),
    (UPat(Ops.CAST, dtypes.bfloat16, (UPat.var("x"),)),
      lambda x: x.cast(dtypes.float).cast(dtypes.bfloat16) if x.dtype!=dtypes.float else None),
    # bfloat16 casting
    (UPat.cvar('x', dtypes.bfloat16), lambda x: cast_float_to_bf16(UOp.const(dtypes.float, x.arg))),
    (UPat(Ops.CAST, dtypes.float, (UPat.var("x", dtypes.bfloat16),)),
     lambda x: (x.bitcast(dtypes.ushort).cast(dtypes.uint)<<16).bitcast(dtypes.float)),
    (UPat(Ops.CAST, dtype=dtypes.bfloat16, src=(UPat.var("x", dtype=dtypes.float),)), cast_float_to_bf16)]) + extra_pm

  def render_vector_prefix(self, dtype:DType) -> str:
    vec, scal = self.render_dtype(dtype), self.render_dtype(dtype.scalar())
    return f"typedef {scal} {vec} __attribute__((ext_vector_type({dtype.count})));\nstatic inline __attribute__((device)) "+ \
           f"{vec} make_{vec}({', '.join([f'{scal} {x}' for x in _nms[:dtype.count]])}) {{ return {{ {', '.join(_nms[:dtype.count])} }}; }}"

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    prefix = ["#define INFINITY (__builtin_inff())","#define NAN (__builtin_nanf(\"\"))","typedef long unsigned int size_t;","#define half _Float16"]
    type_map = { dtypes.bfloat16: "bf16", dtypes.float: "f32", dtypes.half: "f16" }
    used_dtypes = uops_to_dtypes(uops)
    if any(dt.scalar() == dtypes.bfloat16 for dt in used_dtypes): prefix.append("typedef unsigned short hip_bfloat16;")
    prefix += [self.render_vector_prefix(dt) for dt in used_dtypes if dt.count > 1]

    for name, _, dtype_in, dtype_out, _, _, _, _ in wmma_args(uops): # TODO: handle TCs f32_bf16 and bf16_bf16 w/ wrapper
      if self.tensor_cores == tc.amd_cdna:
        prefix.append(f"#define __{name} __builtin_amdgcn_mfma_f32_16x16x16{'f16' if dtype_in == dtypes.half else 'bf16_1k'}")
      # #define __WMMA_16_16_16_half_half __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12
      elif self.tensor_cores == tc.amd_rdna4:
        prefix.append(f"#define __{name} __builtin_amdgcn_wmma_{type_map[dtype_out]}_16x16x16_{type_map[dtype_in]}_w32_gfx12")
      elif dtype_out == dtypes.float:
        prefix.append(f"#define __{name} __builtin_amdgcn_wmma_f32_16x16x16_{'f16' if dtype_in == dtypes.half else 'bf16'}_w32")
      else: prefix.append(f"static inline __attribute__((device)) half8 __{name}"+"""(half16 a, half16 b, half8 c) {
  half16 c_frag = {}; half8 d; for (int n = 0; n < 8; n++) { c_frag[n*2] = c[n]; }
  c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c_frag, false);
  for (int n = 0; n < 8; n++) { d[n] = c_frag[n*2]; } return d;\n}""")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

class NVRenderer(CUDARenderer): device = "NV"
class HIPRenderer(AMDRenderer): device = "HIP"
class QCOMRenderer(OpenCLRenderer): device = "QCOM"
