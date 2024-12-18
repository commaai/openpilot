from typing import Dict, List, Optional, NamedTuple, Tuple, Union, DefaultDict
import math
from collections import defaultdict
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.helpers import ImageDType, dtypes, prod, DType, strip_parens

class CStyleLanguage(NamedTuple):
  size_prefix: str = "int"
  generic_var_prefix: str = ""
  kernel_prefix: str = ""
  buffer_prefix: str = ""
  buffer_suffix: str = ""
  smem_align: str = ""
  smem_prefix: str = ""
  smem_prefix_for_cast: bool = True
  arg_int_prefix: str = ""
  barrier: str = ""
  xid: List[str] = []
  gid: List[str] = []
  lid: List[str] = []
  global_max: List[int] = []
  local_max: List[int] = []
  extra_args: List[str] = []
  float4: Optional[str] = None
  half_prekernel: Optional[str] = None
  uses_vload: bool = False
  external_local_bufs: bool = False
  uses_ptr_arithmetic: bool = False
  launch_bounds: bool = False
  code_for_op: Dict = {
    UnaryOps.NEG: lambda x: f"(-{x})",
    UnaryOps.EXP2: lambda x: f"exp2({x})",
    UnaryOps.LOG2: lambda x: f"log2({x})",
    UnaryOps.SIN: lambda x: f"sin({x})",
    UnaryOps.SQRT: lambda x: f"sqrt({x})",
    BinaryOps.ADD: lambda a,b: f"({a}+{b})", BinaryOps.SUB: lambda a,b: f"({a}-{b})",
    BinaryOps.MUL: lambda a,b: f"({a}*{b})", BinaryOps.DIV: lambda a,b: f"({a}/{b})",
    BinaryOps.MAX: lambda a,b: f"max({a},{b})", BinaryOps.MOD: lambda a,b: f"({a}%{b})",
    BinaryOps.CMPLT: lambda a,b: f"({a}<{b})", TernaryOps.MULACC: lambda a,b,c: f"(({a}*{b})+{c})",
    TernaryOps.WHERE: lambda a,b,c: f"({a}!=0?{b}:{c})"
  }

  # returns a str expression of the casted xs with the given type
  def render_cast(self, x:List[str], var_dtype:DType) -> str:
    if len(x) == 1: return f"({var_dtype.name})({x[0]})"
    assert len(x) == var_dtype.sz, f"cast is wrong size {len(x)} != {var_dtype.sz}"
    assert self.float4 is not None, "cast is not supported on this platform"
    if var_dtype == dtypes._float4: return f"{self.float4}({','.join(x)})"
    if var_dtype == dtypes._float2: return f"{self.float4.replace('float4', 'float2')}({','.join(x)})"
    if var_dtype == dtypes._int2: return f"{self.float4.replace('float4', 'int2')}({','.join(x)})"
    raise NotImplementedError(f"no cast for {var_dtype}")

  # returns a str expression of the const with the given type
  def render_const(self, x:Union[float,int], var_dtype) -> str:
    if math.isnan(x): val = "NAN"
    elif math.isinf(x): val = ("-" if x < 0 else "") + "INFINITY"
    else: val = f"{x}f" if dtypes.is_float(var_dtype) and isinstance(x, float) else f"{int(x)}"
    return self.render_cast([val]*var_dtype.sz, var_dtype) if var_dtype.sz > 1 else val

  # returns a str expression of the loaded value with the output type
  def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert output_dtype == dtypes._float4, f"images must be float4, getting {output_dtype}"
      return f"read_imagef({buf_name}, smp, {idx})"
    if self.uses_vload and buf_dtype == dtypes.float16:
      return f"vload_half{'' if output_dtype.sz == 1 else str(output_dtype.sz)}(0, {buf_name}+{idx})"
    if output_dtype.sz > 1:
      out_val = f"*(({self.smem_prefix if local and self.smem_prefix_for_cast else self.buffer_prefix}{buf_dtype.name}{output_dtype.sz}*)({buf_name}+{idx}))"
    else:
      out_val = f"*({buf_name}+{idx})" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}]"

    return self.render_cast([out_val], output_dtype) if output_dtype != buf_dtype else out_val

  def render_local(self, name:str, size:int):
    return self.smem_align + self.smem_prefix + f"float {name}[{size}];"

  def render_for(self, expr: str, _min:Union[int,str], _max:Union[int,str]) -> str:
    return f"for (int {expr} = {_min}; {expr} < {_max}; ++{expr}) {{"

  def render_if(self, cond: str):
    return f"if ({cond}) {{"

  def render_conditional(self, cond: str, x:str, y:str) -> str:
    return f"({cond})?({x}):{y}"

  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,DType]], local_size:List[int], prekernel:List[str]) -> str:
    tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" if any(isinstance(dtype, ImageDType) for _,dtype in bufs) else ""
    buftypes = [(name,f"{'read_only' if i > 0 else 'write_only'} image2d_t" if dtype.name.startswith('image') else
                self.arg_int_prefix if dtype == dtypes._arg_int32 else
                ("const " if i > 0 else "")+self.buffer_prefix+dtype.name+"*"+self.buffer_suffix) for i,(name,dtype) in enumerate(bufs)]
    prg = ''.join([f"{self.kernel_prefix}void {f'__launch_bounds__ ({prod(local_size)}, 1) ' if self.launch_bounds else ''}{function_name}(",] +
    [', '.join([f'{t} {name}' for name,t in buftypes] + self.extra_args)] +
    [") {\n" + tmp] + ['\n'.join(kernel), "\n}"])
    if self.half_prekernel and any(dtype == dtypes.float16 for _,dtype in bufs): prg = ''.join([f"{self.half_prekernel}", "\n", prg])
    return prg

  # returns a str statement that does the store
  def render_store(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType, idx:str, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert var_dtype == dtypes._float4, "images must be float4"
      return f"write_imagef({buf_name}, {idx}, {var_name});"
    if self.uses_vload and buf_dtype == dtypes.float16:
      return f"vstore_half{'' if var_dtype.sz == 1 else str(var_dtype.sz)}({var_name}, 0, {buf_name}+{idx});"
    if var_dtype.sz > 1:
      return f"*(({self.smem_prefix if local and self.smem_prefix_for_cast else self.buffer_prefix}{buf_dtype.name}{var_dtype.sz}*)({buf_name}+{idx})) = ({buf_dtype.name}{var_dtype.sz}){var_name};"
    return f"*({buf_name}+{idx}) = {var_name};" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}] = {var_name};"

def uops_to_cstyle(lang:CStyleLanguage, function_name:str, uops:List[UOp]) -> Tuple[str, Dict]:
  local_size: List[int] = []
  kernel,prekernel,bufs = [],[],[]
  #pend_close = None
  depth = 1
  def kk(s): kernel.append("  "*depth+s)

  c: DefaultDict[str, int] = defaultdict(int)
  r: Dict[UOp, str] = {}
  def ssa(u, prefix="t"):
    nonlocal c, r
    c[prefix] += 1
    r[u]=f"{prefix}{c[prefix]-1}"
    return r[u]

  child_count: DefaultDict[UOp, int] = defaultdict(int)
  for ru in uops:
    for v in ru.vin:
      child_count[v] += 1

  for u in uops:
    uop,dtype,vin,args,_ = u
    if uop == UOps.LOOP:
      kk(lang.render_for(ssa(u,'ridx'), r[vin[0]], r[vin[1]]))
      depth += 1
    elif uop == UOps.IF:
      kk(lang.render_if(r[vin[0]]))
      depth += 1
    elif uop == UOps.BARRIER:
      kk(lang.barrier)
    elif uop == UOps.END:
      depth -= 1
      kk("}")
    elif uop == UOps.WMMA:
      if args[0] == "METAL":
        # ((lidx2*32)+(lidx3*4)+(lidx4*16)+(lidx5*8)+(lidx6*2))
        kk("{ simdgroup_float8x8 a,b,c;")
        kk(f"a.thread_elements()[0] = {r[vin[0]]}; a.thread_elements()[1] = {r[vin[1]]};")
        kk(f"b.thread_elements()[0] = {r[vin[2]]}; b.thread_elements()[1] = {r[vin[3]]};")
        kk(f"c.thread_elements()[0] = {r[vin[4]]}; c.thread_elements()[1] = {r[vin[5]]};")
        kk("simdgroup_multiply_accumulate(c, a, b, c);")
        kk(f"{r[vin[4]]} = c.thread_elements()[0]; {r[vin[5]]} = c.thread_elements()[1]; }}")
      elif args[0] == "HIP":
        kk("{")
        kk(f"half16 a_frag = {{ {','.join(['(half)'+r[x] for x in vin[0:16]])} }};")
        kk(f"half16 b_frag = {{ {','.join(['(half)'+r[x] for x in vin[16:32]])} }};")
        kk(f"float8 c_frag = {{ {','.join([r[x] for x in vin[32:]])} }};")
        kk("c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);")
        for i in range(8): kk(f"{r[vin[32+i]]} = c_frag[{i}];")
        kk("}")
      else:
        raise NotImplementedError(f"WMMA not implemented for {args}")
    elif uop == UOps.ALU:
      assert dtype is not None
      # remove parens if ALU types are the same. TODO: can do more here
      if vin[0].uop == UOps.ALU and vin[0].arg == args and args in {BinaryOps.ADD, BinaryOps.SUB, BinaryOps.MUL}:
        val = lang.code_for_op[args](strip_parens(r[vin[0]]), *[r[x] for x in vin[1:]])
      else:
        val = lang.code_for_op[args](*[r[x] for x in vin])
      assert child_count[u] != 0, f"childless ALU op found {u}"
      if child_count[u] <= 1 or dtypes.is_int(dtype):  # fix index rendering issue
        r[u] = val
      else:
        kk(f"{lang.generic_var_prefix if lang.generic_var_prefix else dtype.name} {ssa(u,'alu')} = {val};")
    elif uop == UOps.DEFINE_ACC:
      assert dtype is not None
      kk(f"{lang.generic_var_prefix if lang.generic_var_prefix else dtype.name} {ssa(u,'acc')} = {lang.render_const(args, dtype)};")
    elif uop == UOps.SPECIAL:
      xid = lang.gid if args[1].startswith("g") else (lang.xid if args[1].startswith("i") else lang.lid)
      kk(f"{lang.size_prefix} {args[1]} = {xid[args[0]]}; /* {args[2]} */")
      if args[1].startswith("l"): local_size.append(args[2])
      r[u] = args[1]
    elif uop == UOps.CONST:
      r[u] = lang.render_const(args, dtype) if args >= 0 else f"({lang.render_const(args, dtype)})"
    elif uop == UOps.LOAD:
      assert dtype is not None
      val = lang.render_load(dtype, r[vin[0]], vin[0].dtype, strip_parens(r[vin[1]]), vin[0].uop == UOps.DEFINE_LOCAL)
      if len(vin) > 2: val = lang.render_conditional(r[vin[2]], val, r[vin[3]])
      kk(f"{lang.generic_var_prefix if lang.generic_var_prefix else dtype.name} {ssa(u,'val')} = {val};")
    elif uop == UOps.PHI:
      kk(f"{r[vin[0]]} = {r[vin[1]]};")
      r[u] = r[vin[0]]
    elif uop == UOps.STORE:
      assert vin[0].dtype is not None and vin[2].dtype is not None
      kk(lang.render_store(r[vin[0]], vin[0].dtype, r[vin[2]], vin[2].dtype, strip_parens(r[vin[1]]), vin[0].uop == UOps.DEFINE_LOCAL))
    elif uop == UOps.CAST and dtype is not None and dtype.sz > 1:
      val = lang.render_cast([r[x] for x in vin], dtype)
      if child_count[u] <= 1: r[u] = val
      else: kk(f"{lang.generic_var_prefix if lang.generic_var_prefix else dtype.name} {ssa(u,'cast')} = {val};")
    elif uop == UOps.DEFINE_LOCAL:
      if lang.external_local_bufs:
        prekernel.append(lang.render_local(args[0], args[1]))
      else:
        kk(lang.render_local(args[0], args[1]))
      r[u] = args[0]
    elif uop == UOps.DEFINE_GLOBAL:
      bufs.append(args)
      r[u] = args[0]
    elif uop == UOps.GEP:
      r[u] = f"({r[vin[0]]}).{'xyzw'[args]}"
    else:
      raise RuntimeError(f"failed to render {uop}")

  return lang.render_kernel(function_name, kernel, bufs, local_size, prekernel), {}
