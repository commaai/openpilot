from typing import List, Tuple, Optional
from tinygrad.dtype import DType, PtrDType, dtypes
from tinygrad.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite, extra_pm
from tinygrad.helpers import strip_parens
import math

# utility functions for handling packed load/store of < 4-byte data types: bool, char/uchar, short/ushort
unpack_map = {dtypes.bool: dtypes.int, dtypes.char: dtypes.int, dtypes.uchar: dtypes.uint32, dtypes.short: dtypes.int, dtypes.ushort: dtypes.uint32}

def sign_extend(val:UOp, sext_am:int):
  return (UOp.where((val >> (sext_am - 1)) > 0, UOp.const(dtypes.uint32, 0xffffffff) << sext_am, UOp.const(dtypes.uint32, 0)) \
        | val.bitcast(dtypes.uint32)).bitcast(dtypes.int)

# store for char: buf[idx/4] <- (var << (idx%4)*8))
def packed_store(bidx:UOp, var:UOp):
  unpacked_type = unpack_map[var.dtype]
  shift_am = (bidx.src[1].cast(dtypes.uint32)%UOp.const(dtypes.uint32, 4//var.dtype.itemsize))*UOp.const(dtypes.uint32, 8*var.dtype.itemsize)
  new_v = (var & (0xFF if var.dtype.itemsize == 1 else 0xFFFF)).cast(dtypes.uint32) << shift_am
  mask = (((0xFF if var.dtype.itemsize == 1 else 0xFFFF) << shift_am) ^ 0xFFFFFFFF).cast(unpacked_type)
  buf = UOp.load(UOp(Ops.INDEX, bidx.dtype, (bidx.src[0], bidx.src[1]//(4//var.dtype.itemsize))), dtype=unpacked_type)
  return UOp.store(UOp(Ops.INDEX, bidx.dtype, (bidx.src[0], bidx.src[1]//(4//var.dtype.itemsize))), ((buf & mask) | new_v.cast(unpacked_type)))

# load for char: sign_extend(buf[idx/4] >> ((idx%4)*8))
def packed_load(root:UOp, bidx:UOp, dtype:DType, var:Optional[UOp]=None):
  div_idx = bidx.src[1]//(4//dtype.itemsize)
  shift_am = (bidx.src[1].cast(dtypes.uint32)%UOp.const(dtypes.uint32, 4//dtype.itemsize))*UOp.const(dtypes.uint32, 8*dtype.itemsize)
  if var is not None: load = UOp.load(UOp(Ops.INDEX, bidx.dtype, (bidx.src[0], div_idx)), var, root.src[2], dtype=unpack_map[dtype], arg=root.arg)
  else: load = UOp.load(UOp(Ops.INDEX, bidx.dtype, (bidx.src[0], div_idx)), *root.src[1:], dtype=unpack_map[dtype], arg=root.arg)
  val = (load.cast(dtypes.uint32) >> shift_am) & (0xFF if dtype.itemsize == 1 else 0xFFFF)
  return sign_extend(val, 8*dtype.itemsize).cast(dtype) if dtype in [dtypes.char, dtypes.short] else val.cast(dtype)

wgsl_matcher = PatternMatcher([
  (UPat((Ops.CMPLT, Ops.XOR), src=(UPat(name="a", dtype=dtypes.bool), UPat(name="b")), name="c"),
   lambda a,b,c: a.cast(dtypes.int).alu(c.op, b.cast(dtypes.int)).cast(dtypes.bool)),
  (UPat(Ops.LOAD, name="l", src=(UPat.var('b'),)), lambda l,b: packed_load(l,b,l.dtype) if l.dtype.itemsize < 4 else None),
  (UPat(Ops.LOAD, name="l", src=(UPat.var('b'), UPat.var('c'), UPat())),
   lambda l,b,c: packed_load(l,b,l.dtype,c.cast(unpack_map[l.dtype])) if l.dtype.itemsize < 4 else None),
  (UPat.store(UPat.var("bidx"), UPat.var("var")), lambda bidx,var: packed_store(bidx,var) if var.dtype.itemsize < 4 else None),
  # TODO: why is this needed, and only for this MUL order
  (UPat(Ops.MUL, src=(UPat.var("a"), UPat.var("g").where(UPat.cvar("c1"), UPat.cvar("c2")))),
    lambda a,g,c1,c2: g.where(c1, a) if math.isnan(c1.arg) and c2.arg == 1.0 else None),
  ]) + extra_pm

type_map = { dtypes.float: "f32", dtypes.uchar: "u32", dtypes.ushort: "u32", dtypes.short: "i32",
            dtypes.char: "i32", dtypes.int32: "i32", dtypes.uint32: "u32", dtypes.bool: "bool" }
buffer_map = { **type_map, dtypes.bool: "i32" }

class WGSLRenderer(CStyleLanguage):
  device = "WEBGPU"
  global_max = (65535, 65535, 65535)
  local_max = (256, 256, 64)
  code_for_workitem = {"g": lambda x: f"i32(gindex.{'xyz'[int(x)]})", "l": lambda x: f"i32(lindex.{'xyz'[int(x)]})"}
  extra_matcher = wgsl_matcher
  supports_float4 = False
  barrier = "workgroupBarrier();"
  code_for_op = {**CStyleLanguage.code_for_op, Ops.WHERE: lambda a,b,c,dtype: f"select({c},{b},{a})"}
  nan = "nan()"
  type_map = type_map

  string_rewrite = PatternMatcher([
    (UPat(Ops.CONST, dtype=dtypes.bool, name="x"), lambda ctx,x: "true" if x.arg else "false"),
    (UPat(Ops.CONST, dtype=(dtypes.uchar, dtypes.ushort, dtypes.uint32), name="x"), lambda ctx,x: f"bitcast<u32>({x.arg})" \
     if x.arg < 0 else f"{x.arg&0xFFFFFFFF}u"),
    (UPat(Ops.CONST, dtype=dtypes.int32, name="x"), lambda ctx,x: f"bitcast<i32>({x.arg}u)" if x.arg >= 0x80000000 else f"{x.arg}"),
    (UPat(Ops.DEFINE_LOCAL, name="x"), lambda ctx,x: f"var<workgroup> {ctx[x]}: array<{ctx.render_buf_dt(x.dtype.base)}, {x.arg[1]}>;"),
    (UPat(Ops.BITCAST, dtype=(dtypes.char, dtypes.uchar), name="x"), lambda ctx,x: f"bitcast<{type_map[x.dtype]}>({ctx[x.src[0]]}&0xFF)"),
    (UPat(Ops.BITCAST, dtype=(dtypes.short, dtypes.ushort), name="x"), lambda ctx,x: f"bitcast<{type_map[x.dtype]}>({ctx[x.src[0]]}&0xFFFF)"),
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"bitcast<{type_map[x.dtype]}>({ctx[x.src[0]]})"),
    (UPat(Ops.LOAD, src=(UPat.var("b"), UPat.var('v'), UPat.var("g"))), \
     lambda ctx,b,v,g: f"select({ctx[v]}, {ctx.render_load(ctx[b], b.src[0].dtype)}, {ctx[g]})"),
    (UPat(Ops.LOAD, src=(UPat.var('b'),), allow_any_len=True), lambda ctx, b: ctx.render_load(ctx[b], b.src[0].dtype)),
    (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var('idx'))),
    lambda ctx,buf,idx: f"{ctx[buf]}[{strip_parens(ctx[idx]) if idx.arg == Ops.ADD else ctx[idx]}]"),
    (UPat(Ops.STORE, src=(UPat.var('b'), UPat.var("v"))),lambda ctx,b,v:\
     # (load & mask) | var -> mask = v.src[0].src[1], var = v.src[1]
     f"atomicAnd(&{ctx[b]},{ctx[v.src[0].src[1]]});\n  atomicAdd(&{ctx[b]},{ctx[v.src[1]]});" if b.src[0].dtype.itemsize < 4 \
      else f"{ctx[b]} = {ctx[v]};"),
    # fix nan check: 'a != a -> is_nan()'
    (UPat.var("a") != UPat.var("a"), lambda ctx,a: f"is_nan({ctx[a]})"),
  ]) + base_rewrite

  def render_cast(self, dt:DType, val: str) -> str: return f"{self.type_map[dt]}({val})"
  def render_dtype(self, dt:DType, mutable=True) -> str: return "var"
  def render_load(self, x:str, dt:DType) -> str: return f"atomicLoad(&{x})" if dt.itemsize < 4 else x
  def render_buf_dt(self, dt:DType, rw=True) -> str: return f"{f'atomic<{buffer_map[dt]}>' if dt.itemsize < 4 else buffer_map[dt.base]}"
  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:List[UOp], prefix=None) -> str:
    local_size = [num for _, num in sorted([u.arg for u in uops if u.op is Ops.SPECIAL and u.arg[0][0] == 'l'], key=lambda x: x[0])]
    if not local_size: local_size = [1]
    bind_it = iter(range(len(bufs)))
    external_local_bufs = [line.lstrip() for line in kernel if "var<workgroup>" in line]
    kernel[:] = [line for line in kernel if "var<workgroup>" not in line]
    prg = "fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }\n"
    # trick to obfuscate compiler so that nan is detected properly
    prg += "fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }\n"
    prg += "@group(0) @binding(0)\nvar<uniform> INFINITY : f32;\n"
    prg += "\n".join((external_local_bufs or [])+[f"@group(0) @binding({next(bind_it)+1})" +
      f"{'var<storage,read_write>' if isinstance(dtype, PtrDType) else 'var<uniform>'}" +
      f"{name}:{f'array<{self.render_buf_dt(dtype.base,rw)}>' if isinstance(dtype, PtrDType) else buffer_map[dtype]};" for name,(dtype,rw) in bufs])
    prg += f"\n@compute @workgroup_size({','.join([str(x) for x in local_size])}) fn {function_name}(@builtin(workgroup_id) gindex: vec3<u32>,"
    return prg + "@builtin(local_invocation_id) lindex: vec3<u32>) {\n" + "\n".join(kernel) + "\n}"
