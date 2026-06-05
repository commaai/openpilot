from tinygrad.dtype import DType, PtrDType, dtypes, truncate, AddrSpace
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite, extra_pm
from tinygrad.helpers import strip_parens

def _mask(dt:DType): return 0xFF if dt.itemsize == 1 else 0xFFFF

def sign_extend(val:UOp, sext_am:int):
  return (UOp.where((val >> (sext_am - 1)) > 0, UOp.const(dtypes.uint32, 0xffffffff) << sext_am, UOp.const(dtypes.uint32, 0)) \
        | val.bitcast(dtypes.uint32)).bitcast(dtypes.int)

# store for char: buf[idx/4] <- (var << (idx%4)*8))
def packed_store(bidx:UOp, var:UOp, gate:UOp|None=None):
  elems, mask = 4//var.dtype.itemsize, _mask(var.dtype)
  shift_am, div_idx = (bidx.src[1].cast(dtypes.uint32) % elems) * (8*var.dtype.itemsize), bidx.src[1] // elems
  new_v, wmask = (var & mask).cast(dtypes.uint32) << shift_am, ((mask << shift_am) ^ 0xFFFFFFFF).cast(dtypes.uint32)
  idx = UOp(Ops.INDEX, bidx.dtype, (bidx.src[0], div_idx))
  buf = UOp.load(idx, *((UOp.const(dtypes.uint32, 0), gate) if gate is not None else ()), dtype=dtypes.uint32)
  return UOp.store(idx, (buf & wmask) | new_v, *((gate,) if gate is not None else ()))

# load for char: sign_extend(buf[idx/4] >> ((idx%4)*8))
def packed_load(root:UOp, bidx:UOp, dtype:DType, var:UOp|None=None, gate:UOp|None=None):
  elems, mask = 4//dtype.itemsize, _mask(dtype)
  shift_am, div_idx = (bidx.src[1].cast(dtypes.uint32) % elems) * (8*dtype.itemsize), bidx.src[1] // elems
  idx = UOp(Ops.INDEX, bidx.dtype, (bidx.src[0], div_idx))
  load = UOp.load(idx, *((var, gate) if var is not None and gate is not None else root.src[1:]), dtype=dtypes.uint32, arg=root.arg)
  val = (load.cast(dtypes.uint32) >> shift_am) & mask
  return sign_extend(val, 8*dtype.itemsize).cast(dtype) if dtype in [dtypes.char, dtypes.short] else val.cast(dtype)

def is_packed(dt:DType, odt:DType|None = None) -> bool:
  if odt is None: odt = dt
  return dt.itemsize < 4 and dt.base != dtypes.half and (not isinstance(odt, PtrDType) or odt.addrspace != AddrSpace.REG)
def _packed_size(dt:PtrDType): return dt.size // (4//dt.itemsize) if is_packed(dt) else dt.size

def is_nan(a):
  bs, (exp, mant) = a.dtype.bitsize, dtypes.finfo(a.dtype)
  return (a.bitcast(getattr(dtypes, f"uint{bs}")) & ((1 << (bs - 1)) - 1)) > (((1 << exp) - 1) << mant)

wgsl_matcher = PatternMatcher([
  (UPat((Ops.CMPLT, Ops.XOR), src=(UPat(name="a", dtype=dtypes.bool), UPat.var("b")), name="c"),
   lambda a,b,c: a.cast(dtypes.int).alu(c.op, b.cast(dtypes.int)).cast(dtypes.bool)),
  # TODO: load alt value doesnt have to be a const
  (UPat.load(UPat.var("b"), UPat.cvar("c"), UPat.var("gate"), name="l"),
   lambda l,b,c,gate: packed_load(l,b,l.dtype,c.cast(dtypes.uint32),gate) if is_packed(l.dtype, b.dtype) else None),
  (UPat.load(UPat.var("b"), name='l'), lambda l,b: packed_load(l, b, l.dtype) if is_packed(l.dtype, b.dtype) else None),
  (UPat.store(UPat.var("bidx"), UPat.var("var"), UPat.var("gate")),
   lambda bidx,var,gate: packed_store(bidx,var,gate) if is_packed(var.dtype, bidx.dtype) else None),
  (UPat.store(UPat.var("bidx"), UPat.var("var")),
   lambda bidx,var: packed_store(bidx,var) if is_packed(var.dtype, bidx.dtype) else None),
  (UPat.var("a") << UPat.var("b"),lambda a,b:(a.bitcast(dtypes.uint32)<<b.cast(dtypes.uint32)).bitcast(a.dtype) if b.dtype!=dtypes.uint32 else None),
  (UPat.var("x") >> UPat.var("y"), lambda x,y: UOp(Ops.SHR, x.dtype, (x,y.cast(dtypes.uint))) if y.dtype != dtypes.uint else None),
  # fix nan check: 'a != a -> is_nan()'
  (UPat.var("a") != UPat.var("a"), is_nan),
  ]) + extra_pm

class WGSLRenderer(CStyleLanguage):
  global_max = (65535, 65535, 65535)
  local_max = (256, 256, 64)
  code_for_workitem = {"g": lambda x: f"i32(gindex.{'xyz'[int(x)]})", "l": lambda x: f"i32(lindex.{'xyz'[int(x)]})"}
  extra_matcher = wgsl_matcher
  supports_float4 = False
  barrier = "workgroupBarrier();"
  code_for_op = {**CStyleLanguage.code_for_op, Ops.WHERE: lambda a,b,c,dtype: f"select({c},{b},{a})"}
  nan = "nan()"
  type_map = { dtypes.float: "f32", dtypes.uchar: "u32", dtypes.ushort: "u32", dtypes.short: "i32",
              dtypes.char: "i32", dtypes.int32: "i32", dtypes.uint32: "u32", dtypes.bool: "bool", dtypes.half: "f16" }

  string_rewrite = PatternMatcher([
    (UPat(Ops.NEG, dtypes.uints, src=(UPat.var('x'))), lambda ctx,x: f"(0-{ctx[x]})"),
    (UPat.cvar("x", dtype=dtypes.bool), lambda x: "true" if x.arg else "false"),
    (UPat(Ops.CONST, dtype=(dtypes.uchar, dtypes.ushort, dtypes.uint32), name="x"),
     lambda x: f"bitcast<u32>({x.arg})" if x.arg < 0 else f"{x.arg&0xFFFFFFFF}u"),
    (UPat(Ops.CONST, dtype=dtypes.int32, name="x"), lambda ctx,x: f"{truncate[x.dtype](x.arg)}"),
    (UPat(Ops.DEFINE_LOCAL, name="x"), lambda ctx,x: f"var<workgroup> {ctx[x]}: array<{ctx.buf_map(x.dtype.base)},{_packed_size(x.dtype)}>;"),
    (UPat(Ops.DEFINE_REG, name="x"), lambda ctx,x: f"var {ctx[x]}: array<{ctx.buf_map(x.dtype)},{_packed_size(x.dtype)}>;"),
    (UPat(Ops.BITCAST, dtype=dtypes.half, name="x", src=(UPat(dtype=(dtypes.short, dtypes.ushort, dtypes.uint32),),)),
     lambda ctx,x: f"bitcast<vec2<f16>>({ctx[x.src[0]]})[0]"),
    (UPat(Ops.BITCAST, dtype=dtypes.uchar, name="x"), lambda ctx,x: f"bitcast<u32>({ctx[x.src[0]]}&0xFF)"),
    (UPat(Ops.BITCAST, dtype=dtypes.char, name="x"), lambda ctx,x: f"((i32({ctx[x.src[0]]}&0xFF)<<24)>>24)"),
    (UPat(Ops.BITCAST, dtype=dtypes.ushort, name="x"), lambda ctx,x: f"bitcast<u32>(vec2<f16>({ctx[x.src[0]]},0))" \
     if x.src[0].dtype == dtypes.half else f"bitcast<u32>({ctx[x.src[0]]}&0xFFFF)"),
    (UPat(Ops.BITCAST, dtype=dtypes.short, name="x"), lambda ctx,x: f"bitcast<i32>(vec2<f16>({ctx[x.src[0]]},0))" \
     if x.src[0].dtype == dtypes.half else f"((i32({ctx[x.src[0]]}&0xFFFF)<<16)>>16)"),
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"bitcast<{ctx.type_map[x.dtype]}>({ctx[x.src[0]]})"),
    # TODO: load alt value doesnt have to be a const
    (UPat.load(UPat.var("b"), UPat.cvar("v"), UPat.var("gate")),
      lambda ctx,b,v,gate: f"select({ctx[v]}, {ctx.render_load(ctx[b],b.src[0].dtype)}, {ctx[gate]})"),
    (UPat.load(UPat.var("b")), lambda ctx, b: ctx.render_load(ctx[b], b.dtype)),
    (UPat.store(UPat.var("b"), UPat.var("v"), allow_any_len=True),lambda ctx,b,v:\
     # (load & mask) | var -> mask = v.src[0].src[1], var = v.src[1]
     f"atomicAnd(&{ctx[b]},{ctx[v.src[0].src[1]]});\n  atomicAdd(&{ctx[b]},{ctx[v.src[1]]});" if is_packed(b.src[0].dtype) \
      else f"{ctx[b]} = {ctx[v]};"),
    (UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx"))),
     lambda ctx,b,idx: f"{ctx[b]}[{strip_parens(ctx[idx]) if idx.arg is Ops.ADD else ctx[idx]}]"),
  ]) + base_rewrite

  def render_cast(self, dt:DType, val: str) -> str: return f"{self.type_map[dt]}({val})"
  def render_dtype(self, dt:DType, mutable=True) -> str: return "var"
  def render_load(self, x:str, dt:DType) -> str: return f"atomicLoad(&{x})" if is_packed(dt) else x
  def buf_map(self, dt:DType) -> str: return "atomic<u32>" if is_packed(dt) else self.type_map[dt.base]
  def render_kernel(self, function_name:str, kernel:list[str], bufs:list[tuple[str,tuple[DType,bool]]], uops:list[UOp], prefix=None) -> str:
    local_size = [u.src[0].ssimplify() for u in sorted([u for u in uops if u.op is Ops.SPECIAL and u.arg[0] == 'l'], key=lambda u: u.arg)]
    if not local_size: local_size = [1]
    bind_it = iter(range(len(bufs)))
    external_local_bufs = [line.lstrip() for line in kernel if "var<workgroup>" in line]
    kernel[:] = [line for line in kernel if "var<workgroup>" not in line]
    prg = "enable f16;\n" if any(uop.dtype.base == dtypes.half for uop in uops) else ""
    prg += "fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }\n"
    prg += "@group(0) @binding(0)\nvar<uniform> INFINITY : f32;\n"
    prg += "\n".join((external_local_bufs or [])+[f"@group(0) @binding({next(bind_it)+1})" +
      f"{'var<storage,read_write>' if isinstance(dtype, PtrDType) else 'var<uniform>'}" +
      f"{name}:{f'array<{self.buf_map(dtype.base)}>' if isinstance(dtype,PtrDType) else self.buf_map(dtype)};" for name,(dtype,_) in bufs])
    prg += f"\n@compute @workgroup_size({','.join([str(x) for x in local_size])}) fn {function_name}(@builtin(workgroup_id) gindex: vec3<u32>,"
    return prg + "@builtin(local_invocation_id) lindex: vec3<u32>) {\n" + "\n".join(kernel) + "\n}"

  def supported_dtypes(self):
    return {dtypes.bool, dtypes.char, dtypes.uchar, dtypes.short, dtypes.ushort, dtypes.float, dtypes.int32, dtypes.uint32, dtypes.half}
