from tinygrad.dtype import DType, PtrDType, dtypes, AddrSpace
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite, extra_pm
from tinygrad.helpers import strip_parens

def sign_extend(val:UOp, sext_am:int):
  return (UOp.where((val >> (sext_am - 1)) > 0, UOp.const(dtypes.uint32, 0xffffffff) << sext_am, UOp.const(dtypes.uint32, 0)) \
        | val.bitcast(dtypes.uint32)).bitcast(dtypes.int)

# store for char: buf[idx/4] <- (var << (idx%4)*8))
def packed_store(bidx:UOp, var:UOp):
  shift_am = (bidx.src[1].cast(dtypes.uint32)%UOp.const(dtypes.uint32, 4//var.dtype.itemsize))*UOp.const(dtypes.uint32, 8*var.dtype.itemsize)
  new_v = (var & (0xFF if var.dtype.itemsize == 1 else 0xFFFF)).cast(dtypes.uint32) << shift_am
  mask = (((0xFF if var.dtype.itemsize == 1 else 0xFFFF) << shift_am) ^ 0xFFFFFFFF).cast(dtypes.uint32)
  buf = UOp.load(UOp(Ops.INDEX, bidx.dtype, (bidx.src[0], bidx.src[1]//(4//var.dtype.itemsize))), dtype=dtypes.uint32)
  return UOp.store(UOp(Ops.INDEX, bidx.dtype, (bidx.src[0], bidx.src[1]//(4//var.dtype.itemsize))), ((buf & mask) | new_v.cast(dtypes.uint32)))

# load for char: sign_extend(buf[idx/4] >> ((idx%4)*8))
def packed_load(root:UOp, bidx:UOp, dtype:DType, var:UOp|None=None):
  div_idx = bidx.src[1]//(4//dtype.itemsize)
  shift_am = (bidx.src[1].cast(dtypes.uint32)%UOp.const(dtypes.uint32, 4//dtype.itemsize))*UOp.const(dtypes.uint32, 8*dtype.itemsize)
  if var is not None: load = UOp.load(UOp(Ops.INDEX, bidx.dtype, (bidx.src[0], div_idx, bidx.src[2])), var, dtype=dtypes.uint32, arg=root.arg)
  else: load = UOp.load(UOp(Ops.INDEX, bidx.dtype, (bidx.src[0], div_idx)), *root.src[1:], dtype=dtypes.uint32, arg=root.arg)
  val = (load.cast(dtypes.uint32) >> shift_am) & (0xFF if dtype.itemsize == 1 else 0xFFFF)
  return sign_extend(val, 8*dtype.itemsize).cast(dtype) if dtype in [dtypes.char, dtypes.short] else val.cast(dtype)

def is_packed(dt:DType, odt:DType|None = None) -> bool:
  if odt is None: odt = dt
  return dt.itemsize < 4 and dt.base != dtypes.half and (not isinstance(odt, PtrDType) or odt.addrspace != AddrSpace.REG)

wgsl_matcher = PatternMatcher([
  (UPat((Ops.CMPLT, Ops.XOR), src=(UPat(name="a", dtype=dtypes.bool), UPat.var("b")), name="c"),
   lambda a,b,c: a.cast(dtypes.int).alu(c.op, b.cast(dtypes.int)).cast(dtypes.bool)),
  (UPat.load(UPat.var("b"), UPat.cvar("c"), name="l"),
   lambda l,b,c: packed_load(l,b,l.dtype,c.cast(dtypes.uint32)) if is_packed(l.dtype, b.dtype) else None),
  (UPat.load(UPat.var("b"), name='l', allow_any_len=True), lambda l,b: packed_load(l, b, l.dtype) if is_packed(l.dtype, b.dtype) else None),
  (UPat.store(UPat.var("bidx"), UPat.var("var"), allow_any_len=True),
   lambda bidx,var: packed_store(bidx,var) if is_packed(var.dtype, bidx.dtype) else None),
  (UPat.var("a") << UPat.var("b"),lambda a,b:(a.bitcast(dtypes.uint32)<<b.cast(dtypes.uint32)).bitcast(a.dtype) if b.dtype!=dtypes.uint32 else None),
  (UPat.var("x") >> UPat.var("y"), lambda x,y: UOp(Ops.SHR, x.dtype, (x,y.cast(dtypes.uint))) if y.dtype != dtypes.uint else None),
  ]) + extra_pm

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
  type_map = { dtypes.float: "f32", dtypes.uchar: "u32", dtypes.ushort: "u32", dtypes.short: "i32",
              dtypes.char: "i32", dtypes.int32: "i32", dtypes.uint32: "u32", dtypes.bool: "bool", dtypes.half: "f16" }

  string_rewrite = PatternMatcher([
    (UPat.cvar("x", dtype=dtypes.bool), lambda x: "true" if x.arg else "false"),
    (UPat(Ops.CONST, dtype=(dtypes.uchar, dtypes.ushort, dtypes.uint32), name="x"),
     lambda x: f"bitcast<u32>({x.arg})" if x.arg < 0 else f"{x.arg&0xFFFFFFFF}u"),
    (UPat(Ops.DEFINE_LOCAL, name="x"), lambda ctx,x:
      f"var<workgroup> {ctx[x]}: array<{ctx.buf_map(x.dtype.base)},{x.dtype.size//(4//x.dtype.itemsize) if is_packed(x.dtype) else x.dtype.size}>;"),
    (UPat(Ops.DEFINE_REG, name="x"), lambda ctx,x:
     f"var {ctx[x]}: array<{ctx.buf_map(x.dtype)},{x.dtype.size//(4//x.dtype.itemsize) if is_packed(x.dtype) else x.dtype.size}>;"),
    (UPat(Ops.BITCAST, dtype=dtypes.half, name="x", src=(UPat(dtype=(dtypes.short, dtypes.ushort, dtypes.uint32),),)),
     lambda ctx,x: f"bitcast<vec2<f16>>({ctx[x.src[0]]})[0]"),
    (UPat(Ops.BITCAST, dtype=(dtypes.char, dtypes.uchar), name="x"), lambda ctx,x: f"bitcast<{ctx.type_map[x.dtype]}>({ctx[x.src[0]]}&0xFF)"),
    (UPat(Ops.BITCAST, dtype=(dtypes.short, dtypes.ushort), name="x"),lambda ctx,x:f"bitcast<{ctx.type_map[x.dtype]}>(vec2<f16>({ctx[x.src[0]]},0))" \
     if x.src[0].dtype == dtypes.half else f"bitcast<{ctx.type_map[x.dtype]}>({ctx[x.src[0]]}&0xFFFF)"),
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"bitcast<{ctx.type_map[x.dtype]}>({ctx[x.src[0]]})"),
    (UPat.load(UPat.var("b"), UPat.cvar("v")),lambda ctx,b,v: f"select({ctx[v]}, {ctx.render_load(ctx[b],b.src[0].dtype)}, {ctx[b.src[2]]})"),
    (UPat.load(UPat.var("b"), allow_any_len=True), lambda ctx, b: ctx.render_load(ctx[b], b.dtype)),
    (UPat.store(UPat.var("b"), UPat.var("v"), allow_any_len=True),lambda ctx,b,v:\
     # (load & mask) | var -> mask = v.src[0].src[1], var = v.src[1]
     f"atomicAnd(&{ctx[b]},{ctx[v.src[0].src[1]]});\n  atomicAdd(&{ctx[b]},{ctx[v.src[1]]});" if is_packed(b.src[0].dtype) \
      else f"{ctx[b]} = {ctx[v]};"),
    (UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx")), allow_any_len=True),
     lambda ctx,b,idx: f"{ctx[b]}[{strip_parens(ctx[idx]) if idx.arg is Ops.ADD else ctx[idx]}]"),
    # fix nan check: 'a != a -> is_nan()'
    (UPat.var("a") != UPat.var("a"), lambda ctx,a: f"(min({ctx[a]}, 1.0) == 1.0 && max({ctx[a]}, -1.0) == -1.0)"),
  ]) + base_rewrite

  def render_cast(self, dt:DType, val: str) -> str: return f"{self.type_map[dt]}({val})"
  def render_dtype(self, dt:DType, mutable=True) -> str: return "var"
  def render_load(self, x:str, dt:DType) -> str: return f"atomicLoad(&{x})" if is_packed(dt) else x
  def buf_map(self, dt:DType) -> str: return "atomic<u32>" if is_packed(dt) else self.type_map[dt.base]
  def render_kernel(self, function_name:str, kernel:list[str], bufs:list[tuple[str,tuple[DType,bool]]], uops:list[UOp], prefix=None) -> str:
    local_size = [num for _, num in sorted([u.arg for u in uops if u.op is Ops.SPECIAL and u.arg[0][0] == 'l'], key=lambda x: x[0])]
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
