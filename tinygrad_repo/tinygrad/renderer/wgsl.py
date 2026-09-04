from tinygrad.dtype import DType, dtypes, truncate, AddrSpace
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite
from tinygrad.helpers import strip_parens, ceildiv

# a field of `width` bits sitting in the low bits of val: shift it up to the sign bit, then let the arithmetic shift fill
def sign_extend(val:UOp, width:int): return (val << (32-width)).bitcast(dtypes.int) >> (32-width)

# a packed field of dt: the word it lives in, its offset in that word, and its mask. width is 8*itemsize, bool is one bit in a byte
def packed_field(bidx:UOp, dt:DType) -> tuple[UOp, UOp, int]:
  elems, width = 4//dt.itemsize, 8*dt.itemsize
  return bidx.src[0].index(bidx.src[1] // elems), (bidx.src[1].cast(dtypes.uint32) % elems) * width, (1 << width)-1

# store for char: buf[idx/4] <- (var << (idx%4)*8))
def packed_store(s:UOp):
  bidx, var, *gate = s.src
  idx, shift_am, mask = packed_field(bidx, var.dtype)
  # bool does its mask math at int32: renderer rewrites run after weak dtypes are lowered, and bool & 0xFF would create a weakint const
  if var.dtype == dtypes.bool: var = var.cast(dtypes.int32)
  new_v, wmask = (var & mask).cast(dtypes.uint32) << shift_am, ((mask << shift_am) ^ 0xFFFFFFFF).cast(dtypes.uint32)
  buf = idx.cast(dtypes.uint32).load(*((UOp.const(0, dtypes.uint32), *gate) if gate else ()))
  return idx.store((buf & wmask) | new_v, *gate)

# load for char: sign_extend(buf[idx/4] >> ((idx%4)*8))
def packed_load(root:UOp):
  bidx, *alt = root.src
  idx, shift_am, mask = packed_field(bidx, dtype:=root.dtype)
  load = idx.cast(dtypes.uint32).load(*((alt[0].cast(dtypes.uint32), *alt[1:]) if alt else ()), arg=root.arg)
  val = (load >> shift_am) & mask
  return sign_extend(val, 8*dtype.itemsize).cast(dtype) if dtype in [dtypes.char, dtypes.short] else val.cast(dtype)

def is_packed(x:UOp):
  dt = x.src[1].dtype if x.op is Ops.STORE else x.buf_uop.dtype
  return dt.itemsize < 4 and dt != dtypes.half and x.buf_uop.addrspace != AddrSpace.REG
def _packed_size(u:UOp): return ceildiv(u.max_numel(), 4//u.dtype.itemsize) if is_packed(u) else u.max_numel()
def is_nan(a):
  bs, (exp, mant) = a.dtype.bitsize, dtypes.finfo(a.dtype)
  return (a.bitcast(getattr(dtypes, f"uint{bs}")) & ((1 << (bs - 1)) - 1)) > (((1 << exp) - 1) << mant)

# the read-modify-write packed_store emits: a load of the very index being stored to, masked (a gated store loads with 3 srcs)
packed_rmw = UPat(Ops.LOAD, src=(UPat(Ops.CAST, dtype=dtypes.uint32, src=(UPat.var("b"),)),), allow_any_len=True) & UPat.var("wmask")

wgsl_matcher = PatternMatcher([
  (UPat((Ops.CMPLT, Ops.XOR), src=(UPat(name="a", dtype=dtypes.bool), UPat.var("b")), name="c"),
   lambda a,b,c: a.cast(dtypes.int).alu(c.op, b.cast(dtypes.int)).cast(dtypes.bool)),
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX),), allow_any_len=True, name="l"), lambda l: packed_load(l) if is_packed(l) else None),
  (UPat(Ops.STORE, name="s"), lambda s: packed_store(s) if is_packed(s) else None),
  (UPat.var("a") << UPat.var("b"),lambda a,b:(a.bitcast(dtypes.uint32)<<b.cast(dtypes.uint32)).bitcast(a.dtype) if b.dtype!=dtypes.uint32 else None),
  (UPat.var("x") >> UPat.var("y"), lambda x,y: UOp(Ops.SHR, src=(x,y.cast(dtypes.uint))) if y.dtype != dtypes.uint else None),
  # fix nan check: 'a != a -> is_nan()'. the decomp rewrites (a != a).logical_not() to CMPEQ, so match both forms
  (UPat.var("a", dtypes.floats) != UPat.var("a"), is_nan),
  (UPat.var("a", dtypes.floats).alu(Ops.CMPEQ, UPat.var("a")), lambda a: is_nan(a).ne(True)),
  ])

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
    (UPat(Ops.CAST, dtype=dtypes.uint32, src=(UPat(Ops.INDEX, name="x"),)), lambda ctx,x: ctx[x] if is_packed(x) else None),
    (UPat(Ops.NEG, dtypes.uints, src=(UPat.var('x'))), lambda ctx,x: f"(0-{ctx[x]})"),
    (UPat.cvar("c").cast(dtypes.bool), lambda c: "true" if c.val else "false"),
    (UPat.cvar("c").cast((dtypes.uchar, dtypes.ushort, dtypes.uint32)),
     lambda c: f"bitcast<u32>({c.val})" if c.val < 0 else f"{c.val&0xFFFFFFFF}u"),
    # a negative const must state its type: contextual conversion of a bare abstract int rejects it in a u32 position
    (UPat.cvar("c").cast(dtypes.int32, name="x"), lambda ctx,x,c: f"i32({v})" if (v:=truncate[x.dtype](c.val)) < 0 else f"{v}"),
    (UPat(Ops.BUFFER, name="x"), lambda ctx,x:
     f"var{'<workgroup>' if x.addrspace == AddrSpace.LOCAL else ''} {ctx[x]}: array<{ctx.buf_map(x)},{_packed_size(x)}>;"),
    (UPat(Ops.BITCAST, dtype=dtypes.half, name="x", src=(UPat(dtype=(dtypes.short, dtypes.ushort, dtypes.uint32),),)),
     lambda ctx,x: f"bitcast<vec2<f16>>({ctx[x.src[0]]})[0]"),
    (UPat(Ops.BITCAST, dtype=dtypes.uchar, name="x"), lambda ctx,x: f"bitcast<u32>({ctx[x.src[0]]}&0xFF)"),
    (UPat(Ops.BITCAST, dtype=dtypes.char, name="x"), lambda ctx,x: f"((i32({ctx[x.src[0]]}&0xFF)<<24)>>24)"),
    (UPat(Ops.BITCAST, dtype=dtypes.ushort, name="x"), lambda ctx,x: f"bitcast<u32>(vec2<f16>({ctx[x.src[0]]},0))" \
     if x.src[0].dtype == dtypes.half else f"bitcast<u32>({ctx[x.src[0]]}&0xFFFF)"),
    (UPat(Ops.BITCAST, dtype=dtypes.short, name="x"), lambda ctx,x: f"bitcast<i32>(vec2<f16>({ctx[x.src[0]]},0))" \
     if x.src[0].dtype == dtypes.half else f"((i32({ctx[x.src[0]]}&0xFFFF)<<16)>>16)"),
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"bitcast<{ctx.type_map[x.dtype]}>({ctx[x.src[0]]})"),
    (UPat.load(UPat.var("b"), UPat.var("v"), UPat.var("gate")),
      lambda ctx,b,v,gate: f"select({ctx[v]}, {ctx.render_load(ctx[b], b.src[0])}, {ctx[gate]})"),
    (UPat.load(UPat.var("b")), lambda ctx, b: ctx.render_load(ctx[b], b)),
    # packed_store writes (load & wmask) | new_v: atomicAnd clears the field, atomicAdd sets it. new_v is gone when it is 0
    (UPat.store(UPat.var("b"), UPat.any(packed_rmw, packed_rmw | UPat.var("nv"))), lambda ctx,b,wmask,nv=None:
     f"atomicAnd(&{ctx[b]},{ctx[wmask]});"+(f"\n  atomicAdd(&{ctx[b]},{ctx[nv]});" if nv is not None else "") if is_packed(b) else None),
    (UPat.store(UPat.var("b"), UPat.var("v")), lambda ctx,b,v: f"{ctx[b]} = {ctx[v]};"),
    (UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx"))),
     lambda ctx,b,idx: f"{ctx[b]}[{strip_parens(ctx[idx]) if idx.arg is Ops.ADD else ctx[idx]}]"),
  ]) + base_rewrite

  def render_cast(self, u:UOp, val: str) -> str: return f"{self.type_map[u.dtype]}({val})"
  def _render_dtype(self, dtype:DType, sz:int=1, addrspace=AddrSpace.REG, mutable=True, override_ptr=False, shape=None): return "var"
  def render_load(self, x:str, u:UOp) -> str: return f"atomicLoad(&{x})" if is_packed(u) else x
  def buf_map(self, u:UOp) -> str: return "atomic<u32>" if is_packed(u) else self.type_map[u.dtype]
  def render_kernel(self, function_name:str, kernel:list[str], bufs:list[tuple[str,tuple[UOp,bool]]], uops:list[UOp], prefix=None) -> str:
    local_size = [u.src[0].ssimplify() for u in sorted([u for u in uops if u.op is Ops.SPECIAL and u.arg[0] == 'l'], key=lambda u: u.arg)]
    if not local_size: local_size = [1]
    bind_it = iter(range(len(bufs)))
    external_local_bufs = [line.lstrip() for line in kernel if "var<workgroup>" in line]
    kernel[:] = [line for line in kernel if "var<workgroup>" not in line]
    prg = "enable f16;\n" if any(uop.dtype == dtypes.half for uop in uops) else ""
    prg += "fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }\n"
    prg += "@group(0) @binding(0)\nvar<uniform> INFINITY : f32;\n"
    prg += "\n".join((external_local_bufs or [])+[f"@group(0) @binding({next(bind_it)+1})" +
      f"{'var<storage,read_write>' if u.addrspace == AddrSpace.GLOBAL else 'var<uniform>'}" +
      f"{name}:{f'array<{self.buf_map(u)}>' if u.addrspace == AddrSpace.GLOBAL else self.buf_map(u)};" for name,(u,_) in bufs])
    prg += f"\n@compute @workgroup_size({','.join([str(x) for x in local_size])}) fn {function_name}(@builtin(workgroup_id) gindex: vec3<u32>,"
    return prg + "@builtin(local_invocation_id) lindex: vec3<u32>) {\n" + "\n".join(kernel) + "\n}"

  def supported_dtypes(self): return {dtypes.bool, dtypes.char, dtypes.uchar, dtypes.short, dtypes.ushort, dtypes.int32, dtypes.uint32,
                                      dtypes.float, *((dtypes.half,) if "shader-f16" in self.target.arch else ())}
