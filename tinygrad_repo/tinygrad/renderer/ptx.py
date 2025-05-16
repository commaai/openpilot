from typing import cast, Callable
import struct
from collections import defaultdict
from tinygrad.ops import Ops, UOp, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.helpers import flatten, get_single_element

def render_val(x, dtype):
  if dtypes.is_float(dtype):
    if dtype == dtypes.double: return "0d%02X%02X%02X%02X%02X%02X%02X%02X" % tuple(struct.pack("d",x)[::-1])
    if dtype == dtypes.half: return "0x%02X%02X" % tuple(struct.pack("e",x)[::-1])
    return "0f%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
  return str(int(x)) + ("U" if dtypes.is_unsigned(dtype) else "")

asm_for_op: dict[Ops, Callable] = {
  Ops.RECIP: lambda d,a,dt,name: f"rcp{'.approx' if dtypes.is_float(dt) else ''}.{name} {d}, {a};",
  Ops.EXP2: lambda d,a,dt,name: f"ex2.approx.{name} {d}, {a};", Ops.LOG2: lambda d,a,dt,name: f"lg2.approx.{name} {d}, {a};",
  Ops.SIN: lambda d,a,dt,name: f"sin.approx.{name} {d}, {a};", Ops.SQRT: lambda d,a,dt,name: f"sqrt.approx.{name} {d}, {a};",
  Ops.SHR: lambda d,a,b,dt,name: f"shr.{name} {d}, {a}, {b};", Ops.SHL: lambda d,a,b,dt,name: f"shl.b{name[1:]} {d}, {a}, {b};",
  Ops.ADD: lambda d,a,b,dt,name: f"{'or' if dt == dtypes.bool else 'add'}.{name} {d}, {a}, {b};",
  Ops.MUL: lambda d,a,b,dt,name: f"{'and' if dt == dtypes.bool else 'mul'}{'.lo' if dtypes.is_int(dt) else ''}.{name} {d}, {a}, {b};",
  Ops.XOR: lambda d,a,b,dt,name: f"xor.pred {d}, {a}, {b};" if dt == dtypes.bool else f"xor.b{name[1:]} {d}, {a}, {b};",
  Ops.AND: lambda d,a,b,dt, name: f"and.pred {d}, {a}, {b};" if dt == dtypes.bool else f"and.b{name[1:]} {d}, {a}, {b};",
  Ops.OR: lambda d,a,b,dt, name: f"or.pred {d}, {a}, {b};" if dt == dtypes.bool else f"or.b{name[1:]} {d}, {a}, {b};",
  Ops.IDIV: lambda d,a,b,dt,name: f"div.{name} {d}, {a}, {b};", Ops.MOD: lambda d,a,b,dt,name: f"rem.{name} {d}, {a}, {b};",
  Ops.MAX: lambda d,a,b,dt,name: f"max.{name} {d}, {a}, {b};",
  Ops.CMPLT: lambda d,a,b,dt,name: f"setp.lt.{name} {d}, {a}, {b};", Ops.CMPNE: lambda d,a,b,dt,name: f"setp.ne.{name} {d}, {a}, {b};",
  Ops.MULACC: lambda d,a,b,c,dt,name: f"{'fma.rn' if dtypes.is_float(dt) else 'mad.lo'}.{name} {d}, {a}, {b}, {c};",
  Ops.WHERE: lambda d,a,b,c,dt,name: [f"@{a} mov.{name} {d}, {b};", f"@!{a} mov.{name} {d}, {c};"] if dt == dtypes.bool else \
    f"selp.{'b16' if name == 'f16' else name} {d}, {b}, {c}, {a};"
}

supports_half = (Ops.EXP2, Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPLT, Ops.WHERE)
doesnt_support_half: tuple[Ops, ...] = tuple(op for op in asm_for_op.keys() if op not in supports_half)
ptx_matcher = PatternMatcher([
  # bool CMPNE is XOR, bool CMPLT is XOR+AND (universal makes this slow, this is for renderer only)
  (UPat.var('x', dtype=dtypes.bool).ne(UPat.var('y')), lambda x,y: x^y),
  (UPat.var('x', dtype=dtypes.bool)<UPat.var('y'), lambda x,y: (x^True)&y),
  # upcast to float32 all the ops that don't support half
  (UPat(doesnt_support_half, dtype=dtypes.half, name="x"),
    lambda x: (UOp(x.op, dtypes.float32, tuple(vv.cast(dtypes.float32) for vv in x.src), x.arg).cast(dtypes.half))),
  # load/store bool -> uint8
  (UPat(Ops.LOAD, dtypes.bool, src=(UPat(dtype=dtypes.int64),), name="x", allow_any_len=True),
   lambda x: UOp(x.op, dtypes.uint8, x.src[0:1] + ((x.src[1].cast(dtypes.uint8),) if len(x.src) >= 2 else ()) + x.src[2:]).cast(dtypes.bool)),
  (UPat(Ops.STORE, src=(UPat(dtype=dtypes.int64), UPat(dtype=dtypes.bool)), name="x", allow_any_len=True),
   lambda x: UOp(x.op, dtypes.void, x.src[0:1] + (x.src[1].cast(dtypes.uint8),) + x.src[2:])),
  # load/store use pointer arithmetic, and the cast does nothing
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx"))), lambda buf,idx: buf.cast(dtypes.int64) + idx.cast(dtypes.int64)*buf.dtype.itemsize),
  (UPat(Ops.CAST, name="x"), lambda x: x.src[0] if isinstance(x.dtype, PtrDType) or x.src[0].dtype == dtypes.void else None),
  # move mask from INDEX to the load/store to enable pointer arithmetic
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx"), UPat.var("gate"))), UPat.var("alt"))),
   lambda buf,idx,gate,alt: UOp(Ops.LOAD, alt.dtype, (buf.index(idx), alt, gate))),
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx"), UPat())), UPat.var("val"), UPat.var("gate"))),
   lambda buf,idx,val,gate: UOp.store(buf.index(idx), val, gate)),
  # ptx shr and shl instructions require y to be uint
  (UPat.var("x") << UPat.var("y"), lambda x,y: UOp(Ops.SHL, x.dtype, (x,y.cast(dtypes.uint))) if y.dtype != dtypes.uint else None),
  (UPat.var("x") >> UPat.var("y"), lambda x,y: UOp(Ops.SHR, x.dtype, (x,y.cast(dtypes.uint))) if y.dtype != dtypes.uint else None),
])

def mem_type(x: UOp): return 'shared' if any(_x.op is Ops.DEFINE_LOCAL for _x in x.src[0].toposort()) else 'global'

def render_wmma(ctx: "PTXRenderer", wmma: UOp):
  assert ctx.wmma_r, "registry values for wmma must be populated"
  (N, M, K), dtype_in, dtype_out = wmma.arg[1], wmma.arg[2], wmma.arg[3]

  for src, regs in zip(wmma.src, ctx.wmma_r):
    for i, reg in enumerate(regs): # pack input and acc registers
      if (elems_per_reg := 4 // src.dtype.scalar().itemsize) == 1: yield f"mov.b32 {reg}, {ctx.r[src][i]};"
      else: yield f"mov.b32 {reg}, {{{', '.join(ctx.r[src][i * elems_per_reg : (i+1) * elems_per_reg])}}};"

  dt_map_in, dt_map_out = {dtypes.float: "tf32", dtypes.half: "f16"}, {dtypes.float: "f32", dtypes.half: "f16"}
  yield f'mma.sync.aligned.m{M}n{N}k{K}.row.col.{dt_map_out[dtype_out]}.{dt_map_in[dtype_in]}.{dt_map_in[dtype_in]}.{dt_map_out[dtype_out]}{" "*12}'+\
        f'{{{", ".join(ctx.wmma_r[2])}}}, {{{", ".join(ctx.wmma_r[0])}}}, {{{", ".join(ctx.wmma_r[1])}}}, {{{", ".join(ctx.wmma_r[2])}}};'

  for i, reg in enumerate(ctx.wmma_r[2]): # unpack acc registers
    if (elems_per_reg := 4 // dtype_out.itemsize) == 1: yield f"mov.b32 {ctx.r[wmma][i]}, {reg};"
    else: yield f"mov.b32 {{{', '.join(ctx.r[wmma][i * elems_per_reg : (i+1) * elems_per_reg])}}}, {reg};"

def modifier(a: DType, b: DType): return '.rzi' if dtypes.is_int(a) and dtypes.is_float(b) else '.rn' if dtypes.is_float(a) and \
  (a.itemsize < b.itemsize or dtypes.is_int(b) or b == dtypes.bool) else ''

string_rewrite = PatternMatcher([
  (UPat.cvar("x", dtypes.bool), lambda ctx, x: f"setp.ne.s16 {ctx.r[x]}, {render_val(x.arg, x.dtype)}, 0;"),
  (UPat.cvar("x"), lambda ctx, x: f"mov.b{ctx.types[x.dtype][1:]} {ctx.r[x]}, {render_val(x.arg, x.dtype)};"),
  (UPat(Ops.STORE, name="x", src=(UPat.var('bidx'), UPat.var("var")), allow_any_len=True), lambda ctx, x, bidx, var: f"st.{mem_type(bidx)}" + \
    f"{f'.v{cnt}' if ((cnt:=var.dtype.count)>1) else ''}.{ctx.mem_types[var.dtype.scalar()]} " + \
    f"[{ctx.r[bidx]}+0], {('{' + ', '.join(ctx.r[var]) + '}') if var.dtype.count > 1 else ctx.r[var]};"),
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: f"mov.u32 %{x.arg[0]}, %{'ctaid' if x.arg[0][0] == 'g' else 'tid'}.{chr(120+int(x.arg[0][-1]))};"),
  (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda ctx, x: f"ld.param.{ctx.types[dtypes.ulong]} {ctx.r[x]}, [data{x.arg}+0];"),
  (UPat((Ops.CMPLT, Ops.CMPNE), name="x", allow_any_len=True, src=(UPat.var("src0"),)),
    lambda ctx, x, src0: ctx.code_for_op[x.op](ctx.r[x], *[ctx.r[v] for v in x.src], src0.dtype, ctx.types[src0.dtype])),
  (UPat(GroupOp.ALU, name="x"), lambda ctx, x: ctx.code_for_op[x.op](ctx.r[x], *[ctx.r[v] for v in x.src], x.dtype, ctx.types[x.dtype])),
  (UPat(Ops.BITCAST, name="x", src=(UPat.var("a"),), allow_any_len=True), lambda ctx, x, a: f"mov.b{ctx.types[x.dtype][1:]} {ctx.r[x]}, {ctx.r[a]};"),
  (UPat(Ops.CAST, name="x", src=(UPat(dtype=dtypes.bool, name="a"),)),
   lambda ctx, x, a: f"selp.b{ctx.types[x.dtype][1:]} {ctx.r[x]}, {render_val(1, x.dtype)}, {render_val(0, x.dtype)}, {ctx.r[a]};"),
  (UPat(Ops.CAST, name="x", dtype=dtypes.bool, src=(UPat.var("a"),)),
   lambda ctx, x, a: f"setp.ne.b{ctx.types[a.dtype][1:]} {ctx.r[x]}, {ctx.r[a]}, {render_val(0, a.dtype)};"),
  (UPat(Ops.CAST, name="x", src=(UPat.var("a"),)),
   lambda ctx, x, a: f"cvt{modifier(x.dtype, a.dtype)}.{ctx.types[x.dtype]}.{ctx.types[a.dtype]} {ctx.r[x]}, {ctx.r[a]};"),
  (UPat(Ops.LOAD, name="x", src=(UPat.var('loc'), UPat(name='alt'), UPat(name="gate", op=GroupOp.ALU))), lambda ctx, x, loc, alt, gate: flatten([
    [f"mov.{ctx.mem_types[x.dtype.scalar()]} {v}, {render_val(0, x.dtype.scalar())};" for v in ctx.r[x]],
    [f"@{ctx.r[gate]} ld.{mem_type(x)}.v{x.dtype.count}.{ctx.mem_types[x.dtype.scalar()]} {{{', '.join(ctx.r[x])}}}, [{ctx.r[loc]}+0];"]
  ]) if alt.dtype.count > 1 else [
    f"@{ctx.r[gate]} ld.{mem_type(x)}.{ctx.mem_types[x.dtype.scalar()]} {ctx.r[x]}, [{ctx.r[loc]}+0];",
    f"@!{ctx.r[gate]} mov.b{ctx.types[x.dtype.scalar()][1:]} {ctx.r[x]}, {ctx.r[alt]};"]),
  (UPat(Ops.LOAD, name="x", src=(UPat.var('loc'),), allow_any_len=True),
   lambda ctx, x, loc: f"ld.{mem_type(x)}.v{x.dtype.count}.{ctx.mem_types[x.dtype.scalar()]} {{{', '.join(ctx.r[x])}}}, [{ctx.r[loc]}+0];" \
     if x.dtype.count > 1 else f"ld.{mem_type(x)}.{ctx.mem_types[x.dtype]} {ctx.r[x]}, [{ctx.r[loc]}+0];"),
  (UPat(Ops.DEFINE_ACC, name="x", src=(UPat.cvar("pred", dtype=dtypes.bool),), allow_any_len=True), lambda ctx, x, pred: [
    f"setp.ne.s16 {ctx.r[pred]}, {render_val(pred.arg, pred.dtype)}, 0;", f"mov.pred {ctx.r[x]}, {ctx.r[pred]};"]),
  (UPat(Ops.DEFINE_ACC, name="x", src=(UPat.cvar("pred"),), allow_any_len=True),
   lambda ctx, x, pred: f"mov.b{ctx.types[x.dtype][1:]} {ctx.r[x]}, {render_val(pred.arg, x.dtype)};"),
  (UPat(Ops.RANGE, name="x"), lambda ctx, x: [f"mov.u32 {ctx.r[x]}, 0;", "LOOP_" + f"{ctx.r[x][1:]}:"]),
  (UPat(Ops.ASSIGN, name="x", dtype=dtypes.bool), lambda ctx, x: [f"mov.pred {ctx.r[x.src[0]]}, {ctx.r[x.src[1]]};"]),
  (UPat(Ops.ASSIGN, name="x"), lambda ctx, x: f"mov.b{ctx.types[x.dtype][1:]} {ctx.r[x.src[0]]}, {ctx.r[x.src[1]]};"),
  (UPat(Ops.ENDRANGE, name="x", src=(UPat.var("src0"),)), lambda ctx, x, src0: [
    ctx.code_for_op[Ops.ADD](ctx.r[src0], ctx.r[src0], "1", dtypes.int, ctx.types[dtypes.int]),
    ctx.code_for_op[Ops.CMPLT](ctx.r[x], ctx.r[x.src[0]], ctx.r[src0.src[0]], dtypes.int, ctx.types[dtypes.int]),
    f"@{ctx.r[x]} bra LOOP_{ctx.r[src0][1:]};"]),
  (UPat(Ops.DEFINE_LOCAL, name="x"),
   lambda ctx, x: [f".shared .align 4 .b8 {x.arg}[{x.dtype.size*x.dtype.itemsize}];", f"mov.u64 {ctx.r[x]}, {x.arg}[0];"]),
  (UPat(Ops.IF, name="x"), lambda ctx, x: f"@!{ctx.r[x.src[0]]} bra IF_{ctx.r[x.src[0]][1:]}_{ctx.uops.index(x)};"),
  (UPat(Ops.ENDIF, name="x"), lambda ctx, x: f"IF_{ctx.r[x.src[0].src[0]][1:]}_{ctx.uops.index(x.src[0])}:"),
  (UPat(Ops.WMMA, name="x"), lambda ctx, x: list(render_wmma(ctx, x))),
  (UPat(Ops.BARRIER, name="x"), lambda ctx, x: ctx.barrier),
  (UPat(Ops.DEFINE_VAR, name="x"), lambda ctx, x: f"ld.param.{ctx.mem_types[x.dtype]} {ctx.r[x]}, [{x.arg[0]}+0];"),
])

class PTXRenderer(Renderer):
  device = "CUDA"
  suffix = "PTX"
  global_max, local_max, shared_max = CUDARenderer.global_max, CUDARenderer.local_max, CUDARenderer.shared_max
  tc_sm80 = [tc for tc in CUDARenderer.tc_sm80 if tc.dtype_in in [dtypes.half, dtypes.float]]
  code_for_op = asm_for_op
  extra_matcher = ptx_matcher
  def __init__(self, arch:str, device="CUDA"):
    self.device, self.arch = device, arch
    self.tensor_cores = PTXRenderer.tc_sm80 if int(arch[3:]) >= 80 else CUDARenderer.tc_sm75 if int(arch[3:]) >= 75 else []
  def __reduce__(self): return self.__class__, (self.arch, self.device)

  # language options
  kernel_prefix = """.version VERSION
.target TARGET
.address_size 64
.visible .entry"""
  barrier = "bar.sync\t0;"
  # HACK: Use s16 and u16 for int8 and uint8 buffers. This can be wrong in cast.
  types: dict[DType, str] = { dtypes.int8: "s16", dtypes.int16: "s16", dtypes.int32: "s32", dtypes.int64: "s64",
                              dtypes.uint8: "u16", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
                              dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64", dtypes.bool: "pred" }

  mem_types: dict[DType, str] = {**types, dtypes.int8: "s8", dtypes.uint8: "u8", dtypes.bool: "u8", dtypes.float16: "b16"}

  def render_kernel(self, kernel, function_name, bufs, regs) -> str:
    def fmt(line): return line if line[0]=="$" else "\t" + line.replace(" ", "\t" if len(line.split(" ")[0]) > 7 else "\t\t", 1)
    kernel = '\n'.join(map(fmt, [f".reg .{reg.split('_')[-2]} %{reg}<{cnt}>;" for reg,cnt in regs] + kernel + ["ret;"]))
    params = ',\n\t'.join([f".param .{'u64' if dtype.__class__ == PtrDType else self.types[dtype]} {name}" for name,dtype in bufs])
    return f"{self.kernel_prefix} {function_name}(\n\t{params}\n)\n{{\n{kernel}\n}}"

  def render(self, uops:list[UOp]) -> str:
    kernel:list[str] = []
    bufs = []

    c: defaultdict[str, int] = defaultdict(int)
    r: dict[UOp, list[str]|str] = {}
    self.r = r
    self.uops = uops

    def ssa(prefix:str, u:UOp|None=None, dtype:str|None=None) -> str:
      nonlocal c, r
      prefix += f"_{dtype if dtype is not None else self.types[cast(UOp, u).dtype]}_"
      c[prefix] += 1
      return f"%{prefix}{c[prefix]-1}"

    name = "test"
    for u in uops:
      if u.op is Ops.SINK:
        if u.arg is not None: name = u.arg.name
        continue
      if u.op is Ops.VECTORIZE:
        r[u] = [cast(str,r[x]) for x in u.src]
        continue
      if u.op is Ops.GEP:
        r[u] = r[u.src[0]][get_single_element(u.arg)]
        continue
      if u.op in {Ops.CAST, Ops.BITCAST} and (u.src[0].dtype == u.dtype or isinstance(u.src[0].dtype, PtrDType)):
        r[u] = r[u.src[0]]
        continue
      if u.op is Ops.SPECIAL: r[u] = "%" + u.arg[0]
      elif u.op is Ops.DEFINE_VAR: bufs.append((u.arg[0], u.dtype))
      elif u.op is Ops.LOAD:
        assert u.src[0].dtype == dtypes.int64, "load isn't int64"
        r[u] = [ssa('val', dtype=self.types[u.dtype.scalar()]) for _ in range(u.dtype.count)] if u.dtype.count > 1 else ssa('val', u)
      elif u.op is Ops.DEFINE_GLOBAL: bufs.append((f"data{u.arg}", u.dtype))
      elif u.op is Ops.WMMA:
        # registers for packing/unpacking input and acc
        self.wmma_r = [[ssa("wmma_in", dtype="b32") for _ in range(0, len(r[u.src[0]]), 4 // u.arg[2].itemsize)],
                       [ssa("wmma_in", dtype="b32") for _ in range(0, len(r[u.src[1]]), 4 // u.arg[2].itemsize)],
                       [ssa("wmma_acc", dtype="b32") for _ in range(0, len(r[u.src[2]]), 4 // u.arg[3].itemsize)]]
        r[u] = [ssa("wmma", dtype=self.types[u.dtype.scalar()]) for _ in range(u.dtype.count)]
      prefix, dtype = {Ops.CAST: ("cast", None), Ops.BITCAST: ("cast", None), Ops.ENDRANGE: ("pred", "pred"), Ops.RANGE: ("ridx", None),
        Ops.DEFINE_ACC: ("acc", None), Ops.DEFINE_VAR: ("dat", None), Ops.CONST: ("const", None), Ops.DEFINE_LOCAL:("local",self.types[dtypes.ulong]),
        Ops.DEFINE_GLOBAL: ("dat", self.types[dtypes.ulong]), **{op: ("alu", None) for op in GroupOp.ALU}}.get(u.op, (None, None))
      if prefix: r[u] = ssa(prefix, u, dtype)

      if (l:=cast(str|list[str], string_rewrite.rewrite(u, ctx=self))) is None:
        raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
      kernel.extend([l] if isinstance(l, str) else l)

      if u.op is Ops.ASSIGN: r[u] = r[u.src[0]]
      elif u.op is Ops.SPECIAL: kernel = [f".reg .u32 %{u.arg[0]};"] + kernel
    return self.render_kernel(kernel, name, bufs, c.items())
