from typing import cast
import math, struct, sys
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.ops import UOp, PatternMatcher, UPat, Ops, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType, truncate
from tinygrad.helpers import prod, AMX

def ldt(dt:DType):
  if dt.vcount > 1: return f"<{dt.vcount} x {ldt(dt.scalar())}>"
  if isinstance(dt, PtrDType): return ldt(dt.base) + "*"
  return {dtypes.int8: "i8", dtypes.int16: "i16", dtypes.int32: "i32", dtypes.int64: "i64",
          dtypes.uint8: "i8", dtypes.uint16: "i16", dtypes.uint32: "i32", dtypes.uint64: "i64",
          dtypes.float16: "half", dtypes.float32: "float", dtypes.float64: "double", dtypes.bool: "i1", dtypes.void: "void"}[dt]

def lconst(x, dtype:DType):
  if dtype in dtypes.floats:
    if math.isinf(x) or math.isnan(x): return "0x%02X%02X%02X%02X%02X%02X%02X%02X" % tuple(struct.pack("d",x)[::-1])
    return truncate[dtype](x)
  return int(x)

def lcast(input_type:DType, output_type:DType):
  if dtypes.is_float(input_type):
    if dtypes.is_float(output_type): return 'fpext' if output_type.itemsize > input_type.itemsize else 'fptrunc'
    if dtypes.is_int(output_type): return 'fptoui' if dtypes.is_unsigned(output_type) else 'fptosi'
  if dtypes.is_unsigned(input_type) or dtypes.is_bool(input_type):
    if dtypes.is_float(output_type): return 'uitofp'
    if dtypes.is_int(output_type): return 'trunc' if output_type.itemsize < input_type.itemsize else 'zext'
  if dtypes.is_int(input_type):
    if dtypes.is_float(output_type): return 'sitofp'
    if dtypes.is_int(output_type): return 'trunc' if output_type.itemsize < input_type.itemsize else 'sext'
  raise NotImplementedError(f"cast from {input_type} -> {output_type} not implemented")

# https://github.com/corsix/amx
def render_wmma(ctx, wmma: UOp) -> str:
  def AMX(op, gpr): return f'call void asm sideeffect ".word (0x201000+($0<<5)+0$1-((0$1>>4)*6))", "i,r,~{{memory}}"(i32 {op}, i64 {gpr}) #0; AMX'

  return "\n".join([
    *[f'  store {ldt(src.dtype)} {ctx[src]}, {ldt(src.dtype.ptr())} {ctx[wmma]}_amx{i}, align {src.dtype.itemsize}' for i,src in enumerate(wmma.src)],
      f'  call void asm sideeffect "nop\\0Anop\\0Anop\\0A.word ({0x201000 + (17 << 5) + 0})", "~{{memory}}"() #0; AMX set',             # set
    *[f'  {ctx[wmma]}_ld{i} = add i64 {ctx[wmma]}_ptr_amx2, {i*4<<56 | i*64}\n  {AMX(4,f"{ctx[wmma]}_ld{i}")} ldz' for i in range(16)], # ldz
      f'  {AMX(0, f"{ctx[wmma]}_ptr_amx1")} ldx\n  {AMX(1, f"{ctx[wmma]}_ptr_amx0")} ldy\n  {AMX(12, 0)} fma32',                        # ldx ldy fma
    *[f'  {ctx[wmma]}_st{i} = add i64 {ctx[wmma]}_ptr_amx2, {i*4<<56 | i*64}\n  {AMX(5,f"{ctx[wmma]}_st{i}")} stz' for i in range(16)], # stz
      f'  call void asm sideeffect "nop\\0Anop\\0Anop\\0A.word ({0x201000 + (17 << 5) + 1})", "~{{memory}}"() #0; AMX clr',             # clr
      f'  {ctx[wmma]} = load {ldt(wmma.dtype)}, ptr {ctx[wmma]}_amx2, align {wmma.dtype.itemsize}'])

# llvm ops, lop[<dtype>][<op>]
unsigned_lop = { Ops.ADD: "add", Ops.MUL: "mul", Ops.IDIV: "udiv", Ops.MOD: "urem",
                 Ops.CMPLT: "icmp ult", Ops.CMPNE: "icmp ne", Ops.OR: "or", Ops.AND: "and", Ops.XOR: "xor", }
signed_lop = {**unsigned_lop, Ops.CMPLT: "icmp slt", Ops.IDIV: "sdiv", Ops.MOD: "srem"}
flags = " nsz arcp contract afn"
float_lop = {Ops.ADD: "fadd"+flags, Ops.MUL: "fmul"+flags, Ops.CMPLT: f"fcmp{flags} ult", Ops.CMPNE: f"fcmp{flags} une", Ops.FDIV: "fdiv"+flags}
lop = {**{x:unsigned_lop for x in (dtypes.bool,)+dtypes.uints}, **{x:signed_lop for x in dtypes.sints}, **{x:float_lop for x in dtypes.floats}}

base_rewrite = PatternMatcher([
  # memory load/store
  (UPat(Ops.INDEX, name="x"), lambda ctx,x:
   f"  {ctx[x]} = getelementptr inbounds {ldt(x.dtype.base)}, {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, {ldt(x.src[1].dtype)} {ctx[x.src[1]]}"),
  (UPat(Ops.LOAD, src=(UPat.var('idx'), UPat.var('alt'), UPat.var('mask')), name="x"), lambda ctx,x,idx,alt,mask:
   f"  br label {ctx[x]}_entry\n{ctx[x][1:]}_entry:\n"
   f"  br i1 {ctx[mask]}, label {ctx[x]}_load, label {ctx[x]}_exit\n{ctx[x][1:]}_load:\n"
   f"  {ctx[x]}_yes = load {ldt(x.dtype)}, {ldt(idx.dtype)} {ctx[idx]}\n"
   f"  br label {ctx[x]}_exit\n{ctx[x][1:]}_exit:\n"
   f"  {ctx[x]} = phi {ldt(x.dtype)} [{ctx[x]}_yes, {ctx[x]}_load], [{ctx[alt]}, {ctx[x]}_entry]"),
  (UPat(Ops.LOAD, src=(UPat.var('idx'),), name="x"), lambda ctx,x,idx: f"  {ctx[x]} = load {ldt(x.dtype)}, {ldt(idx.dtype)} {ctx[idx]}"),
  (UPat(Ops.STORE, name="x"), lambda ctx,x: f"  store {ldt(x.src[1].dtype)} {ctx[x.src[1]]}, {ldt(x.src[0].dtype)} {ctx[x.src[0]]}"),

  # GEP/VECTORIZE/CAST for float4 support
  (UPat(Ops.GEP, name="x"), lambda ctx,x: f"  {ctx[x]} = extractelement {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, i32 {x.arg[0]}"),
  (UPat(Ops.VECTORIZE, src=UPat.var('y'), name="x"), lambda ctx,x,y:
   f"  {ctx[x]}_z = insertelement <1 x {ldt(y.dtype)}> poison, {ldt(y.dtype)} {ctx[y]}, i32 0\n"
   f"  {ctx[x]} = shufflevector <1 x {ldt(y.dtype)}> {ctx[x]}_z, <1 x {ldt(y.dtype)}> poison, <{x.dtype.count} x i32> zeroinitializer"),
  (UPat(Ops.VECTORIZE, name="x"), lambda ctx,x: "\n".join([(f"  {ctx[x]}_{i}" if i+1 != len(x.src) else f"  {ctx[x]}")+
                                                            f" = insertelement {ldt(x.dtype)} "+(f"{ctx[x]}_{i-1}" if i != 0 else "poison")+
                                                            f", {ldt(u.dtype)} {ctx[u]}, i32 {i}" for i,u in enumerate(x.src)])),
  (UPat(Ops.CAST, name="x"), lambda ctx,x:
   f"  {ctx[x]} = bitcast {ldt(x.src[0].dtype)} {ctx[x.src[0]]} to {ldt(x.dtype)}" if isinstance(x.dtype, PtrDType) else None),

  # unary/binary/ternary ops
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"  {ctx[x]} = bitcast {ldt(x.src[0].dtype)} {ctx[x.src[0]]} to {ldt(x.dtype)}"),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"  {ctx[x]} = {lcast(x.src[0].dtype, x.dtype)} {ldt(x.src[0].dtype)} {ctx[x.src[0]]} to {ldt(x.dtype)}"),
  (UPat(GroupOp.Binary, name="x"), lambda ctx,x:
   f"  {ctx[x]} = {lop[x.src[0].dtype.scalar()][x.op]} {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, {ctx[x.src[1]]}"),
  (UPat(Ops.WHERE, name="x"), lambda ctx,x:
   f"  {ctx[x]} = select {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, {ldt(x.src[1].dtype)} {ctx[x.src[1]]}, {ldt(x.src[2].dtype)} {ctx[x.src[2]]}"),

  # range
  (UPat(Ops.RANGE, name="x"), lambda ctx,x:
   f"  br label %loop_entry_{x.arg}\nloop_entry_{x.arg}:\n"
   f"  br label %loop_body_{x.arg}\nloop_body_{x.arg}:\n"
   f"  {ctx[x]} = phi {ldt(x.dtype)} [{ctx[x.src[0]]}, %loop_entry_{x.arg}], [{ctx[x]}phi, %loop_latch_{x.arg}]"),
  (UPat(Ops.ENDRANGE, name="x"), lambda ctx,x:
   f"  br label %loop_latch_{x.src[0].arg}\nloop_latch_{x.src[0].arg}:\n"
   f"  {ctx[x.src[0]]}phi = add i32 {ctx[x.src[0]]}, 1\n  {ctx[x]} = icmp ult i32 {ctx[x.src[0]]}phi, {ctx[x.src[0].src[1]]}\n"
   f"  br i1 {ctx[x]}, label %loop_body_{x.src[0].arg}, label %loop_exit_{x.src[0].arg}\nloop_exit_{x.src[0].arg}:"),

  # if
  (UPat(Ops.IF, name="x"), lambda ctx,x: f"  br i1 {ctx[x.src[0]]}, label %ifbody_{ctx[x][1:]}, label %ifskip_{ctx[x][1:]}\nifbody_{ctx[x][1:]}:"),
  (UPat(Ops.ENDIF, name="x"), lambda ctx,x: f"  br label %ifskip_{ctx[x.src[0]][1:]}\nifskip_{ctx[x.src[0]][1:]}:"),

  # wmma
  (UPat(Ops.WMMA, name="wmma"), render_wmma),
])

def llvm_bf16_cast(buf:UOp, idx:UOp, root:UOp):
  u16_buf = buf.replace(dtype=dtypes.ushort.ptr(size=cast(PtrDType,buf.dtype).size))
  return UOp.load(UOp.index(u16_buf, idx), dtype=dtypes.ushort).cast(dtypes.uint).mul(1<<16).bitcast(dtypes.float32).cast(root.dtype)

class LLVMRenderer(Renderer):
  device = "LLVM"
  abi = 'win64cc' if sys.platform == 'win32' else None
  supports_float4 = True
  has_local = False
  has_shared = False
  global_max = None
  string_rewrite = base_rewrite
  if AMX: tensor_cores = ClangRenderer.amx_tc

  extra_matcher = PatternMatcher([
    # rewrite RECIP with FDIV
    (UPat(Ops.RECIP, name="x"), lambda x: UOp(Ops.FDIV, x.dtype, (x.const_like(1), x.src[0]))),
    # rewrite cast to bool to CMPNE 0
    (UPat(Ops.CAST, dtype=dtypes.bool, name="x"), lambda x: x.src[0] != x.src[0].const_like(0)),
    # rewrite MAX to CMPLT + WHERE
    (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
    # rewrite bf16 CAST(LOAD) to CAST(BITCAST)
    (UPat(Ops.CAST, name="root", src=(UPat.load(UPat.index(UPat.var("buf"), UPat.var("idx")), dtype=dtypes.bfloat16),)), llvm_bf16_cast),
  ])

  def render(self, uops: list[UOp]) -> str:
    r: dict[UOp, str] = {}
    args: list[str] = []
    kernel: list[str] = []
    end_lines: dict[str, None] = {}
    vc = -1

    acc_to_assign: dict[UOp, UOp] = {}
    for u in uops:
      if u.op is Ops.ASSIGN: # prealloc all assigns
        vc += 1
        r[u] = r[u.src[1]] = f"%assign{vc}"
        assert u.src[0] not in acc_to_assign, "can't assign to DEFINE_ACC twice"
        acc_to_assign[u.src[0]] = u.src[1]
      if u.op is Ops.WMMA: # prealloc aux buffers as AMX can only load from memory
        vc += 1
        r[u] = f"%wmma{vc}"
        for i, dtype in enumerate(u.arg[2].vec(sz) for sz in [prod(size for _, size in upcast) for upcast in u.arg[6]]):
          kernel += [f"  {r[u]}_amx{i} = alloca {ldt(dtype)}, align {dtype.itemsize}",
                     f"  {r[u]}_ptr_amx{i} = ptrtoint {ldt(dtype.ptr())} {r[u]}_amx{i} to i64"]

    name = "test"
    for u in uops:
      if u.op is Ops.NAME:
        name = u.arg
        continue
      if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
        r[u] = f"%data{u.arg}" if u.op is Ops.DEFINE_GLOBAL else f"%{u.arg[0]}"
        # NOTE: MallocAllocator promises 0x20 alignment
        args.append(f"{ldt(u.dtype)}{' noalias align 32' if isinstance(u.dtype, PtrDType) else ''} {r[u]}")
      elif u.op is Ops.ASSIGN: pass  # assign is already handled by the first pass
      elif u.op is Ops.DEFINE_ACC: r[u] = r[u.src[0]]  # a define acc can be used and never be assigned to
      elif u.op is Ops.CONST: r[u] = lconst(u.arg, u.dtype)
      elif u.op is Ops.CAST and ldt(u.dtype) == ldt(u.src[0].dtype): r[u] = r[u.src[0]] # cast from signed to unsigned of the same size is a noop
      else:
        # if it's an assign target, it's already preallocated
        if u not in r:
          vc += 1
          r[u] = f"%v{vc}"

        # do the rendering of the llvm ir code
        if (l:=self.string_rewrite.rewrite(u, ctx=r)) is None:
          raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
        kernel.append(cast(str, l))

        # generate the phi nodes for the assigns
        if u.op is Ops.RANGE:
          for x in acc_to_assign:
            if u in x.src:  # if this range is relevant for this acc
              vc += 1
              kernel.append(f"  %acc{vc} = phi {ldt(x.dtype)}" f"[{r[x]}, %loop_entry_{u.arg}], [{r[acc_to_assign[x]]}, %loop_latch_{u.arg}]")
              r[x] = f"%acc{vc}"

    # output the function. chr(10) is '\n' (python < 3.12 doesn't support backslashes in f-strings)
    return f'''\
define{(' '+self.abi) if self.abi is not None else ''} void @{name}({','.join(args)}) #0 {{
{chr(10).join(kernel)}
  ret void
}}
{chr(10).join(end_lines.keys())}
attributes #0 = {{ nounwind "no-builtins" "no-trapping-math"="true" }}
'''
