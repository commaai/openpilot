from typing import cast
import math, struct, sys
from tinygrad.codegen.opt import tc
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import AMDRenderer
from tinygrad.uop.ops import UOp, PatternMatcher, UPat, Ops, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType, truncate
from tinygrad.helpers import prod, AMX

def ldt(dt:DType):
  if dt.vcount > 1: return f"<{dt.vcount} x {ldt(dt.scalar())}>"
  if isinstance(dt, PtrDType): return ldt(dt.base) + "*"
  return {dtypes.void: "void", dtypes.bool: "i1", dtypes.int8: "i8", dtypes.int16: "i16", dtypes.int32: "i32", dtypes.int64: "i64",
          dtypes.uint8: "i8", dtypes.uint16: "i16", dtypes.uint32: "i32", dtypes.uint64: "i64",
          dtypes.float16: "half", dtypes.bfloat16: "bfloat", dtypes.float32: "float", dtypes.float64: "double"}[dt]

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
def render_wmma_amx(ctx, wmma: UOp) -> str:
  def AMX(op, gpr): return f'call void asm sideeffect ".word (0x201000+($0<<5)+0$1-((0$1>>4)*6))", "i,r,~{{memory}}"(i32 {op}, i64 {gpr}) #0; AMX'

  return "\n".join([
    *[f'  store {ldt(src.dtype)} {ctx[src]}, {ldt(src.dtype.ptr())} {ctx[wmma]}_amx{i}, align {src.dtype.itemsize}' for i,src in enumerate(wmma.src)],
      f'  call void asm sideeffect "nop\\0Anop\\0Anop\\0A.word ({0x201000 + (17 << 5) + 0})", "~{{memory}}"() #0; AMX set',             # set
    *[f'  {ctx[wmma]}_ld{i} = add i64 {ctx[wmma]}_ptr_amx2, {i*4<<56 | i*64}\n  {AMX(4,f"{ctx[wmma]}_ld{i}")} ldz' for i in range(16)], # ldz
      f'  {AMX(0, f"{ctx[wmma]}_ptr_amx1")} ldx\n  {AMX(1, f"{ctx[wmma]}_ptr_amx0")} ldy\n  {AMX(12, 0)} fma32',                        # ldx ldy fma
    *[f'  {ctx[wmma]}_st{i} = add i64 {ctx[wmma]}_ptr_amx2, {i*4<<56 | i*64}\n  {AMX(5,f"{ctx[wmma]}_st{i}")} stz' for i in range(16)], # stz
      f'  call void asm sideeffect "nop\\0Anop\\0Anop\\0A.word ({0x201000 + (17 << 5) + 1})", "~{{memory}}"() #0; AMX clr',             # clr
      f'  {ctx[wmma]} = load {ldt(wmma.dtype)}, ptr {ctx[wmma]}_amx2, align {wmma.dtype.itemsize}'])

def render_wmma_amd(ctx, wmma: UOp, arch: str) -> str:
  dt_map = {dtypes.half: "f16", dtypes.float: "f32", dtypes.bfloat16: "bf16", dtypes.ushort: "bf16"}
  # https://github.com/llvm/llvm-project/blob/main/clang/test/CodeGenOpenCL/builtins-amdgcn-mfma.cl
  if arch.split(":")[0] in {"gfx942", "gfx950"}:
    return f"  {ctx[wmma]} = call {ldt(wmma.dtype)} @llvm.amdgcn.mfma.{dt_map[wmma.src[-1].dtype.scalar()]}" + \
           f".16x16x16{dt_map[wmma.src[0].dtype.scalar()]}(" + ", ".join([f"{ldt(w.dtype)} {ctx[w]}" for w in wmma.src]) + ", i32 0, i32 0, i32 0)"
  # https://github.com/llvm/llvm-project/blob/main/llvm/test/CodeGen/AMDGPU/GlobalISel/llvm.amdgcn.wmma_32.ll
  # example: %wmma0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16(<16 x half> %v99,<16 x half> %v100,<8 x float> %v101)
  return f"  {ctx[wmma]} = call {ldt(wmma.dtype)} @llvm.amdgcn.wmma.{dt_map[wmma.src[-1].dtype.scalar()]}.16x16x16." + \
    f"{dt_map[wmma.src[0].dtype.scalar()]}(" + ", ".join([f"{ldt(w.dtype)} {ctx[w]}" for w in wmma.src]) + (", i1 false)" \
      if wmma.dtype.scalar() != dtypes.float else ")")

# llvm ops, lop[<dtype>][<op>]
unsigned_lop = { Ops.ADD: "add", Ops.MUL: "mul", Ops.IDIV: "udiv", Ops.MOD: "urem",
                 Ops.CMPLT: "icmp ult", Ops.CMPNE: "icmp ne", Ops.CMPEQ: "icmp eq", Ops.OR: "or", Ops.AND: "and", Ops.XOR: "xor",}
signed_lop = {**unsigned_lop, Ops.ADD: "add nsw", Ops.CMPLT: "icmp slt", Ops.IDIV: "sdiv", Ops.MOD: "srem"}
flags = " nsz arcp contract afn"
float_lop = {Ops.ADD: "fadd"+flags, Ops.MUL: "fmul"+flags, Ops.CMPLT: f"fcmp{flags} ult",
    Ops.CMPNE: f"fcmp{flags} une", Ops.CMPEQ: f"fcmp{flags} oeq", Ops.FDIV: "fdiv"+flags}
lop = {**{x:unsigned_lop for x in (dtypes.bool,)+dtypes.uints}, **{x:signed_lop for x in dtypes.sints}, **{x:float_lop for x in dtypes.floats}}

base_rewrite = PatternMatcher([
  # memory load/store
  (UPat(Ops.INDEX, name="x"), lambda ctx,x:
   f"  {ctx[x]} = getelementptr inbounds {ldt(x.dtype.base)}, {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, {ldt(x.src[1].dtype)} {ctx[x.src[1]]}"),
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(), UPat(), UPat.var("mask"))).or_casted("idx"), UPat.var("alt")), name="x"),
   lambda ctx,x,idx,alt,mask:
   f"  br label {ctx[x]}_entry\n{ctx[x][1:]}_entry:\n"
   f"  br i1 {ctx[mask]}, label {ctx[x]}_load, label {ctx[x]}_exit\n{ctx[x][1:]}_load:\n"
   f"  {ctx[x]}_yes = load {ldt(x.dtype)}, {ldt(idx.dtype)} {ctx[idx]}\n"
   f"  br label {ctx[x]}_exit\n{ctx[x][1:]}_exit:\n"
   f"  {ctx[x]} = phi {ldt(x.dtype)} [{ctx[x]}_yes, {ctx[x]}_load], [{ctx[alt]}, {ctx[x]}_entry]"),
  (UPat(Ops.LOAD, src=(UPat.var('idx'),), allow_any_len=True, name="x"),
   lambda ctx,x,idx: f"  {ctx[x]} = load {ldt(x.dtype)}, {ldt(idx.dtype)} {ctx[idx]}"),
  (UPat(Ops.STORE, name="x"), lambda ctx,x: f"  store {ldt(x.src[1].dtype)} {ctx[x.src[1]]}, {ldt(x.src[0].dtype)} {ctx[x.src[0]]}"),

  # GEP/VECTORIZE/CAST for float4 support
  (UPat(Ops.GEP, name="x"), lambda ctx,x: f"  {ctx[x]} = extractelement {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, i32 {x.arg[0]}"),
  (UPat(Ops.VECTORIZE, src=UPat.var('y'), name="x"), lambda ctx,x,y:
   f"  {ctx[x]}_z = insertelement <1 x {ldt(y.dtype)}> poison, {ldt(y.dtype)} {ctx[y]}, i32 0\n"
   f"  {ctx[x]} = shufflevector <1 x {ldt(y.dtype)}> {ctx[x]}_z, <1 x {ldt(y.dtype)}> poison, <{x.dtype.count} x i32> zeroinitializer"),
  (UPat(Ops.VECTORIZE, name="x"), lambda ctx,x: "\n".join([(f"  {ctx[x]}_{i}" if i+1 != len(x.src) else f"  {ctx[x]}")+
                                                            f" = insertelement {ldt(x.dtype)} "+(f"{ctx[x]}_{i-1}" if i != 0 else "poison")+
                                                            f", {ldt(u.dtype)} {ctx[u]}, i32 {i}" for i,u in enumerate(x.src)])),
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
   f"  {ctx[x]} = phi {ldt(x.dtype)} [ 0, %loop_entry_{x.arg} ], [ {ctx[x]}phi, %loop_latch_{x.arg} ]"),
  (UPat(Ops.ENDRANGE, name="x"), lambda ctx,x:
   f"  br label %loop_latch_{x.src[0].arg}\nloop_latch_{x.src[0].arg}:\n"
   f"  {ctx[x.src[0]]}phi = add i32 {ctx[x.src[0]]}, 1\n  {ctx[x]} = icmp ult i32 {ctx[x.src[0]]}phi, {ctx[x.src[0].src[0]]}\n"
   f"  br i1 {ctx[x]}, label %loop_body_{x.src[0].arg}, label %loop_exit_{x.src[0].arg}\nloop_exit_{x.src[0].arg}:"),

  # if
  (UPat(Ops.IF, name="x"), lambda ctx,x: f"  br i1 {ctx[x.src[0]]}, label %ifbody_{ctx[x][1:]}, label %ifskip_{ctx[x][1:]}\nifbody_{ctx[x][1:]}:"),
  (UPat(Ops.ENDIF, name="x"), lambda ctx,x: f"  br label %ifskip_{ctx[x.src[0]][1:]}\nifskip_{ctx[x.src[0]][1:]}:"),

  (UPat(Ops.BARRIER), lambda ctx: "")
])

class LLVMRenderer(Renderer):
  device = "LLVM"
  abi = 'win64cc' if sys.platform == 'win32' else None
  supports_float4 = True
  has_local = False
  global_max: tuple[int, ...] | None = None
  string_rewrite = base_rewrite + PatternMatcher([(UPat(Ops.WMMA, name="wmma"), render_wmma_amx)])
  if AMX: tensor_cores = tc.amx

  extra_matcher = PatternMatcher([
    # rewrite RECIP with FDIV
    (UPat(Ops.RECIP, name="x"), lambda x: UOp(Ops.FDIV, x.dtype, (x.const_like(1), x.src[0]))),
    # rewrite cast to bool to CMPNE 0
    (UPat(Ops.CAST, dtype=dtypes.bool, name="x"), lambda x: x.src[0] != x.src[0].const_like(0)),
    # rewrite MAX to CMPLT + WHERE
    (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
    # copied from cstyle.py, upcast to float32 all the ops that don't support bfloat16
    (UPat((Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN), dtype=dtypes.bfloat16, name="x"),
      lambda x: (UOp(x.op, dtypes.float, tuple(vv.cast(dtypes.float) for vv in x.src), x.arg).cast(dtypes.bfloat16))),
    # copied from cstyle.py, add float intermediate casting
    (UPat(Ops.CAST, name="x", src=UPat.var("y", dtypes.bfloat16)),lambda x,y: y.cast(dtypes.float).cast(x.dtype) if x.dtype!=dtypes.float else None),
    (UPat(Ops.CAST, dtypes.bfloat16, UPat.var("x")),lambda x: x.cast(dtypes.float).cast(dtypes.bfloat16) if x.dtype!=dtypes.float else None),
  ])

  def render(self, uops: list[UOp]) -> str: return "\n".join((k:=self._render_kernel(uops))[0] + (k[1], self._render_footer(uops)))
  def _render_footer(self, uops: list[UOp]) -> str: return 'attributes #0 = { alwaysinline nounwind "no-builtins" "no-trapping-math"="true" }'
  def _render_fn(self, name:str, args:list[tuple[str,DType]], kernel:list[str], prefix:list[str]|None=None) -> str:
    # NOTE: CPUAllocator promises 0x20 alignment
    sargs = ", ".join([f"{ldt(dt)}{' noalias align 32' if isinstance(dt, PtrDType) else ''} {name}" for name,dt in args])
    sprefix = "".join([f" {x}" for x in (prefix or []) + [self.abi] if x is not None])
    return "\n".join([f"define{sprefix} void @{name}({sargs}) #0", "{"] + kernel + ["  ret void\n}"])
  def _render_kernel(self, uops: list[UOp], prefix:list[str]|None=None) -> tuple[tuple[str, ...], str]:
    r: dict[UOp, str] = {}
    args: list[tuple[str, DType]] = []
    kernel: list[str] = []
    vc = -1

    local_args: list[str] = []
    for u in uops:
      if AMX and u.op is Ops.WMMA: # prealloc aux buffers as AMX can only load from memory
        vc += 1
        r[u] = f"%wmma{vc}"
        for i, dtype in enumerate(u.arg[2].vec(sz) for sz in [prod(size for _, size in upcast) for upcast in u.arg[6]]):
          kernel += [f"  {r[u]}_amx{i} = alloca {ldt(dtype)}, align {dtype.itemsize}",
                     f"  {r[u]}_ptr_amx{i} = ptrtoint {ldt(dtype.ptr())} {r[u]}_amx{i} to i64"]

    name = "test"
    for u in uops:
      if u.op is Ops.NOOP: continue
      if u.op is Ops.SINK:
        if u.arg is not None: name = u.arg.function_name
        continue
      if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
        r[u] = f"%data{u.arg}" if u.op is Ops.DEFINE_GLOBAL else f"%{u.arg[0]}"
        args.append((r[u], u.dtype))
      elif u.op in (Ops.DEFINE_LOCAL, Ops.DEFINE_REG):
        r[u] = f"%{'local' if u.op is Ops.DEFINE_LOCAL else 'reg'}_{str(u.arg).replace('(', '').replace(')', '').replace(',', '_').replace(' ', '')}"
        assert isinstance(u.dtype, PtrDType)
        if self.device == "LLVM" or u.op is Ops.DEFINE_REG:
          kernel.append(f"  {r[u]} = alloca [{u.dtype.size} x {ldt(u.dtype.base)}]")
        else:
          local_args.append(f"@{r[u][1:]} = internal unnamed_addr addrspace(3) global [{u.dtype.size} x {ldt(u.dtype)}] undef, align 16")
          kernel.append(f"  {r[u]} = addrspacecast [{u.dtype.size} x {ldt(u.dtype)}] addrspace(3)* @{r[u][1:]} to [{u.dtype.size} x {ldt(u.dtype)}]*")
      elif u.op is Ops.CONST: r[u] = lconst(u.arg, u.dtype)
      elif u.op is Ops.CAST and (ldt(u.dtype) == ldt(u.src[0].dtype) or isinstance(u.dtype, PtrDType)):
        r[u] = r[u.src[0]] # cast from signed to unsigned of the same size is a noop, or pointer cast
      else:
        # if it's an assign target, it's already preallocated
        if u not in r:
          vc += 1
          r[u] = f"%v{vc}"

        # do the rendering of the llvm ir code
        if (l:=self.string_rewrite.rewrite(u, ctx=r)) is None:
          raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
        kernel.append(cast(str, l))
    return tuple(local_args), self._render_fn(name, args, kernel, prefix)

barrier = 'fence syncscope("workgroup") release\ntail call void @llvm.amdgcn.s.barrier()\nfence syncscope("workgroup") acquire\n'
code_for_workitem = {"g": lambda x: f"tail call i32 @llvm.amdgcn.workgroup.id.{chr(120+int(x))}()",
                     "l": lambda x: f"tail call i32 @llvm.amdgcn.workitem.id.{chr(120+int(x))}()"}
class AMDLLVMRenderer(LLVMRenderer):
  device = "AMD"
  has_local = True
  shared_max = AMDRenderer.shared_max
  global_max = AMDRenderer.global_max
  abi = "amdgpu_kernel"
  string_rewrite = PatternMatcher([
    (UPat(Ops.SPECIAL, name="x"), lambda ctx, x: f"  {ctx[x]} = " + f"{ code_for_workitem[x.arg[0][0]](x.arg[0][-1])}; "),
    (UPat(Ops.BARRIER), lambda ctx: barrier),
    (UPat(Ops.CAST, name="x", dtype=dtypes.half.vec(16), src=UPat.var("y", dtypes.half.vec(8))), lambda ctx, x, y: f"  {ctx[x]} = shufflevector "\
      f"<8 x half> {ctx[y]}, <8 x half> zeroinitializer, <16 x i32> <{', '.join([f'i32 {i}, i32 {j}' for i, j in zip(range(0, 8), range(8, 16))])}>"),
    (UPat(Ops.CAST, name="x", dtype=dtypes.half.vec(8), src=UPat.var("y", dtypes.half.vec(16))), lambda ctx, x, y:
      f"  {ctx[x]}= shufflevector <16 x half> {ctx[y]}, <16 x half> undef, <8 x i32> <{', '.join([f'i32 {x}' for x in range(0, 16, 2)])}>"),
  ]) + base_rewrite
  extra_matcher = LLVMRenderer.extra_matcher
  def _render_footer(self, uops: list[UOp]) -> str:
    # TODO: this is copied from cstyle
    requiredMaxThreadsPerBlock = prod(u.arg[1] for u in uops if u.op is Ops.SPECIAL and u.arg[0][0] == "l")
    attributes = ["alwaysinline", "nounwind", '"no-builtins"',
                  f'"amdgpu-flat-work-group-size"="1,{requiredMaxThreadsPerBlock}"', '"no-trapping-math"="true"']
    return 'attributes #0 = { ' + ' '.join(attributes) + ' }'
  def __init__(self, arch:str):
    self.arch = arch
    self.tensor_cores = AMDRenderer.get_tensor_cores(arch)
    self.string_rewrite += PatternMatcher([(UPat(Ops.WMMA, name="wmma"), lambda ctx, wmma, arch=arch: render_wmma_amd(ctx, wmma, arch))])
    if self.arch.split(":")[0] == "gfx1100":
      self.extra_matcher += PatternMatcher([
        (UPat(Ops.WMMA, name="x", dtype=dtypes.half.vec(8)),
          lambda x: UOp(Ops.WMMA, dtypes.half.vec(16), (x.src[0], x.src[1], x.src[2].cast(dtypes.half.vec(16))), (*x.arg,)).cast(dtypes.half.vec(8))),
        (UPat(Ops.WMMA, name="x"), lambda x: UOp(Ops.WMMA, x.dtype, (x.src[0].bitcast(dtypes.uint16.vec(16)), x.src[1].bitcast(dtypes.uint16.vec(16)),
          x.src[2]), x.arg) if x.src[0].dtype == dtypes.bfloat16.vec(16) else None),
      ])
    if self.arch.split(":")[0] == "gfx1201":
      self.extra_matcher += PatternMatcher([
        (UPat(Ops.WMMA, name="x", dtype=dtypes.bfloat16.vec(8)), lambda x: UOp(Ops.WMMA, dtypes.uint16.vec(8),
          (x.src[0].bitcast(dtypes.uint16.vec(8)), x.src[1].bitcast(dtypes.uint16.vec(8)), x.src[2].bitcast(dtypes.uint16.vec(8))), (*x.arg,))
            .bitcast(dtypes.bfloat16.vec(8)) if x.src[0].dtype == dtypes.bfloat16.vec(8) else None),
        (UPat(Ops.WMMA, name="x", dtype=dtypes.float.vec(8)),
          lambda x: UOp(Ops.WMMA, dtypes.float.vec(8), (x.src[0].bitcast(dtypes.uint16.vec(8)), x.src[1].bitcast(dtypes.uint16.vec(8)),
            x.src[2]), (*x.arg,)) if x.src[0].dtype == dtypes.bfloat16.vec(8) else None)
      ])
  def __reduce__(self): return self.__class__, (self.arch,)
