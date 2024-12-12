from typing import List, Dict, cast
import math, struct
from tinygrad.renderer import Renderer
from tinygrad.ops import UOp, PatternMatcher, UPat, Ops, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType, truncate

def ldt(dt:DType):
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
  if dtypes.is_unsigned(input_type) or input_type == dtypes.bool:
    if dtypes.is_float(output_type): return 'uitofp'
    if dtypes.is_int(output_type): return 'trunc' if output_type.itemsize < input_type.itemsize else 'zext'
  if dtypes.is_int(input_type):
    if dtypes.is_float(output_type): return 'sitofp'
    if dtypes.is_int(output_type): return 'trunc' if output_type.itemsize < input_type.itemsize else 'sext'
  raise NotImplementedError(f"cast from {input_type} -> {output_type} not implemented")

# llvm ops, lop[<dtype>][<op>]
unsigned_lop = { Ops.ADD: "add", Ops.MUL: "mul", Ops.IDIV: "udiv", Ops.MOD: "urem",
                 Ops.CMPLT: "icmp ult", Ops.CMPNE: "icmp ne", Ops.OR: "or", Ops.AND: "and", Ops.XOR: "xor", }
signed_lop = {**unsigned_lop, Ops.CMPLT: "icmp slt", Ops.IDIV: "sdiv", Ops.MOD: "srem"}
flags = " nsz arcp contract afn"
float_lop = {Ops.ADD: "fadd"+flags, Ops.MUL: "fmul"+flags, Ops.CMPLT: f"fcmp{flags} ult", Ops.CMPNE: f"fcmp{flags} une", Ops.FDIV: "fdiv"+flags}
lop = {**{x:unsigned_lop for x in (dtypes.bool,)+dtypes.uints}, **{x:signed_lop for x in dtypes.sints}, **{x:float_lop for x in dtypes.floats}}

llvm_rewrite = PatternMatcher([
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

  # unary/binary/ternary ops
  (UPat(Ops.SQRT, name="x"), lambda ctx,x:
   f"  {ctx[x]} = call{flags} {ldt(x.dtype)} @llvm.sqrt.{ldt(x.src[0].dtype)}({ldt(x.src[0].dtype)} {ctx[x.src[0]]})"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"  {ctx[x]} = bitcast {ldt(x.src[0].dtype)} {ctx[x.src[0]]} to {ldt(x.dtype)}"),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"  {ctx[x]} = {lcast(x.src[0].dtype, x.dtype)} {ldt(x.src[0].dtype)} {ctx[x.src[0]]} to {ldt(x.dtype)}"),
  (UPat(GroupOp.Binary, name="x"), lambda ctx,x: f"  {ctx[x]} = {lop[x.src[0].dtype][x.op]} {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, {ctx[x.src[1]]}"),
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
])

class LLVMRenderer(Renderer):
  device = "LLVM"
  supports_float4 = False
  has_local = False
  has_shared = False
  global_max = None

  extra_matcher = PatternMatcher([
    # rewrite RECIP with FDIV
    (UPat(Ops.RECIP, name="x"), lambda x: UOp(Ops.FDIV, x.dtype, (x.const_like(1), x.src[0]))),
    # rewrite cast to bool to CMPNE 0
    (UPat(Ops.CAST, dtype=dtypes.bool, name="x"), lambda x: x.src[0] != x.src[0].const_like(0)),
    # rewrite MAX to CMPLT + WHERE
    (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
  ])

  def render(self, name: str, uops: List[UOp]) -> str:
    r: Dict[UOp, str] = {}
    args: List[str] = []
    kernel: List[str] = []
    end_lines: Dict[str, None] = {}
    vc = -1

    # prealloc all assigns
    acc_to_assign: Dict[UOp, UOp] = {}
    for u in uops:
      if u.op is Ops.ASSIGN:
        vc += 1
        r[u] = r[u.src[1]] = f"%assign{vc}"
        assert u.src[0] not in acc_to_assign, "can't assign to DEFINE_ACC twice"
        acc_to_assign[u.src[0]] = u.src[1]

    for u in uops:
      # hack for defining sqrt function (TODO: can we get a transcendental for this?)
      if u.op is Ops.SQRT: end_lines[f'declare {ldt(u.dtype)} @llvm.sqrt.{ldt(u.dtype)}({ldt(u.dtype)} %".1")'] = None

      if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
        r[u] = f"%data{u.arg}" if u.op is Ops.DEFINE_GLOBAL else f"%{u.arg[0]}"
        args.append(f"{ldt(u.dtype)}{' noalias' if isinstance(u.dtype, PtrDType) else ''} {r[u]}")
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
        if (l:=llvm_rewrite.rewrite(u, ctx=r)) is None: raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
        kernel.append(cast(str, l))

        # generate the phi nodes for the assigns
        if u.op is Ops.RANGE:
          for x in acc_to_assign:
            if u in x.src:  # if this range is relevent for this acc
              vc += 1
              kernel.append(f"  %acc{vc} = phi {ldt(x.dtype)}" f"[{r[x]}, %loop_entry_{u.arg}], [{r[acc_to_assign[x]]}, %loop_latch_{u.arg}]")
              r[x] = f"%acc{vc}"

    # output the function
    return f"define void @{name}({','.join(args)}) {{\n" + '\n'.join(kernel) + "\n  ret void\n}\n"+'\n'.join(end_lines.keys())
