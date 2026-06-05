# flake8: noqa: E702
# allow semicolons to put multiple ops on one line
import sys, struct, functools
from typing import cast
from tinygrad.dtype import dtypes, PtrDType, DType, truncate, AddrSpace
from tinygrad.uop import FastEnum, auto, Ops, GroupOp
from tinygrad.uop.ops import UOp, UPat, PatternMatcher
from tinygrad.renderer.isa import ISARenderer, IselContext, Register, PreRegAllocContext
from tinygrad.helpers import getenv, CPU_COUNT, unwrap, Target

# ***** X86 Ops *****

class X86Ops(FastEnum):
  # NOTE: X86Ops with i suffix are variants that take an immediate, m suffix are variants that can write to memory instead of read from
  # these aren't real instructions
  FRAME_INDEX = auto(); LABEL = auto()
  # index
  LEA = auto()
  # register / memory / immediate moves
  MOV = auto(); MOVm = auto(); MOVi = auto(); MOVABS = auto()
  VMOVSS = auto(); VMOVSD = auto(); VMOVUPS = auto()
  VMOVSSm = auto(); VMOVSDm = auto(); VMOVUPSm = auto()
  # casts
  MOVZX = auto(); MOVSX = auto(); MOVSXD = auto()
  VPMOVZXBW = auto(); VPMOVZXBD = auto(); VPMOVZXBQ = auto()
  VPMOVZXWD = auto(); VPMOVZXWQ = auto(); VPMOVZXDQ = auto()
  VPMOVSXBW = auto(); VPMOVSXBD = auto(); VPMOVSXBQ = auto()
  VPMOVSXWD = auto(); VPMOVSXWQ = auto(); VPMOVSXDQ = auto()
  VCVTDQ2PS = auto(); VCVTDQ2PD = auto(); VCVTTPS2DQ = auto(); VCVTTPD2DQ = auto()
  VCVTPH2PS = auto(); VCVTPS2PH = auto(); VCVTPS2PD = auto(); VCVTPD2PS = auto()
  VCVTSS2SD = auto(); VCVTSD2SS = auto(); VCVTSI2SS = auto(); VCVTSI2SD = auto()
  VCVTTSS2SI = auto(); VCVTTSD2SI = auto()
  # bitcasts
  VMOVD = auto(); VMOVQ = auto(); VMOVDm = auto(); VMOVQm = auto()
  # comparisons
  VUCOMISS = auto(); VUCOMISD = auto()
  VCMPSS = auto(); VCMPSD = auto(); VCMPPS = auto(); VCMPPD = auto()
  VPCMPGTB = auto(); VPCMPGTW = auto(); VPCMPGTD = auto(); VPCMPGTQ = auto()
  VPCMPEQB = auto(); VPCMPEQW = auto(); VPCMPEQD = auto(); VPCMPEQQ = auto()
  SETNE = auto(); SETE = auto(); SETL = auto(); SETB = auto()
  # where
  CMOVNE = auto(); CMOVE = auto(); CMOVL = auto(); CMOVB = auto()
  VPBLENDVB = auto(); VBLENDVPS = auto(); VBLENDVPD = auto()
  # jumps
  JNE = auto(); JE = auto(); JL = auto(); JB = auto(); JGE = auto(); JMP = auto()
  # vectorize / gep
  VSHUFPS = auto(); VSHUFPD = auto(); VINSERTPS = auto(); VPSRLDQ = auto()
  VPEXTRB = auto(); VPEXTRW = auto(); VPEXTRD = auto(); VPEXTRQ = auto()
  VPINSRB = auto(); VPINSRW = auto(); VPINSRD = auto(); VPINSRQ = auto()
  VPBROADCASTB = auto(); VPBROADCASTW = auto(); VPBROADCASTD = auto(); VPBROADCASTQ = auto()
  VBROADCASTSS = auto()
  # int binary
  IDIV = auto(); DIV = auto()
  ADD = auto(); ADDi = auto(); SUB = auto(); SUBi = auto(); IMUL = auto(); IMULi = auto()
  AND = auto(); ANDi = auto(); XOR = auto(); XORi = auto(); OR = auto(); ORi = auto()
  SHL = auto(); SHLi = auto(); SHR = auto(); SHRi = auto(); SAR = auto(); SARi = auto(); CMP = auto(); CMPi = auto()
  # float unary (sometimes not unary)
  VROUNDSS = auto(); VROUNDSD = auto(); VROUNDPS = auto(); VROUNDPD = auto()
  VSQRTSS = auto(); VSQRTSD = auto(); VSQRTPS = auto(); VSQRTPD = auto()
  # float scalar / vector binary
  VADDSS = auto(); VADDSD = auto(); VADDPS = auto(); VADDPD = auto()
  VSUBSS = auto(); VSUBSD = auto(); VSUBPS = auto(); VSUBPD = auto()
  VMULSS = auto(); VMULSD = auto(); VMULPS = auto(); VMULPD = auto()
  VDIVSS = auto(); VDIVSD = auto(); VDIVPS = auto(); VDIVPD = auto()
  VMAXSS = auto(); VMAXSD = auto(); VMAXPS = auto(); VMAXPD = auto()
  VMINSS = auto(); VMINSD = auto(); VMINPS = auto(); VMINPD = auto()
  # int vector binary
  VPADDB = auto(); VPADDW = auto(); VPADDD = auto(); VPADDQ = auto()
  VPSUBB = auto(); VPSUBW = auto(); VPSUBD = auto(); VPSUBQ = auto()
  VPMULLW = auto(); VPMULLD = auto()
  # packed bitwise
  VPAND = auto(); VPOR = auto(); VPXOR = auto()
  # packed variable shifts
  VPSLLVD = auto(); VPSLLVQ = auto(); VPSRLVD = auto(); VPSRLVQ = auto(); VPSRAVD = auto()
  # fused multiply add
  VFMADD213SS = auto(); VFMADD213SD = auto(); VFMADD213PS = auto(); VFMADD213PD = auto()
  # return
  RET = auto()

class X86GroupOp:
  # X86Ops whose first src is also the destination
  TwoAddress = {X86Ops.ADD, X86Ops.ADDi, X86Ops.AND, X86Ops.ANDi, X86Ops.XOR, X86Ops.XORi, X86Ops.OR, X86Ops.ORi, X86Ops.IMUL,
                X86Ops.SUB, X86Ops.SUBi, X86Ops.SHL, X86Ops.SHLi, X86Ops.SHR, X86Ops.SHRi, X86Ops.SAR, X86Ops.SARi,
                X86Ops.IDIV, X86Ops.DIV, X86Ops.VFMADD213SS, X86Ops.VFMADD213SD, X86Ops.VFMADD213PS, X86Ops.VFMADD213PD,
                X86Ops.CMOVNE, X86Ops.CMOVE, X86Ops.CMOVL, X86Ops.CMOVB}

  # X86Ops whose first src can read from memory
  ReadMem1st = {X86Ops.MOV, X86Ops.VMOVSS, X86Ops.VMOVSD, X86Ops.VMOVUPS, X86Ops.MOVZX, X86Ops.MOVSX, X86Ops.MOVSXD, X86Ops.VMOVD, X86Ops.VMOVQ,
                X86Ops.VPMOVZXBW, X86Ops.VPMOVZXBD, X86Ops.VPMOVZXBQ, X86Ops.VPMOVZXWD, X86Ops.VPMOVZXWQ, X86Ops.VPMOVZXDQ,
                X86Ops.VPMOVSXBW, X86Ops.VPMOVSXBD, X86Ops.VPMOVSXBQ, X86Ops.VPMOVSXWD, X86Ops.VPMOVSXWQ, X86Ops.VPMOVSXDQ,
                X86Ops.VCVTDQ2PS, X86Ops.VCVTDQ2PD, X86Ops.VCVTTPS2DQ, X86Ops.VCVTTPD2DQ, X86Ops.VCVTTSS2SI, X86Ops.VCVTTSD2SI,
                X86Ops.VCVTPH2PS, X86Ops.VCVTPS2PD, X86Ops.VCVTPD2PS, X86Ops.VROUNDPS, X86Ops.VROUNDPD, X86Ops.VSQRTPS, X86Ops.VSQRTPD,
                X86Ops.VPBROADCASTB, X86Ops.VPBROADCASTW, X86Ops.VPBROADCASTD, X86Ops.VPBROADCASTQ, X86Ops.VBROADCASTSS,
                X86Ops.CMPi, X86Ops.IMULi, X86Ops.LEA}

  # X86Ops whose second src can read from memory NOTE: some of these are TwoAddress so the second src is actually the first
  ReadMem2nd = {X86Ops.ADD, X86Ops.SUB, X86Ops.AND, X86Ops.OR, X86Ops.XOR, X86Ops.SHL, X86Ops.SHR, X86Ops.SAR, X86Ops.IMUL, X86Ops.CMP,
                X86Ops.VADDSS, X86Ops.VADDSD, X86Ops.VADDPS, X86Ops.VADDPD, X86Ops.VSUBSS, X86Ops.VSUBSD, X86Ops.VSUBPS, X86Ops.VSUBPD,
                X86Ops.VMULSS, X86Ops.VMULSD, X86Ops.VMULPS, X86Ops.VMULPD, X86Ops.VDIVSS, X86Ops.VDIVSD, X86Ops.VDIVPS, X86Ops.VDIVPD,
                X86Ops.VPADDB, X86Ops.VPADDW, X86Ops.VPADDD, X86Ops.VPADDQ, X86Ops.VPSUBB, X86Ops.VPSUBW, X86Ops.VPSUBD, X86Ops.VPSUBQ,
                X86Ops.VPCMPEQB, X86Ops.VPCMPEQW, X86Ops.VPCMPEQD, X86Ops.VPCMPEQQ, X86Ops.VPBLENDVB, X86Ops.VBLENDVPS, X86Ops.VBLENDVPD,
                X86Ops.VPCMPGTB, X86Ops.VPCMPGTW, X86Ops.VPCMPGTD, X86Ops.VPCMPGTQ, X86Ops.VCMPSS, X86Ops.VCMPSD, X86Ops.VCMPPS, X86Ops.VCMPPD,
                X86Ops.VPMULLW, X86Ops.VPMULLD, X86Ops.VROUNDSS, X86Ops.VROUNDSD, X86Ops.VSQRTSS, X86Ops.VSQRTSD, X86Ops.VSHUFPS, X86Ops.VINSERTPS,
                X86Ops.VPINSRB, X86Ops.VPINSRW, X86Ops.VPINSRD, X86Ops.VPINSRQ, X86Ops.VPAND, X86Ops.VPOR, X86Ops.VPXOR, X86Ops.VPSLLVD,
                X86Ops.VPSLLVQ, X86Ops.VPSRLVD, X86Ops.VPSRLVQ, X86Ops.VPSRAVD, X86Ops.CMOVNE, X86Ops.CMOVE, X86Ops.CMOVL, X86Ops.CMOVB,
                X86Ops.VMAXSS, X86Ops.VMAXSD, X86Ops.VMAXPS, X86Ops.VMAXPD, X86Ops.VMINSS, X86Ops.VMINSD, X86Ops.VMINPS, X86Ops.VMINPD,
                X86Ops.VCVTSI2SS, X86Ops.VCVTSI2SD, X86Ops.VCVTSS2SD, X86Ops.VCVTSD2SS, X86Ops.VUCOMISS, X86Ops.VUCOMISD, X86Ops.IDIV, X86Ops.DIV,
                X86Ops.VSHUFPD}

  # X86Ops whose third src can read from memory NOTE: these are TwoAddress so the third src is actually the second
  ReadMem3rd = {X86Ops.VFMADD213SS, X86Ops.VFMADD213SD, X86Ops.VFMADD213PS, X86Ops.VFMADD213PD}

  # X86Ops that can write to memory
  WriteMem = {X86Ops.MOVm, X86Ops.MOVi, X86Ops.VMOVSSm, X86Ops.VMOVSDm, X86Ops.VMOVUPSm, X86Ops.VMOVDm, X86Ops.VMOVQm,
              X86Ops.ADDi, X86Ops.SUBi, X86Ops.ANDi, X86Ops.ORi, X86Ops.XORi, X86Ops.SHLi, X86Ops.SHRi, X86Ops.SARi, X86Ops.SETNE,
              X86Ops.SETE, X86Ops.SETL, X86Ops.SETB, X86Ops.VCVTPS2PH, X86Ops.VPEXTRB, X86Ops.VPEXTRW, X86Ops.VPEXTRD, X86Ops.VPEXTRQ}

  # X86Ops that read flags
  ReadFlags = {X86Ops.CMOVB, X86Ops.CMOVL, X86Ops.CMOVE, X86Ops.CMOVNE, X86Ops.SETB, X86Ops.SETL, X86Ops.SETE, X86Ops.SETNE, X86Ops.JB, X86Ops.JL,
               X86Ops.JE, X86Ops.JNE, X86Ops.JGE}

  # X86Ops that write flags or can modify flags to undefined values
  WriteFlags = {X86Ops.CMP, X86Ops.CMPi, X86Ops.ADD, X86Ops.ADDi, X86Ops.SUB, X86Ops.SUBi, X86Ops.IMUL, X86Ops.IMULi, X86Ops.IDIV, X86Ops.DIV,
                X86Ops.SHL, X86Ops.SHLi, X86Ops.SHR, X86Ops.SHRi, X86Ops.SAR, X86Ops.SARi, X86Ops.AND, X86Ops.ANDi, X86Ops.XOR, X86Ops.XORi,
                X86Ops.OR, X86Ops.ORi, X86Ops.VUCOMISS, X86Ops.VUCOMISD}

  # X86Ops whose first src is the rm field
  Rm1st = ReadMem1st | (ReadMem2nd & TwoAddress) | {X86Ops.VPSRLDQ}

  # X86Ops whose second src is the rm field
  Rm2nd = ReadMem2nd | (ReadMem3rd & TwoAddress)

  All = set(X86Ops)

# ***** X86 legalization *****

extra_matcher = PatternMatcher([
  # bool CMPNE is XOR, bool CMPEQ is XOR+XOR, bool CMPLT is XOR+AND
  (UPat.var('x', dtypes.bool).ne(UPat.var('y')), lambda x,y: x^y),
  (UPat.var('x', dtypes.bool).alu(Ops.CMPEQ, UPat.var('y')), lambda x,y: (x^y)^True),
  (UPat.var('x', dtypes.bool)<UPat.var('y'), lambda x,y: (x^True)&y),
  # can't cast from float16 to ints/float64 directly and vice versa
  (UPat.var("y", dtypes.float16).cast((dtypes.float64,)+dtypes.ints, name="x"), lambda y,x: y.cast(dtypes.float32).cast(x.dtype)),
  (UPat.var("y", (dtypes.float64,)+dtypes.ints).cast(dtypes.float16, name="x"), lambda y,x: y.cast(dtypes.float32).cast(x.dtype)),
  # can't cast from float to int8/16 directly and vice versa
  (UPat.var("y", dtypes.floats).cast(dtypes.int8s+dtypes.int16s, name="x"), lambda y,x: y.cast(dtypes.int32).cast(x.dtype)),
  (UPat.var("y", (dtypes.bool,)+dtypes.int8s+dtypes.int16s).cast(dtypes.floats, name="x"), lambda y,x: y.cast(dtypes.int32).cast(x.dtype)),
  # int/float casts only for signed int
  (UPat.var("y", dtypes.uint32).cast(dtypes.floats, name="x"), lambda y,x: y.cast(dtypes.int64).cast(x.dtype)),
  # casting uint64 to float requires special handling
  (UPat.var("y", dtypes.uint64).cast(dtypes.floats, name="x"), lambda y,x:
   (y >> 1).cast(dtypes.int64).cast(x.dtype) * 2 + (y & 1).cast(dtypes.int64).cast(x.dtype)),
  # no int8 mul or cmove, cast to int16
  (UPat.var("a", dtypes.int8s) * UPat.var("b"), lambda a,b: (a.cast(dtypes.int16) * b.cast(dtypes.int16)).cast(a.dtype)),
  (UPat.var("m").where(UPat.var("a", (dtypes.bool,)+dtypes.int8s), UPat.var("b")),
   lambda m,a,b: m.where(a.cast(dtypes.int16), b.cast(dtypes.int16)).cast(a.dtype) if a.dtype.count == 1 else None),
  # float16 alus are done in float32
  (UPat(GroupOp.ALU, dtypes.float16, name="x"), lambda x: UOp(x.op, dtypes.float.vec(x.dtype.count),
   tuple(s.cast(dtypes.float) if s.dtype != dtypes.bool else s for s in x.src)).cast(x.dtype)),
  (UPat(GroupOp.Comparison, src=(UPat.var("a", dtypes.float16), UPat.var("b")), name="x"),
   lambda x,a,b: UOp(x.op, x.dtype, (a.cast(dtypes.float32), b.cast(dtypes.float32))).cast(x.dtype)),
  # no cmpne for packed ints, y != x => !(y==x)
  (UPat(Ops.CMPNE, src=(UPat.var("y", dtypes.ints), UPat.var("x")), name="cmp"),
   lambda y,x,cmp: UOp(Ops.CMPEQ, cmp.dtype, (y,x))^True if y.dtype.count > 1 else None),
  # float where expects a mask
  (UPat.var("m", dtypes.bool).where(UPat.var("a", dtypes.floats), UPat.var("b")),
   lambda m,a,b: m.cast(a.dtype).ne(0).where(a, b) if m.src[0].dtype not in dtypes.floats else None),
  # rewrite -x -> 0 - x
  (UPat(Ops.NEG, name="x"), lambda x: UOp(Ops.SUB, x.dtype, (x.const_like(0),) + x.src)),
  # TODO: add support for mod, requires support for accessing the 2nd+ reg of a multi output instruction
  (UPat(Ops.CMOD, src=(UPat.var("x"), UPat.var("y"))), lambda x,y: x - y * x.alu(Ops.CDIV, y)),
])

# ***** X86 pre instruction selection *****

def gated_load(ctx, base:UOp, idx:UOp, cast:UOp, alt:UOp, gate:UOp, x:UOp):
  local = UOp(Ops.DEFINE_LOCAL, base.dtype.base.ptr(x.dtype.count, AddrSpace.LOCAL), arg=next(ctx))
  local_idx = local.index(UOp.const(dtypes.int32, 0), ptr=True)
  ptr = gate.where(base.index(idx, ptr=True), local_idx).after((local_idx if x.dtype.count == 1 else local).store(alt))
  return ptr.cast(cast.dtype).load(dtype=x.dtype)

def gated_store(base:UOp, idx:UOp, cast:UOp, gate:UOp, val:UOp):
  local = UOp(Ops.DEFINE_LOCAL, base.dtype.base.ptr(val.dtype.count, AddrSpace.LOCAL), arg=-1)
  ptr = gate.where(base.index(idx, ptr=True), local.index(UOp.const(dtypes.int32, 0), ptr=True))
  return ptr.cast(cast.dtype).store(val)

# these must be done in a separate matcher because they violate the spec
pre_isel_matcher = PatternMatcher([
  # zero extending scalar 32bit int is a noop
  (UPat.var("y", dtypes.uint32).cast(dtypes.int64s, name="x"), lambda y,x: x.replace(op=Ops.NOOP) if y.dtype.count == 1 else None),
  # cast between signed and unsigned int is a noop
  (UPat.var("y", dtypes.ints+(dtypes.bool,)).cast(dtypes.ints, name="x"),
   lambda y,x: x.replace(op=Ops.NOOP) if x.dtype.itemsize == y.dtype.itemsize else None),
  # cast to < scalar int is a noop
  (UPat.var("y", dtypes.ints).cast(dtypes.ints, name="x"),
   lambda y,x: x.replace(op=Ops.NOOP) if x.dtype.itemsize < y.dtype.itemsize and y.dtype.count == 1 else None),
  # bitcasts between scalar floats and ints are real, rest are noops
  (UPat.var("y").bitcast().named("x"), lambda y,x: None if y.dtype in dtypes.floats and x.dtype in dtypes.ints or \
   y.dtype in dtypes.ints and x.dtype in dtypes.floats else x.replace(op=Ops.NOOP)),
  # noop of a noop is removed
  (UPat(Ops.NOOP, src=(UPat(Ops.NOOP),), name="x"), lambda x: x.replace(src=x.src[0].src)),
  # moving elements of a single register to another without shuffling is a noop
  (UPat(Ops.STACK, src=(UPat.var("y"),), allow_any_len=True, name="x"),
   lambda y,x: UOp(Ops.NOOP, x.dtype, y.src) if all(s.op is Ops.GEP and s.src == y.src and s.arg[0] == i for i,s in enumerate(x.src)) else None),
  # gated load/store become a conditional move on the index, the load/store are unconditional
  (UPat.var("base").index(UPat.var("idx")).or_casted(name="cast").load(UPat.var("alt"), UPat.var("gate"), name="x"), gated_load),
  (UPat.var("base").index(UPat.var("idx")).or_casted(name="cast").store(UPat.var("val"), UPat.var("gate")), gated_store),
  # TODO: remove this once we allow all flag producing ops in cmove
  # if gate in scalar int cmove is not a comparison need to add one to set the flag
  (UPat.var("m", dtypes.bool).where(UPat.var("a"), UPat.var("b")),
   lambda m,a,b: m.ne(0).where(a,b) if m.op not in GroupOp.Comparison and a.dtype.count == 1 else None),
])

# ***** X86 registers *****

RAX = Register("rax", 0)
RCX = Register("rcx", 1)
RDX = Register("rdx", 2)
RBX = Register("rbx", 3)
RSP = Register("rsp", 4)
RBP = Register("rbp", 5)
RSI = Register("rsi", 6)
RDI = Register("rdi", 7)
GPR = (RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI) + tuple(Register(f"r{i}", i) for i in range(8, 16))
XMM = tuple(Register(f"xmm{i}", i) for i in range(16))
# gprs you can write to
WGPR = tuple(r for r in GPR if r != RSP)

CALLEE_SAVED = (RBX, RBP, GPR[12], GPR[13], GPR[14], GPR[15]) + ((RSI, RDI) + XMM[6:16] if sys.platform == "win32" else ())

reg_strs = {"rax": {4:"eax", 2:"ax", 1:"al"}, "rcx": {4:"ecx", 2:"cx", 1:"cl"}, "rdx": {4:"edx", 2:"dx", 1:"dl"}, "rbx": {4:"ebx", 2:"bx", 1:"bl"},
        "rsp": {4:"esp", 2:"sp", 1:"spl"}, "rbp": {4:"ebp", 2:"bp", 1:"bpl"}, "rsi": {4:"esi", 2:"si", 1:"sil"}, "rdi": {4:"edi", 2:"di", 1:"dil"},
        **{f"r{i}": {4:f"r{i}d", 2:f"r{i}w", 1:f"r{i}b"} for i in range(8, 16)}, **{f"xmm{i}": {64:f"zmm{i}", 32:f"ymm{i}"} for i in range(16)}}

# ***** X86 instruction selection *****
# if s is used multiple times we don't fold
def is_foldable(ctx:IselContext, x:UOp, s:UOp) -> bool: return len(ctx.uses[s]) == x.src.count(s) == 1
def base(x:UOp, i:int) -> UOp: return s.src[0] if (s:=x.src[i]).op is Ops.GEP else s
def lane(x:UOp, i:int) -> int: return s.arg[0] if (s:=x.src[i]).op is Ops.GEP else 0
def to_int(dt:DType): return {dtypes.float16: dtypes.int16, dtypes.float32: dtypes.int32, dtypes.float64: dtypes.int64}[dt]
def def_reg(dt:DType, reg:Register|None=None) -> UOp: return UOp(Ops.DEFINE_REG, dt, tag=None if reg is None else (reg,))
def imm(dt:DType, v:int) -> UOp: return UOp.const(dt, truncate[dt](v)).rtag()
def to_imm(c:UOp) -> UOp|None:
  if c.op is not Ops.CONST: return None
  if c.dtype is dtypes.int64: return imm(dtypes.int32, c.arg) if not c.overflows(dtypes.int32) else None
  if c.dtype is dtypes.uint64: return imm(dtypes.uint32, c.arg) if not c.overflows(dtypes.uint32) else None
  if c.dtype in dtypes.ints+(dtypes.bool,): return imm(c.dtype, c.arg)
  return None
def cmp(x:UOp) -> UOp:
  if x.src[0].dtype is dtypes.float32: return x.ins(X86Ops.VUCOMISS, dtype=dtypes.void)
  if x.src[0].dtype is dtypes.float64: return x.ins(X86Ops.VUCOMISD, dtype=dtypes.void)
  return x.ins(X86Ops.CMP, dtype=dtypes.void) if (i:=to_imm(x.src[1])) is None else x.ins(X86Ops.CMPi, dtype=dtypes.void, src=(x.src[0], i))
def vcmp(x:UOp) -> UOp:
  v = imm(dtypes.uint8, {Ops.CMPLT: 1, Ops.CMPNE: 4, Ops.CMPEQ: 0}[x.op])
  if x.dtype.scalar() is dtypes.float32: return x.ins(X86Ops.VCMPSS if x.dtype.count == 1 else X86Ops.VCMPPS, src=x.src + (v,))
  return x.ins(X86Ops.VCMPSD if x.dtype.count == 1 else X86Ops.VCMPPD, src=x.src + (v,))

# vshufps xmm2, xmm0, xmm1, imm
# for 128 bit xmm2 selects its lower 2 32 bits from xmm0 and its upper 2 32 bits from xmm1 according to imm
# for 256 bit ymm2 repeats the shuffle for its upper 128 bits selecting from the upper 128 bits of ymm0 and ymm1
def vshufps(x:UOp) -> UOp|None:
  a, b = base(x, 0), base(x, 2)
  if not (a is base(x, 1) and b is base(x, 3)) or any(lane(x, i) > 3 for i in range(4)): return None
  if len(x.src) == 8:
    if not (a is base(x, 4) is base(x, 5) and b is base(x, 6) is base(x, 7)) or any(lane(x, i+4) != lane(x, i)+4 for i in range(4)): return None
  return x.ins(X86Ops.VSHUFPS, src=(a, b, imm(dtypes.uint8, sum(lane(x, i) << 2*i for i in range(4)))))

# vshufpd xmm2, xmm0, xmm1, imm
# for 128 bit xmm2 selects its lower 64 bits from xmm0 and its upper 64 bits from xmm1 according to imm
# for 256 bit ymm2 also selects its upper 128 bits from the upper 128 bits of ymm0 and ymm1 following the same constraint
def vshufpd(x:UOp) -> UOp|None:
  a, b = base(x, 0), base(x, 1)
  if lane(x, 0) > 1 or lane(x, 1) > 1: return None
  if len(x.src) == 4 and not (a is base(x, 2) and b is base(x, 3) and lane(x, 2) > 1 and lane(x, 3) > 1): return None
  return x.ins(X86Ops.VSHUFPD, src=(a, b, imm(dtypes.uint8, sum(lane(x, i) << i for i in range(len(x.src))))))

# vinsertps xmm2, xmm0, xmm1, imm
# inserts any 32 bit element in xmm1 into any position in xmm0 according to immm, result is written to xmm2
# this is the fallback slow case for when you can't match more a powerful shuffle
def vinsertps(x:UOp) -> UOp:
  def _insert(ret:UOp, i:int) -> UOp:
    s, v = base(x, i), lane(x, i)
    # moving the 0th element into the 0th position does nothing
    return s if i == v == 0 else x.ins(X86Ops.VINSERTPS, src=(ret, s, imm(dtypes.uint8, v << 6 | i << 4)))
  return functools.reduce(_insert, range(len(x.src)), def_reg(x.dtype))

# vpinsq xmm2, xmm0, rax, imm
# inserts element in rax into any position in xmm0, result is written to xmm2 according to imm
def vpins(x:UOp) -> UOp:
  op = {1: X86Ops.VPINSRB, 2: X86Ops.VPINSRW, 4: X86Ops.VPINSRD, 8: X86Ops.VPINSRQ}[x.dtype.scalar().itemsize]
  return functools.reduce(lambda ret,i: x.ins(op, src=(ret, x.src[i], imm(dtypes.uint8, i))), range(len(x.src)), def_reg(x.dtype))

# vpbroadcastd xmm1, xmm0
# inserts scalar int in xmm0 into all lanes of xmm1
def vpbroadcast(ctx:IselContext, x:UOp, y:UOp) -> UOp:
  n = x.ins({1: X86Ops.VPBROADCASTB, 2: X86Ops.VPBROADCASTW, 4: X86Ops.VPBROADCASTD, 8: X86Ops.VPBROADCASTQ}[y.dtype.itemsize], src=(y,))
  if y.op is Ops.LOAD and len(y.src) == 1 and is_foldable(ctx, n, y): return n
  # if there isn't a load we can fold we need to move y from gpr to xmm
  # this is hacky but required because int.vec(1) isn't supported
  y = y if y.dtype.itemsize > 1 else y.cast(dtypes.int16)
  return n.replace(src=(y.bitcast({2:dtypes.float16, 4:dtypes.float32, 8:dtypes.float64}[y.dtype.itemsize]),))

# we don't call ctx.vreg on the srcs to avoid duplicates, a rewrite will assign the tuple of valid registers to a vreg
def idiv(ctx:IselContext, x:UOp) -> UOp:
  op = X86Ops.DIV if x.dtype in dtypes.uints else X86Ops.IDIV
  # for >8bit need to zero/sign extend rax to rdx
  if x.dtype in dtypes.int8s: ext = []
  elif x.dtype in dtypes.uints: ext = [x.ins(X86Ops.MOVi, src=(imm(min(dtypes.uint32, x.dtype), 0),), tag=(RDX,))]
  else: ext = [x.ins(X86Ops.SARi, src=(x.src[0], imm(dtypes.uint8, x.dtype.itemsize * 8 - 1)), tag=(RDX,))]
  # for 8bit need to zero/sign extend al to ah
  if x.dtype is dtypes.uint8: dividend = UOp(Ops.INS, arg=X86Ops.MOVZX, dtype=dtypes.int16, src=(x.src[0],), tag=(RAX,))
  elif x.dtype is dtypes.int8: dividend = UOp(Ops.INS, arg=X86Ops.MOVSX, dtype=dtypes.int16, src=(x.src[0],), tag=(RAX,))
  else: dividend = x.ins(X86Ops.MOV, src=(x.src[0],), tag=(RAX,))
  # divisor can't be in rax or rdx
  divisor = x.ins(X86Ops.MOV, src=(x.src[1],), tag=tuple(r for r in WGPR if r not in (RAX, RDX)))
  # for >8bit both rax and rdx are written to
  defs = (ctx.vreg(RAX),) if x.dtype in dtypes.int8s else (ctx.vreg(RAX), ctx.vreg(RDX))
  idiv = x.ins(op, src=(dividend, divisor) + tuple(ext), tag=defs)
  # this move "cleanses" the register constraints (rax/rdx) of idiv as that only applies on definition and not on the uses of idiv
  return x.ins(X86Ops.MOV, src=(idiv,))

def fold_address(x:UOp) -> tuple[UOp, UOp, UOp]:
  def _disp(v:int) -> UOp: return imm(dtypes.int32 if abs(v) > dtypes.int8.max else dtypes.int8, v)
  def _cast(v:UOp) -> UOp: return v.cast(dtypes.int64) if v.vmin < 0 else v
  if x.op is not Ops.INDEX: return (x, UOp(Ops.NOOP), _disp(0))
  base, idx = x.src
  disp_scale = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 1
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: return (base, _cast(idx.src[0]), _disp(idx.src[1].arg * disp_scale))
  if idx.op is Ops.CONST: return (base, UOp(Ops.NOOP), _disp(idx.arg * disp_scale))
  return (base, _cast(idx), _disp(0))

def abi(ctx:IselContext, x:UOp) -> UOp|None:
  if isinstance(x.tag, tuple): return None
  i = ctx.func_args.index(x)
  def _stack_arg(disp:int): return (def_reg(dtypes.uint64, RSP), UOp(Ops.NOOP), UOp(Ops.INS, arg=X86Ops.FRAME_INDEX, dtype=dtypes.int32, tag=disp))
  if sys.platform == "win32": src = (x.replace(tag=((RCX, RDX, GPR[8], GPR[9])[i],)),) if i < 4 else _stack_arg((i-3)*8+32)
  else: src = (x.replace(tag=((RDI, RSI, RDX, RCX, GPR[8], GPR[9])[i],)),) if i < 6 else _stack_arg((i-5)*8)
  # this move "cleanses" the abi register constraint
  return x.ins(X86Ops.MOV, src=src)

def alloc_vregs(ctx:IselContext, x:UOp) -> UOp|None:
  # real registers
  if x.op is Ops.DEFINE_REG and x.tag is not None: return None
  # this is an immediate
  if x.arg is X86Ops.FRAME_INDEX: return None
  # no register definition
  if x.dtype is dtypes.void: return None
  # already allocated vregs
  if isinstance(x.tag, tuple) and x.tag[0]._cons: return None
  # allocate vreg definitions
  defs = []
  if isinstance(x.tag, tuple): defs = [ctx.vreg(x.tag)]
  elif x.dtype in dtypes.ints+(dtypes.bool,) or isinstance(x.dtype, PtrDType): defs = [ctx.vreg(WGPR)]
  elif x.dtype in dtypes.floats or x.dtype.count > 1: defs = [ctx.vreg(XMM)]
  # TODO: add this once the scheduler can track register pressure
  # if x.arg in X86GroupOp.WriteFlags: defs.append(ctx.vreg(RFLAGS))
  return x.replace(tag=tuple(defs))

dts = dtypes.ints + (dtypes.bool, dtypes.float16, dtypes.float32, dtypes.float64)
dt_16bit = tuple(dt.vec(l) for dt in dts for l in [2,1] if l*dt.itemsize == 2 and dt not in dtypes.int16s)
dt_32bit = tuple(dt.vec(l) for dt in dts for l in [4,2,1] if l*dt.itemsize == 4 and dt not in dtypes.int32s)
dt_64bit = tuple(dt.vec(l) for dt in dts for l in [8,4,2,1] if l*dt.itemsize == 8 and dt not in dtypes.int64s)
dt_128bit = tuple(dt.vec(l) for dt in dts for l in [16,8,4,2,1] if l*dt.itemsize == 16)

isel_matcher = PatternMatcher([
  # **** Op -> Op ****
  # cast to pointer is a noop
  (UPat.var("y").cast(name="x"), lambda y,x: y if isinstance(x.dtype, PtrDType) or y.dtype == dtypes.void else None),
  # float gep(0) is a noop as it just moves the 0th element from one xmm register to another
  # this is done here to not interfere with shuffles
  (UPat(dtype=dtypes.floats).gep(0, name="x"), lambda x: x.replace(op=Ops.NOOP, arg=None)),
  # range is lowered to acc, cmp, jmp after regalloc
  (UPat(Ops.RANGE, src=(UPat.cvar("c"),), allow_any_len=True, name="x"), lambda c,x: x.replace(src=(imm(c.dtype, c.arg),) + x.src[1:])),
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: x.replace(tag=(ctx.vreg(WGPR),)) if not isinstance(x.tag, tuple) else None),
  # **** Op -> X86Op ****
  # add callee saved registers to the RET, these will be scheduled at the top of the kernel and will be saved/restored if they are used in regalloc
  # so regalloc builds the prologue/epilogue naturally
  (UPat(Ops.SINK, name="x"), lambda x:
   x.replace(src=(x.ins(X86Ops.RET, src=x.src + tuple(def_reg(dtypes.uint64 if r in GPR else dtypes.float64.vec(2), r) for r in CALLEE_SAVED)),)) \
    if not x.src or x.src[0].arg is not X86Ops.RET else None),
  # function abi constraints
  (UPat((Ops.PARAM, Ops.DEFINE_VAR, Ops.SPECIAL), name="x"), abi),
  # these are treated the same for now
  (UPat(Ops.DEFINE_REG, name="x"), lambda x:
   x.replace(op=Ops.DEFINE_LOCAL, dtype=x.dtype.base.ptr(x.dtype.size, AddrSpace.LOCAL)) if isinstance(x.arg, int) else None),
  # constants that can't be immediates, move them to registers
  (UPat.cvar("x", dtypes.int64s), lambda x: x.ins(X86Ops.MOVABS, src=(imm(x.dtype, x.arg),)) if not x.tag else None),
  (UPat.cvar("x", dtypes.ints+(dtypes.bool,)), lambda x: x.ins(X86Ops.MOVi, src=(imm(x.dtype, x.arg),)) if not x.tag else None),
  (UPat.cvar("x", dtypes.floats), lambda x:
   UOp.const(dt:=to_int(x.dtype), struct.unpack(dt.fmt, struct.pack(x.dtype.fmt, x.arg))[0]).bitcast(x.dtype) if not x.tag else None),
  # TODO: these should use a.maximum(b) / a.minimum(b)
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("b", dtypes.float32), UPat.var("a")), lambda a,b:
   a.ins(X86Ops.VMAXSS if a.dtype.count == 1 else X86Ops.VMAXPS, src=(a, b))),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("b", dtypes.float64), UPat.var("a")), lambda a,b:
   a.ins(X86Ops.VMAXSD if a.dtype.count == 1 else X86Ops.VMAXPD, src=(a, b))),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("a", dtypes.float32), UPat.var("b")), lambda a,b:
   a.ins(X86Ops.VMINSS if a.dtype.count == 1 else X86Ops.VMINPS, src=(a, b))),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("a", dtypes.float64), UPat.var("b")), lambda a,b:
   a.ins(X86Ops.VMINSD if a.dtype.count == 1 else X86Ops.VMINPD, src=(a, b))),
  # conditional moves that use masks NOTE: these currently assume a mask producing cmp exists
  (UPat.var("m").where(UPat.var("a", dtypes.ints), UPat.var("b")), lambda m,a,b:
   a.ins(X86Ops.VPBLENDVB, src=(b, a, m.replace(dtype=m.src[0].dtype))) if a.dtype.count > 1 else None),
  (UPat.var("m").where(UPat.var("a", dtypes.float32), UPat.var("b")), lambda m,a,b:
   a.ins(X86Ops.VBLENDVPS, src=(b, a, m.replace(dtype=m.src[0].dtype)))),
  (UPat.var("m").where(UPat.var("a", dtypes.float64), UPat.var("b")), lambda m,a,b:
   a.ins(X86Ops.VBLENDVPD, src=(b, a, m.replace(dtype=m.src[0].dtype)))),
  # in this case we have a mask producing comparison whose user expects a bool, so we convert to bool
  (UPat(GroupOp.Comparison, dtypes.bool, (UPat.var("y", (dtypes.float32, dtypes.float64)), UPat()), name="x"), lambda y,x:
   x.replace(dtype=y.dtype).bitcast(to_int(y.dtype)).bitwise_and(1).f(Ops.NOOP, dtype=dtypes.bool)),
  # conditional moves that use flags
  (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.sints), UPat()), name="m").where(UPat.var("a"), UPat.var("b")), lambda m,a,b:
   a.ins(X86Ops.CMOVL, src=(b, a, cmp(m)))),
  (UPat(Ops.CMPLT, name="m").where(UPat.var("a"), UPat.var("b")), lambda m,a,b: a.ins(X86Ops.CMOVB, src=(b, a, cmp(m)))),
  (UPat(Ops.CMPEQ, name="m").where(UPat.var("a"), UPat.var("b")), lambda m,a,b: a.ins(X86Ops.CMOVE, src=(b, a, cmp(m)))),
  (UPat(Ops.CMPNE, name="m").where(UPat.var("a"), UPat.var("b")), lambda m,a,b: a.ins(X86Ops.CMOVNE, src=(b, a, cmp(m)))),
  # jumps, use flags
  (UPat(Ops.IF, src=(UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.uints), UPat()), name="y"),), name="x"), lambda y,x: x.ins(X86Ops.JB, src=(cmp(y),))),
  (UPat(Ops.IF, src=(UPat(Ops.CMPLT, name="y"),), name="x"), lambda y,x: x.ins(X86Ops.JL, src=(cmp(y),))),
  (UPat(Ops.IF, src=(UPat(Ops.CMPEQ, name="y"),), name="x"), lambda y,x: x.ins(X86Ops.JE, src=(cmp(y),))),
  (UPat(Ops.IF, src=(UPat(Ops.CMPNE, name="y"),), name="x"), lambda y,x: x.ins(X86Ops.JNE, src=(cmp(y),))),
  # comparisons whose user doesn't use the flag, move flag result to register
  (UPat(Ops.CMPLT, dtypes.bool, (UPat(dtype=dtypes.uints), UPat()), name="x"), lambda x: x.ins(X86Ops.SETB, src=(cmp(x),))),
  (UPat(Ops.CMPLT, dtypes.bool, name="x"), lambda x: x.ins(X86Ops.SETL, src=(cmp(x),))),
  (UPat(Ops.CMPEQ, dtypes.bool, name="x"), lambda x: x.ins(X86Ops.SETE, src=(cmp(x),))),
  (UPat(Ops.CMPNE, dtypes.bool, name="x"), lambda x: x.ins(X86Ops.SETNE, src=(cmp(x),))),
  # comparisons that produce masks (these aren't bool dtype)
  (UPat(GroupOp.Comparison, src=(UPat(dtype=(dtypes.float32, dtypes.float64)), UPat()), name="x"), vcmp),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.int8s), UPat()), name="x"), lambda x: x.ins(X86Ops.VPCMPEQB)),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.int16s), UPat()), name="x"), lambda x: x.ins(X86Ops.VPCMPEQW)),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.int32s), UPat()), name="x"), lambda x: x.ins(X86Ops.VPCMPEQD)),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.int64s), UPat()), name="x"), lambda x: x.ins(X86Ops.VPCMPEQQ)),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.int8s), UPat.var("b")), name="x"), lambda a,b,x: x.ins(X86Ops.VPCMPGTB, src=(b, a))),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.int16s), UPat.var("b")), name="x"), lambda a,b,x: x.ins(X86Ops.VPCMPGTW, src=(b, a))),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.int32s), UPat.var("b")), name="x"), lambda a,b,x: x.ins(X86Ops.VPCMPGTD, src=(b, a))),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.int64s), UPat.var("b")), name="x"), lambda a,b,x: x.ins(X86Ops.VPCMPGTQ, src=(b, a))),
  # float unary
  (UPat.var("y", dtypes.float32).sqrt().named("x"), lambda y,x: x.ins(X86Ops.VSQRTSS, src=(y, y)) if x.dtype.count == 1 else x.ins(X86Ops.VSQRTPS)),
  (UPat.var("y", dtypes.float64).sqrt().named("x"), lambda y,x: x.ins(X86Ops.VSQRTSD, src=(y, y)) if x.dtype.count == 1 else x.ins(X86Ops.VSQRTPD)),
  (UPat.var("y", dtypes.float32).trunc().named("x"), lambda y,x:
   x.ins(X86Ops.VROUNDSS, src=(y, y, imm(dtypes.uint8, 3))) if x.dtype.count == 1 else x.ins(X86Ops.VROUNDPS, src=(y, imm(dtypes.uint8, 3)))),
  (UPat.var("y", dtypes.float64).trunc().named("x"), lambda y,x:
   x.ins(X86Ops.VROUNDSD, src=(y, y, imm(dtypes.uint8, 3))) if x.dtype.count == 1 else x.ins(X86Ops.VROUNDPD, src=(y, imm(dtypes.uint8, 3)))),
  # shufles
  (UPat.var("y", dtypes.float32).broadcast(name="x"), lambda y,x: x.ins(X86Ops.VBROADCASTSS, src=(y,))),
  # for float16 we route the srcs through gprs unless we can fold them, this is suboptimal for values in xmms, in that case we want vpunpcklwd
  (UPat(Ops.STACK, dtypes.float16, name="x"), lambda ctx,x:
   vpins(x.replace(src=tuple(s if s.op is Ops.LOAD and is_foldable(ctx, x, s) else s.bitcast(dtypes.int16) for s in x.src)))),
  (UPat(Ops.STACK, (dtypes.float32.vec(4), dtypes.float32.vec(8)), name="x"), vshufps),
  (UPat(Ops.STACK, (dtypes.float64.vec(2), dtypes.float64.vec(4)), name="x"), vshufpd),
  (UPat(Ops.STACK, dtypes.float32, name="x"), vinsertps),
  (UPat.var("y", dtypes.ints+(dtypes.bool,)).broadcast(name="x"), vpbroadcast),
  (UPat(Ops.STACK, dtypes.ints+(dtypes.bool,), name="x"), vpins),
  # gep
  (UPat.var("y", dtypes.int8s+(dtypes.bool,)).gep(name="x"), lambda y,x: x.ins(X86Ops.VPEXTRB, src=(y, imm(dtypes.uint8, x.arg[0])))),
  (UPat.var("y", dtypes.int16s).gep(name="x"), lambda y,x: x.ins(X86Ops.VPEXTRW, src=(y, imm(dtypes.uint8, x.arg[0])))),
  (UPat.var("y", dtypes.int32s).gep(name="x"), lambda y,x: x.ins(X86Ops.VPEXTRD, src=(y, imm(dtypes.uint8, x.arg[0])))),
  (UPat.var("y", dtypes.int64s).gep(name="x"), lambda y,x: x.ins(X86Ops.VPEXTRQ, src=(y, imm(dtypes.uint8, x.arg[0])))),
  (UPat.var("y", dtypes.floats).gep(name="x"), lambda y,x: x.ins(X86Ops.VPSRLDQ, src=(y, imm(dtypes.uint8, x.arg[0] * x.dtype.itemsize)))),
  # fused multiply add
  ((UPat(Ops.MUL, dtypes.float32, name="a") + UPat.var("b")).named("c"), lambda ctx,a,b,c:
   a.ins(X86Ops.VFMADD213SS if a.dtype.count == 1 else X86Ops.VFMADD213PS, src=(*a.src, b)) if is_foldable(ctx, c, a) else None),
  ((UPat(Ops.MUL, dtypes.float64, name="a") + UPat.var("b")).named("c"), lambda ctx,a,b,c:
   a.ins(X86Ops.VFMADD213SD if a.dtype.count == 1 else X86Ops.VFMADD213PD, src=(*a.src, b)) if is_foldable(ctx, c, a) else None),
  # packed bitwise
  ((UPat() & UPat()).named("x"), lambda x: x.ins(X86Ops.VPAND) if x.dtype.count > 1 else None),
  ((UPat() | UPat()).named("x"), lambda x: x.ins(X86Ops.VPOR) if x.dtype.count > 1 else None),
  ((UPat() ^ UPat()).named("x"), lambda x: x.ins(X86Ops.VPXOR) if x.dtype.count > 1 else None),
  # packed int binary
  ((UPat(dtype=dtypes.int32s) << UPat()).named("x"), lambda x: x.ins(X86Ops.VPSLLVD) if x.dtype.count > 1 else None),
  ((UPat(dtype=dtypes.int64s) << UPat()).named("x"), lambda x: x.ins(X86Ops.VPSLLVQ) if x.dtype.count > 1 else None),
  ((UPat(dtype=dtypes.uint32) >> UPat()).named("x"), lambda x: x.ins(X86Ops.VPSRLVD) if x.dtype.count > 1 else None),
  ((UPat(dtype=dtypes.uint64) >> UPat()).named("x"), lambda x: x.ins(X86Ops.VPSRLVQ) if x.dtype.count > 1 else None),
  ((UPat(dtype=dtypes.int32) >> UPat()).named("x"), lambda x: x.ins(X86Ops.VPSRAVD) if x.dtype.count > 1 else None),
  ((UPat(dtype=dtypes.int8s) + UPat()).named("x"), lambda x: x.ins(X86Ops.VPADDB) if x.dtype.count > 1 else None),
  ((UPat(dtype=dtypes.int16s) + UPat()).named("x"), lambda x: x.ins(X86Ops.VPADDW) if x.dtype.count > 1 else None),
  ((UPat(dtype=dtypes.int32s) + UPat()).named("x"), lambda x: x.ins(X86Ops.VPADDD) if x.dtype.count > 1 else None),
  ((UPat(dtype=dtypes.int64s) + UPat()).named("x"), lambda x: x.ins(X86Ops.VPADDQ) if x.dtype.count > 1 else None),
  (UPat(Ops.SUB, dtypes.int8s, name="x"), lambda x: x.ins(X86Ops.VPSUBB) if x.dtype.count > 1 else None),
  (UPat(Ops.SUB, dtypes.int16s, name="x"), lambda x: x.ins(X86Ops.VPSUBW) if x.dtype.count > 1 else None),
  (UPat(Ops.SUB, dtypes.int32s, name="x"), lambda x: x.ins(X86Ops.VPSUBD) if x.dtype.count > 1 else None),
  (UPat(Ops.SUB, dtypes.int64s, name="x"), lambda x: x.ins(X86Ops.VPSUBQ) if x.dtype.count > 1 else None),
  (UPat(Ops.MUL, dtypes.int16s, name="x"), lambda x: x.ins(X86Ops.VPMULLW) if x.dtype.count > 1 else None),
  (UPat(Ops.MUL, dtypes.int32s, name="x"), lambda x: x.ins(X86Ops.VPMULLD) if x.dtype.count > 1 else None),
  # scalar int binary
  ((UPat(dtype=dtypes.ints).alu(Ops.CDIV, UPat())).named("x"), idiv),
  # scalar int binary with immediate
  (UPat.var("a", dtypes.ints) << UPat.cvar("c"), lambda a,c: a.ins(X86Ops.SHLi, src=(a, imm(dtypes.uint8, c.arg)))),
  (UPat.var("a", dtypes.uints) >> UPat.cvar("c"), lambda a,c: a.ins(X86Ops.SHRi, src=(a, imm(dtypes.uint8, c.arg)))),
  (UPat.var("a", dtypes.sints) >> UPat.cvar("c"), lambda a,c: a.ins(X86Ops.SARi, src=(a, imm(dtypes.uint8, c.arg)))),
  (UPat.var("a", dtypes.ints) + UPat.cvar("c"), lambda a,c: a.ins(X86Ops.ADDi, src=(a, i)) if (i:=to_imm(c)) is not None else None),
  (UPat.var("a", dtypes.ints) * UPat.cvar("c"), lambda a,c: a.ins(X86Ops.IMULi, src=(a, i)) if (i:=to_imm(c)) is not None else None),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) & UPat.cvar("c"), lambda a,c: a.ins(X86Ops.ANDi, src=(a, i)) if (i:=to_imm(c)) is not None else None),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) | UPat.cvar("c"), lambda a,c: a.ins(X86Ops.ORi, src=(a, i)) if (i:=to_imm(c)) is not None else None),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) ^ UPat.cvar("c"), lambda a,c: a.ins(X86Ops.XORi, src=(a, i)) if (i:=to_imm(c)) is not None else None),
  (UPat(Ops.SUB, dtypes.ints, (UPat.var("a"), UPat.cvar("c"))), lambda a,c: a.ins(X86Ops.SUBi, src=(a, i)) if (i:=to_imm(c)) is not None else None),
  # scalar int binary with register
  (UPat.var("a", dtypes.ints) << UPat.var("b"), lambda a,b: a.ins(X86Ops.SHL, src=(a, b))),
  (UPat.var("a", dtypes.uints) >> UPat.var("b"), lambda a,b: a.ins(X86Ops.SHR, src=(a, b))),
  (UPat.var("a", dtypes.sints) >> UPat.var("b"), lambda a,b: a.ins(X86Ops.SAR, src=(a, b))),
  (UPat.var("a", dtypes.ints) + UPat.var("b"), lambda a,b: a.ins(X86Ops.ADD, src=(a, b))),
  (UPat.var("a", dtypes.ints) * UPat.var("b"), lambda a,b: a.ins(X86Ops.IMUL, src=(a, b))),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) & UPat.var("b"), lambda a,b: a.ins(X86Ops.AND, src=(a, b))),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) | UPat.var("b"), lambda a,b: a.ins(X86Ops.OR, src=(a, b))),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) ^ UPat.var("b"), lambda a,b: a.ins(X86Ops.XOR, src=(a, b))),
  (UPat(Ops.SUB, dtypes.ints, (UPat.var("a"), UPat.var("b"))), lambda a,b: a.ins(X86Ops.SUB, src=(a, b))),
  # float binary
  ((UPat(dtype=dtypes.float32) + UPat()).named("x"), lambda x: x.ins(X86Ops.VADDSS if x.dtype.count == 1 else X86Ops.VADDPS)),
  ((UPat(dtype=dtypes.float64) + UPat()).named("x"), lambda x: x.ins(X86Ops.VADDSD if x.dtype.count == 1 else X86Ops.VADDPD)),
  ((UPat(dtype=dtypes.float32) * UPat()).named("x"), lambda x: x.ins(X86Ops.VMULSS if x.dtype.count == 1 else X86Ops.VMULPS)),
  ((UPat(dtype=dtypes.float64) * UPat()).named("x"), lambda x: x.ins(X86Ops.VMULSD if x.dtype.count == 1 else X86Ops.VMULPD)),
  (UPat(Ops.SUB, dtypes.float32, name="x"), lambda x: x.ins(X86Ops.VSUBSS if x.dtype.count == 1 else X86Ops.VSUBPS)),
  (UPat(Ops.SUB, dtypes.float64, name="x"), lambda x: x.ins(X86Ops.VSUBSD if x.dtype.count == 1 else X86Ops.VSUBPD)),
  (UPat(Ops.FDIV, dtypes.float32, name="x"), lambda x: x.ins(X86Ops.VDIVSS if x.dtype.count == 1 else X86Ops.VDIVPS)),
  (UPat(Ops.FDIV, dtypes.float64, name="x"), lambda x: x.ins(X86Ops.VDIVSD if x.dtype.count == 1 else X86Ops.VDIVPD)),
  # casts
  (UPat(dtype=dtypes.int32).cast(dtypes.float32, name="x"), lambda x: x.ins(X86Ops.VCVTDQ2PS) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.int32).cast(dtypes.float64, name="x"), lambda x: x.ins(X86Ops.VCVTDQ2PD) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.float32).cast(dtypes.int32s, name="x"), lambda x: x.ins(X86Ops.VCVTTPS2DQ) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.float64).cast(dtypes.int32s, name="x"), lambda x: x.ins(X86Ops.VCVTTPD2DQ) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.float32).cast(dtypes.float64, name="x"), lambda x: x.ins(X86Ops.VCVTPS2PD) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.float64).cast(dtypes.float32, name="x"), lambda x: x.ins(X86Ops.VCVTPD2PS) if x.dtype.count > 1 else None),
  (UPat(dtype=dtypes.float32).cast(dtypes.float16, name="x"), lambda x: x.ins(X86Ops.VCVTPS2PH, src=x.src + (imm(dtypes.uint8, 4),))),
  (UPat(dtype=dtypes.float16).cast(dtypes.float32, name="x"), lambda x: x.ins(X86Ops.VCVTPH2PS)),
  (UPat(dtype=dtypes.float32).cast(dtypes.int32s+dtypes.int64s, name="x"), lambda x: x.ins(X86Ops.VCVTTSS2SI)),
  (UPat(dtype=dtypes.float64).cast(dtypes.int32s+dtypes.int64s, name="x"), lambda x: x.ins(X86Ops.VCVTTSD2SI)),
  (UPat.var("y", dtypes.float32).cast(dtypes.float64, name="x"), lambda y,x: x.ins(X86Ops.VCVTSS2SD, src=(y, y))),
  (UPat.var("y", dtypes.float64).cast(dtypes.float32, name="x"), lambda y,x: x.ins(X86Ops.VCVTSD2SS, src=(y, y))),
  (UPat.var("y", (dtypes.int32, dtypes.int64)).cast(dtypes.float32, name="x"), lambda y,x: x.ins(X86Ops.VCVTSI2SS, src=(def_reg(x.dtype), y))),
  (UPat.var("y", (dtypes.int32, dtypes.int64)).cast(dtypes.float64, name="x"), lambda y,x: x.ins(X86Ops.VCVTSI2SD, src=(def_reg(x.dtype), y))),
  (UPat(dtype=dtypes.uints+(dtypes.bool,)).cast(dtypes.ints, name="x"), lambda x: x.ins(X86Ops.MOVZX) if x.dtype.count == 1 else None),
  (UPat(dtype=dtypes.int32).cast(dtypes.int64s, name="x"), lambda x: x.ins(X86Ops.MOVSXD) if x.dtype.count == 1 else None),
  (UPat(dtype=dtypes.sints).cast(dtypes.ints, name="x"), lambda x: x.ins(X86Ops.MOVSX) if x.dtype.count == 1 else None),
  (UPat(dtype=(dtypes.uint8, dtypes.bool)).cast(dtypes.int16s, name="x"), lambda x: x.ins(X86Ops.VPMOVZXBW)),
  (UPat(dtype=(dtypes.uint8, dtypes.bool)).cast(dtypes.int32s, name="x"), lambda x: x.ins(X86Ops.VPMOVZXBD)),
  (UPat(dtype=(dtypes.uint8, dtypes.bool)).cast(dtypes.int64s, name="x"), lambda x: x.ins(X86Ops.VPMOVZXBQ)),
  (UPat(dtype=dtypes.uint16).cast(dtypes.int32s, name="x"), lambda x: x.ins(X86Ops.VPMOVZXWD)),
  (UPat(dtype=dtypes.uint16).cast(dtypes.int64s, name="x"), lambda x: x.ins(X86Ops.VPMOVZXWQ)),
  (UPat(dtype=dtypes.uint32).cast(dtypes.int64s, name="x"), lambda x: x.ins(X86Ops.VPMOVZXDQ)),
  (UPat(dtype=dtypes.int8).cast(dtypes.int16s, name="x"), lambda x: x.ins(X86Ops.VPMOVSXBW)),
  (UPat(dtype=dtypes.int8).cast(dtypes.int32s, name="x"), lambda x: x.ins(X86Ops.VPMOVSXBD)),
  (UPat(dtype=dtypes.int8).cast(dtypes.int64s, name="x"), lambda x: x.ins(X86Ops.VPMOVSXBQ)),
  (UPat(dtype=dtypes.int16).cast(dtypes.int32s, name="x"), lambda x: x.ins(X86Ops.VPMOVSXWD)),
  (UPat(dtype=dtypes.int16).cast(dtypes.int64s, name="x"), lambda x: x.ins(X86Ops.VPMOVSXWQ)),
  (UPat(dtype=dtypes.int32).cast(dtypes.int64s, name="x"), lambda x: x.ins(X86Ops.VPMOVSXDQ)),
  # bitcasts
  (UPat.var("y", dtypes.float16).bitcast(dtypes.int16s).named("x"), lambda y,x: x.ins(X86Ops.VPEXTRW, src=(y, imm(dtypes.uint8, 0)))),
  (UPat(dtype=dtypes.int16s).bitcast(dtypes.float16).named("x"), vpins),
  (UPat(dtype=dtypes.int32s).bitcast(dtypes.float32).named("x"), lambda x: x.ins(X86Ops.VMOVD)),
  (UPat(dtype=dtypes.int64s).bitcast(dtypes.float64).named("x"), lambda x: x.ins(X86Ops.VMOVQ)),
  (UPat(dtype=dtypes.float32).bitcast(dtypes.int32s).named("x"), lambda x: x.ins(X86Ops.VMOVDm)),
  (UPat(dtype=dtypes.float64).bitcast(dtypes.int64s).named("x"), lambda x: x.ins(X86Ops.VMOVQm)),
  # index
  (UPat(Ops.INDEX, name="x"), lambda x: x.ins(X86Ops.LEA, src=fold_address(x))),
  # TODO: fuse stores, very few cases -- store cmp becomes setcc, store gep int becomes vpextr, store bitcast to int becomes vmovd/q
  # copy, load, store
  # NOTE: copy here violates the spec, it only happens post register allocation when a reg to reg move needs to be inserted
  (UPat(Ops.COPY, dt_128bit, name="x"), lambda x: x.ins(X86Ops.VMOVUPS)),
  (UPat(Ops.COPY, dt_64bit, name="x"), lambda x: x.ins(X86Ops.VMOVSD)),
  (UPat(Ops.COPY, dt_32bit+dt_16bit, name="x"), lambda x: x.ins(X86Ops.VMOVSS)),
  (UPat(Ops.COPY, dtypes.ints+(dtypes.bool,), name="x"), lambda x: x.ins(X86Ops.MOV)),
  (UPat(Ops.LOAD, dt_128bit, src=(UPat(name="a"),), name="x"), lambda x,a: x.ins(X86Ops.VMOVUPS, src=fold_address(a))),
  (UPat(Ops.LOAD, dt_64bit, src=(UPat(name="a"),), name="x"), lambda x,a: x.ins(X86Ops.VMOVSD, src=fold_address(a))),
  (UPat(Ops.LOAD, dt_32bit, src=(UPat(name="a"),), name="x"), lambda x,a: x.ins(X86Ops.VMOVSS, src=fold_address(a))),
  (UPat(Ops.LOAD, dt_16bit, src=(UPat(name="a"),), name="x"), lambda x,a:
   x.ins(X86Ops.VPINSRW, src=(def_reg(x.dtype, x.tag),) + fold_address(a) + (imm(dtypes.uint8, 0),))),
  (UPat(Ops.LOAD, dtypes.ints+(dtypes.bool,), src=(UPat(name="a"),), name="x"), lambda x,a: x.ins(X86Ops.MOV, src=fold_address(a))),
  (UPat.var("a").store(UPat.var("b", dt_128bit), name="x"), lambda a,b,x: x.ins(X86Ops.VMOVUPSm, src=fold_address(a) + (b,))),
  (UPat.var("a").store(UPat.var("b", dt_64bit), name="x"), lambda a,b,x: x.ins(X86Ops.VMOVSDm, src=fold_address(a) + (b,))),
  (UPat.var("a").store(UPat.var("b", dt_32bit), name="x"), lambda a,b,x: x.ins(X86Ops.VMOVSSm, src=fold_address(a) + (b,))),
  (UPat.var("a").store(UPat.var("b", dt_16bit), name="x"), lambda a,b,x: x.ins(X86Ops.VPEXTRW, src=fold_address(a) + (b, imm(dtypes.uint8, 0)))),
  (UPat.var("a").store(UPat.var("b", dtypes.ints+(dtypes.bool,)), name="x"), lambda a,b,x:
   x.ins(X86Ops.MOVm, src=fold_address(a) + (b,)) if (i:=to_imm(b)) is None else x.ins(X86Ops.MOVi, src=fold_address(a) + (i,))),
  # **** X86Op -> X86Op ****
  # fold loads into X86Ops that allow it, if beneficial
  (UPat(Ops.INS, src=(UPat(Ops.LOAD, src=(UPat(name="a"),), name="y"),), allow_any_len=True, name="x"), lambda ctx,y,a,x:
   x.replace(src=fold_address(a) + x.src[1:]) if x.arg in X86GroupOp.ReadMem1st and is_foldable(ctx, x, y) else None),
  (UPat(Ops.INS, src=(UPat(), UPat(Ops.LOAD, src=(UPat(name="a"),), name="y")), allow_any_len=True, name="x"), lambda ctx,y,a,x:
   x.replace(src=x.src[:1] + fold_address(a) + x.src[2:]) if x.arg in X86GroupOp.ReadMem2nd and is_foldable(ctx, x, y) else None),
  (UPat(Ops.INS, src=(UPat(), UPat(), UPat(Ops.LOAD, src=(UPat(name="a"),), name="y")), allow_any_len=True, name="x"), lambda ctx,y,a,x:
   x.replace(src=x.src[:2] + fold_address(a) + x.src[3:]) if x.arg in X86GroupOp.ReadMem3rd and is_foldable(ctx, x, y) else None),
  # allocate virtual registers
  (UPat((Ops.INS, Ops.DEFINE_REG, Ops.DEFINE_LOCAL), name="x"), alloc_vregs),
])

# ***** pre register allocation *****
# this handles flag clobbers. Unfortunately x86 doesn't have a good way to store/restore the flag register (then regalloc would handle it)
# so we rematerialize. This is different from rematerialization you might want to do in regalloc because it is not optional,
# regalloc shouldn't rematerialize if a src of the instruction is dead, but here you need to as there's no fallback load from stack
def flag_rematerialize(ctx:PreRegAllocContext, x:UOp):
  flag_def = x if x.arg in X86GroupOp.WriteFlags or x.op in (Ops.RANGE, Ops.END) else x.src[-1] if x.arg in X86GroupOp.ReadFlags else None
  if flag_def is None: return None
  if ctx.lock is not None and ctx.lock is not flag_def: ctx.clobbered.add(ctx.lock)
  ctx.lock = flag_def
  if flag_def not in ctx.clobbered: return None
  ctx.clobbered.remove(flag_def)
  return (x, [flag_def, x])

pre_regalloc_matcher = PatternMatcher([
  (UPat((Ops.INS, Ops.RANGE, Ops.END), name="x"), flag_rematerialize),
])

# ***** post register allocation *****
# TODO: control flow should be overhauled so that this isn't necessary
def lower_range(ctx, x:UOp) -> tuple[UOp, list[UOp]]:
  loop_label = "_".join(str(i) for i in x.arg[:-1])
  acc = x.ins(X86Ops.MOVi, src=(imm(x.dtype, 0),) + x.src[1:])
  label = UOp(Ops.INS, arg=X86Ops.LABEL, tag=f".LOOP_{loop_label}")
  cmp = UOp(Ops.INS, arg=X86Ops.CMPi if x.src[0].op is Ops.CONST else X86Ops.CMP, src=(acc, x.src[0]))
  jump_out = UOp(Ops.INS, arg=X86Ops.JGE, src=(cmp,), tag=f".LOOP_OUT_{loop_label}")
  ctx.loop_label[acc] = loop_label
  return (acc, [acc, label, cmp, jump_out])

# final rewrite to match the isa spec
post_regalloc_matcher = PatternMatcher([
  # rewrite FRAME_INDEX to IMM now that the stack size is known
  (UPat(Ops.INS, arg=X86Ops.FRAME_INDEX, name="x"), lambda ctx,x: (nx:=x.const_like(ctx.stack_size + x.tag), [nx])),
  # rewrite RANGE to ACC = 0 -> LABEL -> JUMP if ACC >= loop bound
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: lower_range(ctx, x)),
  # rewrite END to ACC + 1 -> JUMP -> LABEL, also add the out of loop JUMP to the src so this becomes the jump target
  (UPat(Ops.END, name="x"), lambda ctx,x: (jmp:=UOp(Ops.INS, arg=X86Ops.JMP, tag=f".LOOP_{ctx.loop_label[x.src[1]]}"),
   [x.src[1].ins(X86Ops.ADDi, src=(imm(x.src[1].dtype, 1),)), jmp, UOp(Ops.INS, arg=X86Ops.LABEL, tag=f".LOOP_OUT_{ctx.loop_label[x.src[1]]}")])),
  # rewrite two address instructions to two address form, if reused src wasn't coalesced insert a move
  (UPat(Ops.INS, name="x"), lambda ctx,x: (nx:=x.replace(src=x.src[1:]),
   [ctx.ren.copy(x.src[0], x.reg), nx] if x.reg != x.src[0].reg else [nx]) if x.arg in X86GroupOp.TwoAddress else None),
])

# ***** X86 instruction encoding *****

def encode(x:UOp, opc:int, reg:int|None=None, pp:int=0, sel:int=0, we:int=0) -> bytes|None:
  def _encode(reg_uop:UOp|None, rm_uop:UOp, idx_uop:UOp|None=None, disp_uop:UOp|None=None, vvvv_uop:UOp|None=None, imm_uop:UOp|None=None) -> bytes:
    nonlocal reg, opc
    # get the encoding values of the different fields
    reg = cast(int, cast(Register, reg_uop.reg).index if reg_uop is not None else reg)
    rm = cast(Register, rm_uop.reg).index
    idx = cast(Register, idx_uop.reg).index if idx_uop is not None and idx_uop.reg is not None else 4
    rm_sz = 8 if isinstance(rm_uop.dtype, PtrDType) and disp_uop is None else rm_uop.dtype.itemsize
    reg_sz = (reg_uop.dtype.itemsize if not isinstance(reg_uop.dtype, PtrDType) else 8) if reg_uop is not None else 0
    sz = reg_sz or rm_sz

    # encode instruction
    inst = bytes([])
    assert 0 <= reg <= 15 and 0 <= idx <= 15 and 0 <= rm <= 15
    # r extends reg field, x extends index field, b extends rm or base field
    r, _x, b = reg >> 3, idx >> 3, rm >> 3
    if sel: # VEX bytes
      vvvv = cast(Register, vvvv_uop.reg).index if vvvv_uop is not None else 0
      l = (max(reg_sz, rm_sz) > 16) & 0b1
      if sel == 1 and _x == b == we == 0: inst += bytes([0xC5, (~r & 0b1) << 7 | (~vvvv & 0b1111) << 3 | l << 2 | pp])
      else: inst += bytes([0xC4, (~r & 0b1) << 7 | (~_x & 0b1) << 6 | (~b & 0b1) << 5 | sel, we << 7 | (~vvvv & 0b1111) << 3 | l << 2 | pp])
    else: # optional PREFIX and REX bytes
      # PREFIX byte signaling 16 bit variant of instruction
      if sz == 2: inst += bytes([0x66])
      # bit signaling 64 bit variant of instruction
      w = sz == 8
      # REX byte is required when 64 bit or an extended reg is used (index 8 - 15) or lower 8 bits of (rsp, rbp, rsi, rdi) are accessed
      if w | r | _x | b | (reg_sz == 1 & reg >> 2) | (rm_sz == 1 & rm >> 2): inst += bytes([0b0100 << 4 | w << 3 | r << 2 | _x << 1 | b])
      # legacy 8bit opcode is 1 less than 16-64bit variants
      if (rm_sz == 1 or reg_sz == 1) and x.arg not in X86GroupOp.ReadFlags | {X86Ops.LEA}: opc -= 1
    # OPCODE byte
    inst += opc.to_bytes((opc.bit_length() + 7) // 8, 'big')
    # MODRM byte
    # now we only care about the lower 3 bits
    idx, rm, reg = idx & 0b111, rm & 0b111, reg & 0b111
    # 0b00 -- signals memory access with no displacement
    # 0b01 -- signals memory access with 8bit displacement
    # 0b10 -- signals memory access with 32bit displacement
    # 0b11 -- signals no memory access
    if disp_uop is not None:
      assert disp_uop.op is Ops.CONST, "displacement must be a constant"
      assert disp_uop.dtype in (dtypes.int8, dtypes.int32), "displacement can only be 1 or 4 byte signed int"
      # rbp/r13 always require a displacement
      if disp_uop.arg != 0 or rm == 0b101: mod = 0b01 if disp_uop.dtype.itemsize == 1 else 0b10
      else: mod = 0b00
    else: mod = 0b11
    # x 0b0 and idx 0b100 means rsp which means no index exists
    # rm 0b100 (rsp/r12) signals a sib byte is required, rm then is encoded in the base field of SIB
    _rm = rm if idx == 0b100 and _x == 0b0 else 0b100
    inst += bytes([mod << 6 | reg << 3 | _rm])
    # SIB byte
    if _rm == 0b100 and mod != 0b11:
      scale = {1: 0b00, 2: 0b01, 4: 0b10, 8: 0b11}[1 if idx == 0b100 and _x == 0b0 else rm_sz]
      inst += bytes([scale << 6 | idx << 3 | rm])
    # DISP byte
    if mod == 0b01 or mod == 0b10:
      assert disp_uop is not None
      inst += struct.pack(unwrap(disp_uop.dtype.fmt), disp_uop.arg)
    # IMM byte
    if imm_uop is not None:
      if imm_uop.op is Ops.CONST: inst += struct.pack(unwrap(imm_uop.dtype.fmt), imm_uop.arg)
      elif isinstance(imm_uop.reg, Register): inst += bytes([(imm_uop.reg.index & 0b1111) << 4 | 0b0000])
    return inst

  # get the encoding structure of the uop
  # when a uop writes to memory it takes the form of a store, dtype is void, no definition
  address:tuple[UOp|None, ...]
  if x.arg in X86GroupOp.WriteMem:
    if len(x.src) > 3: address, rest = x.src[:3], x.src[3:]
    else: address, rest = (x, None, None), x.src
    return _encode(rest[0], *address, *(None, *rest[1:])) if reg is None else _encode(None, *address, *(None, *rest[:1]))

  if x.arg in X86GroupOp.Rm1st:
    if len(x.src) > 2: address, rest = x.src[:3], x.src[3:]
    else: address, rest = (x.src[0], None, None), x.src[1:]
    imm_uop = rest[:1] if rest and rest[0].op is Ops.CONST else (None,)
    return _encode(x, *address, *(None, *imm_uop)) if reg is None else _encode(None, *address, *(x if sel else None, *imm_uop))

  if x.arg in X86GroupOp.Rm2nd:
    if len(x.src) > 3: address, rest = x.src[1:4], x.src[:1] + x.src[4:]
    else: address, rest = (x.src[1], None, None), x.src[:1] + x.src[2:]
    # cmp/vucomiss reg, rm don't define a new register
    return _encode(x, *address, *rest) if x.dtype is not dtypes.void else _encode(rest[0], *address)

  return None

# https://www.felixcloutier.com/x86/
# legacy version -> VEX version
# prefix field: None -> 0 | 66 -> 1 | F3 -> 2 | F2 -> 3
# opcode map select: 0F -> 1 | 0F38 -> 2 | 0F3A -> 3
encodings = {
  # moves
  X86Ops.MOVABS: lambda x:
   bytes([0b0100 << 4 | 0b1 << 3 | 0b00 << 2 | x.reg.index >> 3, 0xB8 + (x.reg.index & 0b111)]) + struct.pack(x.dtype.fmt, x.src[0].arg),
  X86Ops.MOV: lambda x: encode(x, 0x8B), X86Ops.MOVi: lambda x: encode(x, 0xC7, reg=0),
  X86Ops.MOVm: lambda x: encode(x, 0x89), X86Ops.LEA: lambda x: encode(x, 0x8D),
  X86Ops.VMOVSS: lambda x: encode(x, 0x10, pp=2, sel=1), X86Ops.VMOVSSm: lambda x: encode(x, 0x11, pp=2, sel=1),
  X86Ops.VMOVSD: lambda x: encode(x, 0x10, pp=3, sel=1), X86Ops.VMOVSDm: lambda x: encode(x, 0x11, pp=3, sel=1),
  X86Ops.VMOVUPS: lambda x: encode(x, 0x10, pp=0, sel=1), X86Ops.VMOVUPSm: lambda x: encode(x, 0x11, pp=0, sel=1),
  X86Ops.VMOVD: lambda x: encode(x, 0x6E, pp=1, sel=1), X86Ops.VMOVQ: lambda x: encode(x, 0x6E, pp=1, sel=1, we=1),
  X86Ops.VMOVDm: lambda x: encode(x, 0x7E, pp=1, sel=1), X86Ops.VMOVQm: lambda x: encode(x, 0x7E, pp=1, sel=1, we=1),
  # casts
  X86Ops.MOVZX: lambda x: encode(x, 0x0FB7),
  X86Ops.MOVSX: lambda x: encode(x, 0x0FBF), X86Ops.MOVSXD: lambda x: encode(x, 0x63),
  X86Ops.VPMOVZXBW: lambda x: encode(x, 0x30, pp=1, sel=2), X86Ops.VPMOVZXBD: lambda x: encode(x, 0x31, pp=1, sel=2),
  X86Ops.VPMOVZXBQ: lambda x: encode(x, 0x32, pp=1, sel=2), X86Ops.VPMOVZXWD: lambda x: encode(x, 0x33, pp=1, sel=2),
  X86Ops.VPMOVZXWQ: lambda x: encode(x, 0x34, pp=1, sel=2), X86Ops.VPMOVZXDQ: lambda x: encode(x, 0x35, pp=1, sel=2),
  X86Ops.VPMOVSXBW: lambda x: encode(x, 0x20, pp=1, sel=2), X86Ops.VPMOVSXBD: lambda x: encode(x, 0x21, pp=1, sel=2),
  X86Ops.VPMOVSXBQ: lambda x: encode(x, 0x22, pp=1, sel=2), X86Ops.VPMOVSXWD: lambda x: encode(x, 0x23, pp=1, sel=2),
  X86Ops.VPMOVSXWQ: lambda x: encode(x, 0x24, pp=1, sel=2), X86Ops.VPMOVSXDQ: lambda x: encode(x, 0x25, pp=1, sel=2),
  X86Ops.VCVTSS2SD: lambda x: encode(x, 0x5A, pp=2, sel=1), X86Ops.VCVTSD2SS: lambda x: encode(x, 0x5A, pp=3, sel=1),
  X86Ops.VCVTPH2PS: lambda x: encode(x, 0x13, pp=1, sel=2), X86Ops.VCVTPS2PH: lambda x: encode(x, 0x1D, pp=1, sel=3),
  X86Ops.VCVTDQ2PS: lambda x: encode(x, 0x5B, pp=0, sel=1), X86Ops.VCVTDQ2PD: lambda x: encode(x, 0xE6, pp=2, sel=1),
  X86Ops.VCVTPS2PD: lambda x: encode(x, 0x5A, pp=0, sel=1), X86Ops.VCVTPD2PS: lambda x: encode(x, 0x5A, pp=1, sel=1),
  X86Ops.VCVTTPS2DQ: lambda x: encode(x, 0x5B, pp=2, sel=1), X86Ops.VCVTTPD2DQ: lambda x: encode(x, 0xE6, pp=1, sel=1),
  X86Ops.VCVTSI2SS: lambda x: encode(x, 0x2A, pp=2, sel=1, we=x.src[1].dtype.itemsize == 8),
  X86Ops.VCVTSI2SD: lambda x: encode(x, 0x2A, pp=3, sel=1, we=x.src[1].dtype.itemsize == 8),
  X86Ops.VCVTTSS2SI: lambda x: encode(x, 0x2C, pp=2, sel=1, we=x.dtype.itemsize == 8),
  X86Ops.VCVTTSD2SI: lambda x: encode(x, 0x2C, pp=3, sel=1, we=x.dtype.itemsize == 8),
  # int division
  X86Ops.IDIV: lambda x: encode(x, 0xF7, reg=7), X86Ops.DIV: lambda x: encode(x, 0xF7, reg=6),
  # scalar int binary
  X86Ops.SHLi: lambda x: encode(x, 0xC1, reg=4),
  X86Ops.SHRi: lambda x: encode(x, 0xC1, reg=5), X86Ops.SARi: lambda x: encode(x, 0xC1, reg=7),
  X86Ops.ADD: lambda x: encode(x, 0x03), X86Ops.ADDi: lambda x: encode(x, 0x81, reg=0),
  X86Ops.SUB: lambda x: encode(x, 0x2B), X86Ops.SUBi: lambda x: encode(x, 0x81, reg=5),
  X86Ops.AND: lambda x: encode(x, 0x23), X86Ops.ANDi: lambda x: encode(x, 0x81, reg=4),
  X86Ops.XOR: lambda x: encode(x, 0x33), X86Ops.XORi: lambda x: encode(x, 0x81, reg=6),
  X86Ops.OR: lambda x: encode(x, 0x0B), X86Ops.ORi: lambda x: encode(x, 0x81, reg=1),
  X86Ops.CMP: lambda x: encode(x, 0x3B), X86Ops.CMPi: lambda x: encode(x, 0x81, reg=7),
  X86Ops.IMUL: lambda x: encode(x, 0x0FAF), X86Ops.IMULi: lambda x: encode(x, 0x69),
  X86Ops.SETB: lambda x: encode(x, 0x0F92, reg=0), X86Ops.SETL: lambda x: encode(x, 0x0F9C, reg=0),
  X86Ops.SETE: lambda x: encode(x, 0x0F94, reg=0), X86Ops.SETNE: lambda x: encode(x, 0x0F95, reg=0),
  # packed bitwise NOTE: only bitwise and packed
  X86Ops.VPAND: lambda x: encode(x, 0xDB, pp=1, sel=1), X86Ops.VPXOR: lambda x: encode(x, 0xEF, pp=1, sel=1),
  X86Ops.VPOR: lambda x: encode(x, 0xEB, pp=1, sel=1),
  # unary
  X86Ops.VSQRTSS: lambda x: encode(x, 0x51, pp=2, sel=1), X86Ops.VSQRTPS: lambda x: encode(x, 0x51, pp=0, sel=1),
  X86Ops.VSQRTSD: lambda x: encode(x, 0x51, pp=3, sel=1), X86Ops.VSQRTPD: lambda x: encode(x, 0x51, pp=1, sel=1),
  X86Ops.VROUNDSS: lambda x: encode(x, 0x0A, pp=1, sel=3), X86Ops.VROUNDPS: lambda x: encode(x, 0x08, pp=1, sel=3),
  X86Ops.VROUNDSD: lambda x: encode(x, 0x0B, pp=1, sel=3), X86Ops.VROUNDPD: lambda x: encode(x, 0x09, pp=1, sel=3),
  # packed int binary
  X86Ops.VPSLLVD: lambda x: encode(x, 0x47, pp=1, sel=2), X86Ops.VPSLLVQ: lambda x: encode(x, 0x47, pp=1, sel=2, we=1),
  X86Ops.VPSRLVD: lambda x: encode(x, 0x45, pp=1, sel=2), X86Ops.VPSRLVQ: lambda x: encode(x, 0x45, pp=1, sel=2, we=1),
  X86Ops.VPCMPGTB: lambda x: encode(x, 0x64, pp=1, sel=1), X86Ops.VPCMPGTW: lambda x: encode(x, 0x65, pp=1, sel=1),
  X86Ops.VPCMPGTD: lambda x: encode(x, 0x66, pp=1, sel=1), X86Ops.VPCMPGTQ: lambda x: encode(x, 0x37, pp=1, sel=2),
  X86Ops.VPCMPEQB: lambda x: encode(x, 0x74, pp=1, sel=1), X86Ops.VPCMPEQW: lambda x: encode(x, 0x75, pp=1, sel=1),
  X86Ops.VPCMPEQD: lambda x: encode(x, 0x76, pp=1, sel=1), X86Ops.VPCMPEQQ: lambda x: encode(x, 0x29, pp=1, sel=2),
  X86Ops.VPMULLW: lambda x: encode(x, 0xD5, pp=1, sel=1), X86Ops.VPMULLD: lambda x: encode(x, 0x40, pp=1, sel=2),
  X86Ops.VPADDB: lambda x: encode(x, 0xFC, pp=1, sel=1), X86Ops.VPADDW: lambda x: encode(x, 0xFD, pp=1, sel=1),
  X86Ops.VPADDD: lambda x: encode(x, 0xFE, pp=1, sel=1), X86Ops.VPADDQ: lambda x: encode(x, 0xD4, pp=1, sel=1),
  X86Ops.VPSUBB: lambda x: encode(x, 0xF8, pp=1, sel=1), X86Ops.VPSUBW: lambda x: encode(x, 0xF9, pp=1, sel=1),
  X86Ops.VPSUBD: lambda x: encode(x, 0xFA, pp=1, sel=1), X86Ops.VPSUBQ: lambda x: encode(x, 0xFB, pp=1, sel=1),
  X86Ops.VPSRAVD: lambda x: encode(x, 0x46, pp=1, sel=2),
  # float cmp
  X86Ops.VUCOMISS: lambda x: encode(x, 0x2E, pp=0, sel=1), X86Ops.VUCOMISD: lambda x: encode(x, 0x2E, pp=1, sel=1),
  # scalar / packed float binary
  X86Ops.VADDSS: lambda x: encode(x, 0x58, pp=2, sel=1), X86Ops.VADDPS: lambda x: encode(x, 0x58, pp=0, sel=1),
  X86Ops.VADDSD: lambda x: encode(x, 0x58, pp=3, sel=1), X86Ops.VADDPD: lambda x: encode(x, 0x58, pp=1, sel=1),
  X86Ops.VSUBSS: lambda x: encode(x, 0x5C, pp=2, sel=1), X86Ops.VSUBPS: lambda x: encode(x, 0x5C, pp=0, sel=1),
  X86Ops.VSUBSD: lambda x: encode(x, 0x5C, pp=3, sel=1), X86Ops.VSUBPD: lambda x: encode(x, 0x5C, pp=1, sel=1),
  X86Ops.VMULSS: lambda x: encode(x, 0x59, pp=2, sel=1), X86Ops.VMULPS: lambda x: encode(x, 0x59, pp=0, sel=1),
  X86Ops.VMULSD: lambda x: encode(x, 0x59, pp=3, sel=1), X86Ops.VMULPD: lambda x: encode(x, 0x59, pp=1, sel=1),
  X86Ops.VDIVSS: lambda x: encode(x, 0x5E, pp=2, sel=1), X86Ops.VDIVPS: lambda x: encode(x, 0x5E, pp=0, sel=1),
  X86Ops.VDIVSD: lambda x: encode(x, 0x5E, pp=3, sel=1), X86Ops.VDIVPD: lambda x: encode(x, 0x5E, pp=1, sel=1),
  X86Ops.VCMPSS: lambda x: encode(x, 0xC2, pp=2, sel=1), X86Ops.VCMPPS: lambda x: encode(x, 0xC2, pp=0, sel=1),
  X86Ops.VCMPSD: lambda x: encode(x, 0xC2, pp=3, sel=1), X86Ops.VCMPPD: lambda x: encode(x, 0xC2, pp=1, sel=1),
  X86Ops.VMAXSS: lambda x: encode(x, 0x5F, pp=2, sel=1), X86Ops.VMAXPS: lambda x: encode(x, 0x5F, pp=0, sel=1),
  X86Ops.VMAXSD: lambda x: encode(x, 0x5F, pp=3, sel=1), X86Ops.VMAXPD: lambda x: encode(x, 0x5F, pp=1, sel=1),
  X86Ops.VMINSS: lambda x: encode(x, 0x5D, pp=2, sel=1), X86Ops.VMINPS: lambda x: encode(x, 0x5D, pp=0, sel=1),
  X86Ops.VMINSD: lambda x: encode(x, 0x5D, pp=3, sel=1), X86Ops.VMINPD: lambda x: encode(x, 0x5D, pp=1, sel=1),
  # ternary
  X86Ops.CMOVB: lambda x: encode(x, 0x0F42), X86Ops.CMOVL: lambda x: encode(x, 0x0F4C),
  X86Ops.CMOVE: lambda x: encode(x, 0x0F44), X86Ops.CMOVNE: lambda x: encode(x, 0x0F45),
  X86Ops.VFMADD213SS: lambda x: encode(x, 0xA9, pp=1, sel=2), X86Ops.VFMADD213SD: lambda x: encode(x, 0xA9, pp=1, sel=2, we=1),
  X86Ops.VFMADD213PS: lambda x: encode(x, 0xA8, pp=1, sel=2), X86Ops.VFMADD213PD: lambda x: encode(x, 0xA8, pp=1, sel=2, we=1),
  X86Ops.VBLENDVPS: lambda x: encode(x, 0x4A, pp=1, sel=3), X86Ops.VBLENDVPD: lambda x: encode(x, 0x4B, pp=1, sel=3),
  X86Ops.VPBLENDVB: lambda x: encode(x, 0x4C, pp=1, sel=3),
  # shuffles
  X86Ops.VPBROADCASTB: lambda x: encode(x, 0x78, pp=1, sel=2), X86Ops.VPBROADCASTW: lambda x: encode(x, 0x79, pp=1, sel=2),
  X86Ops.VPBROADCASTD: lambda x: encode(x, 0x58, pp=1, sel=2), X86Ops.VPBROADCASTQ: lambda x: encode(x, 0x59, pp=1, sel=2),
  X86Ops.VBROADCASTSS: lambda x: encode(x, 0x18, pp=1, sel=2), X86Ops.VPSRLDQ: lambda x: encode(x, 0x73, reg=3, pp=1, sel=1),
  X86Ops.VPINSRB: lambda x: encode(x, 0x20, pp=1, sel=3), X86Ops.VPINSRW: lambda x: encode(x, 0xC4, pp=1, sel=1),
  X86Ops.VPINSRD: lambda x: encode(x, 0x22, pp=1, sel=3), X86Ops.VPINSRQ: lambda x: encode(x, 0x22, pp=1, sel=3, we=1),
  X86Ops.VSHUFPS: lambda x: encode(x, 0xC6, pp=0, sel=1), X86Ops.VSHUFPD: lambda x: encode(x, 0xC6, pp=1, sel=1),
  X86Ops.VINSERTPS: lambda x: encode(x, 0x21, pp=1, sel=3),
  # extract
  X86Ops.VPEXTRB: lambda x: encode(x, 0x14, pp=1, sel=3), X86Ops.VPEXTRW: lambda x: encode(x, 0x15, pp=1, sel=3),
  X86Ops.VPEXTRD: lambda x: encode(x, 0x16, pp=1, sel=3), X86Ops.VPEXTRQ: lambda x: encode(x, 0x16, pp=1, sel=3, we=1),
  # jumps are encoded with a placeholder which gets patched later once the real offset is known
  X86Ops.JE: lambda x: bytes([0x0F, 0x84]) + int(0).to_bytes(4),
  X86Ops.JNE: lambda x: bytes([0x0F, 0x85]) + int(0).to_bytes(4),
  X86Ops.JL: lambda x: bytes([0x0F, 0x8C]) + int(0).to_bytes(4),
  X86Ops.JB: lambda x: bytes([0x0F, 0x82]) + int(0).to_bytes(4),
  X86Ops.JGE: lambda x: bytes([0x0F, 0x8D]) + int(0).to_bytes(4),
  X86Ops.JMP: lambda x: bytes([0xE9]) + int(0).to_bytes(4),
  X86Ops.RET: lambda x: bytes([0xC3]),
}

class X86Renderer(ISARenderer):
  device = "CPU"
  has_local = False
  has_threads = bool(getenv("THREADS", 1))
  global_max = (CPU_COUNT.value, 0, 0)
  extra_matcher = extra_matcher
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  pre_regalloc_matcher = pre_regalloc_matcher
  post_regalloc_matcher = post_regalloc_matcher
  code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.AND, Ops.OR, Ops.SHL, Ops.SHR, Ops.NEG, Ops.SUB, Ops.FDIV, Ops.CMPLT, Ops.CMPEQ)}
  def __init__(self, target:Target):
    super().__init__(target)
    from tinygrad.runtime.support.compiler_cpu import X86Compiler
    self.compiler = X86Compiler()
  def is_two_address(self, x:UOp) -> bool: return x.arg in X86GroupOp.TwoAddress
  def stack_pointer(self) -> UOp: return def_reg(dtypes.uint64, RSP)
  # nasty hacks to deal with pointers TODO: rm pointers
  def copy(self, x:UOp, reg:Register):
    dt = dtypes.uint64 if isinstance(x.dtype, PtrDType) else x.dtype
    ret = isel_matcher.rewrite(UOp(Ops.COPY, dt, (x,), tag=reg))
    assert ret is not None
    return ret.replace(dtype=x.dtype)

  def spill(self, disp:UOp, x:UOp) -> UOp:
    nx = x.replace(dtype=dtypes.uint64 if isinstance(x.dtype, PtrDType) else x.dtype)
    ret = isel_matcher.rewrite(self.stack_pointer().index(disp).store(nx))
    assert ret is not None
    return ret.replace(src=(s if s is not nx else x for s in ret.src))

  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp:
    ndt = dtypes.uint64 if isinstance(x.dtype, PtrDType) else x.dtype
    ret = isel_matcher.rewrite(self.stack_pointer().index(disp).load(dtype=ndt, tag=reg))
    assert ret is not None
    return ret.replace(dtype=x.dtype)

  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    def _format_op(x:UOp) -> str: return f"    {(o[7:-1] if (o:=str(x.arg))[-1] in ('i', 'm') else o[7:]).lower():7s}"
    def _format_operands(x:UOp) -> str:
      def _format(src:tuple[UOp, ...]) -> list[str]:
        return [str(s.arg) if s.op is Ops.CONST else reg_strs[o].get(s.dtype.itemsize if not isinstance(s.dtype, PtrDType) else 8, o) if \
                (o:=str(s.reg)) in reg_strs else o for s in src if s.reg is not None]
      def _mem_adress(base:UOp, idx:UOp, disp:UOp) -> list[str]:
        return [f"[{base.reg}" + (f" + {idx.reg}*{base.dtype.itemsize}" if idx.reg else "") + (f" + {disp.arg}" if disp.arg else "") + "]"]

      if len(x.src) > 3 and x.arg in X86GroupOp.WriteMem: ret = _mem_adress(*x.src[:3]) + _format(x.src[3:])
      elif len(x.src) > 2 and x.arg in X86GroupOp.Rm1st: ret = _format((x,)) + _mem_adress(*x.src[:3]) + _format(x.src[3:])
      elif len(x.src) > 3 and x.arg in X86GroupOp.Rm2nd: ret = _format((x, x.src[0])) + _mem_adress(*x.src[1:4]) + _format(x.src[4:])
      else: ret = _format((x,) + x.src)
      return ", ".join(ret)

    asm = [f".{function_name}:"]
    for u in uops:
      if u.op is not Ops.INS: continue
      if u.arg is X86Ops.LABEL: asm.append(f"{str(u.tag)}:")
      elif u.arg is X86Ops.RET: asm.append(_format_op(u))
      else: asm.append(_format_op(u) + " " + _format_operands(u))
    return "\n".join(asm)

  def render(self, uops:list[UOp]) -> str:
    targets: dict[str, int] = {}
    jumps: dict[UOp, int] = {}
    binary = bytearray()
    for u in uops:
      if u.op is not Ops.INS: continue
      if u.arg is X86Ops.LABEL:
        targets[u.tag] = len(binary)
        continue
      if u.arg not in encodings or (l:=encodings[u.arg](u)) is None:
        raise RuntimeError(f"failed to encode {u.arg} with {u.dtype} srcs {[x.dtype for x in u.src]}")
      binary.extend(l)
      if u.arg in (X86Ops.JL, X86Ops.JB, X86Ops.JE, X86Ops.JNE, X86Ops.JGE, X86Ops.JMP): jumps[u] = len(binary)
    # fixup jump targets now that encoding size is known
    for u in uops:
      if (t:=jumps.get(u)) is not None: binary[t-4:t] = (targets[u.tag] - t).to_bytes(4, 'little', signed=True)
    return binary.hex()

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s+(dtypes.bfloat16,)}
