from __future__ import annotations
# flake8: noqa: E702
# allow semicolons to put multiple ops on one line
import struct, functools, itertools
from typing import Any, cast
from tinygrad.dtype import dtypes, DType, truncate, AddrSpace
from tinygrad.uop import FastEnum, auto, Ops, GroupOp
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, promo_dtype
from tinygrad.renderer import Renderer
from tinygrad.device import TinyELF
from tinygrad.helpers import unwrap, Target
from dataclasses import dataclass, field, replace

@dataclass(frozen=True)
class Register:
  name: str
  index: int
  _cons: tuple[Register, ...] = field(default_factory=tuple)
  size: int = 8
  @property
  def cons(self): return self._cons or (self,)
  def __repr__(self): return self.name

class IselContext:
  def __init__(self, sink:UOp):
    self.reg_n = itertools.count()
    params = sorted([u for u in sink.toposort() if u.op is Ops.PARAM], key=lambda u: u.arg.slot)
    self.arg_offsets = {u: o for u, (o, _) in zip(params, TinyELF.iter_sig(tuple((None, u.addrspace, u.dtype, ()) for u in params)))}

  def vreg(self, cons:tuple[Register, ...]|Register, size:int=0):
    cons = cons if isinstance(cons, tuple) else (cons,)
    return Register(f"v{next(self.reg_n)}", 0, cons, size or cons[0].size)

# the registers an INS defines follow its opcode and dtype in its arg
def regs(u:UOp) -> tuple[Register, ...]: return u.arg[2:] if u.op is Ops.INS else ()
def with_regs(u:UOp, r:tuple[Register, ...]) -> UOp: return u.replace(arg=u.arg[:2]+r)

def rdef(u:UOp):
  if u.op in {Ops.AFTER, Ops.BITCAST}: return rdef(u.src[0])
  return r[0] if (r:=regs(u)) else None

class LinearContext:
  def __init__(self, ren:ISARenderer):
    self.ren, self.stack_size = ren, 0
  def insert_linearized(self, uops:list[UOp]) -> list[UOp]: return uops
  def assign_spill_slot(self, r:Register, u:UOp) -> Any: raise NotImplementedError("arch specific")

class ISARenderer(Renderer):
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  pre_regalloc_matcher: PatternMatcher
  post_regalloc_matcher: PatternMatcher
  linear_ctx_type: type = LinearContext

  def is_two_address(self, x:UOp) -> bool: return False
  def spill(self, spill_slot:Any, x:UOp) -> UOp: raise NotImplementedError("arch specific")
  def fill(self, spill_slot:Any, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def asm_str(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")

# ***** X86 Ops *****

class X86Ops(FastEnum):
  # NOTE: X86Ops with i suffix are variants that take an immediate, m suffix are variants that can write to memory instead of read from
  # these aren't real instructions
  LABEL = auto(); IMM = auto(); REG = auto()
  # index
  LEA = auto()
  # register / memory / immediate moves
  MOV = auto(); MOVm = auto(); MOVi = auto(); MOVABS = auto()
  VMOVSS = auto(); VMOVSD = auto(); VMOVUPS = auto()
  VMOVSSm = auto(); VMOVSDm = auto(); VMOVUPSm = auto()
  # casts
  MOVZX = auto(); MOVSX = auto(); MOVSXD = auto()
  VCVTPH2PS = auto(); VCVTPS2PH = auto()
  VCVTSS2SD = auto(); VCVTSD2SS = auto(); VCVTSI2SS = auto(); VCVTSI2SD = auto()
  VCVTTSS2SI = auto(); VCVTTSD2SI = auto()
  # bitcasts
  VMOVD = auto(); VMOVQ = auto(); VMOVDm = auto(); VMOVQm = auto()
  # comparisons
  VCMPSS = auto(); VCMPSD = auto()
  SETNE = auto(); SETE = auto(); SETL = auto(); SETB = auto()
  # where
  CMOVNE = auto(); CMOVE = auto(); CMOVL = auto(); CMOVB = auto()
  VBLENDVPS = auto(); VBLENDVPD = auto()
  # jumps
  JNE = auto(); JE = auto(); JL = auto(); JB = auto(); JGE = auto(); JMP = auto()
  # vectorize / gep
  VINSERTPS = auto(); VPSRLDQ = auto()
  VPEXTRW = auto(); VPEXTRD = auto()
  VPINSRW = auto(); VPINSRD = auto()
  # int binary
  IDIV = auto(); DIV = auto()
  ADD = auto(); ADDi = auto(); SUB = auto(); SUBi = auto(); IMUL = auto(); IMULi = auto()
  AND = auto(); ANDi = auto(); XOR = auto(); XORi = auto(); OR = auto(); ORi = auto()
  SHL = auto(); SHLi = auto(); SHR = auto(); SHRi = auto(); SAR = auto(); SARi = auto(); CMP = auto(); CMPi = auto()
  # float unary (sometimes not unary)
  VROUNDSS = auto(); VROUNDSD = auto(); VSQRTSS = auto(); VSQRTSD = auto()
  # float binary
  VADDSS = auto(); VADDSD = auto(); VSUBSS = auto(); VSUBSD = auto(); VMULSS = auto(); VMULSD = auto(); VDIVSS = auto(); VDIVSD = auto()
  # return
  RET = auto()

class X86GroupOp:
  # X86Ops whose first src is also the destination
  TwoAddress = {X86Ops.ADD, X86Ops.ADDi, X86Ops.AND, X86Ops.ANDi, X86Ops.XOR, X86Ops.XORi, X86Ops.OR, X86Ops.ORi, X86Ops.IMUL,
                X86Ops.SUB, X86Ops.SUBi, X86Ops.SHL, X86Ops.SHLi, X86Ops.SHR, X86Ops.SHRi, X86Ops.SAR, X86Ops.SARi,
                X86Ops.IDIV, X86Ops.DIV, X86Ops.CMOVNE, X86Ops.CMOVE, X86Ops.CMOVL, X86Ops.CMOVB}

  # X86Ops whose second src is the rm field, so that src is what can be a memory operand
  Rm2nd = {X86Ops.ADD, X86Ops.SUB, X86Ops.AND, X86Ops.OR, X86Ops.XOR, X86Ops.IMUL, X86Ops.CMP,
           X86Ops.VADDSS, X86Ops.VADDSD, X86Ops.VSUBSS, X86Ops.VSUBSD, X86Ops.VMULSS, X86Ops.VMULSD, X86Ops.VDIVSS, X86Ops.VDIVSD,
           X86Ops.VBLENDVPS, X86Ops.VBLENDVPD, X86Ops.VCMPSS, X86Ops.VCMPSD, X86Ops.VROUNDSS, X86Ops.VROUNDSD, X86Ops.VSQRTSS, X86Ops.VSQRTSD,
           X86Ops.VINSERTPS, X86Ops.VPINSRW, X86Ops.VPINSRD, X86Ops.CMOVNE, X86Ops.CMOVE, X86Ops.CMOVL, X86Ops.CMOVB,
           X86Ops.VCVTSI2SS, X86Ops.VCVTSI2SD, X86Ops.VCVTSS2SD, X86Ops.VCVTSD2SS, X86Ops.IDIV, X86Ops.DIV}

  # X86Ops that can write to memory
  WriteMem = {X86Ops.MOVm, X86Ops.MOVi, X86Ops.VMOVSSm, X86Ops.VMOVSDm, X86Ops.VMOVUPSm, X86Ops.VMOVDm, X86Ops.VMOVQm,
              X86Ops.ADDi, X86Ops.SUBi, X86Ops.ANDi, X86Ops.ORi, X86Ops.XORi, X86Ops.SHL, X86Ops.SHLi, X86Ops.SHR, X86Ops.SHRi, X86Ops.SAR,
              X86Ops.SARi, X86Ops.SETNE, X86Ops.SETE, X86Ops.SETL, X86Ops.SETB,
              X86Ops.VCVTPS2PH, X86Ops.VPEXTRW, X86Ops.VPEXTRD}

  # X86Ops that read flags
  ReadFlags = {X86Ops.CMOVB, X86Ops.CMOVL, X86Ops.CMOVE, X86Ops.CMOVNE, X86Ops.SETB, X86Ops.SETL, X86Ops.SETE, X86Ops.SETNE, X86Ops.JB, X86Ops.JL,
               X86Ops.JE, X86Ops.JNE, X86Ops.JGE}

  # X86Ops that write flags or can modify flags to undefined values
  WriteFlags = {X86Ops.CMP, X86Ops.CMPi, X86Ops.ADD, X86Ops.ADDi, X86Ops.SUB, X86Ops.SUBi, X86Ops.IMUL, X86Ops.IMULi, X86Ops.IDIV, X86Ops.DIV,
                X86Ops.SHL, X86Ops.SHLi, X86Ops.SHR, X86Ops.SHRi, X86Ops.SAR, X86Ops.SARi, X86Ops.AND, X86Ops.ANDi, X86Ops.XOR, X86Ops.XORi,
                X86Ops.OR, X86Ops.ORi}

  # X86Ops that jump, their first src is what the LABEL they jump to marks
  Jump = {X86Ops.JL, X86Ops.JB, X86Ops.JE, X86Ops.JNE, X86Ops.JGE, X86Ops.JMP}

  # X86Ops whose first src is the rm field. a TwoAddress op drops its first src post regalloc, so its Rm2nd src ends up first
  Rm1st = {X86Ops.MOV, X86Ops.VMOVSS, X86Ops.VMOVSD, X86Ops.VMOVUPS, X86Ops.MOVZX, X86Ops.MOVSX, X86Ops.MOVSXD, X86Ops.VMOVD, X86Ops.VMOVQ,
           X86Ops.VCVTTSS2SI, X86Ops.VCVTTSD2SI, X86Ops.VCVTPH2PS, X86Ops.CMPi, X86Ops.IMULi, X86Ops.LEA, X86Ops.VPSRLDQ} | (Rm2nd & TwoAddress)

# ***** X86 legalization *****
def is_regbuf(x:UOp) -> bool: return x.without_after.addrspace is AddrSpace.REG and x.max_numel() == 1

extra_matcher = PatternMatcher([
  # bool CMPNE is XOR, bool CMPEQ is XOR+XOR, bool CMPLT is XOR+AND
  (UPat.var('x', dtypes.bool).alu(Ops.CMPNE, UPat.var('y')), lambda x,y: x^y),
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
   (y < 2**63).where(y.cast(dtypes.int64).cast(x.dtype), ((y >> 1) | (y & 1)).cast(dtypes.int64).cast(x.dtype) * 2)),
  # no int8 mul or cmove, cast to int16
  (UPat.var("a", dtypes.int8s) * UPat.var("b"), lambda a,b: (a.cast(dtypes.int16) * b.cast(dtypes.int16)).cast(a.dtype)),
  (UPat.var("m").where(UPat.var("a", (dtypes.bool,)+dtypes.int8s), UPat.var("b")),
   lambda m,a,b: m.where(a.cast(dtypes.int16), b.cast(dtypes.int16)).cast(a.dtype)),
  # cast to bool is a nonzero test, this cancels the cast-back the where rule emits so it doesn't survive isel
  (UPat.var("x", dtypes.ints).cast(dtypes.bool), lambda x: x.alu(Ops.CMPNE, UOp.cconst(0, x.dtype))),
  # float16 alus are done in float32
  (UPat(GroupOp.ALU, dtypes.float16, name="x"), lambda x: UOp(x.op,
   src=tuple(s.cast(dtypes.float) if s.dtype != dtypes.bool else s for s in x.src)).cast(x.dtype)),
  (UPat(GroupOp.Comparison, src=[UPat(dtype=dtypes.float16), UPat()], name="x"),
   lambda x: UOp(x.op, src=tuple(s.cast(dtypes.float32) for s in x.src)).cast(x.dtype)),
  # a float WHERE blends at the width of its value, so it needs a comparison at that width to make the mask
  (UPat.var("m", dtypes.bool).where(UPat.var("a", dtypes.floats+(dtypes.weakfloat,)), UPat.var("b")).named("w"),
   lambda m,a,b,w: m.cast(w.dtype).ne(UOp.cconst(0, w.dtype)).where(a, b)
    if w.dtype in dtypes.floats and promo_dtype(m.src) is not w.dtype else None),
  # rewrite -x -> -0.0 - x
  (UPat(Ops.NEG, name="x"), lambda x: UOp(Ops.SUB, src=(x.const_like(-0.0),) + x.src)),
  # TODO: add support for mod, requires support for accessing the 2nd+ reg of a multi output instruction
  (UPat(Ops.TRUNCMOD, src=(UPat.var("x"), UPat.var("y"))), lambda x,y: x - y * x.alu(Ops.TRUNCDIV, y)),
  # scalar reg gated mops become cmovs
  (UPat((Ops.INDEX, Ops.SHRINK), name="addr").load(UPat.var("alt"), UPat.var("gate")),
    lambda addr,alt,gate: gate.where(addr.load(), alt) if is_regbuf(addr.src[0]) else None),
  (UPat((Ops.INDEX, Ops.SHRINK), name="addr").store(UPat.var("val"), UPat.var("gate")),
    lambda addr,val,gate: addr.store(gate.where(val, addr.load())) if is_regbuf(addr.src[0]) else None),
])

# ***** X86 pre instruction selection *****
def scratch_buffer(elem_dt:DType, count:int, slot:int) -> UOp: return UOp.placeholder((count,), elem_dt, slot, AddrSpace.LOCAL)

def gated_load(ctx, addr:UOp, alt:UOp, gate:UOp, x:UOp):
  local = scratch_buffer(addr.src[0].dtype, x.max_numel(), next(ctx))
  local_idx = local.index(UOp.cconst(0, dtypes.int32))
  # the AFTER orders the load after the scratch store
  sel = gate.where(addr, local_idx)
  return UOp(Ops.AFTER, src=(sel, (local_idx if x.max_numel() == 1 else local).store(alt))).load()

def gated_store(addr:UOp, gate:UOp, val:UOp):
  local = scratch_buffer(addr.src[0].dtype, val.max_numel(), -1)
  sel = gate.where(addr, local.index(UOp.cconst(0, dtypes.int32)))
  return UOp(Ops.AFTER, src=(sel,)).store(val)

# a gate the flags can be picked with, or the bool compared to zero that replaces one they can't: only an integer
# comparison sets the flags, see cmp. NOTE: the 0 is int so the bool zero-extends and compares as int (a byte compare renders
# different kernels)
def flag_gate(m:UOp) -> UOp|None:
  return None if m.op in GroupOp.Comparison and m.src[0].dtype not in dtypes.floats else m.cast(dtypes.int).alu(Ops.CMPNE, UOp.cconst(0, dtypes.int))

# legalize the new style graph for isel. NOTE: this runs after the spec is verified, some of these rewrites violate it
pre_isel_matcher = PatternMatcher([
  # vector BUFFERs get modeled in STACK space
  (UPat((Ops.BUFFER, Ops.ALLOC), name="x"), lambda x: x.replace(arg=replace(x.arg, addrspace=AddrSpace.LOCAL))
    if x.addrspace is AddrSpace.REG and x.max_numel() > 1 else None),
  # widening a uint32 is free, the 32bit write that produced it already zeroed the upper half
  (UPat(dtype=dtypes.uint32).cast(dtypes.int64s, name="x"), lambda x: x.replace(op=Ops.BITCAST)),
  (UPat.var("y", dtypes.ints+(dtypes.bool,)).cast(dtypes.ints, name="x"),
   lambda y,x: x.replace(op=Ops.BITCAST) if x.dtype.itemsize == y.dtype.itemsize else None),
  # gated load/store become a conditional move on the address, the load/store are unconditional
  (UPat((Ops.INDEX, Ops.SHRINK), name="addr").load(UPat.var("alt"), UPat.var("gate"), name="x"), gated_load),
  (UPat((Ops.INDEX, Ops.SHRINK), name="addr").store(UPat.var("val"), UPat.var("gate")), gated_store),
  # a conditional backedge picks with the flags, and so does the cmove, which is legalized in isel
  (UPat(Ops.BACKEDGE, src=(UPat(), UPat(), UPat.var("m", dtypes.bool)), name="x"),
   lambda m,x: x.replace(src=x.src[:2]+(g,)) if (g:=flag_gate(m)) is not None else None),
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
XMM = tuple(Register(f"xmm{i}", i, size=16) for i in range(16))
# gprs you can write to
WGPR = tuple(r for r in GPR if r != RSP)

CALLEE_SAVED = (RBX, RBP, GPR[12], GPR[13], GPR[14], GPR[15])

reg_strs = {"rax": {4:"eax", 2:"ax", 1:"al"}, "rcx": {4:"ecx", 2:"cx", 1:"cl"}, "rdx": {4:"edx", 2:"dx", 1:"dl"}, "rbx": {4:"ebx", 2:"bx", 1:"bl"},
        "rsp": {4:"esp", 2:"sp", 1:"spl"}, "rbp": {4:"ebp", 2:"bp", 1:"bpl"}, "rsi": {4:"esi", 2:"si", 1:"sil"}, "rdi": {4:"edi", 2:"di", 1:"dil"},
        **{f"r{i}": {4:f"r{i}d", 2:f"r{i}w", 1:f"r{i}b"} for i in range(8, 16)}}

# a register defined outside the instruction stream: the stack pointer, the args pointer, a callee saved register
def reg(dt:DType, r:Register) -> UOp: return UOp(Ops.INS, arg=(X86Ops.REG, dt, r))
def stack_pointer() -> UOp: return reg(dtypes.uint64, RSP)
def args_pointer() -> UOp: return reg(dtypes.uint64, RDI)

# ***** X86 instruction selection *****
# a register buffer is its own virtual register, named by its identity
def regbuf_reg(x:UOp) -> Register: return Register(f"b{x.key.hex()}", 0, *((XMM, 16) if x.dtype in dtypes.floats else (WGPR, x.dtype.itemsize)))
# a value moved into a register constrained to cons
def pin(ctx:IselContext, x:UOp, cons:Register|tuple[Register, ...]) -> UOp: return x.ins(X86Ops.MOV, ctx.vreg(cons, x.dtype.itemsize), src=(x,))
def base(x:UOp, i:int) -> UOp: return s.src[0] if (s:=x.src[i]).op is Ops.INDEX else s
def lane(x:UOp, i:int) -> int: return s.src[1].src[0].val if (s:=x.src[i]).op is Ops.INDEX else 0
def to_int(dt:DType): return {dtypes.float16: dtypes.int16, dtypes.float32: dtypes.int32, dtypes.float64: dtypes.int64}[dt]
def imm(dt:DType, v:int) -> UOp: return UOp.cconst(truncate[dt](v), dt).ins(X86Ops.IMM)
def is_imm(x:UOp) -> bool: return x.op is Ops.INS and x.arg[0] is X86Ops.IMM
def to_imm(c:UOp) -> UOp|None:
  if not (c.op is Ops.CAST and (v:=c.src[0]).op is Ops.CONST): return None
  if c.dtype in dtypes.int64s: return imm(dtypes.int32, v.val) if not v.overflows(dtypes.int32) else None
  if c.dtype in dtypes.ints+(dtypes.bool,): return imm(c.dtype, v.val)
  return None
# the flag path, which only an integer comparison can take: an x86 float compare sets carry, zero and parity together when an
# operand is NaN, so a NaN reads as "below" and as "equal", and it clears sign and overflow, so nothing reads as "less"
def cmp(x:UOp) -> UOp:
  if x.src[0].dtype in dtypes.floats: raise RuntimeError(f"no flag compare for {x.src[0].dtype}, a float gate must be a mask")
  return x.ins(X86Ops.CMP, dtype=dtypes.void) if (i:=to_imm(x.src[1])) is None else x.ins(X86Ops.CMPi, dtype=dtypes.void, src=(x.src[0], i))
# comparisons that produce masks, the mask has the width of the operands
def mask(x:UOp) -> UOp:
  dt, v = x.src[0].dtype, imm(dtypes.uint8, {Ops.CMPLT: 1, Ops.CMPNE: 4, Ops.CMPEQ: 0}[x.op])
  return x.ins(X86Ops.VCMPSS if dt is dtypes.float32 else X86Ops.VCMPSD, dtype=dt, src=x.src + (v,))

# vinsertps xmm2, xmm0, xmm1, imm
# inserts any 32 bit element in xmm1 into any position in xmm0 according to immm, result is written to xmm2
# this is the fallback slow case for when you can't match more a powerful shuffle
def vinsertps(x:UOp) -> UOp:
  def _insert(ret:UOp, i:int) -> UOp:
    s, v = base(x, i), lane(x, i)
    return x.ins(X86Ops.VINSERTPS, src=(ret, s, imm(dtypes.uint8, v << 6 | i << 4)))
  return functools.reduce(_insert, range(len(x.src)), UOp(Ops.NOOP))

# vpinsrd xmm2, xmm0, eax, imm
# inserts the element in eax into any position in xmm0, result is written to xmm2 according to imm
def vpins(x:UOp, srcs:tuple[UOp, ...]) -> UOp:
  op = {2: X86Ops.VPINSRW, 4: X86Ops.VPINSRD}[x.dtype.itemsize]
  return functools.reduce(lambda ret,i: x.ins(op, src=(ret, srcs[i], imm(dtypes.uint8, i))), range(len(srcs)), UOp(Ops.NOOP))

def idiv(ctx:IselContext, x:UOp) -> UOp:
  op = X86Ops.DIV if x.dtype in dtypes.uints else X86Ops.IDIV
  val = UOp.cconst(0, x.dtype) if x.dtype in dtypes.uints else x.src[0] >> UOp.cconst(x.dtype.itemsize*8-1, x.dtype)
  ext = [] if x.dtype in dtypes.int8s else [pin(ctx, val, RDX)]
  dividend_dtype = dtypes.int16 if x.dtype in dtypes.int8s else x.dtype
  dividend = pin(ctx, x.src[0].cast(dividend_dtype), RAX)
  defs = [ctx.vreg(RAX, x.dtype.itemsize), ctx.vreg(RDX, x.dtype.itemsize)][:1+(x.dtype not in dtypes.int8s)]
  divisor = pin(ctx, x.src[1], tuple(r for r in WGPR if r not in (RAX, RDX)))
  idiv = x.ins(op, *defs, src=(dividend, divisor) + tuple(ext))
  # this move "cleanses" the register constraints (rax/rdx) of idiv
  return x.ins(X86Ops.MOV, src=(idiv,))

# a variable shift count implicitly reads cl so it goes in rcx, the shifted value can't be in rcx
def shift(ctx:IselContext, x:UOp, op:X86Ops) -> UOp:
  val = pin(ctx, x.src[0], tuple(r for r in WGPR if r is not RCX))
  cnt = pin(ctx, x.src[1], RCX)
  return x.ins(op, src=(val, cnt))

# a memory address operand is (base, index, displacement, size)
def fold_address(x:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  def _disp(v:int) -> UOp: return imm(dtypes.int32 if abs(v) > dtypes.int8.max else dtypes.int8, v)
  def _cast(v:UOp) -> UOp: return v.cast(dtypes.int64) if v.vmin < 0 else v.cast(dtypes.uint32) if v.dtype.itemsize < 4 else v
  if x.op not in {Ops.INDEX, Ops.SHRINK}: return (x, UOp(Ops.NOOP), _disp(0), imm(dtypes.uint32, x.dtype.itemsize))
  base, idx = x.src[0], x.src[1]
  # buffers are indexed by element, the stack pointer by byte
  scale = 1 if base in (stack_pointer(), args_pointer()) else base.dtype.itemsize
  sz = imm(dtypes.uint32, base.dtype.itemsize)
  if idx.op is Ops.ADD and (c:=idx.src[1]).op is Ops.CAST and c.src[0].op is Ops.CONST:
    return (base, _cast(idx.src[0]), _disp(c.src[0].val * scale), sz)
  if idx.op is Ops.CAST and idx.src[0].op is Ops.CONST: return (base, UOp(Ops.NOOP), _disp(idx.src[0].val * scale), sz)
  return (base, _cast(idx), _disp(0), sz)

# the value of a BUFFER is its address, it moves through registers and the stack as a 64bit int
def lea(x:UOp) -> UOp: return x.ins(X86Ops.LEA, src=fold_address(x), dtype=dtypes.uint64)
def abi(ctx:IselContext, x:UOp) -> UOp:
  src = fold_address(args_pointer().index(UOp.cconst(ctx.arg_offsets[x], dtypes.uint32)))
  return x.ins(X86Ops.MOV, dtype=x.dtype if x.addrspace is AddrSpace.ALU else dtypes.uint64, src=src)

GPR_DEST_OPS = {X86Ops.VPEXTRW, X86Ops.VPEXTRD, X86Ops.VCVTTSS2SI, X86Ops.VCVTTSD2SI, X86Ops.VMOVDm, X86Ops.VMOVQm}
XMM_OPS = {op for op in X86Ops if op.name.startswith('V')} - GPR_DEST_OPS

def _is_vec_xmm(y: UOp) -> bool:
  return (y.op is Ops.INS and y.arg[0] in XMM_OPS) or (y.op not in (Ops.BUFFER, Ops.ALLOC, Ops.PARAM, Ops.AFTER, Ops.INS) and y.max_numel() > 1)

def _xmm_sz(x: UOp) -> X86Ops:
  bits = x.max_numel() * x.dtype.itemsize
  if bits >= 16: return X86Ops.VMOVUPS
  if bits >= 8: return X86Ops.VMOVSD
  return X86Ops.VMOVSS

def alloc_vregs(ctx:IselContext, x:UOp) -> UOp|None:
  # register placeholders with real registers
  if is_imm(x): return None
  # no register definition
  if x.dtype is dtypes.void: return None
  # already allocated vregs
  if regs(x) and regs(x)[0]._cons: return None
  # a register buffer is a REG, a stack buffer becomes the LEA of its address before regalloc
  if x.op in {Ops.BUFFER, Ops.ALLOC}: return reg(x.dtype, regbuf_reg(x)) if x.addrspace is AddrSpace.REG else None
  if regs(x): defs = [ctx.vreg(regs(x), x.dtype.itemsize)]
  elif x.dtype in dtypes.floats or x.arg[0] in XMM_OPS: defs = [ctx.vreg(XMM)]
  else: defs = [ctx.vreg(WGPR, x.dtype.itemsize)]
  # TODO: add this once the scheduler can track register pressure
  # if x.arg[0] in X86GroupOp.WriteFlags: defs.append(ctx.vreg(RFLAGS))
  return with_regs(x, tuple(defs))

def copy_op(dt:DType) -> X86Ops:
  return {dtypes.float16:X86Ops.VMOVSS, dtypes.float32:X86Ops.VMOVSS, dtypes.float64:X86Ops.VMOVSD, dtypes.bool:X86Ops.MOVZX}.get(dt, X86Ops.MOV)

isel_matcher = PatternMatcher([
  # **** Op -> Op ****
  (UPat(Ops.RANGE, src=(UPat.cvar("c").cast(),), name="x"), lambda c,x: x.replace(src=(imm(x.dtype, c.val),))),
  # BACKEDGE becomes a conditional jump to its RANGE, it uses flags
  (UPat(Ops.BACKEDGE, src=(UPat(), UPat(), UPat(GroupOp.Comparison, name="y")), name="x"), lambda x,y: x.ins({Ops.CMPEQ: X86Ops.JE,
   Ops.CMPNE: X86Ops.JNE, Ops.CMPLT: X86Ops.JB if y.src[0].dtype in dtypes.uints else X86Ops.JL}[y.op], src=(x.src[1], x.src[0], cmp(y)))),
  # **** Op -> X86Op ****
  # function abi constraints
  (UPat(Ops.PARAM, name="x"), abi),
  # conditional moves between addresses, lea both srcs
  (UPat.var("m").where(UPat((Ops.INDEX, Ops.SHRINK), name="a"), UPat((Ops.INDEX, Ops.SHRINK), name="b")), lambda m,a,b:
   m.where(lea(a), lea(b)) if not _is_vec_xmm(a.src[0]) else None),
  # constants that can't be immediates, move them to registers
  (UPat.cvar("c").cast(dtypes.int64s, name="x"), lambda c,x: x.ins(X86Ops.MOVABS, src=(imm(x.dtype, c.val),))),
  (UPat.cvar("c").cast(dtypes.ints+(dtypes.bool,), name="x"), lambda c,x: x.ins(X86Ops.MOVi, src=(imm(x.dtype, c.val),))),
  (UPat.cvar("c").cast(dtypes.floats, name="x"), lambda c,x:
   UOp.cconst(struct.unpack((dt:=to_int(x.dtype)).fmt, struct.pack(x.dtype.fmt, c.val))[0], dt).bitcast(x.dtype)),
  # conditional moves that use masks, the mask has the width of the values
  (UPat(GroupOp.Comparison, src=(UPat(dtype=dtypes.float32), UPat()), name="m").where(UPat.var("a", dtypes.float32), UPat.var("b")), lambda m,a,b:
   a.ins(X86Ops.VBLENDVPS, src=(b, a, mask(m)))),
  (UPat(GroupOp.Comparison, src=(UPat(dtype=dtypes.float64), UPat()), name="m").where(UPat.var("a", dtypes.float64), UPat.var("b")), lambda m,a,b:
   a.ins(X86Ops.VBLENDVPD, src=(b, a, mask(m)))),
  # in this case we have a mask producing comparison whose user expects a bool, so we convert to bool
  (UPat(GroupOp.Comparison, src=(UPat.var("y", (dtypes.float32, dtypes.float64)), UPat()), name="x"), lambda y,x:
   UOp(Ops.AND, src=(mask(x).bitcast(dt:=to_int(y.dtype)), UOp.cconst(1, dt))).bitcast(dtypes.bool)),
  # conditional moves that use flags
  # TODO: remove this once we allow all flag producing ops in cmove
  # the blends took every float gate a mask can serve, so a gate that is still not an integer comparison becomes one here
  (UPat.var("m", dtypes.bool).where(UPat.var("a"), UPat.var("b")), lambda m,a,b: g.where(a, b) if (g:=flag_gate(m)) is not None else None),
  (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.sints), UPat()), name="m").where(UPat.var("a"), UPat.var("b")), lambda m,a,b:
   a.ins(X86Ops.CMOVL, src=(b, a, cmp(m)))),
  (UPat(Ops.CMPLT, name="m").where(UPat.var("a"), UPat.var("b")), lambda m,a,b: a.ins(X86Ops.CMOVB, src=(b, a, cmp(m)))),
  (UPat(Ops.CMPEQ, name="m").where(UPat.var("a"), UPat.var("b")), lambda m,a,b: a.ins(X86Ops.CMOVE, src=(b, a, cmp(m)))),
  (UPat(Ops.CMPNE, name="m").where(UPat.var("a"), UPat.var("b")), lambda m,a,b: a.ins(X86Ops.CMOVNE, src=(b, a, cmp(m)))),
  # comparisons whose user doesn't use the flag, move flag result to register
  (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.uints), UPat()), name="x"), lambda x: x.ins(X86Ops.SETB, src=(cmp(x),))),
  (UPat(Ops.CMPLT, name="x"), lambda x: x.ins(X86Ops.SETL, src=(cmp(x),))),
  (UPat(Ops.CMPEQ, name="x"), lambda x: x.ins(X86Ops.SETE, src=(cmp(x),))),
  (UPat(Ops.CMPNE, name="x"), lambda x: x.ins(X86Ops.SETNE, src=(cmp(x),))),
  # float unary
  (UPat.var("y", dtypes.float32).sqrt().named("x"), lambda y,x: x.ins(X86Ops.VSQRTSS, src=(y, y))),
  (UPat.var("y", dtypes.float64).sqrt().named("x"), lambda y,x: x.ins(X86Ops.VSQRTSD, src=(y, y))),
  (UPat.var("y", dtypes.float32).trunc().named("x"), lambda y,x: x.ins(X86Ops.VROUNDSS, src=(y, y, imm(dtypes.uint8, 3)))),
  (UPat.var("y", dtypes.float64).trunc().named("x"), lambda y,x: x.ins(X86Ops.VROUNDSD, src=(y, y, imm(dtypes.uint8, 3)))),
  # for float16 we route the srcs through gprs, this is suboptimal for values in xmms, in that case we want vpunpcklwd
  (UPat(Ops.STACK, dtypes.float16, name="x"), lambda x: vpins(x, tuple(s.bitcast(dtypes.int16) for s in x.src))),
  (UPat(Ops.STACK, dtypes.float32, name="x"), vinsertps),
  (UPat(Ops.STACK, dtypes.int32s, name="x"), lambda x: vpins(x, x.src)),
  # INDEX on a vector register value extracts a single element
  (UPat.var("y", dtypes.int32s).index(UPat.cvar("c").cast(), name="x"),
   lambda y,c,x: x.ins(X86Ops.VPEXTRD, src=(y, imm(dtypes.uint8, c.val))) if _is_vec_xmm(y) else None),
  (UPat.var("y", dtypes.floats).index(UPat.cvar("c").cast(), name="x"),
   lambda y,c,x: x.ins(X86Ops.VPSRLDQ, src=(y, imm(dtypes.uint8, c.val * x.dtype.itemsize))) if _is_vec_xmm(y) else None),
  # int binary
  ((UPat(dtype=dtypes.ints).alu(Ops.TRUNCDIV, UPat())).named("x"), idiv),
  # int binary with immediate
  (UPat.var("a", dtypes.ints) << UPat.cvar("c").cast(), lambda a,c: a.ins(X86Ops.SHLi, src=(a, imm(dtypes.uint8, c.val)))),
  (UPat.var("a", dtypes.uints) >> UPat.cvar("c").cast(), lambda a,c: a.ins(X86Ops.SHRi, src=(a, imm(dtypes.uint8, c.val)))),
  (UPat.var("a", dtypes.sints) >> UPat.cvar("c").cast(), lambda a,c: a.ins(X86Ops.SARi, src=(a, imm(dtypes.uint8, c.val)))),
  (UPat.var("a", dtypes.ints) + UPat.cvar().cast(name="c"), lambda a,c: a.ins(X86Ops.ADDi, src=(a, i)) if (i:=to_imm(c)) is not None else None),
  (UPat.var("a", dtypes.ints) * UPat.cvar().cast(name="c"), lambda a,c: a.ins(X86Ops.IMULi, src=(a, i)) if (i:=to_imm(c)) is not None else None),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) & UPat.cvar().cast(name="c"),
   lambda a,c: a.ins(X86Ops.ANDi, src=(a, i)) if (i:=to_imm(c)) is not None else None),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) | UPat.cvar().cast(name="c"),
   lambda a,c: a.ins(X86Ops.ORi, src=(a, i)) if (i:=to_imm(c)) is not None else None),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) ^ UPat.cvar().cast(name="c"),
   lambda a,c: a.ins(X86Ops.XORi, src=(a, i)) if (i:=to_imm(c)) is not None else None),
  (UPat(Ops.SUB, dtypes.ints, (UPat.var("a"), UPat.cvar().cast(name="c"))),
   lambda a,c: a.ins(X86Ops.SUBi, src=(a, i)) if (i:=to_imm(c)) is not None else None),
  # int binary with register
  ((UPat(dtype=dtypes.ints) << UPat()).named("x"), lambda ctx,x: shift(ctx, x, X86Ops.SHL)),
  ((UPat(dtype=dtypes.uints) >> UPat()).named("x"), lambda ctx,x: shift(ctx, x, X86Ops.SHR)),
  ((UPat(dtype=dtypes.sints) >> UPat()).named("x"), lambda ctx,x: shift(ctx, x, X86Ops.SAR)),
  (UPat.var("a", dtypes.ints) + UPat.var("b"), lambda a,b: a.ins(X86Ops.ADD, src=(a, b))),
  (UPat.var("a", dtypes.ints) * UPat.var("b"), lambda a,b: a.ins(X86Ops.IMUL, src=(a, b))),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) & UPat.var("b"), lambda a,b: a.ins(X86Ops.AND, src=(a, b))),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) | UPat.var("b"), lambda a,b: a.ins(X86Ops.OR, src=(a, b))),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) ^ UPat.var("b"), lambda a,b: a.ins(X86Ops.XOR, src=(a, b))),
  (UPat(Ops.SUB, dtypes.ints, (UPat.var("a"), UPat.var("b"))), lambda a,b: a.ins(X86Ops.SUB, src=(a, b))),
  # float binary
  ((UPat(dtype=dtypes.float32) + UPat()).named("x"), lambda x: x.ins(X86Ops.VADDSS)),
  ((UPat(dtype=dtypes.float64) + UPat()).named("x"), lambda x: x.ins(X86Ops.VADDSD)),
  ((UPat(dtype=dtypes.float32) * UPat()).named("x"), lambda x: x.ins(X86Ops.VMULSS)),
  ((UPat(dtype=dtypes.float64) * UPat()).named("x"), lambda x: x.ins(X86Ops.VMULSD)),
  (UPat(Ops.SUB, dtypes.float32, name="x"), lambda x: x.ins(X86Ops.VSUBSS)),
  (UPat(Ops.SUB, dtypes.float64, name="x"), lambda x: x.ins(X86Ops.VSUBSD)),
  (UPat(Ops.FDIV, dtypes.float32, name="x"), lambda x: x.ins(X86Ops.VDIVSS)),
  (UPat(Ops.FDIV, dtypes.float64, name="x"), lambda x: x.ins(X86Ops.VDIVSD)),
  # casts
  (UPat(dtype=dtypes.float32).cast(dtypes.float16, name="x"), lambda x: x.ins(X86Ops.VCVTPS2PH, src=x.src + (imm(dtypes.uint8, 4),))),
  (UPat(dtype=dtypes.float16).cast(dtypes.float32, name="x"), lambda x: x.ins(X86Ops.VCVTPH2PS)),
  (UPat(dtype=dtypes.float32).cast(dtypes.int32s+dtypes.int64s, name="x"), lambda x: x.ins(X86Ops.VCVTTSS2SI)),
  (UPat(dtype=dtypes.float64).cast(dtypes.int32s+dtypes.int64s, name="x"), lambda x: x.ins(X86Ops.VCVTTSD2SI)),
  (UPat.var("y", dtypes.float32).cast(dtypes.float64, name="x"), lambda y,x: x.ins(X86Ops.VCVTSS2SD, src=(y, y))),
  (UPat.var("y", dtypes.float64).cast(dtypes.float32, name="x"), lambda y,x: x.ins(X86Ops.VCVTSD2SS, src=(y, y))),
  (UPat.var("y", (dtypes.int32, dtypes.int64)).cast(dtypes.float32, name="x"), lambda y,x: x.ins(X86Ops.VCVTSI2SS, src=(UOp(Ops.NOOP), y))),
  (UPat.var("y", (dtypes.int32, dtypes.int64)).cast(dtypes.float64, name="x"), lambda y,x: x.ins(X86Ops.VCVTSI2SD, src=(UOp(Ops.NOOP), y))),
  (UPat(dtype=(dtypes.uint8, dtypes.uint16, dtypes.bool)).cast(dtypes.ints, name="x"), lambda x:
   x.ins(X86Ops.MOVZX) if x.src[0].dtype.itemsize < x.dtype.itemsize else None),
  (UPat(dtype=dtypes.int32).cast(dtypes.int64s, name="x"), lambda x: x.ins(X86Ops.MOVSXD)),
  (UPat(dtype=dtypes.sints).cast(dtypes.ints, name="x"), lambda x: x.ins(X86Ops.MOVSX) if x.src[0].dtype.itemsize < x.dtype.itemsize else None),
  (UPat(dtype=dtypes.ints).cast(dtypes.ints, name="x"), lambda x: x.ins(X86Ops.MOV)),
  # bitcasts between scalar floats and ints
  (UPat.var("y", dtypes.float16).bitcast(dtypes.int16s).named("x"), lambda y,x: x.ins(X86Ops.VPEXTRW, src=(y, imm(dtypes.uint8, 0)))),
  (UPat(dtype=dtypes.int16s).bitcast(dtypes.float16).named("x"), lambda x: vpins(x, x.src)),
  (UPat(dtype=dtypes.int32s).bitcast(dtypes.float32).named("x"), lambda x: x.ins(X86Ops.VMOVD)),
  (UPat(dtype=dtypes.int64s).bitcast(dtypes.float64).named("x"), lambda x: x.ins(X86Ops.VMOVQ)),
  (UPat(dtype=dtypes.float32).bitcast(dtypes.int32s).named("x"), lambda x: x.ins(X86Ops.VMOVDm)),
  (UPat(dtype=dtypes.float64).bitcast(dtypes.int64s).named("x"), lambda x: x.ins(X86Ops.VMOVQm)),
  # lower register mops: a store is just a copy, load is an anon copy to preserve ordering
  (UPat.var("a").store(UPat.var("val")), lambda a,val:
    UOp(Ops.STACK).after(buf.ins(copy_op(val.dtype), regbuf_reg(buf.without_after), src=(val.after(buf),)))
    if is_regbuf((buf := a.src[0] if a.op is Ops.INDEX else a)) else None),
  (UPat.var("a").load().named("x"), lambda a,x:
    x.ins(copy_op(x.dtype), src=(buf,)) if is_regbuf((buf:=a.src[0] if a.op is Ops.INDEX else a)) else None),
  # index on a buffer (or the stack pointer) computes an address, addresses are 64bit values
  (UPat((Ops.INDEX, Ops.SHRINK), name="x"), lambda x: lea(x) if not _is_vec_xmm(x.src[0]) and x.addrspace is not AddrSpace.REG else None),
  # TODO: fuse stores, very few cases -- store cmp becomes setcc, store gep int becomes vpextr, store bitcast to int becomes vmovd/q
  # load, store
  (UPat(Ops.LOAD, dtypes.floats, src=(UPat(name="a"),), name="x"), lambda x,a: None if a.addrspace is AddrSpace.REG else
   x.ins(X86Ops.VPINSRW, src=(UOp(Ops.NOOP),) + fold_address(a) + (imm(dtypes.uint8, 0),)) if x.max_numel() * x.dtype.itemsize == 2 else
   x.ins(_xmm_sz(x), src=fold_address(a))),
  (UPat(Ops.LOAD, dtypes.ints+(dtypes.bool,), src=(UPat(name="a"),), name="x"), lambda x,a: None if a.addrspace is AddrSpace.REG else
   x.ins(X86Ops.MOV, src=fold_address(a)) if x.max_numel() == 1 else x.ins(_xmm_sz(x), src=fold_address(a))),
  (UPat.var("a").store(UPat.var("b", dtypes.floats), name="x"), lambda a,b,x: None if a.addrspace is AddrSpace.REG else
   x.ins(X86Ops.VPEXTRW, src=fold_address(a) + (b, imm(dtypes.uint8, 0))) if b.max_numel() * b.dtype.itemsize == 2 else
   x.ins(X86Ops[_xmm_sz(b).name+"m"], src=fold_address(a) + (b,))),
  (UPat.var("a").store(UPat.var("b", dtypes.ints+(dtypes.bool,)), name="x"), lambda a,b,x: None if a.addrspace is AddrSpace.REG else
   x.ins(X86Ops[_xmm_sz(b).name+"m"], src=fold_address(a) + (b,)) if b.max_numel() > 1 else
   x.ins(X86Ops.MOVm, src=fold_address(a) + (b,)) if (i:=to_imm(b)) is None else x.ins(X86Ops.MOVi, src=fold_address(a) + (i,))),
  # allocate virtual registers
  (UPat((Ops.INS, Ops.BUFFER, Ops.ALLOC), name="x"), alloc_vregs),
])

# ***** pre register allocation *****
# the flags belong to the last instruction that wrote them. x86 has no good way to store/restore them (then regalloc would
# handle it), so a consumer that no longer owns its compare re-emits it. Unlike a regalloc rematerialization this is not
# optional, there is no fallback load from stack
def flag_rematerialize(ctx:X86LinearContext, x:UOp):
  if x.op in (Ops.RANGE, Ops.END) or x.arg[0] in X86GroupOp.WriteFlags: ctx.lock = x
  elif x.arg[0] in X86GroupOp.ReadFlags and ctx.lock is not (flag_def:=x.src[-1]):
    ctx.lock = flag_def
    return (x, [flag_def, x])
  return None

# the address of a stack buffer keeps the buffer's dtype so the element size of loads and stores through it is known
def alloc_buffer(ctx:X86LinearContext, x:UOp):
  nx = UOp(Ops.INS, fold_address(stack_pointer().index(UOp.cconst(ctx.stack_size, dtypes.uint32))),
           (X86Ops.LEA, x.dtype, Register(f"s{ctx.stack_size}", 0, WGPR, 8)))
  ctx.stack_size += x.max_numel() * x.dtype.itemsize
  return nx, [stack_pointer(), nx]

# TODO: control flow should be overhauled so that this isn't necessary
# a LABEL marks its src: the start of a loop its RANGE, the exit of a loop the start LABEL. a jump's first src is what its LABEL marks
def label(x:UOp) -> UOp: return UOp(Ops.INS, arg=(X86Ops.LABEL, dtypes.void), src=(x,))

# a loop's LABEL follows its RANGE. a bounded RANGE's index is a register named by the RANGE, 0 before the loop, compared to the bound
# after the LABEL and incremented before the END, the body reads it as AFTER(index, RANGE)
def lower_range(x:UOp) -> tuple[UOp, list[UOp]]:
  if x.dtype is dtypes.void: return x, [x, label(x)]
  bound, deps = (x.src[0].src[0], x.src[0].src[1:]) if x.src[0].op is Ops.AFTER else (x.src[0], ())
  idx = UOp(Ops.INS, arg=(X86Ops.MOVi, x.dtype, regbuf_reg(x)), src=(imm(x.dtype, 0),) + deps)
  cmp = UOp(Ops.INS, arg=(X86Ops.CMPi if is_imm(bound) else X86Ops.CMP, dtypes.void), src=(idx, bound))
  return idx.after(x), [idx, x, start:=label(x), cmp, UOp(Ops.INS, arg=(X86Ops.JGE, dtypes.void), src=(start, cmp))]

def lower_end(x:UOp, c:UOp) -> tuple[UOp, list[UOp]]:
  return (end:=x.replace(src=(x.src[0], c.src[1]))), [c.ins(X86Ops.ADDi, *regs(c.src[0]), src=(c, imm(c.dtype, 1))), end]

pre_regalloc_matcher = PatternMatcher([
  (UPat(Ops.BUFFER, name="x"), lambda ctx,x: alloc_buffer(ctx,x) if x.addrspace is not AddrSpace.REG else None),
  (UPat((Ops.INS, Ops.RANGE, Ops.END), name="x"), flag_rematerialize),
  (UPat(Ops.RANGE, name="x"), lower_range),
  (UPat(Ops.END, src=(UPat(), UPat(Ops.AFTER, name="c")), name="x"), lower_end),
])

# ***** post register allocation *****
# final rewrite to match the isa spec
post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x: (x, [x.ins(X86Ops.RET)])),
  # an END jumps back to its loop's LABEL and labels the loop's exit
  (UPat(Ops.END, name="x"), lambda x: (x, [UOp(Ops.INS, arg=(X86Ops.JMP, dtypes.void), src=(x.src[1],)), label(label(x.src[1]))])),
  # rewrite two address instructions to two address form, if reused src wasn't coalesced insert a move
  (UPat(Ops.INS, name="x"), lambda ctx,x: (nx:=x.replace(src=x.src[1:]),
   [ctx.ren.copy(x.src[0], rdef(x)), nx] if rdef(x) != rdef(x.src[0]) else [nx]) if x.arg[0] in X86GroupOp.TwoAddress else None),
])

# ***** X86 instruction encoding *****

def encode(x:UOp, opc:int, reg:int|None=None, pp:int=0, sel:int=0, we:int=0) -> bytes|None:
  def _encode(reg_uop:UOp|None, rm_uop:UOp, idx_uop:UOp|None=None, disp_uop:UOp|None=None, scale_uop:UOp|None=None,
              vvvv_uop:UOp|None=None, imm_uop:UOp|None=None) -> bytes:
    nonlocal reg, opc
    # get the encoding values of the different fields
    reg = cast(int, cast(Register, rdef(reg_uop)).index if reg_uop is not None else reg)
    rm = cast(Register, rdef(rm_uop)).index
    idx = cast(Register, rdef(idx_uop)).index if idx_uop is not None and rdef(idx_uop) is not None else 4
    # TODO: shouldn't rely on bitcast to convey encoding info
    def _sz(x:UOp): return x.dtype.itemsize if x.op is Ops.BITCAST else rdef(x).size
    # for a memory operand the rm size is the element size carried by the address, otherwise it's the size of the value in the register
    rm_sz = scale_uop.src[0].val if scale_uop is not None else _sz(rm_uop)
    reg_sz = _sz(reg_uop) if reg_uop is not None else 0
    sz = reg_sz or rm_sz

    # encode instruction
    inst = bytes([])
    assert 0 <= reg <= 15 and 0 <= idx <= 15 and 0 <= rm <= 15
    # r extends reg field, x extends index field, b extends rm or base field
    r, _x, b = reg >> 3, idx >> 3, rm >> 3
    if sel: # VEX bytes
      vvvv = (vd.index if isinstance(vd := rdef(vvvv_uop), Register) else reg) if vvvv_uop is not None else 0
      if sel == 1 and _x == b == we == 0: inst += bytes([0xC5, (~r & 0b1) << 7 | (~vvvv & 0b1111) << 3 | pp])
      else: inst += bytes([0xC4, (~r & 0b1) << 7 | (~_x & 0b1) << 6 | (~b & 0b1) << 5 | sel, we << 7 | (~vvvv & 0b1111) << 3 | pp])
    else: # optional PREFIX and REX bytes
      # PREFIX byte signaling 16 bit variant of instruction
      if sz == 2: inst += bytes([0x66])
      # bit signaling 64 bit variant of instruction
      w = sz == 8
      # legacy 8bit opcode is 1 less than 16-64bit variants
      demote = (rm_sz == 1 or reg_sz == 1) and x.arg[0] not in X86GroupOp.ReadFlags | {X86Ops.LEA}
      # REX byte is required when 64 bit or an extended reg is used (index 8 - 15) or lower 8 bits of (rsp, rbp, rsi, rdi) are accessed
      if w | r | _x | b | (reg_sz == 1 & reg >> 2) | (rm_sz == 1 & rm >> 2) | (demote and disp_uop is None and rm >= 4):
        inst += bytes([0b0100 << 4 | w << 3 | r << 2 | _x << 1 | b])
      if demote: opc -= 1
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
      assert is_imm(disp_uop), "displacement must be an immediate"
      assert disp_uop.dtype in (dtypes.int8, dtypes.int32), "displacement can only be 1 or 4 byte signed int"
      # rbp/r13 always require a displacement
      if disp_uop.src[0].val != 0 or rm == 0b101: mod = 0b01 if disp_uop.dtype.itemsize == 1 else 0b10
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
      inst += struct.pack(unwrap(disp_uop.dtype.fmt), disp_uop.src[0].val)
    # IMM byte
    if imm_uop is not None:
      if is_imm(imm_uop): inst += struct.pack(unwrap(imm_uop.dtype.fmt), imm_uop.src[0].val)
      elif isinstance(rdef(imm_uop), Register): inst += bytes([(rdef(imm_uop).index & 0b1111) << 4 | 0b0000])
    return inst

  # get the encoding structure of the uop
  # when a uop writes to memory it takes the form of a store, dtype is void, no definition
  address:tuple[UOp|None, ...]
  if x.arg[0] in X86GroupOp.WriteMem:
    if len(x.src) > 4: address, rest = x.src[:4], x.src[4:]
    else: address, rest = (x, None, None, None), x.src
    imm_uop = rest[:1] if rest and is_imm(rest[0]) else (None,)
    return _encode(rest[0], *address, *(None, *rest[1:])) if reg is None else _encode(None, *address, *(None, *imm_uop))

  if x.arg[0] in X86GroupOp.Rm1st:
    if len(x.src) > 3: address, rest = x.src[:4], x.src[4:]
    else: address, rest = (x.src[0], None, None, None), x.src[1:]
    imm_uop = rest[:1] if rest and is_imm(rest[0]) else (None,)
    return _encode(x, *address, *(None, *imm_uop)) if reg is None else _encode(None, *address, *(x if sel else None, *imm_uop))

  if x.arg[0] in X86GroupOp.Rm2nd:
    if len(x.src) > 4: address, rest = x.src[1:5], x.src[:1] + x.src[5:]
    else: address, rest = (x.src[1], None, None, None), x.src[:1] + x.src[2:]
    # cmp reg, rm doesn't define a new register
    return _encode(x, *address, *rest) if x.dtype is not dtypes.void else _encode(rest[0], *address)

  return None

# https://www.felixcloutier.com/x86/
# legacy version -> VEX version
# prefix field: None -> 0 | 66 -> 1 | F3 -> 2 | F2 -> 3
# opcode map select: 0F -> 1 | 0F38 -> 2 | 0F3A -> 3
encodings = {
  # moves
  X86Ops.MOVABS: lambda x:
   bytes([0b0100 << 4 | 0b1 << 3 | 0b00 << 2 | rdef(x).index >> 3, 0xB8 + (rdef(x).index & 0b111)]) + struct.pack(x.dtype.fmt, x.src[0].src[0].val),
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
  X86Ops.VCVTSS2SD: lambda x: encode(x, 0x5A, pp=2, sel=1), X86Ops.VCVTSD2SS: lambda x: encode(x, 0x5A, pp=3, sel=1),
  X86Ops.VCVTPH2PS: lambda x: encode(x, 0x13, pp=1, sel=2), X86Ops.VCVTPS2PH: lambda x: encode(x, 0x1D, pp=1, sel=3),
  # the int src is the 2nd src (the rm field), its width picks the 32 or 64 bit form
  X86Ops.VCVTSI2SS: lambda x: encode(x, 0x2A, pp=2, sel=1, we=x.src[1].dtype.itemsize == 8),
  X86Ops.VCVTSI2SD: lambda x: encode(x, 0x2A, pp=3, sel=1, we=x.src[1].dtype.itemsize == 8),
  X86Ops.VCVTTSS2SI: lambda x: encode(x, 0x2C, pp=2, sel=1, we=x.dtype.itemsize == 8),
  X86Ops.VCVTTSD2SI: lambda x: encode(x, 0x2C, pp=3, sel=1, we=x.dtype.itemsize == 8),
  # int division
  X86Ops.IDIV: lambda x: encode(x, 0xF7, reg=7), X86Ops.DIV: lambda x: encode(x, 0xF7, reg=6),
  # scalar int binary
  X86Ops.SHL: lambda x: encode(x, 0xD3, reg=4), X86Ops.SHLi: lambda x: encode(x, 0xC1, reg=4),
  X86Ops.SHR: lambda x: encode(x, 0xD3, reg=5), X86Ops.SHRi: lambda x: encode(x, 0xC1, reg=5),
  X86Ops.SAR: lambda x: encode(x, 0xD3, reg=7), X86Ops.SARi: lambda x: encode(x, 0xC1, reg=7),
  X86Ops.ADD: lambda x: encode(x, 0x03), X86Ops.ADDi: lambda x: encode(x, 0x81, reg=0),
  X86Ops.SUB: lambda x: encode(x, 0x2B), X86Ops.SUBi: lambda x: encode(x, 0x81, reg=5),
  X86Ops.AND: lambda x: encode(x, 0x23), X86Ops.ANDi: lambda x: encode(x, 0x81, reg=4),
  X86Ops.XOR: lambda x: encode(x, 0x33), X86Ops.XORi: lambda x: encode(x, 0x81, reg=6),
  X86Ops.OR: lambda x: encode(x, 0x0B), X86Ops.ORi: lambda x: encode(x, 0x81, reg=1),
  X86Ops.CMP: lambda x: encode(x, 0x3B), X86Ops.CMPi: lambda x: encode(x, 0x81, reg=7),
  X86Ops.IMUL: lambda x: encode(x, 0x0FAF), X86Ops.IMULi: lambda x: encode(x, 0x69),
  X86Ops.SETB: lambda x: encode(x, 0x0F92, reg=0), X86Ops.SETL: lambda x: encode(x, 0x0F9C, reg=0),
  X86Ops.SETE: lambda x: encode(x, 0x0F94, reg=0), X86Ops.SETNE: lambda x: encode(x, 0x0F95, reg=0),
  # unary
  X86Ops.VSQRTSS: lambda x: encode(x, 0x51, pp=2, sel=1), X86Ops.VSQRTSD: lambda x: encode(x, 0x51, pp=3, sel=1),
  X86Ops.VROUNDSS: lambda x: encode(x, 0x0A, pp=1, sel=3), X86Ops.VROUNDSD: lambda x: encode(x, 0x0B, pp=1, sel=3),
  # float binary
  X86Ops.VADDSS: lambda x: encode(x, 0x58, pp=2, sel=1), X86Ops.VADDSD: lambda x: encode(x, 0x58, pp=3, sel=1),
  X86Ops.VSUBSS: lambda x: encode(x, 0x5C, pp=2, sel=1), X86Ops.VSUBSD: lambda x: encode(x, 0x5C, pp=3, sel=1),
  X86Ops.VMULSS: lambda x: encode(x, 0x59, pp=2, sel=1), X86Ops.VMULSD: lambda x: encode(x, 0x59, pp=3, sel=1),
  X86Ops.VDIVSS: lambda x: encode(x, 0x5E, pp=2, sel=1), X86Ops.VDIVSD: lambda x: encode(x, 0x5E, pp=3, sel=1),
  X86Ops.VCMPSS: lambda x: encode(x, 0xC2, pp=2, sel=1), X86Ops.VCMPSD: lambda x: encode(x, 0xC2, pp=3, sel=1),
  # ternary
  X86Ops.CMOVB: lambda x: encode(x, 0x0F42), X86Ops.CMOVL: lambda x: encode(x, 0x0F4C),
  X86Ops.CMOVE: lambda x: encode(x, 0x0F44), X86Ops.CMOVNE: lambda x: encode(x, 0x0F45),
  X86Ops.VBLENDVPS: lambda x: encode(x, 0x4A, pp=1, sel=3), X86Ops.VBLENDVPD: lambda x: encode(x, 0x4B, pp=1, sel=3),
  # shuffles
  X86Ops.VPSRLDQ: lambda x: encode(x, 0x73, reg=3, pp=1, sel=1),
  X86Ops.VPINSRW: lambda x: encode(x, 0xC4, pp=1, sel=1), X86Ops.VPINSRD: lambda x: encode(x, 0x22, pp=1, sel=3),
  X86Ops.VINSERTPS: lambda x: encode(x, 0x21, pp=1, sel=3),
  # extract
  X86Ops.VPEXTRW: lambda x: encode(x, 0x15, pp=1, sel=3), X86Ops.VPEXTRD: lambda x: encode(x, 0x16, pp=1, sel=3),
  # jumps are encoded with a placeholder which gets patched later once the real offset is known
  X86Ops.JE: lambda x: bytes([0x0F, 0x84]) + int(0).to_bytes(4),
  X86Ops.JNE: lambda x: bytes([0x0F, 0x85]) + int(0).to_bytes(4),
  X86Ops.JL: lambda x: bytes([0x0F, 0x8C]) + int(0).to_bytes(4),
  X86Ops.JB: lambda x: bytes([0x0F, 0x82]) + int(0).to_bytes(4),
  X86Ops.JGE: lambda x: bytes([0x0F, 0x8D]) + int(0).to_bytes(4),
  X86Ops.JMP: lambda x: bytes([0xE9]) + int(0).to_bytes(4),
  X86Ops.RET: lambda x: bytes([0xC3]),
}

class X86LinearContext(LinearContext):
  def __init__(self, ren:X86Renderer):
    super().__init__(ren)
    self.lock: UOp|None = None

  # stack alloc/dealloc and callee saved handling
  def insert_linearized(self, uops:list[UOp]) -> list[UOp]:
    before, after, used = [], [], {r.name for u in uops if isinstance((r:=rdef(u)), Register)}
    for r in CALLEE_SAVED:
      if r.name in used:
        a = reg(dtypes.uint64, r)
        slot = self.assign_spill_slot(r,a)
        before.append(self.ren.spill(slot,a))
        after.append(self.ren.fill(slot,a,r))

    if self.stack_size > 0:
      before.insert(0, stack_pointer().ins(X86Ops.SUBi, RSP, src=(stack_pointer(),imm(dtypes.int32, self.stack_size))))
      after.append(stack_pointer().ins(X86Ops.ADDi, RSP, src=(stack_pointer(),imm(dtypes.int32, self.stack_size))))
    return before + uops[:-1] + after + [uops[-1]]

  def assign_spill_slot(self, r:Register, u:UOp) -> int:
    sz = r.cons[0].size
    offset = self.stack_size + (sz - self.stack_size % sz) %sz
    self.stack_size = offset + sz
    return offset

class X86Renderer(ISARenderer):
  device = "CPU"
  has_local = False
  global_max = (1, 0, 0)
  extra_matcher = extra_matcher
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  pre_regalloc_matcher = pre_regalloc_matcher
  post_regalloc_matcher = post_regalloc_matcher
  code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.AND, Ops.OR, Ops.SHL, Ops.SHR, Ops.NEG, Ops.SUB, Ops.FDIV, Ops.CMPLT, Ops.CMPNE)}
  linear_ctx_type = X86LinearContext
  def __init__(self, target:Target):
    if target.arch.split(",")[0] != "x86_64": raise RuntimeError(f"X86Renderer only supports x86_64, got {target.arch}")
    super().__init__(target)
    from tinygrad.runtime.support.compiler_cpu import X86Compiler
    self.compiler = X86Compiler()
  def is_two_address(self, x:UOp) -> bool: return x.op is Ops.INS and x.arg[0] in X86GroupOp.TwoAddress
  def copy(self, x:UOp, reg:Register) -> UOp: return x.ins(X86Ops.MOV, reg, src=(x,))

  def spill(self, spill_slot:int, x:UOp) -> UOp:
    op = X86Ops.VMOVUPSm if rdef(x).cons[0] in XMM else X86Ops.MOVm
    disp = UOp.cconst(spill_slot, dtypes.int32)
    return UOp(Ops.INS, src=fold_address(stack_pointer().index(disp)) + (x,), arg=(op, dtypes.void))

  def fill(self, spill_slot:int, x:UOp, reg:Register) -> UOp:
    disp = UOp.cconst(spill_slot, dtypes.int32)
    return UOp(Ops.INS, src=fold_address(stack_pointer().index(disp)), arg=(X86Ops.VMOVUPS if reg.cons[0] in XMM else X86Ops.MOV, x.dtype, reg))

  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    def _format_op(x:UOp) -> str: return f"    {(o[7:-1] if (o:=str(x.arg[0]))[-1] in ('i', 'm') else o[7:]).lower():7s}"
    def _format_operands(x:UOp) -> str:
      def _format(src:tuple[UOp, ...]) -> list[str]:
        return [str(s.src[0].val) if is_imm(s) else reg_strs[o].get(rdef(s).size, o) if \
                (o:=str(rdef(s))) in reg_strs else o for s in src if is_imm(s) or rdef(s) is not None]
      def _mem_adress(base:UOp, idx:UOp, disp:UOp, scale:UOp) -> list[str]:
        return [f"[{rdef(base)}" + (f" + {rdef(idx)}*{scale.src[0].val}" if rdef(idx) else "") + (f" + {d}" if (d:=disp.src[0].val) else "") + "]"]

      if len(x.src) > 4 and x.arg[0] in X86GroupOp.WriteMem: ret = _mem_adress(*x.src[:4]) + _format(x.src[4:])
      elif len(x.src) > 3 and x.arg[0] in X86GroupOp.Rm1st: ret = _format((x,)) + _mem_adress(*x.src[:4]) + _format(x.src[4:])
      elif len(x.src) > 4 and x.arg[0] in X86GroupOp.Rm2nd: ret = _format((x, x.src[0])) + _mem_adress(*x.src[1:5]) + _format(x.src[5:])
      else: ret = _format((x,) + x.src)
      return ", ".join(ret)

    asm = [f".{function_name}:"]
    for u in uops:
      if u.op is not Ops.INS or u.arg[0] in {X86Ops.IMM, X86Ops.REG}: continue
      if u.arg[0] is X86Ops.LABEL: asm.append(f".L{uops.index(u)}:")
      elif u.arg[0] is X86Ops.RET: asm.append(_format_op(u))
      elif u.arg[0] in X86GroupOp.Jump: asm.append(_format_op(u) + f" .L{uops.index(label(u.src[0]))}")
      else: asm.append(_format_op(u) + " " + _format_operands(u))
    return "\n".join(asm)

  def render(self, uops:list[UOp]) -> str:
    targets: dict[UOp, int] = {}
    jumps: dict[UOp, int] = {}
    binary = bytearray()
    for u in uops:
      if u.op is not Ops.INS or u.arg[0] in {X86Ops.IMM, X86Ops.REG}: continue
      if u.arg[0] is X86Ops.LABEL:
        targets[u.src[0]] = len(binary)
        continue
      if u.arg[0] not in encodings or (l:=encodings[u.arg[0]](u)) is None:
        raise RuntimeError(f"failed to encode {u.arg[0]} with {u.dtype} srcs {[x.dtype for x in u.src]}")
      binary.extend(l)
      if u.arg[0] in X86GroupOp.Jump: jumps[u] = len(binary)
    # fixup jump targets now that encoding size is known
    for u in uops:
      if (t:=jumps.get(u)) is not None: binary[t-4:t] = (targets[u.src[0]] - t).to_bytes(4, 'little', signed=True)
    return binary.hex()

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s+(dtypes.bfloat16,)}
