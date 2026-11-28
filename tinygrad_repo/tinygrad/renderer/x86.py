# flake8: noqa: E702
# allow semicolons to put multiple ops on one line
import struct, functools, itertools
from tinygrad.dtype import dtypes, DType, AddrSpace
from tinygrad.uop import FastEnum, auto, Ops, GroupOp
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, promo_dtype
from tinygrad.renderer import Renderer
from tinygrad.helpers import unwrap, Target

def vreg(slot:int, dtype:DType, choices:tuple[UOp, ...], name:str="vreg") -> tuple[UOp, ...]:
  return UOp.param(slot, dtype, name=name, addrspace=AddrSpace.REG), UOp(Ops.STACK, src=choices)

class IselContext:
  def __init__(self): self.reg_n = itertools.count()
  def vreg(self, dtype:DType, choices:tuple[UOp, ...]): return vreg(next(self.reg_n), dtype, choices)

# the registers an ins defines exist in the src of NATIVE, with additional definitions in an AFTER
def regs(u:UOp) -> tuple[UOp, ...]:
  if u.op is Ops.BUFFER and u.addrspace is AddrSpace.REG: return regbuf_reg(u)
  if u.op is not Ops.CALL: return ()
  src = u.body.src[0]
  return src.src if src.op is Ops.AFTER else (src,)

def vdef(u:UOp) -> UOp|None:
  # a BITCAST or CAST that survives isel is no instruction, it reads the register of its source
  if u.op in {Ops.AFTER, Ops.BITCAST, Ops.CAST}: return vdef(u.src[0])
  operands = regs(u)
  return operands[0] if operands and operands[0].addrspace is AddrSpace.REG else None

def rdef(u:UOp) -> UOp|None: return physical_reg(p) if (p:=vdef(u)) is not None else None

def register_choices(u:UOp) -> dict[UOp, tuple[UOp, ...]]:
  operands = regs(u)
  return {p:(r,) if (r:=physical_reg(p)) is not None else operands[i+1].src
          for i,p in enumerate(operands) if p.op in (Ops.PARAM, Ops.BUFFER) and p.addrspace is AddrSpace.REG}

class LinearContext:
  def __init__(self, ren:'ISARenderer'): self.ren, self.stack_size = ren, 0
  def insert_linearized(self, uops:list[UOp]) -> list[UOp]: return uops
  def assign_spill_slot(self, r:UOp) -> int: raise NotImplementedError("arch specific")

class ISARenderer(Renderer):
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  pre_regalloc_matcher: PatternMatcher
  post_regalloc_matcher: PatternMatcher
  linear_ctx_type: type[LinearContext] = LinearContext

  def is_two_address(self, x:UOp) -> bool: return False
  def spill(self, spill_slot:int, x:UOp) -> UOp: raise NotImplementedError("arch specific")
  def fill(self, spill_slot:int, x:UOp, reg:UOp) -> UOp: raise NotImplementedError("arch specific")

# ***** X86 Ops *****

class X86Ops(FastEnum):
  # NOTE: X86Ops with i suffix are variants that take an immediate, m suffix are variants that can write to memory instead of read from
  LABEL = auto()
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
  CMOVNE = auto()
  VBLENDVPS = auto(); VBLENDVPD = auto()
  # jumps
  JNE = auto(); JGE = auto(); JMP = auto()
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
                X86Ops.IDIV, X86Ops.DIV, X86Ops.CMOVNE}

  # X86Ops whose second src is the rm field, so that src is what can be a memory operand
  Rm2nd = {X86Ops.ADD, X86Ops.SUB, X86Ops.AND, X86Ops.OR, X86Ops.XOR, X86Ops.IMUL, X86Ops.CMP,
           X86Ops.VADDSS, X86Ops.VADDSD, X86Ops.VSUBSS, X86Ops.VSUBSD, X86Ops.VMULSS, X86Ops.VMULSD, X86Ops.VDIVSS, X86Ops.VDIVSD,
           X86Ops.VBLENDVPS, X86Ops.VBLENDVPD, X86Ops.VCMPSS, X86Ops.VCMPSD, X86Ops.VROUNDSS, X86Ops.VROUNDSD, X86Ops.VSQRTSS, X86Ops.VSQRTSD,
           X86Ops.VINSERTPS, X86Ops.VPINSRW, X86Ops.VPINSRD, X86Ops.CMOVNE,
           X86Ops.VCVTSI2SS, X86Ops.VCVTSI2SD, X86Ops.VCVTSS2SD, X86Ops.VCVTSD2SS, X86Ops.IDIV, X86Ops.DIV}

  # X86Ops that can write to memory
  WriteMem = {X86Ops.MOVm, X86Ops.MOVi, X86Ops.VMOVSSm, X86Ops.VMOVSDm, X86Ops.VMOVUPSm, X86Ops.VMOVDm, X86Ops.VMOVQm,
              X86Ops.ADDi, X86Ops.SUBi, X86Ops.ANDi, X86Ops.ORi, X86Ops.XORi, X86Ops.SHL, X86Ops.SHLi, X86Ops.SHR, X86Ops.SHRi, X86Ops.SAR,
              X86Ops.SARi, X86Ops.SETNE, X86Ops.SETE, X86Ops.SETL, X86Ops.SETB,
              X86Ops.VCVTPS2PH, X86Ops.VPEXTRW, X86Ops.VPEXTRD}

  # X86Ops that read flags
  ReadFlags = {X86Ops.CMOVNE, X86Ops.SETB, X86Ops.SETL, X86Ops.SETE, X86Ops.SETNE, X86Ops.JNE, X86Ops.JGE}

  # X86Ops that jump, their first src is what the LABEL they jump to marks
  Jump = {X86Ops.JNE, X86Ops.JGE, X86Ops.JMP}

  # X86Ops whose first src is the rm field. a TwoAddress op drops its first src post regalloc, so its Rm2nd src ends up first
  Rm1st = {X86Ops.MOV, X86Ops.VMOVSS, X86Ops.VMOVSD, X86Ops.VMOVUPS, X86Ops.MOVZX, X86Ops.MOVSX, X86Ops.MOVSXD, X86Ops.VMOVD, X86Ops.VMOVQ,
           X86Ops.VCVTTSS2SI, X86Ops.VCVTTSD2SI, X86Ops.VCVTPH2PS, X86Ops.CMPi, X86Ops.IMULi, X86Ops.LEA, X86Ops.VPSRLDQ} | (Rm2nd & TwoAddress)

  # X86Ops whose last operand is an immediate
  Immediate = {X86Ops.MOVi, X86Ops.MOVABS, X86Ops.ADDi, X86Ops.SUBi, X86Ops.IMULi, X86Ops.ANDi, X86Ops.ORi, X86Ops.XORi, X86Ops.SHLi, X86Ops.SHRi,
               X86Ops.SARi, X86Ops.CMPi, X86Ops.VCMPSS, X86Ops.VCMPSD, X86Ops.VROUNDSS, X86Ops.VROUNDSD, X86Ops.VINSERTPS, X86Ops.VPSRLDQ,
               X86Ops.VPEXTRW, X86Ops.VPEXTRD, X86Ops.VPINSRW, X86Ops.VPINSRD, X86Ops.VCVTPS2PH}

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
  (UPat(Ops.NEG, name="x"), lambda x: x.const_like(-0.0).alu(Ops.SUB, *x.src)),
  # TODO: add support for mod, requires support for accessing the 2nd+ reg of a multi output instruction
  (UPat(Ops.TRUNCMOD, src=(UPat.var("x"), UPat.var("y"))), lambda x,y: x - y * x.alu(Ops.TRUNCDIV, y)),
  (UPat(Ops.INDEX, name="addr").load(UPat.var("alt"), UPat.var("gate")),
    lambda addr,alt,gate: gate.where(addr.load(), alt) if is_regbuf(addr.src[0]) else None),
  (UPat(Ops.INDEX, name="addr").store(UPat.var("val"), UPat.var("gate")),
    lambda addr,val,gate: addr.store(gate.where(val, addr.load())) if is_regbuf(addr.src[0]) else None),
])

# ***** X86 pre instruction selection *****
def scratch_buffer(elem_dt:DType, count:int, slot:int) -> UOp: return UOp.placeholder((count,), elem_dt, slot, AddrSpace.LOCAL)

def gated_load(ctx:itertools.count, addr:UOp, alt:UOp, gate:UOp, x:UOp):
  local = scratch_buffer(addr.src[0].dtype, x.max_numel(), next(ctx))
  local_idx = local.index(UOp.cconst(0, dtypes.int32))
  # the AFTER orders the load after the scratch store
  sel = gate.where(addr, local_idx)
  return sel.after((local_idx if x.max_numel() == 1 else local).store(alt)).load()

def gated_store(ctx:itertools.count, addr:UOp, gate:UOp, val:UOp):
  local = scratch_buffer(addr.src[0].dtype, val.max_numel(), next(ctx))
  sel = gate.where(addr, local.index(UOp.cconst(0, dtypes.int32)))
  return sel.store(val)

# legalize the new style graph for isel. NOTE: this runs after the spec is verified, some of these rewrites violate it
pre_isel_matcher = PatternMatcher([
  # vector BUFFERs get modeled in scratch space
  (UPat(Ops.BUFFER, name="x"),
   lambda ctx,x: scratch_buffer(x.dtype, n, next(ctx)) if x.addrspace is AddrSpace.REG and (n:=x.max_numel()) > 1 else None),
  # gated load/store become a conditional move on the address, the load/store are unconditional
  (UPat((Ops.INDEX, Ops.SHRINK), name="addr").load(UPat.var("alt"), UPat.var("gate"), name="x"), gated_load),
  (UPat((Ops.INDEX, Ops.SHRINK), name="addr").store(UPat.var("val"), UPat.var("gate")), gated_store),
])

# ***** X86 registers *****
GPR = tuple(UOp.placeholder((1,), dtypes.uint64, i, AddrSpace.REG, name=n) for i,n in
            enumerate(("rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", *(f"r{i}" for i in range(8, 16)))))
RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI = GPR[:8]
XMM = tuple(UOp.placeholder((1,), dtypes.uint64, i, AddrSpace.REG, name=f"xmm{i}") for i in range(16))
physical_regs = frozenset(GPR+XMM)
def physical_reg(u:UOp) -> UOp|None:
  # NOTE: this is not the architectural dtype, it's the widest dtype we use
  r = UOp.placeholder((1,), dtypes.uint64, u.arg.slot, AddrSpace.REG, name=u.arg.name)
  return r if r in physical_regs else None
# gprs you can write to
WGPR = tuple(r for r in GPR if r is not RSP)

CALLEE_SAVED = (RBX, RBP, GPR[12], GPR[13], GPR[14], GPR[15])

# ***** X86 instruction selection *****
# a register buffer is its own virtual register, identified by its buffer slot
def regbuf_reg(x:UOp) -> tuple[UOp, ...]:
  if physical_reg(x) is not None: return (x,)
  return vreg(x.arg.slot, x.dtype, XMM if x.dtype in dtypes.floats else WGPR, name="buffer")

# a value moved into a register constrained to cons
def pin(ctx:IselContext, x:UOp, cons:tuple[UOp, ...]) -> UOp: return UOp.ins(X86Ops.MOV, (x,), x.dtype, *ctx.vreg(x.dtype, cons))
def base(x:UOp, i:int) -> UOp: return s.src[0] if (s:=x.src[i]).op is Ops.INDEX else s
def lane(x:UOp, i:int) -> int: return s.src[1].src[0].val if (s:=x.src[i]).op is Ops.INDEX else 0
def to_int(dt:DType): return {dtypes.float16: dtypes.int16, dtypes.float32: dtypes.int32, dtypes.float64: dtypes.int64}[dt]
def is_imm(x:UOp) -> bool: return x.op is Ops.CAST and x.src[0].op is Ops.CONST and x.dtype not in dtypes.floats
def to_imm(c:UOp) -> UOp|None:
  if not (c.op is Ops.CAST and (v:=c.src[0]).op is Ops.CONST): return None
  if c.dtype in dtypes.int64s: return UOp.cconst(v.val, dtypes.i32) if not v.overflows(dtypes.i32) else None
  if c.dtype in dtypes.ints+(dtypes.bool,): return UOp.cconst(v.val, c.dtype)
  return None
# the flag path, which only an integer comparison can take: an x86 float compare sets carry, zero and parity together when an
# operand is NaN, so a NaN reads as "below" and as "equal", and it clears sign and overflow, so nothing reads as "less"
def cmp(x:UOp) -> UOp:
  if x.src[0].dtype in dtypes.floats: raise RuntimeError(f"no flag compare for {x.src[0].dtype}, a float gate must be a mask")
  return UOp.ins(X86Ops.CMP, x.src, dtypes.void) if (i:=to_imm(x.src[1])) is None else UOp.ins(X86Ops.CMPi, (x.src[0], i), dtypes.void)

# comparisons that produce masks, the mask has the width of the operands
def mask(x:UOp) -> UOp:
  dt, v = x.src[0].dtype, UOp.cconst({Ops.CMPLT: 1, Ops.CMPNE: 4, Ops.CMPEQ: 0}[x.op], dtypes.u8)
  return UOp.ins(X86Ops.VCMPSS if dt is dtypes.float32 else X86Ops.VCMPSD, x.src + (v,), dt)

def test_gate(m:UOp) -> UOp: return UOp.ins(X86Ops.CMPi, (m.cast(dtypes.int), UOp.cconst(0, dtypes.i32)), dtypes.void)

# vinsertps xmm2, xmm0, xmm1, imm
# inserts any 32 bit element in xmm1 into any position in xmm0 according to immm, result is written to xmm2
# this is the fallback slow case for when you can't match more a powerful shuffle
def vinsertps(x:UOp) -> UOp:
  def _insert(ret:UOp, i:int) -> UOp:
    s, v = base(x, i), lane(x, i)
    return UOp.ins(X86Ops.VINSERTPS, (ret, s, UOp.cconst(v << 6 | i << 4, dtypes.u8)), x.dtype)
  return functools.reduce(_insert, range(len(x.src)), UOp(Ops.NOOP))

# vpinsrd xmm2, xmm0, eax, imm
# inserts the element in eax into any position in xmm0, result is written to xmm2 according to imm
def vpins(x:UOp, srcs:tuple[UOp, ...]) -> UOp:
  op = {2: X86Ops.VPINSRW, 4: X86Ops.VPINSRD}[x.dtype.itemsize]
  return functools.reduce(lambda ret,i: UOp.ins(op, (ret, srcs[i], UOp.cconst(i, dtypes.u8)), x.dtype), range(len(srcs)), UOp(Ops.NOOP))

def idiv(ctx:IselContext, x:UOp) -> UOp:
  op = X86Ops.DIV if x.dtype in dtypes.uints else X86Ops.IDIV
  val = UOp.cconst(0, x.dtype) if x.dtype in dtypes.uints else x.src[0] >> UOp.cconst(x.dtype.itemsize*8-1, x.dtype)
  ext = [] if x.dtype in dtypes.int8s else [pin(ctx, val, (RDX,))]
  dividend_dtype = dtypes.int16 if x.dtype in dtypes.int8s else x.dtype
  dividend = pin(ctx, x.src[0].cast(dividend_dtype), (RAX,))
  defs = (RAX, RDX) if ext else (RAX,)
  divisor = pin(ctx, x.src[1], tuple(r for r in WGPR if r not in (RAX, RDX)))
  idiv = UOp.ins(op, (dividend, divisor) + tuple(ext), x.dtype, *defs)
  # this move "cleanses" the register constraints (rax/rdx) of idiv
  return UOp.ins(X86Ops.MOV, (idiv,), x.dtype)

# a variable shift count implicitly reads cl so it goes in rcx, the shifted value can't be in rcx
def shift(ctx:IselContext, x:UOp, op:X86Ops) -> UOp:
  val = pin(ctx, x.src[0], tuple(r for r in WGPR if r is not RCX))
  cnt = pin(ctx, x.src[1], (RCX,))
  return UOp.ins(op, (val, cnt), x.dtype)

def fold_address(x:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  def _disp(v:int) -> UOp: return UOp.cconst(v, dtypes.i8 if dtypes.i8.min <= v <= dtypes.i8.max else dtypes.i32)
  def _cast(v:UOp) -> UOp: return v.cast(dtypes.int64) if v.vmin < 0 else v.cast(dtypes.uint32) if v.dtype.itemsize < 4 else v
  if x.op not in {Ops.INDEX, Ops.SHRINK}: return (x, UOp.cconst(0, dtypes.u64), _disp(0), UOp.cconst(x.dtype.itemsize, dtypes.u32))
  base, idx = x.src[0], x.src[1]
  scale = 1 if base in (RSP, RDI) else base.dtype.itemsize
  sz = UOp.cconst(base.dtype.itemsize, dtypes.u32)
  if idx.op is Ops.ADD and is_imm(c:=idx.src[1]) and dtypes.i32.min <= (disp:=c.src[0].val * scale) <= dtypes.i32.max:
    return (base, _cast(idx.src[0]), _disp(disp), sz)
  if is_imm(idx):
    v = idx.src[0].val
    if dtypes.i32.min <= (disp:=v*scale) <= dtypes.i32.max: return (base, UOp.cconst(0, dtypes.u64), _disp(disp), sz)
    return (base, mov_op(UOp.cconst(v, dtypes.i64 if v < 0 else dtypes.u64)), _disp(0), sz)
  return (base, _cast(idx), _disp(0), sz)

def lea(x:UOp) -> UOp: return UOp.ins(X86Ops.LEA, fold_address(x), dtypes.uint64)

GPR_DEST_OPS = {X86Ops.VPEXTRW, X86Ops.VPEXTRD, X86Ops.VCVTTSS2SI, X86Ops.VCVTTSD2SI, X86Ops.VMOVDm, X86Ops.VMOVQm}
XMM_OPS = {op for op in X86Ops if op.name.startswith('V')} - GPR_DEST_OPS

def _is_vec_xmm(y: UOp) -> bool:
  return (y.op is Ops.CALL and y.body.arg.code in XMM_OPS) or (y.op not in (Ops.BUFFER, Ops.PARAM, Ops.REFPARAM, Ops.AFTER, Ops.CALL) \
      and y.max_numel() > 1)

def _xmm_sz(x: UOp) -> X86Ops:
  bits = x.max_numel() * x.dtype.itemsize
  if bits >= 16: return X86Ops.VMOVUPS
  if bits >= 8: return X86Ops.VMOVSD
  return X86Ops.VMOVSS

def alloc_vregs(ctx:IselContext, x:UOp) -> UOp|None:
  # no register definition
  if x.dtype is dtypes.void: return None
  if (dst:=vdef(x)) is not None:
    # already allocated vregs
    if physical_reg(dst) is None: return None
    defs = tuple(p for choices in register_choices(x).values() for p in ctx.vreg(x.dtype, choices))
  else: defs = ctx.vreg(x.dtype, XMM if x.dtype in dtypes.floats or x.body.arg.code in XMM_OPS else WGPR)
  return UOp.ins(x.body.arg.code, x.src_without_body, x.dtype, *defs)

def copy_op(dt:DType) -> X86Ops:
  return {dtypes.float16:X86Ops.VMOVSS, dtypes.float32:X86Ops.VMOVSS, dtypes.float64:X86Ops.VMOVSD, dtypes.bool:X86Ops.MOVZX}.get(dt, X86Ops.MOV)

def mov_op(s:UOp) -> UOp:
  if s.op is Ops.AFTER: return s.replace(src=(mov_op(s.src[0]),)+s.src[1:])
  return UOp.ins(X86Ops.MOVABS if s.dtype in dtypes.int64s else X86Ops.MOVi, (UOp.cconst(s.src[0].val, s.dtype),), s.dtype) if is_imm(s) else s

def load_const(x:UOp) -> UOp:
  op, n = x.body.arg.code, len(x.src_without_body)
  # keep the instruction's immediate operand as a constant
  immediate = {n-1} if op in X86GroupOp.Immediate else set()
  # memory addresses are (base, index, displacement, scale), keep their immediate operands constant
  immediate |= {1,2,3} if op in X86GroupOp.WriteMem and n>4 else {2,3,4} if op in X86GroupOp.Rm2nd and n>4 else \
      ({1,2,3} if op in X86GroupOp.Rm1st and n>3 else set())
  return x.replace(src=(x.body,)+tuple(s if i in immediate and is_imm(s) else mov_op(s) for i,s in enumerate(x.src_without_body)))

isel_matcher = PatternMatcher([
  (UPat().backedge(UPat(), UPat.var("m", dtypes.bool), name="x"), lambda x,m: UOp.ins(X86Ops.JNE, (x.src[1], x.src[0], test_gate(m)), x.dtype)),
  # function abi constraints
  (UPat(Ops.PARAM), lambda: RDI),
  (UPat(Ops.REFPARAM, name="x"), lambda x: x.src[1]),
  # conditional moves between addresses, lea both srcs
  (UPat.var("m").where(UPat((Ops.INDEX, Ops.SHRINK), name="a"), UPat((Ops.INDEX, Ops.SHRINK), name="b")), lambda m,a,b:
   m.where(lea(a), lea(b)) if not _is_vec_xmm(a.src[0]) else None),
  # pack float const as int
  (UPat.cvar("c").cast(dtypes.floats, name="x"), lambda c,x:
   UOp.cconst(struct.unpack((dt:=to_int(x.dtype)).fmt, struct.pack(x.dtype.fmt, c.val))[0], dt).bitcast(x.dtype)),
  # conditional moves that use masks, the mask has the width of the values
  (UPat(GroupOp.Comparison, src=(UPat(dtype=dtypes.float32), UPat()), name="m").where(UPat.var("a", dtypes.float32), UPat.var("b")), lambda m,a,b:
   UOp.ins(X86Ops.VBLENDVPS, (b, a, mask(m)), a.dtype)),
  (UPat(GroupOp.Comparison, src=(UPat(dtype=dtypes.float64), UPat()), name="m").where(UPat.var("a", dtypes.float64), UPat.var("b")), lambda m,a,b:
   UOp.ins(X86Ops.VBLENDVPD, (b, a, mask(m)), a.dtype)),
  # in this case we have a mask producing comparison whose user expects a bool, mask & 1 is 0 or 1 so the cast to bool is its low byte
  (UPat(GroupOp.Comparison, src=(UPat.var("y", (dtypes.float32, dtypes.float64)), UPat()), name="x"), lambda y,x:
   (mask(x).bitcast(dt:=to_int(y.dtype)) & UOp.cconst(1, dt)).cast(dtypes.bool)),
  # conditional moves use materialized boolean values
  (UPat.var("m", dtypes.bool).where(UPat.var("a"), UPat.var("b")), lambda m,a,b: UOp.ins(X86Ops.CMOVNE, (b, a, test_gate(m)), a.dtype)),
  # comparisons materialize their result in a register
  (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.uints), UPat()), name="x"), lambda x: UOp.ins(X86Ops.SETB, (cmp(x),), x.dtype)),
  (UPat(Ops.CMPLT, name="x"), lambda x: UOp.ins(X86Ops.SETL, (cmp(x),), x.dtype)),
  (UPat(Ops.CMPEQ, name="x"), lambda x: UOp.ins(X86Ops.SETE, (cmp(x),), x.dtype)),
  (UPat(Ops.CMPNE, name="x"), lambda x: UOp.ins(X86Ops.SETNE, (cmp(x),), x.dtype)),
  # float unary
  (UPat.var("y", dtypes.float32).sqrt(), lambda y: UOp.ins(X86Ops.VSQRTSS, (y, y), y.dtype)),
  (UPat.var("y", dtypes.float64).sqrt(), lambda y: UOp.ins(X86Ops.VSQRTSD, (y, y), y.dtype)),
  (UPat.var("y", dtypes.float32).trunc(), lambda y: UOp.ins(X86Ops.VROUNDSS, (y, y, UOp.cconst(3, dtypes.u8)), y.dtype)),
  (UPat.var("y", dtypes.float64).trunc(), lambda y: UOp.ins(X86Ops.VROUNDSD, (y, y, UOp.cconst(3, dtypes.u8)), y.dtype)),
  # for float16 we route the srcs through gprs, this is suboptimal for values in xmms, in that case we want vpunpcklwd
  (UPat(Ops.STACK, dtypes.float16, name="x"), lambda x: vpins(x, tuple(s.bitcast(dtypes.int16) for s in x.src))),
  (UPat(Ops.STACK, dtypes.float32, name="x"), vinsertps),
  (UPat(Ops.STACK, dtypes.int32s, name="x"), lambda x: vpins(x, x.src)),
  # INDEX on a vector register value extracts a single element
  (UPat.var("y", dtypes.int32s).index(UPat.cvar("c").cast(), name="x"),
   lambda y,c,x: UOp.ins(X86Ops.VPEXTRD, (y, UOp.cconst(c.val, dtypes.u8)), x.dtype) if _is_vec_xmm(y) else None),
  (UPat.var("y", dtypes.floats).index(UPat.cvar("c").cast(), name="x"),
   lambda y,c,x: UOp.ins(X86Ops.VPSRLDQ, (y, UOp.cconst(c.val * x.dtype.itemsize, dtypes.u8)), x.dtype) if _is_vec_xmm(y) else None),
  # int binary
  ((UPat(dtype=dtypes.ints).alu(Ops.TRUNCDIV, UPat())).named("x"), idiv),
  # int binary with immediate
  (UPat.var("a", dtypes.ints) << UPat.cvar("c").cast(), lambda a,c: UOp.ins(X86Ops.SHLi, (a, UOp.cconst(c.val & 0xff, dtypes.u8)), a.dtype)),
  (UPat.var("a", dtypes.uints) >> UPat.cvar("c").cast(), lambda a,c: UOp.ins(X86Ops.SHRi, (a, UOp.cconst(c.val & 0xff, dtypes.u8)), a.dtype)),
  (UPat.var("a", dtypes.sints) >> UPat.cvar("c").cast(), lambda a,c: UOp.ins(X86Ops.SARi, (a, UOp.cconst(c.val & 0xff, dtypes.u8)), a.dtype)),
  (UPat.var("a", dtypes.ints) + UPat.cvar().cast(name="c"),
   lambda a,c: UOp.ins(X86Ops.ADDi, (a, i), a.dtype) if (i:=to_imm(c)) is not None else None),
  (UPat.var("a", dtypes.ints) * UPat.cvar().cast(name="c"),
   lambda a,c: UOp.ins(X86Ops.IMULi, (a, i), a.dtype) if (i:=to_imm(c)) is not None else None),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) & UPat.cvar().cast(name="c"),
   lambda a,c: UOp.ins(X86Ops.ANDi, (a, i), a.dtype) if (i:=to_imm(c)) is not None else None),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) | UPat.cvar().cast(name="c"),
   lambda a,c: UOp.ins(X86Ops.ORi, (a, i), a.dtype) if (i:=to_imm(c)) is not None else None),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) ^ UPat.cvar().cast(name="c"),
   lambda a,c: UOp.ins(X86Ops.XORi, (a, i), a.dtype) if (i:=to_imm(c)) is not None else None),
  (UPat(Ops.SUB, dtypes.ints, (UPat.var("a"), UPat.cvar().cast(name="c"))),
   lambda a,c: UOp.ins(X86Ops.SUBi, (a, i), a.dtype) if (i:=to_imm(c)) is not None else None),
  # int binary with register
  ((UPat(dtype=dtypes.ints) << UPat()).named("x"), lambda ctx,x: shift(ctx, x, X86Ops.SHL)),
  ((UPat(dtype=dtypes.uints) >> UPat()).named("x"), lambda ctx,x: shift(ctx, x, X86Ops.SHR)),
  ((UPat(dtype=dtypes.sints) >> UPat()).named("x"), lambda ctx,x: shift(ctx, x, X86Ops.SAR)),
  (UPat.var("a", dtypes.ints) + UPat.var("b"), lambda a,b: UOp.ins(X86Ops.ADD, (a, b), a.dtype)),
  (UPat.var("a", dtypes.ints) * UPat.var("b"), lambda a,b: UOp.ins(X86Ops.IMUL, (a, b), a.dtype)),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) & UPat.var("b"), lambda a,b: UOp.ins(X86Ops.AND, (a, b), a.dtype)),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) | UPat.var("b"), lambda a,b: UOp.ins(X86Ops.OR, (a, b), a.dtype)),
  (UPat.var("a", dtypes.ints+(dtypes.bool,)) ^ UPat.var("b"), lambda a,b: UOp.ins(X86Ops.XOR, (a, b), a.dtype)),
  (UPat(Ops.SUB, dtypes.ints, (UPat.var("a"), UPat.var("b"))), lambda a,b: UOp.ins(X86Ops.SUB, (a, b), a.dtype)),
  # float binary
  ((UPat(dtype=dtypes.float32) + UPat()).named("x"), lambda x: UOp.ins(X86Ops.VADDSS, x.src, x.dtype)),
  ((UPat(dtype=dtypes.float64) + UPat()).named("x"), lambda x: UOp.ins(X86Ops.VADDSD, x.src, x.dtype)),
  ((UPat(dtype=dtypes.float32) * UPat()).named("x"), lambda x: UOp.ins(X86Ops.VMULSS, x.src, x.dtype)),
  ((UPat(dtype=dtypes.float64) * UPat()).named("x"), lambda x: UOp.ins(X86Ops.VMULSD, x.src, x.dtype)),
  (UPat(Ops.SUB, dtypes.float32, name="x"), lambda x: UOp.ins(X86Ops.VSUBSS, x.src, x.dtype)),
  (UPat(Ops.SUB, dtypes.float64, name="x"), lambda x: UOp.ins(X86Ops.VSUBSD, x.src, x.dtype)),
  (UPat(Ops.FDIV, dtypes.float32, name="x"), lambda x: UOp.ins(X86Ops.VDIVSS, x.src, x.dtype)),
  (UPat(Ops.FDIV, dtypes.float64, name="x"), lambda x: UOp.ins(X86Ops.VDIVSD, x.src, x.dtype)),
  # casts
  (UPat(dtype=dtypes.float32).cast(dtypes.float16, name="x"), lambda x: UOp.ins(X86Ops.VCVTPS2PH, x.src + (UOp.cconst(4, dtypes.u8),), x.dtype)),
  (UPat(dtype=dtypes.float16).cast(dtypes.float32, name="x"), lambda x: UOp.ins(X86Ops.VCVTPH2PS, x.src, x.dtype)),
  (UPat(dtype=dtypes.float32).cast(dtypes.int32s+dtypes.int64s, name="x"), lambda x: UOp.ins(X86Ops.VCVTTSS2SI, x.src, x.dtype)),
  (UPat(dtype=dtypes.float64).cast(dtypes.int32s+dtypes.int64s, name="x"), lambda x: UOp.ins(X86Ops.VCVTTSD2SI, x.src, x.dtype)),
  (UPat.var("y", dtypes.float32).cast(dtypes.float64, name="x"), lambda y,x: UOp.ins(X86Ops.VCVTSS2SD, (y, y), x.dtype)),
  (UPat.var("y", dtypes.float64).cast(dtypes.float32, name="x"), lambda y,x: UOp.ins(X86Ops.VCVTSD2SS, (y, y), x.dtype)),
  (UPat.var("y", (dtypes.int32, dtypes.int64)).cast(dtypes.float32, name="x"), lambda y,x: UOp.ins(X86Ops.VCVTSI2SS, (UOp(Ops.NOOP), y), x.dtype)),
  (UPat.var("y", (dtypes.int32, dtypes.int64)).cast(dtypes.float64, name="x"), lambda y,x: UOp.ins(X86Ops.VCVTSI2SD, (UOp(Ops.NOOP), y), x.dtype)),
  (UPat(dtype=(dtypes.uint8, dtypes.uint16, dtypes.bool)).cast(dtypes.ints, name="x"), lambda x:
   UOp.ins(X86Ops.MOVZX, x.src, x.dtype) if x.src[0].dtype.itemsize < x.dtype.itemsize else None),
  (UPat(dtype=dtypes.int32).cast(dtypes.int64s, name="x"), lambda x: UOp.ins(X86Ops.MOVSXD, x.src, x.dtype)),
  (UPat(dtype=dtypes.sints).cast(dtypes.ints, name="x"),
   lambda x: UOp.ins(X86Ops.MOVSX, x.src, x.dtype) if x.src[0].dtype.itemsize < x.dtype.itemsize else None),
  # narrowing is a real move. any 32-bit write zeroes the upper half, so widening a uint32 is free, like a same width cast
  (UPat(dtype=dtypes.ints).cast(dtypes.ints, name="x"),
   lambda x: UOp.ins(X86Ops.MOV, x.src, x.dtype) if x.src[0].dtype.itemsize > x.dtype.itemsize else None),
  # bitcasts between scalar floats and ints
  (UPat.var("y", dtypes.float16).bitcast(dtypes.int16s).named("x"), lambda y,x: UOp.ins(X86Ops.VPEXTRW, (y, UOp.cconst(0, dtypes.u8)), x.dtype)),
  (UPat(dtype=dtypes.int16s).bitcast(dtypes.float16).named("x"), lambda x: vpins(x, x.src)),
  (UPat(dtype=dtypes.int32s).bitcast(dtypes.float32).named("x"), lambda x: UOp.ins(X86Ops.VMOVD, x.src, x.dtype)),
  (UPat(dtype=dtypes.int64s).bitcast(dtypes.float64).named("x"), lambda x: UOp.ins(X86Ops.VMOVQ, x.src, x.dtype)),
  (UPat(dtype=dtypes.float32).bitcast(dtypes.int32s).named("x"), lambda x: UOp.ins(X86Ops.VMOVDm, x.src, x.dtype)),
  (UPat(dtype=dtypes.float64).bitcast(dtypes.int64s).named("x"), lambda x: UOp.ins(X86Ops.VMOVQm, x.src, x.dtype)),
  (UPat.var("a").store(UPat.var("val")), lambda a,val:
    UOp(Ops.STACK).after(UOp.ins(copy_op(val.dtype), (val.after(buf),), buf.dtype, *regbuf_reg(buf.without_after)))
    if is_regbuf((buf := a.src[0] if a.op is Ops.INDEX else a)) else None),
  (UPat.var("a").load().named("x"), lambda a,x:
    UOp.ins(copy_op(x.dtype), (buf,), x.dtype) if is_regbuf((buf:=a.src[0] if a.op is Ops.INDEX else a)) else None),
  # index on a buffer (or the stack pointer) computes an address, addresses are 64bit values
  (UPat((Ops.INDEX, Ops.SHRINK), name="x"), lambda x: lea(x) if not _is_vec_xmm(x.src[0]) and x.addrspace is not AddrSpace.REG else None),
  # TODO: fuse stores, very few cases -- store cmp becomes setcc, store gep int becomes vpextr, store bitcast to int becomes vmovd/q
  # load, store
  (UPat(Ops.LOAD, dtypes.floats, src=(UPat(name="a"),), name="x"), lambda x,a: None if a.addrspace is AddrSpace.REG else
   UOp.ins(X86Ops.VPINSRW, (UOp(Ops.NOOP),) + fold_address(a) + (UOp.cconst(0, dtypes.u8),), x.dtype) if x.max_numel() * x.dtype.itemsize == 2 else
   UOp.ins(_xmm_sz(x), fold_address(a), x.dtype)),
  (UPat(Ops.LOAD, dtypes.ints+(dtypes.bool,), src=(UPat(name="a"),), name="x"), lambda x,a: None if a.addrspace is AddrSpace.REG else
   UOp.ins(X86Ops.MOV, fold_address(a), x.dtype) if x.max_numel() == 1 else UOp.ins(_xmm_sz(x), fold_address(a), x.dtype)),
  (UPat.var("a").store(UPat.var("b", dtypes.floats), name="x"), lambda a,b,x: None if a.addrspace is AddrSpace.REG else
   UOp.ins(X86Ops.VPEXTRW, fold_address(a) + (b, UOp.cconst(0, dtypes.u8)), x.dtype) if b.max_numel() * b.dtype.itemsize == 2 else
   UOp.ins(X86Ops[_xmm_sz(b).name+"m"], fold_address(a) + (b,), x.dtype)),
  (UPat.var("a").store(UPat.var("b", dtypes.ints+(dtypes.bool,)), name="x"), lambda a,b,x: None if a.addrspace is AddrSpace.REG else
   UOp.ins(X86Ops[_xmm_sz(b).name+"m"], fold_address(a) + (b,), x.dtype) if b.max_numel() > 1 else
   UOp.ins(X86Ops.MOVm, fold_address(a) + (b,), x.dtype) if (i:=to_imm(b)) is None else UOp.ins(X86Ops.MOVi, fold_address(a) + (i,), x.dtype)),
  # move const to registers for instructions that don't support immediate operands
  (UPat(Ops.CALL, name="x"), load_const),
  # allocate virtual registers
  (UPat(Ops.CALL, name="x"), alloc_vregs),
])

# ***** pre register allocation *****
class X86LinearContext(LinearContext):
  # stack alloc/dealloc and callee saved handling
  def insert_linearized(self, uops:list[UOp]) -> list[UOp]:
    before, after, used = [], [], {rdef(u) for u in uops}
    for r in CALLEE_SAVED:
      if r in used:
        slot = self.assign_spill_slot(r)
        before.append(self.ren.spill(slot,r))
        after.append(self.ren.fill(slot,r,r))

    if self.stack_size > 0:
      before.insert(0, UOp.ins(X86Ops.SUBi, (RSP, UOp.cconst(self.stack_size, dtypes.i32)), dtypes.u64, RSP))
      after.append(UOp.ins(X86Ops.ADDi, (RSP, UOp.cconst(self.stack_size, dtypes.i32)), dtypes.u64, RSP))
    return before + uops[:-1] + after + [uops[-1]]

  def assign_spill_slot(self, r:UOp) -> int:
    sz = 16 if r in XMM else 8
    offset = self.stack_size + (sz - self.stack_size % sz) %sz
    self.stack_size = offset + sz
    return offset

# the element size of loads and stores is already captured by fold_address before stack buffers become 64-bit pointers
def alloc_buffer(ctx:X86LinearContext, x:UOp):
  dst = vreg(ctx.stack_size, dtypes.uint64, WGPR, name="stack")
  nx = UOp.ins(X86Ops.LEA, fold_address(RSP.index(UOp.cconst(ctx.stack_size, dtypes.uint32))), dtypes.uint64, *dst)
  ctx.stack_size += x.max_numel() * x.dtype.itemsize
  return nx, [RSP, nx]

# TODO: control flow should be overhauled so that this isn't necessary
# a LABEL marks its src: the start of a loop its RANGE, the exit of a loop the start LABEL. a jump's first src is what its LABEL marks
def label(x:UOp) -> UOp: return UOp.ins(X86Ops.LABEL, (x,), dtypes.void)

# a loop's LABEL follows its RANGE. a bounded RANGE's index is a register named by the RANGE, 0 before the loop, compared to the bound
# after the LABEL and incremented before the END, the body reads it as AFTER(index, RANGE)
def lower_range(x:UOp) -> tuple[UOp, list[UOp]]:
  if x.dtype is dtypes.void: return x, [x, label(x)]
  bound = x.src[0].without_after
  idx = UOp.ins(X86Ops.MOVi, (UOp.cconst(0, x.dtype),), x.dtype, *vreg(0, x.dtype, WGPR, name=x.key.hex()))
  cmp = UOp.ins(X86Ops.CMPi if is_imm(bound) else X86Ops.CMP, (idx, bound), dtypes.void)
  return idx.after(x), [idx, x, start:=label(x), cmp, UOp.ins(X86Ops.JGE, (start, cmp), dtypes.void)]

def lower_end(x:UOp, c:UOp) -> tuple[UOp, list[UOp]]:
  inc = UOp.ins(X86Ops.ADDi, (c, UOp.cconst(1, c.dtype)), c.dtype, *regs(c.src[0]))
  return (end:=x.replace(src=(x.src[0], c.src[1]))), [inc, end]

pre_regalloc_matcher = PatternMatcher([
  (UPat(Ops.BUFFER, name="x"), lambda ctx,x: alloc_buffer(ctx,x) if x.addrspace is not AddrSpace.REG else None),
  (UPat(Ops.CALL, name="x"), lambda x: (x, []) if x.body.arg.code in {X86Ops.CMP, X86Ops.CMPi} else
   (x, [x.src[-1], x]) if x.body.arg.code in X86GroupOp.ReadFlags else None),
  (UPat(Ops.RANGE, name="x"), lower_range),
  (UPat(Ops.END, src=(UPat(), UPat(Ops.AFTER, name="c")), name="x"), lower_end),
])

# ***** post register allocation *****
# final rewrite to match the isa spec
post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x: (x, [UOp.ins(X86Ops.RET, x.src, x.dtype)])),
  # an END jumps back to its loop's LABEL and labels the loop's exit
  (UPat(Ops.END, name="x"), lambda x: (x, [UOp.ins(X86Ops.JMP, (x.src[1],), dtypes.void), label(label(x.src[1]))])),
  # rewrite two address instructions to two address form, if reused src wasn't coalesced insert a move
  (UPat(Ops.CALL, name="x"), lambda ctx,x: (nx:=x.replace(src=(x.body,)+x.src[2:]),
   [ctx.ren.copy(x.src[1], rdef(x)), nx] if rdef(x) is not rdef(x.src[1]) else [nx]) if x.body.arg.code in X86GroupOp.TwoAddress else None),
])

# ***** X86 instruction encoding *****

def encode(x:UOp, opc:int, reg:int|None=None, pp:int=0, sel:int=0, we:int=0) -> bytes|None:
  def _encode(reg_uop:UOp|None, rm_uop:UOp, idx_uop:UOp|None=None, disp_uop:UOp|None=None, scale_uop:UOp|None=None,
              vvvv_uop:UOp|None=None, imm_uop:UOp|None=None) -> bytes:
    nonlocal reg, opc
    # get the encoding values of the different fields
    reg = unwrap(rdef(reg_uop)).arg.slot if reg_uop is not None else unwrap(reg)
    rm = unwrap(rdef(rm_uop)).arg.slot
    idx = unwrap(rdef(idx_uop)).arg.slot if idx_uop is not None and rdef(idx_uop) is not None else 4
    # for a memory operand the rm size is the element size carried by the address, otherwise it's the size of the value in the register
    rm_sz = scale_uop.src[0].val if scale_uop is not None else rm_uop.dtype.itemsize
    reg_sz = reg_uop.dtype.itemsize if reg_uop is not None else 0
    sz = reg_sz or rm_sz

    # encode instruction
    inst = bytes([])
    assert 0 <= reg <= 15 and 0 <= idx <= 15 and 0 <= rm <= 15
    # r extends reg field, x extends index field, b extends rm or base field
    r, _x, b = reg >> 3, idx >> 3, rm >> 3
    if sel: # VEX bytes
      vvvv = (vd.arg.slot if (vd:=rdef(vvvv_uop)) is not None else reg) if vvvv_uop is not None else 0
      if sel == 1 and _x == b == we == 0: inst += bytes([0xC5, (~r & 0b1) << 7 | (~vvvv & 0b1111) << 3 | pp])
      else: inst += bytes([0xC4, (~r & 0b1) << 7 | (~_x & 0b1) << 6 | (~b & 0b1) << 5 | sel, we << 7 | (~vvvv & 0b1111) << 3 | pp])
    else: # optional PREFIX and REX bytes
      # PREFIX byte signaling 16 bit variant of instruction
      if sz == 2: inst += bytes([0x66])
      # bit signaling 64 bit variant of instruction
      w = sz == 8
      # legacy 8bit opcode is 1 less than 16-64bit variants
      demote = (rm_sz == 1 or reg_sz == 1) and x.body.arg.code not in X86GroupOp.ReadFlags | {X86Ops.LEA}
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
      elif (r:=rdef(imm_uop)) is not None: inst += bytes([(r.arg.slot & 0b1111) << 4 | 0b0000])
    return inst

  # get the encoding structure of the uop
  # when a uop writes to memory it takes the form of a store, dtype is void, no definition
  address:tuple[UOp|None, ...]
  src = x.src_without_body
  if x.body.arg.code in X86GroupOp.WriteMem:
    if len(src) > 4: address, rest = src[:4], src[4:]
    else: address, rest = (x, None, None, None), src
    imm_uop = rest[:1] if rest and is_imm(rest[0]) else (None,)
    return _encode(rest[0], *address, *(None, *rest[1:])) if reg is None else _encode(None, *address, *(None, *imm_uop))

  if x.body.arg.code in X86GroupOp.Rm1st:
    if len(src) > 3: address, rest = src[:4], src[4:]
    else: address, rest = (src[0], None, None, None), src[1:]
    imm_uop = rest[:1] if rest and is_imm(rest[0]) else (None,)
    return _encode(x, *address, *(None, *imm_uop)) if reg is None else _encode(None, *address, *(x if sel else None, *imm_uop))

  if x.body.arg.code in X86GroupOp.Rm2nd:
    if len(src) > 4: address, rest = src[1:5], src[:1] + src[5:]
    else: address, rest = (src[1], None, None, None), src[:1] + src[2:]
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
   bytes([0x48 | (slot:=unwrap(rdef(x)).arg.slot) >> 3, 0xB8 + (slot & 7)]) + struct.pack(x.dtype.fmt, x.src[1].src[0].val),
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
  X86Ops.VCVTSI2SS: lambda x: encode(x, 0x2A, pp=2, sel=1, we=x.src[2].dtype.itemsize == 8),
  X86Ops.VCVTSI2SD: lambda x: encode(x, 0x2A, pp=3, sel=1, we=x.src[2].dtype.itemsize == 8),
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
  X86Ops.CMOVNE: lambda x: encode(x, 0x0F45),
  X86Ops.VBLENDVPS: lambda x: encode(x, 0x4A, pp=1, sel=3), X86Ops.VBLENDVPD: lambda x: encode(x, 0x4B, pp=1, sel=3),
  # shuffles
  X86Ops.VPSRLDQ: lambda x: encode(x, 0x73, reg=3, pp=1, sel=1),
  X86Ops.VPINSRW: lambda x: encode(x, 0xC4, pp=1, sel=1), X86Ops.VPINSRD: lambda x: encode(x, 0x22, pp=1, sel=3),
  X86Ops.VINSERTPS: lambda x: encode(x, 0x21, pp=1, sel=3),
  # extract
  X86Ops.VPEXTRW: lambda x: encode(x, 0x15, pp=1, sel=3), X86Ops.VPEXTRD: lambda x: encode(x, 0x16, pp=1, sel=3),
  # jumps are encoded with a placeholder which gets patched later once the real offset is known
  X86Ops.JNE: lambda x: bytes([0x0F, 0x85]) + int(0).to_bytes(4),
  X86Ops.JGE: lambda x: bytes([0x0F, 0x8D]) + int(0).to_bytes(4),
  X86Ops.JMP: lambda x: bytes([0xE9]) + int(0).to_bytes(4),
  X86Ops.RET: lambda x: bytes([0xC3]),
}

class X86Renderer(ISARenderer):
  device = "CPU"
  packed_args = True
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
  def is_two_address(self, x:UOp) -> bool: return x.op is Ops.CALL and x.body.arg.code in X86GroupOp.TwoAddress
  def copy(self, x:UOp, reg:UOp) -> UOp: return UOp.ins(X86Ops.MOV, (x,), x.dtype, reg)

  def spill(self, spill_slot:int, x:UOp) -> UOp:
    op = X86Ops.VMOVUPSm if rdef(x) in XMM else X86Ops.MOVm
    disp = UOp.cconst(spill_slot, dtypes.int32)
    return UOp.ins(op, fold_address(RSP.index(disp)) + (x,), dtypes.void)

  def fill(self, spill_slot:int, x:UOp, reg:UOp) -> UOp:
    disp = UOp.cconst(spill_slot, dtypes.int32)
    return UOp.ins(X86Ops.VMOVUPS if reg in XMM else X86Ops.MOV, fold_address(RSP.index(disp)), x.dtype, reg)

  def render(self, uops:list[UOp]) -> str:
    targets: dict[UOp, int] = {}
    jumps: dict[UOp, int] = {}
    binary = bytearray()
    for u in uops:
      if u.op is not Ops.CALL: continue
      if u.body.arg.code is X86Ops.LABEL:
        targets[u.src[1]] = len(binary)
        continue
      if u.body.arg.code not in encodings or (l:=encodings[u.body.arg.code](u)) is None:
        raise RuntimeError(f"failed to encode {u.body.arg.code} with {u.dtype} srcs {[x.dtype for x in u.src_without_body]}")
      binary.extend(l)
      if u.body.arg.code in X86GroupOp.Jump: jumps[u] = len(binary)
    # fixup jump targets now that encoding size is known
    for u in uops:
      if (t:=jumps.get(u)) is not None: binary[t-4:t] = (targets[u.src[1]] - t).to_bytes(4, 'little', signed=True)
    return binary.hex()

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s+(dtypes.bfloat16,)}
