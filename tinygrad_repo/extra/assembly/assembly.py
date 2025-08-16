from typing import Tuple, List, NamedTuple, Any, Dict, Optional, Union, DefaultDict, cast
from tinygrad.codegen.opt.kernel import Ops, MemOp, UOp
from tinygrad.uop.ops import BinaryOps, UnaryOps
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import DEBUG
from tinygrad.uop.ops import Variable, NumNode, MulNode, DivNode, ModNode, LtNode, SumNode, AndNode
import functools
import math
from collections import defaultdict

_type_to_letter = {dtypes.float32: 'f', dtypes.bool: 'p', dtypes.int32: 'i', dtypes.int64: 'a', dtypes.uint32: 'u', dtypes.uint64: 'b', dtypes.float.vec(4): 'x', dtypes.uint8: 'uc', dtypes.float16: 'h',
                   dtypes.int8: 'c', dtypes.uint16: 'us', dtypes.float64: 'd'}

class Register(NamedTuple):
  nm:str
  dtype:DType
  scalar:bool
  off:Optional[int] = None
  def __repr__(self): return self.nm if self.off is None else f"{self.nm}:{self.off}"
  def subregs(self):
    if self.dtype == dtypes.float.vec(4):
      return [Register(self.nm, dtypes.float, False, off=off) for off in range(4)]
    return []

class AssemblyInstruction(NamedTuple):
  op: Ops
  out: Optional[Register]
  vin: List[Union[Register, int, float]]
  arg: Any = None

# warp size of 32, s registers are shared across the warp, v are 32-wide vectors
class AssemblyLanguage:
  supports_load3: bool = False
  sin_is_sin2pi: bool = False
  no_div: bool = False
  #TODO: these should be global vars
  cnts:DefaultDict[Tuple[DType, bool], int] = defaultdict(int)
  tor: Dict[Any, Register] = {}
  ins: List[AssemblyInstruction] = []

  def type_to_letter(self,x): return _type_to_letter[x[0]].upper() if x[1] else _type_to_letter[x[0]]
  def newreg(self, tok, dtype=dtypes.float32, scalar=False) -> Register:
    self.tor[tok] = ret = Register(f"%{self.type_to_letter((dtype, scalar))}{self.cnts[(dtype, scalar)]}", dtype, scalar)
    if dtype == dtypes.float.vec(4):
      for off in range(4):
        self.tor[tok] = Register(ret.nm, dtypes.float, ret.scalar, off)
    self.cnts[(dtype, scalar)] += 1
    return ret

  def render_numnode(self, b) -> Register:
    key = ("num", b)
    if key not in self.tor: self.ins.append(AssemblyInstruction(Ops.LOAD, self.newreg(key, scalar=True, dtype=dtypes.int32), [], b))
    return self.tor[key]

  def render_alu(self, op, a:Register, b:Union[Register, int, float], dtype=dtypes.int32) -> Register:
    key = (op, a, b)
    if key not in self.tor:
      #if not isinstance(b, Register): b = render_numnode(b)
      self.ins.append(AssemblyInstruction(Ops.ALU, self.newreg(key, dtype=dtype, scalar=a.scalar and (not isinstance(b, Register) or b.scalar)), [a, b], op))
    return self.tor[key]

  def render_cast(self, a:Register, new_dtype:DType) -> Register:
    if a.dtype == new_dtype: return a
    key = (a, new_dtype)
    if key not in self.tor:
      self.ins.append(AssemblyInstruction(Ops.CAST, self.newreg(key, dtype=new_dtype), [a]))
    return self.tor[key]

  render_ops: Any = { Variable: lambda self, ops, ctx: ctx.tor[self], NumNode: lambda self, ops, ctx: ctx.render_numnode(self.b),
                 MulNode: lambda self, ops, ctx: ctx.render_alu(BinaryOps.MUL, self.a.render(ops, ctx), self.b),
                 DivNode: lambda self, ops, ctx: ctx.render_alu(BinaryOps.DIV, self.a.render(ops, ctx), self.b),
                 ModNode: lambda self, ops, ctx: ctx.render_alu(BinaryOps.MOD, self.a.render(ops, ctx), self.b),
                 LtNode: lambda self, ops, ctx: ctx.render_alu(BinaryOps.CMPLT, self.a.render(ops, ctx), self.b, dtype=dtypes.bool),
    SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: ctx.render_alu(BinaryOps.ADD, a, b.render(ops,ctx)), self.nodes[1:], self.nodes[0].render(ops,ctx)),
    AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: ctx.render_alu(BinaryOps.MUL, a, b.render(ops,ctx), dtype=dtypes.bool), self.nodes[1:], self.nodes[0].render(ops,ctx)) }

  def addr_w_offset(self, args):
    assert isinstance(args, MemOp)
    idx = args.idx*args.memory_dtype.itemsize
    off = 0  # TODO: should this be None?
    if isinstance(idx, SumNode):
      nums = [n.b for n in idx.nodes if isinstance(n, NumNode)]
      if nums and nums[0] < 4096 and (idx-nums[0]).min >= 0:  # TODO: different for each GPU?
        idx -= nums[0]
        off = cast(int, nums[0])
    reg = idx.render(self.render_ops, self)
    if self.supports_load3:
      if reg.scalar:
        new_reg = self.newreg((reg.nm, 'vec'), dtype=reg.dtype)
        self.ins.append(AssemblyInstruction(Ops.ALU, new_reg, [reg], UnaryOps.NOOP))
        reg = new_reg
      return self.tor[args.name], reg, off
    reg = self.render_alu(BinaryOps.ADD, self.render_cast(reg, dtypes.uint64), self.tor[args.name], dtype=dtypes.uint64)
    return reg, None, off

def uops_to_asmstyle(lang, function_name:str, uops:List[UOp]):
  #TODO: Do not use clear()
  lang.ins.clear()
  lang.tor.clear()
  lang.cnts.clear()
  buf_to_dtype = {args:dtype for uop,dtype,_,args,_ in uops if uop == Ops.DEFINE_GLOBAL}
  global_size, local_size = [], []
  skipload_branch = 0
  lang.ins += [AssemblyInstruction(Ops.SPECIAL, lang.newreg(buf, dtype=dtypes.uint64, scalar=True), [], buf) for buf in buf_to_dtype]
  for u in uops:
    uop,dtype,vin,args,_ = u
    if uop == Ops.DEFINE_LOCAL:
      lang.ins.append(AssemblyInstruction(Ops.DEFINE_LOCAL, None, [], args))
      lang.ins.append(AssemblyInstruction(Ops.ALU, lang.newreg(args[0], dtype=dtypes.uint64), [args[0]], UnaryOps.NOOP))
    elif uop == Ops.LOOP:
      if args[1] == "global":
        for i,var in enumerate(args[0]):
          global_size.append(var.max+1)
          lang.ins.append(AssemblyInstruction(Ops.SPECIAL, lang.newreg(var, dtype=dtypes.int32), [], f"gid{len(args[0])-1-i}"))
      elif args[1] == "local":
        for i,var in enumerate(args[0]):
          local_size.append(var.max+1)
          lang.ins.append(AssemblyInstruction(Ops.SPECIAL, lang.newreg(var, dtype=dtypes.int32), [], f"lid{len(args[0])-1-i}"))
      else:
        for var in args[0]:
          if not isinstance(var, NumNode):  # TODO: why is this coming through?
            lang.ins.append(AssemblyInstruction(Ops.LOAD, lang.newreg(var, dtype=dtypes.int32, scalar=True), [], 0))
            lang.ins.append(AssemblyInstruction(Ops.LABEL, None, [], "$loop_"+var.expr))
    elif uop == Ops.ENDLOOP:
      if args[1] not in ["global", "local", "global+local"]:
        for var in reversed(args[0]):
          if not isinstance(var, NumNode):  # TODO: why is this coming through?
            lang.ins.append(AssemblyInstruction(Ops.ALU, lang.tor[var], [lang.tor[var], 1], BinaryOps.ADD))
            pred = lang.render_alu(BinaryOps.CMPLT, lang.tor[var], var.max+1, dtypes.bool)
            lang.ins.append(AssemblyInstruction(Ops.COND_BRANCH, None, [pred], ("$loop_"+var.expr, True)))
      elif args[1] == "global+local":
        for i, var in enumerate(reversed(args[0])):
          lang.ins.append(AssemblyInstruction(Ops.ENDLOOP, None, [lang.tor[var]], (var.max+1, f"gid{i}")))
      elif args[1] == 'local':
        for i, var in enumerate(reversed(args[0])):
          lang.ins.append(AssemblyInstruction(Ops.ENDLOOP, None, [lang.tor[var]], (var.max+1, f"lid{i}")))
    elif uop == Ops.CAST:
      # TODO: we should reconsider outputting CAST in the linearizer. these are needless copies
      out = lang.newreg(u, dtype)
      for i,sr in enumerate(out.subregs()):
        lang.ins.append(AssemblyInstruction(Ops.ALU, sr, [lang.tor[vin[i]]], UnaryOps.NOOP))
    elif uop == Ops.ALU:
      out = lang.newreg(u, dtype) if u not in lang.tor else lang.tor[u]
      # this is the only thing that can violate SSA
      if args in [BinaryOps.CMPLT]:
        pred_reg = lang.newreg((u, 'pred'), dtype=dtypes.bool)
        lang.ins.append(AssemblyInstruction(Ops.ALU, pred_reg, [lang.tor[x] for x in vin], args))
        lang.ins.append(AssemblyInstruction(Ops.CAST, out, [pred_reg], args))
      elif args == BinaryOps.DIV and lang.no_div:
        tmp = lang.newreg((u, "rcp"))
        lang.ins.append(AssemblyInstruction(Ops.ALU, tmp, [lang.tor[vin[1]]], UnaryOps.RECIP))
        lang.ins.append(AssemblyInstruction(Ops.ALU, out, [lang.tor[vin[0]], tmp], BinaryOps.MUL))
      elif args == UnaryOps.SIN and lang.sin_is_sin2pi:
        tmp = lang.newreg((u, "2pi"))
        lang.ins.append(AssemblyInstruction(Ops.ALU, tmp, [lang.tor[vin[0]], 1/(math.pi*2)], BinaryOps.MUL))
        lang.ins.append(AssemblyInstruction(Ops.ALU, out, [tmp], args))
      else:
        lang.ins.append(AssemblyInstruction(Ops.ALU, out, [lang.tor[x] for x in vin], args))
    elif uop == Ops.DEFINE_REG:
      reg = lang.newreg(u, dtype=dtype)
      lang.ins.append(AssemblyInstruction(Ops.LOAD, reg, [], args))
    elif uop == Ops.SPECIAL:
      lang.tor[u] = lang.tor[args]
    elif uop == Ops.CONST:
      lang.ins.append(AssemblyInstruction(Ops.LOAD, lang.newreg(u, dtype=dtype), [], args))
    elif uop == Ops.LOAD:
      idx, treg, off = lang.addr_w_offset(args)
      reg = lang.newreg(u, dtype=dtype, scalar=(idx.scalar and (not isinstance(treg, Register) or treg.scalar)))
      if args.valid.min == 0:
        lang.ins.append(AssemblyInstruction(Ops.LOAD, reg, [], 0))
        if args.valid.max == 1:
          pred = args.valid.render(lang.render_ops, lang)
          lang.ins.append(AssemblyInstruction(Ops.COND_BRANCH, None, [pred], (f"$skipload_{skipload_branch}", False)))
      if args.valid.max == 1:
          # NOTE: you can't compute the index in here, because it assumes it's all available later
        lang.ins.append(AssemblyInstruction(Ops.LOAD, reg, [idx] + ([treg] if treg is not None else []), (off, 'global' if not args.local else 'shared', args.memory_dtype if args.memory_dtype != dtypes.float else None)))
      if args.valid.min == 0 and args.valid.max == 1:
        lang.ins.append(AssemblyInstruction(Ops.LABEL, None, [], f"$skipload_{skipload_branch}"))
        skipload_branch += 1
    elif uop == Ops.STORE:
      if args is None:
        lang.ins.append(AssemblyInstruction(Ops.ALU, lang.tor[vin[0]], [lang.tor[vin[1]]], UnaryOps.NOOP))
      else:
        idx, treg, off = lang.addr_w_offset(args)
        lang.ins.append(AssemblyInstruction(Ops.STORE, None, [idx, lang.tor[vin[0]]] + ([treg] if treg is not None else []), (off, 'global' if not args.local else 'shared', args.memory_dtype if args.memory_dtype != dtypes.float else None)))

  if DEBUG >= 4:
    for tins in lang.ins: print(tins)
  return global_size, local_size
