import struct
from platform import system
from typing import Tuple, Dict, List, Optional
from tinygrad import dtypes
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps
from tinygrad.codegen.kernel import Ops, UOp
from tinygrad.helpers import CI
from tinygrad.codegen.assembly import uops_to_asmstyle, AssemblyLanguage

def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
def compute_offsets(total):
  quotient, remainder = divmod(total, 4096)
  return [4096]*quotient + [remainder] if remainder else [4096]*quotient

#NOTE: Darwin needs names to start with a "_"
def get_name(name): return ('_' if system() == 'Darwin' else '') + name

class ARM64Language(AssemblyLanguage): pass

def specialize_to_arm64(fn_nm, asm):
  var_size = 16
  prev_uop:Optional[Ops] = None
  ins = []
  x_regs = ['x' + str(i) for i in reversed(range(12))]
  s_regs = ['s' + str(i) for i in reversed(range(3,32)) if i <= 7 or i >= 16]
  type_to_reg = {dtypes.double: "d", dtypes.half: 'h', dtypes.float32: 's', dtypes.bool: 'w', dtypes.int8:'w', dtypes.int32: 'w', dtypes.int64: 'x', dtypes.uint8:'w', dtypes.uint32: 'w', dtypes.uint64: 'x'}
  alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "div", BinaryOps.MAX: "max",
          BinaryOps.MOD: "", BinaryOps.CMPLT: "subs",
          UnaryOps.NOOP: "mov", UnaryOps.NEG: "neg",
          UnaryOps.SIN:'bl ' + get_name('sinf'), UnaryOps.LOG2: 'bl ' + get_name("log2f"), UnaryOps.EXP2: 'bl ' + get_name("exp2f"), UnaryOps.SQRT: 'bl ' + get_name("sqrtf"),
          TernaryOps.MULACC: "madd", TernaryOps.WHERE: "fcsel"}

  def mov_imm(value, reg):
    # Manually move value into reg if value can't fit
    if value.__class__ is not float and abs(value) > abs(65535):
      ins.append(f"movz w15, #{value & 0xffff}")
      ins.append(f"movk w15, #{(value >> 16) & 0xffff}, lsl #16")
      ins.append(f"sxtw {reg}, w15")
    elif reg[0] == 's':
      ins.append(f"movz x15, 0x{float_to_hex(value)[4:]}")
      ins.append(f"movk x15, 0x{float_to_hex(value)[:4]}, lsl #16")
      ins.append("str x15, [sp, 16]")
      ins.append(f"ldr {reg}, [sp, 16]")
    else:
      ins.append(f"mov {reg}, #{value}")

  # Get variables intervals
  live_range:Dict[str, List[int]] = {}
  for i, (uop, out, vin, arg) in enumerate(asm):
    for var in ([v for v in [out] + vin if v is not None and v.__class__ is not int]):
      live_range[var.nm] = [i,i] if var.nm not in live_range else [live_range[var.nm][0], i]

  mem_vars:Dict[str, int] = {}
  rtor:Dict[str, str] = {}
  def allocate_regs(mvars):
    nonlocal var_size
    for v in [v for v in mvars if v is not None and v.__class__ is not int and v.nm not in rtor]:
      available_regs = s_regs if dtypes.is_float(v[1]) else x_regs
      #NOTE: Very simple spill, everything that don't fit in regs goes to mem
      if not available_regs:
        # ARM needs the stack 16-byte aligned
        var_size += 16
        available_regs.append('s0' if dtypes.is_float(out[1]) else 'x12')
        mem_vars[v.nm] = var_size
      rtor[v.nm] = available_regs.pop()

  temp_floats = ['s0', 's1', 's2']
  temp_ints = ['x12', 'x13', 'x16']
  for i, (uop, out, vin, arg) in enumerate(asm):
    # Clear regs out of interval
    for var, reg in list(rtor.items()):
      available_regs = s_regs if reg[0] == 's' else x_regs
      if var[1] not in 'B' and var not in mem_vars and i > live_range[var][1]:
        available_regs.append(rtor.pop(var))
    # Assign a registers to the variables using live ranges.
    allocate_regs([out] + vin)
    # Assign temp regs to vin and load them before direct use
    for i, v in enumerate([v for v in vin if v.__class__ is not int and v.nm in mem_vars]):
      rtor[v.nm] = temp_floats[i] if dtypes.is_float(v[1]) else temp_ints[i]
      # ARM64 addressing constraints https://devblogs.microsoft.com/oldnewthing/20220728-00/?p=106912
      ins.append(f"mov x15, {mem_vars[v.nm]}")
      ins.append(f"ldr {rtor[v.nm]}, [sp, x15]")

    if uop == Ops.SPECIAL:
      if arg.startswith('data'):
        # data 8 to n into the stack
        if int(arg[4:]) >= 8:
          ins.append(f"ldr x15, [x17, #{(int(arg[4:]) - 8) * 8}]")
          ins.append(f"mov {rtor[out.nm]}, x15")
      else:
        ins.append(f"mov {rtor[out.nm]}, #0")
        ins.append(f"loop_{arg}:")
    elif uop == Ops.CAST:
      if arg == BinaryOps.CMPLT:
        if rtor[out.nm][0] == 's':
          mov_imm(0.0, 's0')
          mov_imm(1.0, 's1')
          ins.append(f"fcsel {rtor[out.nm]}, s1, s0, lt")
        if rtor[out.nm][0] == 'x':
          mov_imm(0, 'x14')
          mov_imm(1, 'x15')
          ins.append(f"csel {rtor[out.nm]}, x15, x14, lt")
      else:
        ins.append(f"sxtw {rtor[out.nm]}, w{rtor[vin[0].nm][1:]}")
    elif uop == Ops.ALU:
      if len(vin)==2 and vin[1].__class__ is int: mov_imm(vin[1], 'x15')
      if arg == BinaryOps.MUL and out.dtype == dtypes.bool:
        ins.append(f"ands {','.join('x15' if v.__class__ is int else rtor[v.nm] for v in [out] + vin)}")
      elif arg == TernaryOps.WHERE:
        ins.append(f"fcmp {rtor[vin[0].nm]}, #0.0" if rtor[vin[0].nm][0] == 's' else f"cmp {rtor[vin[0].nm]}, #0")
        ins.append(f"{alu[arg]} {rtor[out.nm]}, {rtor[vin[1].nm]}, {rtor[vin[2].nm]}, ne")
      elif arg in [UnaryOps.LOG2, UnaryOps.SIN, UnaryOps.EXP2, UnaryOps.SQRT]:
        #NOTE: Not a real instruction, use to emulate a ext call in unicorn
        if CI: ins.append(f"{alu[arg]} {rtor[out.nm]} {rtor[vin[0].nm]}")
        else:
          save_regs = [k for k in rtor.keys() if k != out.nm and k not in mem_vars]
          ins.append(f"sub sp, sp, #{(len(save_regs))*16}")
          # Save the registers before they are cleared by func call
          for i,k in enumerate(save_regs,1):
            ins.append(f"str {rtor[k]}, [sp, #{16*i}]")
          ins.append("stp x29, x30, [sp, #0]!")
          ins.append("mov x29, sp")
          ins.append(f"fmov s0, {rtor[vin[0].nm]}")
          ins.append(alu[arg])
          ins.append(f"fmov {rtor[out.nm]}, s0")
          ins.append("mov sp, x29")
          ins.append("ldp x29, x30, [sp], #0")
          for i,k in enumerate(save_regs,1):
            ins.append(f"ldr {rtor[k]}, [sp, #{16*i}]")
          ins.append(f"add sp, sp, #{len(save_regs)*16}")
      elif arg == BinaryOps.CMPLT:
        ins.append(f"{alu[arg]} {','.join('x15' if v.__class__ is int else rtor[v.nm] for v in [out] + vin)}" if not dtypes.is_float(vin[0][1]) else f"fcmp {rtor[vin[0].nm]}, {rtor[vin[1].nm]}")
      elif arg == BinaryOps.MOD:
        rhs = 'x15' if vin[1].__class__ is int else rtor[vin[1].nm]
        ins.append(f"udiv x14, {rtor[vin[0].nm]}, {rhs}")
        ins.append(f"msub {rtor[out.nm]}, x14, {rhs}, {rtor[vin[0].nm]}")
      else:
        ins.append(f"{'f' if dtypes.is_float(vin[0][1]) else 's' if arg == BinaryOps.DIV else ''}{alu[arg]} {', '.join('x15' if v.__class__ is int else rtor[v.nm] for v in [out] + vin)}")
    elif uop == Ops.LOAD:
      if arg.__class__ in (int, float):
        mov_imm(arg, rtor[out.nm])
      else:
        #NOTE: if need casting load var in s/h0 or x/w12 temp regs
        reg_in = type_to_reg[arg[2]] + ('0' if dtypes.is_float(arg[2]) else '12') if arg[2] is not None else rtor[out.nm]
        mov_imm(arg[0], "x15")
        ins.append(f"add x15, {rtor[vin[0].nm]}, x15")
        ins.append(f"ldr{'sb' if arg[2] is not None and arg[2] in (dtypes.int8, dtypes.uint8, dtypes.bool) else ''} {reg_in}, [x15]")
        if arg[2] is not None: ins.append(f"{'fcvt' if arg[2] in [dtypes.half, dtypes.double] else 'scvtf'} {rtor[out.nm]}, {reg_in}")
    elif uop == Ops.STORE:
      #NOTE: if need casting load var in s/h0 or x/w12 temp regs
      reg_out = (type_to_reg[arg[2]] + ('0' if dtypes.is_float(arg[2]) else '12') if arg[2] is not None else rtor[vin[1].nm])
      if arg[2] is not None: ins.append(f"fcvt{'zs' if arg[2] not in [dtypes.half, dtypes.double] else '' } {reg_out}, {rtor[vin[1].nm]}")
      ins.append(f"mov x15, #{arg[0]}")
      ins.append(f"str {reg_out}, [{rtor[vin[0].nm]}, x15, lsl #0]")
    elif uop == Ops.COND_BRANCH:
      #TODO: this is a hack it shouldn't always be a cmp before a cond branch?
      if prev_uop == Ops.LOAD:
        ins.append(f"cmp {rtor[vin[0].nm]}, #0")
      ins.append(f"b.{'lt' if arg[1] else 'ge'} {arg[0][1:]}")
    elif uop == Ops.LABEL:
      ins.append(f"{arg[1:]}:")
    elif uop == Ops.ENDLOOP:
      mov_imm(arg[0], "x15")
      ins.append(f"add {rtor[vin[0].nm]}, {rtor[vin[0].nm]}, #1")
      ins.append(f"cmp {rtor[vin[0].nm]}, x15")
      ins.append(f"b.lt loop_{arg[1]}")
    prev_uop = uop
    # store regs into memory if needed
    if out is not None and out.nm in mem_vars:
      ins.append(f"mov x15, {mem_vars[out.nm]}")
      ins.append(f"str {rtor[out.nm]}, [sp, x15]")
  return "\n".join([f"//varsize {var_size}",".arch armv8-a",".text", f".global {get_name(fn_nm)}",".p2align 2", f"{get_name(fn_nm)}:", "mov x17, sp"] + [f"sub sp, sp, #{offset}" for offset in compute_offsets(var_size)]+ ins + [f"add sp, sp, #{offset}" for offset in compute_offsets(var_size)] +["ret", "\n"])

def uops_to_arm64_asm(fn_nm:str, uops:List[UOp]) -> Tuple[str, List[int], List[int], bool]:
  lang = ARM64Language()
  global_size, local_size = uops_to_asmstyle(lang, fn_nm, uops)
  return specialize_to_arm64(fn_nm, lang.ins), global_size[::-1], local_size[::-1], True