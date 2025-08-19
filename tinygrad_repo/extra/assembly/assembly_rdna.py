import yaml
from typing import Tuple, Set, Dict
from tinygrad import dtypes
from tinygrad.codegen.assembly import AssemblyCodegen, Register
from tinygrad.codegen.opt.kernel import Ops
from tinygrad.uop.ops import BinaryOps, UnaryOps, TernaryOps
from tinygrad.runtime.ops_gpu import ROCM_LLVM_PATH

# ugh, is this really needed?
from extra.helpers import enable_early_exec
early_exec = enable_early_exec()

boilerplate_start = """
.global _start
_start:
.rodata
.align 0x10
.global code.kd
.type code.kd,STT_OBJECT
.amdhsa_kernel code"""

code_start = """.end_amdhsa_kernel
.text
code:
"""

# https://github.com/RadeonOpenCompute/ROCm_Documentation/blob/master/ROCm_Compiler_SDK/ROCm-Codeobj-format.rst
# https://github.com/ROCm-Developer-Tools/ROCm-ComputeABI-Doc/blob/master/AMDGPU-ABI.md#initial-kernel-register-state
# RDNA3 is actually a SIMD machine!
class RDNACodegen(AssemblyCodegen):
  supports_float4: bool = True
  supports_float4_alu: bool = True
  supports_load3: bool = True
  sin_is_sin2pi: bool = True
  no_div: bool = True

  def specialize(self, asm) -> Tuple[str, str]:
    args = []
    for i,b in enumerate(self.bufs): args.append({'.address_space': 'global', '.name': f'buf_{i}', '.offset': i*8, '.size': 8, '.type_name': b.dtype.name+"*", '.value_kind': 'global_buffer'})
    ins = []

    v_cnt = 3  # v[0:2] is local_xyz
    s_cnt = 5  # s[0:1] is the address, s[2:4] is global_xyz

    dtype_to_rdnatype = {dtypes.float32: "f32", dtypes.int64: "i64", dtypes.int32: "i32", dtypes.uint64: "u64", dtypes.bool: "i32"}
    alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", TernaryOps.MULACC: "fma",
           BinaryOps.MAX: "max", UnaryOps.RECIP: "rcp",
           UnaryOps.NOOP: "mov", UnaryOps.SIN: "sin", UnaryOps.LOG2: "log", UnaryOps.EXP2: "exp",
           BinaryOps.CMPLT: "cmp_lt"}

    pend_regs:Set[Register] = set()
    rtor:Dict[Register, str] = {}
    def reg_in(x):
      nonlocal pend_regs
      #print("reg_in", x, rtor[x], pend_regs)
      if x in pend_regs:
        #print("clear")
        ins.append('s_waitcnt lgkmcnt(0), vmcnt(0)')
        pend_regs.clear()
      return rtor[x]
    def reg_out(x):
      return rtor[x]
    for uop, out, vin, arg in asm:
      if uop == Ops.DEFINE_REGISTER:
        if arg[0][0] in [dtypes.uint32, dtypes.uint64, dtypes.int64, dtypes.int32, dtypes.float32, dtypes.float.vec(4)]:
          for i in range(arg[2]):
            # TODO: Re-use gaps created by this to avoid wasting registers
            align = int(arg[0][0].itemsize / 4)
            if arg[0][1]:
              s_cnt += s_cnt % align
              reg_name = f"s[{s_cnt}:{s_cnt + align - 1}]" if align > 1 else f"s{s_cnt}"
              s_cnt += align
            else:
              v_cnt += v_cnt % align
              reg_name = f"v[{v_cnt}:{v_cnt + align - 1}]" if align > 1 else f"v{v_cnt}"
              v_cnt += align
            rtor[Register(f"%{arg[1]}{i}", *arg[0])] = reg_name

            if arg[0][0] == dtypes.float.vec(4):
              for off in range(4):
                reg_name = f"s{s_cnt-align+off}" if arg[0][1] else f"v{v_cnt-align+off}"
                rtor[Register(f"%{arg[1]}{i}", dtypes.float, False, off=off)] = reg_name
        elif arg[0][0] == dtypes.bool:
          for i in range(arg[2]):
            reg_name = "scc" if arg[0][1] else "vcc_lo" # `_lo` suffix since we're running wavefront_size=32
            rtor[Register(f"%{arg[1]}{i}", *arg[0])] = reg_name
        else:
          raise NotImplementedError("DEFINE_REGISTER not implemented for arg: ", arg)
      elif uop == Ops.SPECIAL:
        if arg.startswith('buf'):
          i = int(arg[3:])
          ins.append(f's_load_b64 {reg_out(out)}, s[0:1], {i*8}')
          pend_regs.add(out)
          for r in out.subregs(): pend_regs.add(r)
        elif arg.startswith('gid'):
          ins.append(f'v_mov_b32 {reg_out(out)}, s{2+int(arg[3])}')
          # the docs lied, this is actually y
          if int(arg[3]) == 2: ins.append("v_bfe_u32 v2, v0, 20, 10")  # untested
          if int(arg[3]) == 1: ins.append("v_bfe_u32 v1, v0, 10, 10")
          elif int(arg[3]) == 0: ins.append("v_and_b32_e32 v0, 0x3ff, v0")
          # get local size
          offset = len(args)*8
          args.append({".offset": offset, ".value_kind": f"hidden_group_size_{'xyz'[int(arg[3])]}", ".size": 8})
          ins.append(f's_load_b32 s{2+int(arg[3])}, s[0:1], {offset}')
          ins.append('s_waitcnt vmcnt(0) lgkmcnt(0)')
          pend_regs.clear()
          ins.append(f'v_mul_i32_i24 {reg_out(out)}, {reg_out(out)}, s{2+int(arg[3])}')
          ins.append(f'v_add_nc_u32 {reg_out(out)}, v{int(arg[3])}, {reg_out(out)}')
      elif uop == Ops.CONST:
        if arg == float('inf'): arg = "0x7f800000"
        elif arg == float('-inf'): arg = "0xff800000"
        if out.dtype == dtypes.float.vec(4):
          for off in range(4):
            ins.append(f"{'s_' if out.scalar else 'v_'}mov_b32 {reg_out(Register(out.nm, dtypes.float, False, off=off))}, {arg}")
        else:
          ins.append(f"{'s_' if out.scalar else 'v_'}mov_b32 {reg_out(out)}, {arg}")
      elif uop == Ops.ALU:
        if arg in [BinaryOps.CMPLT]:
          ins.append(f"{'s' if out.scalar else 'v'}_{alu[arg]}_{dtype_to_rdnatype[out.dtype]} {', '.join(reg_in(x) if x.__class__ is Register else str(x) for x in vin)}")
        else:
          alu_arg = alu[arg]
          if arg == TernaryOps.MULACC and out == vin[2]:
            alu_arg = "fmac"
            vin = vin[0:2]
          if out.dtype == dtypes.float.vec(4):
            for rr in zip(*[x.subregs() if x.dtype == dtypes.float.vec(4) else [x,x,x,x] for x in [out]+vin]):
              ins.append(f"{'s_' if rr[0].scalar else 'v_'}{alu_arg}_{dtype_to_rdnatype[rr[0].dtype]} {reg_out(rr[0])}, {', '.join(reg_in(x) if x.__class__ is Register else str(x) for x in rr[1:])}")
          else:
            ins.append(f"{'s_' if out.scalar else 'v_'}{alu_arg}_{dtype_to_rdnatype[out.dtype] if arg != UnaryOps.NOOP else 'b32'}{'_i24' if arg == BinaryOps.MUL and out.dtype != dtypes.float32 and not out.scalar else ''} {reg_out(out)}, {', '.join(reg_in(x) if x.__class__ is Register else str(x) for x in vin)}")
      elif uop == Ops.LOAD:
        if out.scalar:
          # swap arg order
          ins.append(f's_load_b32 {reg_out(out)}, {reg_in(vin[0])}, {reg_in(vin[1])} offset:{arg[0]}')
        else:
          ins.append(f'global_load_{"b128" if out.dtype == dtypes.float.vec(4) else "b32"} {reg_out(out)}, {reg_in(vin[1])}, {reg_in(vin[0])} offset:{arg[0]}')
        pend_regs.add(out)
        for r in out.subregs(): pend_regs.add(r)
      elif uop == Ops.STORE:
        ins.append(f'global_store_{"b128" if vin[1].dtype == dtypes.float.vec(4) else "b32"} {reg_in(vin[2])}, {reg_in(vin[1])}, {reg_in(vin[0])} offset:{arg[0]}')
      elif uop == Ops.LABEL:
        ins.append(f"{arg}:")
      elif uop == Ops.COND_BRANCH:
        ins.append(f"s_cbranch_scc{'1' if arg[1] else '0'} {arg[0]}")
      elif uop == Ops.CAST:
        if vin[0].dtype == dtypes.bool:
          if out.dtype == dtypes.float32:
            ins.append(f"v_cndmask_b32 {reg_out(out)}, 0.0, 1.0, {reg_in(vin[0])}")
        else:
          raise NotImplementedError(f"cast {vin[0].dtype} -> {out.dtype}")
      else:
        raise NotImplementedError(uop)

    ins += ['s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)', 's_endpgm', 's_code_end']

    # dual alu group
    seen = set()
    new_ins = []
    for i,tins in enumerate(ins):
      if tins in seen: continue
      if tins.startswith("v_fmac_f32"):
        for gins in reversed(ins[i+1:]):
          if gins in seen: continue
          if gins.startswith("v_fmac_f32"):
            r0 = [int(x[1:].strip(',')) for x in tins.split(" ")[1:]]
            r1 = [int(x[1:].strip(',')) for x in gins.split(" ")[1:]]
            if r0[0]%2 == r1[0]%2: continue
            if r0[1]%2 == r1[1]%2: continue
            if r0[2]%2 == r1[2]%2: continue
            new_ins.append(tins.replace("v_", "v_dual_")+" :: " + gins.replace("v_", "v_dual_"))
            seen.add(tins)
            seen.add(gins)
            break
      if tins not in seen:
        new_ins.append(tins)
    ins = new_ins

    return 'code', self.assemble(args, ins, v_cnt, s_cnt)

  def assemble(self, args, ins, v_cnt, s_cnt):
    kernel_desc = {'.amdhsa_group_segment_fixed_size': 0, '.amdhsa_private_segment_fixed_size': 0, '.amdhsa_kernarg_size': 0,
                   '.amdhsa_next_free_vgpr': v_cnt,   # this matters!
                   '.amdhsa_reserve_vcc': 0, '.amdhsa_reserve_xnack_mask': 0,
                   '.amdhsa_next_free_sgpr': s_cnt,
                   '.amdhsa_float_round_mode_32': 0, '.amdhsa_float_round_mode_16_64': 0, '.amdhsa_float_denorm_mode_32': 3, '.amdhsa_float_denorm_mode_16_64': 3, '.amdhsa_dx10_clamp': 1, '.amdhsa_ieee_mode': 1,
                   '.amdhsa_fp16_overflow': 0, '.amdhsa_workgroup_processor_mode': 1, '.amdhsa_memory_ordered': 1, '.amdhsa_forward_progress': 0, '.amdhsa_enable_private_segment': 0,
                   '.amdhsa_system_sgpr_workgroup_id_x': 1, '.amdhsa_system_sgpr_workgroup_id_y': 1, '.amdhsa_system_sgpr_workgroup_id_z': 1,
                   '.amdhsa_system_sgpr_workgroup_info': 0, '.amdhsa_system_vgpr_workitem_id': 2, # is amdhsa_system_vgpr_workitem_id real?
                   '.amdhsa_exception_fp_ieee_invalid_op': 0, '.amdhsa_exception_fp_denorm_src': 0, '.amdhsa_exception_fp_ieee_div_zero': 0, '.amdhsa_exception_fp_ieee_overflow': 0, '.amdhsa_exception_fp_ieee_underflow': 0,
                   '.amdhsa_exception_fp_ieee_inexact': 0, '.amdhsa_exception_int_div_zero': 0, '.amdhsa_user_sgpr_dispatch_ptr': 0, '.amdhsa_user_sgpr_queue_ptr': 0, '.amdhsa_user_sgpr_kernarg_segment_ptr': 1,
                   '.amdhsa_user_sgpr_dispatch_id': 0, '.amdhsa_user_sgpr_private_segment_size': 0, '.amdhsa_wavefront_size32': 1, '.amdhsa_uses_dynamic_stack': 0}

    metadata = {'amdhsa.kernels': [{'.args': args,
                  '.group_segment_fixed_size': 0, '.kernarg_segment_align': 8, '.kernarg_segment_size': args[-1][".offset"] + args[-1][".size"],
                  '.language': 'OpenCL C', '.language_version': [1, 2], '.max_flat_workgroup_size': 256,
                  '.name': 'code', '.private_segment_fixed_size': 0, '.sgpr_count': s_cnt, '.sgpr_spill_count': 0,
                  '.symbol': 'code.kd', '.uses_dynamic_stack': False, '.vgpr_count': v_cnt, '.vgpr_spill_count': 0,
                  '.wavefront_size': 32}],
                'amdhsa.target': 'amdgcn-amd-amdhsa--gfx1100', 'amdhsa.version': [1, 2]}

    code = boilerplate_start + "\n" + '\n'.join("%s %d" % x for x in kernel_desc.items()) + "\n" +  code_start + '\n'.join(ins) + "\n.amdgpu_metadata\n" + yaml.dump(metadata) + ".end_amdgpu_metadata"
    obj = early_exec(([ROCM_LLVM_PATH / "llvm-mc", '--arch=amdgcn', '--mcpu=gfx1100', '--triple=amdgcn-amd-amdhsa', '--filetype=obj', '-'], code.encode("utf-8")))
    asm = early_exec(([ROCM_LLVM_PATH / "ld.lld", "/dev/stdin", "-o", "/dev/stdout", "--pie"], obj))
    return asm
