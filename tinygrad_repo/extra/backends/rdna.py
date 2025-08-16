from typing import Dict, Set
import yaml
from tinygrad.codegen.uops import UOpGraph, UOps, UOp
from tinygrad.uop.ops import BinaryOps
from tinygrad.dtype import dtypes

def uops_to_rdna(function_name:str, uops:UOpGraph) -> str:
  replace: Dict[UOp, UOp] = {}
  seen: Set[UOp] = set()
  for u in uops:
    if u in seen: continue
    seen.add(u)
    for o,n in replace.items():
      if o in u.vin and u is not n:
        u.vin = tuple(n if x == o else x for x in u.vin)
    # pointer indexing
    if u.uop in {UOps.LOAD, UOps.STORE} and u.vin[0].dtype.itemsize > 1:
      val = UOp(UOps.CONST, dtypes.int, tuple(), arg=u.vin[0].dtype.itemsize, insert_at=uops.uops.index(u))
      ptr = UOp(UOps.ALU, dtypes.int, (u.vin[1], val), arg=BinaryOps.MUL, insert_at=uops.uops.index(u))
      u.vin = (u.vin[0], ptr) + u.vin[2:]
  #uops.print()

  args = []
  ins = []

  v_cnt = 3  # v[0:2] is local_xyz
  s_cnt = 5  # s[0:1] is the address, s[2:4] is global_xyz

  r: Dict[UOp, str] = {}
  for u in uops:
    if u.uop == UOps.SPECIAL:
      if u.arg[1].startswith("lidx"):
        r[u] = f'v{u.arg[0]}'
      elif u.arg[1].startswith("gidx"):
        r[u] = f's{2+u.arg[0]}'
      else:
        raise NotImplementedError
    elif u.uop == UOps.CONST:
      #r[u] = u.arg

      # TODO: sometimes we can use s
      #r[u] = f"s{s_cnt}"
      #s_cnt += 1
      #ins.append(f"s_mov_b32 {r[u]}, {u.arg}")

      r[u] = f"v{v_cnt}"
      v_cnt += 1
      ins.append(f"v_mov_b32 {r[u]}, {u.arg}")
    elif u.uop == UOps.ALU:
      if u.arg == BinaryOps.ADD:
        r[u] = f"v{v_cnt}"
        v_cnt += 1
        ins.append(f"v_add_f32_e32 {r[u]}, {r[u.vin[0]]}, {r[u.vin[1]]}")
      elif u.arg == BinaryOps.MUL:
        r[u] = f"v{v_cnt}"
        v_cnt += 1
        if dtypes.is_float(u.dtype):
          ins.append(f"v_mul_f32_e32 {r[u]}, {r[u.vin[0]]}, {r[u.vin[1]]}")
        else:
          ins.append(f"v_mul_u32_u24 {r[u]}, {r[u.vin[0]]}, {r[u.vin[1]]}")
      else:
        raise NotImplementedError
    elif u.uop == UOps.LOAD:
      r[u] = f"v{v_cnt}"
      v_cnt += 1
      ins.append(f"global_load_b32 {r[u]}, {r[u.vin[1]]}, {r[u.vin[0]]}")
      ins.append("s_waitcnt vmcnt(0)")
    elif u.uop == UOps.STORE:
      ins.append(f"global_store_b32 {r[u.vin[1]]}, {r[u.vin[2]]}, {r[u.vin[0]]}")
    elif u.uop == UOps.DEFINE_GLOBAL:
      i = u.arg[0]
      args.append({'.address_space': 'global', '.name': f'buf_{i}', '.offset': i*8, '.size': 8,
                   '.type_name': u.dtype.name+"*", '.value_kind': 'global_buffer'})
      s_cnt += s_cnt%2  # skip
      r[u] = f"s[{s_cnt}:{s_cnt+1}]"
      s_cnt += 2
      ins.append(f"s_load_b64 {r[u]}, s[0:1], {i*8}")
      ins.append("s_waitcnt lgkmcnt(0)")
    else:
      raise NotImplementedError(f"can't render {u.uop}")

  # *** boilerplate rendering ***

  metadata = {
    'amdhsa.kernels': [{'.args': args,
      '.group_segment_fixed_size': 0, '.kernarg_segment_align': 8, '.kernarg_segment_size': args[-1][".offset"] + args[-1][".size"],
      '.language': 'OpenCL C', '.language_version': [1, 2], '.max_flat_workgroup_size': 256,
      '.name': function_name, '.private_segment_fixed_size': 0, '.sgpr_count': s_cnt, '.sgpr_spill_count': 0,
      '.symbol': f'{function_name}.kd', '.uses_dynamic_stack': False, '.vgpr_count': v_cnt, '.vgpr_spill_count': 0,
      '.wavefront_size': 32}],
    'amdhsa.target': 'amdgcn-amd-amdhsa--gfx1100', 'amdhsa.version': [1, 2]}

  boilerplate_start = f"""
.rodata
.global {function_name}.kd
.type {function_name}.kd,STT_OBJECT
.align 0x10
.amdhsa_kernel {function_name}"""

  kernel_desc = {
    '.amdhsa_group_segment_fixed_size': 0, '.amdhsa_private_segment_fixed_size': 0, '.amdhsa_kernarg_size': 0,
    '.amdhsa_next_free_vgpr': v_cnt,   # this matters!
    '.amdhsa_reserve_vcc': 0, '.amdhsa_reserve_xnack_mask': 0,
    '.amdhsa_next_free_sgpr': s_cnt,
    '.amdhsa_float_round_mode_32': 0, '.amdhsa_float_round_mode_16_64': 0, '.amdhsa_float_denorm_mode_32': 3, '.amdhsa_float_denorm_mode_16_64': 3,
    '.amdhsa_dx10_clamp': 1, '.amdhsa_ieee_mode': 1, '.amdhsa_fp16_overflow': 0,
    '.amdhsa_workgroup_processor_mode': 1, '.amdhsa_memory_ordered': 1, '.amdhsa_forward_progress': 0, '.amdhsa_enable_private_segment': 0,
    '.amdhsa_system_sgpr_workgroup_id_x': 1, '.amdhsa_system_sgpr_workgroup_id_y': 1, '.amdhsa_system_sgpr_workgroup_id_z': 1,
    '.amdhsa_system_sgpr_workgroup_info': 0, '.amdhsa_system_vgpr_workitem_id': 2, # is amdhsa_system_vgpr_workitem_id real?
    '.amdhsa_exception_fp_ieee_invalid_op': 0, '.amdhsa_exception_fp_denorm_src': 0,
    '.amdhsa_exception_fp_ieee_div_zero': 0, '.amdhsa_exception_fp_ieee_overflow': 0, '.amdhsa_exception_fp_ieee_underflow': 0,
    '.amdhsa_exception_fp_ieee_inexact': 0, '.amdhsa_exception_int_div_zero': 0,
    '.amdhsa_user_sgpr_dispatch_ptr': 0, '.amdhsa_user_sgpr_queue_ptr': 0, '.amdhsa_user_sgpr_kernarg_segment_ptr': 1,
    '.amdhsa_user_sgpr_dispatch_id': 0, '.amdhsa_user_sgpr_private_segment_size': 0, '.amdhsa_wavefront_size32': 1, '.amdhsa_uses_dynamic_stack': 0}

  code_start = f""".end_amdhsa_kernel
.text
.global {function_name}
.type {function_name},@function
.p2align 8
{function_name}:
"""

  ins += ['s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)', 's_endpgm', 's_code_end']
  return ".amdgpu_metadata\n" + yaml.dump(metadata) + ".end_amdgpu_metadata" + \
         boilerplate_start + "\n" + '\n'.join("%s %d" % x for x in kernel_desc.items()) + "\n" + code_start + \
         '\n'.join(ins) + f"\n.size {function_name}, .-{function_name}"
