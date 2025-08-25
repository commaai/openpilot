.global _start
_start:
.rodata
.align 0x10
.global code.kd
.type code.kd,STT_OBJECT
# amd_kernel_code_t (must be at 0x440 for kernel_code_entry_byte_offset to be right)
code.kd:
# amd_kernel_..., amd_machine_...
.long 0,0,0,0
# kernel_code_entry_byte_offset, kernel_code_prefetch_byte_offset
.long 0x00000bc0,0x00000000,0x00000000,0x00000000
# kernel_code_prefetch_byte_size, max_scratch_backing_memory_byte_size
.long 0,0,0,0
# compute_pgm_rsrc1, compute_pgm_rsrc2, kernel_code_properties, workitem_private_segment_byte_size
.long 0x60af0000,0x0000009e,0x00000408,0x00000000
# compute_pgm_rsrc1 |= AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32 | AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64
# compute_pgm_rsrc1 |= AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP | AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE
# compute_pgm_rsrc2 |= AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT = 0xF
# compute_pgm_rsrc2 |= AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X
# kernel_code_properties |= AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR = 1
# kernel_code_properties |= AMD_KERNEL_CODE_PROPERTIES_RESERVED1 = 1
.text
.global code
.type code,STT_FUNC
code:
# https://llvm.org/docs/AMDGPUUsage.html#initial-kernel-execution-state
# s[0:1] contains the kernarg_address
# TODO: can we use s[2:3] if this was really a wave since we only alloced 2 SGPRs?
s_load_b64 s[2:3], s[0:1], null

s_mov_b32 s8, 0
loop:
s_addk_i32 s8, 1
s_cmp_eq_u32 s8, 100000
// FLOPS
s_cbranch_scc0 loop

# wait for the s_load_b64
s_waitcnt lgkmcnt(0)

v_dual_mov_b32 v0, 4 :: v_dual_mov_b32 v1, 2.0
global_store_b32 v0, v1, s[2:3]

# Deallocate all VGPRs for this wave. Use only when next instruction is S_ENDPGM.
s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
s_endpgm
s_code_end

.amdgpu_metadata
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .name:           a
        .offset:         0
        .size:           8
        .type_name:      'float*'
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .language:       OpenCL C
    .language_version:
      - 1
      - 2
    .max_flat_workgroup_size: 256
    .name:           code
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .sgpr_spill_count: 0
    .symbol:         code.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1100
amdhsa.version:
  - 1
  - 2
.end_amdgpu_metadata
