.text
.section	.text.
.global	gemm
.p2align	8
.type	gemm,@function

gemm:
INSTRUCTIONS

.section .rodata,"a",@progbits
.p2align 6, 0x0
.amdhsa_kernel gemm
  # basic memory requirements
  .amdhsa_group_segment_fixed_size 133120
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 28
  # register usage (RSRC1)
  .amdhsa_next_free_vgpr 504
  .amdhsa_next_free_sgpr 96
  # workgroup / workitem IDs (RSRC2)
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  # user SGPRs, we only specify the kernel args ptr in s[0:1]
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_count 2
  .amdhsa_user_sgpr_kernarg_preload_length 0
  .amdhsa_user_sgpr_kernarg_preload_offset 0
  # gfx90a / gfx940 specifics (RSRC3)
  .amdhsa_accum_offset 248
  .amdhsa_uses_dynamic_stack 0
  .amdhsa_tg_split 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .name: gemm
    .symbol: gemm.kd
    .args:
      - .name: C
        .address_space: global
        .offset: 0
        .size: 8
        .value_kind: global_buffer
        .value_type: bf16
      - .name: B
        .address_space: global
        .offset: 8
        .size: 8
        .value_kind: global_buffer
        .value_type: bf16
      - .name: A
        .address_space: global
        .offset: 16
        .size: 8
        .value_kind: global_buffer
        .value_type: bf16
      - .name: sz
        .offset: 24
        .size: 4
        .value_kind: by_value
        .value_type: u32
    .group_segment_fixed_size: 133120
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .max_flat_workgroup_size: 256
    .sgpr_count: 88
    .sgpr_spill_count: 0
    .vgpr_count: 248
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 0
...
.end_amdgpu_metadata
