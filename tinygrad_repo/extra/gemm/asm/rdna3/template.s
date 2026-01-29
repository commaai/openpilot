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
  .amdhsa_group_segment_fixed_size 30336
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 32
  # register usage (RSRC1)
  .amdhsa_next_free_vgpr 256
  .amdhsa_next_free_sgpr 100
  # workgroup / workitem IDs (RSRC2)
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  # user SGPRs: kernarg ptr in s[0:1]
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_count 2
  # gfx10+ / gfx11 specifics (RSRC1[29..31])
  .amdhsa_wavefront_size32 1
  .amdhsa_workgroup_processor_mode 1
  .amdhsa_memory_ordered 1
  .amdhsa_forward_progress 1
  # misc for gfx11
  .amdhsa_dx10_clamp 1
  .amdhsa_ieee_mode 1
  .amdhsa_uses_dynamic_stack 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  generic
        .name:           C
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f16
      - .address_space:  generic
        .name:           A
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f16
      - .address_space:  generic
        .name:           B
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f16
    .group_segment_fixed_size: 30336
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 128
    .name:           gemm
    .private_segment_fixed_size: 0
    .sgpr_count:     70
    .sgpr_spill_count: 0
    .symbol:         gemm.kd
    .vgpr_count:     256
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.version:
  - 1
  - 1
...
.end_amdgpu_metadata
