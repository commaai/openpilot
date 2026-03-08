    .text
    .globl matmul
    .p2align 8
    .type matmul,@function
matmul:
    INSTRUCTION

.rodata
.p2align 6
.amdhsa_kernel matmul
  .amdhsa_next_free_vgpr VGPR_COUNT
  .amdhsa_next_free_sgpr 3
  DIRECTIVE
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: matmul
    .symbol: matmul.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 32
    .max_flat_workgroup_size: 1024
...
.end_amdgpu_metadata
