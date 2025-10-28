    .text
    .globl matmul
    .p2align 8 
    .type matmul,@function
matmul:
    s_mov_b32 s1, 1000000
    s_mov_b32 s2, 0
    inner_loop:
        INSTRUCTION
        s_sub_u32 s1, s1, 1
        s_cmp_lg_i32 s1, s2
        s_cbranch_scc1 inner_loop
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel matmul
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: matmul
    .symbol: matmul.kd
    .kernarg_segment_size:  0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 32
    .max_flat_workgroup_size: 1024
    .args:
...
.end_amdgpu_metadata