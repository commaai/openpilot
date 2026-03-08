# RDNA3 128x128 tiled GEMM kernel - DSL version
# Computes C = A @ B for 4096x4096 float32 matrices using 128x128 tiles
#
# Architecture: RDNA3 (gfx1100)
# Tile size: 128x128 (each workgroup computes one tile of C)
# Workgroup: 128 threads (arranged as 32x4 for coalesced memory access)
# Inner loop: 8 iterations per K-block, processing 8 columns of A and 8 rows of B
#
# Accumulators: 128 vgprs (v[2-129])

import numpy as np
from pathlib import Path
from tinygrad import Tensor, Device, Context, GlobalCounters
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.helpers import getenv, colored
from tinygrad.engine.realize import Estimates
from extra.assembly.amd.dsl import s, v, VCC_LO, NULL
from extra.assembly.amd.autogen.rdna3.ins import *

# =============================================================================
# Kernel constants
# =============================================================================
LDS_SIZE = 8320       # Local data share size in bytes
MATRIX_DIM = 4096     # Matrix dimension N (assumes square NxN matrices)
LDS_A_STRIDE = 0x210  # LDS stride for A tile (528 bytes)
LDS_B_STRIDE = 0x200  # LDS stride for B tile (512 bytes)
LDS_BASE_OFFSET = 0x1080  # Base LDS offset for tiles
ADDR_MASK = 0x3fffff80    # Address alignment mask

# =============================================================================
# Named register assignments (VGPRs)
# =============================================================================
V_LANE_ID = 0             # lane_id set on startup
# Use tile gaps (v146-159) for named regs to minimize max VGPR
V_LANE_ID_MOD8 = 146      # lane_id & 7
V_LANE_MOD8_X4 = 147      # (lane_id & 7) << 2
V_LANE_DIV8_X4 = 150      # ((lane_id >> 3) & 3) << 2
V_LDS_B_BASE = 151        # LDS B-tile base address for inner loop
V_LDS_A_BASE = 154        # LDS A-tile base address for inner loop
V_GLOBAL_A_ADDR = 155     # global memory A prefetch address
V_GLOBAL_B_ADDR = 158     # global memory B prefetch address
V_LDS_A_ADDR = 159        # single base register for A stores
V_LDS_B_ADDR = 162        # single base register for B stores

# LDS tile register destinations - SEPARATE from DATA to avoid overlap
# A on banks 2-3, B on banks 0-1 to avoid bank conflicts in VOPD
V_A_TILE_REGS = [130, 134, 138, 142]  # A tile: banks 2,2,2,2 (130%4=2, etc.)
V_B_TILE_REGS = [132, 136, 140, 144, 148, 152, 156, 160]  # B tile: banks 0,0,0,0,0,0,0,0

# =============================================================================
# Named register assignments (SGPRs)
# =============================================================================
S_OUT_PTR = (0, 1)        # output C matrix base pointer
S_TILE_X = 2              # workgroup_x << 7
S_TILE_Y = 3              # workgroup_y << 7
S_DIM_N = 4               # matrix dimension N
S_LOOP_BOUND = 7          # K-8 (loop termination bound)
S_LOOP_CTR = 12           # loop counter (increments by 8)
S_PREFETCH_FLAG = 13      # prefetch condition flag / row stride in epilogue
S_WORKGROUP_X = 14        # workgroup_id_x
S_WORKGROUP_Y = 15        # workgroup_id_y
# Kernarg load destinations
S_KERNARG_A = (20, 21)    # A pointer from kernarg
S_KERNARG_B = (22, 23)    # B pointer from kernarg
# Prefetch base pointers (8 pairs each, 16KB/256KB apart)
S_PREFETCH_B = 24         # s[24:39] - 8 B tile pointers
S_PREFETCH_A = 40         # s[40:55] - 8 A tile pointers

# =============================================================================
# Data tables
# =============================================================================

# Accumulator grid: ACC_GRID[a_idx][b_idx] = vgpr for C[a,b]
# a_idx: which A value (0-7), b_idx: which B value (0-15)
# Scattered due to VOPD bank constraints (vdst_x % 4 != vdst_y % 4)
# Range is from v2 - v129
ACC_GRID = [
  [  5,  3,  9,  8,   37, 35, 41, 40,   69, 67, 73, 72,  101, 99,105,104],  # a0
  [  4,  2,  7,  6,   36, 34, 39, 38,   68, 66, 71, 70,  100, 98,103,102],  # a1
  [ 17, 16, 13, 11,   49, 48, 45, 43,   81, 80, 77, 75,  113,112,109,107],  # a2
  [ 15, 14, 12, 10,   47, 46, 44, 42,   79, 78, 76, 74,  111,110,108,106],  # a3
  [ 21, 19, 25, 24,   53, 51, 57, 56,   85, 83, 89, 88,  117,115,121,120],  # a4
  [ 20, 18, 23, 22,   52, 50, 55, 54,   84, 82, 87, 86,  116,114,123,122],  # a5
  [125,128, 29, 27,   33, 32, 61, 59,   65, 64, 93, 91,   97, 96,129,127],  # a6
  [119,118, 28, 26,   31, 30, 60, 58,   63, 62, 92, 90,   95, 94,124,126],  # a7
]

# Optimized (a_pair, b_pair) iteration order for better GPU scheduling
# Interleaves A and B pairs to maximize instruction-level parallelism
FMAC_PAIR_ORDER = [
  (0,0),(0,1),(1,1),(1,0), (2,0),(2,1),(3,1),(3,2), (0,2),(0,3),(1,3),(1,2), (2,2),(2,3),(3,3),(3,4),
  (0,4),(0,5),(1,5),(1,4), (2,4),(2,5),(3,5),(3,6), (0,6),(0,7),(1,7),(1,6), (2,6),(2,7),(3,7),(3,0),
]

def derive_fmac_pattern(acc_grid, a_tile_regs=None, b_tile_regs=None):
  """Generate 64 dual FMAC ops from accumulator grid with optimized iteration order."""
  pattern = []
  for idx, (a_pair, b_pair) in enumerate(FMAC_PAIR_ORDER):
    a_even, a_odd = a_pair * 2, a_pair * 2 + 1
    b_even, b_odd = b_pair * 2, b_pair * 2 + 1
    a_base, b_base = a_tile_regs[a_pair], b_tile_regs[b_pair]
    # Op 1: normal order -> C[a_even, b_even] + C[a_odd, b_odd]
    pattern.append((acc_grid[a_even][b_even], acc_grid[a_odd][b_odd],
                   a_base, b_base, a_base+1, b_base+1))
    # Op 2: alternate swapping A vs B to vary register banks
    if idx % 2 == 0:  # swap B
      pattern.append((acc_grid[a_even][b_odd], acc_grid[a_odd][b_even],
                     a_base, b_base+1, a_base+1, b_base))
    else:  # swap A
      pattern.append((acc_grid[a_odd][b_even], acc_grid[a_even][b_odd],
                     a_base+1, b_base, a_base, b_base+1))
  return pattern

# Derived: 64 dual FMAC operations
FMAC_PATTERN = derive_fmac_pattern(ACC_GRID, V_A_TILE_REGS, V_B_TILE_REGS)

def derive_permute_swaps(acc_grid, out_regs):
  """Derive swap sequence to permute accumulators from FMAC layout to output order.

  After FMAC loop: acc_grid[a][b] holds C[a,b]
  Output order: for row_half in 0,1; col_group in 0-3; row_in_group in 0-3; b_off in 0-3
    -> need C[row_half*4 + row_in_group, col_group*4 + b_off] in specified reg order
  """
  def target_ab(i):
    row_half, col_group = i // 64, (i // 16) % 4
    row_in_group, b_off = (i // 4) % 4, i % 4
    return (row_half * 4 + row_in_group, col_group * 4 + b_off)

  reg_contents = {acc_grid[a][b]: (a, b) for a in range(8) for b in range(16)}
  ab_location = {ab: r for r, ab in reg_contents.items()}

  swaps = []
  for i in range(128):
    target_reg, needed_ab = out_regs[i], target_ab(i)
    current_reg = ab_location[needed_ab]
    if current_reg != target_reg:
      swaps.append((current_reg, target_reg))
      ab_at_target = reg_contents.get(target_reg)
      reg_contents[target_reg], ab_location[needed_ab] = needed_ab, target_reg
      if ab_at_target is not None:
        reg_contents[current_reg], ab_location[ab_at_target] = ab_at_target, current_reg
  return swaps

# Derived: swap sequence to arrange accumulators for output
# Each group of 4 registers is ascending for direct global_store_b128
OUT_REGS = [r for i in range(32) for r in range(126 - i*4, 130 - i*4)]
PERMUTE_SWAPS = derive_permute_swaps(ACC_GRID, OUT_REGS)

# =============================================================================
# LDS tile staging registers
# =============================================================================
# DATA regs receive contiguous global prefetch, then write to LDS
# TILE regs receive scattered LDS loads (ds_load_b64 pairs), then feed FMACs
# Contiguous layout with mod4=[3,0,1,2,3,0,1,2] for bank conflict avoidance
V_LDS_A_DATA = [163, 164, 165, 166, 167, 168, 169, 170]
V_LDS_B_DATA = [171, 172, 173, 174, 175, 176, 177, 178]

# Initial tile prefetch: (vdst, saddr_lo) - load into A data regs using B prefetch pointers (s[24:31])
INIT_PREFETCH = [(V_LDS_A_DATA[i], S_PREFETCH_B+2*i) for i in range(4)]

# Global memory prefetch schedule: (vdst1, vdst2, addr_vreg, saddr_lo1, saddr_lo2)
# First 2 pairs from B prefetch pointers (s[32:39]), next 4 pairs from A prefetch pointers (s[40:55])
PREFETCH_LOADS = [(V_LDS_A_DATA[4+2*i], V_LDS_A_DATA[4+2*i+1], V_GLOBAL_B_ADDR, S_PREFETCH_B+8+4*i, S_PREFETCH_B+10+4*i) for i in range(2)] + \
                 [(V_LDS_B_DATA[2*(i-2)], V_LDS_B_DATA[2*(i-2)+1], V_GLOBAL_A_ADDR, S_PREFETCH_A+4*(i-2), S_PREFETCH_A+2+4*(i-2)) for i in range(2, 6)]

# =============================================================================
# Kernel class
# =============================================================================

class Kernel:
  def __init__(self, arch='gfx1100'): self.instructions, self.labels, self.pos, self.arch = [], {}, 0, arch
  def label(self, name): self.labels[name] = self.pos

  def emit(self, inst, target=None):
    self.instructions.append(inst)
    inst._target, inst._pos = target, self.pos
    self.pos += inst.size()
    return inst

  def waitcnt(self, lgkm=None, vm=None):
    """Wait for memory operations. lgkm=N waits until N lgkm ops remain, vm=N waits until N vmem ops remain."""
    vmcnt, lgkmcnt, expcnt = vm if vm is not None else 63, lgkm if lgkm is not None else 63, 7
    waitcnt = (expcnt & 0x7) | ((lgkmcnt & 0x3f) << 4) | ((vmcnt & 0x3f) << 10)
    self.emit(s_waitcnt(simm16=waitcnt))

  def to_asm(self):
    # Patch branch offsets: simm16 = (target_pos - branch_end_pos) / 4
    for inst in self.instructions:
      if inst._target is None: continue
      offset_dwords = (self.labels[inst._target] - inst._pos - inst.size()) // 4
      if not -32768 <= offset_dwords <= 32767: raise ValueError(f"branch to '{inst._target}' offset {offset_dwords} exceeds simm16 range")
      inst.simm16 = offset_dwords

    # TODO: replace this with direct ELF
    body = ['\t' + inst.disasm() for inst in self.instructions]

    # limit wave occupancy by using more LDS
    lds_size = max(LDS_SIZE, 65536//getenv("LIMIT_OCC", 65536))

    # HSA kernel descriptor attributes (zeros included for compatibility)
    hsa = [
      ('group_segment_fixed_size', lds_size), ('private_segment_fixed_size', 0), ('kernarg_size', 36),
      ('user_sgpr_count', 14), ('user_sgpr_dispatch_ptr', 0), ('user_sgpr_queue_ptr', 0),
      ('user_sgpr_kernarg_segment_ptr', 1), ('user_sgpr_dispatch_id', 0), ('user_sgpr_private_segment_size', 0),
      ('wavefront_size32', 1), ('uses_dynamic_stack', 0), ('enable_private_segment', 0),
      ('system_sgpr_workgroup_id_x', 1), ('system_sgpr_workgroup_id_y', 1), ('system_sgpr_workgroup_id_z', 0),
      ('system_sgpr_workgroup_info', 0), ('system_vgpr_workitem_id', 0), ('next_free_vgpr', 179),
      ('next_free_sgpr', 16), ('float_round_mode_32', 0), ('float_round_mode_16_64', 0),
      ('float_denorm_mode_32', 3), ('float_denorm_mode_16_64', 3), ('dx10_clamp', 1), ('ieee_mode', 1),
      ('fp16_overflow', 0), ('workgroup_processor_mode', 0), ('memory_ordered', 1), ('forward_progress', 0),
      ('shared_vgpr_count', 0)]

    return '\n'.join([
      '\t.text', f'\t.amdgcn_target "amdgcn-amd-amdhsa--{self.arch}"',
      '\t.protected\tkernel', '\t.globl\tkernel', '\t.p2align\t8', '\t.type\tkernel,@function', 'kernel:',
      *body,
      '\t.section\t.rodata,"a",@progbits', '\t.p2align\t6, 0x0', '\t.amdhsa_kernel kernel',
      *[f'\t\t.amdhsa_{k} {v}' for k, v in hsa],
      '\t.end_amdhsa_kernel', '\t.text', '.Lfunc_end0:', '\t.size\tkernel, .Lfunc_end0-kernel',
      '\t.amdgpu_metadata', '---', 'amdhsa.kernels:', '  - .args:',
      *[f'      - .address_space: global\n        .offset: {i*8}\n        .size: 8\n        .value_kind: global_buffer' for i in range(3)],
      f'    .group_segment_fixed_size: {lds_size}', '    .kernarg_segment_align: 8',
      '    .kernarg_segment_size: 24', '    .max_flat_workgroup_size: 128', '    .name: kernel',
      '    .private_segment_fixed_size: 0', '    .sgpr_count: 60', '    .symbol: kernel.kd',
      '    .vgpr_count: 179', '    .wavefront_size: 32', f'amdhsa.target: amdgcn-amd-amdhsa--{self.arch}',
      'amdhsa.version:', '  - 1', '  - 2', '...', '\t.end_amdgpu_metadata'])


# =============================================================================
# Kernel builder
# =============================================================================

def build_kernel(arch='gfx1100'):
  k = Kernel(arch)

  # ===========================================================================
  # PROLOGUE: Load kernel arguments, compute tile coordinates and addresses
  # ===========================================================================
  k.emit(s_load_b128(sdata=s[S_KERNARG_A[0]:S_KERNARG_B[1]], sbase=s[0:1], offset=0x0, soffset=NULL))
  k.emit(s_load_b64(sdata=s[S_OUT_PTR[0]:S_OUT_PTR[1]], sbase=s[0:1], offset=0x10, soffset=NULL))
  k.emit(s_mov_b32(s[S_DIM_N], MATRIX_DIM))
  k.emit(s_mov_b32(s[S_LOOP_CTR], 0))  # used by LDS swizzle, always 0 for valid workgroups
  k.emit(s_lshl_b32(s[S_TILE_X], s[S_WORKGROUP_X], 7))
  k.emit(s_lshl_b32(s[S_TILE_Y], s[S_WORKGROUP_Y], 7))

  # Lane-derived values
  k.emit(v_and_b32_e32(v[V_LANE_ID_MOD8], 7, v[V_LANE_ID]))
  k.emit(v_lshrrev_b32_e32(v[4], 3, v[V_LANE_ID]))
  k.emit(v_or_b32_e32(v[1], s[S_TILE_X], v[V_LANE_ID]))
  k.emit(v_or_b32_e32(v[22], s[S_TILE_Y], v[4]))
  k.emit(v_lshlrev_b32_e32(v[V_LANE_MOD8_X4], 2, v[V_LANE_ID_MOD8]))
  k.waitcnt(lgkm=0)

  # Compute 8 A and B matrix tile base pointers for prefetch
  k.emit(s_mov_b64(s[S_PREFETCH_B:S_PREFETCH_B+1], s[S_KERNARG_B[0]:S_KERNARG_B[1]]))  # B[0]: no offset
  for i in range(1, 8):  # B: 16KB apart
    k.emit(s_add_u32(s[S_PREFETCH_B+i*2], s[S_KERNARG_B[0]], i * 0x4000))
    k.emit(s_addc_u32(s[S_PREFETCH_B+i*2+1], s[S_KERNARG_B[1]], 0))
  k.emit(s_mov_b64(s[S_PREFETCH_A:S_PREFETCH_A+1], s[S_KERNARG_A[0]:S_KERNARG_A[1]]))  # A[0]: no offset
  for i in range(1, 8):  # A: 256KB apart
    k.emit(s_add_u32(s[S_PREFETCH_A+i*2], s[S_KERNARG_A[0]], i * 0x40000))
    k.emit(s_addc_u32(s[S_PREFETCH_A+i*2+1], s[S_KERNARG_A[1]], 0))

  # Global prefetch addresses: B = (tile_x + lane_id) * 4, A = ((tile_y << 12) + (lane_id/8)*4K + lane_id%8) * 4
  k.emit(v_add_nc_u32_e32(v[V_GLOBAL_B_ADDR], s[S_TILE_X], v[V_LANE_ID]))
  k.emit(v_lshlrev_b32_e32(v[V_GLOBAL_B_ADDR], 2, v[V_GLOBAL_B_ADDR]))
  k.emit(s_lshl_b32(s[19], s[S_TILE_Y], 12))
  k.emit(v_lshl_add_u32(v[V_GLOBAL_A_ADDR], v[4], 12, v[V_LANE_ID_MOD8]))  # (lane_id/8)*4K + lane_id%8
  k.emit(v_add_nc_u32_e32(v[V_GLOBAL_A_ADDR], s[19], v[V_GLOBAL_A_ADDR]))
  k.emit(v_lshlrev_b32_e32(v[V_GLOBAL_A_ADDR], 2, v[V_GLOBAL_A_ADDR]))

  # Do initial loads
  for vdst, saddr_lo in INIT_PREFETCH:
    k.emit(global_load_b32(vdst=v[vdst], addr=v[V_GLOBAL_B_ADDR], saddr=s[saddr_lo:saddr_lo+1]))
  for iter in range(6):
    vdst1, vdst2, addr, slo1, slo2 = PREFETCH_LOADS[iter]
    k.emit(global_load_b32(vdst=v[vdst1], addr=v[addr], saddr=s[slo1:slo1+1]))
    k.emit(global_load_b32(vdst=v[vdst2], addr=v[addr], saddr=s[slo2:slo2+1]))

  # ===========================================================================
  # LDS store address computation (bank-conflict-avoiding swizzle)
  # ===========================================================================
  # This section computes LDS store addresses with a swizzle pattern to avoid bank conflicts.
  # The swizzle ensures that threads in the same wavefront write to different LDS banks.
  # Formula: swizzled_addr = base + (lane_id & 7) * LDS_A_STRIDE + swizzle_offset
  # where swizzle_offset depends on (lane_id >> 3) to distribute across banks.
  k.emit(v_add_nc_u32_e32(v[9], s[S_LOOP_CTR], v[22]))  # row 0 base
  k.emit(v_and_b32_e32(v[9], ADDR_MASK, v[9]))
  k.emit(v_sub_nc_u32_e32(v[9], v[22], v[9]))  # row 0 swizzle offset
  k.emit(v_lshlrev_b32_e32(v[9], 2, v[9]))  # * 4
  k.emit(v_mad_u32_u24(v[V_LDS_B_ADDR], LDS_A_STRIDE, v[V_LANE_ID_MOD8], v[9]))

  # For V_LDS_A_BASE and epilogue
  k.emit(v_bfe_u32(v[2], v[V_LANE_ID], 3, 2))  # v[2] = (lane_id >> 3) & 3
  k.emit(v_lshlrev_b32_e32(v[V_LANE_DIV8_X4], 2, v[2]))

  # Compute LDS load/store base addresses for inner loop
  k.emit(v_lshlrev_b32_e32(v[2], 4, v[2]))
  k.emit(v_and_b32_e32(v[3], 0x7F, v[1]))  # simplified from 3 lines
  k.emit(v_lshl_or_b32(v[V_LDS_B_BASE], v[V_LANE_ID_MOD8], 4, LDS_BASE_OFFSET))
  k.emit(v_lshl_add_u32(v[V_LDS_A_ADDR], v[3], 2, LDS_BASE_OFFSET))
  k.emit(v_lshlrev_b32_e32(v[3], 2, v[V_LANE_ID]))
  k.emit(v_and_or_b32(v[V_LDS_A_BASE], 0x180, v[3], v[2]))

  # Do initial stores
  k.waitcnt(vm=0)
  for i in range(4):  # A tile: 8 values via 4 stride64 stores
    k.emit(ds_store_2addr_stride64_b32(addr=v[V_LDS_A_ADDR], data0=v[V_LDS_A_DATA[i*2]], data1=v[V_LDS_A_DATA[i*2+1]], offset0=i*4, offset1=i*4+2))
  for i in range(8):  # B tile: 8 values via 8 scalar stores with 64-byte spacing
    offset = i * 64
    k.emit(ds_store_b32(addr=v[V_LDS_B_ADDR], data0=v[V_LDS_B_DATA[i]], offset0=offset & 0xFF, offset1=offset >> 8))

  # Zero all 128 accumulators using VOPD dual moves (64 instructions instead of 128)
  for i in range(0, len(OUT_REGS), 2):
    k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, vdstx=v[OUT_REGS[i]], vdsty=v[OUT_REGS[i+1]], srcx0=0, srcy0=0))
  k.emit(s_add_i32(s[S_LOOP_BOUND], s[S_DIM_N], -8))

  # S_LOOP_CTR is already 0 from prologue initialization
  k.emit(s_branch(), target='LOOP_ENTRY')

  # ===========================================================================
  # MAIN GEMM LOOP
  # ===========================================================================

  NO_DS, NO_GLOBAL = getenv("NO_DS", 0), getenv("NO_GLOBAL", 0)

  k.label('LOOP_INC')
  k.emit(s_add_i32(s[S_LOOP_CTR], s[S_LOOP_CTR], 8))
  k.emit(s_cmp_ge_i32(s[S_LOOP_CTR], s[S_DIM_N]))
  k.emit(s_cbranch_scc1(), target='EPILOGUE')

  k.label('LOOP_ENTRY')
  k.emit(s_cmp_lt_i32(s[S_LOOP_CTR], s[S_LOOP_BOUND]))
  k.emit(s_cselect_b32(s[S_PREFETCH_FLAG], -1, 0))  # s_cselect doesn't modify SCC
  k.emit(s_cbranch_scc0(), target='SKIP_PREFETCH')  # branch if loop_ctr >= loop_bound

  if not NO_GLOBAL:
    # Advance prefetch pointers (VGPR)
    #k.emit(v_add_nc_u32_e32(v[V_GLOBAL_B_ADDR], 0x20000, v[V_GLOBAL_B_ADDR]))
    #k.emit(v_add_nc_u32_e32(v[V_GLOBAL_A_ADDR], 0x20, v[V_GLOBAL_A_ADDR]))

    # Advance prefetch pointers (64-bit adds)
    k.emit(s_clause(simm16=31))
    for i in range(8):
      k.emit(s_add_u32(s[S_PREFETCH_B+i*2], s[S_PREFETCH_B+i*2], 0x20000))
      k.emit(s_addc_u32(s[S_PREFETCH_B+i*2+1], s[S_PREFETCH_B+i*2+1], 0))
    for i in range(8):
      k.emit(s_add_u32(s[S_PREFETCH_A+i*2], s[S_PREFETCH_A+i*2], 0x20))
      k.emit(s_addc_u32(s[S_PREFETCH_A+i*2+1], s[S_PREFETCH_A+i*2+1], 0))

    # do the fetch
    for vdst, saddr_lo in INIT_PREFETCH:
      k.emit(global_load_b32(vdst=v[vdst], addr=v[V_GLOBAL_B_ADDR], saddr=s[saddr_lo:saddr_lo+1]))

  k.label('SKIP_PREFETCH')

  # wait for local stores to finish (either initial or loop)
  # then sync the warp so it's safe to load local
  k.waitcnt(lgkm=0)
  k.emit(s_barrier())

  # 8 inner loop iterations
  for iter in range(8):
    # Load A tile (4 pairs) and B tile (8 pairs) from LDS
    if not NO_DS:
      k.emit(s_clause(simm16=len(V_A_TILE_REGS) + len(V_B_TILE_REGS) - 1))  # 12 loads total: 4 A + 8 B
      # A tile: 4 ds_load_b64
      for i, vdst in enumerate(V_A_TILE_REGS):
        a_off = (i & 1) * 8 + (i >> 1) * 64 + iter * LDS_A_STRIDE
        k.emit(ds_load_b64(vdst=v[vdst:vdst+1], addr=v[V_LDS_A_BASE], offset0=a_off & 0xFF, offset1=a_off >> 8))
      # B tile: 8 ds_load_b64
      for i, vdst in enumerate(V_B_TILE_REGS):
        b_off = (i & 1) * 8 + (i & 2) * 64 + (i >> 2) * 256 + iter * LDS_B_STRIDE
        k.emit(ds_load_b64(vdst=v[vdst:vdst+1], addr=v[V_LDS_B_BASE], offset0=b_off & 0xFF, offset1=b_off >> 8))

    # Issue global prefetch (first 6 iterations only)
    if iter < 6 and not NO_GLOBAL:
      vdst1, vdst2, addr, slo1, slo2 = PREFETCH_LOADS[iter]
      k.emit(global_load_b32(vdst=v[vdst1], addr=v[addr], saddr=s[slo1:slo1+1]))
      k.emit(global_load_b32(vdst=v[vdst2], addr=v[addr], saddr=s[slo2:slo2+1]))

    # 64 dual FMACs
    k.waitcnt(lgkm=0)
    k.emit(s_clause(simm16=len(FMAC_PATTERN)-1))
    for i, (vdst_x, vdst_y, ax, bx, ay, by) in enumerate(FMAC_PATTERN):
      k.emit(VOPD(VOPDOp.V_DUAL_FMAC_F32, VOPDOp.V_DUAL_FMAC_F32,
                  vdstx=v[vdst_x], vdsty=v[vdst_y], srcx0=v[ax], vsrcx1=v[bx], srcy0=v[ay], vsrcy1=v[by]))

  # wait for all global loads to finish
  # then sync the warp so it's safe to store local
  k.waitcnt(vm=0)
  k.emit(s_barrier())

  # Store prefetched data to LDS
  # NOTE: Register naming reflects LDS tile organization, not source matrix:
  #   V_LDS_A_DATA (v155-162) holds data that goes to LDS A-tile region
  #   V_LDS_B_DATA (v163-170) holds data that goes to LDS B-tile region
  # The data sources are swapped: A-tile receives B matrix rows, B-tile receives A matrix columns
  if not NO_DS:
    for i in range(4):  # A tile: 8 values via 4 stride64 stores
      k.emit(ds_store_2addr_stride64_b32(addr=v[V_LDS_A_ADDR], data0=v[V_LDS_A_DATA[i*2]], data1=v[V_LDS_A_DATA[i*2+1]], offset0=i*4, offset1=i*4+2))
    for i in range(8):  # B tile: 8 values via 8 scalar stores with 64-byte spacing
      offset = i * 64
      k.emit(ds_store_b32(addr=v[V_LDS_B_ADDR], data0=v[V_LDS_B_DATA[i]], offset0=offset & 0xFF, offset1=offset >> 8))

  k.emit(s_branch(), target='LOOP_INC')

  # ===========================================================================
  # EPILOGUE: Permute and store results
  # ===========================================================================
  k.label('EPILOGUE')

  # Rearrange accumulators from FMAC layout to contiguous output order
  for a, b in PERMUTE_SWAPS:
    k.emit(v_swap_b32_e32(v[a], v[b]))

  # Compute output base coordinates
  # v[130] = col_base = tile_x + (lane_id & 7) * 4
  # v[131] = row_base = tile_y + (lane_id & 0x60) + ((lane_id >> 3) & 3) * 4
  # v[132] = 0 (for 64-bit address high part)
  k.emit(v_add_nc_u32_e32(v[130], s[S_TILE_X], v[V_LANE_MOD8_X4]))
  k.emit(v_and_b32_e32(v[131], 0x60, v[V_LANE_ID]))
  k.emit(v_add_nc_u32_e32(v[131], s[S_TILE_Y], v[131]))
  k.emit(v_add_nc_u32_e32(v[131], v[V_LANE_DIV8_X4], v[131]))
  k.emit(v_mov_b32_e32(v[132], 0))

  # Precompute row offsets: v[133-136] for rows 0-3, v[137-140] for rows 16-19
  for base, row_off in [(133, 0), (137, 16)]:
    if row_off: k.emit(v_add_nc_u32_e32(v[141], row_off, v[131]))
    k.emit(v_mul_lo_u32(v[base], v[141] if row_off else v[131], s[S_DIM_N]))
    for j in range(3): k.emit(v_add_nc_u32_e32(v[base + 1 + j], s[S_DIM_N], v[base + j]))

  # s[S_PREFETCH_FLAG] = row stride in bytes (N * 4)
  k.emit(s_lshl_b32(s[S_PREFETCH_FLAG], s[S_DIM_N], 2))

  # Store 128 output values as 32 groups of 4 (128-bit stores)
  # Layout: 2 row halves (0-3, 16-19) x 4 col groups x 4 rows = 32 stores of 4 floats
  for i, (row_half, col_off, row_in_group) in enumerate([(rh, co, ri)
      for rh in range(2) for co in [0, 32, 64, 96] for ri in range(4)]):
    row = row_half * 16 + row_in_group
    src = OUT_REGS[i*4]  # first reg of ascending group of 4

    if row_in_group == 0:
      # First row of group: compute full address
      if col_off == 0: k.emit(v_mov_b32_e32(v[141], v[130]))
      else: k.emit(v_add_nc_u32_e32(v[141], col_off, v[130]))
      row_base = 133 + row if row < 4 else 137 + row - 16
      k.emit(v_add_nc_u32_e32(v[141], v[row_base], v[141]))
      k.emit(v_lshlrev_b32_e32(v[141], 2, v[141]))
      k.emit(v_add_co_u32(v[141], VCC_LO, s[S_OUT_PTR[0]], v[141]))
      k.emit(v_add_co_ci_u32_e32(v[142], s[S_OUT_PTR[1]], v[132]))
    else:
      # Subsequent rows: add stride
      k.emit(v_add_co_u32(v[141], VCC_LO, s[S_PREFETCH_FLAG], v[141]))
      k.emit(v_add_co_ci_u32_e32(v[142], v[142], v[132]))

    k.emit(global_store_b128(addr=v[141:142], data=v[src:src+3], saddr=NULL))

  k.emit(s_sendmsg(simm16=3))  # DEALLOC_VGPRS
  k.emit(s_endpgm())

  return k.to_asm()

# =============================================================================
# Test harness
# =============================================================================

N = getenv("N", 4096)
BLOCK_M, BLOCK_N = 128, 128
THREADS = 128

def test_matmul():
  dev = Device[Device.DEFAULT]
  print(f"Device arch: {dev.renderer.arch}")

  if getenv("STOCK", 0):
    # Load the stock kernel from amd_seb/kernel8_batched_gmem.s
    stock_path = Path(__file__).parent / "amd_seb" / "kernel8_batched_gmem.s"
    asm = stock_path.read_text()
    print(f"Loaded stock kernel from {stock_path}")
  else:
    asm = build_kernel(dev.renderer.arch)

  binary = dev.compiler.compile(asm)
  print(f"Compiled! Binary size: {len(binary)} bytes")

  rng = np.random.default_rng(42)
  a = Tensor(rng.random((N, N), dtype=np.float32) - 0.5)
  b = Tensor(rng.random((N, N), dtype=np.float32) - 0.5)
  c = Tensor.empty(N, N)
  Tensor.realize(a, b, c)

  grid, local = (N // BLOCK_N, N // BLOCK_M, 1), (THREADS, 1, 1)
  print(f"Grid: {grid}, Local: {local}")

  dname:str = Device.DEFAULT
  def asm_kernel(A:UOp, B:UOp, C:UOp) -> UOp:
    gidxs = [UOp.special(n, f"gidx{i}") for i,n in enumerate(grid)]
    lidxs = [UOp.special(n, f"lidx{i}") for i,n in enumerate(local)]
    sink = UOp.sink(A.base, B.base, C.base, *gidxs, *lidxs, arg=KernelInfo(name=colored("kernel", "cyan"),
                                                                           estimates=Estimates(ops=N*N*N*2, mem=N*N*4*3)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=asm),
                                 UOp(Ops.BINARY, arg=binary)))
  c = Tensor.custom_kernel(a, b, c, fxn=asm_kernel)[2]
  ei = c.schedule()[0].lower()

  ets = []
  with Context(DEBUG=2):
    for _ in range(getenv("CNT", 5)): ets.append(ei.run(wait=True))
  print(f"REAL TFLOPS {N * N * N * 2 / min(ets) * 1e-12:.2f}")

  if getenv("VERIFY", 1):
    GlobalCounters.reset()
    with Context(DEBUG=2): tc = (a @ b).realize()
    with Context(DEBUG=0): err = (c - tc).square().mean().item()
    print(f"mean squared error {err}")
    if err != err or err > 1e-06: raise RuntimeError("matmul is wrong!")

def run_sqtt():
  """Run with SQTT profiling and write trace files."""
  import subprocess, os

  # Run test_matmul in a subprocess with SQTT enabled from the start (no verify)
  env = {**os.environ, "AMD": "1", "SQTT": "1", "CNT": "1", "PROFILE": "1", "PYTHONPATH": ".", "VERIFY": "0"}
  result = subprocess.run(
    ["python", "-c", "from extra.gemm.amd_asm_matmul import test_matmul; test_matmul()"],
    capture_output=True, text=True, env=env, timeout=120
  )
  print(result.stdout)

  # Run roc.py to extract trace data
  result = subprocess.run(
    ["python", "extra/sqtt/roc.py", "--profile", "/tmp/profile.pkl.tiny", "--kernel", "kernel"],
    capture_output=True, text=True, env={**os.environ, "DEBUG": "5"}, timeout=60
  )
  output = result.stdout + result.stderr

  # Write full output to trace file
  with open("/tmp/sqtt_trace.txt", "w") as f:
    f.write(output)
  print(f"Wrote {len(output)} bytes to /tmp/sqtt_trace.txt")

if __name__ == "__main__":
  if getenv("ASM", 0): print(build_kernel(Device[Device.DEFAULT].arch))
  elif getenv("SQTT", 0): run_sqtt()
  else: test_matmul()
