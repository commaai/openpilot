# unpack the complete kernel descriptor of an amdgpu ELF
# https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html#code-object-v3-kernel-descriptor
import struct, pathlib, sys
from tinygrad.runtime.support.elf import elf_loader

def bits(x, lo, hi): return (x >> lo) & ((1 << (hi - lo + 1)) - 1)
def assert_zero(x, lo, hi): assert bits(x, lo, hi) == 0

with open(sys.argv[1], "rb") as f:
  lib = f.read()

image, sections, relocs = elf_loader(lib)
rodata_entry = next((sh.header.sh_addr for sh in sections if sh.name == ".rodata"))

# rodata is exactly 64 bytes
kd = image[rodata_entry:rodata_entry+64]
desc = int.from_bytes(kd, byteorder="little")

group_segment_fixed_size = bits(desc, 0, 31)
private_segment_fixed_size = bits(desc, 32, 63)
kernarg_size = bits(desc, 64, 95)
reserved_127_96 = bits(desc, 96, 127)
assert reserved_127_96 == 0

print("GROUP_SEGMENT_FIXED_SIZE:", group_segment_fixed_size)
print("PRIVATE_SEGMENT_FIXED_SIZE:", private_segment_fixed_size)
print("KERNARG_SIZE:", kernarg_size)
print("RESERVED 127:96:", reserved_127_96)

entry_off = bits(desc, 128, 191)

# sign-extend manually if needed
if entry_off & (1 << 63):
  entry_off -= 1 << 64

print("KERNEL_CODE_ENTRY_BYTE_OFFSET:", entry_off)

kd_addr = 0x1840
entry_addr = kd_addr + entry_off

print("Computed entry address: 0x%016x" % entry_addr)
print("256B aligned:", entry_addr % 256 == 0)

pgm_rsrc3 = bits(desc, 352, 383)
pgm_rsrc1 = bits(desc, 384, 415)
pgm_rsrc2 = bits(desc, 416, 447)

print("COMPUTE_PGM_RSRC3: 0x%08x" % pgm_rsrc3)
print("COMPUTE_PGM_RSRC1: 0x%08x" % pgm_rsrc1)
print("COMPUTE_PGM_RSRC2: 0x%08x" % pgm_rsrc2)

# rsrc 3 (gfx950)

accum_offset_raw = bits(pgm_rsrc3, 0, 5)
assert_zero(pgm_rsrc3, 6, 15)
tg_split = bits(pgm_rsrc3, 16, 16)
accum_offset_vgprs = (accum_offset_raw + 1) * 4
print("RSRC3.ACCUM_OFFSET (AccVGPR index):", accum_offset_vgprs)
print("RSRC3.TG_SPLIT:", tg_split)

# rsrc 1

vgpr_gran = bits(pgm_rsrc1, 0, 5)
sgpr_gran = bits(pgm_rsrc1, 6, 9)
assert_zero(pgm_rsrc1, 27, 28)

# NOTE: this is vgprs + agprs
vgprs_used = (vgpr_gran + 1) * 8
assert 0 <= vgprs_used <= 512

k = sgpr_gran // 2
sgprs_used = (k + 1) * 16

print("RSRC1.VGPRS:", vgprs_used)
print("RSRC1.SGPRS:", sgprs_used)

assert_zero(pgm_rsrc1, 10, 11)

float_round_mode_32 = bits(pgm_rsrc1, 12, 13)
float_round_mode_16_64 = bits(pgm_rsrc1, 15, 14)
float_denorm_mode_32 = bits(pgm_rsrc1, 16, 17)
float_denorm_mode_16_64 = bits(pgm_rsrc1, 18, 19)

priv = bits(pgm_rsrc1, 20, 20)
assert priv == 0
enable_dx10_clamp_wg_rr_en = bits(pgm_rsrc1, 21, 21)
debug_mode = bits(pgm_rsrc1, 22, 22)
enable_ieee_mode = bits(pgm_rsrc1, 23, 23)
bulky = bits(pgm_rsrc1, 24, 24)
assert bulky == 0
cdbg_user = bits(pgm_rsrc1, 25, 25)
assert cdbg_user == 0
fp16_ovfl = bits(pgm_rsrc1, 26, 26)
assert_zero(pgm_rsrc1, 27, 28)  # reserved
assert_zero(pgm_rsrc1, 29, 29)  # WGP_MODE (reserved on gfx9)
assert_zero(pgm_rsrc1, 30, 30)  # MEM_ORDERED (reserved on gfx9)
assert_zero(pgm_rsrc1, 31, 31)  # FWD_PROGRESS (reserved on gfx9)

# rsrc 2

enable_private_segment = bits(pgm_rsrc2, 0, 0)  # SCRATCH_EN
user_sgpr_count = bits(pgm_rsrc2, 1, 5)         # USER_SGPR
enable_trap_handler = bits(pgm_rsrc2, 6, 6)     # TRAP_PRESENT (must be 0 here)
assert enable_trap_handler == 0

enable_sgpr_workgroup_id_x = bits(pgm_rsrc2, 7, 7)
enable_sgpr_workgroup_id_y = bits(pgm_rsrc2, 8, 8)
enable_sgpr_workgroup_id_z = bits(pgm_rsrc2, 9, 9)
enable_sgpr_workgroup_info = bits(pgm_rsrc2, 10, 10)

enable_vgpr_workitem_id = bits(pgm_rsrc2, 11, 12)  # TIDIG_CMP_CNT enum (0..3)

enable_exception_address_watch = bits(pgm_rsrc2, 13, 13)
assert enable_exception_address_watch == 0
enable_exception_memory = bits(pgm_rsrc2, 14, 14)
assert enable_exception_memory == 0

granulated_lds_size = bits(pgm_rsrc2, 15, 23)
assert granulated_lds_size == 0  # spec: must be 0; CP uses dispatch packet rounding

enable_exception_fp_invalid = bits(pgm_rsrc2, 24, 24)
enable_exception_fp_denorm_src = bits(pgm_rsrc2, 25, 25)
enable_exception_fp_div0 = bits(pgm_rsrc2, 26, 26)
enable_exception_fp_overflow = bits(pgm_rsrc2, 27, 27)
enable_exception_fp_underflow = bits(pgm_rsrc2, 28, 28)
enable_exception_fp_inexact = bits(pgm_rsrc2, 29, 29)
enable_exception_int_div0 = bits(pgm_rsrc2, 30, 30)

assert_zero(pgm_rsrc2, 31, 31)

print("RSRC2.ENABLE_PRIVATE_SEGMENT:", enable_private_segment)
print("RSRC2.USER_SGPR_COUNT:", user_sgpr_count)
print("RSRC2.ENABLE_SGPR_WORKGROUP_ID_X:", enable_sgpr_workgroup_id_x)
print("RSRC2.ENABLE_SGPR_WORKGROUP_ID_Y:", enable_sgpr_workgroup_id_y)
print("RSRC2.ENABLE_SGPR_WORKGROUP_ID_Z:", enable_sgpr_workgroup_id_z)
print("RSRC2.ENABLE_SGPR_WORKGROUP_INFO:", enable_sgpr_workgroup_info)
print("RSRC2.ENABLE_VGPR_WORKITEM_ID (enum):", enable_vgpr_workitem_id)

print("RSRC2.EXC_FP_INVALID:", enable_exception_fp_invalid)
print("RSRC2.EXC_FP_DENORM_SRC:", enable_exception_fp_denorm_src)
print("RSRC2.EXC_FP_DIV0:", enable_exception_fp_div0)
print("RSRC2.EXC_FP_OVERFLOW:", enable_exception_fp_overflow)
print("RSRC2.EXC_FP_UNDERFLOW:", enable_exception_fp_underflow)
print("RSRC2.EXC_FP_INEXACT:", enable_exception_fp_inexact)
print("RSRC2.EXC_INT_DIV0:", enable_exception_int_div0)

# user sgprs

enable_sgpr_private_segment_buffer = bits(desc, 448, 448)
enable_sgpr_dispatch_ptr = bits(desc, 449, 449)
enable_sgpr_queue_ptr = bits(desc, 450, 450)
enable_sgpr_kernarg_segment_ptr = bits(desc, 451, 451)
enable_sgpr_dispatch_id = bits(desc, 452, 452)
enable_sgpr_flat_scratch_init = bits(desc, 453, 453)
enable_sgpr_private_segment_size = bits(desc, 454, 454)

assert_zero(desc, 455, 457)

print("DESC.ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER:", enable_sgpr_private_segment_buffer)
print("DESC.ENABLE_SGPR_DISPATCH_PTR:", enable_sgpr_dispatch_ptr)
print("DESC.ENABLE_SGPR_QUEUE_PTR:", enable_sgpr_queue_ptr)
print("DESC.ENABLE_SGPR_KERNARG_SEGMENT_PTR:", enable_sgpr_kernarg_segment_ptr)
print("DESC.ENABLE_SGPR_DISPATCH_ID:", enable_sgpr_dispatch_id)
print("DESC.ENABLE_SGPR_FLAT_SCRATCH_INIT:", enable_sgpr_flat_scratch_init)
print("DESC.ENABLE_SGPR_PRIVATE_SEGMENT_SIZE:", enable_sgpr_private_segment_size)

assert_zero(desc, 458, 459)

uses_dynamic_stack = bits(desc, 459, 460)
print("DESC.USES_DYNAMIC_STACK:", uses_dynamic_stack)

# gfx950 only
assert_zero(desc, 460, 463)
kernarg_preload_spec_length = bits(desc, 464, 470)
print("DESC.KERNARG_PRELOAD_SPEC_LENGTH:", kernarg_preload_spec_length)
kernarg_preload_spec_offset = bits(desc, 471, 479)
print("DESC.KERNARG_PRELOAD_SPEC_OFFSET:", kernarg_preload_spec_offset)

assert_zero(desc, 480, 511)
