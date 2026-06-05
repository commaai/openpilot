# minimal amdgpu elf packer
import ctypes
from tinygrad.helpers import ceildiv, round_up
from tinygrad.uop.ops import UOp, Ops
from tinygrad.runtime.autogen import amdgpu_kd, hsa, libc
from tinygrad.renderer.amd.dsl import Reg, FixedBitField
from tinygrad.runtime.autogen.amd.common import OpType

# instructions used for padding
from tinygrad.runtime.autogen.amd.rdna3.ins import s_code_end # same encoding as RDNA4
from tinygrad.runtime.autogen.amd.cdna.ins import s_nop as s_nop_cdna

_arch_map = {"gfx9": "cdna", "gfx10": "rdna3", "gfx11": "rdna3", "gfx12": "rdna4"}
def assemble_linear(prg:UOp, lin:UOp, arch:str) -> bytes:
  insts = [u.arg for u in lin.src]

  # ** scan for max vgpr/sgpr/accvgpr
  max_vgpr, max_sgpr, max_accvgpr = 0, 0, 0
  _ACCVGPR_TYPES = {OpType.OPR_ACCVGPR, OpType.OPR_SRC_ACCVGPR}
  for inst in insts:
    # build set of field names that are AccVGPR for this instruction
    accvgpr_fields: set[str] = set()
    for opr_name, (_, _, opr_type) in inst.operands.items():
      if opr_type in _ACCVGPR_TYPES: accvgpr_fields.add(opr_name)
      elif opr_type in {OpType.OPR_VGPR_OR_ACCVGPR, OpType.OPR_SRC_VGPR_OR_ACCVGPR, OpType.OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST}:
        if getattr(inst, 'acc_cd', 0) == 1: accvgpr_fields.add(opr_name)
    for name, field in inst._fields:
      if isinstance(field, FixedBitField): continue
      val = getattr(inst, name)
      if not isinstance(val, Reg): continue
      if 256 <= val.offset < 512:
        if name in accvgpr_fields: max_accvgpr = max(max_accvgpr, (val.offset - 256) + val.sz)
        else: max_vgpr = max(max_vgpr, (val.offset - 256) + val.sz)
      elif val.offset < 106: max_sgpr = max(max_sgpr, val.offset + val.sz)

  # ** scan sink for metadata
  sink, n_bufs, n_vars, lds_size, gids = prg.src[0], 0, 0, 0, set()
  for u in sink.toposort():
    if u.op is Ops.PARAM: n_bufs += 1
    elif u.op is Ops.DEFINE_VAR: n_vars += 1
    elif u.op is Ops.DEFINE_LOCAL: lds_size += u.ptrdtype.size * u.ptrdtype.base.itemsize
    elif u.op is Ops.SPECIAL and u.arg.startswith("gidx"): gids.add(int(u.arg[-1]))
  code_bytes = b"".join(inst.to_bytes() for inst in insts)
  arch = next(v for k, v in _arch_map.items() if arch.startswith(k))
  is_cdna, is_rdna4 = arch == "cdna", arch == "rdna4"

  # ** pad text to ISA alignment
  padding_inst = (s_nop_cdna(0) if is_cdna else s_code_end()).to_bytes()
  text = code_bytes + padding_inst * ((hsa.AMD_ISA_ALIGN_BYTES - len(code_bytes) % hsa.AMD_ISA_ALIGN_BYTES) % hsa.AMD_ISA_ALIGN_BYTES)
  text_offset = round_up(ctypes.sizeof(libc.Elf64_Ehdr), hsa.AMD_ISA_ALIGN_BYTES)

  # ** pack kernel descriptor (rodata)
  # CDNA: total VGPRs = regular VGPRs + AccVGPRs, each rounded to granularity of 4
  accum_offset = round_up(max_vgpr, 4) if max_accvgpr > 0 else 0
  next_free_vgpr = round_up(accum_offset + max_accvgpr, 8) if max_accvgpr > 0 else round_up(max_vgpr, 8)
  next_free_sgpr = round_up(max_sgpr, 8)
  vgpr_granule = max(0, (next_free_vgpr + 7) // 8 - 1)
  # CDNA: add 6 for VCC(2) + FLAT_SCRATCH(2) + XNACK_MASK(2), next_free_sgpr is unused in RDNA.
  sgpr_granule = max(0, ceildiv(next_free_sgpr + 6, 8) - 1) if is_cdna else 0
  desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t()
  desc.group_segment_fixed_size = lds_size
  desc.kernarg_size = n_bufs * 8 + n_vars * 4
  desc.kernel_code_entry_byte_offset = -len(text)

  # https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-compute-pgm-rsrc1-gfx6-gfx12-table
  # NOTE: CU mode is the default
  desc.compute_pgm_rsrc1 = (vgpr_granule << amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT |
                            sgpr_granule << amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT |
                            3 << amdgpu_kd.COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64_SHIFT |
                            (0 if is_rdna4 else 1) << amdgpu_kd.COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_DX10_CLAMP_SHIFT |
                            (0 if is_rdna4 else 1) << amdgpu_kd.COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_IEEE_MODE_SHIFT |
                            (0 if is_cdna else 1) << amdgpu_kd.COMPUTE_PGM_RSRC1_GFX10_PLUS_MEM_ORDERED_SHIFT)
  desc.compute_pgm_rsrc2 = (2 << amdgpu_kd.COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT |
                            int(0 in gids) << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT |
                            int(1 in gids) << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y_SHIFT |
                            int(2 in gids) << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z_SHIFT)
  desc.kernel_code_properties = (1 << amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT |
                                 (0 if is_cdna else 1) << amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32_SHIFT)
  if is_cdna and max_accvgpr > 0:
    desc.compute_pgm_rsrc3 = max(0, accum_offset // 4 - 1) << amdgpu_kd.COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET_SHIFT
  rodata = bytes(desc)

  # ** pack ELF
  sh_names:list[int] = []
  strtab = bytearray(b"\x00")
  for name in [".text", ".rodata", ".strtab"]:
    sh_names.append(len(strtab))
    strtab += name.encode("ascii") + b"\x00"

  rodata_offset = round_up(text_offset + (text_size := len(text)), hsa.AMD_KERNEL_CODE_ALIGN_BYTES)
  strtab_offset = rodata_offset + (rodata_size := len(rodata))
  shdr_offset   = strtab_offset + (strtab_size := len(strtab))

  sections = [(libc.SHT_PROGBITS, libc.SHF_ALLOC | libc.SHF_EXECINSTR, text_offset, text_offset, text_size),
              (libc.SHT_PROGBITS, libc.SHF_ALLOC, rodata_offset, rodata_offset, rodata_size),
              (libc.SHT_STRTAB, 0, 0, strtab_offset, strtab_size)]
  shdrs = (libc.Elf64_Shdr * len(sections))()
  for i, s in enumerate(sections): shdrs[i] = libc.Elf64_Shdr(sh_names[i], *s)

  ehdr = libc.Elf64_Ehdr()
  ehdr.e_ident[:5], ehdr.e_shoff, ehdr.e_shnum, ehdr.e_shstrndx = b"\x7FELF\x02", shdr_offset, len(sections), 2

  elf = bytearray(shdr_offset + ctypes.sizeof(shdrs))
  elf[0:ctypes.sizeof(ehdr)] = bytes(ehdr)
  elf[text_offset:text_offset+text_size] = text
  elf[rodata_offset:rodata_offset+rodata_size] = rodata
  elf[strtab_offset:strtab_offset+strtab_size] = strtab
  elf[shdr_offset:shdr_offset+ctypes.sizeof(shdrs)] = bytes(shdrs)
  binary = bytes(elf)

  return binary
