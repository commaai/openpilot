import struct, tinygrad.runtime.autogen.libc as libc
from dataclasses import dataclass
from tinygrad.helpers import getbits, i2u

@dataclass(frozen=True)
class ElfSection: name:str; header:libc.Elf64_Shdr; content:bytes # noqa: E702

def elf_loader(blob:bytes, force_section_align:int=1) -> tuple[memoryview, list[ElfSection], list[tuple]]:
  def _strtab(blob: bytes, idx: int) -> str: return blob[idx:blob.find(b'\x00', idx)].decode('utf-8')

  header = libc.Elf64_Ehdr.from_buffer_copy(blob)
  section_headers = (libc.Elf64_Shdr * header.e_shnum).from_buffer_copy(blob[header.e_shoff:])
  sh_strtab = blob[(shstrst:=section_headers[header.e_shstrndx].sh_offset):shstrst+section_headers[header.e_shstrndx].sh_size]
  sections = [ElfSection(_strtab(sh_strtab, sh.sh_name), sh, blob[sh.sh_offset:sh.sh_offset+sh.sh_size]) for sh in section_headers]

  def _to_carray(sh, ctype): return (ctype * (sh.header.sh_size // sh.header.sh_entsize)).from_buffer_copy(sh.content)
  rel = [(sh, sh.name[4:], _to_carray(sh, libc.Elf64_Rel)) for sh in sections if sh.header.sh_type == libc.SHT_REL]
  rela = [(sh, sh.name[5:], _to_carray(sh, libc.Elf64_Rela)) for sh in sections if sh.header.sh_type == libc.SHT_RELA]
  symtab = [_to_carray(sh, libc.Elf64_Sym) for sh in sections if sh.header.sh_type == libc.SHT_SYMTAB][0]
  progbits = [sh for sh in sections if sh.header.sh_type == libc.SHT_PROGBITS]

  # Prealloc image for all fixed addresses.
  image = bytearray(max([sh.header.sh_addr + sh.header.sh_size for sh in progbits if sh.header.sh_addr != 0] + [0]))
  for sh in progbits:
    if sh.header.sh_addr != 0: image[sh.header.sh_addr:sh.header.sh_addr+sh.header.sh_size] = sh.content
    else:
      image += b'\0' * (((align:=max(sh.header.sh_addralign, force_section_align)) - len(image) % align) % align) + sh.content
      sh.header.sh_addr = len(image) - len(sh.content)

  # Relocations
  relocs = []
  for sh, trgt_sh_name, c_rels in rel + rela:
    target_image_off = next(tsh for tsh in sections if tsh.name == trgt_sh_name).header.sh_addr
    rels = [(r.r_offset, symtab[libc.ELF64_R_SYM(r.r_info)], libc.ELF64_R_TYPE(r.r_info), getattr(r, "r_addend", 0)) for r in c_rels]
    for roff, sym, r_type_, r_addend in rels:
      if sym.st_shndx == 0: raise RuntimeError(f'Attempting to relocate against an undefined symbol {repr(_strtab(sh_strtab, sym.st_name))}')
    relocs += [(target_image_off + roff, sections[sym.st_shndx].header.sh_addr + sym.st_value, rtype, raddend) for roff, sym, rtype, raddend in rels]

  return memoryview(image), sections, relocs

def relocate(instr: int, ploc: int, tgt: int, r_type: int):
  match r_type:
    # https://refspecs.linuxfoundation.org/elf/x86_64-abi-0.95.pdf
    case libc.R_X86_64_PC32: return i2u(32, tgt-ploc)
    # https://github.com/ARM-software/abi-aa/blob/main/aaelf64/aaelf64.rst for definitions of relocations
    # https://www.scs.stanford.edu/~zyedidia/arm64/index.html for instruction encodings
    case libc.R_AARCH64_ADR_PREL_PG_HI21:
      rel_pg = (tgt & ~0xFFF) - (ploc & ~0xFFF)
      return instr | (getbits(rel_pg, 12, 13) << 29) | (getbits(rel_pg, 14, 32) << 5)
    case libc.R_AARCH64_ADD_ABS_LO12_NC: return instr | (getbits(tgt, 0, 11) << 10)
    case libc.R_AARCH64_LDST16_ABS_LO12_NC: return instr | (getbits(tgt, 1, 11) << 10)
    case libc.R_AARCH64_LDST32_ABS_LO12_NC: return instr | (getbits(tgt, 2, 11) << 10)
    case libc.R_AARCH64_LDST64_ABS_LO12_NC: return instr | (getbits(tgt, 3, 11) << 10)
    case libc.R_AARCH64_LDST128_ABS_LO12_NC: return instr | (getbits(tgt, 4, 11) << 10)
  raise NotImplementedError(f"Encountered unknown relocation type {r_type}")

def jit_loader(obj: bytes) -> bytes:
  image, _, relocs = elf_loader(obj)
  # This is needed because we have an object file, not a .so that has all internal references (like loads of constants from .rodata) resolved.
  for ploc,tgt,r_type,r_addend in relocs:
    image[ploc:ploc+4] = struct.pack("<I", relocate(struct.unpack("<I", image[ploc:ploc+4])[0], ploc, tgt+r_addend, r_type))
  return bytes(image)
