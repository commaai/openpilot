"""Shared test helpers for AMD tests."""
import ctypes
from tinygrad.helpers import unwrap
from tinygrad.runtime.autogen import llvm
from tinygrad.runtime.support.elf import elf_loader

ARCH_TO_TARGET:dict[str, list[str]] = {
  "rdna3":["gfx1100", "gfx1151"],
  "rdna4":["gfx1200", "gfx1201"],
  "cdna":["gfx950", "gfx942"],
}

TARGET_TO_ARCH:dict[str, str] = {t:arch for arch,targets in ARCH_TO_TARGET.items() for t in targets}

_DPP16_RANGE_OPS = {0x100: "row_shl", 0x110: "row_shr", 0x120: "row_ror", 0x150: "row_newbcast", 0x160: "row_share", 0x170: "row_xmask"}
_DPP16_EXACT_OPS = {0x130: ("wave_shl", 1), 0x134: ("wave_rol", 1), 0x138: ("wave_shr", 1), 0x13c: ("wave_ror", 1),
                    0x140: ("row_mirror", 0), 0x141: ("row_half_mirror", 0), 0x142: ("row_bcast", 15), 0x143: ("row_bcast", 31)}

def get_target(arch:str) -> str: return ARCH_TO_TARGET[arch][0]

def decode_dpp16(dpp: int) -> tuple[str, int | tuple[int, int, int, int]]:
  """Decode a DPP16 control word into a symbolic operation and argument."""
  if dpp < 0x100: return "quad_perm", ((dpp >> 0) & 0x3, (dpp >> 2) & 0x3, (dpp >> 4) & 0x3, (dpp >> 6) & 0x3)
  if dpp in _DPP16_EXACT_OPS: return _DPP16_EXACT_OPS[dpp]
  if (base := dpp & 0x1f0) in _DPP16_RANGE_OPS: return _DPP16_RANGE_OPS[base], dpp & 0xf
  return "dpp", dpp

def get_mattr(arch:str) -> str:
  return {"rdna3":"+real-true16,+wavefrontsize32", "rdna4":"+real-true16,+wavefrontsize32", "cdna":"+wavefrontsize64"}[arch]

# LLVM in-process assembler/disassembler (replaces llvm-mc and llvm-objdump subprocesses)
_SENTINEL = b'\xde\xad\xbe\xef'
_SENTINEL_ASM = '.byte 0xde, 0xad, 0xbe, 0xef'

def _cerr(): return ctypes.pointer(ctypes.pointer(ctypes.c_char()))
def _expect(x, err, ret=None):
  if x: raise RuntimeError(unwrap(ctypes.cast(err.contents, ctypes.c_char_p).value).decode() if not isinstance(err, str) else err)
  return ret

def _init_llvm():
  for component in ['Target', 'TargetInfo', 'TargetMC', 'AsmParser', 'AsmPrinter', 'Disassembler']:
    getattr(llvm, f'LLVMInitializeAMDGPU{component}')()

def _create_target_machine(mcpu:str, mattr:str) -> llvm.LLVMTargetMachineRef:
  target = _expect(llvm.LLVMGetTargetFromTriple(b'amdgcn-amd-amdhsa', ctypes.pointer(tgt:=llvm.LLVMTargetRef()), err:=_cerr()), err, tgt)
  return llvm.LLVMCreateTargetMachine(target, b'amdgcn-amd-amdhsa', mcpu.encode(), mattr.encode(),
                                      llvm.LLVMCodeGenLevelDefault, llvm.LLVMRelocDefault, llvm.LLVMCodeModelDefault)

def _emit_obj(asm_text:str, mcpu:str, mattr:str, diag_errors:list[str]|None=None) -> bytes:
  """Assemble raw asm text into an ELF object using LLVM in-process."""
  _init_llvm()
  tm = _create_target_machine(mcpu, mattr)
  ctx = llvm.LLVMContextCreate()
  try:
    errors = diag_errors if diag_errors is not None else []
    @llvm.LLVMDiagnosticHandler
    def handle_diag(diag_ref, _arg):
      if llvm.LLVMGetDiagInfoSeverity(diag_ref) == llvm.LLVMDSError:
        errors.append(ctypes.string_at(llvm.LLVMGetDiagInfoDescription(diag_ref)).decode())
    llvm.LLVMContextSetDiagnosticHandler(ctx, handle_diag, None)
    mod = llvm.LLVMModuleCreateWithNameInContext(b'asm', ctx)
    llvm.LLVMSetTarget(mod, b'amdgcn-amd-amdhsa')
    asm_bytes = asm_text.encode()
    llvm.LLVMSetModuleInlineAsm2(mod, asm_bytes, len(asm_bytes))
    buf = llvm.LLVMMemoryBufferRef()
    _expect(llvm.LLVMTargetMachineEmitToMemoryBuffer(tm, mod, llvm.LLVMObjectFile, err:=_cerr(), ctypes.pointer(buf)), err)
    obj = ctypes.string_at(llvm.LLVMGetBufferStart(buf), llvm.LLVMGetBufferSize(buf))
    llvm.LLVMDisposeMemoryBuffer(buf)
    llvm.LLVMDisposeModule(mod)
    return obj
  finally:
    llvm.LLVMContextDispose(ctx)
    llvm.LLVMDisposeTargetMachine(tm)

def _extract_text(obj:bytes) -> bytes:
  """Extract .text section from ELF object bytes."""
  return next(s.content for s in elf_loader(obj)[1] if s.name == ".text")

def llvm_assemble(instrs:list[str], mcpu:str, mattr:str) -> list[bytes]:
  """Assemble instructions in one LLVM emission, return per-instruction bytes."""
  if not instrs: return []
  parts = []
  for instr in instrs:
    parts.append(instr)
    parts.append(_SENTINEL_ASM)
  text = _extract_text(_emit_obj('.text\n' + '\n'.join(parts) + '\n', mcpu, mattr))
  results, start = [], 0
  for _ in instrs:
    idx = text.find(_SENTINEL, start)
    assert idx != -1, "sentinel not found in .text section"
    results.append(bytes(text[start:idx]))
    start = idx + len(_SENTINEL)
  return results

def llvm_disasm(code:bytes, mcpu:str, mattr:str) -> list[str]:
  """Disassemble raw bytes into instruction strings using LLVM."""
  _init_llvm()
  dc = llvm.LLVMCreateDisasmCPUFeatures(b'amdgcn-amd-amdhsa', mcpu.encode(), mattr.encode(), None, 0,
                                         llvm.LLVMOpInfoCallback(0), llvm.LLVMSymbolLookupCallback(0))
  if not dc: raise RuntimeError(f"failed to create disasm context for {mcpu}")
  llvm.LLVMSetDisasmOptions(dc, 2 | 4)  # PrintImmHex | AsmPrinterVariant
  try:
    buf = ctypes.create_string_buffer(256)
    arr = (ctypes.c_uint8 * len(code)).from_buffer_copy(code)
    results, offset = [], 0
    while offset < len(code):
      size = llvm.LLVMDisasmInstruction(dc, ctypes.cast(ctypes.addressof(arr) + offset, ctypes.POINTER(ctypes.c_uint8)),
                                        len(code) - offset, 0, buf, 256)
      if size == 0: break
      results.append(buf.value.decode().strip())
      offset += size
    return results
  finally:
    llvm.LLVMDisasmDispose(dc)

def llvm_filter_valid_asm(tests:list[tuple[str, bytes]], mcpu:str, mattr:str) -> list[tuple[str, bytes]]:
  """Filter out tests where original ASM isn't valid on target, and where LLVM roundtrip doesn't match."""
  if not tests: return []
  # Assemble all instructions at once with sentinels and diagnostic handler to detect failures
  parts, diag_errors = [], []  # type: ignore[var-annotated]
  for asm, _ in tests:
    parts.append(asm)
    parts.append(_SENTINEL_ASM)
  text = _extract_text(_emit_obj('.text\n' + '\n'.join(parts) + '\n', mcpu, mattr, diag_errors))
  results, start = [], 0
  for _ in tests:
    idx = text.find(_SENTINEL, start)
    assert idx != -1, "sentinel not found in .text section"
    results.append(bytes(text[start:idx]))
    start = idx + len(_SENTINEL)
  # Invalid instructions produce 0 bytes; also filter where LLVM roundtrip doesn't match original
  return [(asm, data) for (asm, data), chunk in zip(tests, results) if len(chunk) > 0 and chunk == data]
