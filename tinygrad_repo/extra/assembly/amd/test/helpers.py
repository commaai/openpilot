"""Shared test helpers for RDNA3 tests."""
import shutil
from dataclasses import dataclass

@dataclass
class KernelInfo:
  code: bytes
  src: str
  global_size: tuple[int, int, int]
  local_size: tuple[int, int, int]
  buf_idxs: list[int]  # indices into shared buffer pool
  buf_sizes: list[int]  # sizes for each buffer index

# LLVM tool detection (shared across test files)
def get_llvm_mc():
  """Find llvm-mc executable, preferring newer versions."""
  for p in ['llvm-mc', 'llvm-mc-21', 'llvm-mc-20']:
    if shutil.which(p): return p
  raise FileNotFoundError("llvm-mc not found")

def get_llvm_objdump():
  """Find llvm-objdump executable, preferring newer versions."""
  for p in ['llvm-objdump', 'llvm-objdump-21', 'llvm-objdump-20']:
    if shutil.which(p): return p
  raise FileNotFoundError("llvm-objdump not found")

ARCH_TO_TARGET:dict[str, list[str]] = {
  "rdna3":["gfx1100"],
  "rdna4":["gfx1200"],
  "cdna":["gfx950", "gfx942"],
}

TARGET_TO_ARCH:dict[str, str] = {t:arch for arch,targets in ARCH_TO_TARGET.items() for t in targets}

def get_target(arch:str) -> str: return ARCH_TO_TARGET[arch][0]

def get_mattr(arch:str) -> str:
  return {"rdna3":"+real-true16,+wavefrontsize32", "rdna4":"+real-true16,+wavefrontsize32", "cdna":"+wavefrontsize64"}[arch]