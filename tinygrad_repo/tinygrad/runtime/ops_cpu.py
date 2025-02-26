import platform, subprocess, sys
from tinygrad.helpers import capstone_flatdump, getenv
from tinygrad.device import Compiled, Compiler, MallocAllocator, CPUProgram
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.renderer.cstyle import ClangRenderer

class ClangJITCompiler(Compiler):
  def __init__(self, cachekey="compile_clang_jit"): super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # -fno-math-errno is required for __builtin_sqrt to become an instruction instead of a function call
    # x18 is a reserved platform register. It is clobbered on context switch in macos and is used to store TEB pointer in windows on arm, don't use it
    target = 'x86_64' if sys.platform == 'win32' else platform.machine()
    args = ['-march=native', f'--target={target}-none-unknown-elf', '-O2', '-fPIC', '-ffreestanding', '-fno-math-errno', '-nostdlib']
    arch_args = ['-ffixed-x18'] if target == 'arm64' else []
    obj = subprocess.check_output([getenv("CC", 'clang'), '-c', '-x', 'c', *args, *arch_args, '-', '-o', '-'], input=src.encode('utf-8'))
    return jit_loader(obj)

  def disassemble(self, lib:bytes): return capstone_flatdump(lib)

class ClangDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, MallocAllocator, ClangRenderer(), ClangJITCompiler(), CPUProgram)

CPUDevice = ClangDevice