import ctypes, platform, sys, subprocess
from tinygrad.device import Compiler
from tinygrad.helpers import OSX, getenv, capstone_flatdump, DEBUG, unwrap
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.runtime.autogen import llvm

class ClangJITCompiler(Compiler):
  def __init__(self, cachekey="compile_clang_jit"): super().__init__(cachekey)

  def compile_to_obj(self, src:str) -> bytes:
    """Compile C source to ELF object file (before linking)."""
    # -fno-math-errno is required for __builtin_sqrt to become an instruction instead of a function call
    # x18 is a reserved platform register. It is clobbered on context switch in macos and is used to store TEB pointer in windows on arm, don't use it
    target = 'x86_64' if sys.platform == 'win32' else platform.machine()
    # on arm march means "runs on this arch and superset" instead of "optimize for this arch". x86 march == arm mcpu
    arch = {'x86_64': '-march=native', 'AMD64': '-march=native', 'riscv64': '-march=rv64g'}.get(platform.machine(), "-mcpu=native")
    args = [arch, f'--target={target}-none-unknown-elf', '-O2', '-fPIC', '-ffreestanding', '-fno-math-errno', '-nostdlib', '-fno-ident']
    arch_args = ['-ffixed-x18'] if target == 'arm64' else []
    return subprocess.check_output([getenv("CC", 'clang'), '-c', '-x', 'c', *args, *arch_args, '-', '-o', '-'], input=src.encode('utf-8'))

  def compile(self, src:str) -> bytes: return jit_loader(self.compile_to_obj(src))

  def disassemble(self, lib:bytes): return capstone_flatdump(lib)

def cerr(): return ctypes.pointer(ctypes.pointer(ctypes.c_char()))

def expect(x, err, ret=None):
  if x: raise RuntimeError(unwrap(ctypes.cast(err.contents, ctypes.c_char_p).value).decode() if not isinstance(err, str) else err)
  return ret

class LLVMCompiler(Compiler):
  jit = True
  target_arch = {'arm64': 'AArch64', 'aarch64': 'AArch64', 'x86_64': 'X86', 'AMD64': 'X86', 'riscv64': 'riscv64'}[platform.machine()]
  def __init__(self, processor:str, feats:str, cache_key=None):
    for component in ['Target', 'TargetInfo', 'TargetMC', 'AsmParser', 'AsmPrinter']: getattr(llvm, f'LLVMInitialize{self.target_arch}{component}')()

    triple = {'AArch64': b'aarch64-none-unknown-elf', 'X86': b'x86_64-none-unknown-elf', 'AMDGPU': b'amdgcn-amd-amdhsa'}[self.target_arch]
    target = expect(llvm.LLVMGetTargetFromTriple(triple, ctypes.pointer(tgt:=llvm.LLVMTargetRef()), err:=cerr()), err, tgt)
    if DEBUG >= 3: print(f"LLVM init for {processor!r} with {feats!r}")
    self.target_machine = llvm.LLVMCreateTargetMachine(target, triple, processor.encode(), feats.encode(),
                                                       llvm.LLVMCodeGenLevelDefault, llvm.LLVMRelocPIC, llvm.LLVMCodeModelDefault)

    self.pbo = llvm.LLVMCreatePassBuilderOptions()
    if (opt:=bool(getenv("LLVMOPT", "1"))):
      self.passes = b'default<O2>'
      llvm.LLVMPassBuilderOptionsSetLoopUnrolling(self.pbo, True)
      llvm.LLVMPassBuilderOptionsSetLoopVectorization(self.pbo, True)
      llvm.LLVMPassBuilderOptionsSetSLPVectorization(self.pbo, True)
      llvm.LLVMPassBuilderOptionsSetVerifyEach(self.pbo, True)
    else:
      self.passes = b'default<O0>'

    # Create a per-instance context instead of using the global context to avoid shared state between parallel test processes
    self.context = llvm.LLVMContextCreate()
    self.diag_msgs: list[str] = []
    @llvm.LLVMDiagnosticHandler
    def handle_diag(diag_ref, _arg):
      severity = llvm.LLVMGetDiagInfoSeverity(diag_ref)
      msg = ctypes.string_at(llvm.LLVMGetDiagInfoDescription(diag_ref)).decode()
      if severity == llvm.LLVMDSError:
        self.diag_msgs.append(msg)
    self.handle_diag = handle_diag
    llvm.LLVMContextSetDiagnosticHandler(self.context, handle_diag, None)
    super().__init__(cache_key or f"compile_llvm_{processor}_{feats}{'_jit' if self.jit else ''}{'_opt' if opt else ''}")

  def __del__(self):
    llvm.LLVMDisposePassBuilderOptions(self.pbo)
    llvm.LLVMContextDispose(self.context)

  def compile_to_obj(self, src:str) -> bytes:
    self.diag_msgs.clear()
    src_buf = llvm.LLVMCreateMemoryBufferWithMemoryRangeCopy(ctypes.create_string_buffer(src_bytes:=src.encode()), len(src_bytes), b'src')
    mod = expect(llvm.LLVMParseIRInContext(self.context, src_buf, ctypes.pointer(m:=llvm.LLVMModuleRef()), err:=cerr()), err, m)
    expect(llvm.LLVMVerifyModule(mod, llvm.LLVMReturnStatusAction, err:=cerr()), err)
    expect(llvm.LLVMRunPasses(mod, self.passes, self.target_machine, self.pbo), 'failed to run passes')
    if DEBUG >= 7: print(ctypes.string_at(llvm.LLVMPrintModuleToString(mod)).decode())
    obj_buf = expect(llvm.LLVMTargetMachineEmitToMemoryBuffer(self.target_machine, mod, llvm.LLVMObjectFile, err:=cerr(),
                                                              buf:=llvm.LLVMMemoryBufferRef()), err, buf)
    llvm.LLVMDisposeModule(mod)
    obj = ctypes.string_at(llvm.LLVMGetBufferStart(obj_buf), llvm.LLVMGetBufferSize(obj_buf))
    llvm.LLVMDisposeMemoryBuffer(obj_buf)
    if self.diag_msgs: raise RuntimeError("llvm diagnostic: " + "\n".join(self.diag_msgs))
    return obj

  def compile(self, src:str) -> bytes: return jit_loader(self.compile_to_obj(src)) if self.jit else self.compile_to_obj(src)

  def disassemble(self, lib:bytes): capstone_flatdump(lib)

class CPULLVMCompiler(LLVMCompiler):
  def __init__(self, cache_key=None):
    # +reserve-x18 here does the same thing as -ffixed-x18 in ops_cpu.py, see comments there for why it's needed on arm osx
    cpu, feats = ctypes.string_at(llvm.LLVMGetHostCPUName()), (b'+reserve-x18,' if OSX else b'') + ctypes.string_at(llvm.LLVMGetHostCPUFeatures())
    super().__init__(cpu.decode(), feats.decode(), cache_key)
