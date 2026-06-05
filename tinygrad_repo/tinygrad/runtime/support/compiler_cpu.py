import ctypes, subprocess
from tinygrad.device import Compiler
from tinygrad.helpers import getenv, capstone_flatdump, DEBUG, unwrap
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.runtime.autogen import llvm

class ClangJITCompiler(Compiler):
  def __init__(self, arch:list[str], cachekey="compile_clang_jit"):
    assert len(arch) >= 2, f"invalid arch string: {','.join(arch)!r}, expected '<arch>,<cpu>,[<feats>]' (eg. 'x86_64,znver2')"
    self.arch, cpu, *feats = arch
    match self.arch:
      case "x86_64": self.args = [f"-march={cpu}"] + [f"-mno{f}" if f.startswith("-") else f"-m{f}" for f in feats]
      # on arm march means "runs on this arch and superset" instead of "optimize for this arch". x86 march == arm mcpu
      # x18 is a reserved platform register. It is clobbered on context switch in macos and is used to store TEB pointer in windows on arm
      case "arm64": self.args = ["-ffixed-x18", "-mcpu=" + "+".join([cpu] + ["no"+f[1:] if f.startswith("-") else f for f in feats])]
      case "riscv64": self.args = ["-march=" + "_".join(["rv64g" if cpu == "native" else cpu] + feats)]
      case _: raise RuntimeError(f"unsupported arch: {self.arch!r}")
    super().__init__(f"{cachekey}_{'_'.join(arch)}")

  def compile_to_obj(self, src:str) -> bytes:
    """Compile C source to ELF object file (before linking)."""
    # -fno-math-errno is required for __builtin_sqrt to become an instruction instead of a function call
    return subprocess.check_output([getenv("CC", 'clang'), '-c', '-x', 'c', '-O2', '-fPIC', '-ffreestanding', '-fno-math-errno', '-nostdlib',
                                    '-fno-ident', f'--target={self.arch}-none-unknown-elf', *self.args, '-', '-o', '-'], input=src.encode('utf-8'))

  def compile(self, src:str) -> bytes: return jit_loader(self.compile_to_obj(src))

  def disassemble(self, lib:bytes): return capstone_flatdump(lib, self.arch)

def cerr(): return ctypes.pointer(ctypes.pointer(ctypes.c_char()))

def expect(x, err, ret=None):
  if x: raise RuntimeError(unwrap(ctypes.cast(err.contents, ctypes.c_char_p).value).decode() if not isinstance(err, str) else err)
  return ret

class LLVMCompiler(Compiler):
  jit = True
  def __init__(self, arch:str, processor:str, feats:str, cache_key=None):
    for component in ['Target', 'TargetInfo', 'TargetMC', 'AsmParser', 'AsmPrinter']:
      getattr(llvm, "LLVMInitialize" + {'arm64': 'AArch64', 'x86_64': 'X86', 'riscv64': 'riscv64'}.get(arch, "AMDGPU") + component)()

    triple = {'arm64': b'aarch64-none-unknown-elf', 'x86_64': b'x86_64-none-unknown-elf', 'AMDGPU': b'amdgcn-amd-amdhsa'}[arch]
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


class CPULLVMCompiler(LLVMCompiler):
  def __init__(self, arch:list[str], cache_key=None):
    assert len(arch) >= 2, f"invalid arch string: {','.join(arch)!r}, expected '<arch>,<cpu>,[<feats>]' (eg. 'x86_64,znver2')"
    self.arch, cpu, *feats = arch
    featstr = ','.join(f if f.startswith('-') else '+'+f for f in feats)
    if cpu == "native":
      cpu = ctypes.string_at(llvm.LLVMGetHostCPUName()).decode()
      featstr = (featstr + "," if featstr else "") + ctypes.string_at(llvm.LLVMGetHostCPUFeatures()).decode()
    # +reserve-x18 here does the same thing as -ffixed-x18 in ClangJITCompiler, see comments there for why it's needed on arm osx
    super().__init__(self.arch, cpu, ('+reserve-x18,' if self.arch == "arm64" else '') + featstr, cache_key)

  def disassemble(self, lib:bytes): capstone_flatdump(lib, self.arch)

class X86Compiler(Compiler):
  def __init__(self): super().__init__(None)
  def compile(self, src:str) -> bytes: return bytes.fromhex(src)
  def disassemble(self, lib:bytes): return capstone_flatdump(lib, "x86_64")
