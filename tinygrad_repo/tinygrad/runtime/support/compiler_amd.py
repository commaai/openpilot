import ctypes, subprocess
import tinygrad.runtime.autogen.comgr as comgr
from tinygrad.device import Compiler, CompileError
from tinygrad.runtime.ops_llvm import LLVMCompiler
from tinygrad.helpers import OSX, to_char_p_p

def amdgpu_disassemble(lib:bytes):
  asm = subprocess.check_output(["llvm-objdump" if OSX else "/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], input=lib)
  print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

def check(status):
  if status != 0:
    comgr.amd_comgr_status_string(status, ctypes.byref(status_str := ctypes.POINTER(ctypes.c_char)()))
    raise RuntimeError(f"comgr fail {status}, {ctypes.string_at(status_str).decode()}")

def _get_comgr_data(data_set, data_type):
  check(comgr.amd_comgr_action_data_get_data(data_set, data_type, 0, ctypes.byref(data_exec := comgr.amd_comgr_data_t())))
  check(comgr.amd_comgr_get_data(data_exec, ctypes.byref(sz := ctypes.c_uint64()), None))
  check(comgr.amd_comgr_get_data(data_exec, ctypes.byref(sz), (dat := ctypes.create_string_buffer(sz.value))))
  check(comgr.amd_comgr_release_data(data_exec))
  return bytes(dat)

# amd_comgr_action_info_set_options was deprecated
def set_options(action_info, options:bytes):
  # TODO: this type should be correct in the autogen stub
  comgr.amd_comgr_action_info_set_option_list.argtypes = [comgr.amd_comgr_action_info_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), comgr.size_t]
  return comgr.amd_comgr_action_info_set_option_list(action_info, to_char_p_p(options_list:=options.split(b' ')), len(options_list))

# AMD_COMGR_SAVE_TEMPS=1 AMD_COMGR_REDIRECT_LOGS=stdout AMD_COMGR_EMIT_VERBOSE_LOGS=1
def compile_hip(prg:str, arch="gfx1100", asm=False) -> bytes:
  check(comgr.amd_comgr_create_action_info(ctypes.byref(action_info := comgr.amd_comgr_action_info_t())))
  check(comgr.amd_comgr_action_info_set_language(action_info, comgr.AMD_COMGR_LANGUAGE_HIP))
  check(comgr.amd_comgr_action_info_set_isa_name(action_info, b"amdgcn-amd-amdhsa--" + arch.encode()))
  check(comgr.amd_comgr_action_info_set_logging(action_info, True))

  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_src := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_bc := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_reloc := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_exec := comgr.amd_comgr_data_set_t())))

  check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_SOURCE, ctypes.byref(data_src := comgr.amd_comgr_data_t())))
  check(comgr.amd_comgr_set_data(data_src, len(rprg := prg.encode()), rprg))

  if asm:
    check(comgr.amd_comgr_set_data_name(data_src, b"<null>.s"))
    check(comgr.amd_comgr_data_set_add(data_set_src, data_src))
    status = comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE, action_info, data_set_src, data_set_reloc)
    if status != 0:
      print(_get_comgr_data(data_set_reloc, comgr.AMD_COMGR_DATA_KIND_LOG).decode())
      raise RuntimeError("assemble failed")
  else:
    check(comgr.amd_comgr_set_data_name(data_src, b"<null>"))
    check(comgr.amd_comgr_data_set_add(data_set_src, data_src))
    # -include hiprtc_runtime.h was removed
    check(set_options(action_info, f"-O3 -mcumode --hip-version=6.0.32830 -DHIP_VERSION_MAJOR=6 -DHIP_VERSION_MINOR=0 -DHIP_VERSION_PATCH=32830 -D__HIPCC_RTC__ -std=c++14 -nogpuinc -Wno-gnu-line-marker -Wno-missing-prototypes --offload-arch={arch} -I/opt/rocm/include -Xclang -disable-llvm-passes".encode())) # noqa: E501
    status = comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, action_info, data_set_src, data_set_bc)
    if status != 0:
      print(_get_comgr_data(data_set_bc, comgr.AMD_COMGR_DATA_KIND_LOG).decode())
      raise RuntimeError("compile failed")
    check(set_options(action_info, b"-O3 -mllvm -amdgpu-internalize-symbols"))
    check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, action_info, data_set_bc, data_set_reloc))

  check(set_options(action_info, b""))
  check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, action_info, data_set_reloc, data_set_exec))
  ret = _get_comgr_data(data_set_exec, comgr.AMD_COMGR_DATA_KIND_EXECUTABLE)
  check(comgr.amd_comgr_release_data(data_src))
  for x in [data_set_src, data_set_bc, data_set_reloc, data_set_exec]: check(comgr.amd_comgr_destroy_data_set(x))
  check(comgr.amd_comgr_destroy_action_info(action_info))
  return ret

class HIPCompiler(Compiler):
  def __init__(self, arch:str):
    self.arch = arch
    super().__init__(f"compile_hip_{self.arch}")
  def compile(self, src:str) -> bytes:
    try: return compile_hip(src, self.arch, src.split('\n', 1)[0].strip() == '.text')
    except RuntimeError as e: raise CompileError(e) from e
  def disassemble(self, lib:bytes): amdgpu_disassemble(lib)

class AMDLLVMCompiler(LLVMCompiler):
  jit = False
  target_arch = "AMDGPU"
  def __init__(self, arch: str):
    self.arch = arch
    super().__init__(self.arch, "+cumode")
  def __reduce__(self): return (AMDLLVMCompiler, (self.arch,))
  def compile(self, src:str) -> bytes:
    try: return super().compile(src)
    except RuntimeError as e:
      if "undefined value '@llvm.amdgcn." in str(e): raise CompileError(str(e) + "AMD with LLVM backend requires LLVM >= 18") from e
      raise CompileError(e) from e
  def disassemble(self, lib:bytes): amdgpu_disassemble(lib)
