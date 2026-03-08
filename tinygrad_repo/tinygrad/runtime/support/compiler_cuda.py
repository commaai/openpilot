import subprocess, hashlib, tempfile, ctypes, re, pathlib
from typing import Callable
from tinygrad.helpers import to_char_p_p, colored, getenv, system
from tinygrad.runtime.support.c import init_c_var
from tinygrad.runtime.autogen import nvrtc, nvjitlink as jitlink
from tinygrad.device import Compiler, CompileError

CUDA_PATH = getenv("CUDA_PATH", "")

def _get_bytes(arg, get_str, get_sz, check) -> bytes:
  x = ctypes.create_string_buffer(init_c_var(ctypes.c_size_t, lambda x: check(get_sz(arg, ctypes.byref(x)))).value)
  check(get_str(arg, x))
  return ctypes.string_at(x, size=len(x))

def nvrtc_check(status, ctx=None):
  if status != 0:
    err_log = _get_bytes(ctx, nvrtc.nvrtcGetProgramLog, nvrtc.nvrtcGetProgramLogSize, lambda _: None).decode() if ctx else ""
    raise CompileError(f"Nvrtc Error {status}, {ctypes.string_at(nvrtc.nvrtcGetErrorString(status)).decode()}\n{err_log}")

def jitlink_check(status, ctx=None):
  if status != 0:
    err_log = _get_bytes(ctx, jitlink.nvJitLinkGetErrorLog, jitlink.nvJitLinkGetErrorLogSize, lambda _: None).decode() if ctx else ""
    raise CompileError(f"jitlink Error {status}, {jitlink.nvJitLinkResult.get(status)}\n{err_log}")

def pretty_ptx(s):
  # all expressions match `<valid_before><expr><valid_after>` and replace it with `<valid_before>color(<expr>)<valid_after>`
  s = re.sub(r'([!@<\[\s,\+\-;\n])((?:[_%$][\w%\$_]+(?:\.[xyz])?\:?)|(?:buf\d+))([<>\]\s,\+\-;\n\)])',
             lambda m:m[1]+colored(m[2], "blue")+m[3], s, flags=re.M) # identifiers
  s = re.sub(r'(.)((?:b|s|u|f)(?:8|16|32|64)|pred)([\.\s])', lambda m:m[1]+colored(m[2], "green")+m[3], s, flags=re.M) # types
  s = re.sub(r'^(\s*)([\w]+)(.*?;$)', lambda m:m[1]+colored(m[2], "yellow")+m[3], s, flags=re.M) # instructions
  s = re.sub(r'([<>\[\]\s,\+\-;])((?:0[fF][0-9a-fA-F]{8})|(?:[0-9]+)|(?:0[xX][0-9a-fA-F]+))([<>\[\]\s,\+\-;])',
             lambda m:m[1]+colored(m[2], "yellow")+m[3], s, flags=re.M) # numbers
  s = re.sub(r'(\.)(param|reg|global)', lambda m:m[1]+colored(m[2], "magenta"), s, flags=re.M) # space
  s = re.sub(r'(\.)(version|target|address_size|visible|entry)', lambda m:m[1]+colored(m[2], "magenta"), s, flags=re.M) # derivatives
  return s

def cuda_disassemble(lib:bytes, arch:str, ptx=False):
  try:
    fn = (pathlib.Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(lib).hexdigest()}").as_posix()
    with open(fn, "wb") as f: f.write(lib.rstrip(b'\x00') if ptx else lib)
    if ptx: system(f"ptxas -arch={arch} -o {fn} {fn}")
    print(system(f'nvdisasm {fn}'))
  except Exception as e: print("Failed to generate SASS", str(e), "Make sure your PATH contains ptxas/nvdisasm binary of compatible version.")

class CUDACompiler(Compiler):
  def __init__(self, arch:str, cache_key:str="cuda"):
    self.arch, self.compile_options = arch, [f'--gpu-architecture={arch}']
    self.compile_options += [f"-I{CUDA_PATH}/include"] if CUDA_PATH else ["-I/usr/local/cuda/include", "-I/usr/include", "-I/opt/cuda/include"]
    nvrtc_check(nvrtc.nvrtcVersion((nvrtcMajor := ctypes.c_int()), (nvrtcMinor := ctypes.c_int())))
    if (nvrtcMajor.value, nvrtcMinor.value) >= (12, 4): self.compile_options.append("--minimal")
    super().__init__(f"compile_{cache_key}_{self.arch}")
  def _compile_program(self, src:str, nvrtc_get_content:Callable, nvrtc_get_size:Callable) -> bytes:
    nvrtc_check(nvrtc.nvrtcCreateProgram(ctypes.byref(prog := nvrtc.nvrtcProgram()), src.encode(), "<null>".encode(), 0, None, None))
    nvrtc_check(nvrtc.nvrtcCompileProgram(prog, len(self.compile_options), to_char_p_p([o.encode() for o in self.compile_options])), prog)
    data = _get_bytes(prog, nvrtc_get_content, nvrtc_get_size, nvrtc_check)
    nvrtc_check(nvrtc.nvrtcDestroyProgram(ctypes.byref(prog)))
    return data
  def compile(self, src:str) -> bytes: return self._compile_program(src, nvrtc.nvrtcGetPTX, nvrtc.nvrtcGetPTXSize)
  def disassemble(self, lib:bytes): cuda_disassemble(lib, self.arch, ptx=True)

class NVCompiler(CUDACompiler):
  def __init__(self, arch:str): super().__init__(arch, cache_key="nv")
  def compile(self, src:str) -> bytes: return self._compile_program(src, nvrtc.nvrtcGetCUBIN, nvrtc.nvrtcGetCUBINSize)
  def disassemble(self, lib:bytes): cuda_disassemble(lib, self.arch)

class NVCCCompiler(Compiler):
  def __init__(self, arch:str, extra_options:list[str]=[]):
    self.arch, self.extra_options = arch, extra_options
    super().__init__(f"compile_nvcc_{self.arch}_{hashlib.sha256(' '.join(extra_options).encode()).hexdigest()[:8]}")
  def compile(self, src:str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".cu") as srcf, tempfile.NamedTemporaryFile(suffix=".ptx") as libf:
      srcf.write(src.encode())
      srcf.flush()
      subprocess.run(["nvcc", f"-arch={self.arch}", "-ptx", "-o", libf.name, srcf.name] + self.extra_options, check=True)
      return libf.read()
  def disassemble(self, lib:bytes): cuda_disassemble(lib, self.arch, ptx=True)

class PTXCompiler(Compiler):
  def __init__(self, arch:str, cache_key="ptx"):
    self.arch = arch
    super().__init__(f"compile_{cache_key}_{self.arch}")
  def compile(self, src:str) -> bytes:
    return src.replace("TARGET", self.arch).replace("VERSION", "8.7" if (ver:=int(self.arch[3:]))>=120 else ("7.8" if ver>=89 else "7.5")).encode()
  def disassemble(self, lib:bytes): cuda_disassemble(lib, self.arch, ptx=True)

class NVPTXCompiler(PTXCompiler):
  def __init__(self, arch:str):
    nvrtc_check(jitlink.nvJitLinkVersion(ctypes.byref(ctypes.c_uint()), ctypes.byref(ctypes.c_uint())))
    super().__init__(arch, cache_key="nv_ptx")
  def compile(self, src:str) -> bytes:
    jitlink_check(jitlink.nvJitLinkCreate(handle := jitlink.nvJitLinkHandle(), 1, to_char_p_p([f'-arch={self.arch}'.encode()])), handle)
    jitlink_check(jitlink.nvJitLinkAddData(handle, jitlink.NVJITLINK_INPUT_PTX, ptxsrc:=super().compile(src), len(ptxsrc), "<null>".encode()), handle)
    jitlink_check(jitlink.nvJitLinkComplete(handle), handle)
    data = _get_bytes(handle, jitlink.nvJitLinkGetLinkedCubin, jitlink.nvJitLinkGetLinkedCubinSize, jitlink_check)
    jitlink_check(jitlink.nvJitLinkDestroy(handle))
    return data
  def disassemble(self, lib:bytes): cuda_disassemble(lib, self.arch)
