import ctypes, ctypes.util
from tinygrad.helpers import getenv

def _try_dlopen_gpuocelot():
  GPUOCELOT_PATHS = [ctypes.util.find_library("gpuocelot")] if ctypes.util.find_library("gpuocelot") is not None else []
  GPUOCELOT_PATHS += ["libgpuocelot.so", "/usr/local/lib/libgpuocelot.so",
                      "libgpuocelot.dylib", "/usr/local/lib/libgpuocelot.dylib", "/opt/homebrew/lib/libgpuocelot.dylib"]
  for path in GPUOCELOT_PATHS:
    try:
      gpuocelot_lib = ctypes.CDLL(path)
      gpuocelot_lib.ptx_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    except OSError: pass
    else: return gpuocelot_lib
  print("Could not find libgpuocelot.so")
  return None

class PythonRemu:
  """Python RDNA3 emulator wrapper that matches the libremu.so interface."""
  valid_mem_ranges: set[tuple[int, int]] = set()

  def run_asm(self, lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int) -> int:
    from extra.assembly.rdna3.emu import run_asm, set_valid_mem_ranges
    # Pad ranges to handle GPU loads that may read past small buffers (e.g. s_load_b128 on 12-byte buffer)
    set_valid_mem_ranges({(start, size + 4096) for start, size in self.valid_mem_ranges})
    return run_asm(lib, lib_sz, gx, gy, gz, lx, ly, lz, args_ptr)

def _try_dlopen_remu():
  # Use Python emulator only if PYTHON_REMU=1
  if getenv("PYTHON_REMU"):
    return PythonRemu()
  REMU_PATHS = ["extra/remu/target/release/libremu.so", "libremu.so", "/usr/local/lib/libremu.so",
               "extra/remu/target/release/libremu.dylib", "libremu.dylib", "/usr/local/lib/libremu.dylib", "/opt/homebrew/lib/libremu.dylib"]
  for path in REMU_PATHS:
    try:
      remu = ctypes.CDLL(path)
      remu.run_asm.restype = ctypes.c_int32
      remu.run_asm.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    except OSError: pass
    else: return remu
  print("Could not find libremu.so")
  return None
