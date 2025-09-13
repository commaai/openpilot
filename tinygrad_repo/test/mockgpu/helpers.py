import ctypes, ctypes.util

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

def _try_dlopen_remu():
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
