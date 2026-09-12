import ctypes
from tinygrad.runtime.support import c

gpuocelot_lib = c.DLL("ocelot", "gpuocelot")
@gpuocelot_lib.bind(None, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
def ptx_run(source:bytes, n_args:int, args:c.POINTER[ctypes.c_void_p], blck_x:int, blck_y:int, blck_z:int,
            grid_x:int, grid_y:int, grid_z:int, shared_mem_size:int): pass

class PythonRemu:
  """Python RDNA3/RDNA4 emulator wrapper used by mockgpu."""
  valid_mem_ranges: set[tuple[int, int]] = set()
  rsrc2: int = 0x19c  # Default: USER_SGPR_COUNT=14, enable X and Y workgroup IDs
  scratch_size: int = 0  # private_segment_fixed_size from kernel descriptor
  arch: str = "rdna3"  # Architecture: rdna3 or rdna4
  user_data: list[int] = []  # All COMPUTE_USER_DATA registers (loaded into s[0:N])

  def run_asm(self, lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int) -> int:
    from test.mockgpu.amd.emu import run_asm
    return run_asm(lib, lib_sz, gx, gy, gz, lx, ly, lz, args_ptr, self.rsrc2, self.scratch_size, self.arch, self.user_data)
