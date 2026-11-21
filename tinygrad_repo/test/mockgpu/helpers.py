import ctypes
from tinygrad.runtime.support import c

gpuocelot_lib = c.DLL("ocelot", "gpuocelot")
@gpuocelot_lib.bind(None, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
def ptx_run(source:bytes, n_args:int, args:c.POINTER[ctypes.c_void_p], blck_x:int, blck_y:int, blck_z:int,
            grid_x:int, grid_y:int, grid_z:int, shared_mem_size:int): pass
