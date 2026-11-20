# the hip api on top of mock cuda, kernels run on the amd emulator
import ctypes
from types import SimpleNamespace
from tinygrad.runtime.autogen import hip, libc
from tinygrad.runtime.ops_amd import _amd_program_image
from tinygrad.runtime.ops_hip import HIPIFY
from tinygrad.runtime.support.elf import elf_loader
from test.mockgpu.cuda import cuda as mc
from test.mockgpu.amd.emu import run_asm

programs:dict[int, tuple] = {} # a function handle is the address of its module's loaded image

def run_kernel(func:int, args:int, size:int, gx:int, gy:int, gz:int, lx:int, ly:int, lz:int, smem:int) -> bool:
  image, data, text_end = programs[func]
  entry, entry_size = ctypes.addressof(image) + data.entry_point_offset, text_end - data.entry_point_offset
  return run_asm(entry, entry_size, gx, gy, gz, lx, ly, lz, args, data.rsrc2, data.private_segment_size) == 0
mc.run_kernel = run_kernel

def hipModuleLoadData(module, image):
  elf = libc.Elf64_Ehdr.from_address(image) # the section headers end the elf
  gfx1100 = SimpleNamespace(iface=SimpleNamespace(props={"lds_size_in_kb": 64}), target=(11, 0, 0))
  lib = ctypes.string_at(image, elf.e_shoff + elf.e_shnum * elf.e_shentsize)
  data, img = _amd_program_image(gfx1100, lib)
  text = next(sh.header for sh in elf_loader(lib)[1] if sh.name == ".text")
  buf = ctypes.create_string_buffer(img, len(img))
  programs[ctypes.addressof(buf)] = (buf, data, text.sh_addr + text.sh_size)
  return mc.out(module, ctypes.addressof(buf))

def hipGetDevicePropertiesR0600(prop, device_id):
  prop[0].gcnArchName = b"gfx1100"
  return hip.hipSuccess

fns = {fn.__name__: getattr(mc, cu) for cu, fn in HIPIFY.items() if fn.__name__.startswith("hip")}
fns |= dict(hipHostMalloc=mc.cuMemHostAlloc, hipModuleLoadData=hipModuleLoadData, hipGetDevicePropertiesR0600=hipGetDevicePropertiesR0600,
  hipMemcpyAsync=lambda dst, src, size, kind, stream: mc.cuMemcpyAsync(dst, src, size, stream),
  hipStreamWaitValue64=lambda stream, ptr, value, flags, mask: mc.cuStreamWaitValue64_v2(stream, ptr, value, flags))
for name, fn in fns.items(): setattr(hip.dll, name, ctypes.CFUNCTYPE(getattr(hip, name).restype, *getattr(hip, name).argtypes)(fn))
hip.dll._loaded_[hip.dll.nm] = hip.dll
