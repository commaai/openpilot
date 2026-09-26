import ctypes, struct, platform, pathlib, os, binascii
from hexdump import hexdump
from tinygrad.helpers import to_mv, DEBUG, getenv
from tinygrad.runtime.autogen import libc, cuda
from tinygrad.device import CPUProgram
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.ops_cuda import cu_time_execution

def _hook(fxn_address_value, tramp):
  page_address = (fxn_address_value//0x1000)*0x1000
  ret = libc.mprotect(page_address, 0x2000, 7)
  assert ret == 0
  libc.memcpy(fxn_address_value, tramp, len(tramp))
  ret = libc.mprotect(page_address, 0x2000, 5)
  assert ret == 0
  CPUProgram.rt_lib["__clear_cache"](fxn_address_value, fxn_address_value + len(tramp))

def install_hook(c_function, python_function):
  python_function_addr = ctypes.cast(ctypes.byref(python_function), ctypes.POINTER(ctypes.c_ulong)).contents.value
  # AARCH64 trampoline to ioctl
  if (processor:=platform.processor()) == "aarch64":
    # 0x0000000000000000:  70 00 00 10    adr x16, #0xc
    # 0x0000000000000004:  10 02 40 F9    ldr x16, [x16]
    # 0x0000000000000008:  00 02 1F D6    br  x16
    tramp = b"\x70\x00\x00\x10\x10\x02\x40\xf9\x00\x02\x1f\xd6"
    tramp += struct.pack("Q", python_function_addr)
  elif processor == "x86_64":
    # 0x0000000000000000:  49 BB aa aa aa aa aa aa aa aa    movabs r11, <address>
    # 0x000000000000000a:  41 FF E3                         jmp    r11
    tramp = b"\x49\xBB" + struct.pack("Q", python_function_addr) + b"\x41\xFF\xE3"
  else:
    raise Exception(f"processor {processor} not supported")
  tramp = ctypes.create_string_buffer(tramp)

  # get real function address
  fxn_address = ctypes.cast(ctypes.byref(c_function), ctypes.POINTER(ctypes.c_ulong))
  fxn_address_value = fxn_address.contents.value
  #print(f"** hooking function at 0x{fxn_address_value}")

  orig_save = (ctypes.c_char*len(tramp))()
  libc.memcpy(orig_save, fxn_address_value, len(tramp))
  _hook(fxn_address_value, tramp)

  def original(*args):
    _hook(fxn_address_value, orig_save)
    ret = c_function(*args)
    _hook(fxn_address_value, tramp)
    return ret
  return original

hooked = {}

allocated_memory = {}
function_names = {}

seen_modules = set()

@ctypes.CFUNCTYPE(ctypes.c_int)
def dummy():
  print("**** dummy function hook ****")
  return -1

@ctypes.CFUNCTYPE(*([cuda.cuInit.restype] + cuda.cuInit.argtypes))
def cuInit(flags):
  print("call cuInit", flags)
  return hooked["cuInit"](flags)

@ctypes.CFUNCTYPE(*([cuda.cuMemHostAlloc.restype] + cuda.cuMemHostAlloc.argtypes))
def cuMemHostAlloc(pp, bytesize, flags):
  print(f"cuMemHostAlloc {bytesize}")
  return hooked["cuMemHostAlloc"](pp, bytesize, flags)

@ctypes.CFUNCTYPE(*([cuda.cuModuleLoadData.restype] + cuda.cuModuleLoadData.argtypes))
def cuModuleLoadData(module, image):
  ret = hooked["cuModuleLoadData"](module, image)
  module_address = ctypes.addressof(module.contents.contents)
  print(f"cuModuleLoadData 0x{image:x} -> 0x{module_address:X}")
  seen_modules.add(module_address)

  #images, sections, relocs = elf_loader(bytes(to_mv(image, 0x100000)))
  #for s in sections: print(s)

  #print('\n'.join([x for x in maps.split("\n") if 'libcuda' in x]))

  #hexdump(to_mv(image, 0x1000))
  #image, sections, relocs = elf_loader(to_mv(image))
  #print(sections)
  return ret

@ctypes.CFUNCTYPE(*([cuda.cuModuleGetFunction.restype] + cuda.cuModuleGetFunction.argtypes))
def cuModuleGetFunction(hfunc, hmod, name):
  ret = hooked["cuModuleGetFunction"](hfunc, hmod, name)
  python_name = ctypes.string_at(name).decode()

  # pip install git+https://github.com/wbenny/pydemangler.git
  import pydemangler
  demangled_name = pydemangler.demangle(python_name)
  if demangled_name is not None: python_name = demangled_name

  print(f"called cuModuleGetFunction 0x{ctypes.addressof(hmod.contents):X} {python_name}")
  function_names[ctypes.addressof(hfunc.contents.contents)] = python_name
  return ret

@ctypes.CFUNCTYPE(*([cuda.cuMemAlloc_v2.restype] + cuda.cuMemAlloc_v2.argtypes))
def cuMemAlloc_v2(dptr, bytesize):
  ret = hooked["cuMemAlloc_v2"](dptr, bytesize)
  cuda_address = ctypes.addressof(dptr.contents)
  allocated_memory[cuda_address] = bytesize
  print(f"cuMemAlloc_v2 {bytesize} 0x{cuda_address:X}")
  return ret

@ctypes.CFUNCTYPE(*([cuda.cuLaunchKernel.restype] + cuda.cuLaunchKernel.argtypes))
def cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra):
  tm = cu_time_execution(lambda:
    hooked["cuLaunchKernel"](f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra), True)

  name = function_names[ctypes.addressof(f.contents)]
  print(f"{tm*1e6:9.2f} us -- cuLaunchKernel <<{gridDimX:6d}, {gridDimY:5d}, {gridDimZ:5d}>>",
    f"<<{blockDimX:4d}, {blockDimY:4d}, {blockDimZ:4d}>> {sharedMemBytes} {name}")

  if extra: hexdump(to_mv(extra, 0x100))

  if getenv("PARAMS") and kernelParams:
    #print(f"params @ 0x{ctypes.addressof(kernelParams.contents):X}")
    params = []
    while True:
      ret = cuda.cuFuncGetParamInfo(f, len(params), ctypes.byref(paramOffset:=ctypes.c_size_t()), ctypes.byref(paramSize:=ctypes.c_size_t()))
      if ret != 0: break
      params.append((paramOffset.value, paramSize.value))
    #params_dat = to_mv(kernelParams.contents, params[-1][0] + params[-1][1])
    params_ptr = to_mv(kernelParams, len(params)*8).cast("Q")
    #params_dat = to_mv(kernelParams.contents, params[-1][0] + params[-1][1])
    for i,(off,sz) in enumerate(params):
      hexdump(to_mv(params_ptr[i], sz))


    #hexdump(params_dat)
    #for i,(off,sz) in enumerate(params):
    #  print(f"{i}: offset:{off:3d} size:{sz:3d}") # --", binascii.hexlify(dat).decode())
    #  hexdump(params_dat[off:off+sz])
    #if name == "exp2_kernel_vectorized4_kernel":
    #  ptr_0 = struct.unpack("Q", params_dat[0x10:0x18])[0]
    #  hexdump(to_mv(ptr_0, 0x80))
    #ptr_1 = struct.unpack("Q", to_mv(ptr_0, 8))[0]

  #print(f"params 0x{ctypes.addressof(kernelParams):X}")
  #hexdump(to_mv(kernelParams, 0x100))
  #print(f"data 0x{to_mv(kernelParams, 8).cast('Q')[0]:X}")
  #hexdump(to_mv(kernelParams.contents, 0x80))
  #for i,addr in enumerate(to_mv(kernelParams.contents, 0x100).cast("Q")): print(f"{i*8:3d}: {addr:X}")

  return 0

if __name__ == "__main__":
  #out = cuda.CUmoduleLoadingMode()
  #print(cuda.cuModuleGetLoadingMode(ctypes.byref(out)))
  #print(out.value)

  hooked['cuInit'] = install_hook(cuda.cuInit, cuInit)
  hooked['cuModuleGetFunction'] = install_hook(cuda.cuModuleGetFunction, cuModuleGetFunction)
  hooked['cuLaunchKernel'] = install_hook(cuda.cuLaunchKernel, cuLaunchKernel)

  # memory stuff
  hooked['cuMemAlloc_v2'] = install_hook(cuda.cuMemAlloc_v2, cuMemAlloc_v2)
  hooked['cuMemHostAlloc'] = install_hook(cuda.cuMemHostAlloc, cuMemHostAlloc)

  # module loading + not used module loading
  hooked['cuModuleLoadData'] = install_hook(cuda.cuModuleLoadData, cuModuleLoadData)
  install_hook(cuda.cuModuleLoad, dummy)
  install_hook(cuda.cuModuleLoadDataEx, dummy)
  install_hook(cuda.cuModuleLoadFatBinary, dummy)

  # library stuff (doesn't seem used)
  #install_hook(cuda.cuLibraryLoadData, dummy)
  #install_hook(cuda.cuLibraryLoadFromFile, dummy)
  #install_hook(cuda.cuLibraryGetModule, dummy)

  #install_hook(cuda.cuMemAllocManaged, dummy)

  # unused
  #install_hook(cuda.cuFuncGetModule, dummy)
  #install_hook(cuda.cuModuleGetGlobal_v2, dummy)

  # hook v1
  #install_hook(cuda._libraries['libcuda.so'].cuModuleGetGlobal, dummy)
  #install_hook(cuda._libraries['libcuda.so'].cuMemAlloc, dummy)
  #install_hook(cuda._libraries['libcuda.so'].cuLinkComplete, dummy)

  #nvjitlink = ctypes.CDLL("/home/tiny/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12")
  #install_hook(nvjitlink.nvJitLinkCreate, dummy)
  #nvrtc = ctypes.CDLL("/home/tiny/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.11.2")
  #nvrtc = ctypes.CDLL("/usr/local/cuda-12.4/targets/x86_64-linux/lib/libnvrtc.so.12.4.127")
  #from tinygrad.runtime.autogen import nvrtc
  #install_hook(nvrtc.nvrtcCreateProgram, dummy)
  #install_hook(nvrtc.nvJitLinkCreate, dummy)

  #import tinygrad.runtime.autogen.nvrtc as nvrtc
  #install_hook(nvrtc.nvJitLinkCreate, dummy)
  #install_hook(nvrtc.nvrtcCreateProgram, dummy)

  #hooked['cuLinkCreate'] = install_hook(cuda.cuLinkCreate, dummy)

  if getenv("TINYGRAD"):
    from tinygrad import Tensor
    (Tensor.zeros(6, device="CUDA").contiguous()*2).realize()
    exit(0)

  print("importing torch...")
  import torch
  print("torch", torch.__version__, torch.__file__)

  if getenv("RESNET"):
    import torchvision.models as models
    model = models.resnet18(pretrained=True)
    model = model.cuda()
    model.eval()

    if getenv("COMPILE"): model = torch.compile(model)

    X = torch.rand(getenv("BS", 1), 3, 288, 288, device='cuda')
    model(X)

    print("\n\n\n****** second run ******\n")
    model(X)
  else:
    a = torch.zeros(4, 4).cuda()
    b = torch.zeros(4, 4).cuda()
    print("tensor created")
    print(f"a: 0x{a.data_ptr():X}")
    print(f"b: 0x{b.data_ptr():X}")
    a += 1
    b += 2
    a = a.exp2()
    b = b.exp2()
    a += b
    #c = a @ b
    print("tensor math done", a.cpu().numpy())

  # confirm cuda library is right
  #maps = pathlib.Path("/proc/self/maps").read_text()
  #print('\n'.join([x for x in maps.split("\n") if 'cuda' in x or 'nv' in x]))
