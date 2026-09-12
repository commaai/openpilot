import ctypes, struct, platform, pathlib, os, binascii, itertools
from hexdump import hexdump
from tinygrad.helpers import to_mv, DEBUG, getenv, colored, time_to_str
from tinygrad.runtime.autogen import libc, cuda
from tinygrad.device import CPUProgram, Device
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.ops_cuda import cu_time_execution

print(f"hooking CUDA runtime, running with {Device.DEFAULT}")

# TODO: regen and make cuda 12 default?
cuda.cuFuncGetParamInfo = cuda._libraries['libcuda.so'].cuFuncGetParamInfo
cuda.cuFuncGetParamInfo.restype = cuda.CUresult
cuda.cuFuncGetParamInfo.argtypes = [cuda.CUfunction, cuda.size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]

ignore_dispatch = [False] # default valus is False
def push_ignore_dispatch(val):
  global ignore_dispatch
  ignore_dispatch.append(val)

def pop_ignore_dispatch():
  global ignore_dispatch
  ignore_dispatch.pop()

hooked = {}
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

allocated_memory_enum = 0
allocated_memory = {}
function_names = {}
tiny_devs = {}

seen_modules = set()

global_events = []
class HookEvent: pass
class HookMemAllocEvent(HookEvent):
  def __init__(self, cuda_address, bytesize, enum): self.cuda_address, self.bytesize, self.enum = cuda_address, bytesize, enum
  def __repr__(self): return f"tensor alloc: {self.enum}: {self.cuda_address:#x} - {self.bytesize:#x} bytes"
class HookConstParamEvent(HookEvent):
  def __init__(self, value): self.value = value
  def __repr__(self): return f"const({self.value:#x})"
class HookTensorParamEvent(HookEvent):
  def __init__(self, cuda_address, offset, enum): self.cuda_address, self.offset, self.enum = cuda_address, offset, enum
  def __repr__(self): return f"tensor{self.enum}({self.cuda_address:#x}, {self.offset=:#x})"
class HookKernelCallEvent(HookEvent):
  def __init__(self, grid, block, tm, ptm, name, params): self.grid, self.block, self.tm, self.ptm, self.name, self.params = grid, block, tm, ptm, name, params
  def __repr__(self): return f"kernel call <<{self.grid}>> <<{self.block}>> {self.ptm}\n | {self.params}\n | {self.name}"

def collect_events(clear=False):
  global global_events
  x = global_events
  if clear: global_events = []
  return x

@ctypes.CFUNCTYPE(*([cuda.cuDeviceGet.restype] + cuda.cuDeviceGet.argtypes))
def cuDeviceGet(device, ordinal):
  tiny_devs[ordinal] = Device[f"{Device.DEFAULT}:{ordinal}"]
  device.contents.value = ordinal
  return cuda.CUDA_SUCCESS

@ctypes.CFUNCTYPE(*([cuda.cuMemHostAlloc.restype] + cuda.cuMemHostAlloc.argtypes))
def cuMemHostAlloc(pp, bytesize, flags):
  print(f"cuMemHostAlloc {bytesize}")
  return hooked["cuMemHostAlloc"](pp, bytesize, flags)

@ctypes.CFUNCTYPE(*([cuda.cuModuleLoadData.restype] + cuda.cuModuleLoadData.argtypes))
def cuModuleLoadData(module, image):
  ret = hooked["cuModuleLoadData"](module, image)
  module_address = ctypes.addressof(module.contents.contents)
  seen_modules.add(module_address)
  return ret

@ctypes.CFUNCTYPE(*([cuda.cuModuleGetFunction.restype] + cuda.cuModuleGetFunction.argtypes))
def cuModuleGetFunction(hfunc, hmod, name):
  ret = hooked["cuModuleGetFunction"](hfunc, hmod, name)
  python_name = ctypes.string_at(name).decode()

  # pip install git+https://github.com/wbenny/pydemangler.git
  import pydemangler
  demangled_name = pydemangler.demangle(python_name)
  if demangled_name is not None: python_name = demangled_name

  # print(f"called cuModuleGetFunction 0x{ctypes.addressof(hmod.contents):X} {python_name}")
  function_names[ctypes.addressof(hfunc.contents.contents)] = python_name
  return ret

@ctypes.CFUNCTYPE(*([cuda.cuMemAlloc_v2.restype] + cuda.cuMemAlloc_v2.argtypes))
def cuMemAlloc_v2(dptr, bytesize):
  global allocated_memory_enum, text_prefix

  ret = hooked["cuMemAlloc_v2"](dptr, bytesize)
  cuda_address = dptr.contents.value
  allocated_memory[cuda_address] = (bytesize, allocated_memory_enum)

  global_events.append(HookMemAllocEvent(cuda_address, bytesize, allocated_memory_enum))
  if DEBUG >= 3: print(global_events[-1])

  allocated_memory_enum += 1
  return ret

@ctypes.CFUNCTYPE(*([cuda.cuLaunchKernel.restype] + cuda.cuLaunchKernel.argtypes))
def cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra):
  global ignore_dispatch

  name = function_names[ctypes.addressof(f.contents)]
  if ignore_dispatch[-1]:
    if DEBUG >= 4: print(f"ignoring dispatch {name}")
    return 0

  tm = cu_time_execution(lambda:
    hooked["cuLaunchKernel"](f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra), True)

  ptm = colored(time_to_str(tm, w=9), "yellow" if tm > 0.01 else "green")

  params = []
  while True:
    ret = cuda.cuFuncGetParamInfo(f, len(params), ctypes.byref(paramOffset:=ctypes.c_size_t()), ctypes.byref(paramSize:=ctypes.c_size_t()))
    if ret != 0: break
    params.append((paramOffset.value, paramSize.value))

  ev_params = []
  if extra: params_ptr = to_mv(extra, 5*8).cast("Q")
  else: params_ptr = to_mv(kernelParams, len(params)*8).cast("Q")

  for i,(off,sz) in enumerate(params):
    sz_to_let = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}
    if sz >= 8:
      for j in range(sz//8):
        if extra: value = to_mv(params_ptr[1] + off, sz).cast("Q")[0]
        else: value = to_mv(params_ptr[i] + j*8, 8).cast('Q')[0]

        has_in_allocated_mem, lcoff, alnum = False, 0, -1
        for taddr, (tsz, talnum) in allocated_memory.items():
          if taddr <= value < taddr + tsz:
            has_in_allocated_mem = True
            lcoff = value - taddr
            alnum = talnum
            break

        if has_in_allocated_mem: ev_params.append(HookTensorParamEvent(value, lcoff, alnum))
        else: ev_params.append(HookConstParamEvent(value))
    else:
      if extra: value = to_mv(params_ptr[1] + off, sz).cast(sz_to_let[sz])[0]
      else: value = to_mv(params_ptr[i], sz).cast(sz_to_let[sz])[0]
      ev_params.append(HookConstParamEvent(value))

  global_events.append(HookKernelCallEvent((gridDimX, gridDimY, gridDimZ), (blockDimX, blockDimY, blockDimZ), tm, ptm, name, ev_params))
  if DEBUG >= 3: print(global_events[-1])

  return 0

def create_hook(func_name, restype, argtypes):
  def hook_template(*args):
    # print(func_name, flush=True)
    return hooked[func_name](*args)
  return ctypes.CFUNCTYPE(restype, *argtypes)(hook_template)

def install_hooks():
  hooked['cuModuleGetFunction'] = install_hook(cuda.cuModuleGetFunction, cuModuleGetFunction)
  hooked['cuLaunchKernel'] = install_hook(cuda.cuLaunchKernel, cuLaunchKernel)

  # memory stuff
  hooked['cuMemAlloc_v2'] = install_hook(cuda.cuMemAlloc_v2, cuMemAlloc_v2)
  hooked['cuMemHostAlloc'] = install_hook(cuda.cuMemHostAlloc, cuMemHostAlloc)

  # module loading + not used module loading
  hooked['cuModuleLoadData'] = install_hook(cuda.cuModuleLoadData, cuModuleLoadData)

NVPROFILER = os.environ.get("NV_COMPUTE_PROFILER_PERFWORKS_DIR", None) # realize and wait each aten call
if NVPROFILER is None: install_hooks()
else:
  print("Detected NSIGHT Profiled, hooking not avail.")
  cuda._libraries['libcuda.so'] = ctypes.CDLL(NVPROFILER + "/libcuda-injection.so")
