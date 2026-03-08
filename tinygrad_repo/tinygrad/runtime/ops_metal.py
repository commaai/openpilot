import subprocess, pathlib, struct, ctypes, tempfile, functools, contextlib, decimal, platform
from tinygrad.helpers import prod, to_mv, getenv, round_up, cache_dir, PROFILE, ProfileRangeEvent, cpu_profile, unwrap, suppress_finalizing
import tinygrad.runtime.support.objc as objc
from tinygrad.device import Compiled, Compiler, CompileError, LRUAllocator, ProfileDeviceEvent, CompilerSet, CompilerPair
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.runtime.autogen import metal
from tinygrad.runtime.support.c import DLL

# 13 is requestType that metal uses to compile source code into MTLB, there aren't any docs or symbols.
REQUEST_TYPE_COMPILE = 13

# Must be loaded for default Metal Device: https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice?language=objc
DLL("CoreGraphics", "CoreGraphics")

# FIXME: these need autogen to support objc categories
# https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/ObjectiveC/Chapters/ocCategories.html
@functools.cache
def to_ns_str(s: str): return ctypes.cast(objc.msg("stringWithUTF8String:")(metal.NSString._objc_class_, s.encode()), metal.NSString)
def from_ns_str(s): return bytes(objc.msg("UTF8String", ctypes.c_char_p)(s)).decode()

def wait_check(cbuf:metal.MTLCommandBuffer):
  cbuf.waitUntilCompleted()
  error_check(cbuf.error().retained())

def cmdbuf_label(cbuf:metal.MTLCommandBuffer) -> str|None: return from_ns_str(label) if (label:=cbuf.label()).value is not None else None

def error_check(error: metal.NSError, error_constructor: type[Exception] = RuntimeError):
  if error.value is None: return None
  raise error_constructor(from_ns_str(error.localizedDescription().retained()))

class MetalDevice(Compiled):
  def __init__(self, device:str):
    self.sysdevice = metal.MTLCreateSystemDefaultDevice()
    self.mtl_queue = self.sysdevice.newCommandQueueWithMaxCommandBufferCount(1024)
    if self.mtl_queue is None: raise RuntimeError("Cannot allocate a new command queue")
    self.mtl_buffers_in_flight: list[metal.MTLCommandBuffer] = []
    self.timeline_signal = self.sysdevice.newSharedEvent()
    self.timeline_value = 0

    Compiled.profile_events += [ProfileDeviceEvent(device)]

    from tinygrad.runtime.graph.metal import MetalGraph
    # NOTE: GitHub CI macOS runners use paravirtualized metal which is broken with graph.
    # This can be reproduced locally with any virtualization software (like utm) that can create macOS VMs with apple's own virtualization framework.
    super().__init__(device, MetalAllocator(self), CompilerSet([CompilerPair(MetalRenderer, MetalCompiler), CompilerPair(MetalRenderer, Compiler)]),
      functools.partial(MetalProgram, self), MetalGraph if 'virtual' not in from_ns_str(self.sysdevice.name()).lower() else None)

  def synchronize(self):
    for cbuf in self.mtl_buffers_in_flight:
      wait_check(cbuf)
      st, en = decimal.Decimal(cbuf.GPUStartTime()) * 1000000, decimal.Decimal(cbuf.GPUEndTime()) * 1000000
      # NOTE: command buffers from MetalGraph are not profiled here
      if PROFILE and (lb:=cmdbuf_label(cbuf)) is not None and not lb.startswith("batched"):
        Compiled.profile_events += [ProfileRangeEvent(self.device, lb, st, en, is_copy=lb.startswith("COPY"))]
    self.mtl_buffers_in_flight.clear()

def metal_src_to_library(device:MetalDevice, src:str) -> metal.MTLLibrary:
  options = metal.MTLCompileOptions.new()
  options.setFastMathEnabled(getenv("METAL_FAST_MATH"))
  library = device.sysdevice.newLibraryWithSource_options_error(to_ns_str(src), options, ctypes.byref(compileError:=metal.NSError().retained()))
  error_check(compileError, CompileError)
  return library

class MetalCompiler(Compiler):
  # Opening METAL after LLVM doesn't fail because ctypes.CDLL opens with RTLD_LOCAL but MTLCompiler opens it's own llvm with RTLD_GLOBAL
  # This means that MTLCompiler's llvm will create it's own instances of global state because RTLD_LOCAL doesn't export symbols, but if RTLD_GLOBAL
  # library is loaded first then RTLD_LOCAL library will just use it's symbols. On linux there is RTLD_DEEPBIND to prevent that, but on macos there
  # doesn't seem to be anything we can do.
  with contextlib.suppress(FileNotFoundError, ModuleNotFoundError):
    import tinygrad.runtime.autogen.llvm # noqa: F401
  support = DLL("MTLCompiler", "MTLCompiler")
  support.MTLCodeGenServiceCreate.restype = ctypes.c_void_p

  def __init__(self):
    self.cgs = ctypes.c_void_p(MetalCompiler.support.MTLCodeGenServiceCreate(b"tinygrad"))
    super().__init__("compile_metal_direct")
  def __reduce__(self): return (MetalCompiler,()) # force pickle to create new instance for each multiprocessing fork
  def compile(self, src:str) -> bytes:
    ret: Exception|bytes = CompileError("MTLCodeGenServiceBuildRequest returned without calling the callback")
    @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p)
    def callback(blockptr, error, dataPtr, dataLen, errorMessage):
      nonlocal ret
      if error == 0:
        reply = bytes(to_mv(dataPtr, dataLen))
        # offset from beginning to data = header size + warning size
        ret = reply[sum(struct.unpack('<LL', reply[8:16])):]
      else:
        ret = CompileError(errorMessage.decode())

    # no changes for compute in 2.0 - 2.4 specs, use 2.0 as default for old versions.
    macos_major = int(platform.mac_ver()[0].split('.')[0])
    metal_version = "metal3.1" if macos_major >= 14 else "metal3.0" if macos_major >= 13 else "macos-metal2.0"

    # llvm will create modules.timestamp in cache path and cache compilation of metal stdlib (250ms => 8ms compilation time)
    # note that llvm won't necessarily create anything else here as apple has prebuilt versions of many standard libraries
    params = f'-fno-fast-math -std={metal_version} --driver-mode=metal -x metal -fmodules-cache-path="{cache_dir}" -fno-caret-diagnostics'
    # source blob has to be padded to multiple of 4 but at least one 'b\x00' should be added, params blob just has to be null terminated
    src_padded, params_padded = src.encode() + b'\x00'*(round_up(len(src) + 1, 4) - len(src)), params.encode() + b'\x00'
    request = struct.pack('<QQ', len(src_padded), len(params_padded)) + src_padded + params_padded
    # The callback is actually not a callback but a block which is apple's non-standard extension to add closures to C.
    # See https://clang.llvm.org/docs/Block-ABI-Apple.html#high-level for struct layout.
    # Fields other than invoke are unused in this case so we can just use ctypes.byref with negative offset to invoke field, add blockptr as a first
    # argument and pretend it's a normal callback
    MetalCompiler.support.MTLCodeGenServiceBuildRequest(self.cgs, None, REQUEST_TYPE_COMPILE, request, len(request), ctypes.byref(callback, -0x10))
    if isinstance(ret, Exception): raise ret
    assert ret[:4] == b"MTLB" and ret[-4:] == b"ENDT", f"Invalid Metal library. {ret!r}"
    return ret
  def disassemble(self, lib:bytes):
    with tempfile.NamedTemporaryFile(delete=True) as shader:
      shader.write(lib)
      shader.flush()
      proc = subprocess.Popen(f"cd {pathlib.Path(__file__).parents[2]}/extra/disassemblers/applegpu && python3 compiler_explorer.py {shader.name}",
                              stdout=subprocess.PIPE, shell=True, text=True, bufsize=1)
      for line in unwrap(proc.stdout): print(line, end="")
      ret = proc.wait()
      if ret: print("Disassembler Error: Make sure you have https://github.com/dougallj/applegpu cloned to tinygrad/extra/disassemblers/applegpu")

class MetalProgram:
  def __init__(self, dev:MetalDevice, name:str, lib:bytes, **kwargs):
    self.dev, self.name, self.lib = dev, name, lib
    if lib[:4] == b"MTLB":
      # binary metal library
      data = objc.dispatch_data_create(lib, len(lib), None, None)
      self.library = self.dev.sysdevice.newLibraryWithData_error(data, ctypes.byref(error_lib:=metal.NSError().retained())).retained()
      error_check(error_lib)
    else:
      # metal source. rely on OS caching
      try: self.library = metal_src_to_library(self.dev, lib.decode())
      except CompileError as e: raise RuntimeError from e
    self.fxn = self.library.newFunctionWithName(to_ns_str(name)).retained()
    descriptor = metal.MTLComputePipelineDescriptor.new()
    descriptor.setComputeFunction(self.fxn)
    descriptor.setSupportIndirectCommandBuffers(True)
    self.pipeline_state = self.dev.sysdevice.newComputePipelineStateWithDescriptor_options_reflection_error(descriptor, metal.MTLPipelineOptionNone,
      None, ctypes.byref(error_pipeline_creation:=metal.NSError().retained()))
    error_check(error_pipeline_creation)
    # cache these msg calls
    self.max_total_threads: int = self.pipeline_state.maxTotalThreadsPerThreadgroup()

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    if prod(local_size) > self.max_total_threads:
      exec_width = self.pipeline_state.threadExecutionWidth()
      memory_length = self.pipeline_state.staticThreadgroupMemoryLength()
      raise RuntimeError(f"local size {local_size} bigger than {self.max_total_threads} with exec width {exec_width} memory length {memory_length}")
    command_buffer = self.dev.mtl_queue.commandBuffer().retained() # FIXME: is this really ARC?
    encoder = command_buffer.computeCommandEncoder().retained() # FIXME: is this really ARC?
    encoder.setComputePipelineState(self.pipeline_state)
    for i,a in enumerate(bufs): encoder.setBuffer_offset_atIndex(a.buf, a.offset, i)
    for i,a in enumerate(vals, start=len(bufs)): encoder.setBytes_length_atIndex(bytes(ctypes.c_int(a)), 4, i)
    encoder.dispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(*global_size), metal.MTLSize(*local_size))
    encoder.endEncoding()
    command_buffer.setLabel(to_ns_str(self.name)) # TODO: is this always needed?
    command_buffer.commit()
    self.dev.mtl_buffers_in_flight.append(command_buffer)
    if wait:
      wait_check(command_buffer)
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()

class MetalBuffer:
  def __init__(self, buf:metal.MTLBuffer, size:int, offset=0): self.buf, self.size, self.offset = buf, size, offset

class MetalAllocator(LRUAllocator[MetalDevice]):
  def _alloc(self, size:int, options) -> MetalBuffer:
    if options.external_ptr: return MetalBuffer(metal.MTLBuffer(options.external_ptr), size)

    # Buffer is explicitly released in _free() rather than garbage collected via reference count
    ret = self.dev.sysdevice.newBufferWithLength_options(size, metal.MTLResourceStorageModeShared)
    ret.retain = False
    if ret.value is None: raise MemoryError(f"Metal OOM while allocating {size=}")
    return MetalBuffer(ret, size)
  @suppress_finalizing
  def _free(self, opaque:MetalBuffer, options):
    if not options.external_ptr: opaque.buf.release
  def _transfer(self, dest:MetalBuffer, src:MetalBuffer, sz:int, src_dev:MetalDevice, dest_dev:MetalDevice):
    dest_dev.synchronize()
    src_command_buffer = src_dev.mtl_queue.commandBuffer().retained()
    encoder = src_command_buffer.blitCommandEncoder().retained()
    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(src.buf, src.offset, dest.buf, dest.offset, sz)
    encoder.endEncoding()
    if src_dev != dest_dev:
      src_command_buffer.encodeSignalEvent_value(ctypes.cast(src_dev.timeline_signal, metal.MTLEvent), src_dev.timeline_value)
      dest_command_buffer = dest_dev.mtl_queue.commandBuffer().retained()
      dest_command_buffer.encodeWaitForEvent_value(ctypes.cast(src_dev.timeline_signal, metal.MTLEvent), src_dev.timeline_value)
      dest_command_buffer.commit()
      dest_dev.mtl_buffers_in_flight.append(dest_command_buffer)
      src_dev.timeline_value += 1
    src_command_buffer.setLabel(to_ns_str(f"COPY {src_dev.device} -> {dest_dev.device}"))
    src_command_buffer.commit()
    src_dev.mtl_buffers_in_flight.append(src_command_buffer)
    # Transfers currently synchronize the completion. Otherwise, copies can sometimes lead to incorrect values.
    # There is no real metal multidevice support for now, so transfer is used only for tests.
    src_dev.synchronize()
  def _cp_mv(self, dst, src, prof_desc):
    with cpu_profile(prof_desc, self.dev.device, is_copy=True): dst[:] = src
  def _as_buffer(self, src:MetalBuffer) -> memoryview:
    self.dev.synchronize()
    return to_mv(src.buf.contents(), src.size + src.offset)[src.offset:]
  def _copyin(self, dest:MetalBuffer, src:memoryview): self._cp_mv(self._as_buffer(dest), src, "TINY -> METAL")
  def _copyout(self, dest:memoryview, src:MetalBuffer): self._cp_mv(dest, self._as_buffer(src), "METAL -> TINY")
  def _offset(self, buf:MetalBuffer, size:int, offset:int): return MetalBuffer(buf.buf, size, offset)
