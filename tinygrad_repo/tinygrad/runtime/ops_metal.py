from __future__ import annotations
import os, pathlib, struct, ctypes, tempfile, functools
from typing import List, Any, Union, Tuple, cast
from tinygrad.helpers import prod, to_mv, getenv, round_up, cache_dir, T, init_c_struct_t
from tinygrad.device import Compiled, Compiler, CompileError, LRUAllocator
from tinygrad.renderer.cstyle import MetalRenderer

class objc_id(ctypes.c_void_p): # This prevents ctypes from converting response to plain int, and dict.fromkeys() can use it to dedup
  def __hash__(self): return hash(self.value)
  def __eq__(self, other): return self.value == other.value

class objc_instance(objc_id): # method with name "new", "alloc" should be freed after use
  def __del__(self): msg(self, "release")

@functools.lru_cache(None)
def sel(name: str): return libobjc.sel_registerName(name.encode())

class MTLResourceOptions:
  MTLResourceCPUCacheModeDefaultCache = 0
  MTLResourceStorageModeShared = 0 << 4

class MTLPipelineOption:
  MTLPipelineOptionNone = 0

# 13 is requestType that metal uses to compile source code into MTLB, there aren't any docs or symbols.
REQUEST_TYPE_COMPILE = 13

libobjc = ctypes.CDLL("/usr/lib/libobjc.dylib")
libmetal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
compiler = ctypes.CDLL("/System/Library/PrivateFrameworks/MTLCompiler.framework/MTLCompiler")
# Must be loaded for default Metal Device: https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice?language=objc
ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
libdispatch = ctypes.CDLL("/usr/lib/libSystem.dylib") # libdispatch is part of libSystem on mac
libobjc.objc_getClass.restype = objc_id
libobjc.sel_registerName.restype = objc_id
libmetal.MTLCreateSystemDefaultDevice.restype = objc_instance
compiler.MTLCodeGenServiceCreate.restype = ctypes.c_void_p
libdispatch.dispatch_data_create.restype = objc_instance

# Ignore mypy error reporting incompatible default, because typevar default only works on python 3.12
def msg(ptr: objc_id, selector: str, /, *args: Any, restype: type[T] = objc_id) -> T: # type: ignore [assignment]
  sender = libobjc["objc_msgSend"] # Using attribute access returns a new reference so setting restype is safe
  sender.restype = restype
  return sender(ptr, sel(selector), *args)

def to_ns_str(s: str): return msg(libobjc.objc_getClass(b"NSString"), "stringWithUTF8String:", s.encode(), restype=objc_instance)

def to_struct(*t: int, _type: type = ctypes.c_ulong): return init_c_struct_t(tuple([(f"field{i}", _type) for i in range(len(t))]))(*t)

def wait_check(cbuf: Any):
  msg(cbuf, "waitUntilCompleted")
  error_check(msg(cbuf, "error", restype=objc_instance))

def elapsed_time(cbuf: objc_id):
  return cast(float, msg(cbuf, "GPUEndTime", restype=ctypes.c_double)) - cast(float, msg(cbuf, "GPUStartTime", restype=ctypes.c_double))

def error_check(error: objc_instance, error_constructor: type[Exception] = RuntimeError):
  if error.value is None: return None
  raise error_constructor(bytes(msg(msg(error, "localizedDescription", restype=objc_instance), "UTF8String", restype=ctypes.c_char_p)).decode())

def metal_src_to_library(device:MetalDevice, src:str) -> objc_instance:
  options = msg(libobjc.objc_getClass(b"MTLCompileOptions"), "new", restype=objc_instance)
  msg(options, "setFastMathEnabled:", getenv("METAL_FAST_MATH"))
  library = msg(device.sysdevice, "newLibraryWithSource:options:error:", to_ns_str(src), options,
                ctypes.byref(compileError:=objc_instance()), restype=objc_instance)
  error_check(compileError, CompileError)
  return library

class MetalCompiler(Compiler):
  def __init__(self):
    self.cgs = ctypes.c_void_p(compiler.MTLCodeGenServiceCreate(b"tinygrad"))
    super().__init__("compile_metal_direct")
  def __reduce__(self): return (MetalCompiler,()) # force pickle to create new instance for each multiprocessing fork
  def compile(self, src:str) -> bytes:
    ret: Union[Exception, bytes] = CompileError("MTLCodeGenServiceBuildRequest returned without calling the callback")
    @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p)
    def callback(blockptr, error, dataPtr, dataLen, errorMessage):
      nonlocal ret
      if error == 0:
        reply = bytes(to_mv(dataPtr, dataLen))
        # offset from beginning to data = header size + warning size
        ret = reply[sum(struct.unpack('<LL', reply[8:16])):]
      else:
        ret = CompileError(errorMessage.decode())
    # llvm will create modules.timestamp in cache path and cache compilation of metal stdlib (250ms => 8ms compilation time)
    # note that llvm won't necessarily create anything else here as apple has prebuilt versions of many standard libraries
    params = f'-fno-fast-math -std=metal3.1 --driver-mode=metal -x metal -fmodules-cache-path="{cache_dir}"'
    # source blob has to be padded to multiple of 4 but at least one 'b\x00' should be added, params blob just has to be null terminated
    src_padded, params_padded = src.encode() + b'\x00'*(round_up(len(src) + 1, 4) - len(src)), params.encode() + b'\x00'
    request = struct.pack('<QQ', len(src_padded), len(params_padded)) + src_padded + params_padded
    # The callback is actually not a callback but a block which is apple's non-standard extension to add closures to C.
    # See https://clang.llvm.org/docs/Block-ABI-Apple.html#high-level for struct layout.
    # Fields other than invoke are unused in this case so we can just use ctypes.byref with negative offset to invoke field, add blockptr as a first
    # argument and pretend it's a normal callback
    compiler.MTLCodeGenServiceBuildRequest(self.cgs, None, REQUEST_TYPE_COMPILE, request, len(request), ctypes.byref(callback, -0x10))
    if isinstance(ret, Exception): raise ret
    assert ret[:4] == b"MTLB" and ret[-4:] == b"ENDT", f"Invalid Metal library. {ret!r}"
    return ret
  def disassemble(self, lib:bytes):
    with tempfile.NamedTemporaryFile(delete=True) as shader:
      shader.write(lib)
      shader.flush()
      ret = os.system(f"cd {pathlib.Path(__file__).parents[2]}/extra/disassemblers/applegpu && python3 compiler_explorer.py {shader.name}")
      if ret: print("Disassembler Error: Make sure you have https://github.com/dougallj/applegpu cloned to tinygrad/extra/disassemblers/applegpu")

class MetalProgram:
  def __init__(self, dev:MetalDevice, name:str, lib:bytes):
    self.dev, self.name, self.lib = dev, name, lib
    if lib[:4] == b"MTLB":
      # binary metal library
      data = libdispatch.dispatch_data_create(lib, len(lib), None, None)
      self.library = msg(self.dev.sysdevice, "newLibraryWithData:error:", data, ctypes.byref(error_lib:=objc_instance()), restype=objc_instance)
      error_check(error_lib)
    else:
      # metal source. rely on OS caching
      try: self.library = metal_src_to_library(self.dev, lib.decode())
      except CompileError as e: raise RuntimeError from e
    self.fxn = msg(self.library, "newFunctionWithName:", to_ns_str(name), restype=objc_instance)
    descriptor = msg(libobjc.objc_getClass(b"MTLComputePipelineDescriptor"), "new", restype=objc_instance)
    msg(descriptor, "setComputeFunction:", self.fxn)
    msg(descriptor, "setSupportIndirectCommandBuffers:", True)
    self.pipeline_state = msg(self.dev.sysdevice, "newComputePipelineStateWithDescriptor:options:reflection:error:",
      descriptor, MTLPipelineOption.MTLPipelineOptionNone, None, ctypes.byref(error_pipeline_creation:=objc_instance()), restype=objc_instance)
    error_check(error_pipeline_creation)

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    max_total_threads = msg(self.pipeline_state, "maxTotalThreadsPerThreadgroup", restype=ctypes.c_ulong)
    if prod(local_size) > cast(int, max_total_threads):
      exec_width = msg(self.pipeline_state, "threadExecutionWidth", restype=ctypes.c_ulong)
      memory_length = msg(self.pipeline_state, "staticThreadgroupMemoryLength", restype=ctypes.c_ulong)
      raise RuntimeError(f"local size {local_size} bigger than {max_total_threads} with exec width {exec_width} memory length {memory_length}")
    command_buffer = msg(self.dev.mtl_queue, "commandBuffer", restype=objc_instance)
    encoder = msg(command_buffer, "computeCommandEncoder", restype=objc_instance)
    msg(encoder, "setComputePipelineState:", self.pipeline_state)
    for i,a in enumerate(bufs): msg(encoder, "setBuffer:offset:atIndex:", a.buf, a.offset, i)
    for i,a in enumerate(vals, start=len(bufs)): msg(encoder, "setBytes:length:atIndex:", bytes(ctypes.c_int(a)), 4, i)
    msg(encoder, "dispatchThreadgroups:threadsPerThreadgroup:", to_struct(*global_size), to_struct(*local_size))
    msg(encoder, "endEncoding")
    msg(command_buffer, "commit")
    if wait:
      wait_check(command_buffer)
      return elapsed_time(command_buffer)
    self.dev.mtl_buffers_in_flight.append(command_buffer)

class MetalBuffer:
  def __init__(self, buf:Any, size:int, offset=0): self.buf, self.size, self.offset = buf, size, offset

class MetalAllocator(LRUAllocator):
  def __init__(self, dev:MetalDevice):
    self.dev:MetalDevice = dev
    super().__init__()
  def _alloc(self, size:int, options) -> MetalBuffer:
    # Buffer is explicitly released in _free() rather than garbage collected via reference count
    ret = msg(self.dev.sysdevice, "newBufferWithLength:options:", ctypes.c_ulong(size), MTLResourceOptions.MTLResourceStorageModeShared,
              restype=objc_id)
    if ret.value is None: raise MemoryError(f"Metal OOM while allocating {size=}")
    return MetalBuffer(ret, size)
  def _free(self, opaque:MetalBuffer, options): msg(opaque.buf, "release")
  def _transfer(self, dest:MetalBuffer, src:MetalBuffer, sz:int, src_dev:MetalDevice, dest_dev:MetalDevice):
    dest_dev.synchronize()
    src_command_buffer = msg(src_dev.mtl_queue, "commandBuffer", restype=objc_instance)
    encoder = msg(src_command_buffer, "blitCommandEncoder", restype=objc_instance)
    msg(encoder, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:", src.buf, ctypes.c_ulong(src.offset),
        dest.buf, ctypes.c_ulong(dest.offset), ctypes.c_ulong(sz))
    msg(encoder, "endEncoding")
    if src_dev != dest_dev:
      msg(src_command_buffer, "encodeSignalEvent:value:", src_dev.timeline_signal, src_dev.timeline_value)
      dest_command_buffer = msg(dest_dev.mtl_queue, "commandBuffer", restype=objc_instance)
      msg(dest_command_buffer, "encodeWaitForEvent:value:", src_dev.timeline_signal, src_dev.timeline_value)
      msg(dest_command_buffer, "commit")
      dest_dev.mtl_buffers_in_flight.append(dest_command_buffer)
      src_dev.timeline_value += 1
    msg(src_command_buffer, "commit")
    src_dev.mtl_buffers_in_flight.append(src_command_buffer)
  def _as_buffer(self, src:MetalBuffer) -> memoryview:
    self.dev.synchronize()
    return to_mv(cast(int, msg(src.buf, "contents", restype=objc_id).value), src.size + src.offset)[src.offset:]
  def _copyin(self, dest:MetalBuffer, src:memoryview): self._as_buffer(dest)[:] = src
  def _copyout(self, dest:memoryview, src:MetalBuffer): dest[:] = self._as_buffer(src)
  def _offset(self, buf:MetalBuffer, size:int, offset:int): return MetalBuffer(buf.buf, size, offset)

class MetalDevice(Compiled):
  def __init__(self, device:str):
    self.sysdevice = libmetal.MTLCreateSystemDefaultDevice()
    self.mtl_queue = msg(self.sysdevice, "newCommandQueueWithMaxCommandBufferCount:", 1024, restype=objc_instance)
    if self.mtl_queue is None: raise RuntimeError("Cannot allocate a new command queue")
    self.mtl_buffers_in_flight: List[Any] = []
    self.mv_in_metal: List[memoryview] = []
    self.timeline_signal = msg(self.sysdevice, "newSharedEvent", restype=objc_instance)
    self.timeline_value = 0

    from tinygrad.runtime.graph.metal import MetalGraph
    super().__init__(device, MetalAllocator(self), MetalRenderer(), MetalCompiler() if getenv("METAL_DIRECT", 1) else Compiler(),
                     functools.partial(MetalProgram, self), MetalGraph)
  def synchronize(self):
    for cbuf in self.mtl_buffers_in_flight: wait_check(cbuf)
    self.mv_in_metal.clear()
    self.mtl_buffers_in_flight.clear()
