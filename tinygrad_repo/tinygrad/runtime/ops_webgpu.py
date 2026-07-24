import functools, struct
from tinygrad.device import Compiled, Allocator, BufferSpec
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.helpers import round_up, suppress_finalizing, getenv, to_mv
from tinygrad.runtime.autogen import webgpu
from tinygrad.runtime.support import c
from typing import Callable
import ctypes

backend_types = {v: k for k, v in webgpu.enum_WGPUBackendType.items()}
instance = webgpu.wgpuCreateInstance(webgpu.WGPUInstanceDescriptor(features=webgpu.WGPUInstanceFeatures(timedWaitAnyEnable=True)))

def from_wgpu_str(string_view:webgpu.WGPUStringView) -> str: return ctypes.string_at(string_view.data, string_view.length).decode()
def to_wgpu_str(_str:str) -> webgpu.WGPUStringView: return webgpu.WGPUStringView(data=ctypes.create_string_buffer(_str.encode()), length=len(_str))

# gets a memoryview from a buffer, which is assumed to have MAP_READ (see _readable_buffer)
def buf_to_mv(buf:webgpu.WGPUBuffer) -> memoryview:
  BufferMapAsync(buf, webgpu.WGPUMapMode_Read, 0, size:=webgpu.wgpuBufferGetSize(buf))
  return to_mv(webgpu.wgpuBufferGetConstMappedRange(buf, 0, size), size)

# turns a webgpu function returning a future into python-synchronous function
# the new function handles the status code and optional error message, returning the other callback arguments
def synchronous(status_enum:dict[int, str], has_emsg:bool=False):
  def wrap(fn:Callable[..., webgpu.WGPUFuture]) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args):
      status, payload, emsg = 0, [], None

      @next(ty for nm, ty, *_ in fn.argtypes[-1]._real_fields_ if nm == "callback") # type: ignore
      def cb(s:int, *args):
        nonlocal status, payload, emsg
        # the last two arguments are "userdata1" and "userdata2", which we drop
        # we must process wgpu strings in this callback, as they will be freed after we return
        status, (*payload, emsg) = s, [from_wgpu_str(a) if type(a) is webgpu.WGPUStringView else a for a in args[:-2]] + ([] if has_emsg else [None])

      future = fn(*args, fn.argtypes[-1](mode=webgpu.WGPUCallbackMode_WaitAnyOnly, callback=cb)) # type: ignore
      if (future_status:=webgpu.wgpuInstanceWaitAny(instance, 1, webgpu.WGPUFutureWaitInfo(future), 2**64-1)) != webgpu.WGPUWaitStatus_Success:
        raise RuntimeError(f"error while waiting for future ({fn.__name__}): {webgpu.enum_WGPUWaitStatus.get(future_status)}")

      if status != 1: raise RuntimeError(f"[{status_enum.get(status)}]{emsg or ''}")
      return payload if len(payload) > 1 else payload[0] if len(payload) == 1 else None
    return wrapper
  return wrap

BufferMapAsync = synchronous(webgpu.enum_WGPUBufferMapAsyncStatus, True)(webgpu.wgpuBufferMapAsync2)
DevicePopErrorScope = synchronous(webgpu.enum_WGPUPopErrorScopeStatus)(webgpu.wgpuDevicePopErrorScope2)
DeviceCreateComputePipeline = synchronous(webgpu.enum_WGPUCreatePipelineAsyncStatus, True)(webgpu.wgpuDeviceCreateComputePipelineAsync2)
InstanceRequestAdapter = synchronous(webgpu.enum_WGPURequestAdapterStatus, True)(webgpu.wgpuInstanceRequestAdapter2)
AdapterRequestDevice = synchronous(webgpu.enum_WGPURequestDeviceStatus, True)(webgpu.wgpuAdapterRequestDevice2)
QueueOnSubmittedWorkDone = synchronous(webgpu.enum_WGPUQueueWorkDoneStatus)(webgpu.wgpuQueueOnSubmittedWorkDone2)

class WebGPUProgram:
  def __init__(self, dev:'WebGpuDevice', name:str, lib:bytes, **kwargs):
    self.dev, self.name = dev, to_wgpu_str(name)

    # Creating shader module
    shader = webgpu.WGPUShaderModuleWGSLDescriptor(code=to_wgpu_str(lib.decode()),
                                                   chain=webgpu.WGPUChainedStruct(sType=webgpu.WGPUSType_ShaderSourceWGSL))
    module = webgpu.WGPUShaderModuleDescriptor(nextInChain=ctypes.cast(ctypes.pointer(shader), ctypes.POINTER(webgpu.struct_WGPUChainedStruct)))

    # Check compiler error
    webgpu.wgpuDevicePushErrorScope(self.dev.device_res, webgpu.WGPUErrorFilter_Validation)
    self.prg = webgpu.wgpuDeviceCreateShaderModule(self.dev.device_res, module)
    if err := self.dev.pop_error(): raise RuntimeError(f"Shader compilation failed: {err}")

  @suppress_finalizing
  def __del__(self): webgpu.wgpuShaderModuleRelease(self.prg)

  def __call__(self, *bufs:webgpu.WGPUBuffer, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               vals:tuple[int, ...]=(), wait=False, **kw) -> float|None:
    wait = wait and webgpu.WGPUFeatureName_TimestampQuery in self.dev.features

    # Creating bind group layout
    def bgl_entry(n:int, ty:str):
      return webgpu.WGPUBindGroupLayoutEntry(binding=n, visibility=webgpu.WGPUShaderStage_Compute,
                                             buffer=webgpu.WGPUBufferBindingLayout(type=getattr(webgpu, f'WGPUBufferBindingType_{ty}')))
    bind_entries = (webgpu.WGPUBindGroupLayoutEntry * (1+len(bufs)+len(vals)))(
      bgl_entry(0, 'Uniform'), *(bgl_entry(i+1, 'Uniform' if i >= len(bufs) else 'Storage') for i in range(len(bufs)+len(vals))))

    webgpu.wgpuDevicePushErrorScope(self.dev.device_res, webgpu.WGPUErrorFilter_Validation)
    bind_layout = webgpu.wgpuDeviceCreateBindGroupLayout(self.dev.device_res,
                                                         webgpu.WGPUBindGroupLayoutDescriptor(entryCount=len(bind_entries), entries=bind_entries))

    if err := self.dev.pop_error(): raise RuntimeError(f"Error creating bind group layout: {err}")

    # Creating pipeline layout
    pipeline_layout_desc = webgpu.WGPUPipelineLayoutDescriptor(bindGroupLayoutCount=1, bindGroupLayouts=(webgpu.WGPUBindGroupLayout*1)(bind_layout))

    webgpu.wgpuDevicePushErrorScope(self.dev.device_res, webgpu.WGPUErrorFilter_Validation)
    pipeline_layout = webgpu.wgpuDeviceCreatePipelineLayout(self.dev.device_res, pipeline_layout_desc)
    if err := self.dev.pop_error(): raise RuntimeError(f"Error creating pipeline layout: {err}")

    # Creating bind group
    def bg_entry(n:int, x:webgpu.WGPUBuffer|int|float):
      buf = x if isinstance(x, webgpu.WGPUBuffer) else self.dev.create_uniform(x)
      return webgpu.WGPUBindGroupEntry(binding=n, buffer=buf, offset=0, size=webgpu.wgpuBufferGetSize(buf))
    bindings = (webgpu.WGPUBindGroupEntry * (1+len(bufs)+len(vals)))(bg_entry(0, float('inf')), *(bg_entry(i+1, x) for i,x in enumerate(bufs+vals)))

    bind_group_desc = webgpu.WGPUBindGroupDescriptor(layout=bind_layout, entryCount=len(bindings), entries=bindings)
    webgpu.wgpuDevicePushErrorScope(self.dev.device_res, webgpu.WGPUErrorFilter_Validation)
    bind_group = webgpu.wgpuDeviceCreateBindGroup(self.dev.device_res, bind_group_desc)
    if err := self.dev.pop_error(): raise RuntimeError(f"Error creating bind group: {err}")

    # Creating compute pipeline
    compute_desc = webgpu.WGPUComputePipelineDescriptor(layout=pipeline_layout,
                                                        compute=webgpu.WGPUComputeState(module=self.prg, entryPoint=self.name))
    pipeline_result = DeviceCreateComputePipeline(self.dev.device_res, compute_desc)

    command_encoder = webgpu.wgpuDeviceCreateCommandEncoder(self.dev.device_res, webgpu.WGPUCommandEncoderDescriptor())
    comp_pass_desc = webgpu.WGPUComputePassDescriptor()

    if wait:
      query_set = webgpu.wgpuDeviceCreateQuerySet(self.dev.device_res, webgpu.WGPUQuerySetDescriptor(type=webgpu.WGPUQueryType_Timestamp, count=2))
      query_buf = webgpu.wgpuDeviceCreateBuffer(
        self.dev.device_res, webgpu.WGPUBufferDescriptor(size=16, usage=webgpu.WGPUBufferUsage_QueryResolve | webgpu.WGPUBufferUsage_CopySrc))
      comp_pass_desc.timestampWrites = c.pointer(webgpu.WGPUComputePassTimestampWrites(querySet=query_set, beginningOfPassWriteIndex=0,
                                                                                       endOfPassWriteIndex=1))

    # Begin compute pass
    compute_pass = webgpu.wgpuCommandEncoderBeginComputePass(command_encoder, comp_pass_desc)
    webgpu.wgpuComputePassEncoderSetPipeline(compute_pass, pipeline_result)
    webgpu.wgpuComputePassEncoderSetBindGroup(compute_pass, 0, bind_group, 0, None)
    webgpu.wgpuComputePassEncoderDispatchWorkgroups(compute_pass, *global_size)
    webgpu.wgpuComputePassEncoderEnd(compute_pass)

    if wait: webgpu.wgpuCommandEncoderResolveQuerySet(command_encoder, query_set, 0, 2, query_buf, 0)

    cmd_buf = webgpu.wgpuCommandEncoderFinish(command_encoder, webgpu.WGPUCommandBufferDescriptor())
    webgpu.wgpuQueueSubmit(self.dev.queue, 1, (webgpu.WGPUCommandBuffer*1)(cmd_buf))

    # release created objects
    webgpu.wgpuBindGroupLayoutRelease(bind_layout)
    webgpu.wgpuPipelineLayoutRelease(pipeline_layout)
    webgpu.wgpuBindGroupRelease(bind_group)
    webgpu.wgpuComputePipelineRelease(pipeline_result)
    webgpu.wgpuCommandEncoderRelease(command_encoder)
    webgpu.wgpuComputePassEncoderRelease(compute_pass)
    webgpu.wgpuCommandBufferRelease(cmd_buf)

    if wait:
      time = ((timestamps:=buf_to_mv(tmp_buf:=self.dev._readable_buffer(query_buf)).cast("Q").tolist())[1] - timestamps[0]) / 1e9
      self.dev.free(query_buf)
      self.dev.free(tmp_buf)
      webgpu.wgpuQuerySetDestroy(query_set)
      webgpu.wgpuQuerySetRelease(query_set)
      return time
    return None

class WebGpuAllocator(Allocator['WebGpuDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> webgpu.WGPUBuffer:
    # WebGPU buffers have to be 4-byte aligned
    return webgpu.wgpuDeviceCreateBuffer(self.dev.device_res, webgpu.WGPUBufferDescriptor(size=round_up(size, 4),
      usage=webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst | webgpu.WGPUBufferUsage_CopySrc))
  def _copyin(self, dest:webgpu.WGPUBuffer, src:memoryview):
    if src.nbytes % 4:
      padded_src = bytearray(round_up(src.nbytes, 4))
      padded_src[:src.nbytes] = src
    self.dev.write_buffer(dest, padded_src if src.nbytes % 4 else src)
  def _copyout(self, dest:memoryview, src:webgpu.WGPUBuffer):
    dest[:] = buf_to_mv(tmp_buf:=self.dev._readable_buffer(src))[:dest.nbytes]
    self.dev.free(tmp_buf)

  def _free(self, opaque:webgpu.WGPUBuffer, options:BufferSpec): self.dev.free(opaque)

class WebGpuDevice(Compiled):
  def __init__(self, device:str):
    # Requesting an adapter
    adapter_res = InstanceRequestAdapter(instance, webgpu.WGPURequestAdapterOptions(
      powerPreference=webgpu.WGPUPowerPreference_HighPerformance, backendType=backend_types.get(getenv("WEBGPU_BACKEND", ""), 0)))

    # Get supported features
    webgpu.wgpuAdapterGetFeatures(adapter_res, supported_features:=webgpu.WGPUSupportedFeatures())
    self.features = [feat for i in range(supported_features.featureCount)
                     if (feat:=supported_features.features[i]) in [webgpu.WGPUFeatureName_TimestampQuery, webgpu.WGPUFeatureName_ShaderF16]]
    webgpu.wgpuSupportedFeaturesFreeMembers(supported_features)
    dev_desc = webgpu.WGPUDeviceDescriptor(requiredFeatureCount=len(self.features),
                                           requiredFeatures=(webgpu.WGPUFeatureName * len(self.features))(*self.features))

    # Limits
    webgpu.wgpuAdapterGetLimits(adapter_res, supported_limits:=webgpu.WGPUSupportedLimits())
    dev_desc.requiredLimits = c.pointer(webgpu.WGPURequiredLimits(limits=supported_limits.limits))

    # Requesting a device
    self.device_res = AdapterRequestDevice(adapter_res, dev_desc)
    self.queue = webgpu.wgpuDeviceGetQueue(self.device_res)

    webgpu.wgpuAdapterRelease(adapter_res)

    super().__init__(device, WebGpuAllocator(self), [WGSLRenderer], functools.partial(WebGPUProgram, self),
                     arch="shader-f16" * (webgpu.WGPUFeatureName_ShaderF16 in self.features))

  def synchronize(self): QueueOnSubmittedWorkDone(self.queue)

  @suppress_finalizing
  def free(self, buf:webgpu.WGPUBuffer):
    if webgpu.wgpuBufferGetMapState(buf) == webgpu.WGPUBufferMapState_Mapped: webgpu.wgpuBufferUnmap(buf)
    webgpu.wgpuBufferDestroy(buf)
    webgpu.wgpuBufferRelease(buf)

  def pop_error(self) -> str: return DevicePopErrorScope(self.device_res)[1]
  def create_uniform(self, val:int|float) -> webgpu.WGPUBuffer:
    buf = webgpu.wgpuDeviceCreateBuffer(self.device_res,
                                        webgpu.WGPUBufferDescriptor(size=4, usage=webgpu.WGPUBufferUsage_Uniform | webgpu.WGPUBufferUsage_CopyDst))
    self.write_buffer(buf, val.to_bytes(4, "little") if isinstance(val, int) else struct.pack('<f', val))
    return buf
  def _readable_buffer(self, buf:webgpu.WGPUBuffer) -> webgpu.WGPUBuffer:
    size = webgpu.wgpuBufferGetSize(buf)
    ret = webgpu.wgpuDeviceCreateBuffer(self.device_res,
      webgpu.WGPUBufferDescriptor(size=size, usage=webgpu.WGPUBufferUsage_CopyDst | webgpu.WGPUBufferUsage_MapRead, mappedAtCreation=False))

    # copy_buffer_to_buffer
    encoder = webgpu.wgpuDeviceCreateCommandEncoder(self.device_res, webgpu.WGPUCommandEncoderDescriptor())
    webgpu.wgpuCommandEncoderCopyBufferToBuffer(encoder, buf, 0, ret, 0, size)
    cmd_buf = webgpu.wgpuCommandEncoderFinish(encoder, webgpu.WGPUCommandBufferDescriptor())
    webgpu.wgpuQueueSubmit(self.queue, 1, (webgpu.WGPUCommandBuffer*1)(cmd_buf))
    webgpu.wgpuCommandBufferRelease(cmd_buf)
    webgpu.wgpuCommandEncoderRelease(encoder)

    return ret
  def write_buffer(self, buf:webgpu.WGPUBuffer, src:memoryview|bytearray|bytes):
    webgpu.wgpuQueueWriteBuffer(self.queue, buf, 0, (ctypes.c_uint8 * len(src)).from_buffer_copy(src), len(src))
