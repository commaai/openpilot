import functools, struct
from tinygrad.device import  Compiled, Allocator, Compiler
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.helpers import round_up, OSX
from tinygrad.runtime.autogen import webgpu
from typing import List, Any
import ctypes
import os

backend_types = {v: k for k, v in webgpu.WGPUBackendType__enumvalues.items() }

try:
  instance = webgpu.wgpuCreateInstance(webgpu.WGPUInstanceDescriptor(features = webgpu.WGPUInstanceFeatures(timedWaitAnyEnable = True)))
except AttributeError:
  raise RuntimeError("Cannot find dawn library. Install it with: " + ("brew tap wpmed92/dawn && brew install dawn" if OSX else
  "sudo curl -L https://github.com/wpmed92/pydawn/releases/download/v0.1.6/libwebgpu_dawn.so -o /usr/lib/libwebgpu_dawn.so"))

def to_c_string(_str): return ctypes.create_string_buffer(_str.encode('utf-8'))

def from_wgpu_str(string_view): return ctypes.string_at(string_view.data, string_view.length).decode("utf-8")

def to_wgpu_str(_str):
  return webgpu.WGPUStringView(data=ctypes.cast(ctypes.pointer(to_c_string(_str)), ctypes.POINTER(ctypes.c_char)), length=len(_str))

def _wait(future):
  assert webgpu.wgpuInstanceWaitAny(instance, 1, webgpu.WGPUFutureWaitInfo(future=future), 2**64-1) == webgpu.WGPUWaitStatus_Success, "Future failed"

def write_buffer(device, buf, offset, src):
  src = bytearray(src)
  webgpu.wgpuQueueWriteBuffer(webgpu.wgpuDeviceGetQueue(device), buf, offset, (ctypes.c_uint8 * len(src)).from_buffer(src), len(src))

def _run(async_fun, cb_info_type, cb_type, status_enum, res_idx, msg_idx, *params):
  result: List[Any] = []

  def cb(*params):
    result[:] = params
    if msg_idx: result[msg_idx] = from_wgpu_str(result[msg_idx])

  cb_info = cb_info_type(nextInChain=None, mode=webgpu.WGPUCallbackMode_WaitAnyOnly, callback=cb_type(cb))
  _wait(async_fun(*params, cb_info))

  if result[0] != 1: raise RuntimeError(f"[{status_enum[result[0]] if status_enum else 'ERROR'}]{result[msg_idx] if msg_idx else ''}")
  return result[res_idx] if res_idx else None

def copy_buffer_to_buffer(dev, src, src_offset, dst, dst_offset, size):
  encoder = webgpu.wgpuDeviceCreateCommandEncoder(dev, webgpu.WGPUCommandEncoderDescriptor())
  webgpu.wgpuCommandEncoderCopyBufferToBuffer(encoder, src, src_offset, dst, dst_offset, size)
  cb = webgpu.wgpuCommandEncoderFinish(encoder, webgpu.WGPUCommandBufferDescriptor())
  webgpu.wgpuQueueSubmit(webgpu.wgpuDeviceGetQueue(dev), 1, (webgpu.WGPUCommandBuffer*1)(cb))
  webgpu.wgpuCommandBufferRelease(cb)
  webgpu.wgpuCommandEncoderRelease(encoder)

def read_buffer(dev, buf):
  size = webgpu.wgpuBufferGetSize(buf)
  tmp_buffer = webgpu.wgpuDeviceCreateBuffer(dev, webgpu.WGPUBufferDescriptor(size=size,
    usage=webgpu.WGPUBufferUsage_CopyDst | webgpu.WGPUBufferUsage_MapRead, mappedAtCreation=False))
  copy_buffer_to_buffer(dev, buf, 0, tmp_buffer, 0, size)
  _run(webgpu.wgpuBufferMapAsync2, webgpu.WGPUBufferMapCallbackInfo2, webgpu.WGPUBufferMapCallback2, webgpu.WGPUBufferMapAsyncStatus__enumvalues,
    None, 0, tmp_buffer, webgpu.WGPUMapMode_Read, 0, size)
  void_ptr = ctypes.cast(webgpu.wgpuBufferGetConstMappedRange(tmp_buffer, 0, size), ctypes.c_void_p)
  buf_copy = bytearray((ctypes.c_uint8 * size).from_address(void_ptr.value))
  webgpu.wgpuBufferUnmap(tmp_buffer)
  webgpu.wgpuBufferDestroy(tmp_buffer)
  return memoryview(buf_copy).cast("B")

def pop_error(device):
  return _run(webgpu.wgpuDevicePopErrorScopeF, webgpu.WGPUPopErrorScopeCallbackInfo, webgpu.WGPUPopErrorScopeCallback, None, 2, 2, device)

def create_uniform(wgpu_device, val):
  buf = webgpu.wgpuDeviceCreateBuffer(wgpu_device,
    webgpu.WGPUBufferDescriptor(size=4, usage=webgpu.WGPUBufferUsage_Uniform | webgpu.WGPUBufferUsage_CopyDst))
  write_buffer(wgpu_device, buf, 0, val.to_bytes(4, "little") if isinstance(val, int) else struct.pack('<f', val))
  return buf

class WebGPUProgram:
  def __init__(self, dev, name:str, lib:bytes):
    (self.dev, self.timestamp_supported) = dev

    # Creating shader module
    shader = webgpu.WGPUShaderModuleWGSLDescriptor(code=to_wgpu_str(lib.decode()),
      chain=webgpu.WGPUChainedStruct(sType=webgpu.WGPUSType_ShaderSourceWGSL))
    module = webgpu.WGPUShaderModuleDescriptor()
    module.nextInChain = ctypes.cast(ctypes.pointer(shader), ctypes.POINTER(webgpu.struct_WGPUChainedStruct))

    # Check compiler error
    webgpu.wgpuDevicePushErrorScope(self.dev, webgpu.WGPUErrorFilter_Validation)
    shader_module = webgpu.wgpuDeviceCreateShaderModule(self.dev, module)

    if err := pop_error(self.dev): raise RuntimeError(f"Shader compilation failed: {err}")

    self.name, self.lib, self.prg = name, lib, shader_module
  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False):
    wait = wait and self.timestamp_supported
    tmp_bufs = [*bufs]
    buf_patch = False

    # WebGPU does not allow using the same buffer for input and output
    for i in range(1, len(bufs)):
      if bufs[i] == bufs[0]:
        tmp_bufs[0] = webgpu.wgpuDeviceCreateBuffer(self.dev,
          webgpu.WGPUBufferDescriptor(size=webgpu.wgpuBufferGetSize(bufs[0]), usage=webgpu.wgpuBufferGetUsage(bufs[0])))
        buf_patch = True

    # Creating bind group layout
    binding_layouts = [webgpu.WGPUBindGroupLayoutEntry(binding=0, visibility= webgpu.WGPUShaderStage_Compute,
      buffer=webgpu.WGPUBufferBindingLayout(type=webgpu.WGPUBufferBindingType_Uniform))]
    binding_layouts += [webgpu.WGPUBindGroupLayoutEntry(binding=i+1, visibility=webgpu.WGPUShaderStage_Compute,
      buffer=webgpu.WGPUBufferBindingLayout(type=webgpu.WGPUBufferBindingType_Uniform if i >= len(tmp_bufs)
      else webgpu.WGPUBufferBindingType_Storage)) for i in range(len(tmp_bufs)+len(vals))]

    bl_arr_type = webgpu.WGPUBindGroupLayoutEntry * len(binding_layouts)
    webgpu.wgpuDevicePushErrorScope(self.dev, webgpu.WGPUErrorFilter_Validation)
    bind_group_layouts = [webgpu.wgpuDeviceCreateBindGroupLayout(self.dev, webgpu.WGPUBindGroupLayoutDescriptor(
      entryCount=len(binding_layouts), entries=ctypes.cast(bl_arr_type(*binding_layouts), ctypes.POINTER(webgpu.WGPUBindGroupLayoutEntry))))]

    if bg_layout_err := pop_error(self.dev): raise RuntimeError(f"Error creating bind group layout: {bg_layout_err}")

    # Creating pipeline layout
    pipeline_layout_desc = webgpu.WGPUPipelineLayoutDescriptor(bindGroupLayoutCount=len(bind_group_layouts),
      bindGroupLayouts = (webgpu.WGPUBindGroupLayout * len(bind_group_layouts))(*bind_group_layouts))

    webgpu.wgpuDevicePushErrorScope(self.dev, webgpu.WGPUErrorFilter_Validation)
    pipeline_layout = webgpu.wgpuDeviceCreatePipelineLayout(self.dev, pipeline_layout_desc)

    if pipe_err := pop_error(self.dev): raise RuntimeError(f"Error creating pipeline layout: {pipe_err}")

    # Creating bind group
    bindings = [webgpu.WGPUBindGroupEntry(binding=0, buffer=create_uniform(self.dev, float('inf')), offset=0, size=4)]
    bindings += [webgpu.WGPUBindGroupEntry(binding=i+1, buffer=create_uniform(self.dev, x) if i >= len(tmp_bufs) else x, offset=0,
      size=4 if i >= len(tmp_bufs) else webgpu.wgpuBufferGetSize(x)) for i,x in enumerate(tuple(tmp_bufs)+vals)]

    bg_arr_type = webgpu.WGPUBindGroupEntry * len(bindings)
    bind_group_desc = webgpu.WGPUBindGroupDescriptor(layout=bind_group_layouts[0], entryCount=len(bindings), entries=bg_arr_type(*bindings))
    webgpu.wgpuDevicePushErrorScope(self.dev, webgpu.WGPUErrorFilter_Validation)
    bind_group = webgpu.wgpuDeviceCreateBindGroup(self.dev, bind_group_desc)

    if bind_err := pop_error(self.dev): raise RuntimeError(f"Error creating bind group: {bind_err}")

    # Creating compute pipeline
    compute_desc = webgpu.WGPUComputePipelineDescriptor(layout=pipeline_layout,
      compute=webgpu.WGPUComputeState(module=self.prg, entryPoint=to_wgpu_str(self.name)))
    pipeline_result = _run(webgpu.wgpuDeviceCreateComputePipelineAsync2, webgpu.WGPUCreateComputePipelineAsyncCallbackInfo2,
    webgpu.WGPUCreateComputePipelineAsyncCallback2, webgpu.WGPUCreatePipelineAsyncStatus__enumvalues, 1, None, self.dev, compute_desc)

    command_encoder = webgpu.wgpuDeviceCreateCommandEncoder(self.dev, webgpu.WGPUCommandEncoderDescriptor())
    comp_pass_desc = webgpu.WGPUComputePassDescriptor(nextInChain=None)

    if wait:
      query_set = webgpu.wgpuDeviceCreateQuerySet(self.dev, webgpu.WGPUQuerySetDescriptor(type=webgpu.WGPUQueryType_Timestamp, count=2))
      query_buf = webgpu.wgpuDeviceCreateBuffer(self.dev,
        webgpu.WGPUBufferDescriptor(size=16, usage=webgpu.WGPUBufferUsage_QueryResolve | webgpu.WGPUBufferUsage_CopySrc))
      comp_pass_desc.timestampWrites = ctypes.pointer(webgpu.WGPUComputePassTimestampWrites(
        querySet=query_set, beginningOfPassWriteIndex=0, endOfPassWriteIndex=1))

    # Begin compute pass
    compute_pass = webgpu.wgpuCommandEncoderBeginComputePass(command_encoder, comp_pass_desc)
    webgpu.wgpuComputePassEncoderSetPipeline(compute_pass, pipeline_result)
    webgpu.wgpuComputePassEncoderSetBindGroup(compute_pass, 0, bind_group, 0, None)
    webgpu.wgpuComputePassEncoderDispatchWorkgroups(compute_pass, *global_size)
    webgpu.wgpuComputePassEncoderEnd(compute_pass)

    if wait: webgpu.wgpuCommandEncoderResolveQuerySet(command_encoder, query_set, 0, 2, query_buf, 0)

    cmd_buf = webgpu.wgpuCommandEncoderFinish(command_encoder, webgpu.WGPUCommandBufferDescriptor())
    webgpu.wgpuQueueSubmit(webgpu.wgpuDeviceGetQueue(self.dev), 1, (webgpu.WGPUCommandBuffer*1)(cmd_buf))

    if buf_patch:
      copy_buffer_to_buffer(self.dev, tmp_bufs[0], 0, bufs[0], 0, webgpu.wgpuBufferGetSize(bufs[0]))
      webgpu.wgpuBufferDestroy(tmp_bufs[0])

    if wait:
      time = ((timestamps:=read_buffer(self.dev, query_buf).cast("Q").tolist())[1] - timestamps[0]) / 1e9
      webgpu.wgpuBufferDestroy(query_buf)
      webgpu.wgpuQuerySetDestroy(query_set)
      return time

class WebGpuAllocator(Allocator):
  def __init__(self, dev): self.dev = dev
  def _alloc(self, size: int, options):
    # WebGPU buffers have to be 4-byte aligned
    return webgpu.wgpuDeviceCreateBuffer(self.dev, webgpu.WGPUBufferDescriptor(size=round_up(size, 4),
      usage=webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst | webgpu.WGPUBufferUsage_CopySrc))
  def _copyin(self, dest, src: memoryview):
    if src.nbytes % 4:
      padded_src = bytearray(round_up(src.nbytes, 4))
      padded_src[:src.nbytes] = src
    write_buffer(self.dev, dest, 0, padded_src if src.nbytes % 4 else src)
  def _copyout(self, dest: memoryview, src):
    buffer_data = read_buffer(self.dev, src)
    dest[:] = buffer_data[:dest.nbytes] if webgpu.wgpuBufferGetSize(src)  > dest.nbytes else buffer_data
  def _free(self, opaque, options):
    webgpu.wgpuBufferDestroy(opaque)

class WebGpuDevice(Compiled):
  def __init__(self, device:str):
    # Requesting an adapter
    adapter_res = _run(webgpu.wgpuInstanceRequestAdapterF, webgpu.WGPURequestAdapterCallbackInfo, webgpu.WGPURequestAdapterCallback,
    webgpu.WGPURequestAdapterStatus__enumvalues, 1, 2, instance,

    webgpu.WGPURequestAdapterOptions(powerPreference=webgpu.WGPUPowerPreference_HighPerformance,
      backendType=backend_types.get(os.getenv("WEBGPU_BACKEND", ""), 0)))

    # Get supported features
    supported_features = webgpu.WGPUSupportedFeatures()
    webgpu.wgpuAdapterGetFeatures(adapter_res, supported_features)
    supported = [supported_features.features[i] for i in range(supported_features.featureCount)]
    features = [feat for feat in [webgpu.WGPUFeatureName_TimestampQuery, webgpu.WGPUFeatureName_ShaderF16] if feat in supported]
    dev_desc = webgpu.WGPUDeviceDescriptor(requiredFeatureCount=len(features),requiredFeatures=(webgpu.WGPUFeatureName * len(features))(*features))

    # Limits
    supported_limits = webgpu.WGPUSupportedLimits()
    webgpu.wgpuAdapterGetLimits(adapter_res, ctypes.cast(ctypes.pointer(supported_limits),ctypes.POINTER(webgpu.struct_WGPUSupportedLimits)))
    limits = webgpu.WGPURequiredLimits(limits=supported_limits.limits)
    dev_desc.requiredLimits = ctypes.cast(ctypes.pointer(limits),ctypes.POINTER(webgpu.struct_WGPURequiredLimits))

    # Requesting a device
    device_res = _run(webgpu.wgpuAdapterRequestDeviceF, webgpu.WGPURequestDeviceCallbackInfo, webgpu.WGPURequestDeviceCallback,
    webgpu.WGPURequestDeviceStatus__enumvalues, 1, 2, adapter_res, dev_desc)

    super().__init__(device, WebGpuAllocator(device_res), WGSLRenderer(), Compiler(),
      functools.partial(WebGPUProgram, (device_res, webgpu.WGPUFeatureName_TimestampQuery in supported)))

  def synchronize(self):
    _run(webgpu.wgpuQueueOnSubmittedWorkDone2, webgpu.WGPUQueueWorkDoneCallbackInfo2, webgpu.WGPUQueueWorkDoneCallback2,
    webgpu.WGPUQueueWorkDoneStatus__enumvalues, None, None, webgpu.wgpuDeviceGetQueue(self.runtime.args[0][0]))
