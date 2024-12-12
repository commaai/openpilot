import functools
from tinygrad.device import  Compiled, Allocator, Compiler
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.helpers import round_up
import wgpu
import struct

def create_uniform(wgpu_device, val) -> wgpu.GPUBuffer:
  buf = wgpu_device.create_buffer(size=4, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
  if isinstance(val, int): wgpu_device.queue.write_buffer(buf, 0, val.to_bytes(4, "little"))
  else: wgpu_device.queue.write_buffer(buf, 0, struct.pack('<f', val))
  return buf

class WebGPUProgram:
  def __init__(self, dev, name:str, lib:bytes):
    (self.dev, self.timestamp_supported) = dev
    self.name, self.lib, self.prg = name, lib, self.dev.create_shader_module(code=lib.decode())   # NOTE: this is the compiler
  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False):
    wait = wait and self.timestamp_supported
    binding_layouts = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform }}]
    binding_layouts += [{"binding": i+1, "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": wgpu.BufferBindingType.uniform if i >= len(bufs) else wgpu.BufferBindingType.storage }} for i in range(len(bufs)+len(vals))]  # noqa: E501
    bindings = [{"binding": 0, "resource": {"buffer": create_uniform(self.dev, float('inf')), "offset": 0, "size": 4}}]
    bindings += [{"binding": i+1, "resource": {"buffer": create_uniform(self.dev, x) if i >= len(bufs) else x, "offset": 0,
                                            "size": 4 if i >= len(bufs) else x.size}} for i,x in enumerate(bufs+vals)]  # noqa: E501
    bind_group_layout = self.dev.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = self.dev.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    bind_group = self.dev.create_bind_group(layout=bind_group_layout, entries=bindings)
    compute_pipeline = self.dev.create_compute_pipeline(layout=pipeline_layout,compute={"module": self.prg, "entry_point": self.name},)
    command_encoder = self.dev.create_command_encoder()
    if wait:
      query_set = self.dev.create_query_set(type=wgpu.QueryType.timestamp, count=2)
      query_buf = self.dev.create_buffer(size=16, usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
      timestamp_writes = {"query_set": query_set, "beginning_of_pass_write_index": 0, "end_of_pass_write_index": 1}
    compute_pass = command_encoder.begin_compute_pass(timestamp_writes=timestamp_writes if wait else None) # pylint: disable=E0606
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999) # last 2 not used
    compute_pass.dispatch_workgroups(*global_size)  # x y z
    compute_pass.end()
    if wait:
      command_encoder.resolve_query_set(query_set=query_set, first_query=0, query_count=2, destination=query_buf, destination_offset=0)
    self.dev.queue.submit([command_encoder.finish()])
    return ((timestamps:=self.dev.queue.read_buffer(query_buf).cast("Q").tolist())[1] - timestamps[0]) / 1e9 if wait else None

# WebGPU buffers have to be 4-byte aligned
class WebGpuAllocator(Allocator):
  def __init__(self, dev): self.dev = dev
  def _alloc(self, size: int, options):
    return self.dev.create_buffer(size=round_up(size, 4), usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
  def _copyin(self, dest, src: memoryview):
    if src.nbytes % 4:
      padded_src = bytearray(round_up(src.nbytes, 4))
      padded_src[:src.nbytes] = src
    self.dev.queue.write_buffer(dest, 0, padded_src if src.nbytes % 4 else src)
  def _copyout(self, dest: memoryview, src):
    buffer_data = self.dev.queue.read_buffer(src, 0)
    dest[:] = buffer_data[:dest.nbytes] if src._nbytes > dest.nbytes else buffer_data

class WebGpuDevice(Compiled):
  def __init__(self, device:str):
    adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
    timestamp_supported = wgpu.FeatureName.timestamp_query in adapter.features
    wgpu_device = adapter.request_device(required_features=[wgpu.FeatureName.timestamp_query] if timestamp_supported else [])
    super().__init__(device, WebGpuAllocator(wgpu_device), WGSLRenderer(), Compiler(),
                     functools.partial(WebGPUProgram, (wgpu_device, timestamp_supported)))
