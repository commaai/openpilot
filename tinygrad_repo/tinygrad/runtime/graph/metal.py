from typing import List, Any, Dict, cast, Optional
import ctypes
from tinygrad.dtype import dtypes
from tinygrad.helpers import dedup, getenv
from tinygrad.device import Buffer
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.ops import Variable
from tinygrad.runtime.ops_metal import wait_check, msg, libobjc, to_struct, objc_instance,\
  MTLResourceOptions, elapsed_time, objc_id

class MTLIndirectCommandType:
  MTLIndirectCommandTypeConcurrentDispatch = (1 << 5)

class MTLResourceUsage:
  MTLResourceUsageRead = 0b01
  MTLResourceUsageWrite = 0b10

class MetalGraph(GraphRunner):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    if not all(isinstance(ji.prg, CompiledRunner) for ji in jit_cache): raise GraphException

    # create metal batch exec
    icb_descriptor = msg(libobjc.objc_getClass(b"MTLIndirectCommandBufferDescriptor"), "new", restype=objc_instance)
    msg(icb_descriptor, "setCommandTypes:", MTLIndirectCommandType.MTLIndirectCommandTypeConcurrentDispatch)
    msg(icb_descriptor, "setInheritBuffers:", False)
    msg(icb_descriptor, "setInheritPipelineState:", False)
    msg(icb_descriptor, "setMaxKernelBufferBindCount:", 31)

    self.icb = msg(self.dev.sysdevice, "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",
      icb_descriptor, len(jit_cache), MTLResourceOptions.MTLResourceCPUCacheModeDefaultCache, restype=objc_instance)
    if self.icb.value is None: raise GraphException("create indirect command buffer failed, does your system support this?")
    icb_label = bytes(msg(msg(self.icb, "description", restype=objc_instance), "UTF8String", restype=ctypes.c_char_p)).decode()
    self.needs_icb_fix = int("AGXG15XFamilyIndirectCommandBuffer" not in icb_label)    # not required on M3

    if len(self.vars): self.int_buf = self.dev.allocator.alloc(len(self.vars)*dtypes.int32.itemsize)
    all_resources = [self.int_buf.buf] if len(self.vars) else []
    all_pipelines = []
    for j,ji in enumerate(jit_cache):
      prg: CompiledRunner = cast(CompiledRunner, ji.prg)
      icb_command = msg(self.icb, "indirectComputeCommandAtIndex:", j, restype=objc_instance)
      all_pipelines.append(prg._prg.pipeline_state)
      msg(icb_command, "setComputePipelineState:", prg._prg.pipeline_state)
      for i,b in enumerate(ji.bufs):
        if b is not None and b not in input_rawbuffers:
          msg(icb_command, "setKernelBuffer:offset:atIndex:", b._buf.buf, b._buf.offset, i)
          all_resources.append(b._buf.buf)
      for i,v in enumerate(prg.p.vars): msg(icb_command, "setKernelBuffer:offset:atIndex:", self.int_buf.buf, self.vars.index(v)*4, len(ji.bufs)+i)

      global_size, local_size = prg.p.launch_dims(var_vals)
      msg(icb_command, "concurrentDispatchThreadgroups:threadsPerThreadgroup:", to_struct(*global_size), to_struct(*local_size))
      msg(icb_command, "setBarrier")

    self.all_resources = dedup(all_resources)
    self.all_pipelines = dedup(all_pipelines)
    self.command_buffer: Any = None
    if len(self.vars): self.int_buf_view = self.dev.allocator._as_buffer(self.int_buf).cast('i')
    self.range = to_struct(0, len(jit_cache))

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:

    if self.command_buffer is not None and self.command_buffer in self.dev.mtl_buffers_in_flight: wait_check(self.command_buffer)
    all_resources = dedup(self.all_resources + [x._buf.buf for x in input_rawbuffers])

    for (j,i),input_idx in self.input_replace.items():
      computeCommand = msg(self.icb, "indirectComputeCommandAtIndex:", j, restype=objc_id)
      msg(computeCommand, "setKernelBuffer:offset:atIndex:", input_rawbuffers[input_idx]._buf.buf,
                                                                                 input_rawbuffers[input_idx]._buf.offset, i)

    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      computeCommand = msg(self.icb, "indirectComputeCommandAtIndex:", j, restype=objc_id)
      msg(computeCommand, "concurrentDispatchThreadgroups:threadsPerThreadgroup:", to_struct(*global_dims), to_struct(*local_dims))
    for j, var in enumerate(self.vars): self.int_buf_view[j] = var_vals[var]

    command_buffer = msg(self.dev.mtl_queue, "commandBuffer", restype=objc_instance)
    encoder = msg(command_buffer, "computeCommandEncoder", restype=objc_instance)
    msg(encoder, "useResources:count:usage:", (objc_id * len(all_resources))(*all_resources), len(all_resources),
        MTLResourceUsage.MTLResourceUsageRead | MTLResourceUsage.MTLResourceUsageWrite)

    # NOTE: the pipelines likely need to be added to the used resources to fix the crash on M1/M2, but I haven't figured out how
    # this is a O(n) hack to get them used. what should work is:
    #encoder.useResources_count_usage_(self.all_pipelines, len(self.all_pipelines), Metal.MTLResourceUsageRead)
    # but it fails with "Invalid Resource (00000009:kIOGPUCommandBufferCallbackErrorInvalidResource)"
    # to repro the crash (which can also crash other running GPU apps), run with FIX_METAL_ICB=0
    if getenv("FIX_METAL_ICB", self.needs_icb_fix):
      for ps in self.all_pipelines:
        msg(encoder, "setComputePipelineState:", ps)
        msg(encoder, "dispatchThreadgroups:threadsPerThreadgroup:", to_struct(0,0,0), to_struct(0,0,0))

    msg(encoder, "executeCommandsInBuffer:withRange:", self.icb, self.range)
    msg(encoder, "endEncoding")
    msg(command_buffer, "commit")
    self.command_buffer = command_buffer

    if wait:
      wait_check(command_buffer)
      return elapsed_time(command_buffer)
    self.dev.mtl_buffers_in_flight.append(command_buffer)
    return None
