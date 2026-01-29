from typing import Any, cast
import ctypes, re, decimal
from tinygrad.dtype import dtypes
from tinygrad.helpers import dedup, getenv, merge_dicts, PROFILE
from tinygrad.device import Buffer, ProfileGraphEntry, ProfileGraphEvent
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.runtime.ops_metal import wait_check, to_ns_str
from tinygrad.runtime.autogen import metal
from tinygrad.runtime.support import objc

class MetalGraph(GraphRunner):
  def __init__(self, jit_cache: list[ExecItem], input_buffers: list[Buffer], var_vals: dict[str, int],
               orig_valid_positions: dict[int, set[int]]|None = None):
    super().__init__(jit_cache, input_buffers, var_vals, orig_valid_positions)
    if not all(isinstance(ji.prg, CompiledRunner) for ji in jit_cache): raise GraphException

    # create metal batch exec
    icb_descriptor = metal.MTLIndirectCommandBufferDescriptor.new()
    icb_descriptor.setCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
    icb_descriptor.setInheritBuffers(False)
    icb_descriptor.setInheritPipelineState(False)
    icb_descriptor.setMaxKernelBufferBindCount(31)

    self.icb = self.dev.sysdevice.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(icb_descriptor, len(jit_cache),
                                                                                                 metal.MTLResourceCPUCacheModeDefaultCache)
    if self.icb.value is None: raise GraphException("create indirect command buffer failed, does your system support this?")
    # TODO: needs categories
    icb_label = bytes(objc.msg("UTF8String", ctypes.c_char_p)(objc.msg("description")(self.icb).retained())).decode()
    self.needs_icb_fix = int((m := re.search(r'AGXG(\d+)XFamily', icb_label)) is None or int(m.group(1)) < 15) # not required on M3+

    self.fixedvars = merge_dicts([ji.fixedvars for ji in jit_cache])
    self.varlist = self.vars + list(self.fixedvars.keys())
    if len(self.varlist): self.int_buf = self.dev.allocator.alloc(len(self.varlist)*dtypes.int32.itemsize)

    all_pipelines, all_resources = [], [self.int_buf.buf] if len(self.varlist) else []
    for j,ji in enumerate(jit_cache):
      prg: CompiledRunner = cast(CompiledRunner, ji.prg)
      icb_command = self.icb.indirectComputeCommandAtIndex(j).retained()
      all_pipelines.append(prg._prg.pipeline_state)
      icb_command.setComputePipelineState(prg._prg.pipeline_state)
      for i,b in enumerate(ji.bufs):
        if b is not None and (j,i) not in self.input_replace:
          icb_command.setKernelBuffer_offset_atIndex(b._buf.buf, b._buf.offset, i)
          all_resources.append(b._buf.buf)
      for i,v in enumerate(prg.p.vars): icb_command.setKernelBuffer_offset_atIndex(self.int_buf.buf, self.varlist.index(v.expr)*4, len(ji.bufs)+i)

      global_size, local_size = prg.p.launch_dims(var_vals)
      icb_command.concurrentDispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(*global_size), metal.MTLSize(*local_size))
      icb_command.setBarrier()

    self.all_resources = dedup(all_resources)
    self.all_pipelines = dedup(all_pipelines)
    self.command_buffer: Any = None
    if len(self.varlist): self.int_buf_view = self.dev.allocator._as_buffer(self.int_buf).cast('i')
    for var in self.fixedvars: self.int_buf_view[self.varlist.index(var)] = self.fixedvars[var]
    self.range = metal.NSRange(0, len(jit_cache))

  def __call__(self, input_buffers: list[Buffer], var_vals: dict[str, int], wait=False) -> float|None:
    if self.command_buffer is not None and self.command_buffer in self.dev.mtl_buffers_in_flight: wait_check(self.command_buffer)
    # NOTE: old command buffer may not be inflight anymore
    if self.command_buffer is not None and PROFILE: self.collect_timestamps()

    all_resources = dedup(self.all_resources + [input_buffers[input_idx]._buf.buf for input_idx in self.input_replace.values()])
    for (j,i),input_idx in self.input_replace.items():
      computeCommand = self.icb.indirectComputeCommandAtIndex(j)
      computeCommand.setKernelBuffer_offset_atIndex(input_buffers[input_idx]._buf.buf, input_buffers[input_idx]._buf.offset, i)

    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      computeCommand = self.icb.indirectComputeCommandAtIndex(j)
      computeCommand.concurrentDispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(*global_dims), metal.MTLSize(*local_dims))
    for var in self.vars: self.int_buf_view[self.varlist.index(var)] = var_vals[var]

    command_buffer = self.dev.mtl_queue.commandBuffer().retained()
    encoder = command_buffer.computeCommandEncoder().retained()
    encoder.useResources_count_usage(ctypes.cast((metal.MTLBuffer * len(all_resources))(*all_resources), ctypes.POINTER(metal.MTLResource)),
                                     len(all_resources), metal.MTLResourceUsageRead | metal.MTLResourceUsageWrite)

    # NOTE: the pipelines likely need to be added to the used resources to fix the crash on M1/M2, but I haven't figured out how
    # this is a O(n) hack to get them used. what should work is:
    #encoder.useResources_count_usage_(self.all_pipelines, len(self.all_pipelines), Metal.MTLResourceUsageRead)
    # but it fails with "Invalid Resource (00000009:kIOGPUCommandBufferCallbackErrorInvalidResource)"
    # to repro the crash (which can also crash other running GPU apps), run with FIX_METAL_ICB=0
    if getenv("FIX_METAL_ICB", self.needs_icb_fix):
      for ps in self.all_pipelines:
        encoder.setComputePipelineState(ps)
        encoder.dispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(0,0,0), metal.MTLSize(0,0,0))

    encoder.executeCommandsInBuffer_withRange(self.icb, self.range)
    encoder.endEncoding()
    command_buffer.setLabel(to_ns_str(f"batched {len(self.jit_cache)}"))
    command_buffer.commit()
    self.command_buffer = command_buffer

    self.dev.mtl_buffers_in_flight.append(command_buffer)
    if wait:
      wait_check(command_buffer)
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    return None

  def collect_timestamps(self):
    # create a graph event and evenly space each program
    st, en = decimal.Decimal(self.command_buffer.GPUStartTime()) * 1000000, decimal.Decimal(self.command_buffer.GPUEndTime()) * 1000000
    ents = [ProfileGraphEntry(self.device, cast(CompiledRunner, ji.prg)._prg.name, i, i+1, is_copy=False) for i,ji in enumerate(self.jit_cache)]
    step = (en-st)/len(ents)
    self.dev.profile_events += [ProfileGraphEvent(ents, [], [st+step*i for i in range(len(ents)+1)])]

  def __del__(self):
    if PROFILE and self.command_buffer is not None:
      wait_check(self.command_buffer)
      self.collect_timestamps()
