# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
dll = c.DLL('opencl', 'OpenCL')
class struct__cl_platform_id(ctypes.Structure): pass
cl_platform_id: TypeAlias = c.POINTER[struct__cl_platform_id]
class struct__cl_device_id(ctypes.Structure): pass
cl_device_id: TypeAlias = c.POINTER[struct__cl_device_id]
class struct__cl_context(ctypes.Structure): pass
cl_context: TypeAlias = c.POINTER[struct__cl_context]
class struct__cl_command_queue(ctypes.Structure): pass
cl_command_queue: TypeAlias = c.POINTER[struct__cl_command_queue]
class struct__cl_mem(ctypes.Structure): pass
cl_mem: TypeAlias = c.POINTER[struct__cl_mem]
class struct__cl_program(ctypes.Structure): pass
cl_program: TypeAlias = c.POINTER[struct__cl_program]
class struct__cl_kernel(ctypes.Structure): pass
cl_kernel: TypeAlias = c.POINTER[struct__cl_kernel]
class struct__cl_event(ctypes.Structure): pass
cl_event: TypeAlias = c.POINTER[struct__cl_event]
class struct__cl_sampler(ctypes.Structure): pass
cl_sampler: TypeAlias = c.POINTER[struct__cl_sampler]
cl_bool: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_bitfield: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_properties: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_device_type: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_platform_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_device_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_device_fp_config: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_device_mem_cache_type: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_device_local_mem_type: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_device_exec_capabilities: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_device_svm_capabilities: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_command_queue_properties: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_device_partition_property: TypeAlias = Annotated[int, ctypes.c_int64]
cl_device_affinity_domain: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_context_properties: TypeAlias = Annotated[int, ctypes.c_int64]
cl_context_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_queue_properties: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_command_queue_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_channel_order: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_channel_type: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_mem_flags: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_svm_mem_flags: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_mem_object_type: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_mem_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_mem_migration_flags: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_image_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_buffer_create_type: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_addressing_mode: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_filter_mode: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_sampler_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_map_flags: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_pipe_properties: TypeAlias = Annotated[int, ctypes.c_int64]
cl_pipe_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_program_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_program_build_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_program_binary_type: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_build_status: TypeAlias = Annotated[int, ctypes.c_int32]
cl_kernel_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_kernel_arg_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_kernel_arg_address_qualifier: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_kernel_arg_access_qualifier: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_kernel_arg_type_qualifier: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_kernel_work_group_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_kernel_sub_group_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_event_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_command_type: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_profiling_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_sampler_properties: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_kernel_exec_info: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_device_atomic_capabilities: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_device_device_enqueue_capabilities: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_khronos_vendor_id: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_mem_properties: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_version: TypeAlias = Annotated[int, ctypes.c_uint32]
@c.record
class struct__cl_image_format(c.Struct):
  SIZE = 8
  image_channel_order: Annotated[cl_channel_order, 0]
  image_channel_data_type: Annotated[cl_channel_type, 4]
cl_image_format: TypeAlias = struct__cl_image_format
@c.record
class struct__cl_image_desc(c.Struct):
  SIZE = 72
  image_type: Annotated[cl_mem_object_type, 0]
  image_width: Annotated[size_t, 8]
  image_height: Annotated[size_t, 16]
  image_depth: Annotated[size_t, 24]
  image_array_size: Annotated[size_t, 32]
  image_row_pitch: Annotated[size_t, 40]
  image_slice_pitch: Annotated[size_t, 48]
  num_mip_levels: Annotated[cl_uint, 56]
  num_samples: Annotated[cl_uint, 60]
  buffer: Annotated[cl_mem, 64]
  mem_object: Annotated[cl_mem, 64]
size_t: TypeAlias = Annotated[int, ctypes.c_uint64]
cl_uint: TypeAlias = Annotated[int, ctypes.c_uint32]
cl_image_desc: TypeAlias = struct__cl_image_desc
@c.record
class struct__cl_buffer_region(c.Struct):
  SIZE = 16
  origin: Annotated[size_t, 0]
  size: Annotated[size_t, 8]
cl_buffer_region: TypeAlias = struct__cl_buffer_region
@c.record
class struct__cl_name_version(c.Struct):
  SIZE = 68
  version: Annotated[cl_version, 0]
  name: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[64]], 4]
cl_name_version: TypeAlias = struct__cl_name_version
cl_int: TypeAlias = Annotated[int, ctypes.c_int32]
@dll.bind
def clGetPlatformIDs(num_entries:cl_uint, platforms:c.POINTER[cl_platform_id], num_platforms:c.POINTER[cl_uint]) -> cl_int: ...
@dll.bind
def clGetPlatformInfo(platform:cl_platform_id, param_name:cl_platform_info, param_value_size:size_t, param_value:ctypes.c_void_p, param_value_size_ret:c.POINTER[size_t]) -> cl_int: ...
@dll.bind
def clGetDeviceIDs(platform:cl_platform_id, device_type:cl_device_type, num_entries:cl_uint, devices:c.POINTER[cl_device_id], num_devices:c.POINTER[cl_uint]) -> cl_int: ...
@dll.bind
def clGetDeviceInfo(device:cl_device_id, param_name:cl_device_info, param_value_size:size_t, param_value:ctypes.c_void_p, param_value_size_ret:c.POINTER[size_t]) -> cl_int: ...
@dll.bind
def clCreateSubDevices(in_device:cl_device_id, properties:c.POINTER[cl_device_partition_property], num_devices:cl_uint, out_devices:c.POINTER[cl_device_id], num_devices_ret:c.POINTER[cl_uint]) -> cl_int: ...
@dll.bind
def clRetainDevice(device:cl_device_id) -> cl_int: ...
@dll.bind
def clReleaseDevice(device:cl_device_id) -> cl_int: ...
@dll.bind
def clSetDefaultDeviceCommandQueue(context:cl_context, device:cl_device_id, command_queue:cl_command_queue) -> cl_int: ...
cl_ulong: TypeAlias = Annotated[int, ctypes.c_uint64]
@dll.bind
def clGetDeviceAndHostTimer(device:cl_device_id, device_timestamp:c.POINTER[cl_ulong], host_timestamp:c.POINTER[cl_ulong]) -> cl_int: ...
@dll.bind
def clGetHostTimer(device:cl_device_id, host_timestamp:c.POINTER[cl_ulong]) -> cl_int: ...
@dll.bind
def clCreateContext(properties:c.POINTER[cl_context_properties], num_devices:cl_uint, devices:c.POINTER[cl_device_id], pfn_notify:c.CFUNCTYPE[None, [c.POINTER[Annotated[bytes, ctypes.c_char]], ctypes.c_void_p, size_t, ctypes.c_void_p]], user_data:ctypes.c_void_p, errcode_ret:c.POINTER[cl_int]) -> cl_context: ...
@dll.bind
def clCreateContextFromType(properties:c.POINTER[cl_context_properties], device_type:cl_device_type, pfn_notify:c.CFUNCTYPE[None, [c.POINTER[Annotated[bytes, ctypes.c_char]], ctypes.c_void_p, size_t, ctypes.c_void_p]], user_data:ctypes.c_void_p, errcode_ret:c.POINTER[cl_int]) -> cl_context: ...
@dll.bind
def clRetainContext(context:cl_context) -> cl_int: ...
@dll.bind
def clReleaseContext(context:cl_context) -> cl_int: ...
@dll.bind
def clGetContextInfo(context:cl_context, param_name:cl_context_info, param_value_size:size_t, param_value:ctypes.c_void_p, param_value_size_ret:c.POINTER[size_t]) -> cl_int: ...
@dll.bind
def clSetContextDestructorCallback(context:cl_context, pfn_notify:c.CFUNCTYPE[None, [cl_context, ctypes.c_void_p]], user_data:ctypes.c_void_p) -> cl_int: ...
@dll.bind
def clCreateCommandQueueWithProperties(context:cl_context, device:cl_device_id, properties:c.POINTER[cl_queue_properties], errcode_ret:c.POINTER[cl_int]) -> cl_command_queue: ...
@dll.bind
def clRetainCommandQueue(command_queue:cl_command_queue) -> cl_int: ...
@dll.bind
def clReleaseCommandQueue(command_queue:cl_command_queue) -> cl_int: ...
@dll.bind
def clGetCommandQueueInfo(command_queue:cl_command_queue, param_name:cl_command_queue_info, param_value_size:size_t, param_value:ctypes.c_void_p, param_value_size_ret:c.POINTER[size_t]) -> cl_int: ...
@dll.bind
def clCreateBuffer(context:cl_context, flags:cl_mem_flags, size:size_t, host_ptr:ctypes.c_void_p, errcode_ret:c.POINTER[cl_int]) -> cl_mem: ...
@dll.bind
def clCreateSubBuffer(buffer:cl_mem, flags:cl_mem_flags, buffer_create_type:cl_buffer_create_type, buffer_create_info:ctypes.c_void_p, errcode_ret:c.POINTER[cl_int]) -> cl_mem: ...
@dll.bind
def clCreateImage(context:cl_context, flags:cl_mem_flags, image_format:c.POINTER[cl_image_format], image_desc:c.POINTER[cl_image_desc], host_ptr:ctypes.c_void_p, errcode_ret:c.POINTER[cl_int]) -> cl_mem: ...
@dll.bind
def clCreatePipe(context:cl_context, flags:cl_mem_flags, pipe_packet_size:cl_uint, pipe_max_packets:cl_uint, properties:c.POINTER[cl_pipe_properties], errcode_ret:c.POINTER[cl_int]) -> cl_mem: ...
@dll.bind
def clCreateBufferWithProperties(context:cl_context, properties:c.POINTER[cl_mem_properties], flags:cl_mem_flags, size:size_t, host_ptr:ctypes.c_void_p, errcode_ret:c.POINTER[cl_int]) -> cl_mem: ...
@dll.bind
def clCreateImageWithProperties(context:cl_context, properties:c.POINTER[cl_mem_properties], flags:cl_mem_flags, image_format:c.POINTER[cl_image_format], image_desc:c.POINTER[cl_image_desc], host_ptr:ctypes.c_void_p, errcode_ret:c.POINTER[cl_int]) -> cl_mem: ...
@dll.bind
def clRetainMemObject(memobj:cl_mem) -> cl_int: ...
@dll.bind
def clReleaseMemObject(memobj:cl_mem) -> cl_int: ...
@dll.bind
def clGetSupportedImageFormats(context:cl_context, flags:cl_mem_flags, image_type:cl_mem_object_type, num_entries:cl_uint, image_formats:c.POINTER[cl_image_format], num_image_formats:c.POINTER[cl_uint]) -> cl_int: ...
@dll.bind
def clGetMemObjectInfo(memobj:cl_mem, param_name:cl_mem_info, param_value_size:size_t, param_value:ctypes.c_void_p, param_value_size_ret:c.POINTER[size_t]) -> cl_int: ...
@dll.bind
def clGetImageInfo(image:cl_mem, param_name:cl_image_info, param_value_size:size_t, param_value:ctypes.c_void_p, param_value_size_ret:c.POINTER[size_t]) -> cl_int: ...
@dll.bind
def clGetPipeInfo(pipe:cl_mem, param_name:cl_pipe_info, param_value_size:size_t, param_value:ctypes.c_void_p, param_value_size_ret:c.POINTER[size_t]) -> cl_int: ...
@dll.bind
def clSetMemObjectDestructorCallback(memobj:cl_mem, pfn_notify:c.CFUNCTYPE[None, [cl_mem, ctypes.c_void_p]], user_data:ctypes.c_void_p) -> cl_int: ...
@dll.bind
def clSVMAlloc(context:cl_context, flags:cl_svm_mem_flags, size:size_t, alignment:cl_uint) -> ctypes.c_void_p: ...
@dll.bind
def clSVMFree(context:cl_context, svm_pointer:ctypes.c_void_p) -> None: ...
@dll.bind
def clCreateSamplerWithProperties(context:cl_context, sampler_properties:c.POINTER[cl_sampler_properties], errcode_ret:c.POINTER[cl_int]) -> cl_sampler: ...
@dll.bind
def clRetainSampler(sampler:cl_sampler) -> cl_int: ...
@dll.bind
def clReleaseSampler(sampler:cl_sampler) -> cl_int: ...
@dll.bind
def clGetSamplerInfo(sampler:cl_sampler, param_name:cl_sampler_info, param_value_size:size_t, param_value:ctypes.c_void_p, param_value_size_ret:c.POINTER[size_t]) -> cl_int: ...
@dll.bind
def clCreateProgramWithSource(context:cl_context, count:cl_uint, strings:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], lengths:c.POINTER[size_t], errcode_ret:c.POINTER[cl_int]) -> cl_program: ...
@dll.bind
def clCreateProgramWithBinary(context:cl_context, num_devices:cl_uint, device_list:c.POINTER[cl_device_id], lengths:c.POINTER[size_t], binaries:c.POINTER[c.POINTER[Annotated[int, ctypes.c_ubyte]]], binary_status:c.POINTER[cl_int], errcode_ret:c.POINTER[cl_int]) -> cl_program: ...
@dll.bind
def clCreateProgramWithBuiltInKernels(context:cl_context, num_devices:cl_uint, device_list:c.POINTER[cl_device_id], kernel_names:c.POINTER[Annotated[bytes, ctypes.c_char]], errcode_ret:c.POINTER[cl_int]) -> cl_program: ...
@dll.bind
def clCreateProgramWithIL(context:cl_context, il:ctypes.c_void_p, length:size_t, errcode_ret:c.POINTER[cl_int]) -> cl_program: ...
@dll.bind
def clRetainProgram(program:cl_program) -> cl_int: ...
@dll.bind
def clReleaseProgram(program:cl_program) -> cl_int: ...
@dll.bind
def clBuildProgram(program:cl_program, num_devices:cl_uint, device_list:c.POINTER[cl_device_id], options:c.POINTER[Annotated[bytes, ctypes.c_char]], pfn_notify:c.CFUNCTYPE[None, [cl_program, ctypes.c_void_p]], user_data:ctypes.c_void_p) -> cl_int: ...
@dll.bind
def clCompileProgram(program:cl_program, num_devices:cl_uint, device_list:c.POINTER[cl_device_id], options:c.POINTER[Annotated[bytes, ctypes.c_char]], num_input_headers:cl_uint, input_headers:c.POINTER[cl_program], header_include_names:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], pfn_notify:c.CFUNCTYPE[None, [cl_program, ctypes.c_void_p]], user_data:ctypes.c_void_p) -> cl_int: ...
@dll.bind
def clLinkProgram(context:cl_context, num_devices:cl_uint, device_list:c.POINTER[cl_device_id], options:c.POINTER[Annotated[bytes, ctypes.c_char]], num_input_programs:cl_uint, input_programs:c.POINTER[cl_program], pfn_notify:c.CFUNCTYPE[None, [cl_program, ctypes.c_void_p]], user_data:ctypes.c_void_p, errcode_ret:c.POINTER[cl_int]) -> cl_program: ...
@dll.bind
def clSetProgramReleaseCallback(program:cl_program, pfn_notify:c.CFUNCTYPE[None, [cl_program, ctypes.c_void_p]], user_data:ctypes.c_void_p) -> cl_int: ...
@dll.bind
def clSetProgramSpecializationConstant(program:cl_program, spec_id:cl_uint, spec_size:size_t, spec_value:ctypes.c_void_p) -> cl_int: ...
@dll.bind
def clUnloadPlatformCompiler(platform:cl_platform_id) -> cl_int: ...
@dll.bind
def clGetProgramInfo(program:cl_program, param_name:cl_program_info, param_value_size:size_t, param_value:ctypes.c_void_p, param_value_size_ret:c.POINTER[size_t]) -> cl_int: ...
@dll.bind
def clGetProgramBuildInfo(program:cl_program, device:cl_device_id, param_name:cl_program_build_info, param_value_size:size_t, param_value:ctypes.c_void_p, param_value_size_ret:c.POINTER[size_t]) -> cl_int: ...
@dll.bind
def clCreateKernel(program:cl_program, kernel_name:c.POINTER[Annotated[bytes, ctypes.c_char]], errcode_ret:c.POINTER[cl_int]) -> cl_kernel: ...
@dll.bind
def clCreateKernelsInProgram(program:cl_program, num_kernels:cl_uint, kernels:c.POINTER[cl_kernel], num_kernels_ret:c.POINTER[cl_uint]) -> cl_int: ...
@dll.bind
def clCloneKernel(source_kernel:cl_kernel, errcode_ret:c.POINTER[cl_int]) -> cl_kernel: ...
@dll.bind
def clRetainKernel(kernel:cl_kernel) -> cl_int: ...
@dll.bind
def clReleaseKernel(kernel:cl_kernel) -> cl_int: ...
@dll.bind
def clSetKernelArg(kernel:cl_kernel, arg_index:cl_uint, arg_size:size_t, arg_value:ctypes.c_void_p) -> cl_int: ...
@dll.bind
def clSetKernelArgSVMPointer(kernel:cl_kernel, arg_index:cl_uint, arg_value:ctypes.c_void_p) -> cl_int: ...
@dll.bind
def clSetKernelExecInfo(kernel:cl_kernel, param_name:cl_kernel_exec_info, param_value_size:size_t, param_value:ctypes.c_void_p) -> cl_int: ...
@dll.bind
def clGetKernelInfo(kernel:cl_kernel, param_name:cl_kernel_info, param_value_size:size_t, param_value:ctypes.c_void_p, param_value_size_ret:c.POINTER[size_t]) -> cl_int: ...
@dll.bind
def clGetKernelArgInfo(kernel:cl_kernel, arg_indx:cl_uint, param_name:cl_kernel_arg_info, param_value_size:size_t, param_value:ctypes.c_void_p, param_value_size_ret:c.POINTER[size_t]) -> cl_int: ...
@dll.bind
def clGetKernelWorkGroupInfo(kernel:cl_kernel, device:cl_device_id, param_name:cl_kernel_work_group_info, param_value_size:size_t, param_value:ctypes.c_void_p, param_value_size_ret:c.POINTER[size_t]) -> cl_int: ...
@dll.bind
def clGetKernelSubGroupInfo(kernel:cl_kernel, device:cl_device_id, param_name:cl_kernel_sub_group_info, input_value_size:size_t, input_value:ctypes.c_void_p, param_value_size:size_t, param_value:ctypes.c_void_p, param_value_size_ret:c.POINTER[size_t]) -> cl_int: ...
@dll.bind
def clWaitForEvents(num_events:cl_uint, event_list:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clGetEventInfo(event:cl_event, param_name:cl_event_info, param_value_size:size_t, param_value:ctypes.c_void_p, param_value_size_ret:c.POINTER[size_t]) -> cl_int: ...
@dll.bind
def clCreateUserEvent(context:cl_context, errcode_ret:c.POINTER[cl_int]) -> cl_event: ...
@dll.bind
def clRetainEvent(event:cl_event) -> cl_int: ...
@dll.bind
def clReleaseEvent(event:cl_event) -> cl_int: ...
@dll.bind
def clSetUserEventStatus(event:cl_event, execution_status:cl_int) -> cl_int: ...
@dll.bind
def clSetEventCallback(event:cl_event, command_exec_callback_type:cl_int, pfn_notify:c.CFUNCTYPE[None, [cl_event, cl_int, ctypes.c_void_p]], user_data:ctypes.c_void_p) -> cl_int: ...
@dll.bind
def clGetEventProfilingInfo(event:cl_event, param_name:cl_profiling_info, param_value_size:size_t, param_value:ctypes.c_void_p, param_value_size_ret:c.POINTER[size_t]) -> cl_int: ...
@dll.bind
def clFlush(command_queue:cl_command_queue) -> cl_int: ...
@dll.bind
def clFinish(command_queue:cl_command_queue) -> cl_int: ...
@dll.bind
def clEnqueueReadBuffer(command_queue:cl_command_queue, buffer:cl_mem, blocking_read:cl_bool, offset:size_t, size:size_t, ptr:ctypes.c_void_p, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueReadBufferRect(command_queue:cl_command_queue, buffer:cl_mem, blocking_read:cl_bool, buffer_origin:c.POINTER[size_t], host_origin:c.POINTER[size_t], region:c.POINTER[size_t], buffer_row_pitch:size_t, buffer_slice_pitch:size_t, host_row_pitch:size_t, host_slice_pitch:size_t, ptr:ctypes.c_void_p, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueWriteBuffer(command_queue:cl_command_queue, buffer:cl_mem, blocking_write:cl_bool, offset:size_t, size:size_t, ptr:ctypes.c_void_p, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueWriteBufferRect(command_queue:cl_command_queue, buffer:cl_mem, blocking_write:cl_bool, buffer_origin:c.POINTER[size_t], host_origin:c.POINTER[size_t], region:c.POINTER[size_t], buffer_row_pitch:size_t, buffer_slice_pitch:size_t, host_row_pitch:size_t, host_slice_pitch:size_t, ptr:ctypes.c_void_p, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueFillBuffer(command_queue:cl_command_queue, buffer:cl_mem, pattern:ctypes.c_void_p, pattern_size:size_t, offset:size_t, size:size_t, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueCopyBuffer(command_queue:cl_command_queue, src_buffer:cl_mem, dst_buffer:cl_mem, src_offset:size_t, dst_offset:size_t, size:size_t, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueCopyBufferRect(command_queue:cl_command_queue, src_buffer:cl_mem, dst_buffer:cl_mem, src_origin:c.POINTER[size_t], dst_origin:c.POINTER[size_t], region:c.POINTER[size_t], src_row_pitch:size_t, src_slice_pitch:size_t, dst_row_pitch:size_t, dst_slice_pitch:size_t, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueReadImage(command_queue:cl_command_queue, image:cl_mem, blocking_read:cl_bool, origin:c.POINTER[size_t], region:c.POINTER[size_t], row_pitch:size_t, slice_pitch:size_t, ptr:ctypes.c_void_p, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueWriteImage(command_queue:cl_command_queue, image:cl_mem, blocking_write:cl_bool, origin:c.POINTER[size_t], region:c.POINTER[size_t], input_row_pitch:size_t, input_slice_pitch:size_t, ptr:ctypes.c_void_p, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueFillImage(command_queue:cl_command_queue, image:cl_mem, fill_color:ctypes.c_void_p, origin:c.POINTER[size_t], region:c.POINTER[size_t], num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueCopyImage(command_queue:cl_command_queue, src_image:cl_mem, dst_image:cl_mem, src_origin:c.POINTER[size_t], dst_origin:c.POINTER[size_t], region:c.POINTER[size_t], num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueCopyImageToBuffer(command_queue:cl_command_queue, src_image:cl_mem, dst_buffer:cl_mem, src_origin:c.POINTER[size_t], region:c.POINTER[size_t], dst_offset:size_t, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueCopyBufferToImage(command_queue:cl_command_queue, src_buffer:cl_mem, dst_image:cl_mem, src_offset:size_t, dst_origin:c.POINTER[size_t], region:c.POINTER[size_t], num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueMapBuffer(command_queue:cl_command_queue, buffer:cl_mem, blocking_map:cl_bool, map_flags:cl_map_flags, offset:size_t, size:size_t, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event], errcode_ret:c.POINTER[cl_int]) -> ctypes.c_void_p: ...
@dll.bind
def clEnqueueMapImage(command_queue:cl_command_queue, image:cl_mem, blocking_map:cl_bool, map_flags:cl_map_flags, origin:c.POINTER[size_t], region:c.POINTER[size_t], image_row_pitch:c.POINTER[size_t], image_slice_pitch:c.POINTER[size_t], num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event], errcode_ret:c.POINTER[cl_int]) -> ctypes.c_void_p: ...
@dll.bind
def clEnqueueUnmapMemObject(command_queue:cl_command_queue, memobj:cl_mem, mapped_ptr:ctypes.c_void_p, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueMigrateMemObjects(command_queue:cl_command_queue, num_mem_objects:cl_uint, mem_objects:c.POINTER[cl_mem], flags:cl_mem_migration_flags, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueNDRangeKernel(command_queue:cl_command_queue, kernel:cl_kernel, work_dim:cl_uint, global_work_offset:c.POINTER[size_t], global_work_size:c.POINTER[size_t], local_work_size:c.POINTER[size_t], num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueNativeKernel(command_queue:cl_command_queue, user_func:c.CFUNCTYPE[None, [ctypes.c_void_p]], args:ctypes.c_void_p, cb_args:size_t, num_mem_objects:cl_uint, mem_list:c.POINTER[cl_mem], args_mem_loc:c.POINTER[ctypes.c_void_p], num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueMarkerWithWaitList(command_queue:cl_command_queue, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueBarrierWithWaitList(command_queue:cl_command_queue, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueSVMFree(command_queue:cl_command_queue, num_svm_pointers:cl_uint, svm_pointers:c.Array[ctypes.c_void_p, Literal[0]], pfn_free_func:c.CFUNCTYPE[None, [cl_command_queue, cl_uint, c.Array[ctypes.c_void_p, Literal[0]], ctypes.c_void_p]], user_data:ctypes.c_void_p, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueSVMMemcpy(command_queue:cl_command_queue, blocking_copy:cl_bool, dst_ptr:ctypes.c_void_p, src_ptr:ctypes.c_void_p, size:size_t, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueSVMMemFill(command_queue:cl_command_queue, svm_ptr:ctypes.c_void_p, pattern:ctypes.c_void_p, pattern_size:size_t, size:size_t, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueSVMMap(command_queue:cl_command_queue, blocking_map:cl_bool, flags:cl_map_flags, svm_ptr:ctypes.c_void_p, size:size_t, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueSVMUnmap(command_queue:cl_command_queue, svm_ptr:ctypes.c_void_p, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueSVMMigrateMem(command_queue:cl_command_queue, num_svm_pointers:cl_uint, svm_pointers:c.POINTER[ctypes.c_void_p], sizes:c.POINTER[size_t], flags:cl_mem_migration_flags, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clGetExtensionFunctionAddressForPlatform(platform:cl_platform_id, func_name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> ctypes.c_void_p: ...
@dll.bind
def clCreateImage2D(context:cl_context, flags:cl_mem_flags, image_format:c.POINTER[cl_image_format], image_width:size_t, image_height:size_t, image_row_pitch:size_t, host_ptr:ctypes.c_void_p, errcode_ret:c.POINTER[cl_int]) -> cl_mem: ...
@dll.bind
def clCreateImage3D(context:cl_context, flags:cl_mem_flags, image_format:c.POINTER[cl_image_format], image_width:size_t, image_height:size_t, image_depth:size_t, image_row_pitch:size_t, image_slice_pitch:size_t, host_ptr:ctypes.c_void_p, errcode_ret:c.POINTER[cl_int]) -> cl_mem: ...
@dll.bind
def clEnqueueMarker(command_queue:cl_command_queue, event:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueWaitForEvents(command_queue:cl_command_queue, num_events:cl_uint, event_list:c.POINTER[cl_event]) -> cl_int: ...
@dll.bind
def clEnqueueBarrier(command_queue:cl_command_queue) -> cl_int: ...
@dll.bind
def clUnloadCompiler() -> cl_int: ...
@dll.bind
def clGetExtensionFunctionAddress(func_name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> ctypes.c_void_p: ...
@dll.bind
def clCreateCommandQueue(context:cl_context, device:cl_device_id, properties:cl_command_queue_properties, errcode_ret:c.POINTER[cl_int]) -> cl_command_queue: ...
@dll.bind
def clCreateSampler(context:cl_context, normalized_coords:cl_bool, addressing_mode:cl_addressing_mode, filter_mode:cl_filter_mode, errcode_ret:c.POINTER[cl_int]) -> cl_sampler: ...
@dll.bind
def clEnqueueTask(command_queue:cl_command_queue, kernel:cl_kernel, num_events_in_wait_list:cl_uint, event_wait_list:c.POINTER[cl_event], event:c.POINTER[cl_event]) -> cl_int: ...
c.init_records()
CL_NAME_VERSION_MAX_NAME_SIZE = 64 # type: ignore
CL_SUCCESS = 0 # type: ignore
CL_DEVICE_NOT_FOUND = -1 # type: ignore
CL_DEVICE_NOT_AVAILABLE = -2 # type: ignore
CL_COMPILER_NOT_AVAILABLE = -3 # type: ignore
CL_MEM_OBJECT_ALLOCATION_FAILURE = -4 # type: ignore
CL_OUT_OF_RESOURCES = -5 # type: ignore
CL_OUT_OF_HOST_MEMORY = -6 # type: ignore
CL_PROFILING_INFO_NOT_AVAILABLE = -7 # type: ignore
CL_MEM_COPY_OVERLAP = -8 # type: ignore
CL_IMAGE_FORMAT_MISMATCH = -9 # type: ignore
CL_IMAGE_FORMAT_NOT_SUPPORTED = -10 # type: ignore
CL_BUILD_PROGRAM_FAILURE = -11 # type: ignore
CL_MAP_FAILURE = -12 # type: ignore
CL_MISALIGNED_SUB_BUFFER_OFFSET = -13 # type: ignore
CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = -14 # type: ignore
CL_COMPILE_PROGRAM_FAILURE = -15 # type: ignore
CL_LINKER_NOT_AVAILABLE = -16 # type: ignore
CL_LINK_PROGRAM_FAILURE = -17 # type: ignore
CL_DEVICE_PARTITION_FAILED = -18 # type: ignore
CL_KERNEL_ARG_INFO_NOT_AVAILABLE = -19 # type: ignore
CL_INVALID_VALUE = -30 # type: ignore
CL_INVALID_DEVICE_TYPE = -31 # type: ignore
CL_INVALID_PLATFORM = -32 # type: ignore
CL_INVALID_DEVICE = -33 # type: ignore
CL_INVALID_CONTEXT = -34 # type: ignore
CL_INVALID_QUEUE_PROPERTIES = -35 # type: ignore
CL_INVALID_COMMAND_QUEUE = -36 # type: ignore
CL_INVALID_HOST_PTR = -37 # type: ignore
CL_INVALID_MEM_OBJECT = -38 # type: ignore
CL_INVALID_IMAGE_FORMAT_DESCRIPTOR = -39 # type: ignore
CL_INVALID_IMAGE_SIZE = -40 # type: ignore
CL_INVALID_SAMPLER = -41 # type: ignore
CL_INVALID_BINARY = -42 # type: ignore
CL_INVALID_BUILD_OPTIONS = -43 # type: ignore
CL_INVALID_PROGRAM = -44 # type: ignore
CL_INVALID_PROGRAM_EXECUTABLE = -45 # type: ignore
CL_INVALID_KERNEL_NAME = -46 # type: ignore
CL_INVALID_KERNEL_DEFINITION = -47 # type: ignore
CL_INVALID_KERNEL = -48 # type: ignore
CL_INVALID_ARG_INDEX = -49 # type: ignore
CL_INVALID_ARG_VALUE = -50 # type: ignore
CL_INVALID_ARG_SIZE = -51 # type: ignore
CL_INVALID_KERNEL_ARGS = -52 # type: ignore
CL_INVALID_WORK_DIMENSION = -53 # type: ignore
CL_INVALID_WORK_GROUP_SIZE = -54 # type: ignore
CL_INVALID_WORK_ITEM_SIZE = -55 # type: ignore
CL_INVALID_GLOBAL_OFFSET = -56 # type: ignore
CL_INVALID_EVENT_WAIT_LIST = -57 # type: ignore
CL_INVALID_EVENT = -58 # type: ignore
CL_INVALID_OPERATION = -59 # type: ignore
CL_INVALID_GL_OBJECT = -60 # type: ignore
CL_INVALID_BUFFER_SIZE = -61 # type: ignore
CL_INVALID_MIP_LEVEL = -62 # type: ignore
CL_INVALID_GLOBAL_WORK_SIZE = -63 # type: ignore
CL_INVALID_PROPERTY = -64 # type: ignore
CL_INVALID_IMAGE_DESCRIPTOR = -65 # type: ignore
CL_INVALID_COMPILER_OPTIONS = -66 # type: ignore
CL_INVALID_LINKER_OPTIONS = -67 # type: ignore
CL_INVALID_DEVICE_PARTITION_COUNT = -68 # type: ignore
CL_INVALID_PIPE_SIZE = -69 # type: ignore
CL_INVALID_DEVICE_QUEUE = -70 # type: ignore
CL_INVALID_SPEC_ID = -71 # type: ignore
CL_MAX_SIZE_RESTRICTION_EXCEEDED = -72 # type: ignore
CL_FALSE = 0 # type: ignore
CL_TRUE = 1 # type: ignore
CL_BLOCKING = CL_TRUE # type: ignore
CL_NON_BLOCKING = CL_FALSE # type: ignore
CL_PLATFORM_PROFILE = 0x0900 # type: ignore
CL_PLATFORM_VERSION = 0x0901 # type: ignore
CL_PLATFORM_NAME = 0x0902 # type: ignore
CL_PLATFORM_VENDOR = 0x0903 # type: ignore
CL_PLATFORM_EXTENSIONS = 0x0904 # type: ignore
CL_PLATFORM_HOST_TIMER_RESOLUTION = 0x0905 # type: ignore
CL_PLATFORM_NUMERIC_VERSION = 0x0906 # type: ignore
CL_PLATFORM_EXTENSIONS_WITH_VERSION = 0x0907 # type: ignore
CL_DEVICE_TYPE_DEFAULT = (1 << 0) # type: ignore
CL_DEVICE_TYPE_CPU = (1 << 1) # type: ignore
CL_DEVICE_TYPE_GPU = (1 << 2) # type: ignore
CL_DEVICE_TYPE_ACCELERATOR = (1 << 3) # type: ignore
CL_DEVICE_TYPE_CUSTOM = (1 << 4) # type: ignore
CL_DEVICE_TYPE_ALL = 0xFFFFFFFF # type: ignore
CL_DEVICE_TYPE = 0x1000 # type: ignore
CL_DEVICE_VENDOR_ID = 0x1001 # type: ignore
CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002 # type: ignore
CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = 0x1003 # type: ignore
CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004 # type: ignore
CL_DEVICE_MAX_WORK_ITEM_SIZES = 0x1005 # type: ignore
CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR = 0x1006 # type: ignore
CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT = 0x1007 # type: ignore
CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT = 0x1008 # type: ignore
CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG = 0x1009 # type: ignore
CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT = 0x100A # type: ignore
CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE = 0x100B # type: ignore
CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100C # type: ignore
CL_DEVICE_ADDRESS_BITS = 0x100D # type: ignore
CL_DEVICE_MAX_READ_IMAGE_ARGS = 0x100E # type: ignore
CL_DEVICE_MAX_WRITE_IMAGE_ARGS = 0x100F # type: ignore
CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010 # type: ignore
CL_DEVICE_IMAGE2D_MAX_WIDTH = 0x1011 # type: ignore
CL_DEVICE_IMAGE2D_MAX_HEIGHT = 0x1012 # type: ignore
CL_DEVICE_IMAGE3D_MAX_WIDTH = 0x1013 # type: ignore
CL_DEVICE_IMAGE3D_MAX_HEIGHT = 0x1014 # type: ignore
CL_DEVICE_IMAGE3D_MAX_DEPTH = 0x1015 # type: ignore
CL_DEVICE_IMAGE_SUPPORT = 0x1016 # type: ignore
CL_DEVICE_MAX_PARAMETER_SIZE = 0x1017 # type: ignore
CL_DEVICE_MAX_SAMPLERS = 0x1018 # type: ignore
CL_DEVICE_MEM_BASE_ADDR_ALIGN = 0x1019 # type: ignore
CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE = 0x101A # type: ignore
CL_DEVICE_SINGLE_FP_CONFIG = 0x101B # type: ignore
CL_DEVICE_GLOBAL_MEM_CACHE_TYPE = 0x101C # type: ignore
CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE = 0x101D # type: ignore
CL_DEVICE_GLOBAL_MEM_CACHE_SIZE = 0x101E # type: ignore
CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F # type: ignore
CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = 0x1020 # type: ignore
CL_DEVICE_MAX_CONSTANT_ARGS = 0x1021 # type: ignore
CL_DEVICE_LOCAL_MEM_TYPE = 0x1022 # type: ignore
CL_DEVICE_LOCAL_MEM_SIZE = 0x1023 # type: ignore
CL_DEVICE_ERROR_CORRECTION_SUPPORT = 0x1024 # type: ignore
CL_DEVICE_PROFILING_TIMER_RESOLUTION = 0x1025 # type: ignore
CL_DEVICE_ENDIAN_LITTLE = 0x1026 # type: ignore
CL_DEVICE_AVAILABLE = 0x1027 # type: ignore
CL_DEVICE_COMPILER_AVAILABLE = 0x1028 # type: ignore
CL_DEVICE_EXECUTION_CAPABILITIES = 0x1029 # type: ignore
CL_DEVICE_QUEUE_PROPERTIES = 0x102A # type: ignore
CL_DEVICE_QUEUE_ON_HOST_PROPERTIES = 0x102A # type: ignore
CL_DEVICE_NAME = 0x102B # type: ignore
CL_DEVICE_VENDOR = 0x102C # type: ignore
CL_DRIVER_VERSION = 0x102D # type: ignore
CL_DEVICE_PROFILE = 0x102E # type: ignore
CL_DEVICE_VERSION = 0x102F # type: ignore
CL_DEVICE_EXTENSIONS = 0x1030 # type: ignore
CL_DEVICE_PLATFORM = 0x1031 # type: ignore
CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032 # type: ignore
CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF = 0x1034 # type: ignore
CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035 # type: ignore
CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR = 0x1036 # type: ignore
CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT = 0x1037 # type: ignore
CL_DEVICE_NATIVE_VECTOR_WIDTH_INT = 0x1038 # type: ignore
CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG = 0x1039 # type: ignore
CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT = 0x103A # type: ignore
CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE = 0x103B # type: ignore
CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF = 0x103C # type: ignore
CL_DEVICE_OPENCL_C_VERSION = 0x103D # type: ignore
CL_DEVICE_LINKER_AVAILABLE = 0x103E # type: ignore
CL_DEVICE_BUILT_IN_KERNELS = 0x103F # type: ignore
CL_DEVICE_IMAGE_MAX_BUFFER_SIZE = 0x1040 # type: ignore
CL_DEVICE_IMAGE_MAX_ARRAY_SIZE = 0x1041 # type: ignore
CL_DEVICE_PARENT_DEVICE = 0x1042 # type: ignore
CL_DEVICE_PARTITION_MAX_SUB_DEVICES = 0x1043 # type: ignore
CL_DEVICE_PARTITION_PROPERTIES = 0x1044 # type: ignore
CL_DEVICE_PARTITION_AFFINITY_DOMAIN = 0x1045 # type: ignore
CL_DEVICE_PARTITION_TYPE = 0x1046 # type: ignore
CL_DEVICE_REFERENCE_COUNT = 0x1047 # type: ignore
CL_DEVICE_PREFERRED_INTEROP_USER_SYNC = 0x1048 # type: ignore
CL_DEVICE_PRINTF_BUFFER_SIZE = 0x1049 # type: ignore
CL_DEVICE_IMAGE_PITCH_ALIGNMENT = 0x104A # type: ignore
CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT = 0x104B # type: ignore
CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS = 0x104C # type: ignore
CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE = 0x104D # type: ignore
CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES = 0x104E # type: ignore
CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE = 0x104F # type: ignore
CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE = 0x1050 # type: ignore
CL_DEVICE_MAX_ON_DEVICE_QUEUES = 0x1051 # type: ignore
CL_DEVICE_MAX_ON_DEVICE_EVENTS = 0x1052 # type: ignore
CL_DEVICE_SVM_CAPABILITIES = 0x1053 # type: ignore
CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE = 0x1054 # type: ignore
CL_DEVICE_MAX_PIPE_ARGS = 0x1055 # type: ignore
CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS = 0x1056 # type: ignore
CL_DEVICE_PIPE_MAX_PACKET_SIZE = 0x1057 # type: ignore
CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT = 0x1058 # type: ignore
CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT = 0x1059 # type: ignore
CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT = 0x105A # type: ignore
CL_DEVICE_IL_VERSION = 0x105B # type: ignore
CL_DEVICE_MAX_NUM_SUB_GROUPS = 0x105C # type: ignore
CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS = 0x105D # type: ignore
CL_DEVICE_NUMERIC_VERSION = 0x105E # type: ignore
CL_DEVICE_EXTENSIONS_WITH_VERSION = 0x1060 # type: ignore
CL_DEVICE_ILS_WITH_VERSION = 0x1061 # type: ignore
CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION = 0x1062 # type: ignore
CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES = 0x1063 # type: ignore
CL_DEVICE_ATOMIC_FENCE_CAPABILITIES = 0x1064 # type: ignore
CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT = 0x1065 # type: ignore
CL_DEVICE_OPENCL_C_ALL_VERSIONS = 0x1066 # type: ignore
CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x1067 # type: ignore
CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT = 0x1068 # type: ignore
CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT = 0x1069 # type: ignore
CL_DEVICE_OPENCL_C_FEATURES = 0x106F # type: ignore
CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES = 0x1070 # type: ignore
CL_DEVICE_PIPE_SUPPORT = 0x1071 # type: ignore
CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED = 0x1072 # type: ignore
CL_FP_DENORM = (1 << 0) # type: ignore
CL_FP_INF_NAN = (1 << 1) # type: ignore
CL_FP_ROUND_TO_NEAREST = (1 << 2) # type: ignore
CL_FP_ROUND_TO_ZERO = (1 << 3) # type: ignore
CL_FP_ROUND_TO_INF = (1 << 4) # type: ignore
CL_FP_FMA = (1 << 5) # type: ignore
CL_FP_SOFT_FLOAT = (1 << 6) # type: ignore
CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT = (1 << 7) # type: ignore
CL_NONE = 0x0 # type: ignore
CL_READ_ONLY_CACHE = 0x1 # type: ignore
CL_READ_WRITE_CACHE = 0x2 # type: ignore
CL_LOCAL = 0x1 # type: ignore
CL_GLOBAL = 0x2 # type: ignore
CL_EXEC_KERNEL = (1 << 0) # type: ignore
CL_EXEC_NATIVE_KERNEL = (1 << 1) # type: ignore
CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = (1 << 0) # type: ignore
CL_QUEUE_PROFILING_ENABLE = (1 << 1) # type: ignore
CL_QUEUE_ON_DEVICE = (1 << 2) # type: ignore
CL_QUEUE_ON_DEVICE_DEFAULT = (1 << 3) # type: ignore
CL_CONTEXT_REFERENCE_COUNT = 0x1080 # type: ignore
CL_CONTEXT_DEVICES = 0x1081 # type: ignore
CL_CONTEXT_PROPERTIES = 0x1082 # type: ignore
CL_CONTEXT_NUM_DEVICES = 0x1083 # type: ignore
CL_CONTEXT_PLATFORM = 0x1084 # type: ignore
CL_CONTEXT_INTEROP_USER_SYNC = 0x1085 # type: ignore
CL_DEVICE_PARTITION_EQUALLY = 0x1086 # type: ignore
CL_DEVICE_PARTITION_BY_COUNTS = 0x1087 # type: ignore
CL_DEVICE_PARTITION_BY_COUNTS_LIST_END = 0x0 # type: ignore
CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN = 0x1088 # type: ignore
CL_DEVICE_AFFINITY_DOMAIN_NUMA = (1 << 0) # type: ignore
CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE = (1 << 1) # type: ignore
CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE = (1 << 2) # type: ignore
CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE = (1 << 3) # type: ignore
CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE = (1 << 4) # type: ignore
CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE = (1 << 5) # type: ignore
CL_DEVICE_SVM_COARSE_GRAIN_BUFFER = (1 << 0) # type: ignore
CL_DEVICE_SVM_FINE_GRAIN_BUFFER = (1 << 1) # type: ignore
CL_DEVICE_SVM_FINE_GRAIN_SYSTEM = (1 << 2) # type: ignore
CL_DEVICE_SVM_ATOMICS = (1 << 3) # type: ignore
CL_QUEUE_CONTEXT = 0x1090 # type: ignore
CL_QUEUE_DEVICE = 0x1091 # type: ignore
CL_QUEUE_REFERENCE_COUNT = 0x1092 # type: ignore
CL_QUEUE_PROPERTIES = 0x1093 # type: ignore
CL_QUEUE_SIZE = 0x1094 # type: ignore
CL_QUEUE_DEVICE_DEFAULT = 0x1095 # type: ignore
CL_QUEUE_PROPERTIES_ARRAY = 0x1098 # type: ignore
CL_MEM_READ_WRITE = (1 << 0) # type: ignore
CL_MEM_WRITE_ONLY = (1 << 1) # type: ignore
CL_MEM_READ_ONLY = (1 << 2) # type: ignore
CL_MEM_USE_HOST_PTR = (1 << 3) # type: ignore
CL_MEM_ALLOC_HOST_PTR = (1 << 4) # type: ignore
CL_MEM_COPY_HOST_PTR = (1 << 5) # type: ignore
CL_MEM_HOST_WRITE_ONLY = (1 << 7) # type: ignore
CL_MEM_HOST_READ_ONLY = (1 << 8) # type: ignore
CL_MEM_HOST_NO_ACCESS = (1 << 9) # type: ignore
CL_MEM_SVM_FINE_GRAIN_BUFFER = (1 << 10) # type: ignore
CL_MEM_SVM_ATOMICS = (1 << 11) # type: ignore
CL_MEM_KERNEL_READ_AND_WRITE = (1 << 12) # type: ignore
CL_MIGRATE_MEM_OBJECT_HOST = (1 << 0) # type: ignore
CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED = (1 << 1) # type: ignore
CL_R = 0x10B0 # type: ignore
CL_A = 0x10B1 # type: ignore
CL_RG = 0x10B2 # type: ignore
CL_RA = 0x10B3 # type: ignore
CL_RGB = 0x10B4 # type: ignore
CL_RGBA = 0x10B5 # type: ignore
CL_BGRA = 0x10B6 # type: ignore
CL_ARGB = 0x10B7 # type: ignore
CL_INTENSITY = 0x10B8 # type: ignore
CL_LUMINANCE = 0x10B9 # type: ignore
CL_Rx = 0x10BA # type: ignore
CL_RGx = 0x10BB # type: ignore
CL_RGBx = 0x10BC # type: ignore
CL_DEPTH = 0x10BD # type: ignore
CL_sRGB = 0x10BF # type: ignore
CL_sRGBx = 0x10C0 # type: ignore
CL_sRGBA = 0x10C1 # type: ignore
CL_sBGRA = 0x10C2 # type: ignore
CL_ABGR = 0x10C3 # type: ignore
CL_SNORM_INT8 = 0x10D0 # type: ignore
CL_SNORM_INT16 = 0x10D1 # type: ignore
CL_UNORM_INT8 = 0x10D2 # type: ignore
CL_UNORM_INT16 = 0x10D3 # type: ignore
CL_UNORM_SHORT_565 = 0x10D4 # type: ignore
CL_UNORM_SHORT_555 = 0x10D5 # type: ignore
CL_UNORM_INT_101010 = 0x10D6 # type: ignore
CL_SIGNED_INT8 = 0x10D7 # type: ignore
CL_SIGNED_INT16 = 0x10D8 # type: ignore
CL_SIGNED_INT32 = 0x10D9 # type: ignore
CL_UNSIGNED_INT8 = 0x10DA # type: ignore
CL_UNSIGNED_INT16 = 0x10DB # type: ignore
CL_UNSIGNED_INT32 = 0x10DC # type: ignore
CL_HALF_FLOAT = 0x10DD # type: ignore
CL_FLOAT = 0x10DE # type: ignore
CL_UNORM_INT_101010_2 = 0x10E0 # type: ignore
CL_MEM_OBJECT_BUFFER = 0x10F0 # type: ignore
CL_MEM_OBJECT_IMAGE2D = 0x10F1 # type: ignore
CL_MEM_OBJECT_IMAGE3D = 0x10F2 # type: ignore
CL_MEM_OBJECT_IMAGE2D_ARRAY = 0x10F3 # type: ignore
CL_MEM_OBJECT_IMAGE1D = 0x10F4 # type: ignore
CL_MEM_OBJECT_IMAGE1D_ARRAY = 0x10F5 # type: ignore
CL_MEM_OBJECT_IMAGE1D_BUFFER = 0x10F6 # type: ignore
CL_MEM_OBJECT_PIPE = 0x10F7 # type: ignore
CL_MEM_TYPE = 0x1100 # type: ignore
CL_MEM_FLAGS = 0x1101 # type: ignore
CL_MEM_SIZE = 0x1102 # type: ignore
CL_MEM_HOST_PTR = 0x1103 # type: ignore
CL_MEM_MAP_COUNT = 0x1104 # type: ignore
CL_MEM_REFERENCE_COUNT = 0x1105 # type: ignore
CL_MEM_CONTEXT = 0x1106 # type: ignore
CL_MEM_ASSOCIATED_MEMOBJECT = 0x1107 # type: ignore
CL_MEM_OFFSET = 0x1108 # type: ignore
CL_MEM_USES_SVM_POINTER = 0x1109 # type: ignore
CL_MEM_PROPERTIES = 0x110A # type: ignore
CL_IMAGE_FORMAT = 0x1110 # type: ignore
CL_IMAGE_ELEMENT_SIZE = 0x1111 # type: ignore
CL_IMAGE_ROW_PITCH = 0x1112 # type: ignore
CL_IMAGE_SLICE_PITCH = 0x1113 # type: ignore
CL_IMAGE_WIDTH = 0x1114 # type: ignore
CL_IMAGE_HEIGHT = 0x1115 # type: ignore
CL_IMAGE_DEPTH = 0x1116 # type: ignore
CL_IMAGE_ARRAY_SIZE = 0x1117 # type: ignore
CL_IMAGE_BUFFER = 0x1118 # type: ignore
CL_IMAGE_NUM_MIP_LEVELS = 0x1119 # type: ignore
CL_IMAGE_NUM_SAMPLES = 0x111A # type: ignore
CL_PIPE_PACKET_SIZE = 0x1120 # type: ignore
CL_PIPE_MAX_PACKETS = 0x1121 # type: ignore
CL_PIPE_PROPERTIES = 0x1122 # type: ignore
CL_ADDRESS_NONE = 0x1130 # type: ignore
CL_ADDRESS_CLAMP_TO_EDGE = 0x1131 # type: ignore
CL_ADDRESS_CLAMP = 0x1132 # type: ignore
CL_ADDRESS_REPEAT = 0x1133 # type: ignore
CL_ADDRESS_MIRRORED_REPEAT = 0x1134 # type: ignore
CL_FILTER_NEAREST = 0x1140 # type: ignore
CL_FILTER_LINEAR = 0x1141 # type: ignore
CL_SAMPLER_REFERENCE_COUNT = 0x1150 # type: ignore
CL_SAMPLER_CONTEXT = 0x1151 # type: ignore
CL_SAMPLER_NORMALIZED_COORDS = 0x1152 # type: ignore
CL_SAMPLER_ADDRESSING_MODE = 0x1153 # type: ignore
CL_SAMPLER_FILTER_MODE = 0x1154 # type: ignore
CL_SAMPLER_MIP_FILTER_MODE = 0x1155 # type: ignore
CL_SAMPLER_LOD_MIN = 0x1156 # type: ignore
CL_SAMPLER_LOD_MAX = 0x1157 # type: ignore
CL_SAMPLER_PROPERTIES = 0x1158 # type: ignore
CL_MAP_READ = (1 << 0) # type: ignore
CL_MAP_WRITE = (1 << 1) # type: ignore
CL_MAP_WRITE_INVALIDATE_REGION = (1 << 2) # type: ignore
CL_PROGRAM_REFERENCE_COUNT = 0x1160 # type: ignore
CL_PROGRAM_CONTEXT = 0x1161 # type: ignore
CL_PROGRAM_NUM_DEVICES = 0x1162 # type: ignore
CL_PROGRAM_DEVICES = 0x1163 # type: ignore
CL_PROGRAM_SOURCE = 0x1164 # type: ignore
CL_PROGRAM_BINARY_SIZES = 0x1165 # type: ignore
CL_PROGRAM_BINARIES = 0x1166 # type: ignore
CL_PROGRAM_NUM_KERNELS = 0x1167 # type: ignore
CL_PROGRAM_KERNEL_NAMES = 0x1168 # type: ignore
CL_PROGRAM_IL = 0x1169 # type: ignore
CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT = 0x116A # type: ignore
CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT = 0x116B # type: ignore
CL_PROGRAM_BUILD_STATUS = 0x1181 # type: ignore
CL_PROGRAM_BUILD_OPTIONS = 0x1182 # type: ignore
CL_PROGRAM_BUILD_LOG = 0x1183 # type: ignore
CL_PROGRAM_BINARY_TYPE = 0x1184 # type: ignore
CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE = 0x1185 # type: ignore
CL_PROGRAM_BINARY_TYPE_NONE = 0x0 # type: ignore
CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT = 0x1 # type: ignore
CL_PROGRAM_BINARY_TYPE_LIBRARY = 0x2 # type: ignore
CL_PROGRAM_BINARY_TYPE_EXECUTABLE = 0x4 # type: ignore
CL_BUILD_SUCCESS = 0 # type: ignore
CL_BUILD_NONE = -1 # type: ignore
CL_BUILD_ERROR = -2 # type: ignore
CL_BUILD_IN_PROGRESS = -3 # type: ignore
CL_KERNEL_FUNCTION_NAME = 0x1190 # type: ignore
CL_KERNEL_NUM_ARGS = 0x1191 # type: ignore
CL_KERNEL_REFERENCE_COUNT = 0x1192 # type: ignore
CL_KERNEL_CONTEXT = 0x1193 # type: ignore
CL_KERNEL_PROGRAM = 0x1194 # type: ignore
CL_KERNEL_ATTRIBUTES = 0x1195 # type: ignore
CL_KERNEL_ARG_ADDRESS_QUALIFIER = 0x1196 # type: ignore
CL_KERNEL_ARG_ACCESS_QUALIFIER = 0x1197 # type: ignore
CL_KERNEL_ARG_TYPE_NAME = 0x1198 # type: ignore
CL_KERNEL_ARG_TYPE_QUALIFIER = 0x1199 # type: ignore
CL_KERNEL_ARG_NAME = 0x119A # type: ignore
CL_KERNEL_ARG_ADDRESS_GLOBAL = 0x119B # type: ignore
CL_KERNEL_ARG_ADDRESS_LOCAL = 0x119C # type: ignore
CL_KERNEL_ARG_ADDRESS_CONSTANT = 0x119D # type: ignore
CL_KERNEL_ARG_ADDRESS_PRIVATE = 0x119E # type: ignore
CL_KERNEL_ARG_ACCESS_READ_ONLY = 0x11A0 # type: ignore
CL_KERNEL_ARG_ACCESS_WRITE_ONLY = 0x11A1 # type: ignore
CL_KERNEL_ARG_ACCESS_READ_WRITE = 0x11A2 # type: ignore
CL_KERNEL_ARG_ACCESS_NONE = 0x11A3 # type: ignore
CL_KERNEL_ARG_TYPE_NONE = 0 # type: ignore
CL_KERNEL_ARG_TYPE_CONST = (1 << 0) # type: ignore
CL_KERNEL_ARG_TYPE_RESTRICT = (1 << 1) # type: ignore
CL_KERNEL_ARG_TYPE_VOLATILE = (1 << 2) # type: ignore
CL_KERNEL_ARG_TYPE_PIPE = (1 << 3) # type: ignore
CL_KERNEL_WORK_GROUP_SIZE = 0x11B0 # type: ignore
CL_KERNEL_COMPILE_WORK_GROUP_SIZE = 0x11B1 # type: ignore
CL_KERNEL_LOCAL_MEM_SIZE = 0x11B2 # type: ignore
CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x11B3 # type: ignore
CL_KERNEL_PRIVATE_MEM_SIZE = 0x11B4 # type: ignore
CL_KERNEL_GLOBAL_WORK_SIZE = 0x11B5 # type: ignore
CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE = 0x2033 # type: ignore
CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE = 0x2034 # type: ignore
CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT = 0x11B8 # type: ignore
CL_KERNEL_MAX_NUM_SUB_GROUPS = 0x11B9 # type: ignore
CL_KERNEL_COMPILE_NUM_SUB_GROUPS = 0x11BA # type: ignore
CL_KERNEL_EXEC_INFO_SVM_PTRS = 0x11B6 # type: ignore
CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM = 0x11B7 # type: ignore
CL_EVENT_COMMAND_QUEUE = 0x11D0 # type: ignore
CL_EVENT_COMMAND_TYPE = 0x11D1 # type: ignore
CL_EVENT_REFERENCE_COUNT = 0x11D2 # type: ignore
CL_EVENT_COMMAND_EXECUTION_STATUS = 0x11D3 # type: ignore
CL_EVENT_CONTEXT = 0x11D4 # type: ignore
CL_COMMAND_NDRANGE_KERNEL = 0x11F0 # type: ignore
CL_COMMAND_TASK = 0x11F1 # type: ignore
CL_COMMAND_NATIVE_KERNEL = 0x11F2 # type: ignore
CL_COMMAND_READ_BUFFER = 0x11F3 # type: ignore
CL_COMMAND_WRITE_BUFFER = 0x11F4 # type: ignore
CL_COMMAND_COPY_BUFFER = 0x11F5 # type: ignore
CL_COMMAND_READ_IMAGE = 0x11F6 # type: ignore
CL_COMMAND_WRITE_IMAGE = 0x11F7 # type: ignore
CL_COMMAND_COPY_IMAGE = 0x11F8 # type: ignore
CL_COMMAND_COPY_IMAGE_TO_BUFFER = 0x11F9 # type: ignore
CL_COMMAND_COPY_BUFFER_TO_IMAGE = 0x11FA # type: ignore
CL_COMMAND_MAP_BUFFER = 0x11FB # type: ignore
CL_COMMAND_MAP_IMAGE = 0x11FC # type: ignore
CL_COMMAND_UNMAP_MEM_OBJECT = 0x11FD # type: ignore
CL_COMMAND_MARKER = 0x11FE # type: ignore
CL_COMMAND_ACQUIRE_GL_OBJECTS = 0x11FF # type: ignore
CL_COMMAND_RELEASE_GL_OBJECTS = 0x1200 # type: ignore
CL_COMMAND_READ_BUFFER_RECT = 0x1201 # type: ignore
CL_COMMAND_WRITE_BUFFER_RECT = 0x1202 # type: ignore
CL_COMMAND_COPY_BUFFER_RECT = 0x1203 # type: ignore
CL_COMMAND_USER = 0x1204 # type: ignore
CL_COMMAND_BARRIER = 0x1205 # type: ignore
CL_COMMAND_MIGRATE_MEM_OBJECTS = 0x1206 # type: ignore
CL_COMMAND_FILL_BUFFER = 0x1207 # type: ignore
CL_COMMAND_FILL_IMAGE = 0x1208 # type: ignore
CL_COMMAND_SVM_FREE = 0x1209 # type: ignore
CL_COMMAND_SVM_MEMCPY = 0x120A # type: ignore
CL_COMMAND_SVM_MEMFILL = 0x120B # type: ignore
CL_COMMAND_SVM_MAP = 0x120C # type: ignore
CL_COMMAND_SVM_UNMAP = 0x120D # type: ignore
CL_COMMAND_SVM_MIGRATE_MEM = 0x120E # type: ignore
CL_COMPLETE = 0x0 # type: ignore
CL_RUNNING = 0x1 # type: ignore
CL_SUBMITTED = 0x2 # type: ignore
CL_QUEUED = 0x3 # type: ignore
CL_BUFFER_CREATE_TYPE_REGION = 0x1220 # type: ignore
CL_PROFILING_COMMAND_QUEUED = 0x1280 # type: ignore
CL_PROFILING_COMMAND_SUBMIT = 0x1281 # type: ignore
CL_PROFILING_COMMAND_START = 0x1282 # type: ignore
CL_PROFILING_COMMAND_END = 0x1283 # type: ignore
CL_PROFILING_COMMAND_COMPLETE = 0x1284 # type: ignore
CL_DEVICE_ATOMIC_ORDER_RELAXED = (1 << 0) # type: ignore
CL_DEVICE_ATOMIC_ORDER_ACQ_REL = (1 << 1) # type: ignore
CL_DEVICE_ATOMIC_ORDER_SEQ_CST = (1 << 2) # type: ignore
CL_DEVICE_ATOMIC_SCOPE_WORK_ITEM = (1 << 3) # type: ignore
CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP = (1 << 4) # type: ignore
CL_DEVICE_ATOMIC_SCOPE_DEVICE = (1 << 5) # type: ignore
CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES = (1 << 6) # type: ignore
CL_DEVICE_QUEUE_SUPPORTED = (1 << 0) # type: ignore
CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT = (1 << 1) # type: ignore
CL_KHRONOS_VENDOR_ID_CODEPLAY = 0x10004 # type: ignore
CL_VERSION_MAJOR_BITS = (10) # type: ignore
CL_VERSION_MINOR_BITS = (10) # type: ignore
CL_VERSION_PATCH_BITS = (12) # type: ignore
CL_VERSION_MAJOR_MASK = ((1 << CL_VERSION_MAJOR_BITS) - 1) # type: ignore
CL_VERSION_MINOR_MASK = ((1 << CL_VERSION_MINOR_BITS) - 1) # type: ignore
CL_VERSION_PATCH_MASK = ((1 << CL_VERSION_PATCH_BITS) - 1) # type: ignore
CL_VERSION_MAJOR = lambda version: ((version) >> (CL_VERSION_MINOR_BITS + CL_VERSION_PATCH_BITS)) # type: ignore
CL_VERSION_MINOR = lambda version: (((version) >> CL_VERSION_PATCH_BITS) & CL_VERSION_MINOR_MASK) # type: ignore
CL_VERSION_PATCH = lambda version: ((version) & CL_VERSION_PATCH_MASK) # type: ignore
CL_MAKE_VERSION = lambda major,minor,patch: ((((major) & CL_VERSION_MAJOR_MASK) << (CL_VERSION_MINOR_BITS + CL_VERSION_PATCH_BITS)) | (((minor) & CL_VERSION_MINOR_MASK) << CL_VERSION_PATCH_BITS) | ((patch) & CL_VERSION_PATCH_MASK)) # type: ignore