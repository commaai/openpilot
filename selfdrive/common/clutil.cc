#include "selfdrive/common/clutil.h"

#include <sys/stat.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "selfdrive/common/util.h"

namespace {  // helper functions

template <typename Func, typename Id, typename Name>
std::string get_info(Func get_info_func, Id id, Name param_name) {
  size_t size = 0;
  CL_CHECK(get_info_func(id, param_name, 0, NULL, &size));
  std::string info(size, '\0');
  CL_CHECK(get_info_func(id, param_name, size, info.data(), NULL));
  return info;
}
inline std::string get_platform_info(cl_platform_id id, cl_platform_info name) { return get_info(&clGetPlatformInfo, id, name); }
inline std::string get_device_info(cl_device_id id, cl_device_info name) { return get_info(&clGetDeviceInfo, id, name); }

void cl_print_info(cl_platform_id platform, cl_device_id device) {
  size_t work_group_size = 0;
  cl_device_type device_type = 0;
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(work_group_size), &work_group_size, NULL);
  clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
  const char *type_str = "Other...";
  switch (device_type) {
    case CL_DEVICE_TYPE_CPU: type_str ="CL_DEVICE_TYPE_CPU"; break;
    case CL_DEVICE_TYPE_GPU: type_str = "CL_DEVICE_TYPE_GPU"; break;
    case CL_DEVICE_TYPE_ACCELERATOR: type_str = "CL_DEVICE_TYPE_ACCELERATOR"; break;
  }

  std::cout << "vendor: " << get_platform_info(platform, CL_PLATFORM_VENDOR) << std::endl
            << "platform version: " << get_platform_info(platform, CL_PLATFORM_VERSION) << std::endl
            << "profile: " << get_platform_info(platform, CL_PLATFORM_PROFILE) << std::endl
            << "extensions: " << get_platform_info(platform, CL_PLATFORM_EXTENSIONS) << std::endl
            << "name :" << get_device_info(device, CL_DEVICE_NAME) << std::endl
            << "device version :" << get_device_info(device, CL_DEVICE_VERSION) << std::endl
            << "max work group size :" << work_group_size << std::endl
            << "type = " << device_type << " = " << type_str << std::endl;
}

void cl_print_build_errors(cl_program program, cl_device_id device) {
  cl_build_status status;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, NULL);
  size_t log_size;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
  std::string log(log_size, '\0');
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], NULL);

  std::cout << "build failed; status=" << status << ", log:" << std::endl << log << std::endl; 
}

}  // namespace

cl_device_id cl_get_device_id(cl_device_type device_type) {
  cl_uint num_platforms = 0;
  CL_CHECK(clGetPlatformIDs(0, NULL, &num_platforms));
  std::unique_ptr<cl_platform_id[]> platform_ids = std::make_unique<cl_platform_id[]>(num_platforms);
  CL_CHECK(clGetPlatformIDs(num_platforms, &platform_ids[0], NULL));

  for (size_t i = 0; i < num_platforms; ++i) {
    std::cout << "platform[" << i << "] CL_PLATFORM_NAME: " << get_platform_info(platform_ids[i], CL_PLATFORM_NAME) << std::endl;
    // Get first device
    if (cl_device_id device_id = NULL; clGetDeviceIDs(platform_ids[i], device_type, 1, &device_id, NULL) == 0 && device_id) {
      cl_print_info(platform_ids[i], device_id);
      return device_id;
    }
  }
  std::cout << "No valid openCL platform found" << std::endl;
  assert(0);
  return nullptr;
}

cl_program cl_program_from_file(cl_context ctx, cl_device_id device_id, const char* path, const char* args) {
  std::string src = util::read_file(path);
  assert(src.length() > 0);
  cl_program prg = CL_CHECK_ERR(clCreateProgramWithSource(ctx, 1, (const char*[]){src.c_str()}, NULL, &err));
  if (int err = clBuildProgram(prg, 1, &device_id, args, NULL, NULL); err != 0) {
    cl_print_build_errors(prg, device_id);
    assert(0);
  }
  return prg;
}

// Given a cl code and return a string representation
#define CL_ERR_TO_STR(err) case err: return #err
const char* cl_get_error_string(int err) {
  switch (err) {
    CL_ERR_TO_STR(CL_SUCCESS);
    CL_ERR_TO_STR(CL_DEVICE_NOT_FOUND);
    CL_ERR_TO_STR(CL_DEVICE_NOT_AVAILABLE);
    CL_ERR_TO_STR(CL_COMPILER_NOT_AVAILABLE);
    CL_ERR_TO_STR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    CL_ERR_TO_STR(CL_OUT_OF_RESOURCES);
    CL_ERR_TO_STR(CL_OUT_OF_HOST_MEMORY);
    CL_ERR_TO_STR(CL_PROFILING_INFO_NOT_AVAILABLE);
    CL_ERR_TO_STR(CL_MEM_COPY_OVERLAP);
    CL_ERR_TO_STR(CL_IMAGE_FORMAT_MISMATCH);
    CL_ERR_TO_STR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
    CL_ERR_TO_STR(CL_MAP_FAILURE);
    CL_ERR_TO_STR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
    CL_ERR_TO_STR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
    CL_ERR_TO_STR(CL_COMPILE_PROGRAM_FAILURE);
    CL_ERR_TO_STR(CL_LINKER_NOT_AVAILABLE);
    CL_ERR_TO_STR(CL_LINK_PROGRAM_FAILURE);
    CL_ERR_TO_STR(CL_DEVICE_PARTITION_FAILED);
    CL_ERR_TO_STR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
    CL_ERR_TO_STR(CL_INVALID_VALUE);
    CL_ERR_TO_STR(CL_INVALID_DEVICE_TYPE);
    CL_ERR_TO_STR(CL_INVALID_PLATFORM);
    CL_ERR_TO_STR(CL_INVALID_DEVICE);
    CL_ERR_TO_STR(CL_INVALID_CONTEXT);
    CL_ERR_TO_STR(CL_INVALID_QUEUE_PROPERTIES);
    CL_ERR_TO_STR(CL_INVALID_COMMAND_QUEUE);
    CL_ERR_TO_STR(CL_INVALID_HOST_PTR);
    CL_ERR_TO_STR(CL_INVALID_MEM_OBJECT);
    CL_ERR_TO_STR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    CL_ERR_TO_STR(CL_INVALID_IMAGE_SIZE);
    CL_ERR_TO_STR(CL_INVALID_SAMPLER);
    CL_ERR_TO_STR(CL_INVALID_BINARY);
    CL_ERR_TO_STR(CL_INVALID_BUILD_OPTIONS);
    CL_ERR_TO_STR(CL_INVALID_PROGRAM);
    CL_ERR_TO_STR(CL_INVALID_PROGRAM_EXECUTABLE);
    CL_ERR_TO_STR(CL_INVALID_KERNEL_NAME);
    CL_ERR_TO_STR(CL_INVALID_KERNEL_DEFINITION);
    CL_ERR_TO_STR(CL_INVALID_KERNEL);
    CL_ERR_TO_STR(CL_INVALID_ARG_INDEX);
    CL_ERR_TO_STR(CL_INVALID_ARG_VALUE);
    CL_ERR_TO_STR(CL_INVALID_ARG_SIZE);
    CL_ERR_TO_STR(CL_INVALID_KERNEL_ARGS);
    CL_ERR_TO_STR(CL_INVALID_WORK_DIMENSION);
    CL_ERR_TO_STR(CL_INVALID_WORK_GROUP_SIZE);
    CL_ERR_TO_STR(CL_INVALID_WORK_ITEM_SIZE);
    CL_ERR_TO_STR(CL_INVALID_GLOBAL_OFFSET);
    CL_ERR_TO_STR(CL_INVALID_EVENT_WAIT_LIST);
    CL_ERR_TO_STR(CL_INVALID_EVENT);
    CL_ERR_TO_STR(CL_INVALID_OPERATION);
    CL_ERR_TO_STR(CL_INVALID_GL_OBJECT);
    CL_ERR_TO_STR(CL_INVALID_BUFFER_SIZE);
    CL_ERR_TO_STR(CL_INVALID_MIP_LEVEL);
    CL_ERR_TO_STR(CL_INVALID_GLOBAL_WORK_SIZE);
    CL_ERR_TO_STR(CL_INVALID_PROPERTY);
    CL_ERR_TO_STR(CL_INVALID_IMAGE_DESCRIPTOR);
    CL_ERR_TO_STR(CL_INVALID_COMPILER_OPTIONS);
    CL_ERR_TO_STR(CL_INVALID_LINKER_OPTIONS);
    CL_ERR_TO_STR(CL_INVALID_DEVICE_PARTITION_COUNT);
    case -69: return "CL_INVALID_PIPE_SIZE";
    case -70: return "CL_INVALID_DEVICE_QUEUE";
    case -71: return "CL_INVALID_SPEC_ID";
    case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
    case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
    case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
    case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
    case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
    case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
    case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
    case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
    case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
    case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
    case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
    case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
    case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
    case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
    case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
    case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
    case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
    case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
    default: return "CL_UNKNOWN_ERROR";
  }
}
