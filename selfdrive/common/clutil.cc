#include "clutil.h"

#include <assert.h>
#include <inttypes.h>
#include <string.h>
#include <sys/stat.h>
#include <memory>
#include <iostream>
#include <vector>
#include "util.h"
#include "utilpp.h"

#define CL_ERR_TO_STR(err) case err: return #err

namespace {  // helper functions

std::string get_platform_info(cl_platform_id platform, cl_platform_info param_name) {
  size_t size = 0;
  CL_CHECK(clGetPlatformInfo(platform, param_name, 0, NULL, &size));
  std::string ret;
  ret.resize(size, '\0');
  CL_CHECK(clGetPlatformInfo(platform, param_name, size, &ret[0], NULL));
  return ret;
}

void cl_print_info(cl_platform_id platform, cl_device_id device) {
  
  std::cout << "vendor: " << get_platform_info(platform, CL_PLATFORM_VENDOR) << std::endl
            << "platform version: " << get_platform_info(platform, CL_PLATFORM_VERSION) << std::endl
            << "profile: " << get_platform_info(platform, CL_PLATFORM_PROFILE) << std::endl
            << "extensions: " << get_platform_info(platform, CL_PLATFORM_EXTENSIONS) << std::endl;

  char str[4096] = {};
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(str), str, NULL);
  std::cout << "name :" << str << std::endl;

  clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(str), str, NULL);
  std::cout << "device version :" << str << std::endl;

  size_t sz = 0;
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(sz), &sz, NULL);
  std::cout << "max work group size :" << sz << std::endl;

  cl_device_type type = 0;
  clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
  const char *type_str = "Other...";
  switch (type) {
    case CL_DEVICE_TYPE_CPU: type_str ="CL_DEVICE_TYPE_CPU"; break;
    case CL_DEVICE_TYPE_GPU: type_str = "CL_DEVICE_TYPE_GPU"; break;
    case CL_DEVICE_TYPE_ACCELERATOR: type_str = "CL_DEVICE_TYPE_ACCELERATOR"; break;
  }
  std::cout << "type = " << std::hex << std::showbase << type << " = " << type_str << std::endl;
}

void cl_print_build_errors(cl_program program, cl_device_id device) {
  cl_build_status status;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS,
                        sizeof(cl_build_status), &status, NULL);

  size_t log_size;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

  std::unique_ptr<char[]> log = std::make_unique<char[]>(log_size + 1);
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, &log[0], NULL);
  std::cout << "build failed; status=" << status << ", log:" << std::endl << &log[0] << std::endl; 
}

}  // namespace

cl_device_id cl_get_device_id(cl_device_type device_type) {
  cl_uint num_platforms = 0;
  CL_CHECK(clGetPlatformIDs(0, NULL, &num_platforms));
  std::unique_ptr<cl_platform_id[]> platform_ids = std::make_unique<cl_platform_id[]>(num_platforms);
  CL_CHECK(clGetPlatformIDs(num_platforms, &platform_ids[0], NULL));

  for (size_t i = 0; i < num_platforms; i++) {
    std::string platform_name = get_platform_info(platform_ids[i], CL_PLATFORM_NAME);
    std::cout << "platform[" << i << "] CL_PLATFORM_NAME: " << platform_name << std::endl;

    cl_uint num_devices;
    int err = clGetDeviceIDs(platform_ids[i], device_type, 0, NULL, &num_devices);
    if (err != 0 || !num_devices) {
      continue;
    }
    // Get first device
    cl_device_id device_id = NULL;
    CL_CHECK(clGetDeviceIDs(platform_ids[i], device_type, 1, &device_id, NULL));
    cl_print_info(platform_ids[i], device_id);
    return device_id;
  }
  std::cout << "No valid openCL platform found" << std::endl;
  assert(0);
  return nullptr;
}

cl_program cl_program_from_file(cl_context ctx, cl_device_id device_id, const char* path, const char* args) {
  char* src = (char*)read_file(path, nullptr);
  assert(src != nullptr);
  cl_program prg = CL_CHECK_ERR(clCreateProgramWithSource(ctx, 1, (const char**)&src, NULL, &err));
  free(src);
  if (int err = clBuildProgram(prg, 1, &device_id, args, NULL, NULL); err != 0) {
    cl_print_build_errors(prg, device_id);
    assert(0);
  }
  return prg;
}

// Given a cl code and return a string represenation
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
    CL_ERR_TO_STR(CL_BUILD_PROGRAM_FAILURE);
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
    CL_ERR_TO_STR(CL_INVALID_PIPE_SIZE);
    CL_ERR_TO_STR(CL_INVALID_DEVICE_QUEUE);

    default:
      return "UNKNOWN ERROR CODE";
  }
}
