#include "common/clutil.h"

#include <cassert>
#include <iostream>
#include <memory>

#include "common/util.h"
#include "common/swaglog.h"

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

  LOGD("vendor: %s", get_platform_info(platform, CL_PLATFORM_VENDOR).c_str());
  LOGD("platform version: %s", get_platform_info(platform, CL_PLATFORM_VERSION).c_str());
  LOGD("profile: %s", get_platform_info(platform, CL_PLATFORM_PROFILE).c_str());
  LOGD("extensions: %s", get_platform_info(platform, CL_PLATFORM_EXTENSIONS).c_str());
  LOGD("name: %s", get_device_info(device, CL_DEVICE_NAME).c_str());
  LOGD("device version: %s", get_device_info(device, CL_DEVICE_VERSION).c_str());
  LOGD("max work group size: %zu", work_group_size);
  LOGD("type = %d, %s", (int)device_type, type_str);
}

void cl_print_build_errors(cl_program program, cl_device_id device) {
  cl_build_status status;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, NULL);
  size_t log_size;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
  std::string log(log_size, '\0');
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], NULL);

  LOGE("build failed; status=%d, log: %s", status, log.c_str());
}

}  // namespace

cl_device_id cl_get_device_id(cl_device_type device_type) {
  cl_uint num_platforms = 0;
  CL_CHECK(clGetPlatformIDs(0, NULL, &num_platforms));
  std::unique_ptr<cl_platform_id[]> platform_ids = std::make_unique<cl_platform_id[]>(num_platforms);
  CL_CHECK(clGetPlatformIDs(num_platforms, &platform_ids[0], NULL));

  for (size_t i = 0; i < num_platforms; ++i) {
    LOGD("platform[%zu] CL_PLATFORM_NAME: %s", i, get_platform_info(platform_ids[i], CL_PLATFORM_NAME).c_str());

    // Get first device
    if (cl_device_id device_id = NULL; clGetDeviceIDs(platform_ids[i], device_type, 1, &device_id, NULL) == 0 && device_id) {
      cl_print_info(platform_ids[i], device_id);
      return device_id;
    }
  }
  LOGE("No valid openCL platform found");
  assert(0);
  return nullptr;
}

cl_context cl_create_context(cl_device_id device_id) {
  return CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));
}

void cl_release_context(cl_context context) {
  clReleaseContext(context);
}

cl_program cl_program_from_file(cl_context ctx, cl_device_id device_id, const char* path, const char* args) {
  return cl_program_from_source(ctx, device_id, util::read_file(path), args);
}

cl_program cl_program_from_source(cl_context ctx, cl_device_id device_id, const std::string& src, const char* args) {
  const char *csrc = src.c_str();
  cl_program prg = CL_CHECK_ERR(clCreateProgramWithSource(ctx, 1, &csrc, NULL, &err));
  if (int err = clBuildProgram(prg, 1, &device_id, args, NULL, NULL); err != 0) {
    cl_print_build_errors(prg, device_id);
    assert(0);
  }
  return prg;
}
