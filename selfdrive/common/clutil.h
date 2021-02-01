#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mutex>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define CL_CHECK(_expr)          \
  do {                           \
    assert(CL_SUCCESS == _expr); \
  } while (0)

#define CL_CHECK_ERR(_expr)           \
  ({                                  \
    cl_int err = CL_INVALID_VALUE;    \
    __typeof__(_expr) _ret = _expr;   \
    assert(_ret&& err == CL_SUCCESS); \
    _ret;                             \
  })

cl_device_id cl_get_device_id(cl_device_type device_type);
cl_program cl_program_from_file(cl_context ctx, cl_device_id device_id, const char* path, const char* args);
const char* cl_get_error_string(int err);

class CLContext {
 public:
  CLContext() = default;
  CLContext(cl_device_type device_type) { init(device_type); }
  ~CLContext() { if (context_) clReleaseContext(context_); }
  static CLContext getDefault() {
    std::call_once(default_initialized_, [] { default_.init(CL_DEVICE_TYPE_DEFAULT); });
    return default_;
  }

private:
  void init(cl_device_type device_type) {
    device_id_ = cl_get_device_id(device_type);
    // TODO: do this for QCOM2 too
#if defined(QCOM)
    const cl_context_properties props[] = {CL_CONTEXT_PRIORITY_HINT_QCOM, CL_PRIORITY_HINT_HIGH_QCOM, 0};
    context_ = CL_CHECK_ERR(clCreateContext(props, 1, &device_id_, NULL, NULL, &err));
#else
    context_ = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id_, NULL, NULL, &err));
#endif
  }

  cl_context context_ = nullptr;
  cl_device_id device_id_ = nullptr;
  static std::once_flag default_initialized_;
  static CLContext default_;
};
