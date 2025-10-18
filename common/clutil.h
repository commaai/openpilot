#pragma once

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <string>

#define CL_CHECK(_expr)          \
  do {                           \
    assert(CL_SUCCESS == (_expr)); \
  } while (0)

#define CL_CHECK_ERR(_expr)           \
  ({                                  \
    cl_int err = CL_INVALID_VALUE;    \
    __typeof__(_expr) _ret = _expr;   \
    assert(_ret&& err == CL_SUCCESS); \
    _ret;                             \
  })

cl_device_id cl_get_device_id(cl_device_type device_type);
cl_context cl_create_context(cl_device_id device_id);
void cl_release_context(cl_context context);
cl_program cl_program_from_source(cl_context ctx, cl_device_id device_id, const std::string& src, const char* args = nullptr);
cl_program cl_program_from_file(cl_context ctx, cl_device_id device_id, const char* path, const char* args);
