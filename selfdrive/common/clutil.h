#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
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

typedef struct CLContext{
  cl_context context;
  cl_device_id device_id;
}CLContext;

CLContext cl_init_context(cl_device_type device_type);
cl_device_id cl_get_device_id(cl_device_type device_type);
void cl_free_context(CLContext *ctx);
cl_program cl_program_from_file(CLContext *ctx, const char* path, const char* args);
const char* cl_get_error_string(int err);

#ifdef __cplusplus
}
#endif
