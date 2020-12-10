#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

void clu_init(void);

cl_device_id cl_get_device_id(cl_device_type device_type);
cl_program cl_program_from_string(cl_context ctx, cl_device_id device_id,
                                        const char* src, const char* args,
                                        const char* file, int line, const char* function);
cl_program cl_program_from_file(cl_context ctx, cl_device_id device_id, const char* path, const char* args,
                                      const char* file, int line, const char* function);
const char* cl_get_error_string(int err);

#define CLU_LOAD_FROM_STRING(ctx, device_id, src, args) \
  cl_program_from_string(ctx, device_id, src, args, __FILE__, __LINE__, __func__);
 #define CLU_LOAD_FROM_FILE(ctx, device_id, path, args) \
  cl_program_from_file(ctx, device_id, path, args, __FILE__, __LINE__, __func__);

#ifdef __cplusplus
}
#endif
