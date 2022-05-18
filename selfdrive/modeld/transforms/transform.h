#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "common/mat.h"

typedef struct {
  cl_kernel krnl;
  cl_mem m_y_cl, m_uv_cl;
} Transform;

void transform_init(Transform* s, cl_context ctx, cl_device_id device_id);

void transform_destroy(Transform* transform);

void transform_queue(Transform* s, cl_command_queue q,
                     cl_mem yuv, int in_width, int in_height,
                     cl_mem out_y, cl_mem out_u, cl_mem out_v,
                     int out_width, int out_height,
                     const mat3& projection);
