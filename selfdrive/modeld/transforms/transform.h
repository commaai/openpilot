#pragma once

#include <inttypes.h>
#include <stdbool.h>

#include "clutil.h"
#include "common/mat.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  cl_kernel krnl;
  cl_mem m_y_cl, m_uv_cl;
} Transform;

void transform_init(Transform* s, CLContext *ctx);

void transform_destroy(Transform* transform);

void transform_queue(Transform* s, cl_command_queue q,
                     cl_mem yuv, int in_width, int in_height,
                     cl_mem out_y, cl_mem out_u, cl_mem out_v,
                     int out_width, int out_height,
                     mat3 projection);

#ifdef __cplusplus
}
#endif
