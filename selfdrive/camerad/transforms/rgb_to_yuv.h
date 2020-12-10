#pragma once

#include <inttypes.h>
#include <stdbool.h>

#include "clutil.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int width, height;
  cl_kernel rgb_to_yuv_krnl;
} RGBToYUVState;

void rgb_to_yuv_init(RGBToYUVState* s, CLContext *ctx, int width, int height, int rgb_stride);

void rgb_to_yuv_destroy(RGBToYUVState* s);

void rgb_to_yuv_queue(RGBToYUVState* s, cl_command_queue q, cl_mem rgb_cl, cl_mem yuv_cl);

#ifdef __cplusplus
}
#endif
