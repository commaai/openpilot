#ifndef RGB_TO_YUV_H
#define RGB_TO_YUV_H

#include <inttypes.h>
#include <stdbool.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int width, height;
  cl_kernel rgb_to_yuv_krnl;
} RGBToYUVState;

void rgb_to_yuv_init(RGBToYUVState* s, cl_context ctx, cl_device_id device_id, int width, int height, int rgb_stride);

void rgb_to_yuv_destroy(RGBToYUVState* s);

void rgb_to_yuv_queue(RGBToYUVState* s, cl_command_queue q, cl_mem rgb_cl, cl_mem yuv_cl);

#ifdef __cplusplus
}
#endif

#endif  // RGB_TO_YUV_H
