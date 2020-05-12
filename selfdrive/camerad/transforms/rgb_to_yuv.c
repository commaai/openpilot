#include <string.h>
#include <assert.h>

#include "clutil.h"

#include "rgb_to_yuv.h"

void rgb_to_yuv_init(RGBToYUVState* s, cl_context ctx, cl_device_id device_id, int width, int height, int rgb_stride) {
  int err = 0;
  memset(s, 0, sizeof(*s));
  printf("width %d, height %d, rgb_stride %d\n", width, height, rgb_stride);
  assert(width % 2 == 0);
  assert(height % 2 == 0);
  s->width = width;
  s->height = height;
  char args[1024];
  snprintf(args, sizeof(args),
           "-cl-fast-relaxed-math -cl-denorms-are-zero "
#ifdef CL_DEBUG
           "-DCL_DEBUG "
#endif
           "-DWIDTH=%d -DHEIGHT=%d -DUV_WIDTH=%d -DUV_HEIGHT=%d -DRGB_STRIDE=%d -DRGB_SIZE=%d",
           width, height, width/ 2, height / 2, rgb_stride, width * height);
  cl_program prg = CLU_LOAD_FROM_FILE(ctx, device_id, "transforms/rgb_to_yuv.cl", args);

  s->rgb_to_yuv_krnl = clCreateKernel(prg, "rgb_to_yuv", &err);
  assert(err == 0);
  // done with this
  err = clReleaseProgram(prg);
  assert(err == 0);
}

void rgb_to_yuv_destroy(RGBToYUVState* s) {
  int err = 0;
  err = clReleaseKernel(s->rgb_to_yuv_krnl);
  assert(err == 0);
}

void rgb_to_yuv_queue(RGBToYUVState* s, cl_command_queue q, cl_mem rgb_cl, cl_mem yuv_cl) {
  int err = 0;
  err = clSetKernelArg(s->rgb_to_yuv_krnl, 0, sizeof(cl_mem), &rgb_cl);
  assert(err == 0);
  err = clSetKernelArg(s->rgb_to_yuv_krnl, 1, sizeof(cl_mem), &yuv_cl);
  assert(err == 0);
  const size_t work_size[2] = {
    (size_t)(s->width + (s->width % 4 == 0 ? 0 : (4 - s->width % 4))) / 4, 
    (size_t)(s->height + (s->height % 4 == 0 ? 0 : (4 - s->height % 4))) / 4
  };
  cl_event event;
  err = clEnqueueNDRangeKernel(q, s->rgb_to_yuv_krnl, 2, NULL, &work_size[0], NULL, 0, 0, &event);
  assert(err == 0);
  clWaitForEvents(1, &event);
  clReleaseEvent(event);
}
