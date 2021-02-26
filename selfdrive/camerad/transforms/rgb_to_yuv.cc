#include <string.h>
#include <assert.h>

#include "clutil.h"

#include "rgb_to_yuv.h"

void rgb_to_yuv_init(RGBToYUVState* s, cl_context ctx, cl_device_id device_id, int width, int height, int rgb_stride) {
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
  cl_program prg = cl_program_from_file(ctx, device_id, "transforms/rgb_to_yuv.cl", args);

  s->rgb_to_yuv_krnl = CL_CHECK_ERR(clCreateKernel(prg, "rgb_to_yuv", &err));
  // done with this
  CL_CHECK(clReleaseProgram(prg));
}

void rgb_to_yuv_destroy(RGBToYUVState* s) {
  CL_CHECK(clReleaseKernel(s->rgb_to_yuv_krnl));
}

void rgb_to_yuv_queue(RGBToYUVState* s, cl_command_queue q, cl_mem rgb_cl, cl_mem yuv_cl) {
  CL_CHECK(clSetKernelArg(s->rgb_to_yuv_krnl, 0, sizeof(cl_mem), &rgb_cl));
  CL_CHECK(clSetKernelArg(s->rgb_to_yuv_krnl, 1, sizeof(cl_mem), &yuv_cl));
  const size_t work_size[2] = {
    (size_t)(s->width + (s->width % 4 == 0 ? 0 : (4 - s->width % 4))) / 4,
    (size_t)(s->height + (s->height % 4 == 0 ? 0 : (4 - s->height % 4))) / 4
  };
  cl_event event;
  CL_CHECK(clEnqueueNDRangeKernel(q, s->rgb_to_yuv_krnl, 2, NULL, &work_size[0], NULL, 0, 0, &event));
  CL_CHECK(clWaitForEvents(1, &event));
  CL_CHECK(clReleaseEvent(event));
}
