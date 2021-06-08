#include "selfdrive/camerad/transforms/rgb_to_yuv.h"

#include <cassert>
#include <cstdio>

Rgb2Yuv::Rgb2Yuv(cl_context ctx, cl_device_id device_id, int width, int height, int rgb_stride) {
  assert(width % 2 == 0 && height % 2 == 0);
  char args[1024];
  snprintf(args, sizeof(args),
           "-cl-fast-relaxed-math -cl-denorms-are-zero "
#ifdef CL_DEBUG
           "-DCL_DEBUG "
#endif
           "-DWIDTH=%d -DHEIGHT=%d -DUV_WIDTH=%d -DUV_HEIGHT=%d -DRGB_STRIDE=%d -DRGB_SIZE=%d",
           width, height, width / 2, height / 2, rgb_stride, width * height);

  cl_program prg = cl_program_from_file(ctx, device_id, "transforms/rgb_to_yuv.cl", args);
  krnl = CL_CHECK_ERR(clCreateKernel(prg, "rgb_to_yuv", &err));
  CL_CHECK(clReleaseProgram(prg));

  work_size[0] = (width + (width % 4 == 0 ? 0 : (4 - width % 4))) / 4;
  work_size[1] = (height + (height % 4 == 0 ? 0 : (4 - height % 4))) / 4;
}

Rgb2Yuv::~Rgb2Yuv() {
  CL_CHECK(clReleaseKernel(krnl));
}

void Rgb2Yuv::queue(cl_command_queue q, cl_mem rgb_cl, cl_mem yuv_cl) {
  CL_CHECK(clSetKernelArg(krnl, 0, sizeof(cl_mem), &rgb_cl));
  CL_CHECK(clSetKernelArg(krnl, 1, sizeof(cl_mem), &yuv_cl));
  cl_event event;
  CL_CHECK(clEnqueueNDRangeKernel(q, krnl, 2, NULL, &work_size[0], NULL, 0, 0, &event));
  CL_CHECK(clWaitForEvents(1, &event));
  CL_CHECK(clReleaseEvent(event));
}
