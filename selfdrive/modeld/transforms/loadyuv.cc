#include "selfdrive/modeld/transforms/loadyuv.h"

#include <cassert>
#include <cstdio>
#include <cstring>

void loadyuv_init(LoadYUVState* s, cl_context ctx, cl_device_id device_id, int width, int height) {
  memset(s, 0, sizeof(*s));

  s->width = width;
  s->height = height;

  char args[1024];
  snprintf(args, sizeof(args),
           "-cl-fast-relaxed-math -cl-denorms-are-zero "
           "-DTRANSFORMED_WIDTH=%d -DTRANSFORMED_HEIGHT=%d",
           width, height);
  cl_program prg = cl_program_from_file(ctx, device_id, "transforms/loadyuv.cl", args);

  s->loadys_krnl = CL_CHECK_ERR(clCreateKernel(prg, "loadys", &err));
  s->loaduv_krnl = CL_CHECK_ERR(clCreateKernel(prg, "loaduv", &err));

  // done with this
  CL_CHECK(clReleaseProgram(prg));
}

void loadyuv_destroy(LoadYUVState* s) {
  CL_CHECK(clReleaseKernel(s->loadys_krnl));
  CL_CHECK(clReleaseKernel(s->loaduv_krnl));
}

void loadyuv_queue(LoadYUVState* s, cl_command_queue q,
                   cl_mem y_cl, cl_mem u_cl, cl_mem v_cl,
                   cl_mem out_cl) {
  CL_CHECK(clSetKernelArg(s->loadys_krnl, 0, sizeof(cl_mem), &y_cl));
  CL_CHECK(clSetKernelArg(s->loadys_krnl, 1, sizeof(cl_mem), &out_cl));

  const size_t loadys_work_size = (s->width*s->height)/8;
  CL_CHECK(clEnqueueNDRangeKernel(q, s->loadys_krnl, 1, NULL,
                               &loadys_work_size, NULL, 0, 0, NULL));

  const size_t loaduv_work_size = ((s->width/2)*(s->height/2))/8;
  cl_int loaduv_out_off = (s->width*s->height);

  CL_CHECK(clSetKernelArg(s->loaduv_krnl, 0, sizeof(cl_mem), &u_cl));
  CL_CHECK(clSetKernelArg(s->loaduv_krnl, 1, sizeof(cl_mem), &out_cl));
  CL_CHECK(clSetKernelArg(s->loaduv_krnl, 2, sizeof(cl_int), &loaduv_out_off));

  CL_CHECK(clEnqueueNDRangeKernel(q, s->loaduv_krnl, 1, NULL,
                               &loaduv_work_size, NULL, 0, 0, NULL));

  loaduv_out_off += (s->width/2)*(s->height/2);

  CL_CHECK(clSetKernelArg(s->loaduv_krnl, 0, sizeof(cl_mem), &v_cl));
  CL_CHECK(clSetKernelArg(s->loaduv_krnl, 1, sizeof(cl_mem), &out_cl));
  CL_CHECK(clSetKernelArg(s->loaduv_krnl, 2, sizeof(cl_int), &loaduv_out_off));

  CL_CHECK(clEnqueueNDRangeKernel(q, s->loaduv_krnl, 1, NULL,
                               &loaduv_work_size, NULL, 0, 0, NULL));
}
