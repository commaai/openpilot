#include <string.h>
#include <assert.h>

#include "clutil.h"

#include "loadyuv.h"

void loadyuv_init(LoadYUVState* s, cl_context ctx, cl_device_id device_id, int width, int height) {
  int err = 0;
  memset(s, 0, sizeof(*s));

  s->width = width;
  s->height = height;

  char args[1024];
  snprintf(args, sizeof(args),
           "-cl-fast-relaxed-math -cl-denorms-are-zero "
           "-DTRANSFORMED_WIDTH=%d -DTRANSFORMED_HEIGHT=%d",
           width, height);
  cl_program prg = CLU_LOAD_FROM_FILE(ctx, device_id, "transforms/loadyuv.cl", args);

  s->loadys_krnl = clCreateKernel(prg, "loadys", &err);
  assert(err == 0);
  s->loaduv_krnl = clCreateKernel(prg, "loaduv", &err);
  assert(err == 0);

  // done with this
  err = clReleaseProgram(prg);
  assert(err == 0);
}

void loadyuv_destroy(LoadYUVState* s) {
  int err = 0;

  err = clReleaseKernel(s->loadys_krnl);
  assert(err == 0);
  err = clReleaseKernel(s->loaduv_krnl);
  assert(err == 0);
}

void loadyuv_queue(LoadYUVState* s, cl_command_queue q,
                   cl_mem y_cl, cl_mem u_cl, cl_mem v_cl,
                   cl_mem out_cl) {
  int err = 0;

  err = clSetKernelArg(s->loadys_krnl, 0, sizeof(cl_mem), &y_cl);
  assert(err == 0);
  err = clSetKernelArg(s->loadys_krnl, 1, sizeof(cl_mem), &out_cl);
  assert(err == 0);

  const size_t loadys_work_size = (s->width*s->height)/8;
  err = clEnqueueNDRangeKernel(q, s->loadys_krnl, 1, NULL,
                               &loadys_work_size, NULL, 0, 0, NULL);
  assert(err == 0);

  const size_t loaduv_work_size = ((s->width/2)*(s->height/2))/8;
  cl_int loaduv_out_off = (s->width*s->height);

  err = clSetKernelArg(s->loaduv_krnl, 0, sizeof(cl_mem), &u_cl);
  assert(err == 0);
  err = clSetKernelArg(s->loaduv_krnl, 1, sizeof(cl_mem), &out_cl);
  assert(err == 0);
  err = clSetKernelArg(s->loaduv_krnl, 2, sizeof(cl_int), &loaduv_out_off);
  assert(err == 0);

  err = clEnqueueNDRangeKernel(q, s->loaduv_krnl, 1, NULL,
                               &loaduv_work_size, NULL, 0, 0, NULL);
  assert(err == 0);

  loaduv_out_off += (s->width/2)*(s->height/2);

  err = clSetKernelArg(s->loaduv_krnl, 0, sizeof(cl_mem), &v_cl);
  assert(err == 0);
  err = clSetKernelArg(s->loaduv_krnl, 1, sizeof(cl_mem), &out_cl);
  assert(err == 0);
  err = clSetKernelArg(s->loaduv_krnl, 2, sizeof(cl_int), &loaduv_out_off);
  assert(err == 0);

  err = clEnqueueNDRangeKernel(q, s->loaduv_krnl, 1, NULL,
                               &loaduv_work_size, NULL, 0, 0, NULL);
  assert(err == 0);
}
