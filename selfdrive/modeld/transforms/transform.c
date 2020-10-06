#include <string.h>
#include <assert.h>

#include "clutil.h"

#include "transform.h"

void transform_init(Transform* s, cl_context ctx, cl_device_id device_id) {
  int err = 0;
  memset(s, 0, sizeof(*s));

  cl_program prg = CLU_LOAD_FROM_FILE(ctx, device_id, "transforms/transform.cl", "");

  s->krnl = clCreateKernel(prg, "warpPerspective", &err);
  assert(err == 0);

  // done with this
  err = clReleaseProgram(prg);
  assert(err == 0);

  s->m_y_cl = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 3*3*sizeof(float), NULL, &err);
  assert(err == 0);

  s->m_uv_cl = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 3*3*sizeof(float), NULL, &err);
  assert(err == 0);
}

void transform_destroy(Transform* s) {
  int err = 0;

  err = clReleaseMemObject(s->m_y_cl);
  assert(err == 0);
  err = clReleaseMemObject(s->m_uv_cl);
  assert(err == 0);

  err = clReleaseKernel(s->krnl);
  assert(err == 0);
}

void transform_queue(Transform* s,
                     cl_command_queue q,
                     cl_mem in_yuv, int in_width, int in_height,
                     cl_mem out_y, cl_mem out_u, cl_mem out_v,
                     int out_width, int out_height,
                     mat3 projection) {
  int err = 0;
  const int zero = 0;

  // sampled using pixel center origin
  // (because thats how fastcv and opencv does it)

  mat3 projection_y = projection;

  // in and out uv is half the size of y.
  mat3 projection_uv = transform_scale_buffer(projection, 0.5);

  err = clEnqueueWriteBuffer(q, s->m_y_cl, CL_TRUE, 0, 3*3*sizeof(float), (void*)projection_y.v, 0, NULL, NULL);
  assert(err == 0);
  err = clEnqueueWriteBuffer(q, s->m_uv_cl, CL_TRUE, 0, 3*3*sizeof(float), (void*)projection_uv.v, 0, NULL, NULL);
  assert(err == 0);

  const int in_y_width = in_width;
  const int in_y_height = in_height;
  const int in_uv_width = in_width/2;
  const int in_uv_height = in_height/2;
  const int in_y_offset = 0;
  const int in_u_offset = in_y_offset + in_y_width*in_y_height;
  const int in_v_offset = in_u_offset + in_uv_width*in_uv_height;

  const int out_y_width = out_width;
  const int out_y_height = out_height;
  const int out_uv_width = out_width/2;
  const int out_uv_height = out_height/2;

  err = clSetKernelArg(s->krnl, 0, sizeof(cl_mem), &in_yuv);
  assert(err == 0);

  err = clSetKernelArg(s->krnl, 1, sizeof(cl_int), &in_y_width);
  assert(err == 0);
  err = clSetKernelArg(s->krnl, 2, sizeof(cl_int), &in_y_offset);
  assert(err == 0);
  err = clSetKernelArg(s->krnl, 3, sizeof(cl_int), &in_y_height);
  assert(err == 0);
  err = clSetKernelArg(s->krnl, 4, sizeof(cl_int), &in_y_width);
  assert(err == 0);

  err = clSetKernelArg(s->krnl, 5, sizeof(cl_mem), &out_y);
  assert(err == 0);

  err = clSetKernelArg(s->krnl, 6, sizeof(cl_int), &out_y_width);
  assert(err == 0);
  err = clSetKernelArg(s->krnl, 7, sizeof(cl_int), &zero);
  assert(err == 0);
  err = clSetKernelArg(s->krnl, 8, sizeof(cl_int), &out_y_height);
  assert(err == 0);
  err = clSetKernelArg(s->krnl, 9, sizeof(cl_int), &out_y_width);
  assert(err == 0);

  err = clSetKernelArg(s->krnl, 10, sizeof(cl_mem), &s->m_y_cl);
  assert(err == 0);

  const size_t work_size_y[2] = {out_y_width, out_y_height};

  err = clEnqueueNDRangeKernel(q, s->krnl, 2, NULL,
                              (const size_t*)&work_size_y, NULL, 0, 0, NULL);
  assert(err == 0);


  const size_t work_size_uv[2] = {out_uv_width, out_uv_height};

  err = clSetKernelArg(s->krnl, 1, sizeof(cl_int), &in_uv_width);
  assert(err == 0);
  err = clSetKernelArg(s->krnl, 2, sizeof(cl_int), &in_u_offset);
  assert(err == 0);
  err = clSetKernelArg(s->krnl, 3, sizeof(cl_int), &in_uv_height);
  assert(err == 0);
  err = clSetKernelArg(s->krnl, 4, sizeof(cl_int), &in_uv_width);
  assert(err == 0);

  err = clSetKernelArg(s->krnl, 5, sizeof(cl_mem), &out_u);
  assert(err == 0);

  err = clSetKernelArg(s->krnl, 6, sizeof(cl_int), &out_uv_width);
  assert(err == 0);
  err = clSetKernelArg(s->krnl, 7, sizeof(cl_int), &zero);
  assert(err == 0);
  err = clSetKernelArg(s->krnl, 8, sizeof(cl_int), &out_uv_height);
  assert(err == 0);
  err = clSetKernelArg(s->krnl, 9, sizeof(cl_int), &out_uv_width);
  assert(err == 0);

  err = clSetKernelArg(s->krnl, 10, sizeof(cl_mem), &s->m_uv_cl);
  assert(err == 0);

  err = clEnqueueNDRangeKernel(q, s->krnl, 2, NULL,
                              (const size_t*)&work_size_uv, NULL, 0, 0, NULL);
  assert(err == 0);


  err = clSetKernelArg(s->krnl, 2, sizeof(cl_int), &in_v_offset);
  assert(err == 0);
  err = clSetKernelArg(s->krnl, 5, sizeof(cl_mem), &out_v);
  assert(err == 0);


  err = clEnqueueNDRangeKernel(q, s->krnl, 2, NULL,
                              (const size_t*)&work_size_uv, NULL, 0, 0, NULL);
  assert(err == 0);
}
