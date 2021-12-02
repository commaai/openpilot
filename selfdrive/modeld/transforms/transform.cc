#include "selfdrive/modeld/transforms/transform.h"

#include <cassert>
#include <cstring>

#include "selfdrive/common/clutil.h"

void transform_init(Transform* s, cl_context ctx, cl_device_id device_id) {
  memset(s, 0, sizeof(*s));

  cl_program prg = cl_program_from_file(ctx, device_id, "transforms/transform.cl", "");
  s->krnl = CL_CHECK_ERR(clCreateKernel(prg, "warpPerspective", &err));
  // done with this
  CL_CHECK(clReleaseProgram(prg));

  s->m_y_cl = CL_CHECK_ERR(clCreateBuffer(ctx, CL_MEM_READ_WRITE, 3*3*sizeof(float), NULL, &err));
  s->m_uv_cl = CL_CHECK_ERR(clCreateBuffer(ctx, CL_MEM_READ_WRITE, 3*3*sizeof(float), NULL, &err));
}

void transform_destroy(Transform* s) {
  CL_CHECK(clReleaseMemObject(s->m_y_cl));
  CL_CHECK(clReleaseMemObject(s->m_uv_cl));
  CL_CHECK(clReleaseKernel(s->krnl));
}

void transform_queue(Transform* s,
                     cl_command_queue q,
                     cl_mem in_yuv, int in_width, int in_height,
                     cl_mem out_y, cl_mem out_u, cl_mem out_v,
                     int out_width, int out_height,
                     const mat3& projection) {
  const int zero = 0;

  // sampled using pixel center origin
  // (because thats how fastcv and opencv does it)

  mat3 projection_y = projection;

  // in and out uv is half the size of y.
  mat3 projection_uv = transform_scale_buffer(projection, 0.5);

  CL_CHECK(clEnqueueWriteBuffer(q, s->m_y_cl, CL_TRUE, 0, 3*3*sizeof(float), (void*)projection_y.v, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(q, s->m_uv_cl, CL_TRUE, 0, 3*3*sizeof(float), (void*)projection_uv.v, 0, NULL, NULL));

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

  CL_CHECK(clSetKernelArg(s->krnl, 0, sizeof(cl_mem), &in_yuv));
  CL_CHECK(clSetKernelArg(s->krnl, 1, sizeof(cl_int), &in_y_width));
  CL_CHECK(clSetKernelArg(s->krnl, 2, sizeof(cl_int), &in_y_offset));
  CL_CHECK(clSetKernelArg(s->krnl, 3, sizeof(cl_int), &in_y_height));
  CL_CHECK(clSetKernelArg(s->krnl, 4, sizeof(cl_int), &in_y_width));
  CL_CHECK(clSetKernelArg(s->krnl, 5, sizeof(cl_mem), &out_y));
  CL_CHECK(clSetKernelArg(s->krnl, 6, sizeof(cl_int), &out_y_width));
  CL_CHECK(clSetKernelArg(s->krnl, 7, sizeof(cl_int), &zero));
  CL_CHECK(clSetKernelArg(s->krnl, 8, sizeof(cl_int), &out_y_height));
  CL_CHECK(clSetKernelArg(s->krnl, 9, sizeof(cl_int), &out_y_width));
  CL_CHECK(clSetKernelArg(s->krnl, 10, sizeof(cl_mem), &s->m_y_cl));

  const size_t work_size_y[2] = {(size_t)out_y_width, (size_t)out_y_height};

  CL_CHECK(clEnqueueNDRangeKernel(q, s->krnl, 2, NULL,
                              (const size_t*)&work_size_y, NULL, 0, 0, NULL));

  const size_t work_size_uv[2] = {(size_t)out_uv_width, (size_t)out_uv_height};

  CL_CHECK(clSetKernelArg(s->krnl, 1, sizeof(cl_int), &in_uv_width));
  CL_CHECK(clSetKernelArg(s->krnl, 2, sizeof(cl_int), &in_u_offset));
  CL_CHECK(clSetKernelArg(s->krnl, 3, sizeof(cl_int), &in_uv_height));
  CL_CHECK(clSetKernelArg(s->krnl, 4, sizeof(cl_int), &in_uv_width));
  CL_CHECK(clSetKernelArg(s->krnl, 5, sizeof(cl_mem), &out_u));
  CL_CHECK(clSetKernelArg(s->krnl, 6, sizeof(cl_int), &out_uv_width));
  CL_CHECK(clSetKernelArg(s->krnl, 7, sizeof(cl_int), &zero));
  CL_CHECK(clSetKernelArg(s->krnl, 8, sizeof(cl_int), &out_uv_height));
  CL_CHECK(clSetKernelArg(s->krnl, 9, sizeof(cl_int), &out_uv_width));
  CL_CHECK(clSetKernelArg(s->krnl, 10, sizeof(cl_mem), &s->m_uv_cl));
  
  CL_CHECK(clEnqueueNDRangeKernel(q, s->krnl, 2, NULL,
                              (const size_t*)&work_size_uv, NULL, 0, 0, NULL));
  CL_CHECK(clSetKernelArg(s->krnl, 2, sizeof(cl_int), &in_v_offset));
  CL_CHECK(clSetKernelArg(s->krnl, 5, sizeof(cl_mem), &out_v));

  CL_CHECK(clEnqueueNDRangeKernel(q, s->krnl, 2, NULL,
                              (const size_t*)&work_size_uv, NULL, 0, 0, NULL));
}
