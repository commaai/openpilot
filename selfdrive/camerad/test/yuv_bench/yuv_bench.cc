#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <cstring>
#include <unistd.h>

// #include <opencv2/opencv.hpp>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "common/util.h"
#include "common/timing.h"
#include "common/mat.h"
#include "clutil.h"

int main() {

  int rgb_width = 1164;
  int rgb_height = 874;

  int rgb_stride = rgb_width*3;

  size_t out_size = rgb_width*rgb_height*3/2;

  uint8_t* rgb_buf = (uint8_t*)calloc(1, rgb_width*rgb_height*3);
  uint8_t* out = (uint8_t*)calloc(1, out_size);

  for (int y=0; y<rgb_height; y++) {
    for (int k=0; k<rgb_stride; k++) {
      rgb_buf[y*rgb_stride + k] = k ^ y;
    }
  }


  // init cl
  /* Get Platform and Device Info */
  cl_platform_id platform_id = NULL;
  cl_uint num_platforms_unused;
  int err = clGetPlatformIDs(1, &platform_id, &num_platforms_unused);
  if (err != 0) {
    fprintf(stderr, "cl error: %d\n", err);
  }
  assert(err == 0);

  cl_device_id device_id = NULL;
  cl_uint num_devices_unused;
  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id,
                       &num_devices_unused);
  if (err != 0) {
    fprintf(stderr, "cl error: %d\n", err);
  }
  assert(err == 0);

  cl_print_info(platform_id, device_id);
  printf("\n");

  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  assert(err == 0);

  cl_program prg = cl_create_program_from_file(context, "yuv.cl");

  err = clBuildProgram(prg, 1, &device_id, "", NULL, NULL);
  if (err != 0) {
    cl_print_build_errors(prg, device_id);
  }
  cl_check_error(err);


  cl_kernel krnl = clCreateKernel(prg, "RGB2YUV_YV12_IYUV", &err);
  assert(err == 0);


  cl_mem inbuf_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     rgb_width*rgb_height*3, (void*)rgb_buf, &err);
  cl_check_error(err);
  cl_mem out_cl = clCreateBuffer(context, CL_MEM_READ_WRITE, out_size, NULL, &err);
  cl_check_error(err);


  // load into net
  err = clSetKernelArg(krnl, 0, sizeof(cl_mem), &inbuf_cl); //srcptr
  assert(err == 0);

  int zero = 0;
  err = clSetKernelArg(krnl, 1, sizeof(cl_int), &rgb_stride); //src_step
  assert(err == 0);
  err = clSetKernelArg(krnl, 2, sizeof(cl_int), &zero); //src_offset
  assert(err == 0);

  err = clSetKernelArg(krnl, 3, sizeof(cl_mem), &out_cl); //dstptr
  assert(err == 0);
  err = clSetKernelArg(krnl, 4, sizeof(cl_int), &rgb_width); //dst_step
  assert(err == 0);
  err = clSetKernelArg(krnl, 5, sizeof(cl_int), &zero); //dst_offset
  assert(err == 0);

  const int rows = rgb_height * 3 / 2;
  err = clSetKernelArg(krnl, 6, sizeof(cl_int), &rows); //rows
  assert(err == 0);
  err = clSetKernelArg(krnl, 7, sizeof(cl_int), &rgb_width); //cols
  assert(err == 0);

  cl_command_queue q = clCreateCommandQueue(context, device_id, 0, &err);
  assert(err == 0);
  const size_t work_size[2] = {rgb_width/2, rows/3};

  err = clEnqueueNDRangeKernel(q, krnl, 2, NULL,
                              (const size_t*)&work_size, NULL, 0, 0, NULL);
  cl_check_error(err);
  clFinish(q);



  double t1 = millis_since_boot();
  for (int k=0; k<32; k++) {
    err = clEnqueueNDRangeKernel(q, krnl, 2, NULL,
                                (const size_t*)&work_size, NULL, 0, 0, NULL);
    cl_check_error(err);
  }
  clFinish(q);
  double t2 = millis_since_boot();
  printf("t: %.2f\n", (t2-t1)/32.);

  uint8_t* out_ptr = (uint8_t*)clEnqueueMapBuffer(q, out_cl, CL_FALSE,
                                        CL_MAP_READ, 0, out_size,
                                        0, NULL, NULL, &err);
  assert(err == 0);
  clFinish(q);


  FILE* of = fopen("out_cl.bin", "wb");
  fwrite(out_ptr, out_size, 1, of);
  fclose(of);


// #endif


    return 0;
}