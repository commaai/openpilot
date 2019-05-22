#include <memory.h>
#include <iostream>
#include <getopt.h>
#include <math.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <iomanip>
#include <thread>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <cassert>
#include <cstdint>

#ifdef ANDROID

#define MAXE 0
#include <unistd.h>

#else
// The libyuv implementation on ARM is slightly different than on x86
// Our implementation matches the ARM version, so accept errors of 1
#define MAXE 1

#endif

#include <libyuv.h>

#include <CL/cl.h>

#include "clutil.h"
#include "rgb_to_yuv.h"


static inline double millis_since_boot() {
  struct timespec t;
  clock_gettime(CLOCK_BOOTTIME, &t);
  return t.tv_sec * 1000.0 + t.tv_nsec * 1e-6;
}

void cl_init(cl_device_id &device_id, cl_context &context) {
  int err;
  cl_platform_id platform_id = NULL;
  cl_uint num_devices;
  cl_uint num_platforms;

  err = clGetPlatformIDs(1, &platform_id, &num_platforms);
  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
                       &device_id, &num_devices);
  cl_print_info(platform_id, device_id);
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
}


bool compare_results(uint8_t *a, uint8_t *b, int len, int stride, int width, int height, uint8_t *rgb) {
  int min_diff = 0., max_diff = 0., max_e = 0.;
  int e1 = 0, e0 = 0;
  int e0y = 0, e0u = 0, e0v = 0, e1y = 0, e1u = 0, e1v = 0;
  int max_e_i = 0;
  for (int i = 0;i < len;i++) {
    int e = ((int)a[i]) - ((int)b[i]);
    if(e < min_diff) {
      min_diff = e;
    }
    if(e > max_diff) {
      max_diff = e;
    }
    int e_abs = std::abs(e);
    if(e_abs > max_e) {
      max_e = e_abs;
      max_e_i = i;
    }
    if(e_abs < 1) {
      e0++;
      if(i < stride * height)
        e0y++;
      else if(i < stride * height + stride * height / 4)
        e0u++;
      else
        e0v++;
    } else {
      e1++;
      if(i < stride * height)
        e1y++;
      else if(i < stride * height + stride * height / 4)
        e1u++;
      else
        e1v++;
    }
  }
  //printf("max diff : %d, min diff : %d, e < 1: %d, e >= 1: %d\n", max_diff, min_diff, e0, e1);
  //printf("Y: e < 1: %d, e >= 1: %d, U: e < 1: %d, e >= 1: %d, V: e < 1: %d, e >= 1: %d\n", e0y, e1y, e0u, e1u, e0v, e1v);
  if(max_e <= MAXE) {
    return true;
  }
  int row = max_e_i / stride;
  if(row < height) {
    printf("max error is Y: %d = (libyuv: %u - cl: %u), row: %d, col: %d\n", max_e, a[max_e_i], b[max_e_i], row, max_e_i % stride);
  } else if(row >= height && row < (height + height / 4)) {
    printf("max error is U: %d = %u - %u, row: %d, col: %d\n", max_e, a[max_e_i], b[max_e_i], (row - height) / 2, max_e_i % stride / 2);
  } else {
    printf("max error is V: %d = %u - %u, row: %d, col: %d\n", max_e, a[max_e_i], b[max_e_i], (row - height - height / 4) / 2, max_e_i % stride / 2);
  }
  return false;
}

int main(int argc, char** argv) {
  srand(1337);

  clu_init();
  cl_device_id device_id;
  cl_context context;
  cl_init(device_id, context)	;

  int err;
  const cl_queue_properties props[] = {0}; //CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_HIGH_KHR, 0};
  cl_command_queue q = clCreateCommandQueueWithProperties(context, device_id, props, &err);
  if(err != 0) {
    std::cout << "clCreateCommandQueueWithProperties error: " << err << std::endl;
  }

  int width = 1164;
  int height = 874;

  int opt = 0;
  while ((opt = getopt(argc, argv, "f")) != -1)
    {
      switch (opt)
        {
        case 'f':
          std::cout << "Using front camera dimensions" << std::endl;
          int width = 1152;
          int height = 846;
        }
  }

  std::cout << "Width: " << width << " Height: " << height << std::endl;
  uint8_t *rgb_frame = new uint8_t[width * height * 3];


  RGBToYUVState rgb_to_yuv_state;
  rgb_to_yuv_init(&rgb_to_yuv_state, context, device_id, width, height, width * 3);

  int frame_yuv_buf_size = width * height * 3 / 2;
  cl_mem yuv_cl = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_yuv_buf_size, (void*)NULL, &err);
  uint8_t *frame_yuv_buf = new uint8_t[frame_yuv_buf_size];
  uint8_t *frame_yuv_ptr_y = frame_yuv_buf;
  uint8_t *frame_yuv_ptr_u = frame_yuv_buf + (width * height);
  uint8_t *frame_yuv_ptr_v = frame_yuv_ptr_u + ((width/2) * (height/2));

  cl_mem rgb_cl = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * 3, (void*)NULL, &err);
  int mismatched = 0;
  int counter = 0;
  srand (time(NULL));

  for (int i = 0; i < 100; i++){
    for (int i = 0; i < width * height * 3; i++){
      rgb_frame[i] = (uint8_t)rand();
    }

    double t1 = millis_since_boot();
    libyuv::RGB24ToI420((uint8_t*)rgb_frame, width * 3,
                        frame_yuv_ptr_y, width,
                        frame_yuv_ptr_u, width/2,
                        frame_yuv_ptr_v, width/2,
                        width, height);
    double t2 = millis_since_boot();
    //printf("Libyuv: rgb to yuv: %.2fms\n", t2-t1);

    clEnqueueWriteBuffer(q, rgb_cl, CL_TRUE, 0, width * height * 3, (void *)rgb_frame, 0, NULL, NULL);
    t1 = millis_since_boot();
    rgb_to_yuv_queue(&rgb_to_yuv_state, q, rgb_cl, yuv_cl);
    t2 = millis_since_boot();

    //printf("OpenCL: rgb to yuv: %.2fms\n", t2-t1);
    uint8_t *yyy = (uint8_t *)clEnqueueMapBuffer(q, yuv_cl, CL_TRUE,
                                                 CL_MAP_READ, 0, frame_yuv_buf_size,
                                                 0, NULL, NULL, &err);
    if(!compare_results(frame_yuv_ptr_y, yyy, frame_yuv_buf_size, width, width, height, (uint8_t*)rgb_frame))
      mismatched++;
    clEnqueueUnmapMemObject(q, yuv_cl, yyy, 0, NULL, NULL);

    // std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if(counter++ % 100 == 0)
      printf("Matched: %d, Mismatched: %d\n", counter - mismatched, mismatched);

  }
  printf("Matched: %d, Mismatched: %d\n", counter - mismatched, mismatched);

  delete[] frame_yuv_buf;
  rgb_to_yuv_destroy(&rgb_to_yuv_state);
  clReleaseContext(context);
  delete[] rgb_frame;

  if (mismatched == 0)
    return 0;
  else
    return -1;
}
