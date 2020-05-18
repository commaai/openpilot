#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <assert.h>
#include <time.h>

/*
block7b_project_conv (Conv2D)   (None, 8, 16, 352)   743424      block7b_activation[0][0]
8448*8*4 = 8*16*2112 = 270336 = input = 128*2112
2112*88*4 = 743424 = weights = 2112*352
1408*8*4 = 8*16*352 = 45056 = output = 128*352

FLOPS = 128*2112*352 = 95158272 = 95 MFLOPS
RAM = 128*2112 + 2112*352 + 128*352 = 1058816 = 1 M accesses

# 22 groups
128*2112 + 2112*16 + 128*16 = 306176
306176*22 = 6735872 real accesses

This is a 128x2112 by 2112x352 matrix multiply

work_size = {88, 4, 8}
Each kernel run computes 16 outputs

0x7f7e8a6380                 convolution_horizontal_reduced_reads_1x1 --   88    4    8  --    4    4    8
  image2d_t input = 0x7f7f490b00 image 8448 x 8 rp 67840
  short startPackedInputChannel = 0
  short numPackedInputChannelsForGroup = 528
  short totalNumPackedInputChannels = 528
  short packedOuputChannelOffset = 0
  short totalNumPackedOutputChannels = 88
  image2d_t weights = 0x7f7f52fb80 image 2112 x 88 rp 16896
  float* biases = 0x7f7f564d80 buffer 1408
  short filterSizeX = 1
  short filterSizeY = 1
  image2d_t output = 0x7f7f490e80 image 1408 x 8 rp 11264
  short paddingX = 0
  short paddingY = 0
  short strideX = 1
  short strideY = 1
  short neuron = 0
  float a = 1.000000
  float b = 1.000000
  float min_clamp = 0.000000
  float max_clamp = 0.000000
  float* parameters = 0x0
  float* batchNormBiases = 0x0
  short numOutputColumns = 16
*/

#define GEMM
#define IMAGE

void dump_maps() {
  FILE *f = fopen("/proc/self/maps", "rb");
  char maps[0x100000];
  int len = fread(maps, 1, sizeof(maps), f);
  maps[len] = '\0';
  maps[0x800] = '\0';
  fclose(f);
  printf("%s\n", maps);
}

static inline uint64_t nanos_since_boot() {
  struct timespec t;
  clock_gettime(CLOCK_BOOTTIME, &t);
  return t.tv_sec * 1000000000ULL + t.tv_nsec;
}

int main(int argc, char *argv[]) {
  cl_int err;

  // cl init
  cl_device_id device_id;
  cl_context context;
  cl_command_queue q;
  {
    cl_platform_id platform_id[2];
    cl_uint num_devices;
    cl_uint num_platforms;

    err = clGetPlatformIDs(sizeof(platform_id)/sizeof(cl_platform_id), platform_id, &num_platforms);
    assert(err == 0);

    err = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &num_devices);
    assert(err == 0);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    assert(err == 0);

    q = clCreateCommandQueue(context, device_id, 0, &err);
    assert(err == 0);
  }
  printf("cl ready\n");

  char tmp[0x10000];
  memset(tmp, 0, sizeof(tmp));
  FILE *f = fopen(argv[1], "rb");
  fread(tmp, 1, sizeof(tmp), f);
  fclose(f);

  const char *strings[1];
  size_t lengths[1];
  strings[0] = tmp;
  lengths[0] = strlen(tmp);

  cl_program prog = clCreateProgramWithSource(context, 1, strings, lengths, &err);
  assert(err == 0);
  printf("creating program\n");

  err = clBuildProgram(prog, 1, &device_id, "-D AVANTE_IS_GPU_A530_64", NULL, NULL);

  if (err != 0) {
    printf("got err %d\n", err);
    size_t length;
    char buffer[2048];
    clGetProgramBuildInfo(prog, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
    buffer[length] = '\0';
    printf("%s\n", buffer);
  }
  assert(err == 0);
  printf("built program\n");


#ifdef GEMM
  // 128x2112 by 2112x352
  int M,N,K;

  M = N = K = 1024;
  //M = 128; K = 2112; N = 352;
  
  cl_kernel kern = clCreateKernel(prog, "gemm", &err);
  assert(err == 0);
  printf("creating kernel %p\n", kern);

  cl_mem A,B,C;
  A = clCreateBuffer(context, CL_MEM_READ_WRITE, M*K*2, NULL, &err);
  assert(err == 0);
  B = clCreateBuffer(context, CL_MEM_READ_WRITE, K*N*2, NULL, &err);
  assert(err == 0);
  C = clCreateBuffer(context, CL_MEM_READ_WRITE, M*N*2, NULL, &err);
  assert(err == 0);
  printf("created buffers\n");

#ifdef IMAGE
  cl_image_format fmt;
  fmt.image_channel_order = CL_RGBA;
  fmt.image_channel_data_type = CL_HALF_FLOAT;

  cl_image_desc desc;
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_depth = 0; desc.image_slice_pitch = 0; desc.num_mip_levels = 0; desc.num_samples = 0;

  desc.image_width = K; desc.image_height = M/4;
  desc.buffer = A;
  desc.image_row_pitch = desc.image_width*8;
  A = clCreateImage(context, CL_MEM_READ_WRITE, &fmt, &desc, NULL, &err);
  assert(err == 0);

  desc.image_width = K; desc.image_height = N/4;
  desc.buffer = B; desc.image_row_pitch = desc.image_width*8;
  B = clCreateImage(context, CL_MEM_READ_WRITE, &fmt, &desc, NULL, &err);
  assert(err == 0);

  desc.image_width = M/4; desc.image_height = N;
  desc.buffer = C; desc.image_row_pitch = desc.image_width*8;
  C = clCreateImage(context, CL_MEM_READ_WRITE, &fmt, &desc, NULL, &err);
  assert(err == 0);
  printf("created images\n");
#endif

  clSetKernelArg(kern, 0, sizeof(int), &M);
  clSetKernelArg(kern, 1, sizeof(int), &N);
  clSetKernelArg(kern, 2, sizeof(int), &K);

  clSetKernelArg(kern, 3, sizeof(cl_mem), &A);
  clSetKernelArg(kern, 4, sizeof(cl_mem), &B);
  clSetKernelArg(kern, 5, sizeof(cl_mem), &C);
  printf("set args\n");

#ifdef IMAGE
  size_t global_work_size[3] = {M/4, N/4, 1};
  size_t local_work_size[3] = {4, 64, 1};
#else
  size_t global_work_size[3] = {128, 128, 1};
  size_t local_work_size[3] = {2, 128, 1};
#endif

#else
  cl_kernel kern = clCreateKernel(prog, "convolution_horizontal_reduced_reads_1x1", &err);
  assert(err == 0);
  printf("creating kernel\n");

  cl_mem input;
  cl_mem weights;
  cl_mem weights_buffer;
  cl_mem biases;
  cl_mem outputs;

  cl_image_format fmt;
  fmt.image_channel_order = CL_RGBA;
  fmt.image_channel_data_type = CL_HALF_FLOAT;

  cl_image_desc desc;
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_depth = 0; desc.image_slice_pitch = 0; desc.num_mip_levels = 0; desc.num_samples = 0;
  desc.buffer = NULL;

  biases = clCreateBuffer(context, CL_MEM_READ_WRITE, 1408, NULL, &err);
  assert(err == 0);

  desc.image_width = 8448; desc.image_height = 8; desc.image_row_pitch = 67840;
  desc.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, desc.image_height * desc.image_row_pitch, NULL, &err);
  assert(err == 0);
  input = clCreateImage(context, CL_MEM_READ_WRITE, &fmt, &desc, NULL, &err);
  assert(err == 0);

  desc.image_width = 2112; desc.image_height = 88; desc.image_row_pitch = 16896;
  weights_buffer = desc.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, desc.image_height * desc.image_row_pitch, NULL, &err);
  assert(err == 0);
  weights = clCreateImage(context, CL_MEM_READ_WRITE, &fmt, &desc, NULL, &err);
  assert(err == 0);

  desc.image_width = 1408; desc.image_height = 8; desc.image_row_pitch = 11264;
  desc.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, desc.image_height * desc.image_row_pitch, NULL, &err);
  assert(err == 0);
  outputs = clCreateImage(context, CL_MEM_READ_WRITE, &fmt, &desc, NULL, &err);
  assert(err == 0);

  void *n = NULL;
  uint16_t v;
  float fl;

  clSetKernelArg(kern, 0, sizeof(cl_mem), &input);
  v = 0; clSetKernelArg(kern, 1, sizeof(v), &v);
  v = 528; clSetKernelArg(kern, 2, sizeof(v), &v);
  v = 528; clSetKernelArg(kern, 3, sizeof(v), &v);
  v = 0; clSetKernelArg(kern, 4, sizeof(v), &v);
  v = 88; clSetKernelArg(kern, 5, sizeof(v), &v);
  clSetKernelArg(kern, 6, sizeof(cl_mem), &weights);
  //clSetKernelArg(kern, 6, sizeof(cl_mem), &weights_buffer);
  clSetKernelArg(kern, 7, sizeof(cl_mem), &biases);
  v = 1; clSetKernelArg(kern, 8, sizeof(v), &v);
  v = 1; clSetKernelArg(kern, 9, sizeof(v), &v);
  clSetKernelArg(kern, 10, sizeof(cl_mem), &outputs);
  v = 0; clSetKernelArg(kern, 11, sizeof(v), &v);
  v = 0; clSetKernelArg(kern, 12, sizeof(v), &v);
  v = 1; clSetKernelArg(kern, 13, sizeof(v), &v);
  v = 1; clSetKernelArg(kern, 14, sizeof(v), &v);
  v = 0; clSetKernelArg(kern, 15, sizeof(v), &v);
  fl = 1.0; clSetKernelArg(kern, 16, sizeof(fl), &fl);
  fl = 0.0; clSetKernelArg(kern, 17, sizeof(fl), &fl);
  fl = 0.0; clSetKernelArg(kern, 18, sizeof(fl), &fl);
  fl = 0.0; clSetKernelArg(kern, 19, sizeof(fl), &fl);
  clSetKernelArg(kern, 20, sizeof(n), &n);
  clSetKernelArg(kern, 21, sizeof(n), &n);
  v = 16; clSetKernelArg(kern, 22, sizeof(v), &v);
  
  size_t global_work_size[3] = {88, 4, 8};
  size_t local_work_size[3] = {4, 4, 8};
#endif

  printf("ready to enqueue\n");
  for (int i = 0; i < 20; i++) {
    cl_event event;
    err = clEnqueueNDRangeKernel(q, kern, 3, NULL, global_work_size, local_work_size, 0, NULL, &event);
    assert(err == 0);

    uint64_t tb = nanos_since_boot();
    err = clWaitForEvents(1, &event);
    assert(err == 0);
    uint64_t te = nanos_since_boot();
    uint64_t us = (te-tb)/1000;

    float s = 1000000.0/us;

#ifdef GEMM
    float flops = M*N*K*s;
    float rams = (M*N + N*K + M*K)*s;
#else
    float flops = 95158272.0*s;
    float rams = 1058816.0*s;
    //float rams = 6735872.0*s;
#endif

    printf("%2d: wait %lu us -- %.2f GFLOPS -- %.2f GB/s\n", i, us, flops/1e9, rams*2/1e9);
  }

  size_t binary_size = 0;
  err = clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(binary_size), &binary_size, NULL);
  assert(err == 0);
  assert(binary_size > 0);

  uint8_t *binary_buf = (uint8_t *)malloc(binary_size);
  assert(binary_buf);

  uint8_t* bufs[1] = { binary_buf, };
  err = clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(bufs), &bufs, NULL);
  assert(err == 0);

  FILE *g = fopen("/tmp/bin.bin", "wb");
  fwrite(binary_buf, 1, binary_size, g);
  fclose(g);

  /*dump_maps();
  for (uint64_t i = 0x7ffbd2000; i < 0x800000000; i += 0x1000) {
    uint64_t cmd = *((uint64_t*)i);
    printf("%llx: %llx\n", i, cmd);
  }*/


  return 0;
}

