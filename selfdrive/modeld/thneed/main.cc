#include <cassert>
#include <stdlib.h>
#include <dlfcn.h>
#include <CL/cl.h>
#include "../runners/snpemodel.h"
#include "../models/driving.h"
#include <time.h>

static inline uint64_t nanos_since_boot() {
  struct timespec t;
  clock_gettime(CLOCK_BOOTTIME, &t);
  return t.tv_sec * 1000000000ULL + t.tv_nsec;
}


#include <string>
#include <map>
using namespace std;

int do_print = 0;

#define TEMPORAL_SIZE 512
#define DESIRE_LEN 8
#define TRAFFIC_CONVENTION_LEN 2

FILE *f = NULL;

cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret) {
  cl_program (*my_clCreateProgramWithSource)(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret) = NULL;
  my_clCreateProgramWithSource = reinterpret_cast<decltype(my_clCreateProgramWithSource)>(dlsym(RTLD_NEXT, "REAL_clCreateProgramWithSource"));
  //printf("clCreateProgramWithSource: %d\n", count);

  if (f == NULL) {
    f = fopen("/tmp/kernels.cl", "w");
  }

  fprintf(f, "/* ************************ PROGRAM BREAK ****************************/\n");
  for (int i = 0; i < count; i++) {
    fprintf(f, "%s\n", strings[i]);
    if (i != 0) fprintf(f, "/* ************************ SECTION BREAK ****************************/\n");
  }
  fflush(f);

  return my_clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
}

map<cl_kernel, string> kernels;
map<cl_kernel, void*> kernel_inputs;
map<cl_kernel, void*> kernel_outputs;

cl_kernel clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret) {
  cl_kernel (*my_clCreateKernel)(cl_program program, const char *kernel_name, cl_int *errcode_ret) = NULL;
  my_clCreateKernel = reinterpret_cast<decltype(my_clCreateKernel)>(dlsym(RTLD_NEXT, "REAL_clCreateKernel"));
  cl_kernel ret = my_clCreateKernel(program, kernel_name, errcode_ret);

  printf("clCreateKernel: %s -> %p\n", kernel_name, ret);
  kernels.insert(make_pair(ret, kernel_name));
  return ret;
}

cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
  cl_int (*my_clSetKernelArg)(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) = NULL;
  my_clSetKernelArg = reinterpret_cast<decltype(my_clSetKernelArg)>(dlsym(RTLD_NEXT, "REAL_clSetKernelArg"));

  char arg_type[0x100];
  char arg_name[0x100];
  clGetKernelArgInfo(kernel, arg_index, CL_KERNEL_ARG_TYPE_NAME, sizeof(arg_type), arg_type, NULL);
  clGetKernelArgInfo(kernel, arg_index, CL_KERNEL_ARG_NAME, sizeof(arg_name), arg_name, NULL);
  printf("  %s %s", arg_type, arg_name);

  if (strcmp(arg_name, "input") == 0) kernel_inputs[kernel] = (void*)arg_value;
  if (strcmp(arg_name, "output") == 0) kernel_outputs[kernel] = (void*)arg_value;
  if (strcmp(arg_name, "accumulator") == 0) assert(kernel_inputs[kernel] = (void*)arg_value);

  if (arg_size == 1) {
    printf(" = %d", *((char*)arg_value));
  } else if (arg_size == 2) {
    printf(" = %d", *((short*)arg_value));
  } else if (arg_size == 4) {
    if (strcmp(arg_type, "float") == 0) {
      printf(" = %f", *((float*)arg_value));
    } else {
      printf(" = %d", *((int*)arg_value));
    }
  } else if (arg_size == 8) {
    printf(" = %p", (void*)arg_value);
  }
  printf("\n");
  cl_int ret = my_clSetKernelArg(kernel, arg_index, arg_size, arg_value);
  return ret;
}

uint64_t start_time = 0;
uint64_t tns = 0;

cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue,
  cl_kernel kernel,
  cl_uint work_dim,
  const size_t *global_work_offset,
  const size_t *global_work_size,
  const size_t *local_work_size,
  cl_uint num_events_in_wait_list,
  const cl_event *event_wait_list,
  cl_event *event) {

  // SNPE doesn't use these
  assert(num_events_in_wait_list == 0);
  assert(global_work_offset == NULL);

  cl_int (*my_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *) = NULL;
  my_clEnqueueNDRangeKernel = reinterpret_cast<decltype(my_clEnqueueNDRangeKernel)>(dlsym(RTLD_NEXT, "REAL_clEnqueueNDRangeKernel"));


  uint64_t tb = nanos_since_boot();
  cl_int ret = my_clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
    global_work_offset, global_work_size, local_work_size,
    num_events_in_wait_list, event_wait_list, event);
  uint64_t te = nanos_since_boot();

  if (do_print) {
    tns += te-tb;
    printf("%10lu %10lu running -- %p %p -- %60s -- %p -> %p ", (tb-start_time)/1000, (tns/1000), kernel, event, kernels[kernel].c_str(), kernel_inputs[kernel], kernel_outputs[kernel]);
    printf("global -- ");
    for (int i = 0; i < work_dim; i++) {
      printf("%4zu ", global_work_size[i]);
    }
    printf("local -- ");
    for (int i = 0; i < work_dim; i++) {
      printf("%4zu ", local_work_size[i]);
    }
    printf("\n");
  }

  return ret;
}

cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret) {
  cl_mem (*my_clCreateBuffer)(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret) = NULL;
  my_clCreateBuffer = reinterpret_cast<decltype(my_clCreateBuffer)>(dlsym(RTLD_NEXT, "REAL_clCreateBuffer"));

  cl_mem ret = my_clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
  printf("%p = clCreateBuffer %zu\n", ret, size);
  return ret;
}

cl_mem clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, const cl_image_desc *image_desc, void *host_ptr, cl_int *errcode_ret) {
  cl_mem (*my_clCreateImage)(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, const cl_image_desc *image_desc, void *host_ptr, cl_int *errcode_ret) = NULL;
  my_clCreateImage = reinterpret_cast<decltype(my_clCreateImage)>(dlsym(RTLD_NEXT, "REAL_clCreateImage"));

  // SNPE only uses this
  assert(CL_MEM_OBJECT_IMAGE2D == image_desc->image_type);

  map<cl_mem_object_type, string> lc = {
    {CL_MEM_OBJECT_BUFFER, "CL_MEM_OBJECT_BUFFER"},
    {CL_MEM_OBJECT_IMAGE2D, "CL_MEM_OBJECT_IMAGE2D"},
    {CL_MEM_OBJECT_IMAGE3D, "CL_MEM_OBJECT_IMAGE3D"},
    {CL_MEM_OBJECT_IMAGE2D_ARRAY, "CL_MEM_OBJECT_IMAGE2D_ARRAY"},
    {CL_MEM_OBJECT_IMAGE1D, "CL_MEM_OBJECT_IMAGE1D"},
    {CL_MEM_OBJECT_IMAGE1D_ARRAY, "CL_MEM_OBJECT_IMAGE1D_ARRAY"},
    {CL_MEM_OBJECT_IMAGE1D_BUFFER, "CL_MEM_OBJECT_IMAGE1D_BUFFER"}};

  cl_mem ret = my_clCreateImage(context, flags, image_format, image_desc, host_ptr, errcode_ret);
  printf("%p = clCreateImage %s\n", ret, lc[image_desc->image_type].c_str());
  return ret;
}

void *dlsym(void *handle, const char *symbol) {
  void *(*my_dlsym)(void *handle, const char *symbol) = (void *(*)(void *handle, const char *symbol))((uintptr_t)dlopen-0x2d4);
  if (memcmp("REAL_", symbol, 5) == 0) {
    return my_dlsym(handle, symbol+5);
  } else if (strcmp("clCreateProgramWithSource", symbol) == 0) {
    return (void*)clCreateProgramWithSource;
  } else if (strcmp("clCreateKernel", symbol) == 0) {
    return (void*)clCreateKernel;
  } else if (strcmp("clEnqueueNDRangeKernel", symbol) == 0) {
    return (void*)clEnqueueNDRangeKernel;
  } else if (strcmp("clSetKernelArg", symbol) == 0) {
    return (void*)clSetKernelArg;
  } else if (strcmp("clCreateBuffer", symbol) == 0) {
    return (void*)clCreateBuffer;
  } else if (strcmp("clCreateImage", symbol) == 0) {
    return (void*)clCreateImage;
  } else {
    //printf("dlsym %s\n", symbol);
    return my_dlsym(handle, symbol);
  }
}

int main(int argc, char* argv[]) {
  float *output = (float*)calloc(0x10000, sizeof(float));
  SNPEModel mdl(argv[1], output, 0, USE_GPU_RUNTIME);

  float state[TEMPORAL_SIZE];
  mdl.addRecurrent(state, TEMPORAL_SIZE);

  float desire[DESIRE_LEN];
  mdl.addDesire(desire, DESIRE_LEN);

  float traffic_convention[TRAFFIC_CONVENTION_LEN];
  mdl.addTrafficConvention(traffic_convention, TRAFFIC_CONVENTION_LEN);

  float *input = (float*)calloc(0x1000000, sizeof(float));;
  printf("************** execute 1 **************\n");
  do_print = 0;
  mdl.execute(input, 0);
  printf("************** execute 2 **************\n");
  do_print = 0;
  mdl.execute(input, 0);
  printf("************** execute 3 **************\n");
  do_print = 1;
  start_time = nanos_since_boot();
  mdl.execute(input, 0);
}

