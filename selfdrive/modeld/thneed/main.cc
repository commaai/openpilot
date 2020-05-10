#include <cassert>
#include <stdlib.h>
#include <dlfcn.h>
#include <CL/cl.h>
#include "../runners/snpemodel.h"
#include "../models/driving.h"

#include <string>
#include <map>
using namespace std;

#define TEMPORAL_SIZE 512
#define DESIRE_LEN 8
#define TRAFFIC_CONVENTION_LEN 2

FILE *f = NULL;

cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret) {
  cl_program (*my_clCreateProgramWithSource)(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret) = NULL;
  my_clCreateProgramWithSource = reinterpret_cast<decltype(my_clCreateProgramWithSource)>(dlsym(RTLD_NEXT, "REAL_clCreateProgramWithSource"));
  printf("clCreateProgramWithSource: %d\n", count);

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

cl_kernel clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret) {
  cl_kernel (*my_clCreateKernel)(cl_program program, const char *kernel_name, cl_int *errcode_ret) = NULL;
  my_clCreateKernel = reinterpret_cast<decltype(my_clCreateKernel)>(dlsym(RTLD_NEXT, "REAL_clCreateKernel"));
  cl_kernel ret = my_clCreateKernel(program, kernel_name, errcode_ret);

  //printf("clCreateKernel: %s -> %p\n", kernel_name, ret);
  kernels.insert(make_pair(ret, kernel_name));
  return ret;
}

cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue,
  cl_kernel kernel,
  cl_uint work_dim,
  const size_t *global_work_offset,
  const size_t *global_work_size,
  const size_t *local_work_size,
  cl_uint num_events_in_wait_list,
  const cl_event *event_wait_list,
  cl_event *event) {

  cl_int (*my_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *) = NULL;
  my_clEnqueueNDRangeKernel = reinterpret_cast<decltype(my_clEnqueueNDRangeKernel)>(dlsym(RTLD_NEXT, "REAL_clEnqueueNDRangeKernel"));

  printf("running %s ", kernels[kernel].c_str());
  for (int i = 0; i < work_dim; i++) {
    printf("%4zu ", global_work_size[i]);
  }
  printf("\n");

  cl_int ret = my_clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
    global_work_offset, global_work_size, local_work_size,
    num_events_in_wait_list, event_wait_list, event);
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
  mdl.execute(input, 0);
}

