#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <CL/cl.h>
#include <stdint.h>
#include <time.h>

static inline uint64_t nanos_since_boot() {
  struct timespec t;
  clock_gettime(CLOCK_BOOTTIME, &t);
  return t.tv_sec * 1000000000ULL + t.tv_nsec;
}

struct kernel {
  cl_kernel k;
  const char *name;
  cl_program p;
};


int k_index = 0;
struct kernel kk[0x1000] = {0};

FILE *f = NULL;

cl_program clCreateProgramWithSource(cl_context context,
  cl_uint count,
  const char **strings,
  const size_t *lengths,
  cl_int *errcode_ret) {
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

  cl_program (*my_clCreateProgramWithSource)(cl_context context,
    cl_uint count,
    const char **strings,
    const size_t *lengths,
    cl_int *errcode_ret) = dlsym(RTLD_NEXT, "REAL_clCreateProgramWithSource");

  return my_clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
}

cl_program clCreateProgramWithBinary(cl_context context,
  cl_uint num_devices,
  const cl_device_id *device_list,
  const size_t *lengths,
  const unsigned char **binaries,
  cl_int *binary_status,
  cl_int *errcode_ret) {
  printf("clCreateProgramWithBinary\n");

  cl_program (*my_clCreateProgramWithBinary)(cl_context context,
    cl_uint num_devices,
    const cl_device_id *device_list,
    const size_t *lengths,
    const unsigned char **binaries,
    cl_int *binary_status,
    cl_int *errcode_ret) = dlsym(RTLD_NEXT, "REAL_clCreateProgramWithBinary");

  return my_clCreateProgramWithBinary(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
}

cl_kernel clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret) {
  cl_kernel (*my_clCreateKernel)(cl_program program, const char *kernel_name, cl_int *errcode_ret);
  my_clCreateKernel = dlsym(RTLD_NEXT, "REAL_clCreateKernel");
  cl_kernel ret = my_clCreateKernel(program, kernel_name, errcode_ret);
  //printf("clCreateKernel: %s -> %p\n", kernel_name, ret);
  
  char *tmp = (char*)malloc(strlen(kernel_name)+1);
  strcpy(tmp, kernel_name);

  kk[k_index].k = ret;
  kk[k_index].name = tmp;
  kk[k_index].p = program;
  k_index++;
  return ret;
}


uint64_t start_time = 0;
int cnt = 0;

cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue,
  cl_kernel kernel,
  cl_uint work_dim,
  const size_t *global_work_offset,
  const size_t *global_work_size,
  const size_t *local_work_size,
  cl_uint num_events_in_wait_list,
  const cl_event *event_wait_list,
  cl_event *event) {

  cl_int (*my_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint,
    const size_t *, const size_t *, const size_t *,
    cl_uint, const cl_event *, cl_event *) = NULL;
  my_clEnqueueNDRangeKernel = dlsym(RTLD_NEXT, "REAL_clEnqueueNDRangeKernel");

  if (start_time == 0) {
    start_time = nanos_since_boot();
  }

  // get kernel name
  const char *name = NULL;
  cl_program p;
  for (int i = 0; i < k_index; i++) {
    if (kk[i].k == kernel) {
      name = kk[i].name;
      p = kk[i].p;
      break;
    }
  }

  uint64_t tb = nanos_since_boot();
  cl_int ret = my_clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
    global_work_offset, global_work_size, local_work_size,
    num_events_in_wait_list, event_wait_list, event);
  uint64_t te = nanos_since_boot();

  printf("%10lu run%8d in %5ld us command_queue:%p work_dim:%d event:%p  ", (tb-start_time)/1000, cnt++, (te-tb)/1000,
    command_queue, work_dim, event);
  for (int i = 0; i < work_dim; i++) {
    printf("%4zu ", global_work_size[i]);
  }
  printf("%p %s\n", p, name);
  return ret;
}

void *dlsym(void *handle, const char *symbol) {
  void *(*my_dlsym)(void *handle, const char *symbol) = (void*)dlopen-0x2d4;
  if (memcmp("REAL_", symbol, 5) == 0) {
    return my_dlsym(handle, symbol+5);
  } else if (strcmp("clEnqueueNDRangeKernel", symbol) == 0) {
    return clEnqueueNDRangeKernel;
  } else if (strcmp("clCreateKernel", symbol) == 0) {
    return clCreateKernel;
  } else if (strcmp("clCreateProgramWithSource", symbol) == 0) {
    return clCreateProgramWithSource;
  } else if (strcmp("clCreateProgramWithBinary", symbol) == 0) {
    return clCreateProgramWithBinary;
  } else {
    printf("dlsym %s\n", symbol);
    return my_dlsym(handle, symbol);
  }
}

