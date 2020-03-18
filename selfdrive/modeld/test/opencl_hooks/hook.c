#include <stdio.h>
#include <dlfcn.h>
#include <CL/cl.h>

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
  my_clEnqueueNDRangeKernel = dlsym(RTLD_NEXT, "clEnqueueNDRangeKernel");
  printf("hook clEnqueueNDRangeKernel(%p) %d\n", my_clEnqueueNDRangeKernel, work_dim);

  return my_clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
    global_work_offset, global_work_size, local_work_size,
    num_events_in_wait_list, event_wait_list, event);
}

void *dlsym(void *handle, const char *symbol) {
  void *(*my_dlsym)(void *handle, const char *symbol) = (void*)dlopen-0x2d4;
  if (strcmp("clEnqueueNDRangeKernel", symbol) == 0) {
    return clEnqueueNDRangeKernel;
  } else {
    printf("dlsym %s\n", symbol);
    return my_dlsym(handle, symbol);
  }
}

