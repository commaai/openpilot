#include "visionbuf.h"

#include <stdio.h>
#include <fcntl.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include "common/clutil.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int offset = 0;
void *malloc_with_fd(size_t len, int *fd) {
  char full_path[0x100];
#ifdef __APPLE__
  snprintf(full_path, sizeof(full_path)-1, "/tmp/visionbuf_%d_%d", getpid(), offset++);
#else
  snprintf(full_path, sizeof(full_path)-1, "/dev/shm/visionbuf_%d_%d", getpid(), offset++);
#endif
  *fd = open(full_path, O_RDWR | O_CREAT, 0777);
  assert(*fd >= 0);
  unlink(full_path);
  ftruncate(*fd, len);
  void *addr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, 0);
  assert(addr != MAP_FAILED);
  return addr;
}

VisionBuf visionbuf_allocate(size_t len) {
  // const size_t alignment = 4096;
  // void* addr = aligned_alloc(alignment, alignment * ((len - 1) / alignment + 1));
  //void* addr = calloc(1, len);

  int fd;
  void *addr = malloc_with_fd(len, &fd);

  return (VisionBuf){
      .len = len, .addr = addr, .handle = 1, .fd = fd,
  };
}

VisionBuf visionbuf_allocate_cl(size_t len, cl_device_id device_id, cl_context ctx) {
#if __OPENCL_VERSION__ >= 200
  void* host_ptr =
      clSVMAlloc(ctx, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, len, 0);
  assert(host_ptr);
#else
  int fd;
  void* host_ptr = malloc_with_fd(len, &fd);

  cl_command_queue q = CL_CHECK_ERR(clCreateCommandQueue(ctx, device_id, 0, &err));
#endif

  cl_mem mem = CL_CHECK_ERR(clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, len, host_ptr, &err));

  return (VisionBuf){
      .len = len, .addr = host_ptr, .handle = 0, .fd = fd,
      .device_id = device_id, .ctx = ctx, .buf_cl = mem,

#if __OPENCL_VERSION__ < 200
      .copy_q = q,
#endif

  };
}

void visionbuf_sync(const VisionBuf* buf, int dir) {
  if (!buf->buf_cl) return;

#if __OPENCL_VERSION__ < 200
  if (dir == VISIONBUF_SYNC_FROM_DEVICE) {
    CL_CHECK(clEnqueueReadBuffer(buf->copy_q, buf->buf_cl, CL_FALSE, 0, buf->len, buf->addr, 0, NULL, NULL));
  } else {
    CL_CHECK(clEnqueueWriteBuffer(buf->copy_q, buf->buf_cl, CL_FALSE, 0, buf->len, buf->addr, 0, NULL, NULL));
  }
  clFinish(buf->copy_q);
#endif
}

void visionbuf_free(const VisionBuf* buf) {
  if (buf->handle) {
    munmap(buf->addr, buf->len);
    close(buf->fd);
  } else {
    CL_CHECK(clReleaseMemObject(buf->buf_cl));
#if __OPENCL_VERSION__ >= 200
    clSVMFree(buf->ctx, buf->addr);
#else
    CL_CHECK(clReleaseCommandQueue(buf->copy_q));
    munmap(buf->addr, buf->len);
    close(buf->fd);
#endif
  }
}
