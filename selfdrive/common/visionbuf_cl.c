#include "visionbuf.h"

#include <fcntl.h>
#include <assert.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

VisionBuf visionbuf_allocate(size_t len) {
  // const size_t alignment = 4096;
  // void* addr = aligned_alloc(alignment, alignment * ((len - 1) / alignment + 1));
  void* addr = calloc(1, len);

  return (VisionBuf){
      .len = len, .addr = addr, .handle = 1, .fd = -1,
  };
}

cl_mem visionbuf_to_cl(const VisionBuf* buf, cl_device_id device_id, cl_context ctx) {
  // HACK because this platform is just for convenience
  VisionBuf *w_buf = (VisionBuf*)buf;
  cl_mem ret;
  *w_buf = visionbuf_allocate_cl(buf->len, device_id, ctx, &ret);
  return ret;
}

VisionBuf visionbuf_allocate_cl(size_t len, cl_device_id device_id, cl_context ctx, cl_mem *out_mem) {
  int err;
  assert(out_mem);

#if __OPENCL_VERSION__ >= 200
  void* host_ptr =
      clSVMAlloc(ctx, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, len, 0);
  assert(host_ptr);
#else
  void* host_ptr = calloc(1, len);

  cl_command_queue q = clCreateCommandQueue(ctx, device_id, 0, &err);
  assert(err == 0);
#endif

  cl_mem mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, len, host_ptr, &err);
  assert(err == 0);

  *out_mem = mem;

  return (VisionBuf){
      .len = len, .addr = host_ptr, .handle = 0, .fd = -1,
      .device_id = device_id, .ctx = ctx, .buf_cl = mem,

#if __OPENCL_VERSION__ < 200
      .copy_q = q,
#endif

  };
}

void visionbuf_sync(const VisionBuf* buf, int dir) {
  int err = 0;
  if (!buf->buf_cl) return;

#if __OPENCL_VERSION__ < 200
  if (dir == VISIONBUF_SYNC_FROM_DEVICE) {
    err = clEnqueueReadBuffer(buf->copy_q, buf->buf_cl, CL_FALSE, 0, buf->len, buf->addr, 0, NULL, NULL);
  } else {
    err = clEnqueueWriteBuffer(buf->copy_q, buf->buf_cl, CL_FALSE, 0, buf->len, buf->addr, 0, NULL, NULL);
  }
  assert(err == 0);
  clFinish(buf->copy_q);
#endif
}

void visionbuf_free(const VisionBuf* buf) {
  if (buf->handle) {
    free(buf->addr);
  } else {
    int err = clReleaseMemObject(buf->buf_cl);
    assert(err == 0);
#if __OPENCL_VERSION__ >= 200
    clSVMFree(buf->ctx, buf->addr);
#else
    free(buf->addr);
#endif
  }
}
