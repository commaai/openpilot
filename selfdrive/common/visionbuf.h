#ifndef IONBUF_H
#define IONBUF_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VisionBuf {
  size_t len;
  void* addr;
  int handle;
  int fd;

  cl_context ctx;
  cl_device_id device_id;
  cl_mem buf_cl;
  cl_command_queue copy_q;
} VisionBuf;

#define VISIONBUF_SYNC_FROM_DEVICE 0
#define VISIONBUF_SYNC_TO_DEVICE 1

VisionBuf visionbuf_allocate(size_t len);
VisionBuf visionbuf_allocate_cl(size_t len, cl_device_id device_id, cl_context ctx, cl_mem *out_mem);
cl_mem visionbuf_to_cl(const VisionBuf* buf, cl_device_id device_id, cl_context ctx);
void visionbuf_sync(const VisionBuf* buf, int dir);
void visionbuf_free(const VisionBuf* buf);

#ifdef __cplusplus
}
#endif

#endif
