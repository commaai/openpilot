#pragma once

#include "clutil.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VisionBuf {
  size_t len;
  size_t mmap_len;
  void* addr;
  int handle;
  int fd;

  CLContext *ctx;
  cl_mem buf_cl;
  cl_command_queue copy_q;
} VisionBuf;

#define VISIONBUF_SYNC_FROM_DEVICE 0
#define VISIONBUF_SYNC_TO_DEVICE 1

VisionBuf visionbuf_allocate(size_t len);
VisionBuf visionbuf_allocate_cl(CLContext *ctx, size_t len);
void visionbuf_sync(const VisionBuf* buf, int dir);
void visionbuf_free(const VisionBuf* buf);

#ifdef __cplusplus
}
#endif
