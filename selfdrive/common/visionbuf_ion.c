#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/ion.h>
#include <CL/cl_ext.h>

#include <msm_ion.h>

#include "visionbuf.h"


// just hard-code these for convenience
// size_t device_page_size = 0;
// clGetDeviceInfo(device_id, CL_DEVICE_PAGE_SIZE_QCOM,
//                 sizeof(device_page_size), &device_page_size,
//                 NULL);

// size_t padding_cl = 0;
// clGetDeviceInfo(device_id, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM,
//                 sizeof(padding_cl), &padding_cl,
//                 NULL);
#define DEVICE_PAGE_SIZE_CL 4096
#define PADDING_CL 0

static int ion_fd = -1;
static void ion_init() {
  if (ion_fd == -1) {
    ion_fd = open("/dev/ion", O_RDWR | O_NONBLOCK);
  }
}

VisionBuf visionbuf_allocate(size_t len) {
  int err;

  ion_init();

  struct ion_allocation_data ion_alloc = {0};
  ion_alloc.len = len + PADDING_CL;
  ion_alloc.align = 4096;
  ion_alloc.heap_id_mask = 1 << ION_IOMMU_HEAP_ID;
  ion_alloc.flags = ION_FLAG_CACHED;

  err = ioctl(ion_fd, ION_IOC_ALLOC, &ion_alloc);
  assert(err == 0);

  struct ion_fd_data ion_fd_data = {0};
  ion_fd_data.handle = ion_alloc.handle;
  err = ioctl(ion_fd, ION_IOC_SHARE, &ion_fd_data);
  assert(err == 0);

  void *addr = mmap(NULL, ion_alloc.len,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED, ion_fd_data.fd, 0);
  assert(addr != MAP_FAILED);

  memset(addr, 0, ion_alloc.len);

  return (VisionBuf){
    .len = len,
    .mmap_len = ion_alloc.len,
    .addr = addr,
    .handle = ion_alloc.handle,
    .fd = ion_fd_data.fd,
  };
}

VisionBuf visionbuf_allocate_cl(size_t len, cl_device_id device_id, cl_context ctx, cl_mem *out_mem) {
  VisionBuf r = visionbuf_allocate(len);
  *out_mem = visionbuf_to_cl(&r, device_id, ctx);
  r.buf_cl = *out_mem;
  return r;
}

cl_mem visionbuf_to_cl(const VisionBuf* buf, cl_device_id device_id, cl_context ctx) {
  int err = 0;

  assert(((uintptr_t)buf->addr % DEVICE_PAGE_SIZE_CL) == 0);

  cl_mem_ion_host_ptr ion_cl = {0};
  ion_cl.ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_QCOM;
  ion_cl.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
  ion_cl.ion_filedesc = buf->fd;
  ion_cl.ion_hostptr = buf->addr;

  cl_mem mem = clCreateBuffer(ctx,
                              CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                              buf->len, &ion_cl, &err);
  assert(err == 0);

  return mem;
}

void visionbuf_sync(const VisionBuf* buf, int dir) {
  int err;

  struct ion_fd_data fd_data = {0};
  fd_data.fd = buf->fd;
  err = ioctl(ion_fd, ION_IOC_IMPORT, &fd_data);
  assert(err == 0);

  struct ion_flush_data flush_data = {0};
  flush_data.handle = fd_data.handle;
  flush_data.vaddr = buf->addr;
  flush_data.offset = 0;
  flush_data.length = buf->len;

  // ION_IOC_INV_CACHES ~= DMA_FROM_DEVICE
  // ION_IOC_CLEAN_CACHES ~= DMA_TO_DEVICE
  // ION_IOC_CLEAN_INV_CACHES ~= DMA_BIDIRECTIONAL

  struct ion_custom_data custom_data = {0};

  switch (dir) {
  case VISIONBUF_SYNC_FROM_DEVICE:
    custom_data.cmd = ION_IOC_INV_CACHES;
    break;
  case VISIONBUF_SYNC_TO_DEVICE:
    custom_data.cmd = ION_IOC_CLEAN_CACHES;
    break;
  default:
    assert(0);
  }

  custom_data.arg = (unsigned long)&flush_data;
  err = ioctl(ion_fd, ION_IOC_CUSTOM, &custom_data);
  assert(err == 0);

  struct ion_handle_data handle_data = {0};
  handle_data.handle = fd_data.handle;
  err = ioctl(ion_fd, ION_IOC_FREE, &handle_data);
  assert(err == 0);
}

void visionbuf_free(const VisionBuf* buf) {
  clReleaseMemObject(buf->buf_cl);
  munmap(buf->addr, buf->mmap_len);
  close(buf->fd);
  struct ion_handle_data handle_data = {
    .handle = buf->handle,
  };
  int ret = ioctl(ion_fd, ION_IOC_FREE, &handle_data);
  assert(ret == 0);
}
