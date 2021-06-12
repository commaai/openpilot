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

void VisionBuf::allocate(size_t len) {
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

  this->owner = true;
  this->len = len;
  this->mmap_len = ion_alloc.len;
  this->addr = addr;
  this->handle = ion_alloc.handle;
  this->fd = ion_fd_data.fd;
}

void VisionBuf::import(){
  int err;
  assert(this->fd >= 0);

  ion_init();

  // Get handle
  struct ion_fd_data fd_data = {0};
  fd_data.fd = this->fd;
  err = ioctl(ion_fd, ION_IOC_IMPORT, &fd_data);
  assert(err == 0);

  this->owner = false;
  this->handle = fd_data.handle;
  this->addr = mmap(NULL, this->mmap_len, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd, 0);
  assert(this->addr != MAP_FAILED);
}

void VisionBuf::init_cl(cl_device_id device_id, cl_context ctx) {
  int err;

  assert(((uintptr_t)this->addr % DEVICE_PAGE_SIZE_CL) == 0);

  cl_mem_ion_host_ptr ion_cl = {0};
  ion_cl.ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_QCOM;
  ion_cl.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
  ion_cl.ion_filedesc = this->fd;
  ion_cl.ion_hostptr = this->addr;

  this->buf_cl = clCreateBuffer(ctx,
                              CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                              this->len, &ion_cl, &err);
  assert(err == 0);
}


void VisionBuf::sync(int dir) {
  int err;

  struct ion_flush_data flush_data = {0};
  flush_data.handle = this->handle;
  flush_data.vaddr = this->addr;
  flush_data.offset = 0;
  flush_data.length = this->len;

  // ION_IOC_INV_CACHES ~= DMA_FROM_DEVICE
  // ION_IOC_CLEAN_CACHES ~= DMA_TO_DEVICE
  // ION_IOC_CLEAN_INV_CACHES ~= DMA_BIDIRECTIONAL

  struct ion_custom_data custom_data = {0};

   assert(dir == VISIONBUF_SYNC_FROM_DEVICE || dir == VISIONBUF_SYNC_TO_DEVICE);
   custom_data.cmd = (dir == VISIONBUF_SYNC_FROM_DEVICE) ?
     ION_IOC_INV_CACHES : ION_IOC_CLEAN_CACHES;

  custom_data.arg = (unsigned long)&flush_data;
  err = ioctl(ion_fd, ION_IOC_CUSTOM, &custom_data);
  assert(err == 0);
}

void VisionBuf::free() {
  if (this->buf_cl){
    int err = clReleaseMemObject(this->buf_cl);
    assert(err == 0);
  }

  munmap(this->addr, this->mmap_len);
  close(this->fd);

  // Free the ION buffer if we also shared it
  if (this->owner){
    struct ion_handle_data handle_data = {.handle = this->handle};
    int ret = ioctl(ion_fd, ION_IOC_FREE, &handle_data);
    assert(ret == 0);
  }
}
