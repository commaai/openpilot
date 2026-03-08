#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/ion.h>

#include <msm_ion.h>

#include "msgq/visionipc/visionbuf.h"

// keep trying if x gets interrupted by a signal
#define HANDLE_EINTR(x)                                       \
  ({                                                          \
    decltype(x) ret;                                          \
    int try_cnt = 0;                                          \
    do {                                                      \
      ret = (x);                                              \
    } while (ret == -1 && errno == EINTR && try_cnt++ < 100); \
    ret;                                                      \
  })

struct IonFileHandle {
  IonFileHandle() {
    fd = open("/dev/ion", O_RDWR | O_NONBLOCK);
    assert(fd >= 0);
  }
  ~IonFileHandle() {
    close(fd);
  }
  int fd = -1;
};

int ion_fd() {
  static IonFileHandle fh;
  return fh.fd;
}

void VisionBuf::allocate(size_t length) {
  struct ion_allocation_data ion_alloc = {0};
  ion_alloc.len = length + sizeof(uint64_t);
  ion_alloc.align = 4096;
  ion_alloc.heap_id_mask = 1 << ION_IOMMU_HEAP_ID;
  ion_alloc.flags = ION_FLAG_CACHED;

  int err = HANDLE_EINTR(ioctl(ion_fd(), ION_IOC_ALLOC, &ion_alloc));
  assert(err == 0);

  struct ion_fd_data ion_fd_data = {0};
  ion_fd_data.handle = ion_alloc.handle;
  err = HANDLE_EINTR(ioctl(ion_fd(), ION_IOC_SHARE, &ion_fd_data));
  assert(err == 0);

  void *mmap_addr = mmap(NULL, ion_alloc.len,
                         PROT_READ | PROT_WRITE,
                         MAP_SHARED, ion_fd_data.fd, 0);
  assert(mmap_addr != MAP_FAILED);

  memset(mmap_addr, 0, ion_alloc.len);

  this->len = length;
  this->mmap_len = ion_alloc.len;
  this->addr = mmap_addr;
  this->handle = ion_alloc.handle;
  this->fd = ion_fd_data.fd;
  this->frame_id = (uint64_t*)((uint8_t*)this->addr + this->len);
}

void VisionBuf::import(){
  int err;
  assert(this->fd >= 0);

  // Get handle
  struct ion_fd_data fd_data = {0};
  fd_data.fd = this->fd;
  err = HANDLE_EINTR(ioctl(ion_fd(), ION_IOC_IMPORT, &fd_data));
  assert(err == 0);

  this->handle = fd_data.handle;
  this->addr = mmap(NULL, this->mmap_len, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd, 0);
  assert(this->addr != MAP_FAILED);

  this->frame_id = (uint64_t*)((uint8_t*)this->addr + this->len);
}

void VisionBuf::init_yuv(size_t init_width, size_t init_height, size_t init_stride, size_t init_uv_offset){
  this->width = init_width;
  this->height = init_height;
  this->stride = init_stride;
  this->uv_offset = init_uv_offset;

  this->y = (uint8_t *)this->addr;
  this->uv = this->y + this->uv_offset;
}

int VisionBuf::sync(int dir) {
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
  return HANDLE_EINTR(ioctl(ion_fd(), ION_IOC_CUSTOM, &custom_data));
}

int VisionBuf::free() {
  int err = munmap(this->addr, this->mmap_len);
  if (err != 0) return err;

  err = close(this->fd);
  if (err != 0) return err;

  struct ion_handle_data handle_data = {.handle = this->handle};
  return HANDLE_EINTR(ioctl(ion_fd(), ION_IOC_FREE, &handle_data));
}

uint64_t VisionBuf::get_frame_id() {
  return *frame_id;
}

void VisionBuf::set_frame_id(uint64_t id) {
  *frame_id = id;
}
