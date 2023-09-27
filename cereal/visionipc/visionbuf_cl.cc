#include "visionbuf.h"

#include <atomic>
#include <stdio.h>
#include <fcntl.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>

std::atomic<int> offset = 0;

static void *malloc_with_fd(size_t len, int *fd) {
  char full_path[0x100];

#ifdef __APPLE__
  snprintf(full_path, sizeof(full_path)-1, "/tmp/visionbuf_%d_%d", getpid(), offset++);
#else
  snprintf(full_path, sizeof(full_path)-1, "/dev/shm/visionbuf_%d_%d", getpid(), offset++);
#endif

  *fd = open(full_path, O_RDWR | O_CREAT, 0664);
  assert(*fd >= 0);

  unlink(full_path);

  ftruncate(*fd, len);
  void *addr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, 0);
  assert(addr != MAP_FAILED);

  return addr;
}

void VisionBuf::allocate(size_t length) {
  this->len = length;
  this->mmap_len = this->len;
  this->addr = malloc_with_fd(this->len, &this->fd);
  this->frame_id = (uint64_t*)((uint8_t*)this->addr + this->len);
}

void VisionBuf::init_cl(cl_device_id device_id, cl_context ctx){
  int err;

  this->copy_q = clCreateCommandQueue(ctx, device_id, 0, &err);
  assert(err == 0);

  this->buf_cl = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, this->len, this->addr, &err);
  assert(err == 0);
}


void VisionBuf::import(){
  assert(this->fd >= 0);
  this->addr = mmap(NULL, this->mmap_len, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd, 0);
  assert(this->addr != MAP_FAILED);

  this->frame_id = (uint64_t*)((uint8_t*)this->addr + this->len);
}


int VisionBuf::sync(int dir) {
  int err = 0;
  if (!this->buf_cl) return 0;

  if (dir == VISIONBUF_SYNC_FROM_DEVICE) {
    err = clEnqueueReadBuffer(this->copy_q, this->buf_cl, CL_FALSE, 0, this->len, this->addr, 0, NULL, NULL);
  } else {
    err = clEnqueueWriteBuffer(this->copy_q, this->buf_cl, CL_FALSE, 0, this->len, this->addr, 0, NULL, NULL);
  }

  if (err == 0){
    err = clFinish(this->copy_q);
  }

  return err;
}

int VisionBuf::free() {
  int err = 0;
  if (this->buf_cl){
    err = clReleaseMemObject(this->buf_cl);
    if (err != 0) return err;

    err = clReleaseCommandQueue(this->copy_q);
    if (err != 0) return err;
  }

  err = munmap(this->addr, this->len);
  if (err != 0) return err;

  err = close(this->fd);
  return err;
}
