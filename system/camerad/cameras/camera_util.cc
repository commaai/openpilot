#include "system/camerad/cameras/camera_util.h"

#include <cassert>

#include <sys/ioctl.h>
#include <sys/mman.h>

#include "common/swaglog.h"
#include "common/util.h"

// ************** low level camera helpers ****************
int do_cam_control(int fd, int op_code, void *handle, int size) {
  struct cam_control camcontrol = {0};
  camcontrol.op_code = op_code;
  camcontrol.handle = (uint64_t)handle;
  if (size == 0) {
    camcontrol.size = 8;
    camcontrol.handle_type = CAM_HANDLE_MEM_HANDLE;
  } else {
    camcontrol.size = size;
    camcontrol.handle_type = CAM_HANDLE_USER_POINTER;
  }

  int ret = HANDLE_EINTR(ioctl(fd, VIDIOC_CAM_CONTROL, &camcontrol));
  if (ret == -1) {
    LOGE("VIDIOC_CAM_CONTROL error: op_code %d - errno %d", op_code, errno);
  }
  return ret;
}

std::optional<int32_t> device_acquire(int fd, int32_t session_handle, void *data, uint32_t num_resources) {
  struct cam_acquire_dev_cmd cmd = {
      .session_handle = session_handle,
      .handle_type = CAM_HANDLE_USER_POINTER,
      .num_resources = (uint32_t)(data ? num_resources : 0),
      .resource_hdl = (uint64_t)data,
  };
  int err = do_cam_control(fd, CAM_ACQUIRE_DEV, &cmd, sizeof(cmd));
  return err == 0 ? std::make_optional(cmd.dev_handle) : std::nullopt;
}

int device_config(int fd, int32_t session_handle, int32_t dev_handle, uint64_t packet_handle) {
  struct cam_config_dev_cmd cmd = {
      .session_handle = session_handle,
      .dev_handle = dev_handle,
      .packet_handle = packet_handle,
  };
  return do_cam_control(fd, CAM_CONFIG_DEV, &cmd, sizeof(cmd));
}

int device_control(int fd, int op_code, int session_handle, int dev_handle) {
  // start stop and release are all the same
  struct cam_start_stop_dev_cmd cmd { .session_handle = session_handle, .dev_handle = dev_handle };
  return do_cam_control(fd, op_code, &cmd, sizeof(cmd));
}

void *alloc_w_mmu_hdl(int video0_fd, int len, uint32_t *handle, int align, int flags, int mmu_hdl, int mmu_hdl2) {
  struct cam_mem_mgr_alloc_cmd mem_mgr_alloc_cmd = {0};
  mem_mgr_alloc_cmd.len = len;
  mem_mgr_alloc_cmd.align = align;
  mem_mgr_alloc_cmd.flags = flags;
  mem_mgr_alloc_cmd.num_hdl = 0;
  if (mmu_hdl != 0) {
    mem_mgr_alloc_cmd.mmu_hdls[0] = mmu_hdl;
    mem_mgr_alloc_cmd.num_hdl++;
  }
  if (mmu_hdl2 != 0) {
    mem_mgr_alloc_cmd.mmu_hdls[1] = mmu_hdl2;
    mem_mgr_alloc_cmd.num_hdl++;
  }

  do_cam_control(video0_fd, CAM_REQ_MGR_ALLOC_BUF, &mem_mgr_alloc_cmd, sizeof(mem_mgr_alloc_cmd));
  *handle = mem_mgr_alloc_cmd.out.buf_handle;

  void *ptr = NULL;
  if (mem_mgr_alloc_cmd.out.fd > 0) {
    ptr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, mem_mgr_alloc_cmd.out.fd, 0);
    assert(ptr != MAP_FAILED);
  }

  // LOGD("allocated: %x %d %llx mapped %p", mem_mgr_alloc_cmd.out.buf_handle, mem_mgr_alloc_cmd.out.fd, mem_mgr_alloc_cmd.out.vaddr, ptr);

  return ptr;
}

void release(int video0_fd, uint32_t handle) {
  int ret;
  struct cam_mem_mgr_release_cmd mem_mgr_release_cmd = {0};
  mem_mgr_release_cmd.buf_handle = handle;

  ret = do_cam_control(video0_fd, CAM_REQ_MGR_RELEASE_BUF, &mem_mgr_release_cmd, sizeof(mem_mgr_release_cmd));
  assert(ret == 0);
}

void release_fd(int video0_fd, uint32_t handle) {
  // handle to fd
  close(handle>>16);
  release(video0_fd, handle);
}

void *MemoryManager::alloc_buf(int size, uint32_t *handle) {
  lock.lock();
  void *ptr;
  if (!cached_allocations[size].empty()) {
    ptr = cached_allocations[size].front();
    cached_allocations[size].pop();
    *handle = handle_lookup[ptr];
  } else {
    ptr = alloc_w_mmu_hdl(video0_fd, size, handle);
    handle_lookup[ptr] = *handle;
    size_lookup[ptr] = size;
  }
  lock.unlock();
  return ptr;
}

void MemoryManager::free(void *ptr) {
  lock.lock();
  cached_allocations[size_lookup[ptr]].push(ptr);
  lock.unlock();
}

MemoryManager::~MemoryManager() {
  for (auto& x : cached_allocations) {
    while (!x.second.empty()) {
      void *ptr = x.second.front();
      x.second.pop();
      LOGD("freeing cached allocation %p with size %d", ptr, size_lookup[ptr]);
      munmap(ptr, size_lookup[ptr]);
      release_fd(video0_fd, handle_lookup[ptr]);
      handle_lookup.erase(ptr);
      size_lookup.erase(ptr);
    }
  }
}
