#pragma once

#include <sys/mman.h>

#include <cassert>
#include <functional>
#include <map>
#include <memory>
#include <queue>

#include "common/util.h"
#include "media/cam_req_mgr.h"

void *alloc_w_mmu_hdl(int video0_fd, int len, uint32_t *handle, int align = 8,
                      int flags = CAM_MEM_FLAG_KMD_ACCESS | CAM_MEM_FLAG_UMD_ACCESS | CAM_MEM_FLAG_CMD_BUF_TYPE,
                      int mmu_hdl = 0, int mmu_hdl2 = 0);
void release(int video0_fd, uint32_t handle);

class MemoryManager {
 public:
  void init(int _video0_fd) { video0_fd = _video0_fd; }
  ~MemoryManager();

  template <class T>
  auto alloc(int len, uint32_t *handle) {
    return std::unique_ptr<T, std::function<void(void *)>>((T *)alloc_buf(len, handle), [this](void *ptr) { this->free(ptr); });
  }

 private:
  void *alloc_buf(int len, uint32_t *handle);
  void free(void *ptr);

  std::map<void *, uint32_t> handle_lookup;
  std::map<void *, int> size_lookup;
  std::map<int, std::queue<void *>> cached_allocations;
  int video0_fd;
};

class SpectraBuf {
public:
  SpectraBuf() = default;
  ~SpectraBuf();
  void init(int fd, int s, int a, bool shared_access, int mmu_hdl = 0, int mmu_hdl2 = 0, int count = 1);
  uint32_t aligned_size() {
    return ALIGNED_SIZE(size, alignment);
  }

  int video_fd = -1;
  unsigned char *ptr = nullptr;
  int size = 0, alignment = 0, handle = 0, mmap_size = 0;
};
