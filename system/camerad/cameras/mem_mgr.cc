#include "system/camerad/cameras/mem_mgr.h"

#include <cstring>

#include "common/swaglog.h"
#include "system/camerad/cameras/utils.h"

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
  struct cam_mem_mgr_release_cmd mem_mgr_release_cmd = {0};
  mem_mgr_release_cmd.buf_handle = handle;

  int ret = do_cam_control(video0_fd, CAM_REQ_MGR_RELEASE_BUF, &mem_mgr_release_cmd, sizeof(mem_mgr_release_cmd));
  assert(ret == 0);
}

// *** MemoryManager ***

void *MemoryManager::alloc_buf(int size, uint32_t *handle) {
  void *ptr;
  auto &cache = cached_allocations[size];
  if (!cache.empty()) {
    ptr = cache.front();
    cache.pop();
    *handle = handle_lookup[ptr];
  } else {
    ptr = alloc_w_mmu_hdl(video0_fd, size, handle);
    handle_lookup[ptr] = *handle;
    size_lookup[ptr] = size;
  }
  memset(ptr, 0, size);
  return ptr;
}

void MemoryManager::free(void *ptr) {
  cached_allocations[size_lookup[ptr]].push(ptr);
}

MemoryManager::~MemoryManager() {
  for (auto &x : cached_allocations) {
    while (!x.second.empty()) {
      void *ptr = x.second.front();
      x.second.pop();
      LOGD("freeing cached allocation %p with size %d", ptr, size_lookup[ptr]);
      munmap(ptr, size_lookup[ptr]);

      // release fd
      close(handle_lookup[ptr] >> 16);
      release(video0_fd, handle_lookup[ptr]);

      handle_lookup.erase(ptr);
      size_lookup.erase(ptr);
    }
  }
}

// SpectraBuf

SpectraBuf::~SpectraBuf() {
  if (video_fd >= 0 && ptr) {
    munmap(ptr, mmap_size);
    release(video_fd, handle);
  }
}

void SpectraBuf::init(int fd, int s, int a, bool shared_access, int mmu_hdl, int mmu_hdl2, int count) {
  video_fd = fd;
  size = s;
  alignment = a;
  mmap_size = aligned_size() * count;

  uint32_t flags = CAM_MEM_FLAG_HW_READ_WRITE | CAM_MEM_FLAG_KMD_ACCESS | CAM_MEM_FLAG_UMD_ACCESS | CAM_MEM_FLAG_CMD_BUF_TYPE;
  if (shared_access) {
    flags |= CAM_MEM_FLAG_HW_SHARED_ACCESS;
  }

  void *p = alloc_w_mmu_hdl(video_fd, mmap_size, (uint32_t *)&handle, alignment, flags, mmu_hdl, mmu_hdl2);
  ptr = (unsigned char *)p;
  assert(ptr != NULL);
}
