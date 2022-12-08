#pragma once

#include <map>
#include <mutex>
#include <optional>
#include <queue>

#include <media/cam_req_mgr.h>

std::optional<int32_t> device_acquire(int fd, int32_t session_handle, void *data, uint32_t num_resources=1);
int device_config(int fd, int32_t session_handle, int32_t dev_handle, uint64_t packet_handle);
int device_control(int fd, int op_code, int session_handle, int dev_handle);
int do_cam_control(int fd, int op_code, void *handle, int size);
void *alloc_w_mmu_hdl(int video0_fd, int len, uint32_t *handle, int align = 8, int flags = CAM_MEM_FLAG_KMD_ACCESS | CAM_MEM_FLAG_UMD_ACCESS | CAM_MEM_FLAG_CMD_BUF_TYPE,
                      int mmu_hdl = 0, int mmu_hdl2 = 0);
void release(int video0_fd, uint32_t handle);

class MemoryManager {
  public:
    void init(int _video0_fd) { video0_fd = _video0_fd; }
    void *alloc(int len, uint32_t *handle);
    void free(void *ptr);
    ~MemoryManager();
  private:
    std::mutex lock;
    std::map<void *, uint32_t> handle_lookup;
    std::map<void *, int> size_lookup;
    std::map<int, std::queue<void *> > cached_allocations;
    int video0_fd;
};
