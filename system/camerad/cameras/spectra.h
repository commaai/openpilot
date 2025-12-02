#pragma once

#include <sys/mman.h>
#include <functional>
#include <memory>
#include <queue>
#include <optional>
#include <utility>

#include "media/cam_req_mgr.h"

#include "common/util.h"
#include "common/swaglog.h"
#include "system/camerad/cameras/hw.h"
#include "system/camerad/cameras/camera_common.h"
#include "system/camerad/sensors/sensor.h"

#define MAX_IFE_BUFS 20

const int MIPI_SETTLE_CNT = 33;  // Calculated by camera_freqs.py

// For use with the Titan 170 ISP in the SDM845
// https://github.com/commaai/agnos-kernel-sdm845

// CSLDeviceType/CSLPacketOpcodesIFE from camx
// cam_packet_header.op_code = (device << 24) | (opcode);
#define CSLDeviceTypeImageSensor (0x01 << 24)
#define CSLDeviceTypeIFE         (0x0F << 24)
#define CSLDeviceTypeBPS         (0x10 << 24)
#define OpcodesIFEInitialConfig  0x0
#define OpcodesIFEUpdate         0x1

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
  ~MemoryManager();

  template <class T>
  auto alloc(int len, uint32_t *handle) {
    return std::unique_ptr<T, std::function<void(void *)>>((T*)alloc_buf(len, handle), [this](void *ptr) { this->free(ptr); });
  }

private:
  void *alloc_buf(int len, uint32_t *handle);
  void free(void *ptr);

  std::map<void *, uint32_t> handle_lookup;
  std::map<void *, int> size_lookup;
  std::map<int, std::queue<void *> > cached_allocations;
  int video0_fd;
};

class SpectraMaster {
public:
  void init();

  unique_fd video0_fd;
  unique_fd cam_sync_fd;
  unique_fd isp_fd;
  unique_fd icp_fd;
  int device_iommu = -1;
  int cdm_iommu = -1;
  int icp_device_iommu = -1;
  MemoryManager mem_mgr;
};

class SpectraBuf {
public:
  SpectraBuf() = default;

  ~SpectraBuf() {
    if (video_fd >= 0 && ptr) {
      munmap(ptr, mmap_size);
      release(video_fd, handle);
    }
  }

  void init(SpectraMaster *m, int s, int a, bool shared_access, int mmu_hdl = 0, int mmu_hdl2 = 0, int count = 1) {
    video_fd = m->video0_fd;
    size = s;
    alignment = a;
    mmap_size = aligned_size() * count;

    uint32_t flags = CAM_MEM_FLAG_HW_READ_WRITE | CAM_MEM_FLAG_KMD_ACCESS | CAM_MEM_FLAG_UMD_ACCESS | CAM_MEM_FLAG_CMD_BUF_TYPE;
    if (shared_access) {
      flags |= CAM_MEM_FLAG_HW_SHARED_ACCESS;
    }

    void *p = alloc_w_mmu_hdl(video_fd, mmap_size, (uint32_t*)&handle, alignment, flags, mmu_hdl, mmu_hdl2);
    ptr = (unsigned char*)p;
    assert(ptr != NULL);
  };

  uint32_t aligned_size() {
    return ALIGNED_SIZE(size, alignment);
  };

  int video_fd = -1;
  unsigned char *ptr = nullptr;
  int size = 0, alignment = 0, handle = 0, mmap_size = 0;
};

class SpectraCamera {
public:
  SpectraCamera(SpectraMaster *master, const CameraConfig &config);
  ~SpectraCamera();

  void camera_open(VisionIpcServer *v, cl_device_id device_id, cl_context ctx);
  bool handle_camera_event(const cam_req_mgr_message *event_data);
  void camera_close();
  void camera_map_bufs();
  void config_bps(int idx, int request_id);
  void config_ife(int idx, int request_id, bool init=false);

  int clear_req_queue();
  void enqueue_frame(uint64_t request_id);

  int sensors_init();
  void sensors_start();
  void sensors_poke(int request_id);
  void sensors_i2c(const struct i2c_random_wr_payload* dat, int len, int op_code, bool data_word);

  bool openSensor();
  void configISP();
  void configICP();
  void configCSIPHY();
  void linkDevices();
  void destroySyncObjectAt(int index);

  // *** state ***

  int ife_buf_depth = -1;
  bool open = false;
  bool enabled = true;
  CameraConfig cc;
  std::unique_ptr<const SensorInfo> sensor;

  // YUV image size
  uint32_t stride;
  uint32_t y_height;
  uint32_t uv_height;
  uint32_t uv_offset;
  uint32_t yuv_size;

  unique_fd sensor_fd;
  unique_fd csiphy_fd;

  int32_t session_handle = -1;
  int32_t sensor_dev_handle = -1;
  int32_t isp_dev_handle = -1;
  int32_t icp_dev_handle = -1;
  int32_t csiphy_dev_handle = -1;

  int32_t link_handle = -1;

  SpectraBuf ife_cmd;
  SpectraBuf ife_gamma_lut;
  SpectraBuf ife_linearization_lut;
  SpectraBuf ife_vignetting_lut;

  SpectraBuf bps_cmd;
  SpectraBuf bps_cdm_buffer;
  SpectraBuf bps_cdm_program_array;
  SpectraBuf bps_cdm_striping_bl;
  SpectraBuf bps_iq;
  SpectraBuf bps_striping;
  SpectraBuf bps_linearization_lut;
  std::vector<uint32_t> bps_lin_reg;
  std::vector<uint32_t> bps_ccm_reg;

  int buf_handle_yuv[MAX_IFE_BUFS] = {};
  int buf_handle_raw[MAX_IFE_BUFS] = {};
  int sync_objs_ife[MAX_IFE_BUFS] = {};
  int sync_objs_bps[MAX_IFE_BUFS] = {};
  uint64_t request_id_last = 0;
  uint64_t last_requeue_ts = 0;
  uint64_t frame_id_raw_last = 0;
  int invalid_request_count = 0;
  bool skip_expected = true;

  CameraBuf buf;
  SpectraMaster *m;

private:
  void clearAndRequeue(uint64_t from_request_id);
  bool validateEvent(uint64_t request_id, uint64_t frame_id_raw);
  bool waitForFrameReady(uint64_t request_id);
  bool processFrame(int buf_idx, uint64_t request_id, uint64_t frame_id_raw, uint64_t timestamp);
  static bool syncFirstFrame(int camera_id, uint64_t request_id, uint64_t raw_id, uint64_t timestamp);
  struct SyncData {
    uint64_t timestamp;
    uint64_t frame_id_offset = 0;
  };
  inline static std::map<int, SyncData> camera_sync_data;
  inline static bool first_frame_synced = false;

  // a mode for stressing edge cases: realignment, sync failures, etc.
  inline bool stress_test(std::string log) {
    static double last_trigger = 0;
    static double prob = std::stod(util::getenv("SPECTRA_ERROR_PROB", "-1"));
    static double dt = std::stod(util::getenv("SPECTRA_ERROR_DT", "1"));
    bool triggered = (prob > 0) && \
                     ((static_cast<double>(rand()) / RAND_MAX) < prob) && \
                     (millis_since_boot() - last_trigger) > dt;
    if (triggered) {
      last_trigger = millis_since_boot();
      LOGE("stress test (cam %d): %s", cc.camera_num, log.c_str());
    }
    return triggered;
  }
};
