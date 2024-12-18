#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "media/cam_req_mgr.h"

#include "common/util.h"
#include "system/camerad/cameras/tici.h"
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

  std::mutex lock;
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
};

class SpectraBuf {
public:
  void init(SpectraMaster *m, int s, int a, int flags, int mmu_hdl = 0, int mmu_hdl2 = 0, int count=1) {
    size = s;
    alignment = a;
    void *p = alloc_w_mmu_hdl(m->video0_fd, ALIGNED_SIZE(size, alignment)*count, (uint32_t*)&handle, alignment, flags, mmu_hdl, mmu_hdl2);
    ptr = (unsigned char*)p;
    assert(ptr != NULL);
  };

  uint32_t aligned_size() {
    return ALIGNED_SIZE(size, alignment);
  };

  unsigned char *ptr;
  int size, alignment, handle;
};

class SpectraCamera {
public:
  SpectraCamera(SpectraMaster *master, const CameraConfig &config, bool raw);
  ~SpectraCamera();

  void camera_open(VisionIpcServer *v, cl_device_id device_id, cl_context ctx);
  void handle_camera_event(const cam_req_mgr_message *event_data);
  void camera_close();
  void camera_map_bufs();
  void config_bps(int idx, int request_id);
  void config_ife(int idx, int request_id, bool init=false);

  int clear_req_queue();
  void enqueue_buffer(int i, bool dp);
  void enqueue_req_multi(uint64_t start, int n, bool dp);

  int sensors_init();
  void sensors_start();
  void sensors_poke(int request_id);
  void sensors_i2c(const struct i2c_random_wr_payload* dat, int len, int op_code, bool data_word);

  bool openSensor();
  void configISP();
  void configICP();
  void configCSIPHY();
  void linkDevices();

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

  int buf_handle_yuv[MAX_IFE_BUFS] = {};
  int buf_handle_raw[MAX_IFE_BUFS] = {};
  int sync_objs[MAX_IFE_BUFS] = {};
  int sync_objs_bps_out[MAX_IFE_BUFS] = {};
  uint64_t request_ids[MAX_IFE_BUFS] = {};
  uint64_t request_id_last = 0;
  uint64_t frame_id_last = 0;
  uint64_t idx_offset = 0;
  bool skipped = true;

  bool is_raw;

  CameraBuf buf;
  MemoryManager mm;
  SpectraMaster *m;
};
