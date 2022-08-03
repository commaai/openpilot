#pragma once

#include <cstdint>
#include <map>
#include <utility>

#include <media/cam_req_mgr.h>

#include "system/camerad/cameras/camera_common.h"
#include "common/util.h"

#define FRAME_BUF_COUNT 4

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

class CameraState {
public:
  MultiCameraState *multi_cam_state;
  CameraInfo ci;
  bool enabled;

  std::mutex exp_lock;

  int exposure_time;
  bool dc_gain_enabled;
  float analog_gain_frac;

  float cur_ev[3];
  float min_ev, max_ev;

  float measured_grey_fraction;
  float target_grey_fraction;
  int gain_idx;

  unique_fd sensor_fd;
  unique_fd csiphy_fd;

  int camera_num;

  void handle_camera_event(void *evdat);
  void set_camera_exposure(float grey_frac);

  void sensors_start();

  void camera_open();
  void camera_init(MultiCameraState *multi_cam_state, VisionIpcServer * v, int camera_id, int camera_num, unsigned int fps, cl_device_id device_id, cl_context ctx, VisionStreamType yuv_type, bool enabled);
  void camera_close();

  std::map<uint16_t, uint16_t> ar0231_parse_registers(uint8_t *data, std::initializer_list<uint16_t> addrs);

  int32_t session_handle;
  int32_t sensor_dev_handle;
  int32_t isp_dev_handle;
  int32_t csiphy_dev_handle;

  int32_t link_handle;

  int buf0_handle;
  int buf_handle[FRAME_BUF_COUNT];
  int sync_objs[FRAME_BUF_COUNT];
  int request_ids[FRAME_BUF_COUNT];
  int request_id_last;
  int frame_id_last;
  int idx_offset;
  bool skipped;
  int camera_id;

  CameraBuf buf;
  MemoryManager mm;

private:
  void config_isp(int io_mem_handle, int fence, int request_id, int buf0_mem_handle, int buf0_offset);
  void enqueue_req_multi(int start, int n, bool dp);
  void enqueue_buffer(int i, bool dp);
  int clear_req_queue();

  int sensors_init();
  void sensors_poke(int request_id);
  void sensors_i2c(struct i2c_random_wr_payload* dat, int len, int op_code, bool data_word);

  // Register parsing
  std::map<uint16_t, std::pair<int, int>> ar0231_register_lut;
  std::map<uint16_t, std::pair<int, int>> ar0231_build_register_lut(uint8_t *data);
};

typedef struct MultiCameraState {
  unique_fd video0_fd;
  unique_fd cam_sync_fd;
  unique_fd isp_fd;
  int device_iommu;
  int cdm_iommu;

  CameraState road_cam;
  CameraState wide_road_cam;
  CameraState driver_cam;

  PubMaster *pm;
} MultiCameraState;
