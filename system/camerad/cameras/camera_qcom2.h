#pragma once

#include <cstdint>
#include <map>
#include <utility>

#include <media/cam_req_mgr.h>

#include "system/camerad/cameras/camera_common.h"
#include "system/camerad/cameras/camera_util.h"
#include "common/params.h"
#include "common/util.h"

#define FRAME_BUF_COUNT 4
#define ANALOG_GAIN_MAX_CNT 55

class CameraState {
public:
  MultiCameraState *multi_cam_state;
  CameraInfo ci;
  bool enabled;

  std::mutex exp_lock;

  int exposure_time;
  bool dc_gain_enabled;
  int dc_gain_weight;
  int gain_idx;
  float analog_gain_frac;

  int exposure_time_min;
  int exposure_time_max;

  float dc_gain_factor;
  int dc_gain_min_weight;
  int dc_gain_max_weight;
  float dc_gain_on_grey;
  float dc_gain_off_grey;

  float sensor_analog_gains[ANALOG_GAIN_MAX_CNT];
  int analog_gain_min_idx;
  int analog_gain_max_idx;
  int analog_gain_rec_idx;
  int analog_gain_cost_delta;
  float analog_gain_cost_low;
  float analog_gain_cost_high;

  float cur_ev[3];
  float min_ev, max_ev;
  float best_ev_score;
  int new_exp_g;
  int new_exp_t;

  float measured_grey_fraction;
  float target_grey_fraction;
  float target_grey_factor;

  unique_fd sensor_fd;
  unique_fd csiphy_fd;

  int camera_num;

  void handle_camera_event(void *evdat);
  void update_exposure_score(float desired_ev, int exp_t, int exp_g_idx, float exp_gain);
  void set_camera_exposure(float grey_frac);

  void sensors_start();

  void camera_open(MultiCameraState *multi_cam_state, int camera_num, bool enabled);
  void camera_set_parameters();
  void camera_map_bufs(MultiCameraState *s);
  void camera_init(MultiCameraState *s, VisionIpcServer *v, int camera_id, unsigned int fps, cl_device_id device_id, cl_context ctx, VisionStreamType yuv_type);
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
