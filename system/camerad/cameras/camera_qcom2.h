#pragma once

#include <memory>
#include <utility>

#include "system/camerad/cameras/camera_common.h"
#include "system/camerad/cameras/camera_util.h"
#include "system/camerad/sensors/sensor.h"
#include "common/params.h"
#include "common/util.h"

#define FRAME_BUF_COUNT 4

#define ROAD_FL_MM 8.0f
#define WIDE_FL_MM 1.71f
#define DRIVER_FL_MM 1.71f

class CameraState {
public:
  MultiCameraState *multi_cam_state;
  std::unique_ptr<const SensorInfo> ci;
  bool enabled;

  std::mutex exp_lock;

  int exposure_time;
  bool dc_gain_enabled;
  int dc_gain_weight;
  int gain_idx;
  float analog_gain_frac;

  float cur_ev[3];
  float best_ev_score;
  int new_exp_g;
  int new_exp_t;

  Rect ae_xywh;
  float measured_grey_fraction;
  float target_grey_fraction;

  unique_fd sensor_fd;
  unique_fd csiphy_fd;

  int camera_num;
  float fl_pix;

  void handle_camera_event(void *evdat);
  void update_exposure_score(float desired_ev, int exp_t, int exp_g_idx, float exp_gain);
  void set_camera_exposure(float grey_frac);

  void sensors_start();

  void camera_open(MultiCameraState *multi_cam_state, int camera_num, bool enabled);
  void set_exposure_rect();
  void sensor_set_parameters();
  void camera_map_bufs(MultiCameraState *s);
  void camera_init(MultiCameraState *s, VisionIpcServer *v, cl_device_id device_id, cl_context ctx, VisionStreamType yuv_type, float focal_len);
  void camera_close();

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

  CameraBuf buf;
  MemoryManager mm;

  void config_isp(int io_mem_handle, int fence, int request_id, int buf0_mem_handle, int buf0_offset);
  void enqueue_req_multi(int start, int n, bool dp);
  void enqueue_buffer(int i, bool dp);
  int clear_req_queue();

  int sensors_init();
  void sensors_poke(int request_id);
  void sensors_i2c(const struct i2c_random_wr_payload* dat, int len, int op_code, bool data_word);

private:
  // for debugging
  Params params;
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
