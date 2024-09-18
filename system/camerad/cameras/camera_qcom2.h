#pragma once

#include <memory>
#include <utility>

#include "system/camerad/cameras/camera_common.h"
#include "system/camerad/cameras/camera_util.h"
#include "system/camerad/cameras/tici.h"
#include "system/camerad/cameras/spectra.h"
#include "system/camerad/sensors/sensor.h"
#include "common/util.h"


class CameraState : public SpectraCamera {
public:
  std::mutex exp_lock;

  int exposure_time = 5;
  bool dc_gain_enabled = false;
  int dc_gain_weight = 0;
  int gain_idx = 0;
  float analog_gain_frac = 0;

  float cur_ev[3] = {};
  float best_ev_score = 0;
  int new_exp_g = 0;
  int new_exp_t = 0;

  Rect ae_xywh = {};
  float measured_grey_fraction = 0;
  float target_grey_fraction = 0.3;

  float fl_pix = 0;

  CameraState(MultiCameraState *multi_camera_state, const CameraConfig &config);
  void handle_camera_event(void *evdat);
  void update_exposure_score(float desired_ev, int exp_t, int exp_g_idx, float exp_gain);
  void set_camera_exposure(float grey_frac);

  void sensors_start();

  void camera_map_bufs();
  void camera_init(VisionIpcServer *v, cl_device_id device_id, cl_context ctx);
  void enqueue_req_multi(uint64_t start, int n, bool dp);
  void enqueue_buffer(int i, bool dp);
  void sensors_poke(int request_id);

  // these stay
  void set_exposure_rect();
  void sensor_set_parameters();
  void run();
};

class MultiCameraState {
public:
  MultiCameraState();
  ~MultiCameraState() {
    if (pm != nullptr) {
      delete pm;
    }
  };

  unique_fd video0_fd;
  unique_fd cam_sync_fd;
  unique_fd isp_fd;
  int device_iommu = -1;
  int cdm_iommu = -1;

  CameraState road_cam;
  CameraState wide_road_cam;
  CameraState driver_cam;

  PubMaster *pm = nullptr;
};
