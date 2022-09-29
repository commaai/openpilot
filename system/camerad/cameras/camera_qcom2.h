#pragma once

#include <cstdint>
#include <map>
#include <utility>

#include <media/cam_req_mgr.h>

#include "system/camerad/cameras/camera.h"
#include "system/camerad/cameras/camera_common.h"
#include "system/camerad/cameras/camera_util.h"
#include "common/params.h"
#include "common/util.h"

#define FRAME_BUF_COUNT 4

class CameraState {
public:
  void handle_camera_event(void *evdat);
  void set_camera_exposure(float grey_frac);
  void camera_open(MultiCameraState *multi_cam_state, int camera_num, bool enabled);
  void camera_init(MultiCameraState *s, VisionIpcServer *v, unsigned int fps, cl_device_id device_id, cl_context ctx, VisionStreamType yuv_type);
  void camera_close();

  void camera_set_parameters();
  void camera_map_bufs(MultiCameraState *s);

  MultiCameraState *multi_cam_state;
  std::unique_ptr<AbstractCamera> camera;

  std::mutex exp_lock;

  int exposure_time;
  bool dc_gain_enabled;
  int dc_gain_weight;
  int gain_idx;
  float analog_gain_frac;

  float cur_ev[3];

  float measured_grey_fraction;
  float target_grey_fraction;

  int buf_handle[FRAME_BUF_COUNT];
  int sync_objs[FRAME_BUF_COUNT];
  int request_ids[FRAME_BUF_COUNT];
  int request_id_last;
  int frame_id_last;
  int idx_offset;
  bool skipped;

  CameraBuf buf;

private:
  void enqueue_req_multi(int start, int n, bool dp);
  void enqueue_buffer(int i, bool dp);
  int clear_req_queue();
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
