#pragma once

#include <pthread.h>

#include <cstdint>

#include <media/cam_req_mgr.h>

#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/common/util.h"

#define FRAME_BUF_COUNT 4

#define ANALOG_GAIN_MAX_IDX 10 // 0xF is bypass
#define EXPOSURE_TIME_MIN 2 // with HDR, fastest ss
#define EXPOSURE_TIME_MAX 1904 // with HDR, slowest ss

#define EF_LOWPASS_K 0.35

#define DEBAYER_LOCAL_WORKSIZE 16

typedef struct CameraState {
  MultiCameraState *multi_cam_state;
  CameraInfo ci;

  std::mutex exp_lock;
  float analog_gain_frac;
  uint16_t analog_gain;
  bool dc_gain_enabled;
  int exposure_time;
  int exposure_time_min;
  int exposure_time_max;
  float ef_filtered;

  unique_fd sensor_fd;
  unique_fd csiphy_fd;

  int camera_num;

  uint32_t session_handle;

  uint32_t sensor_dev_handle;
  uint32_t isp_dev_handle;
  uint32_t csiphy_dev_handle;

  int32_t link_handle;

  int buf0_handle;
  int buf_handle[FRAME_BUF_COUNT];
  int sync_objs[FRAME_BUF_COUNT];
  int request_ids[FRAME_BUF_COUNT];
  int request_id_last;
  int frame_id_last;
  int idx_offset;
  bool skipped;

  struct cam_req_mgr_session_info req_mgr_session_info;

  CameraBuf buf;
} CameraState;

typedef struct MultiCameraState {
  int device;

  unique_fd video0_fd;
  unique_fd video1_fd;
  unique_fd isp_fd;
  int device_iommu;
  int cdm_iommu;


  CameraState road_cam;
  CameraState wide_road_cam;
  CameraState driver_cam;

  pthread_mutex_t isp_lock;

  SubMaster *sm;
  PubMaster *pm;
} MultiCameraState;
