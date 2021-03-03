#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#include "camera_common.h"
#include "media/cam_req_mgr.h"

#define FRAME_BUF_COUNT 4

#define ANALOG_GAIN_MAX_IDX 10 // 0xF is bypass
#define EXPOSURE_TIME_MIN 2 // with HDR, fastest ss
#define EXPOSURE_TIME_MAX 1757 // with HDR, slowest ss

#define HLC_THRESH 222
#define HLC_A 80
#define HISTO_CEIL_K 5

#define EF_LOWPASS_K 0.35

#define DEBAYER_LOCAL_WORKSIZE 16

typedef struct CameraState {
  CameraInfo ci;

  float analog_gain_frac;
  uint16_t analog_gain;
  bool dc_gain_enabled;
  int exposure_time;
  int exposure_time_min;
  int exposure_time_max;
  float ef_filtered;

  int device_iommu;
  int cdm_iommu;

  int video0_fd;
  int video1_fd;
  int isp_fd;

  int sensor_fd;
  int csiphy_fd;

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

  int video0_fd;
  int video1_fd;
  int isp_fd;

  CameraState road_cam;
  CameraState wide_road_cam;
  CameraState driver_cam;

  pthread_mutex_t isp_lock;

  SubMaster *sm;
  PubMaster *pm;
} MultiCameraState;
