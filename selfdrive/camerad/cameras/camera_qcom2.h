#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <czmq.h>

#include "camera_common.h"
#include "media/cam_req_mgr.h"

#define FRAME_BUF_COUNT 4

typedef struct CameraState {
  CameraInfo ci;

  //float digital_gain;
  //int digital_gain_pre;
  float analog_gain_frac;
  uint16_t analog_gain;
  uint8_t dc_opstate;
  bool dc_gain_enabled;
  int exposure_time;

  mat3 transform;

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

  int debayer_cl_localMemSize;
  size_t debayer_cl_globalWorkSize[2];
  size_t debayer_cl_localWorkSize[2];

  struct cam_req_mgr_session_info req_mgr_session_info;

  CameraBuf buf;
} CameraState;

typedef struct MultiCameraState {
  int device;

  int video0_fd;
  int video1_fd;
  int isp_fd;

  CameraState rear;
  CameraState front;
  CameraState wide;
#ifdef NOSCREEN
  zsock_t *rgb_sock;
#endif

  pthread_mutex_t isp_lock;

  SubMaster *sm;
  PubMaster *pm;
} MultiCameraState;

void cameras_init(MultiCameraState *s, cl_device_id device_id, cl_context ctx);
void cameras_open(MultiCameraState *s);
void cameras_run(MultiCameraState *s);
void camera_autoexposure(CameraState *s, float grey_frac);
#ifdef NOSCREEN
void sendrgb(MultiCameraState *s, uint8_t* dat, int len, uint8_t cam_id);
#endif
