#ifndef CAMERA_H
#define CAMERA_H

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#include "common/mat.h"
#include "common/visionbuf.h"
#include "common/buffering.h"

#include "camera_common.h"

#include "media/cam_req_mgr.h"

#define FRAME_BUF_COUNT 4
#define METADATA_BUF_COUNT 4

#ifdef __cplusplus
extern "C" {
#endif


typedef struct CameraState {
  CameraInfo ci;
  FrameMetadata camera_bufs_metadata[FRAME_BUF_COUNT];
  TBuffer camera_tb;

  int frame_size;
  float digital_gain;
  mat3 transform;

  int device_iommu;
  int cdm_iommu;

  int video0_fd;
  int video1_fd;
  int isp_fd;

  int sensor_fd;
  int csiphy_fd;

  int camera_num;

  VisionBuf *bufs;

  uint32_t session_handle;

  uint32_t sensor_dev_handle;
  uint32_t isp_dev_handle;
  uint32_t csiphy_dev_handle;

  int32_t link_handle;

  int buf0_handle;
  int buf_handle[FRAME_BUF_COUNT];
  int sched_request_id;
  int request_ids[FRAME_BUF_COUNT];
  int sync_objs[FRAME_BUF_COUNT];

  struct cam_req_mgr_session_info req_mgr_session_info;
} CameraState;

typedef struct DualCameraState {
  int device;

  int video0_fd;
  int video1_fd;
  int isp_fd;

  CameraState rear;
  CameraState front;
  CameraState wide;
} DualCameraState;

void cameras_init(DualCameraState *s);
void cameras_open(DualCameraState *s, VisionBuf *camera_bufs_rear, VisionBuf *camera_bufs_focus, VisionBuf *camera_bufs_stats, VisionBuf *camera_bufs_front);
void cameras_run(DualCameraState *s);
void camera_autoexposure(CameraState *s, float grey_frac);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif

