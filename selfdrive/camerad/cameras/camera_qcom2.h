#ifndef CAMERA_H
#define CAMERA_H

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <czmq.h>

#include "common/mat.h"
#include "common/visionbuf.h"
#include "common/buffering.h"

#include "camera_common.h"

#include "media/cam_req_mgr.h"

#define FRAME_BUF_COUNT 4

#define ANALOG_GAIN_MAX_IDX 15 // 0xF is bypass
#define EXPOSURE_TIME_MIN 8 // min time limited by HDR exp factor
#define EXPOSURE_TIME_MAX 1132 // with HDR, no slower than 1/25 sec (1416 lines)

#define HLC_THRESH 240
#define HLC_A 80

#define EF_LOWPASS_K 0.35

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CameraState {
  CameraInfo ci;
  FrameMetadata camera_bufs_metadata[FRAME_BUF_COUNT];
  TBuffer camera_tb;

  int frame_size;
  float analog_gain_frac;
  uint16_t analog_gain;
  bool dc_gain_enabled;
  int exposure_time;
  int exposure_time_min;
  int exposure_time_max;
  float ef_filtered;

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
  int sync_objs[FRAME_BUF_COUNT];
  int request_ids[FRAME_BUF_COUNT];
  int request_id_last;
  int frame_id_last;
  int idx_offset;
  bool skipped;

  struct cam_req_mgr_session_info req_mgr_session_info;

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
} MultiCameraState;

void cameras_init(MultiCameraState *s);
void cameras_open(MultiCameraState *s, VisionBuf *camera_bufs_rear, VisionBuf *camera_bufs_focus, VisionBuf *camera_bufs_stats, VisionBuf *camera_bufs_front, VisionBuf *camera_bufs_wide);
void cameras_run(MultiCameraState *s);
void camera_autoexposure(CameraState *s, float grey_frac);
#ifdef NOSCREEN
void sendrgb(MultiCameraState *s, uint8_t* dat, int len, uint8_t cam_id);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif

