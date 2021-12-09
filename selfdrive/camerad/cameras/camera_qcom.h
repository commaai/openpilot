#pragma once

#include <atomic>

#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/camerad/imgproc/utils.h"
#include "selfdrive/camerad/include/msm_cam_sensor.h"
#include "selfdrive/camerad/include/msmb_camera.h"
#include "selfdrive/camerad/include/msmb_isp.h"
#include "selfdrive/camerad/include/msmb_ispif.h"

#define FRAME_BUF_COUNT 4
#define METADATA_BUF_COUNT 4

#define NUM_FOCUS 8

#define LP3_AF_DAC_DOWN 366
#define LP3_AF_DAC_UP 634
#define LP3_AF_DAC_M 440
#define LP3_AF_DAC_3SIG 52

#define FOCUS_RECOVER_PATIENCE 50 // 2.5 seconds of complete blur
#define FOCUS_RECOVER_STEPS 240 // 6 seconds

typedef struct CameraState CameraState;

typedef int (*camera_apply_exposure_func)(CameraState *s, int gain, int integ_lines, uint32_t frame_length);

typedef struct StreamState {
  struct msm_isp_buf_request buf_request;
  struct msm_vfe_axi_stream_request_cmd stream_req;
  struct msm_isp_qbuf_info qbuf_info[FRAME_BUF_COUNT];
  VisionBuf *bufs;
} StreamState;

typedef struct CameraState {
  int camera_num;
  int camera_id;

  int fps;
  CameraInfo ci;

  unique_fd csid_fd;
  unique_fd csiphy_fd;
  unique_fd sensor_fd;
  unique_fd isp_fd;

  struct msm_vfe_axi_stream_cfg_cmd stream_cfg;

  StreamState ss[3];
  CameraBuf buf;

  std::mutex frame_info_lock;
  FrameMetadata frame_metadata[METADATA_BUF_COUNT];
  int frame_metadata_idx;

  // exposure
  uint32_t pixel_clock, line_length_pclk;
  uint32_t frame_length;
  unsigned int max_gain;
  float cur_exposure_frac, cur_gain_frac;
  int cur_gain, cur_integ_lines;

  float measured_grey_fraction;
  float target_grey_fraction;

  std::atomic<float> digital_gain;
  camera_apply_exposure_func apply_exposure;

  // rear camera only,used for focusing
  unique_fd actuator_fd;
  std::atomic<float> focus_err;
  std::atomic<float> last_sag_acc_z;
  std::atomic<float> lens_true_pos;
  std::atomic<int> self_recover; // af recovery counter, neg is patience, pos is active
  uint16_t cur_step_pos;
  uint16_t cur_lens_pos;
  int16_t focus[NUM_FOCUS];
  uint8_t confidence[NUM_FOCUS];
} CameraState;


struct MultiCameraState : public CameraServerBase {
  unique_fd ispif_fd;
  unique_fd msmcfg_fd;
  unique_fd v4l_fd;
  uint16_t lapres[(ROI_X_MAX-ROI_X_MIN+1)*(ROI_Y_MAX-ROI_Y_MIN+1)];

  VisionBuf focus_bufs[FRAME_BUF_COUNT];
  VisionBuf stats_bufs[FRAME_BUF_COUNT];

  CameraState road_cam;
  CameraState driver_cam;

  LapConv *lap_conv;
};

void actuator_move(CameraState *s, uint16_t target);
int sensor_write_regs(CameraState *s, struct msm_camera_i2c_reg_array* arr, size_t size, int data_type);
