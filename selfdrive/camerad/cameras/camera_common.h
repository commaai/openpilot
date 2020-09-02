#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "common/buffering.h"
#include "common/mat.h"
#include "common/visionbuf.h"
#include "messaging.hpp"
#include "transforms/rgb_to_yuv.h"

#include "common/visionipc.h"

#define CAMERA_ID_IMX298 0
#define CAMERA_ID_IMX179 1
#define CAMERA_ID_S5K3P8SP 2
#define CAMERA_ID_OV8865 3
#define CAMERA_ID_IMX298_FLIPPED 4
#define CAMERA_ID_OV10640 5
#define CAMERA_ID_LGC920 6
#define CAMERA_ID_LGC615 7
#define CAMERA_ID_AR0231 8
#define CAMERA_ID_MAX 9

#define LOG_CAMERA_ID_FCAMERA 0
#define LOG_CAMERA_ID_DCAMERA 1
#define LOG_CAMERA_ID_ECAMERA 2
#define LOG_CAMERA_ID_QCAMERA 3
#define LOG_CAMERA_ID_MAX 4

#define UI_BUF_COUNT 4
#define YUV_COUNT 40

typedef struct CameraInfo {
  const char *name;
  int frame_width, frame_height;
  int frame_stride;
  bool bayer;
  int bayer_flip;
  bool hdr;
} CameraInfo;

typedef struct LogCameraInfo {
  const char* filename;
  const char* frame_packet_name;
  const char* encode_idx_name;
  VisionStreamType stream_type;
  int frame_width, frame_height;
  int fps;
  int bitrate;
  bool is_h265;
  bool downscale;
  bool has_qcamera;
} LogCameraInfo;

typedef struct FrameMetadata {
  uint32_t frame_id;
  uint64_t timestamp_eof;
  unsigned int frame_length;
  unsigned int integ_lines;
  unsigned int global_gain;
  unsigned int lens_pos;
  float lens_sag;
  float lens_err;
  float lens_true_pos;
  float gain_frac;
} FrameMetadata;

extern CameraInfo cameras_supported[CAMERA_ID_MAX];

typedef struct {
  uint8_t *y, *u, *v;
} YUVBuf;

struct CameraState;
class CameraBuf {
  public:
  cl_kernel krnl_debayer;
  cl_command_queue q;

  TBuffer ui_tb;
  TBuffer *yuv_tb;

  Pool yuv_pool;
  VisionBuf yuv_ion[YUV_COUNT];
  YUVBuf yuv_bufs[YUV_COUNT];
  FrameMetadata yuv_metas[YUV_COUNT];
  size_t yuv_buf_size;
  int yuv_width, yuv_height;
  RGBToYUVState rgb_to_yuv_state;

  int rgb_width, rgb_height, rgb_stride;
  VisionBuf rgb_bufs[UI_BUF_COUNT];

  VisionBuf *camera_bufs;

  VisionBuf *cur_rgb_buf;
  YUVBuf *cur_yuv_buf;
  VisionBuf *cur_yuv_ion_buf;

  mat3 yuv_transform;

  void init(cl_device_id device_id, cl_context context, CameraState *s, const char *name);
  void free();
  bool acquire(CameraState *s);
  void release();
  void stop();
  const FrameMetadata &frameMetaData() const { return yuv_metas[cur_yuv_idx]; }

  private:
  int cur_yuv_idx, cur_rgb_idx;
};

struct MultiCameraState;
void fill_frame_data(cereal::FrameData::Builder &framed, const FrameMetadata &frame_data, uint32_t cnt);
void common_camera_process_buf(MultiCameraState *s, const CameraBuf *b, int cnt, PubMaster *pm);
