#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <memory>
#include <thread>
#include "common/mat.h"
#include "common/swaglog.h"
#include "visionbuf.h"
#include "common/visionimg.h"
#include "imgproc/utils.h"
#include "messaging.hpp"
#include "transforms/rgb_to_yuv.h"

#include "visionipc.h"
#include "visionipc_server.h"

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

#define UI_BUF_COUNT 4
#define YUV_COUNT 40
#define LOG_CAMERA_ID_FCAMERA 0
#define LOG_CAMERA_ID_DCAMERA 1
#define LOG_CAMERA_ID_ECAMERA 2
#define LOG_CAMERA_ID_QCAMERA 3
#define LOG_CAMERA_ID_MAX 4

const bool env_send_front = getenv("SEND_FRONT") != NULL;
const bool env_send_rear = getenv("SEND_REAR") != NULL;
const bool env_send_wide = getenv("SEND_WIDE") != NULL;

typedef void (*release_cb)(void *cookie, int buf_idx);

typedef struct CameraInfo {
  const char* name;
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
  uint64_t timestamp_sof; // only set on tici
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

typedef struct CameraExpInfo {
  int op_id;
  float grey_frac;
} CameraExpInfo;

extern CameraInfo cameras_supported[CAMERA_ID_MAX];

struct MultiCameraState;
struct CameraState;

class CameraBuf {
public:
  VisionIpcServer *vipc_server;
  CameraState *camera_state;
  cl_kernel krnl_debayer;
  cl_command_queue q;

  RGBToYUVState rgb_to_yuv_state;

  mat3 yuv_transform;

  FrameMetadata yuv_metas[YUV_COUNT];

  VisionStreamType rgb_type, yuv_type;
  int rgb_width, rgb_height, rgb_stride;

  FrameMetadata cur_frame_data;

  VisionBuf *cur_rgb_buf;
  VisionBuf *cur_yuv_buf;
  int cur_buf_idx;

  std::mutex frame_queue_mutex;
  std::condition_variable frame_queue_cv;
  std::queue<size_t> frame_queue;

  std::unique_ptr<VisionBuf[]> camera_bufs;
  std::unique_ptr<FrameMetadata[]> camera_bufs_metadata;

  int frame_buf_count;

  release_cb release_callback;

  CameraBuf() = default;
  ~CameraBuf();
  void init(cl_device_id device_id, cl_context context, CameraState *s, VisionIpcServer * v, int frame_cnt, VisionStreamType rgb_type, VisionStreamType yuv_type, release_cb release_callback=nullptr);
  bool acquire();
  void release();
  void queue(size_t buf_idx);
};

typedef void (*process_thread_cb)(MultiCameraState *s, CameraState *c, int cnt);

void fill_frame_data(cereal::FrameData::Builder &framed, const FrameMetadata &frame_data, uint32_t cnt);
void fill_frame_image(cereal::FrameData::Builder &framed, const CameraBuf *b);
void set_exposure_target(CameraState *c, const uint8_t *pix_ptr, int x_start, int x_end, int x_skip, int y_start, int y_end, int y_skip);
std::thread start_process_thread(MultiCameraState *cameras, const char *tname,
                                    CameraState *cs, process_thread_cb callback);
void common_camera_process_front(SubMaster *sm, PubMaster *pm, CameraState *c, int cnt);
