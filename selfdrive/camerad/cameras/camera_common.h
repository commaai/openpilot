#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <thread>

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_server.h"
#include "selfdrive/camerad/transforms/rgb_to_yuv.h"
#include "selfdrive/common/mat.h"
#include "selfdrive/common/queue.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/visionimg.h"

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

#define HLC_THRESH 222
#define HLC_A 80
#define HISTO_CEIL_K 5

const bool env_send_driver = getenv("SEND_DRIVER") != NULL;
const bool env_send_road = getenv("SEND_ROAD") != NULL;
const bool env_send_wide_road = getenv("SEND_WIDE_ROAD") != NULL;

typedef void (*release_cb)(void *cookie, int buf_idx);

typedef struct CameraInfo {
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

struct MultiCameraState;
struct CameraState;

class CameraBuf {
private:
  VisionIpcServer *vipc_server;
  CameraState *camera_state;
  cl_kernel krnl_debayer;

  std::unique_ptr<Rgb2Yuv> rgb2yuv;

  VisionStreamType rgb_type, yuv_type;

  int cur_buf_idx;

  SafeQueue<int> safe_queue;

  int frame_buf_count;
  release_cb release_callback;

public:
  cl_command_queue q;
  FrameMetadata cur_frame_data;
  VisionBuf *cur_rgb_buf;
  VisionBuf *cur_yuv_buf;
  std::unique_ptr<VisionBuf[]> camera_bufs;
  std::unique_ptr<FrameMetadata[]> camera_bufs_metadata;
  int rgb_width, rgb_height, rgb_stride;

  mat3 yuv_transform;

  CameraBuf() = default;
  ~CameraBuf();
  void init(cl_device_id device_id, cl_context context, CameraState *s, VisionIpcServer * v, int frame_cnt, VisionStreamType rgb_type, VisionStreamType yuv_type, release_cb release_callback=nullptr);
  bool acquire();
  void release();
  void queue(size_t buf_idx);
};

typedef void (*process_thread_cb)(MultiCameraState *s, CameraState *c, int cnt);

void fill_frame_data(cereal::FrameData::Builder &framed, const FrameMetadata &frame_data);
kj::Array<uint8_t> get_frame_image(const CameraBuf *b);
float set_exposure_target(const CameraBuf *b, int x_start, int x_end, int x_skip, int y_start, int y_end, int y_skip, int analog_gain, bool hist_ceil, bool hl_weighted);
std::thread start_process_thread(MultiCameraState *cameras, CameraState *cs, process_thread_cb callback);
void common_process_driver_camera(SubMaster *sm, PubMaster *pm, CameraState *c, int cnt);

void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx);
void cameras_open(MultiCameraState *s);
void cameras_run(MultiCameraState *s);
void cameras_close(MultiCameraState *s);
void camera_autoexposure(CameraState *s, float grey_frac);
