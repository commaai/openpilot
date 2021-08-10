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

#define LOG_CAMERA_ID_FCAMERA 0
#define LOG_CAMERA_ID_DCAMERA 1
#define LOG_CAMERA_ID_ECAMERA 2
#define LOG_CAMERA_ID_QCAMERA 3
#define LOG_CAMERA_ID_MAX 4

#define HLC_THRESH 222
#define HLC_A 80
#define HISTO_CEIL_K 5

typedef void (*release_cb)(void *cookie, int buf_idx);

enum CameraType {
  RoadCam = 0,
  DriverCam,
  WideRoadCam
};

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
  bool trigger_rotate;
} LogCameraInfo;

typedef struct FrameMetadata {
  uint32_t frame_id;
  unsigned int frame_length;

  // Timestamps
  uint64_t timestamp_sof; // only set on tici
  uint64_t timestamp_eof;

  // Exposure
  unsigned int integ_lines;
  bool high_conversion_gain;
  float gain;
  float measured_grey_fraction;
  float target_grey_fraction;

  // Focus
  unsigned int lens_pos;
  float lens_sag;
  float lens_err;
  float lens_true_pos;
} FrameMetadata;

typedef struct CameraExpInfo {
  int op_id;
  float grey_frac;
} CameraExpInfo;

class CameraServer;
struct CameraState;
typedef void (*process_thread_cb)(CameraServer *s, CameraState *c, cereal::FrameData::Builder &framed, int cnt);

class CameraServerBase {
public:
  CameraServerBase();
  virtual ~CameraServerBase();
  void start();

  cl_device_id device_id;
  cl_context context;
  VisionIpcServer *vipc_server;
  PubMaster *pm;

protected:
  virtual void run() = 0;
  void start_process_thread(CameraState *cs, process_thread_cb callback = nullptr, bool is_frame_stream = false);
  std::vector<std::thread> camera_threads;

private:
  void process_camera(CameraState *cs, process_thread_cb callback, bool is_frame_stream);
};

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
  void init(CameraServer *server, CameraState *s, int frame_cnt, release_cb release_callback=nullptr);
  bool acquire();
  void release();
  void queue(size_t buf_idx);
};

float set_exposure_target(const CameraBuf *b, int x_start, int x_end, int x_skip, int y_start, int y_end, int y_skip);
void camera_autoexposure(CameraState *s, float grey_frac);
