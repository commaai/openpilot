#pragma once

#include <fcntl.h>
#include <memory>
#include <thread>

#include "cereal/messaging/messaging.h"
#include "msgq/visionipc/visionipc_server.h"
#include "common/queue.h"
#include "common/util.h"

const int YUV_BUFFER_COUNT = 20;

enum CameraType {
  RoadCam = 0,
  DriverCam,
  WideRoadCam
};

// for debugging
const bool env_disable_road = getenv("DISABLE_ROAD") != NULL;
const bool env_disable_wide_road = getenv("DISABLE_WIDE_ROAD") != NULL;
const bool env_disable_driver = getenv("DISABLE_DRIVER") != NULL;
const bool env_debug_frames = getenv("DEBUG_FRAMES") != NULL;
const bool env_log_raw_frames = getenv("LOG_RAW_FRAMES") != NULL;
const bool env_ctrl_exp_from_params = getenv("CTRL_EXP_FROM_PARAMS") != NULL;

typedef struct FrameMetadata {
  uint32_t frame_id;
  uint32_t request_id;

  // Timestamps
  uint64_t timestamp_sof;
  uint64_t timestamp_eof;

  // Exposure
  unsigned int integ_lines;
  bool high_conversion_gain;
  float gain;
  float measured_grey_fraction;
  float target_grey_fraction;

  float processing_time;
} FrameMetadata;

struct MultiCameraState;
class CameraState;
class ImgProc;

class CameraBuf {
private:
  VisionIpcServer *vipc_server;
  ImgProc *imgproc = nullptr;
  VisionStreamType stream_type;
  int cur_buf_idx;
  SafeQueue<int> safe_queue;
  int frame_buf_count;

public:
  cl_command_queue q;
  FrameMetadata cur_frame_data;
  VisionBuf *cur_yuv_buf;
  VisionBuf *cur_camera_buf;
  std::unique_ptr<VisionBuf[]> camera_bufs;
  std::unique_ptr<FrameMetadata[]> camera_bufs_metadata;
  int rgb_width, rgb_height;

  CameraBuf() = default;
  ~CameraBuf();
  void init(cl_device_id device_id, cl_context context, CameraState *s, VisionIpcServer * v, int frame_cnt, VisionStreamType type);
  bool acquire();
  void queue(size_t buf_idx);
};

typedef void (*process_thread_cb)(MultiCameraState *s, CameraState *c, int cnt);

void fill_frame_data(cereal::FrameData::Builder &framed, const FrameMetadata &frame_data, CameraState *c);
kj::Array<uint8_t> get_raw_frame_image(const CameraBuf *b);
float set_exposure_target(const CameraBuf *b, Rect ae_xywh, int x_skip, int y_skip);
std::thread start_process_thread(MultiCameraState *cameras, CameraState *cs, process_thread_cb callback);

void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx);
void cameras_open(MultiCameraState *s);
void cameras_run(MultiCameraState *s);
void cameras_close(MultiCameraState *s);
void camerad_thread();

int open_v4l_by_name_and_index(const char name[], int index = 0, int flags = O_RDWR | O_NONBLOCK);
