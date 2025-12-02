#pragma once

#include <memory>

#include "cereal/messaging/messaging.h"
#include "msgq/visionipc/visionipc_server.h"
#include "common/util.h"


const int VIPC_BUFFER_COUNT = 18;

typedef struct FrameMetadata {
  uint32_t frame_id;
  uint32_t request_id;
  uint64_t timestamp_sof;
  uint64_t timestamp_eof;
  float processing_time;
} FrameMetadata;

class SpectraCamera;

class CameraBuf {
private:
  int frame_buf_count;

public:
  VisionIpcServer *vipc_server;
  VisionStreamType stream_type;

  int cur_buf_idx;
  FrameMetadata cur_frame_data;
  VisionBuf *cur_yuv_buf;
  VisionBuf *cur_camera_buf;
  std::unique_ptr<VisionBuf[]> camera_bufs_raw;
  uint32_t out_img_width, out_img_height;

  CameraBuf() = default;
  ~CameraBuf();
  void init(cl_device_id device_id, cl_context context, SpectraCamera *cam, VisionIpcServer * v, int frame_cnt, VisionStreamType type);
  void sendFrameToVipc();
};

void camerad_thread();
kj::Array<uint8_t> get_raw_frame_image(const CameraBuf *b);
float calculate_exposure_value(const CameraBuf *b, Rect ae_xywh, int x_skip, int y_skip);
int open_v4l_by_name_and_index(const char name[], int index = 0, int flags = O_RDWR | O_NONBLOCK);
