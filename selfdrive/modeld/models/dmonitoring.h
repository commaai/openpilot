#pragma once

#include <vector>

#include "cereal/messaging/messaging.h"
#include "selfdrive/modeld/models/commonmodel.h"
#include "selfdrive/modeld/runners/run.h"

const int OUTPUT_SIZE = 39;

typedef struct DMResult {
  float face_orientation[3];
  float face_orientation_meta[3];
  float face_position[2];
  float face_position_meta[2];
  float face_prob;
  float left_eye_prob;
  float right_eye_prob;
  float left_blink_prob;
  float right_blink_prob;
  float sg_prob;
  float poor_vision;
  float partial_face;
  float distracted_pose;
  float distracted_eyes;
  float occluded_prob;
  float dsp_execution_time;
  float model_execution_time;
} DMResult;

struct Rect {
  int x, y, w, h;
};

class YUVBuf {
public:
  void init(int width, int height, bool black = false) {
    buf.resize(width * height * 3 / 2);
    y = buf.data();
    u = y + width * height;
    v = u + (width / 2) * (height / 2);
    if (black) {
      // needed on comma two to make the padded border black
      // equivalent to RGB(0,0,0) in YUV space
      memset(y, 16, width * height);
      memset(u, 128, (width / 2) * (height / 2));
      memset(v, 128, (width / 2) * (height / 2));
    }
  }
  uint8_t *y, *u, *v;
  std::vector<uint8_t> buf;
};

class DMModel {
public:
  DMModel(int width, int height);
  DMResult eval_frame(uint8_t *stream_buf, int width, int height);
  void publish(PubMaster &pm, uint32_t frame_id, const DMResult &res);

private:
  const YUVBuf &crop_yuv(uint8_t *raw, int width, int height);

  const int MODEL_WIDTH = 320;
  const int MODEL_HEIGHT = 640;

  std::unique_ptr<RunModel> m;
  bool is_rhd;
  float output[OUTPUT_SIZE];
  YUVBuf resized_buf;
  YUVBuf cropped_buf;
  YUVBuf premirror_cropped_buf;
  std::vector<float> net_input_buf;
  float tensor[UINT8_MAX + 1];
  Rect crop_rect;
};
