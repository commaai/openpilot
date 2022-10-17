#pragma once

#include "cereal/messaging/messaging.h"
#include "common/util.h"
#include "selfdrive/modeld/models/commonmodel.h"
#include "selfdrive/modeld/runners/run.h"

constexpr int FEATURE_LEN = 64;
constexpr int INPUT_SIZE = 256*256;
constexpr int OUTPUT_SIZE = FEATURE_LEN;

typedef struct NavModelResult {
  std::array<float, FEATURE_LEN> features;
  float dsp_execution_time;
} NavModelResult;

typedef struct NavModelState {
  RunModel *m;
  float net_input_buf[INPUT_SIZE];  // TODO: make this uint8_t
  float output[OUTPUT_SIZE];
} NavModelState;

void navmodel_init(NavModelState* s);
NavModelResult navmodel_eval_frame(NavModelState* s, VisionBuf* buf);
void navmodel_publish(PubMaster &pm, uint32_t frame_id, const NavModelResult &model_res, float execution_time);
void navmodel_free(NavModelState* s);
