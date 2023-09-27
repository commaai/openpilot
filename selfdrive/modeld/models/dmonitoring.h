#pragma once

#include <vector>

#include "cereal/messaging/messaging.h"
#include "common/util.h"
#include "selfdrive/modeld/models/commonmodel.h"
#include "selfdrive/modeld/runners/run.h"

#define CALIB_LEN 3

#define OUTPUT_SIZE 84
#define REG_SCALE 0.25f

typedef struct DriverStateResult {
  float face_orientation[3];
  float face_orientation_std[3];
  float face_position[2];
  float face_position_std[2];
  float face_prob;
  float left_eye_prob;
  float right_eye_prob;
  float left_blink_prob;
  float right_blink_prob;
  float sunglasses_prob;
  float occluded_prob;
  float ready_prob[4];
  float not_ready_prob[2];
} DriverStateResult;

typedef struct DMonitoringModelResult {
  DriverStateResult driver_state_lhd;
  DriverStateResult driver_state_rhd;
  float poor_vision_prob;
  float wheel_on_right_prob;
  float dsp_execution_time;
} DMonitoringModelResult;

typedef struct DMonitoringModelState {
  RunModel *m;
  float output[OUTPUT_SIZE];
  std::vector<uint8_t> net_input_buf;
  float calib[CALIB_LEN];
} DMonitoringModelState;

void dmonitoring_init(DMonitoringModelState* s);
DMonitoringModelResult dmonitoring_eval_frame(DMonitoringModelState* s, void* stream_buf, int width, int height, int stride, int uv_offset, float *calib);
void dmonitoring_publish(PubMaster &pm, uint32_t frame_id, const DMonitoringModelResult &model_res, float execution_time, kj::ArrayPtr<const float> raw_pred);
void dmonitoring_free(DMonitoringModelState* s);

