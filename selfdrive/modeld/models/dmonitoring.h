#ifndef DMONITORING_H
#define DMONITORING_H

#include "common/util.h"
#include "commonmodel.h"
#include "runners/run.h"

#include "messaging.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#define OUTPUT_SIZE 33
#define RHD_CHECK_INTERVAL 10

typedef struct DMonitoringResult {
  float face_orientation[3];
  float face_orientation_meta[3];
  float face_position[2];
  float face_position_meta[2];
  float face_prob;
  float left_eye_prob;
  float right_eye_prob;
  float left_blink_prob;
  float right_blink_prob;
} DMonitoringResult;

typedef struct DMonitoringModelState {
  RunModel *m;
  bool is_rhd;
  bool is_rhd_checked;
  float output[OUTPUT_SIZE];
} DMonitoringModelState;

void dmonitoring_init(DMonitoringModelState* s);
DMonitoringResult dmonitoring_eval_frame(DMonitoringModelState* s, void* stream_buf, int width, int height);
void dmonitoring_publish(PubMaster &pm, uint32_t frame_id, const DMonitoringResult res);
void dmonitoring_free(DMonitoringModelState* s);

#ifdef __cplusplus
}
#endif

#endif
