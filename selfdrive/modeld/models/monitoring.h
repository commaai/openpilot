#ifndef MONITORING_H
#define MONITORING_H

#include "common/util.h"
#include "commonmodel.h"
#include "runners/run.h"

#include "cereal/gen/cpp/log.capnp.h"
#include <capnp/serialize.h>
#include "messaging.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#define OUTPUT_SIZE 33

typedef struct MonitoringResult {
  float face_orientation[3];
  float face_position[2];
  float face_prob;
  float left_eye_prob;
  float right_eye_prob;
  float left_blink_prob;
  float right_blink_prob;
} MonitoringResult;

typedef struct MonitoringState {
  RunModel *m;
  float output[OUTPUT_SIZE];
} MonitoringState;

void monitoring_init(MonitoringState* s);
MonitoringResult monitoring_eval_frame(MonitoringState* s, void* stream_buf, int width, int height);
void monitoring_publish(PubSocket *sock, uint32_t frame_id, const MonitoringResult res);
void monitoring_free(MonitoringState* s);

#ifdef __cplusplus
}
#endif

#endif
