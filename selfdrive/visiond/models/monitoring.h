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

#define OUTPUT_SIZE_DEPRECATED 8
#define OUTPUT_SIZE 33

typedef struct MonitoringResult {
  float descriptor_DEPRECATED[OUTPUT_SIZE_DEPRECATED - 1];
  float std_DEPRECATED;

  float face_orientation[3];
  float face_position[2];
  float face_prob;
  float left_eye_prob;
  float right_eye_prob;
  float left_blink_prob;
  float right_blink_prob;
} MonitoringResult;

typedef struct MonitoringState {
  ModelInput in;
  RunModel *m;
  float output[OUTPUT_SIZE];
} MonitoringState;

void monitoring_init(MonitoringState* s, cl_device_id device_id, cl_context context);
MonitoringResult monitoring_eval_frame(MonitoringState* s, cl_command_queue q, cl_mem yuv_cl, int width, int height);
void monitoring_publish(PubSocket *sock, uint32_t frame_id, const MonitoringResult res, float ir_target);
void monitoring_free(MonitoringState* s);

float ir_target_set(float *cur_front_gain, const MonitoringResult res);

#ifdef __cplusplus
}
#endif

#endif
