#ifndef MODEL_H
#define MODEL_H

// gate this here
#define TEMPORAL
#define DESIRE

#ifdef DESIRE
  #define DESIRE_SIZE 8
#endif

#include "common/mat.h"
#include "common/modeldata.h"
#include "common/util.h"

#include "commonmodel.h"
#include "runners/run.h"

#include "cereal/gen/cpp/log.capnp.h"
#include <czmq.h>
#include <capnp/serialize.h>
#include "messaging.hpp"


typedef struct ModelState {
  ModelInput in;
  float *output;
  RunModel *m;
#ifdef DESIRE
  float *desire;
#endif
} ModelState;

void model_init(ModelState* s, cl_device_id device_id,
                cl_context context, int temporal);
ModelData model_eval_frame(ModelState* s, cl_command_queue q,
                           cl_mem yuv_cl, int width, int height,
                           mat3 transform, void* sock, float *desire_in);
void model_free(ModelState* s);
void poly_fit(float *in_pts, float *in_stds, float *out);

void model_publish(PubSocket* sock, uint32_t frame_id,
                   const ModelData data, uint64_t timestamp_eof);
#endif
