#ifndef MODEL_H
#define MODEL_H

// gate this here
#define TEMPORAL

#include "common/mat.h"
#include "common/modeldata.h"

#include "commonmodel.h"
#include "runners/run.h"

typedef struct ModelState {
  ModelInput in;
  float *output;
  RunModel *m;
} ModelState;

void model_init(ModelState* s, cl_device_id device_id,
                cl_context context, int temporal);
ModelData model_eval_frame(ModelState* s, cl_command_queue q,
                           cl_mem yuv_cl, int width, int height,
                           mat3 transform, void* sock);
void model_free(ModelState* s);
void poly_fit(float *in_pts, float *in_stds, float *out);

#endif
