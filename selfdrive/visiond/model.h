#ifndef MODEL_H
#define MODEL_H

// gate this here
//#define BIGMODEL
#define TEMPORAL

#include "common/mat.h"
#include "common/modeldata.h"

#include "commonmodel.h"
#include "snpemodel.h"

typedef struct ModelState {
  ModelInput in;
  float *output;
  SNPEModel *m;
} ModelState;

void model_init(ModelState* s, cl_device_id device_id,
                cl_context context, int temporal);
ModelData model_eval_frame(ModelState* s, cl_command_queue q,
                           cl_mem yuv_cl, int width, int height,
                           mat3 transform, void* sock);
void model_free(ModelState* s);

#endif
