#include "common/timing.h"
#include "model.h"

#ifdef BIGMODEL
  #define MODEL_WIDTH 864
  #define MODEL_HEIGHT 288
  #define MODEL_NAME "driving_bigmodel_dlc"
#else
  #define MODEL_WIDTH 320
  #define MODEL_HEIGHT 160
  #define MODEL_NAME "driving_model_dlc"
#endif

#define OUTPUT_SIZE 161

#ifdef TEMPORAL
  #define TEMPORAL_SIZE 512
#else
  #define TEMPORAL_SIZE 0
#endif

extern const uint8_t driving_model_data[] asm("_binary_" MODEL_NAME "_start");
extern const uint8_t driving_model_end[] asm("_binary_" MODEL_NAME "_end");
const size_t driving_model_size = driving_model_end - driving_model_data;

void model_init(ModelState* s, cl_device_id device_id, cl_context context, int temporal) {
  model_input_init(&s->in, MODEL_WIDTH, MODEL_HEIGHT, device_id, context);
  const int output_size = OUTPUT_SIZE + TEMPORAL_SIZE;
  s->output = (float*)malloc(output_size * sizeof(float));
  memset(s->output, 0, output_size * sizeof(float));
  s->m = new SNPEModel(driving_model_data, driving_model_size, s->output, output_size);
#ifdef TEMPORAL
  assert(temporal);
  s->m->addRecurrent(&s->output[OUTPUT_SIZE], TEMPORAL_SIZE);
#else
  assert(!temporal);
#endif
}

ModelData model_eval_frame(ModelState* s, cl_command_queue q,
                           cl_mem yuv_cl, int width, int height,
                           mat3 transform, void* sock) {
  struct {
    float *path;
    float *left_lane;
    float *right_lane;
    float *lead;
  } net_outputs = {NULL};

  //for (int i = 0; i < OUTPUT_SIZE + TEMPORAL_SIZE; i++) { printf("%f ", s->output[i]); } printf("\n");

  float *net_input_buf = model_input_prepare(&s->in, q, yuv_cl, width, height, transform);
  s->m->execute(net_input_buf);

  // net outputs
  net_outputs.path = &s->output[0];
  net_outputs.left_lane = &s->output[51];
  net_outputs.right_lane = &s->output[51+53];
  net_outputs.lead = &s->output[51+53+53];

  ModelData model = {0};

  for (int i=0; i<MODEL_PATH_DISTANCE; i++) {
    model.path.points[i] = net_outputs.path[i];
    model.left_lane.points[i] = net_outputs.left_lane[i] + 1.8;
    model.right_lane.points[i] = net_outputs.right_lane[i] - 1.8;
  }

  model.path.std = sqrt(2.) / net_outputs.path[MODEL_PATH_DISTANCE];
  model.left_lane.std = sqrt(2.) / net_outputs.left_lane[MODEL_PATH_DISTANCE];
  model.right_lane.std = sqrt(2.) / net_outputs.right_lane[MODEL_PATH_DISTANCE];

  float softmax_buff[2];
  model.path.prob = 1.;
  softmax(&net_outputs.left_lane[MODEL_PATH_DISTANCE + 1], softmax_buff, 2);
  model.left_lane.prob = softmax_buff[0];

  softmax(&net_outputs.right_lane[MODEL_PATH_DISTANCE + 1], softmax_buff, 2);
  model.right_lane.prob = softmax_buff[0];

  const double max_dist = 140.0;
  model.lead.dist = net_outputs.lead[0] * max_dist;
  model.lead.dist = model.lead.dist > 0. ? model.lead.dist : 0.;

  model.lead.std =  max_dist * sqrt(2.) / net_outputs.lead[1];
  softmax(&net_outputs.lead[2], softmax_buff, 2);
  model.lead.prob = softmax_buff[0];

  return model;
}

void model_free(ModelState* s) {
  model_input_free(&s->in);
  delete s->m;
}

