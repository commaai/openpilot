#include <string.h>
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef QCOM
  #include <eigen3/Eigen/Dense>
#else
  #include <Eigen/Dense>
#endif

#include "common/timing.h"
#include "driving.h"

#ifdef MEDMODEL
  #define MODEL_WIDTH 512
  #define MODEL_HEIGHT 256
  #define MODEL_NAME "driving_model_dlc"
#else
  #define MODEL_WIDTH 320
  #define MODEL_HEIGHT 160
  #define MODEL_NAME "driving_model_dlc"
#endif

#define OUTPUT_SIZE (200 + 2*201 + 26)
#define LEAD_MDN_N 5

#ifdef TEMPORAL
  #define TEMPORAL_SIZE 512
#else
  #define TEMPORAL_SIZE 0
#endif

Eigen::Matrix<float, MODEL_PATH_DISTANCE, POLYFIT_DEGREE> vander;

void model_init(ModelState* s, cl_device_id device_id, cl_context context, int temporal) {
  model_input_init(&s->in, MODEL_WIDTH, MODEL_HEIGHT, device_id, context);
  const int output_size = OUTPUT_SIZE + TEMPORAL_SIZE;
  s->output = (float*)malloc(output_size * sizeof(float));
  memset(s->output, 0, output_size * sizeof(float));
  s->m = new DefaultRunModel("../../models/driving_model.dlc", s->output, output_size);
#ifdef TEMPORAL
  assert(temporal);
  s->m->addRecurrent(&s->output[OUTPUT_SIZE], TEMPORAL_SIZE);
#endif

  // Build Vandermonde matrix
  for(int i = 0; i < MODEL_PATH_DISTANCE; i++) {
    for(int j = 0; j < POLYFIT_DEGREE; j++) {
      vander(i, j) = pow(i, POLYFIT_DEGREE-j-1);
    }
  }
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

  //printf("readinggggg \n");
  //FILE *f = fopen("goof_frame", "r");
  //fread(net_input_buf, sizeof(float), MODEL_HEIGHT*MODEL_WIDTH*3/2, f);
  //fclose(f);
  //sleep(1);
  //printf("done \n");
  s->m->execute(net_input_buf);

  // net outputs
  net_outputs.path = &s->output[0];
  net_outputs.left_lane = &s->output[MODEL_PATH_DISTANCE*2];
  net_outputs.right_lane = &s->output[MODEL_PATH_DISTANCE*2 + MODEL_PATH_DISTANCE*2 + 1];
  net_outputs.lead = &s->output[MODEL_PATH_DISTANCE*2 + (MODEL_PATH_DISTANCE*2 + 1)*2];

  ModelData model = {0};

  for (int i=0; i<MODEL_PATH_DISTANCE; i++) {
    model.path.points[i] = net_outputs.path[i];
    model.left_lane.points[i] = net_outputs.left_lane[i] + 1.8;
    model.right_lane.points[i] = net_outputs.right_lane[i] - 1.8;
    model.path.stds[i] = softplus(net_outputs.path[MODEL_PATH_DISTANCE + i]);
    model.left_lane.stds[i] = softplus(net_outputs.left_lane[MODEL_PATH_DISTANCE + i]);
    model.right_lane.stds[i] = softplus(net_outputs.right_lane[MODEL_PATH_DISTANCE + i]);
  }

  model.path.std = softplus(net_outputs.path[MODEL_PATH_DISTANCE + MODEL_PATH_DISTANCE/2]);
  model.left_lane.std = softplus(net_outputs.left_lane[MODEL_PATH_DISTANCE + MODEL_PATH_DISTANCE/2]);
  model.right_lane.std = softplus(net_outputs.right_lane[MODEL_PATH_DISTANCE + MODEL_PATH_DISTANCE/2]);

  model.path.prob = 1.;
  model.left_lane.prob = sigmoid(net_outputs.left_lane[MODEL_PATH_DISTANCE*2]);
  model.right_lane.prob = sigmoid(net_outputs.right_lane[MODEL_PATH_DISTANCE*2]);

  poly_fit(model.path.points, model.path.stds, model.path.poly);
  poly_fit(model.left_lane.points, model.left_lane.stds, model.left_lane.poly);
  poly_fit(model.right_lane.points, model.right_lane.stds, model.right_lane.poly);

  const double max_dist = 140.0;
  const double max_rel_vel = 10.0;
  int mdn_max_idx = 0;
  for (int i=1; i<LEAD_MDN_N; i++) {
    if (net_outputs.lead[i*5 + 4] > net_outputs.lead[mdn_max_idx*5 + 4]) {
      mdn_max_idx = i;
    }
  }
  model.lead.prob = sigmoid(net_outputs.lead[LEAD_MDN_N*5]);
  model.lead.dist = net_outputs.lead[mdn_max_idx*5] * max_dist;
  model.lead.std = softplus(net_outputs.lead[mdn_max_idx*5 + 2]) * max_dist;
  model.lead.rel_v = net_outputs.lead[mdn_max_idx*5 + 1] * max_rel_vel;
  model.lead.rel_v_std = softplus(net_outputs.lead[mdn_max_idx*5 + 3]) * max_rel_vel;
  return model;
}

void model_free(ModelState* s) {
  free(s->output);
  model_input_free(&s->in);
  delete s->m;
}

void poly_fit(float *in_pts, float *in_stds, float *out) {
  // References to inputs
  Eigen::Map<Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> > pts(in_pts, MODEL_PATH_DISTANCE);
  Eigen::Map<Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> > std(in_stds, MODEL_PATH_DISTANCE);
  Eigen::Map<Eigen::Matrix<float, POLYFIT_DEGREE, 1> > p(out, POLYFIT_DEGREE);

  // Build Least Squares equations
  Eigen::Matrix<float, MODEL_PATH_DISTANCE, POLYFIT_DEGREE> lhs = vander.array().colwise() / std.array();
  Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> rhs = pts.array() / std.array();

  // Solve inplace
  Eigen::ColPivHouseholderQR<Eigen::Ref<Eigen::MatrixXf> > qr(lhs);
  p = qr.solve(rhs);
}
