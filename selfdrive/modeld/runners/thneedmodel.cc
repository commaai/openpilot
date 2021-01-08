#include "thneedmodel.h"

ThneedModel::ThneedModel(const char *path, float *loutput, size_t loutput_size, int runtime) {
  thneed = new Thneed(true);
  thneed->load(path);

  output = loutput;
  output_size = loutput_size;

  assert(runtime==USE_GPU_RUNTIME);
}

void ThneedModel::addRecurrent(float *state, int state_size) {
  recurrent = state;
  recurrent_size = state_size;
}

void ThneedModel::addTrafficConvention(float *state, int state_size) {
  trafficConvention = state;
}

void ThneedModel::addDesire(float *state, int state_size) {
  desire = state;
}

void ThneedModel::execute(float *net_input_buf, int buf_size) {
  float *inputs[4] = {recurrent, trafficConvention, desire, net_input_buf};
  thneed->execute(inputs, output);
}

