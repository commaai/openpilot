#include "thneedmodel.h"
#include <assert.h>

ThneedModel::ThneedModel(const char *path, float *loutput, size_t loutput_size, int runtime) {
  thneed = new Thneed(true);
  thneed->record = 0;
  thneed->load(path);
  thneed->clexec();
  thneed->record = THNEED_RECORD;
  thneed->clexec();
  thneed->stop();

  output = loutput;
  assert(runtime==USE_GPU_RUNTIME);
}

void ThneedModel::addRecurrent(float *state, int state_size) {
  recurrent = state;
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

