#include "selfdrive/modeld/runners/thneedmodel.h"

#include <cassert>

ThneedModel::ThneedModel(const char *path, float *loutput, size_t loutput_size, int runtime) {
  thneed = new Thneed(true);
  thneed->record = 0;
  thneed->load(path);
  thneed->clexec();
  thneed->find_inputs_outputs();

  recorded = false;
  output = loutput;
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
  if (!recorded) {
    thneed->record = THNEED_RECORD;
    thneed->copy_inputs(inputs);
    thneed->clexec();
    thneed->copy_output(output);
    thneed->stop();

    recorded = true;
  } else {
    thneed->execute(inputs, output);
  }
}

