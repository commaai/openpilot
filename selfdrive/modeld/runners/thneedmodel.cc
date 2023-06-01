#include "selfdrive/modeld/runners/thneedmodel.h"

#include <cassert>

ThneedModel::ThneedModel(const char *path, float *loutput, size_t loutput_size, int runtime, bool luse_extra, bool luse_tf8, cl_context context) {
  thneed = new Thneed(true, context);
  thneed->load(path);
  thneed->clexec();

  recorded = false;
  output = loutput;
  use_extra = luse_extra;
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

void ThneedModel::addDrivingStyle(float *state, int state_size) {
    drivingStyle = state;
}

void ThneedModel::addNavFeatures(float *state, int state_size) {
  navFeatures = state;
}

void ThneedModel::addImage(float *image_input_buf, int buf_size) {
  input = image_input_buf;
}

void ThneedModel::addExtra(float *extra_input_buf, int buf_size) {
  extra = extra_input_buf;
}

void* ThneedModel::getInputBuf() {
  if (use_extra && thneed->input_clmem.size() > 5) return &(thneed->input_clmem[5]);
  else if (!use_extra && thneed->input_clmem.size() > 4) return &(thneed->input_clmem[4]);
  else return nullptr;
}

void* ThneedModel::getExtraBuf() {
  if (thneed->input_clmem.size() > 4) return &(thneed->input_clmem[4]);
  else return nullptr;
}

void ThneedModel::execute() {
  if (!recorded) {
    thneed->record = true;
    if (use_extra) {
      float *inputs[6] = {recurrent, navFeatures, trafficConvention, desire, extra, input};
      thneed->copy_inputs(inputs);
    } else {
      float *inputs[5] = {recurrent, navFeatures, trafficConvention, desire, input};
      thneed->copy_inputs(inputs);
    }
    thneed->clexec();
    thneed->copy_output(output);
    thneed->stop();

    recorded = true;
  } else {
    if (use_extra) {
      float *inputs[6] = {recurrent, navFeatures, trafficConvention, desire, extra, input};
      thneed->execute(inputs, output);
    } else {
      float *inputs[5] = {recurrent, navFeatures, trafficConvention, desire, input};
      thneed->execute(inputs, output);
    }
  }
}
