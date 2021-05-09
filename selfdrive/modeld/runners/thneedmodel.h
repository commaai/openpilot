#pragma once

#include "selfdrive/modeld/runners/runmodel.h"
#include "selfdrive/modeld/thneed/thneed.h"

class ThneedModel : public RunModel {
public:
  ThneedModel(const char *path, float *loutput, size_t loutput_size, int runtime);
  void addRecurrent(float *state, int state_size);
  void addTrafficConvention(float *state, int state_size);
  void addDesire(float *state, int state_size);
  void execute(float *net_input_buf, int buf_size);
private:
  Thneed *thneed = NULL;
  bool recorded;

  float *output;

  // recurrent and desire
  float *recurrent;
  float *trafficConvention;
  float *desire;
};

