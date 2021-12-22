#pragma once

#include "selfdrive/modeld/runners/runmodel.h"
#include "selfdrive/modeld/thneed/thneed.h"

class ThneedModel : public RunModel {
public:
  ThneedModel(const char *path, float *loutput, size_t loutput_size, int runtime);
  void addRecurrent(float *state, int state_size);
  void addTrafficConvention(float *state, int state_size);
  void addDesire(float *state, int state_size);
  void addImage(float *image_buf, int buf_size);
  void execute();
  void* getInputBuf();
private:
  Thneed *thneed = NULL;
  bool recorded;

  float *input;
  float *output;

  // recurrent and desire
  float *recurrent;
  float *trafficConvention;
  float *desire;
};

