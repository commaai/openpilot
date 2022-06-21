#pragma once

#include "selfdrive/modeld/runners/runmodel.h"
#include "selfdrive/modeld/thneed/thneed.h"

class ThneedModel : public RunModel {
public:
  ThneedModel(const char *path, float *loutput, size_t loutput_size, int runtime, bool luse_extra = false);
  void addRecurrent(float *state, int state_size);
  void addTrafficConvention(float *state, int state_size);
  void addDesire(float *state, int state_size);
  void addImage(float *image_buf, int buf_size);
  void addExtra(float *image_buf, int buf_size);
  void execute();
  void* getInputBuf();
  void* getExtraBuf();
private:
  Thneed *thneed = NULL;
  bool recorded;
  bool use_extra;

  float *input;
  float *extra;
  float *output;

  // recurrent and desire
  float *recurrent;
  float *trafficConvention;
  float *desire;
};

