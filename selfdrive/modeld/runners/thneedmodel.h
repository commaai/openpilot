#ifndef THNEEDMODEL_H
#define THNEEDMODEL_H

#include "runmodel.h"

#define USE_CPU_RUNTIME 0
#define USE_GPU_RUNTIME 1
#define USE_DSP_RUNTIME 2

#include "thneed/thneed.h"

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

#endif

