#pragma once

#include <vector>

#include "selfdrive/modeld/runners/runmodel.h"
#include "selfdrive/modeld/thneed/thneed.h"

class ThneedModel : public RunModel {
public:
  ThneedModel(const char *path, float *_output, size_t _output_size, int runtime, bool use_tf8 = false, cl_context context = NULL);
  void addInput(const char *name, float *buffer, int size);
  void updateInput(const char *name, float *buffer, int size);
  void *getCLBuffer(const char *name);
  void execute();
private:
  Thneed *thneed = NULL;
  bool recorded;

  std::vector<ModelInput> inputs;
  float *output;
};
