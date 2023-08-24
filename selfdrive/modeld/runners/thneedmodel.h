#pragma once

#include <string>

#include "selfdrive/modeld/runners/runmodel.h"
#include "selfdrive/modeld/thneed/thneed.h"

class ThneedModel : public RunModel {
public:
  ThneedModel(const std::string path, float *_output, size_t _output_size, int runtime, bool use_tf8 = false, cl_context context = NULL);
  void *getCLBuffer(const std::string name);
  void execute();
private:
  Thneed *thneed = NULL;
  bool recorded;
  float *output;
};
