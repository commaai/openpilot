#pragma once

#include <string>
#include <vector>

#include "selfdrive/modeld/runners/runmodel.h"
#include "selfdrive/modeld/thneed/thneed.h"

class ThneedModel : public RunModel {
public:
  ThneedModel(const std::string path, float *_output, size_t _output_size, int runtime, bool use_tf8 = false, cl_context context = NULL);
  void addInput(const std::string name, float *buffer, int size);
  void setInputBuffer(const std::string name, float *buffer, int size);
  void *getCLBuffer(const std::string name);
  void execute();
private:
  Thneed *thneed = NULL;
  bool recorded;

  std::vector<ModelInput> inputs;
  float *output;
};
