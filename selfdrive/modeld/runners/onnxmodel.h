#pragma once

#include "selfdrive/modeld/runners/runmodel.h"

class ONNXModel : public RunModel {
public:
  ONNXModel(const std::string path, float *output, size_t output_size, int runtime, bool _use_tf8 = false, cl_context context = NULL);
	~ONNXModel();
  void execute();
private:
  int proc_pid;
  float *output;
  size_t output_size;
  bool use_tf8;

  // pipe to communicate to onnx_runner subprocess
  void pread(float *buf, int size);
  void pwrite(float *buf, int size);
  int pipein[2];
  int pipeout[2];
};
