#pragma once

#include <cstdlib>

#include "selfdrive/modeld/runners/runmodel.h"

class ONNXModel : public RunModel {
public:
  ONNXModel(const char *path, float *output, size_t output_size, int runtime);
	~ONNXModel();
  void addRecurrent(float *state, int state_size);
  void addDesire(float *state, int state_size);
  void addTrafficConvention(float *state, int state_size);
  void execute(float *net_input_buf, int buf_size);
private:
  int proc_pid;

  float *output;
  size_t output_size;

  float *rnn_input_buf = NULL;
  int rnn_state_size;
  float *desire_input_buf = NULL;
  int desire_state_size;
  float *traffic_convention_input_buf = NULL;
  int traffic_convention_size;

  // pipe to communicate to keras subprocess
  void pread(float *buf, int size);
  void pwrite(float *buf, int size);
  int pipein[2];
  int pipeout[2];
};

