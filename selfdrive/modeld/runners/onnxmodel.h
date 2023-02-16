#pragma once

#include <cstdlib>

#include "selfdrive/modeld/runners/runmodel.h"

class ONNXModel : public RunModel {
public:
  ONNXModel(const char *path, float *output, size_t output_size, int runtime, bool use_extra = false, bool _use_tf8 = false, cl_context context = NULL);
	~ONNXModel();
  void addRecurrent(float *state, int state_size);
  void addDesire(float *state, int state_size);
  void addNavFeatures(float *state, int state_size);
  void addDrivingStyle(float *state, int state_size);
  void addTrafficConvention(float *state, int state_size);
  void addCalib(float *state, int state_size);
  void addImage(float *image_buf, int buf_size);
  void addExtra(float *image_buf, int buf_size);
  void execute();
private:
  int proc_pid;

  float *output;
  size_t output_size;

  float *rnn_input_buf = NULL;
  int rnn_state_size;
  float *desire_input_buf = NULL;
  int desire_state_size;
  float *nav_features_input_buf = NULL;
  int nav_features_size;
  float *driving_style_input_buf = NULL;
  int driving_style_size;
  float *traffic_convention_input_buf = NULL;
  int traffic_convention_size;
  float *calib_input_buf = NULL;
  int calib_size;
  float *image_input_buf = NULL;
  int image_buf_size;
  bool use_tf8;
  float *extra_input_buf = NULL;
  int extra_buf_size;
  bool use_extra;

  // pipe to communicate to keras subprocess
  void pread(float *buf, int size);
  void pwrite(float *buf, int size);
  int pipein[2];
  int pipeout[2];
};

