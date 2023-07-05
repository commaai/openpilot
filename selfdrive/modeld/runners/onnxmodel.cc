#include "selfdrive/modeld/runners/onnxmodel.h"

#include <poll.h>
#include <unistd.h>

#include <cassert>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/swaglog.h"
#include "common/util.h"

ONNXModel::ONNXModel(const char *path, float *_output, size_t _output_size, int runtime, bool _use_extra, bool _use_tf8, cl_context context) {
  LOGD("loading model %s", path);

  output = _output;
  output_size = _output_size;
  use_extra = _use_extra;
  use_tf8 = _use_tf8;

  int err = pipe(pipein);
  assert(err == 0);
  err = pipe(pipeout);
  assert(err == 0);

  std::string exe_dir = util::dir_name(util::readlink("/proc/self/exe"));
  std::string onnx_runner = exe_dir + "/runners/onnx_runner.py";
  std::string tf8_arg = use_tf8 ? "--use_tf8" : "";

  proc_pid = fork();
  if (proc_pid == 0) {
    LOGD("spawning onnx process %s", onnx_runner.c_str());
    char *argv[] = {(char*)onnx_runner.c_str(), (char*)path, (char*)tf8_arg.c_str(), nullptr};
    dup2(pipein[0], 0);
    dup2(pipeout[1], 1);
    close(pipein[0]);
    close(pipein[1]);
    close(pipeout[0]);
    close(pipeout[1]);
    execvp(onnx_runner.c_str(), argv);
  }

  // parent
  close(pipein[0]);
  close(pipeout[1]);
}

ONNXModel::~ONNXModel() {
  close(pipein[1]);
  close(pipeout[0]);
  kill(proc_pid, SIGTERM);
}

void ONNXModel::pwrite(float *buf, int size) {
  char *cbuf = (char *)buf;
  int tw = size*sizeof(float);
  while (tw > 0) {
    int err = write(pipein[1], cbuf, tw);
    //printf("host write %d\n", err);
    assert(err >= 0);
    cbuf += err;
    tw -= err;
  }
  LOGD("host write of size %d done", size);
}

void ONNXModel::pread(float *buf, int size) {
  char *cbuf = (char *)buf;
  int tr = size*sizeof(float);
  struct pollfd fds[1];
  fds[0].fd = pipeout[0];
  fds[0].events = POLLIN;
  while (tr > 0) {
    int err;
    err = poll(fds, 1, 10000);  // 10 second timeout
    assert(err == 1 || (err == -1 && errno == EINTR));
    LOGD("host read remaining %d/%d poll %d", tr, size*sizeof(float), err);
    err = read(pipeout[0], cbuf, tr);
    assert(err > 0 || (err == 0 && errno == EINTR));
    cbuf += err;
    tr -= err;
  }
  LOGD("host read done");
}

void ONNXModel::addRecurrent(float *state, int state_size) {
  rnn_input_buf = state;
  rnn_state_size = state_size;
}

void ONNXModel::addDesire(float *state, int state_size) {
  desire_input_buf = state;
  desire_state_size = state_size;
}

void ONNXModel::addNavFeatures(float *state, int state_size) {
  nav_features_input_buf = state;
  nav_features_size = state_size;
}

void ONNXModel::addDrivingStyle(float *state, int state_size) {
    driving_style_input_buf = state;
    driving_style_size = state_size;
}

void ONNXModel::addTrafficConvention(float *state, int state_size) {
  traffic_convention_input_buf = state;
  traffic_convention_size = state_size;
}

void ONNXModel::addCalib(float *state, int state_size) {
  calib_input_buf = state;
  calib_size = state_size;
}

void ONNXModel::addImage(float *image_buf, int buf_size) {
  image_input_buf = image_buf;
  image_buf_size = buf_size;
}

void ONNXModel::addExtra(float *image_buf, int buf_size) {
  extra_input_buf = image_buf;
  extra_buf_size = buf_size;
}

void ONNXModel::execute() {
  // order must be this
  if (image_input_buf != NULL) {
    pwrite(image_input_buf, image_buf_size);
  }
  if (extra_input_buf != NULL) {
    pwrite(extra_input_buf, extra_buf_size);
  }
  if (desire_input_buf != NULL) {
    pwrite(desire_input_buf, desire_state_size);
  }
  if (traffic_convention_input_buf != NULL) {
    pwrite(traffic_convention_input_buf, traffic_convention_size);
  }
  if (driving_style_input_buf != NULL) {
    pwrite(driving_style_input_buf, driving_style_size);
  }
  if (nav_features_input_buf != NULL) {
    pwrite(nav_features_input_buf, nav_features_size);
  }
  if (calib_input_buf != NULL) {
    pwrite(calib_input_buf, calib_size);
  }
  if (rnn_input_buf != NULL) {
    pwrite(rnn_input_buf, rnn_state_size);
  }
  pread(output, output_size);
}

