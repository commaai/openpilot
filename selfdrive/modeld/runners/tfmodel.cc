#include "tfmodel.h"
#include <stdio.h>
#include <string>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdexcept>
#include "common/util.h"
#include "common/utilpp.h"
#include "common/swaglog.h"
#include <cassert>


TFModel::TFModel(const char *path, float *_output, size_t _output_size, int runtime) {
  output = _output;
  output_size = _output_size;

  char tmp[1024];
  strncpy(tmp, path, sizeof(tmp));
  strstr(tmp, ".dlc")[0] = '\0';
  strcat(tmp, ".keras");
  LOGD("loading model %s", tmp);

  assert(pipe(pipein) == 0);
  assert(pipe(pipeout) == 0);

  std::string exe_dir = util::dir_name(util::readlink("/proc/self/exe"));
  std::string keras_runner = exe_dir + "/runners/keras_runner.py";

  proc_pid = fork();
  if (proc_pid == 0) {
    LOGD("spawning keras process %s", keras_runner.c_str());
    char *argv[] = {(char*)keras_runner.c_str(), tmp, NULL};
    dup2(pipein[0], 0);
    dup2(pipeout[1], 1);
    close(pipein[0]);
    close(pipein[1]);
    close(pipeout[0]);
    close(pipeout[1]);
    execvp(keras_runner.c_str(), argv);
  }

  // parent
  close(pipein[0]);
  close(pipeout[1]);
}

TFModel::~TFModel() {
  close(pipein[1]);
  close(pipeout[0]);
  kill(proc_pid, SIGTERM);
}

void TFModel::pwrite(float *buf, int size) {
  char *cbuf = (char *)buf;
  int tw = size*sizeof(float);
  while (tw > 0) {
    int err = write(pipein[1], cbuf, tw);
    //printf("host write %d\n", err);
    assert(err >= 0);
    cbuf += err;
    tw -= err;
  }
  //printf("host write done\n");
}

void TFModel::pread(float *buf, int size) {
  char *cbuf = (char *)buf;
  int tr = size*sizeof(float);
  while (tr > 0) {
    int err = read(pipeout[0], cbuf, tr);
    //printf("host read %d/%d\n", err, tr);
    assert(err >= 0);
    cbuf += err;
    tr -= err;
  }
  //printf("host read done\n");
}

void TFModel::addRecurrent(float *state, int state_size) {
  rnn_input_buf = state;
  rnn_state_size = state_size;
}

void TFModel::addDesire(float *state, int state_size) {
  desire_input_buf = state;
  desire_state_size = state_size;
}

void TFModel::addTrafficConvention(float *state, int state_size) {
  traffic_convention_input_buf = state;
  traffic_convention_size = state_size;
}

void TFModel::execute(float *net_input_buf, int buf_size) {
  // order must be this
  pwrite(net_input_buf, buf_size);
  if (desire_input_buf != NULL) {
    pwrite(desire_input_buf, desire_state_size);
  }
  if (rnn_input_buf != NULL) {
    pwrite(rnn_input_buf, rnn_state_size);
  }
  pread(output, output_size);
}

