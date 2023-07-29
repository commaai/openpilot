#include "selfdrive/modeld/runners/onnxmodel.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <poll.h>
#include <unistd.h>

#include "common/util.h"

ONNXModel::ONNXModel(const std::string path, float *_output, size_t _output_size, int runtime, bool _use_tf8, cl_context context) {
  LOGD("loading model %s", path.c_str());

  output = _output;
  output_size = _output_size;
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
    char *argv[] = {(char*)onnx_runner.c_str(), (char*)path.c_str(), (char*)tf8_arg.c_str(), nullptr};
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

void ONNXModel::execute() {
  for (auto &input : inputs) {
    pwrite(input->buffer, input->size);
  }
  pread(output, output_size);
}
