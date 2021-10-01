#pragma once
class RunModel {
public:
  virtual void addRecurrent(float *state, int state_size) {}
  virtual void addDesire(float *state, int state_size) {}
  virtual void addTrafficConvention(float *state, int state_size) {}
  virtual void execute(float *net_input_buf, int buf_size) {}
  virtual void* getInputBuf() { return nullptr; }
};

