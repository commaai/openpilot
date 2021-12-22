#pragma once
class RunModel {
public:
  virtual ~RunModel() {}
  virtual void addRecurrent(float *state, int state_size) {}
  virtual void addDesire(float *state, int state_size) {}
  virtual void addTrafficConvention(float *state, int state_size) {}
  virtual void addImage(float *image_buf, int buf_size) {}
  virtual void execute() {}
  virtual void* getInputBuf() { return nullptr; }
};

