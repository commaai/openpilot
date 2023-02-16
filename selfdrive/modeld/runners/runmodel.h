#pragma once
#include "common/clutil.h"
class RunModel {
public:
  virtual ~RunModel() {}
  virtual void addRecurrent(float *state, int state_size) {}
  virtual void addDesire(float *state, int state_size) {}
  virtual void addNavFeatures(float *state, int state_size) {}
  virtual void addDrivingStyle(float *state, int state_size) {}
  virtual void addTrafficConvention(float *state, int state_size) {}
  virtual void addCalib(float *state, int state_size) {}
  virtual void addImage(float *image_buf, int buf_size) {}
  virtual void addExtra(float *image_buf, int buf_size) {}
  virtual void execute() {}
  virtual void* getInputBuf() { return nullptr; }
  virtual void* getExtraBuf() { return nullptr; }
};

