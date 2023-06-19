#pragma once
#include "common/clutil.h"

struct ModelInput {
  const char* name;
  float *buffer;
  int size;

  ModelInput(const char *_name, float *_buffer, int _size) : name(_name), buffer(_buffer), size(_size) {}
};

class RunModel {
public:
  virtual ~RunModel() {}
  virtual void addInput(const char *name, float *buffer, int size) {}
  virtual void updateInput(const char *name, float *buffer, int size) {}
  virtual void* getCLBuffer(const char *name) { return nullptr; }
  virtual void execute() {}
};
