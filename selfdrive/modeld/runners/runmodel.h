#pragma once
#include "common/clutil.h"

struct ModelInput {
  const char* name;
  int size;
  float *buffer;

  ModelInput(const char *_name, int _size, float *_buffer) : name(_name), size(_size), buffer(_buffer) {}
};

class RunModel {
public:
  virtual ~RunModel() {}
  virtual void addInput(const char *name, int size, float *buffer) {}
  virtual void execute() {}
};
