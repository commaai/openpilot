#pragma once
#include <string>
#include "common/clutil.h"

struct ModelInput {
  const std::string name;
  float *buffer;
  int size;

  ModelInput(const std::string _name, float *_buffer, int _size) : name(_name), buffer(_buffer), size(_size) {}
};

class RunModel {
public:
  virtual ~RunModel() {}
  virtual void addInput(const std::string name, float *buffer, int size) {}
  virtual void setInputBuffer(const std::string name, float *buffer, int size) {}
  virtual void* getCLBuffer(const std::string name) { return nullptr; }
  virtual void execute() {}
};
