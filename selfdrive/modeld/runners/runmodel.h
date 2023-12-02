#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cassert>

#include "common/clutil.h"
#include "common/swaglog.h"

#define USE_CPU_RUNTIME 0
#define USE_GPU_RUNTIME 1
#define USE_DSP_RUNTIME 2

struct ModelInput {
  const std::string name;
  float *buffer;
  int size;

  ModelInput(const std::string _name, float *_buffer, int _size) : name(_name), buffer(_buffer), size(_size) {}
  virtual void setBuffer(float *_buffer, int _size) {
    assert(size == _size || size == 0);
    buffer = _buffer;
    size = _size;
  }
};

class RunModel {
public:
  std::vector<std::unique_ptr<ModelInput>> inputs;

  virtual ~RunModel() {}
  virtual void execute() {}
  virtual void* getCLBuffer(const std::string name) { return nullptr; }

  virtual void addInput(const std::string name, float *buffer, int size) {
    inputs.push_back(std::unique_ptr<ModelInput>(new ModelInput(name, buffer, size)));
  }
  virtual void setInputBuffer(const std::string name, float *buffer, int size) {
    for (auto &input : inputs) {
      if (name == input->name) {
        input->setBuffer(buffer, size);
        return;
      }
    }
    LOGE("Tried to update input `%s` but no input with this name exists", name.c_str());
    assert(false);
  }
};
