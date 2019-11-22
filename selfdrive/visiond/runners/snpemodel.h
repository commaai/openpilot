#ifndef SNPEMODEL_H
#define SNPEMODEL_H

#include <SNPE/SNPE.hpp>
#include <SNPE/SNPEBuilder.hpp>
#include <SNPE/SNPEFactory.hpp>
#include <DlContainer/IDlContainer.hpp>
#include <DlSystem/DlError.hpp>
#include <DlSystem/ITensor.hpp>
#include <DlSystem/ITensorFactory.hpp>
#include <DlSystem/IUserBuffer.hpp>
#include <DlSystem/IUserBufferFactory.hpp>

#include "runmodel.h"

#define USE_CPU_RUNTIME 0
#define USE_GPU_RUNTIME 1
#define USE_DSP_RUNTIME 2

class SNPEModel : public RunModel {
public:
  SNPEModel(const char *path, float *output, size_t output_size, int runtime);
  ~SNPEModel() {
    if (model_data) free(model_data);
  }
  void addRecurrent(float *state, int state_size);
  void addDesire(float *state, int state_size);
  void execute(float *net_input_buf);
private:
  uint8_t *model_data = NULL;

  // snpe model stuff
  std::unique_ptr<zdl::SNPE::SNPE> snpe;

  // snpe input stuff
  zdl::DlSystem::UserBufferMap inputMap;
  std::unique_ptr<zdl::DlSystem::IUserBuffer> inputBuffer;

  // snpe output stuff
  zdl::DlSystem::UserBufferMap outputMap;
  std::unique_ptr<zdl::DlSystem::IUserBuffer> outputBuffer;
  float *output;

  // recurrent and desire
  std::unique_ptr<zdl::DlSystem::IUserBuffer> addExtra(float *state, int state_size, int idx);
  std::unique_ptr<zdl::DlSystem::IUserBuffer> recurrentBuffer;
  std::unique_ptr<zdl::DlSystem::IUserBuffer> desireBuffer;
};

#endif

