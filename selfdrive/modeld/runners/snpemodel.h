#pragma once
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <vector>
#include <DlContainer/IDlContainer.hpp>
#include <DlSystem/DlError.hpp>
#include <DlSystem/ITensor.hpp>
#include <DlSystem/ITensorFactory.hpp>
#include <DlSystem/IUserBuffer.hpp>
#include <DlSystem/IUserBufferFactory.hpp>
#include <SNPE/SNPE.hpp>
#include <SNPE/SNPEBuilder.hpp>
#include <SNPE/SNPEFactory.hpp>

#include "runmodel.h"

#define USE_CPU_RUNTIME 0
#define USE_GPU_RUNTIME 1
#define USE_DSP_RUNTIME 2

#ifdef USE_THNEED
#include "selfdrive/modeld/thneed/thneed.h"
#endif

struct SNPEModelInput {
  const char* name;
  int size;
  float *buffer;
  zdl::DlSystem::IUserBuffer *snpe_buffer;

  SNPEModelInput(const char *_name, int _size, float *_buffer, zdl::DlSystem::IUserBuffer *_snpe_buffer) : name(_name), size(_size), buffer(_buffer), snpe_buffer(_snpe_buffer) {}
};

class SNPEModel : public RunModel {
public:
  SNPEModel(const char *path, float *loutput, size_t loutput_size, int runtime, bool luse_extra = false, bool use_tf8 = false, cl_context context = NULL);
  void addInput(const char *name, int size, float *buffer);
  void execute();

#ifdef USE_THNEED
  std::unique_ptr<Thneed> thneed;
  bool thneed_recorded = false;
#endif

private:
  std::string model_data;

#ifdef QCOM2
  zdl::DlSystem::Runtime_t snpe_runtime;
#endif

  // snpe model stuff
  std::unique_ptr<zdl::SNPE::SNPE> snpe;
  zdl::DlSystem::UserBufferMap input_map;
  zdl::DlSystem::UserBufferMap output_map;
  std::unique_ptr<zdl::DlSystem::IUserBuffer> output_buffer;

  std::vector<SNPEModelInput> inputs;

  bool use_tf8;
  float *output;
  size_t output_size;
};
