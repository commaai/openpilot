#pragma clang diagnostic ignored "-Wexceptions"

#include "selfdrive/modeld/runners/snpemodel.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

#include "selfdrive/common/util.h"
#include "selfdrive/common/timing.h"

void PrintErrorStringAndExit() {
  std::cerr << zdl::DlSystem::getLastErrorString() << std::endl;
  std::exit(EXIT_FAILURE);
}

SNPEModel::SNPEModel(const char *path, float *loutput, size_t loutput_size, int runtime, bool luse_extra) {
  output = loutput;
  output_size = loutput_size;
  use_extra = luse_extra;
#ifdef QCOM2
  if (runtime==USE_GPU_RUNTIME) {
    Runtime = zdl::DlSystem::Runtime_t::GPU;
  } else if (runtime==USE_DSP_RUNTIME) {
    Runtime = zdl::DlSystem::Runtime_t::DSP;
  } else {
    Runtime = zdl::DlSystem::Runtime_t::CPU;
  }
  assert(zdl::SNPE::SNPEFactory::isRuntimeAvailable(Runtime));
#endif
  model_data = util::read_file(path);
  assert(model_data.size() > 0);

  // load model
  std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open((uint8_t*)model_data.data(), model_data.size());
  if (!container) { PrintErrorStringAndExit(); }
  printf("loaded model with size: %lu\n", model_data.size());

  // create model runner
  zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
  while (!snpe) {
#ifdef QCOM2
    snpe = snpeBuilder.setOutputLayers({})
                      .setRuntimeProcessor(Runtime)
                      .setUseUserSuppliedBuffers(true)
                      .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
                      .build();
#else
    snpe = snpeBuilder.setOutputLayers({})
                      .setUseUserSuppliedBuffers(true)
                      .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
                      .build();
#endif
    if (!snpe) std::cerr << zdl::DlSystem::getLastErrorString() << std::endl;
  }

  // get input and output names
  const auto &strListi_opt = snpe->getInputTensorNames();
  if (!strListi_opt) throw std::runtime_error("Error obtaining Input tensor names");
  const auto &strListi = *strListi_opt;
  //assert(strListi.size() == 1);
  const char *input_tensor_name = strListi.at(0);

  const auto &strListo_opt = snpe->getOutputTensorNames();
  if (!strListo_opt) throw std::runtime_error("Error obtaining Output tensor names");
  const auto &strListo = *strListo_opt;
  assert(strListo.size() == 1);
  const char *output_tensor_name = strListo.at(0);

  printf("model: %s -> %s\n", input_tensor_name, output_tensor_name);

  zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
  zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();

  // create input buffer
  {
    const auto &inputDims_opt = snpe->getInputDimensions(input_tensor_name);
    const zdl::DlSystem::TensorShape& bufferShape = *inputDims_opt;
    std::vector<size_t> strides(bufferShape.rank());
    strides[strides.size() - 1] = sizeof(float);
    size_t product = 1;
    for (size_t i = 0; i < bufferShape.rank(); i++) product *= bufferShape[i];
    size_t stride = strides[strides.size() - 1];
    for (size_t i = bufferShape.rank() - 1; i > 0; i--) {
      stride *= bufferShape[i];
      strides[i-1] = stride;
    }
    printf("input product is %lu\n", product);
    inputBuffer = ubFactory.createUserBuffer(NULL, product*sizeof(float), strides, &userBufferEncodingFloat);

    inputMap.add(input_tensor_name, inputBuffer.get());
  }

  if (use_extra) {
    const char *extra_tensor_name = strListi.at(1);
    const auto &extraDims_opt = snpe->getInputDimensions(extra_tensor_name);
    const zdl::DlSystem::TensorShape& bufferShape = *extraDims_opt;
    std::vector<size_t> strides(bufferShape.rank());
    strides[strides.size() - 1] = sizeof(float);
    size_t product = 1;
    for (size_t i = 0; i < bufferShape.rank(); i++) product *= bufferShape[i];
    size_t stride = strides[strides.size() - 1];
    for (size_t i = bufferShape.rank() - 1; i > 0; i--) {
      stride *= bufferShape[i];
      strides[i-1] = stride;
    }
    printf("extra product is %lu\n", product);
    extraBuffer = ubFactory.createUserBuffer(NULL, product*sizeof(float), strides, &userBufferEncodingFloat);

    inputMap.add(extra_tensor_name, extraBuffer.get());
  }

  // create output buffer
  {
    const zdl::DlSystem::TensorShape& bufferShape = snpe->getInputOutputBufferAttributes(output_tensor_name)->getDims();
    if (output_size != 0) {
      assert(output_size == bufferShape[1]);
    } else {
      output_size = bufferShape[1];
    }

    std::vector<size_t> outputStrides = {output_size * sizeof(float), sizeof(float)};
    outputBuffer = ubFactory.createUserBuffer(output, output_size * sizeof(float), outputStrides, &userBufferEncodingFloat);
    outputMap.add(output_tensor_name, outputBuffer.get());
  }
}

void SNPEModel::addRecurrent(float *state, int state_size) {
  recurrent = state;
  recurrent_size = state_size;
  recurrentBuffer = this->addExtra(state, state_size, 3);
}

void SNPEModel::addTrafficConvention(float *state, int state_size) {
  trafficConvention = state;
  trafficConventionBuffer = this->addExtra(state, state_size, 2);
}

void SNPEModel::addDesire(float *state, int state_size) {
  desire = state;
  desireBuffer = this->addExtra(state, state_size, 1);
}

void SNPEModel::addCalib(float *state, int state_size) {
  calib = state;
  calibBuffer = this->addExtra(state, state_size, 1);
}

void SNPEModel::addImage(float *image_buf, int buf_size) {
  input = image_buf;
  input_size = buf_size;
}

void SNPEModel::addExtra(float *image_buf, int buf_size) {
  extra = image_buf;
  extra_size = buf_size;
}

std::unique_ptr<zdl::DlSystem::IUserBuffer> SNPEModel::addExtra(float *state, int state_size, int idx) {
  // get input and output names
  const auto real_idx = idx + (use_extra ? 1 : 0);
  const auto &strListi_opt = snpe->getInputTensorNames();
  if (!strListi_opt) throw std::runtime_error("Error obtaining Input tensor names");
  const auto &strListi = *strListi_opt;
  const char *input_tensor_name = strListi.at(real_idx);
  printf("adding index %d: %s\n", real_idx, input_tensor_name);

  zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
  zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
  std::vector<size_t> retStrides = {state_size * sizeof(float), sizeof(float)};
  auto ret = ubFactory.createUserBuffer(state, state_size * sizeof(float), retStrides, &userBufferEncodingFloat);
  inputMap.add(input_tensor_name, ret.get());
  return ret;
}

void SNPEModel::execute() {
#ifdef USE_THNEED
  if (Runtime == zdl::DlSystem::Runtime_t::GPU) {
    if (thneed == NULL) {
      bool ret = inputBuffer->setBufferAddress(input);
      assert(ret == true);
      if (use_extra) {
        assert(extra != NULL);
        bool extra_ret = extraBuffer->setBufferAddress(extra);
        assert(extra_ret == true);
      }
      if (!snpe->execute(inputMap, outputMap)) {
        PrintErrorStringAndExit();
      }
      memset(recurrent, 0, recurrent_size*sizeof(float));
      thneed = new Thneed();
      if (!snpe->execute(inputMap, outputMap)) {
        PrintErrorStringAndExit();
      }
      thneed->stop();
      printf("thneed cached\n");

      // doing self test
      float *outputs_golden = (float *)malloc(output_size*sizeof(float));
      memcpy(outputs_golden, output, output_size*sizeof(float));
      memset(output, 0, output_size*sizeof(float));
      memset(recurrent, 0, recurrent_size*sizeof(float));
      uint64_t start_time = nanos_since_boot();
      if (extra != NULL) {
        float *inputs[5] = {recurrent, trafficConvention, desire, extra, input};
        thneed->execute(inputs, output);
      } else {
        float *inputs[4] = {recurrent, trafficConvention, desire, input};
        thneed->execute(inputs, output);
      }
      uint64_t elapsed_time = nanos_since_boot() - start_time;
      printf("ran model in %.2f ms\n", float(elapsed_time)/1e6);

      if (memcmp(output, outputs_golden, output_size*sizeof(float)) == 0) {
        printf("thneed selftest passed\n");
      } else {
        for (int i = 0; i < output_size; i++) {
          printf("mismatch %3d: %f %f\n", i, output[i], outputs_golden[i]);
        }
        assert(false);
      }
      free(outputs_golden);
    } else {
      if (use_extra) {
        float *inputs[5] = {recurrent, trafficConvention, desire, extra, input};
        thneed->execute(inputs, output);
      } else {
        float *inputs[4] = {recurrent, trafficConvention, desire, input};
        thneed->execute(inputs, output);
      }
    }
  } else {
#endif
    bool ret = inputBuffer->setBufferAddress(input);
    assert(ret == true);
    if (use_extra) {
      bool extra_ret = extraBuffer->setBufferAddress(extra);
      assert(extra_ret == true);
    }
    if (!snpe->execute(inputMap, outputMap)) {
      PrintErrorStringAndExit();
    }
#ifdef USE_THNEED
  }
#endif
}

