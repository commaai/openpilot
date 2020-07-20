#pragma clang diagnostic ignored "-Wexceptions"

#include <cassert>
#include <stdlib.h>
#include "common/util.h"
#include "snpemodel.h"

void PrintErrorStringAndExit() {
  std::cerr << zdl::DlSystem::getLastErrorString() << std::endl;
  std::exit(EXIT_FAILURE);
}

SNPEModel::SNPEModel(const char *path, float *loutput, size_t loutput_size, int runtime) {
  output = loutput;
  output_size = loutput_size;
#ifdef QCOM
  if (runtime==USE_GPU_RUNTIME) {
    Runtime = zdl::DlSystem::Runtime_t::GPU;
  } else if (runtime==USE_DSP_RUNTIME) {
    Runtime = zdl::DlSystem::Runtime_t::DSP;
  } else {
    Runtime = zdl::DlSystem::Runtime_t::CPU;
  }
  assert(zdl::SNPE::SNPEFactory::isRuntimeAvailable(Runtime));
#endif
  size_t model_size;
  model_data = (uint8_t *)read_file(path, &model_size);
  assert(model_data);

  // load model
  std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open(model_data, model_size);
  if (!container) { PrintErrorStringAndExit(); }
  printf("loaded model with size: %lu\n", model_size);

  // create model runner
  zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
  while (!snpe) {
#ifdef QCOM
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

std::unique_ptr<zdl::DlSystem::IUserBuffer> SNPEModel::addExtra(float *state, int state_size, int idx) {
  // get input and output names
  const auto &strListi_opt = snpe->getInputTensorNames();
  if (!strListi_opt) throw std::runtime_error("Error obtaining Input tensor names");
  const auto &strListi = *strListi_opt;
  const char *input_tensor_name = strListi.at(idx);
  printf("adding index %d: %s\n", idx, input_tensor_name);

  zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
  zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
  std::vector<size_t> retStrides = {state_size * sizeof(float), sizeof(float)};
  auto ret = ubFactory.createUserBuffer(state, state_size * sizeof(float), retStrides, &userBufferEncodingFloat);
  inputMap.add(input_tensor_name, ret.get());
  return ret;
}

void SNPEModel::execute(float *net_input_buf, int buf_size) {
#ifdef USE_THNEED
  if (Runtime == zdl::DlSystem::Runtime_t::GPU) {
    float *inputs[4] = {recurrent, trafficConvention, desire, net_input_buf};
    if (thneed == NULL) {
      assert(inputBuffer->setBufferAddress(net_input_buf));
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
      thneed->execute(inputs, output);

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
      thneed->execute(inputs, output);
    }
  } else {
#endif
    assert(inputBuffer->setBufferAddress(net_input_buf));
    if (!snpe->execute(inputMap, outputMap)) {
      PrintErrorStringAndExit();
    }
#ifdef USE_THNEED
  }
#endif
}

