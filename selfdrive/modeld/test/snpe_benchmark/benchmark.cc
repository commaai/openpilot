#include <SNPE/SNPE.hpp>
#include <SNPE/SNPEBuilder.hpp>
#include <SNPE/SNPEFactory.hpp>
#include <DlContainer/IDlContainer.hpp>
#include <DlSystem/DlError.hpp>
#include <DlSystem/ITensor.hpp>
#include <DlSystem/ITensorFactory.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

int64_t timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p) {
  return ((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) - ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec);
}

void PrintErrorStringAndExit() {
  cout << "ERROR!" << endl;
  const char* const errStr = zdl::DlSystem::getLastErrorString();
  std::cerr << errStr << std::endl;
  std::exit(EXIT_FAILURE);
}


zdl::DlSystem::Runtime_t checkRuntime() {
  static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();
  static zdl::DlSystem::Runtime_t Runtime;
  std::cout << "SNPE Version: " << Version.asString().c_str() << std::endl; //Print Version number
  if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP)) {
    std::cout << "Using DSP runtime" << std::endl;
    Runtime = zdl::DlSystem::Runtime_t::DSP;
  } else if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU)) {
    std::cout << "Using GPU runtime" << std::endl;
    Runtime = zdl::DlSystem::Runtime_t::GPU;
  } else {
    std::cout << "Using cpu runtime" << std::endl;
    Runtime = zdl::DlSystem::Runtime_t::CPU;
  }
  return Runtime;
}

void test(char *filename) {
  static zdl::DlSystem::Runtime_t runtime = checkRuntime();
  std::unique_ptr<zdl::DlContainer::IDlContainer> container;
  container = zdl::DlContainer::IDlContainer::open(filename);

  if (!container) { PrintErrorStringAndExit(); }
  cout << "start build" << endl;
  std::unique_ptr<zdl::SNPE::SNPE> snpe;
  {
    snpe = NULL;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    snpe = snpeBuilder.setOutputLayers({})
      .setRuntimeProcessor(runtime)
      .setUseUserSuppliedBuffers(false)
      //.setDebugMode(true)
      .build();
    if (!snpe) {
      cout << "ERROR!" << endl;
      const char* const errStr = zdl::DlSystem::getLastErrorString();
      std::cerr << errStr << std::endl;
    }
    cout << "ran snpeBuilder" << endl;
  }

  const auto &strList_opt = snpe->getInputTensorNames();
  if (!strList_opt) throw std::runtime_error("Error obtaining input tensor names");

  cout << "get input tensor names done" << endl;
  const auto &strList = *strList_opt;
  static zdl::DlSystem::TensorMap inputTensorMap;
  static zdl::DlSystem::TensorMap outputTensorMap;
  vector<std::unique_ptr<zdl::DlSystem::ITensor> > inputs;
  for (int i = 0; i < strList.size(); i++) {
    cout << "input name: " << strList.at(i) << endl;

    const auto &inputDims_opt = snpe->getInputDimensions(strList.at(i));
    const auto &inputShape = *inputDims_opt;
    inputs.push_back(zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape));
    inputTensorMap.add(strList.at(i), inputs[i].get());
  }

  struct timespec start, end;
  cout << "**** starting benchmark ****" << endl;
  for (int i = 0; i < 50; i++) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    int err = snpe->execute(inputTensorMap, outputTensorMap);
    assert(err == true);
    clock_gettime(CLOCK_MONOTONIC, &end);
    uint64_t timeElapsed = timespecDiff(&end, &start);
    printf("time: %f ms\n", timeElapsed*1.0/1e6);
  }
}

void get_testframe(int index, std::unique_ptr<zdl::DlSystem::ITensor> &input) {
  FILE * pFile;
  string filepath="/data/ipt/quantize_samples/sample_input_"+std::to_string(index);
  pFile = fopen(filepath.c_str(),"rb");
  int length = 1*6*160*320*4;
  float * frame_buffer = new float[length/4]; // 32/8
  fread(frame_buffer, length, 1, pFile);
  // std::cout << *(frame_buffer+length/4-1) << std::endl;
  std::copy(frame_buffer, frame_buffer+(length/4), input->begin());
}

void SaveITensor(const std::string& path, const zdl::DlSystem::ITensor* tensor)
{
   std::ofstream os(path, std::ofstream::binary);
   if (!os)
   {
      std::cerr << "Failed to open output file for writing: " << path << "\n";
      std::exit(EXIT_FAILURE);
   }
   for ( auto it = tensor->cbegin(); it != tensor->cend(); ++it )
   {
      float f = *it;
      if (!os.write(reinterpret_cast<char*>(&f), sizeof(float)))
      {
         std::cerr << "Failed to write data to: " << path << "\n";
         std::exit(EXIT_FAILURE);
      }
   }
}

void testrun(char* modelfile) {
  static zdl::DlSystem::Runtime_t runtime = checkRuntime();
  std::unique_ptr<zdl::DlContainer::IDlContainer> container;
  container = zdl::DlContainer::IDlContainer::open(modelfile);

  if (!container) { PrintErrorStringAndExit(); }
  cout << "start build" << endl;
  std::unique_ptr<zdl::SNPE::SNPE> snpe;
  {
    snpe = NULL;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    snpe = snpeBuilder.setOutputLayers({})
      .setRuntimeProcessor(runtime)
      .setUseUserSuppliedBuffers(false)
      //.setDebugMode(true)
      .build();
    if (!snpe) {
      cout << "ERROR!" << endl;
      const char* const errStr = zdl::DlSystem::getLastErrorString();
      std::cerr << errStr << std::endl;
    }
    cout << "ran snpeBuilder" << endl;
  }

  const auto &strList_opt = snpe->getInputTensorNames();
  if (!strList_opt) throw std::runtime_error("Error obtaining input tensor names");
  cout << "get input tensor names done" << endl;

  const auto &strList = *strList_opt;
  static zdl::DlSystem::TensorMap inputTensorMap;
  static zdl::DlSystem::TensorMap outputTensorMap;

  assert (strList.size() == 1);
  const auto &inputDims_opt = snpe->getInputDimensions(strList.at(0));
  const auto &inputShape = *inputDims_opt;
  std::cout << "winkwink" << std::endl;

  for (int i=0;i<10000;i++) {
    std::unique_ptr<zdl::DlSystem::ITensor> input;
    input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);
    get_testframe(i,input);
    snpe->execute(input.get(), outputTensorMap);
    zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();
    std::for_each( tensorNames.begin(), tensorNames.end(), [&](const char* name) {
      std::ostringstream path;
      path << "/data/opt/Result_" << std::to_string(i) << ".raw";
      auto tensorPtr = outputTensorMap.getTensor(name);
      SaveITensor(path.str(), tensorPtr);
    });
  }
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("usage: %s <filename>\n", argv[0]);
    return -1;
  }

  if (argc == 2) {
    while (true) test(argv[1]);
  } else if (argc == 3) {
    testrun(argv[1]);
  }
  return 0;
}

