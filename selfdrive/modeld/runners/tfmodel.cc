#include "tfmodel.h"
#include <string>
#include <string.h>
#include <stdlib.h>
#include <stdexcept>
#include "common/util.h"
#include "common/swaglog.h"
#include <cassert>

void TFModel::status_check() const {
  if (TF_GetCode(this->status) != TF_OK) {
    throw std::runtime_error(TF_Message(status));
  }
}

TF_Tensor *TFModel::allocate_tensor_for_output(TF_Output out, float *dat) {
  int num_dims = TF_GraphGetTensorNumDims(graph, out, status);
  status_check();
  int64_t *dims = new int64_t[num_dims];
  TF_GraphGetTensorShape(graph, out, dims, num_dims, status);
  status_check();
  dims[0] = 1;

  int total = 1;
  for (int i = 0; i < num_dims; i++) total *= dims[i];
  //printf("dims %d total %d wdat %p\n", num_dims, total, dat);

  // don't deallocate the buffers
  auto d = [](void* ddata, size_t, void* arg) {};
  TF_Tensor *ret = TF_NewTensor(TF_FLOAT, dims, num_dims, (void*)dat, sizeof(float)*total, d, NULL);

  //TF_Tensor *ret = TF_AllocateTensor(TF_FLOAT, dims, num_dims, sizeof(float)*total);
  //memcpy(TF_TensorData(ret), dat, sizeof(float)*total);

  assert(ret);
  delete[] dims;

  return ret;
}

TFModel::TFModel(const char *path, float *_output, size_t _output_size, int runtime) {
  // load model
  {
    TF_Buffer* buf;
    size_t model_size;
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s.pb", path);
    LOGD("loading model %s", tmp);
    uint8_t *model_data = (uint8_t *)read_file(tmp, &model_size);
    assert(model_data);
    buf = TF_NewBuffer();
    buf->data = model_data;
    buf->length = model_size;
    buf->data_deallocator = [](void *data, size_t) { free(data); };
    LOGD("loaded model of size %d", model_size);

    // import graph
    status = TF_NewStatus();
    graph = TF_NewGraph();
    TF_ImportGraphDefOptions *opts = TF_NewImportGraphDefOptions();
    // TODO: fix the GPU, currently it hangs if you set this to /gpu:0
    //TF_ImportGraphDefOptionsSetDefaultDevice(opts, "/cpu:0");
    TF_GraphImportGraphDef(graph, buf, opts, status);
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(buf);
    status_check();
    LOGD("imported graph");
  }

  // set up session
  TF_SessionOptions* sess_opts = TF_NewSessionOptions();

  // don't use all GPU memory
  /*uint8_t config[15] = {0x32, 0xb, 0x9, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x20, 0x1, 0x38, 0x1};
  double gpu_memory_fraction = 0.2;
  auto bytes = reinterpret_cast<std::uint8_t*>(&gpu_memory_fraction);
  for (std::size_t i = 0; i < sizeof(gpu_memory_fraction); ++i) {
    config[i + 3] = bytes[i];
  }
  TF_SetConfig(sess_opts, config, sizeof(config), status);
  status_check();*/

  // make session
  session = TF_NewSession(graph, sess_opts, status);
  TF_DeleteSessionOptions(sess_opts);
  status_check();

  // find tensors
  // TODO: make this generic
  input_operation = {TF_GraphOperationByName(graph, "lambda/div"), 0};
  if (input_operation.oper == NULL) {
    input_operation = {TF_GraphOperationByName(graph, "vision_lambda/div"), 0};
  }
  assert(input_operation.oper != NULL);

  output_operation = {TF_GraphOperationByName(graph, "outputs/outputs/Identity"), 0};
  if (output_operation.oper == NULL) {
    output_operation = {TF_GraphOperationByName(graph, "outputs/concat"), 0};
  }
  assert(output_operation.oper != NULL);

  // output tensor is good to bind now
  output = _output;
  output_size = _output_size;
}

TFModel::~TFModel() {
  TF_DeleteSession(session, status);
  status_check();
  TF_DeleteGraph(graph);
  TF_DeleteStatus(status);
}

void TFModel::addRecurrent(float *state, int state_size) {
  rnn_operation.oper = TF_GraphOperationByName(graph, "rnn_state");
  rnn_operation.index = 0;
  assert(rnn_operation.oper != NULL);

  rnn_input_buf = state;
}

void TFModel::addDesire(float *state, int state_size) {
  desire_operation.oper = TF_GraphOperationByName(graph, "desire");
  desire_operation.index = 0;
  assert(desire_operation.oper != NULL);

  desire_input_buf = state;
}

void TFModel::execute(float *net_input_buf) {
  TF_Tensor *input_tensor = allocate_tensor_for_output(input_operation, net_input_buf);
  assert(input_tensor);
  TF_Tensor *output_tensor = NULL;

  if (rnn_input_buf == NULL) {
    TF_SessionRun(session, NULL,
      &input_operation, &input_tensor, 1,
      &output_operation, &output_tensor, 1,
      NULL, 0, NULL, status);
  } else {
    //printf("%f %f %f\n", net_input_buf[0], rnn_input_buf[0], desire_input_buf[0]);
    TF_Tensor *rnn_tensor = allocate_tensor_for_output(rnn_operation, rnn_input_buf);
    TF_Tensor *desire_tensor = allocate_tensor_for_output(desire_operation, desire_input_buf);
    TF_Output io[] = {input_operation, rnn_operation, desire_operation};
    TF_Tensor* id[] = {input_tensor, rnn_tensor, desire_tensor};
    TF_SessionRun(session, NULL,
      io, id, 3,
      &output_operation, &output_tensor, 1,
      NULL, 0, NULL, status);
    TF_DeleteTensor(rnn_tensor);
    TF_DeleteTensor(desire_tensor);
  }
  TF_DeleteTensor(input_tensor);
  status_check();
  assert(output_tensor);
  memcpy((void*)output, TF_TensorData(output_tensor), output_size*sizeof(float));
  TF_DeleteTensor(output_tensor);
}


