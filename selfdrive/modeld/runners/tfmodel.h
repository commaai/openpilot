#ifndef TFMODEL_H
#define TFMODEL_H

#include <stdlib.h>
#include "runmodel.h"

#include "tensorflow/c/c_api.h"

struct TFState;

class TFModel : public RunModel {
public:
  TFModel(const char *path, float *output, size_t output_size, int runtime);
	~TFModel();
  void addRecurrent(float *state, int state_size);
  void addDesire(float *state, int state_size);
  void execute(float *net_input_buf);
private:
  void status_check() const;
  TF_Tensor *allocate_tensor_for_output(TF_Output out, float *dat);

  float *output;
  size_t output_size;

  TF_Session* session;
  TF_Graph* graph;
  TF_Status* status;

  TF_Output input_operation;
  TF_Output rnn_operation;
  TF_Output desire_operation;
  TF_Output output_operation;

  float *rnn_input_buf = NULL;
  float *desire_input_buf = NULL;
};

#endif

