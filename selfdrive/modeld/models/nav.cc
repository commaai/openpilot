#include <cstdio>
#include <cstring>

#include "cereal/visionipc/visionipc_client.h"
#include "common/mat.h"
#include "common/modeldata.h"
#include "common/params.h"
#include "common/timing.h"
#include "system/hardware/hw.h"

#include "selfdrive/modeld/models/nav.h"


void navmodel_init(NavModelState* s) {
#ifdef USE_ONNX_MODEL
  s->m = new ONNXModel("models/navmodel.onnx", &s->output[0], OUTPUT_SIZE, USE_DSP_RUNTIME, false, false); // TODO: Set _use_tf8=true for quantized models?
#else
  s->m = new SNPEModel("models/navmodel_q.dlc", &s->output[0], OUTPUT_SIZE, USE_DSP_RUNTIME, false, true);
#endif
}

NavModelResult navmodel_eval_frame(NavModelState* s, VisionBuf* buf) {
  // convert from uint8 to float32
  // memcpy(s->net_input_buf, buf->addr, INPUT_SIZE);
  for (int i=0; i<INPUT_SIZE; i++) {
    s->net_input_buf[i] = ((uint8_t*)buf->addr)[i];
  }

  double t1 = millis_since_boot();
  s->m->addImage(s->net_input_buf, INPUT_SIZE);
  s->m->execute();
  double t2 = millis_since_boot();

  NavModelResult model_res = {0};
  model_res.dsp_execution_time = (t2 - t1) / 1000.;
  // TODO: Fill in features

  return model_res;
}

void navmodel_publish(PubMaster &pm, uint32_t frame_id, const NavModelResult &model_res, float execution_time) {
  // make msg
  MessageBuilder msg;
  auto framed = msg.initEvent().initNavModel();
  framed.setFrameId(frame_id);
  framed.setModelExecutionTime(execution_time);
  framed.setDspExecutionTime(model_res.dsp_execution_time);

  // TODO: Fill in features

  pm.send("navModel", msg);
}

void navmodel_free(NavModelState* s) {
  delete s->m;
}
