#include "selfdrive/modeld/models/nav.h"

#include <cstdio>
#include <cstring>

#include "common/mat.h"
#include "common/modeldata.h"
#include "common/timing.h"


void navmodel_init(NavModelState* s) {
  #ifdef USE_ONNX_MODEL
    s->m = new ONNXModel("models/navmodel.onnx", &s->output[0], NAV_NET_OUTPUT_SIZE, USE_DSP_RUNTIME, true);
  #else
    s->m = new SNPEModel("models/navmodel_q.dlc", &s->output[0], NAV_NET_OUTPUT_SIZE, USE_DSP_RUNTIME, true);
  #endif

  s->m->addInput("map", NULL, 0);
}

NavModelResult* navmodel_eval_frame(NavModelState* s, VisionBuf* buf) {
  memcpy(s->net_input_buf, buf->addr, NAV_INPUT_SIZE);

  double t1 = millis_since_boot();
  s->m->setInputBuffer("map", (float*)s->net_input_buf, NAV_INPUT_SIZE/sizeof(float));
  s->m->execute();
  double t2 = millis_since_boot();

  NavModelResult *model_res = (NavModelResult*)&s->output;
  model_res->dsp_execution_time = (t2 - t1) / 1000.;
  return model_res;
}

void fill_plan(cereal::NavModelData::Builder &framed, const NavModelOutputPlan &plan) {
  std::array<float, TRAJECTORY_SIZE> pos_x, pos_y;
  std::array<float, TRAJECTORY_SIZE> pos_x_std, pos_y_std;

  for (int i=0; i<TRAJECTORY_SIZE; i++) {
    pos_x[i] = plan.mean[i].x;
    pos_y[i] = plan.mean[i].y;
    pos_x_std[i] = exp(plan.std[i].x);
    pos_y_std[i] = exp(plan.std[i].y);
  }

  auto position = framed.initPosition();
  position.setX(to_kj_array_ptr(pos_x));
  position.setY(to_kj_array_ptr(pos_y));
  position.setXStd(to_kj_array_ptr(pos_x_std));
  position.setYStd(to_kj_array_ptr(pos_y_std));
}

void navmodel_publish(PubMaster &pm, VisionIpcBufExtra &extra, const NavModelResult &model_res, float execution_time, bool route_valid) {
  // make msg
  MessageBuilder msg;
  auto evt = msg.initEvent();
  auto framed = evt.initNavModel();
  evt.setValid(extra.valid && route_valid);
  framed.setFrameId(extra.frame_id);
  framed.setLocationMonoTime(extra.timestamp_sof);
  framed.setModelExecutionTime(execution_time);
  framed.setDspExecutionTime(model_res.dsp_execution_time);
  framed.setFeatures(to_kj_array_ptr(model_res.features.values));
  framed.setDesirePrediction(to_kj_array_ptr(model_res.desire_pred.values));
  fill_plan(framed, model_res.plan);

  pm.send("navModel", msg);
}

void navmodel_free(NavModelState* s) {
  delete s->m;
}
