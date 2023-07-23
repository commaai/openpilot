#include <cstring>

#include "libyuv.h"

#include "common/mat.h"
#include "common/modeldata.h"
#include "common/params.h"
#include "common/timing.h"
#include "system/hardware/hw.h"

#include "selfdrive/modeld/models/dmonitoring.h"

constexpr int MODEL_WIDTH = 1440;
constexpr int MODEL_HEIGHT = 960;

template <class T>
static inline T *get_buffer(std::vector<T> &buf, const size_t size) {
  if (buf.size() < size) buf.resize(size);
  return buf.data();
}

void dmonitoring_init(DMonitoringModelState* s) {

#ifdef USE_ONNX_MODEL
  s->m = new ONNXModel("models/dmonitoring_model.onnx", &s->output[0], OUTPUT_SIZE, USE_DSP_RUNTIME, true);
#else
  s->m = new SNPEModel("models/dmonitoring_model_q.dlc", &s->output[0], OUTPUT_SIZE, USE_DSP_RUNTIME, true);
#endif

  s->m->addInput("input_imgs", NULL, 0);
  s->m->addInput("calib", s->calib, CALIB_LEN);
}

void parse_driver_data(DriverStateResult &ds_res, const DMonitoringModelState* s, int out_idx_offset) {
  for (int i = 0; i < 3; ++i) {
    ds_res.face_orientation[i] = s->output[out_idx_offset+i] * REG_SCALE;
    ds_res.face_orientation_std[i] = exp(s->output[out_idx_offset+6+i]);
  }
  for (int i = 0; i < 2; ++i) {
    ds_res.face_position[i] = s->output[out_idx_offset+3+i] * REG_SCALE;
    ds_res.face_position_std[i] = exp(s->output[out_idx_offset+9+i]);
  }
  for (int i = 0; i < 4; ++i) {
    ds_res.ready_prob[i] = sigmoid(s->output[out_idx_offset+35+i]);
  }
  for (int i = 0; i < 2; ++i) {
    ds_res.not_ready_prob[i] = sigmoid(s->output[out_idx_offset+39+i]);
  }
  ds_res.face_prob = sigmoid(s->output[out_idx_offset+12]);
  ds_res.left_eye_prob = sigmoid(s->output[out_idx_offset+21]);
  ds_res.right_eye_prob = sigmoid(s->output[out_idx_offset+30]);
  ds_res.left_blink_prob = sigmoid(s->output[out_idx_offset+31]);
  ds_res.right_blink_prob = sigmoid(s->output[out_idx_offset+32]);
  ds_res.sunglasses_prob = sigmoid(s->output[out_idx_offset+33]);
  ds_res.occluded_prob = sigmoid(s->output[out_idx_offset+34]);
}

void fill_driver_data(cereal::DriverStateV2::DriverData::Builder ddata, const DriverStateResult &ds_res) {
  ddata.setFaceOrientation(ds_res.face_orientation);
  ddata.setFaceOrientationStd(ds_res.face_orientation_std);
  ddata.setFacePosition(ds_res.face_position);
  ddata.setFacePositionStd(ds_res.face_position_std);
  ddata.setFaceProb(ds_res.face_prob);
  ddata.setLeftEyeProb(ds_res.left_eye_prob);
  ddata.setRightEyeProb(ds_res.right_eye_prob);
  ddata.setLeftBlinkProb(ds_res.left_blink_prob);
  ddata.setRightBlinkProb(ds_res.right_blink_prob);
  ddata.setSunglassesProb(ds_res.sunglasses_prob);
  ddata.setOccludedProb(ds_res.occluded_prob);
  ddata.setReadyProb(ds_res.ready_prob);
  ddata.setNotReadyProb(ds_res.not_ready_prob);
}

DMonitoringModelResult dmonitoring_eval_frame(DMonitoringModelState* s, void* stream_buf, int width, int height, int stride, int uv_offset, float *calib) {
  int v_off = height - MODEL_HEIGHT;
  int h_off = (width - MODEL_WIDTH) / 2;
  int yuv_buf_len = MODEL_WIDTH * MODEL_HEIGHT;

  uint8_t *raw_buf = (uint8_t *) stream_buf;
  // vertical crop free
  uint8_t *raw_y_start = raw_buf + stride * v_off;

  uint8_t *net_input_buf = get_buffer(s->net_input_buf, yuv_buf_len);

  // here makes a uint8 copy
  for (int r = 0; r < MODEL_HEIGHT; ++r) {
    memcpy(net_input_buf + r * MODEL_WIDTH, raw_y_start + r * stride + h_off, MODEL_WIDTH);
  }

  // printf("preprocess completed. %d \n", yuv_buf_len);
  // FILE *dump_yuv_file = fopen("/tmp/rawdump.yuv", "wb");
  // fwrite(net_input_buf, yuv_buf_len, sizeof(uint8_t), dump_yuv_file);
  // fclose(dump_yuv_file);

  double t1 = millis_since_boot();
  s->m->setInputBuffer("input_imgs", (float*)net_input_buf, yuv_buf_len / sizeof(float));
  for (int i = 0; i < CALIB_LEN; i++) {
    s->calib[i] = calib[i];
  }
  s->m->execute();
  double t2 = millis_since_boot();

  DMonitoringModelResult model_res = {0};
  parse_driver_data(model_res.driver_state_lhd, s, 0);
  parse_driver_data(model_res.driver_state_rhd, s, 41);
  model_res.poor_vision_prob = sigmoid(s->output[82]);
  model_res.wheel_on_right_prob = sigmoid(s->output[83]);
  model_res.dsp_execution_time = (t2 - t1) / 1000.;

  return model_res;
}

void dmonitoring_publish(PubMaster &pm, uint32_t frame_id, const DMonitoringModelResult &model_res, float execution_time, kj::ArrayPtr<const float> raw_pred) {
  // make msg
  MessageBuilder msg;
  auto framed = msg.initEvent().initDriverStateV2();
  framed.setFrameId(frame_id);
  framed.setModelExecutionTime(execution_time);
  framed.setDspExecutionTime(model_res.dsp_execution_time);

  framed.setPoorVisionProb(model_res.poor_vision_prob);
  framed.setWheelOnRightProb(model_res.wheel_on_right_prob);
  fill_driver_data(framed.initLeftDriverData(), model_res.driver_state_lhd);
  fill_driver_data(framed.initRightDriverData(), model_res.driver_state_rhd);

  if (send_raw_pred) {
    framed.setRawPredictions(raw_pred.asBytes());
  }

  pm.send("driverStateV2", msg);
}

void dmonitoring_free(DMonitoringModelState* s) {
  delete s->m;
}
