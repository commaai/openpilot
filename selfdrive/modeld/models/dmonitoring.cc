#include "selfdrive/modeld/models/dmonitoring.h"

#include <cstring>

#include "libyuv.h"
#include "selfdrive/common/modeldata.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/hardware/hw.h"

static Rect get_crop_rect(bool is_rhd, int width, int height) {
  Rect crop_rect;
  if (width == TICI_CAM_WIDTH) {
    const int cropped_height = tici_dm_crop::width / 1.33;
    crop_rect = {width / 2 - tici_dm_crop::width / 2 + tici_dm_crop::x_offset,
                 height / 2 - cropped_height / 2 + tici_dm_crop::y_offset,
                 cropped_height / 2,
                 cropped_height};
    if (!is_rhd) {
      crop_rect.x += tici_dm_crop::width - crop_rect.w;
    }
  } else {
    const int adapt_width = 372;
    crop_rect = {0, 0, adapt_width, height};
    if (!is_rhd) {
      crop_rect.x += width - crop_rect.w;
    }
  }
  return crop_rect;
}

DMModel::DMModel(int width, int height) {
  is_rhd = Params().getBool("IsRHD");
  for (int x = 0; x < std::size(tensor); ++x) {
    tensor[x] = (x - 128.f) * 0.0078125f;
  }

  crop_rect = get_crop_rect(is_rhd, width, height);
  cropped_buf.init(crop_rect.w, crop_rect.h);
  premirror_cropped_buf.init(crop_rect.w, crop_rect.h);

  resized_buf.init(MODEL_WIDTH, MODEL_HEIGHT, true);
  net_input_buf.resize((MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2) * 6);  // Y|u|v -> y|y|y|y|u|v);

#ifdef USE_ONNX_MODEL
  m = std::make_unique<ONNXModel>("../../models/dmonitoring_model.onnx", &output[0], OUTPUT_SIZE, USE_DSP_RUNTIME);
#else
  m = std::make_unique<SNPEModel>("../../models/dmonitoring_model_q.dlc", &output[0], OUTPUT_SIZE, USE_DSP_RUNTIME);
#endif
}

const YUVBuf &DMModel::crop_yuv(uint8_t *raw, int width, int height) {
  uint8_t *raw_y = raw;
  uint8_t *raw_u = raw_y + (width * height);
  uint8_t *raw_v = raw_u + ((width / 2) * (height / 2));
  for (int r = 0; r < crop_rect.h / 2; r++) {
    memcpy(cropped_buf.y + 2 * r * crop_rect.w, raw_y + (2 * r + crop_rect.y) * width + crop_rect.x, crop_rect.w);
    memcpy(cropped_buf.y + (2 * r + 1) * crop_rect.w, raw_y + (2 * r + crop_rect.y + 1) * width + crop_rect.x, crop_rect.w);
    memcpy(cropped_buf.u + r * (crop_rect.w / 2), raw_u + (r + (crop_rect.y / 2)) * width / 2 + (crop_rect.x / 2), crop_rect.w / 2);
    memcpy(cropped_buf.v + r * (crop_rect.w / 2), raw_v + (r + (crop_rect.y / 2)) * width / 2 + (crop_rect.x / 2), crop_rect.w / 2);
  }

  if (!is_rhd) {
    return cropped_buf;
  } else {
    libyuv::I420Mirror(cropped_buf.y, crop_rect.w,
                       cropped_buf.u, crop_rect.w / 2,
                       cropped_buf.v, crop_rect.w / 2,
                       premirror_cropped_buf.y, crop_rect.w,
                       premirror_cropped_buf.u, crop_rect.w / 2,
                       premirror_cropped_buf.v, crop_rect.w / 2,
                       crop_rect.w, crop_rect.h);
    return premirror_cropped_buf;
  }
}

DMResult DMModel::eval_frame(uint8_t *stream_buf, int width, int height) {
  const double start_tm = millis_since_boot();

  const YUVBuf &cropped = crop_yuv(stream_buf, width, height);
  libyuv::FilterMode mode = libyuv::FilterModeEnum::kFilterBilinear;
  if (Hardware::TICI()) {
    libyuv::I420Scale(cropped.y, crop_rect.w,
                      cropped.u, crop_rect.w / 2,
                      cropped.v, crop_rect.w / 2,
                      crop_rect.w, crop_rect.h,
                      resized_buf.y, MODEL_WIDTH,
                      resized_buf.u, MODEL_WIDTH / 2,
                      resized_buf.v, MODEL_WIDTH / 2,
                      MODEL_WIDTH, MODEL_HEIGHT,
                      mode);
  } else {
    const int source_height = 0.7 * MODEL_HEIGHT;
    const int extra_height = (MODEL_HEIGHT - source_height) / 2;
    const int extra_width = (MODEL_WIDTH - source_height / 2) / 2;
    const int source_width = source_height / 2 + extra_width;
    libyuv::I420Scale(cropped.y, crop_rect.w,
                      cropped.u, crop_rect.w / 2,
                      cropped.v, crop_rect.w / 2,
                      crop_rect.w, crop_rect.h,
                      resized_buf.y + extra_height * MODEL_WIDTH, MODEL_WIDTH,
                      resized_buf.u + extra_height / 2 * MODEL_WIDTH / 2, MODEL_WIDTH / 2,
                      resized_buf.v + extra_height / 2 * MODEL_WIDTH / 2, MODEL_WIDTH / 2,
                      source_width, source_height,
                      mode);
  }

  float *input = net_input_buf.data();
  // one shot conversion, O(n) anyway
  // yuvframe2tensor, normalize
  for (int r = 0; r < MODEL_HEIGHT/2; r++) {
    for (int c = 0; c < MODEL_WIDTH/2; c++) {
      // Y_ul
      input[(r*MODEL_WIDTH/2) + c + (0*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = tensor[resized_buf.y[(2*r)*MODEL_WIDTH + 2*c]];
      // Y_dl
      input[(r*MODEL_WIDTH/2) + c + (1*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = tensor[resized_buf.y[(2*r+1)*MODEL_WIDTH + 2*c]];
      // Y_ur
      input[(r*MODEL_WIDTH/2) + c + (2*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = tensor[resized_buf.y[(2*r)*MODEL_WIDTH + 2*c+1]];
      // Y_dr
      input[(r*MODEL_WIDTH/2) + c + (3*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = tensor[resized_buf.y[(2*r+1)*MODEL_WIDTH + 2*c+1]];
      // U
      input[(r*MODEL_WIDTH/2) + c + (4*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = tensor[resized_buf.u[r*MODEL_WIDTH/2 + c]];
      // V
      input[(r*MODEL_WIDTH/2) + c + (5*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = tensor[resized_buf.v[r*MODEL_WIDTH/2 + c]];
    }
  }

  double t1 = millis_since_boot();
  m->execute(net_input_buf.data(), net_input_buf.size());
  double t2 = millis_since_boot();

  DMResult ret = {};
  for (int i = 0; i < 3; ++i) {
    ret.face_orientation[i] = output[i];
    ret.face_orientation_meta[i] = softplus(output[6 + i]);
  }
  for (int i = 0; i < 2; ++i) {
    ret.face_position[i] = output[3 + i];
    ret.face_position_meta[i] = softplus(output[9 + i]);
  }
  ret.face_prob = output[12];
  ret.left_eye_prob = output[21];
  ret.right_eye_prob = output[30];
  ret.left_blink_prob = output[31];
  ret.right_blink_prob = output[32];
  ret.sg_prob = output[33];
  ret.poor_vision = output[34];
  ret.partial_face = output[35];
  ret.distracted_pose = output[36];
  ret.distracted_eyes = output[37];
  ret.occluded_prob = output[38];
  ret.dsp_execution_time = (t2 - t1) / 1000.;
  ret.model_execution_time = (millis_since_boot() - start_tm) / 1000.;
  return ret;
}

void DMModel::publish(PubMaster &pm, uint32_t frame_id, const DMResult &res) {
  MessageBuilder msg;
  auto framed = msg.initEvent().initDriverState();
  framed.setFrameId(frame_id);
  framed.setModelExecutionTime(res.model_execution_time);
  framed.setDspExecutionTime(res.dsp_execution_time);

  framed.setFaceOrientation(res.face_orientation);
  framed.setFaceOrientationStd(res.face_orientation_meta);
  framed.setFacePosition(res.face_position);
  framed.setFacePositionStd(res.face_position_meta);
  framed.setFaceProb(res.face_prob);
  framed.setLeftEyeProb(res.left_eye_prob);
  framed.setRightEyeProb(res.right_eye_prob);
  framed.setLeftBlinkProb(res.left_blink_prob);
  framed.setRightBlinkProb(res.right_blink_prob);
  framed.setSunglassesProb(res.sg_prob);
  framed.setPoorVision(res.poor_vision);
  framed.setPartialFace(res.partial_face);
  framed.setDistractedPose(res.distracted_pose);
  framed.setDistractedEyes(res.distracted_eyes);
  framed.setOccludedProb(res.occluded_prob);
  if (send_raw_pred) {
    framed.setRawPredictions(kj::ArrayPtr(output).asBytes());
  }

  pm.send("driverState", msg);
}
