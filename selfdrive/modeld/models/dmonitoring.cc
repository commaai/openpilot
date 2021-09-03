#include <cstring>

#include "libyuv.h"

#include "selfdrive/common/mat.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/hardware/hw.h"

#include "selfdrive/modeld/models/dmonitoring.h"

#define MODEL_WIDTH 320
#define MODEL_HEIGHT 640
#define FULL_W 852 // should get these numbers from camerad

static void crop_yuv(uint8_t *raw, int width, int height, const YUVBuf *buf, const Rect &rect) {
  uint8_t *raw_y = raw;
  uint8_t *raw_u = raw_y + (width * height);
  uint8_t *raw_v = raw_u + ((width / 2) * (height / 2));
  for (int r = 0; r < rect.h / 2; r++) {
    memcpy(buf->y + 2 * r * rect.w, raw_y + (2 * r + rect.y) * width + rect.x, rect.w);
    memcpy(buf->y + (2 * r + 1) * rect.w, raw_y + (2 * r + rect.y + 1) * width + rect.x, rect.w);
    memcpy(buf->u + r * (rect.w / 2), raw_u + (r + (rect.y / 2)) * width / 2 + (rect.x / 2), rect.w / 2);
    memcpy(buf->v + r * (rect.w / 2), raw_v + (r + (rect.y / 2)) * width / 2 + (rect.x / 2), rect.w / 2);
  }
}

void dmonitoring_init(DMonitoringModelState* s, int width, int height) {
  s->is_rhd = Params().getBool("IsRHD");
  for (int x = 0; x < std::size(s->tensor); ++x) {
    s->tensor[x] = (x - 128.f) * 0.0078125f;
  }

#ifdef USE_ONNX_MODEL
  s->m = new ONNXModel("../../models/dmonitoring_model.onnx", &s->output[0], OUTPUT_SIZE, USE_DSP_RUNTIME);
#else
  s->m = new SNPEModel("../../models/dmonitoring_model_q.dlc", &s->output[0], OUTPUT_SIZE, USE_DSP_RUNTIME);
#endif

  // initialize buffers
  if (Hardware::TICI()) {
    const int full_width_tici = 1928;
    const int full_height_tici = 1208;
    const int adapt_width_tici = 668;
    const int cropped_height = adapt_width_tici / 1.33;
    s->crop_rect = {full_width_tici / 2 - adapt_width_tici / 2,
                    full_height_tici / 2 - cropped_height / 2 - 196,
                    cropped_height / 2,
                    cropped_height};
    if (!s->is_rhd) {
      s->crop_rect.x += adapt_width_tici - s->crop_rect.w + 32;
    }

  } else {
    s->crop_rect = {0, 0, height / 2, height};
    if (!s->is_rhd) {
      s->crop_rect.x += width - s->crop_rect.w;
    }
  }

  s->cropped_buf.init(s->crop_rect.w, s->crop_rect.h);
  s->premirror_cropped_buf.init(s->crop_rect.w, s->crop_rect.h);
  s->resized_buf.init(MODEL_WIDTH, MODEL_HEIGHT);
  s->net_input_buf.resize((MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2) * 6);  // Y|u|v -> y|y|y|y|u|v
}

DMonitoringResult dmonitoring_eval_frame(DMonitoringModelState* s, void* stream_buf, int width, int height) {
  const int crop_w = s->crop_rect.w;
  const int crop_h = s->crop_rect.h;
  if (!s->is_rhd) {
    crop_yuv((uint8_t *)stream_buf, width, height, &s->cropped_buf, s->crop_rect);
  } else {
    crop_yuv((uint8_t *)stream_buf, width, height, &s->premirror_cropped_buf, s->crop_rect);
    libyuv::I420Mirror(s->premirror_cropped_buf.y, crop_w,
                       s->premirror_cropped_buf.u, crop_w / 2,
                       s->premirror_cropped_buf.v, crop_w / 2,
                       s->cropped_buf.y, crop_w,
                       s->cropped_buf.u, crop_w / 2,
                       s->cropped_buf.v, crop_w / 2,
                       crop_w, crop_h);
  }

  libyuv::I420Scale(s->cropped_buf.y, crop_w,
                    s->cropped_buf.u, crop_w / 2,
                    s->cropped_buf.v, crop_w / 2,
                    crop_w, crop_h,
                    s->resized_buf.y, MODEL_WIDTH,
                    s->resized_buf.u, MODEL_WIDTH / 2,
                    s->resized_buf.v, MODEL_WIDTH / 2,
                    MODEL_WIDTH, MODEL_HEIGHT,
                    libyuv::FilterModeEnum::kFilterBilinear);

  // one shot conversion, O(n) anyway
  // yuvframe2tensor, normalize
  for (int r = 0; r < MODEL_HEIGHT/2; r++) {
    for (int c = 0; c < MODEL_WIDTH/2; c++) {
      // Y_ul
      s->net_input_buf[(r*MODEL_WIDTH/2) + c + (0*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = s->tensor[s->resized_buf.y[(2*r)*MODEL_WIDTH + 2*c]];
      // Y_dl
      s->net_input_buf[(r*MODEL_WIDTH/2) + c + (1*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = s->tensor[s->resized_buf.y[(2*r+1)*MODEL_WIDTH + 2*c]];
      // Y_ur
      s->net_input_buf[(r*MODEL_WIDTH/2) + c + (2*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = s->tensor[s->resized_buf.y[(2*r)*MODEL_WIDTH + 2*c+1]];
      // Y_dr
      s->net_input_buf[(r*MODEL_WIDTH/2) + c + (3*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = s->tensor[s->resized_buf.y[(2*r+1)*MODEL_WIDTH + 2*c+1]];
      // U
      s->net_input_buf[(r*MODEL_WIDTH/2) + c + (4*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = s->tensor[s->resized_buf.u[r*MODEL_WIDTH/2 + c]];
      // V
      s->net_input_buf[(r*MODEL_WIDTH/2) + c + (5*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = s->tensor[s->resized_buf.v[r*MODEL_WIDTH/2 + c]];
    }
  }

  //printf("preprocess completed. %d \n", yuv_buf_len);
  //FILE *dump_yuv_file = fopen("/tmp/rawdump.yuv", "wb");
  //fwrite(raw_buf, height*width*3/2, sizeof(uint8_t), dump_yuv_file);
  //fclose(dump_yuv_file);

  // *** testing ***
  // idat = np.frombuffer(open("/tmp/inputdump.yuv", "rb").read(), np.float32).reshape(6, 160, 320)
  // imshow(cv2.cvtColor(tensor_to_frames(idat[None]/0.0078125+128)[0], cv2.COLOR_YUV2RGB_I420))

  //FILE *dump_yuv_file2 = fopen("/tmp/inputdump.yuv", "wb");
  //fwrite(net_input_buf, MODEL_HEIGHT*MODEL_WIDTH*3/2, sizeof(float), dump_yuv_file2);
  //fclose(dump_yuv_file2);

  double t1 = millis_since_boot();
  s->m->execute(s->net_input_buf.data(), s->net_input_buf.size());
  double t2 = millis_since_boot();

  DMonitoringResult ret = {0};
  for (int i = 0; i < 3; ++i) {
    ret.face_orientation[i] = s->output[i];
    ret.face_orientation_meta[i] = softplus(s->output[6 + i]);
  }
  for (int i = 0; i < 2; ++i) {
    ret.face_position[i] = s->output[3 + i];
    ret.face_position_meta[i] = softplus(s->output[9 + i]);
  }
  ret.face_prob = s->output[12];
  ret.left_eye_prob = s->output[21];
  ret.right_eye_prob = s->output[30];
  ret.left_blink_prob = s->output[31];
  ret.right_blink_prob = s->output[32];
  ret.sg_prob = s->output[33];
  ret.poor_vision = s->output[34];
  ret.partial_face = s->output[35];
  ret.distracted_pose = s->output[36];
  ret.distracted_eyes = s->output[37];
  ret.dsp_execution_time = (t2 - t1) / 1000.;
  return ret;
}

void dmonitoring_publish(PubMaster &pm, uint32_t frame_id, const DMonitoringResult &res, float execution_time, kj::ArrayPtr<const float> raw_pred) {
  // make msg
  MessageBuilder msg;
  auto framed = msg.initEvent().initDriverState();
  framed.setFrameId(frame_id);
  framed.setModelExecutionTime(execution_time);
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
  if (send_raw_pred) {
    framed.setRawPredictions(raw_pred.asBytes());
  }

  pm.send("driverState", msg);
}

void dmonitoring_free(DMonitoringModelState* s) {
  delete s->m;
}
