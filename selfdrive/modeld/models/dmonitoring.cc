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

#if defined(QCOM) || defined(QCOM2)
#define input_lambda(x) (x - 128.f) * 0.0078125f
#else
#define input_lambda(x) x // for non SNPE running platforms, assume keras model instead has lambda layer
#endif

void dmonitoring_init(DMonitoringModelState* s) {
  const char *model_path = "../../models/dmonitoring_model_q.dlc";
  int runtime = USE_DSP_RUNTIME;
  s->m = new DefaultRunModel(model_path, &s->output[0], OUTPUT_SIZE, runtime);
  s->is_rhd = Params().getBool("IsRHD");
}

template <class T>
static inline T *get_buffer(std::vector<T> &buf, const size_t size) {
  if (buf.size() < size) buf.resize(size);
  return buf.data();
}

static inline auto get_yuv_buf(std::vector<uint8_t> &buf, const int width, int height) {
  uint8_t *y = get_buffer(buf, width * height * 3 / 2);
  uint8_t *u = y + width * height;
  uint8_t *v = u + (width /2) * (height / 2);
  return std::make_tuple(y, u, v);
}

struct Rect {int x, y, w, h;};
void crop_yuv(uint8_t *raw, int width, int height, uint8_t *y, uint8_t *u, uint8_t *v, const Rect &rect) {
  uint8_t *raw_y = raw;
  uint8_t *raw_u = raw_y + (width * height);
  uint8_t *raw_v = raw_u + ((width / 2) * (height / 2));
  for (int r = 0; r < rect.h / 2; r++) {
    memcpy(y + 2 * r * rect.w, raw_y + (2 * r + rect.y) * width + rect.x, rect.w);
    memcpy(y + (2 * r + 1) * rect.w, raw_y + (2 * r + rect.y + 1) * width + rect.x, rect.w);
    memcpy(u + r * (rect.w / 2), raw_u + (r + (rect.y / 2)) * width / 2 + (rect.x / 2), rect.w / 2);
    memcpy(v + r * (rect.w / 2), raw_v + (r + (rect.y / 2)) * width / 2 + (rect.x / 2), rect.w / 2);
  }
}

DMonitoringResult dmonitoring_eval_frame(DMonitoringModelState* s, void* stream_buf, int width, int height) {
  Rect crop_rect;
  if (Hardware::TICI()) {
    const int full_width_tici = 1928;
    const int full_height_tici = 1208;
    const int adapt_width_tici = 668;
    const int cropped_height = adapt_width_tici / 1.33;
    crop_rect = {full_width_tici / 2 - adapt_width_tici / 2,
                 full_height_tici / 2 - cropped_height / 2 - 196,
                 cropped_height / 2,
                 cropped_height};
    if (!s->is_rhd) {
      crop_rect.x += adapt_width_tici - crop_rect.w + 32;
    }

  } else {
    crop_rect = {0, 0, height / 2, height};
    if (!s->is_rhd) {
      crop_rect.x += width - crop_rect.w;
    }
  }

  int resized_width = MODEL_WIDTH;
  int resized_height = MODEL_HEIGHT;

  auto [cropped_y, cropped_u, cropped_v] = get_yuv_buf(s->cropped_buf, crop_rect.w, crop_rect.h);
  if (!s->is_rhd) {
    crop_yuv((uint8_t *)stream_buf, width, height, cropped_y, cropped_u, cropped_v, crop_rect);
  } else {
    auto [mirror_y, mirror_u, mirror_v] = get_yuv_buf(s->premirror_cropped_buf, crop_rect.w, crop_rect.h);
    crop_yuv((uint8_t *)stream_buf, width, height, mirror_y, mirror_u, mirror_v, crop_rect);
    libyuv::I420Mirror(mirror_y, crop_rect.w,
                       mirror_u, crop_rect.w / 2,
                       mirror_v, crop_rect.w / 2,
                       cropped_y, crop_rect.w,
                       cropped_u, crop_rect.w / 2,
                       cropped_v, crop_rect.w / 2,
                       crop_rect.w, crop_rect.h);
  }

  auto [resized_buf, resized_u, resized_v] = get_yuv_buf(s->resized_buf, resized_width, resized_height);
  uint8_t *resized_y = resized_buf;
  libyuv::FilterMode mode = libyuv::FilterModeEnum::kFilterBilinear;
  libyuv::I420Scale(cropped_y, crop_rect.w,
                    cropped_u, crop_rect.w / 2,
                    cropped_v, crop_rect.w / 2,
                    crop_rect.w, crop_rect.h,
                    resized_y, resized_width,
                    resized_u, resized_width / 2,
                    resized_v, resized_width / 2,
                    resized_width, resized_height,
                    mode);

  int yuv_buf_len = (MODEL_WIDTH/2) * (MODEL_HEIGHT/2) * 6; // Y|u|v -> y|y|y|y|u|v
  float *net_input_buf = get_buffer(s->net_input_buf, yuv_buf_len);
  // one shot conversion, O(n) anyway
  // yuvframe2tensor, normalize
  for (int r = 0; r < MODEL_HEIGHT/2; r++) {
    for (int c = 0; c < MODEL_WIDTH/2; c++) {
      // Y_ul
      net_input_buf[(r*MODEL_WIDTH/2) + c + (0*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = input_lambda(resized_buf[(2*r)*resized_width + (2*c)]);
      // Y_dl
      net_input_buf[(r*MODEL_WIDTH/2) + c + (1*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = input_lambda(resized_buf[(2*r+1)*resized_width + (2*c)]);
      // Y_ur
      net_input_buf[(r*MODEL_WIDTH/2) + c + (2*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = input_lambda(resized_buf[(2*r)*resized_width + (2*c+1)]);
      // Y_dr
      net_input_buf[(r*MODEL_WIDTH/2) + c + (3*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = input_lambda(resized_buf[(2*r+1)*resized_width + (2*c+1)]);
      // U
      net_input_buf[(r*MODEL_WIDTH/2) + c + (4*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = input_lambda(resized_buf[(resized_width*resized_height) + r*resized_width/2 + c]);
      // V
      net_input_buf[(r*MODEL_WIDTH/2) + c + (5*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = input_lambda(resized_buf[(resized_width*resized_height) + ((resized_width/2)*(resized_height/2)) + c + (r*resized_width/2)]);
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
  s->m->execute(net_input_buf, yuv_buf_len);
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
  ret.eyes_on_road = s->output[38];
  ret.phone_use = s->output[39];
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
  framed.setEyesOnRoad(res.eyes_on_road);
  framed.setPhoneUse(res.phone_use);
  if (send_raw_pred) {
    framed.setRawPredictions(raw_pred.asBytes());
  }

  pm.send("driverState", msg);
}

void dmonitoring_free(DMonitoringModelState* s) {
  delete s->m;
}
