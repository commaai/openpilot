#include <string.h>
#include "dmonitoring.h"
#include "common/mat.h"
#include "common/timing.h"
#include "common/params.h"

#include <libyuv.h>

#define MODEL_WIDTH 320
#define MODEL_HEIGHT 640
#define FULL_W 852 // should get these numbers from camerad

#if defined(QCOM) || defined(QCOM2)
#define input_lambda(x) (x - 128.f) * 0.0078125f
#else
#define input_lambda(x) x // for non SNPE running platforms, assume keras model instead has lambda layer
#endif

void dmonitoring_init(DMonitoringModelState* s) {
#if defined(QCOM) || defined(QCOM2)
  const char* model_path = "../../models/dmonitoring_model_q.dlc";
#else
  const char* model_path = "../../models/dmonitoring_model.dlc";
#endif
#ifdef QCOM2
  int runtime = USE_CPU_RUNTIME;
#else
  int runtime = USE_DSP_RUNTIME;
#endif
  s->m = new DefaultRunModel(model_path, (float*)&s->output, OUTPUT_SIZE, runtime);
  s->is_rhd = Params().read_db_bool("IsRHD");
}

template <class T>
static inline T *get_buffer(std::vector<T> &buf, const size_t size) {
  if (buf.size() < size) {
    buf.resize(size);
  }
  return buf.data();
}

DMonitoringResult dmonitoring_eval_frame(DMonitoringModelState* s, void* stream_buf, int width, int height) {
  uint8_t *raw_buf = (uint8_t*) stream_buf;
  uint8_t *raw_y_buf = raw_buf;
  uint8_t *raw_u_buf = raw_y_buf + (width * height);
  uint8_t *raw_v_buf = raw_u_buf + ((width/2) * (height/2));

#ifndef QCOM2
  const int cropped_width = height/2;
  const int cropped_height = height;
  const int global_x_offset = 0;
  const int global_y_offset = 0;
  const int crop_x_offset = width - cropped_width;
  const int crop_y_offset = 0;
#else
  const int full_width_tici = 1928;
  const int full_height_tici = 1208;
  const int adapt_width_tici = 808;

  const int cropped_height = adapt_width_tici / 1.33;
  const int cropped_width = cropped_height / 2;
  const int global_x_offset = full_width_tici / 2 - adapt_width_tici / 2;
  const int global_y_offset = full_height_tici / 2 - cropped_height / 2;
  const int crop_x_offset = adapt_width_tici - cropped_width;
  const int crop_y_offset = 0;
#endif

  int resized_width = MODEL_WIDTH;
  int resized_height = MODEL_HEIGHT;

  uint8_t *cropped_y_buf = get_buffer(s->cropped_buf, cropped_width*cropped_height*3/2);
  uint8_t *cropped_u_buf = cropped_y_buf + (cropped_width * cropped_height);
  uint8_t *cropped_v_buf = cropped_u_buf + ((cropped_width/2) * (cropped_height/2));

  if (!s->is_rhd) {
    for (int r = 0; r < cropped_height/2; r++) {
      memcpy(cropped_y_buf + 2*r*cropped_width, raw_y_buf + (2*r + global_y_offset + crop_y_offset)*width + global_x_offset + crop_x_offset, cropped_width);
      memcpy(cropped_y_buf + (2*r+1)*cropped_width, raw_y_buf + (2*r + global_y_offset + crop_y_offset + 1)*width + global_x_offset + crop_x_offset, cropped_width);
      memcpy(cropped_u_buf + r*cropped_width/2, raw_u_buf + (r + (global_y_offset + crop_y_offset)/2)*width/2 + (global_x_offset + crop_x_offset)/2, cropped_width/2);
      memcpy(cropped_v_buf + r*cropped_width/2, raw_v_buf + (r + (global_y_offset + crop_y_offset)/2)*width/2 + (global_x_offset + crop_x_offset)/2, cropped_width/2);
    }
  } else {
    uint8_t *premirror_cropped_y_buf = get_buffer(s->premirror_cropped_buf, cropped_width*cropped_height*3/2);
    uint8_t *premirror_cropped_u_buf = premirror_cropped_y_buf + (cropped_width * cropped_height);
    uint8_t *premirror_cropped_v_buf = premirror_cropped_u_buf + ((cropped_width/2) * (cropped_height/2));
    for (int r = 0; r < cropped_height/2; r++) {
      memcpy(premirror_cropped_y_buf + (2*r)*cropped_width, raw_y_buf + (2*r + global_y_offset + crop_y_offset)*width + global_x_offset, cropped_width);
      memcpy(premirror_cropped_y_buf + (2*r+1)*cropped_width, raw_y_buf + (2*r + global_y_offset + crop_y_offset + 1)*width + global_x_offset, cropped_width);
      memcpy(premirror_cropped_u_buf + r*cropped_width/2, raw_u_buf + (r + (global_y_offset + crop_y_offset)/2)*width/2 + global_x_offset/2, cropped_width/2);
      memcpy(premirror_cropped_v_buf + r*cropped_width/2, raw_v_buf + (r + (global_y_offset + crop_y_offset)/2)*width/2 + global_x_offset/2, cropped_width/2);
    }
    libyuv::I420Mirror(premirror_cropped_y_buf, cropped_width,
                       premirror_cropped_u_buf, cropped_width/2,
                       premirror_cropped_v_buf, cropped_width/2,
                       cropped_y_buf, cropped_width,
                       cropped_u_buf, cropped_width/2,
                       cropped_v_buf, cropped_width/2,
                       cropped_width, cropped_height);
  }

  uint8_t *resized_buf = get_buffer(s->resized_buf, resized_width*resized_height*3/2);
  uint8_t *resized_y_buf = resized_buf;
  uint8_t *resized_u_buf = resized_y_buf + (resized_width * resized_height);
  uint8_t *resized_v_buf = resized_u_buf + ((resized_width/2) * (resized_height/2));

  libyuv::FilterMode mode = libyuv::FilterModeEnum::kFilterBilinear;
  libyuv::I420Scale(cropped_y_buf, cropped_width,
                    cropped_u_buf, cropped_width/2,
                    cropped_v_buf, cropped_width/2,
                    cropped_width, cropped_height,
                    resized_y_buf, resized_width,
                    resized_u_buf, resized_width/2,
                    resized_v_buf, resized_width/2,
                    resized_width, resized_height,
                    mode);

  int yuv_buf_len = (MODEL_WIDTH/2) * (MODEL_HEIGHT/2) * 6; // Y|u|v -> y|y|y|y|u|v
  float *net_input_buf = get_buffer(s->net_input_buf, yuv_buf_len);
  // one shot conversion, O(n) anyway
  // yuvframe2tensor, normalize
  for (int r = 0; r < MODEL_HEIGHT/2; r++) {
    for (int c = 0; c < MODEL_WIDTH/2; c++) {
      // Y_ul
      net_input_buf[(c*MODEL_HEIGHT/2) + r] = input_lambda(resized_buf[(2*r*resized_width) + (2*c)]);
      // Y_ur
      net_input_buf[(c*MODEL_HEIGHT/2) + r + (2*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = input_lambda(resized_buf[(2*r*resized_width) + (2*c+1)]);
      // Y_dl
      net_input_buf[(c*MODEL_HEIGHT/2) + r + ((MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = input_lambda(resized_buf[(2*r*resized_width+1) + (2*c)]);
      // Y_dr
      net_input_buf[(c*MODEL_HEIGHT/2) + r + (3*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = input_lambda(resized_buf[(2*r*resized_width+1) + (2*c+1)]);
      // U
      net_input_buf[(c*MODEL_HEIGHT/2) + r + (4*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = input_lambda(resized_buf[(resized_width*resized_height) + (r*resized_width/2) + c]);
      // V
      net_input_buf[(c*MODEL_HEIGHT/2) + r + (5*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = input_lambda(resized_buf[(resized_width*resized_height) + ((resized_width/2)*(resized_height/2)) + (r*resized_width/2) + c]);
    }
  }

  //printf("preprocess completed. %d \n", yuv_buf_len);
  //FILE *dump_yuv_file = fopen("/tmp/rawdump.yuv", "wb");
  //fwrite(raw_buf, height*width*3/2, sizeof(uint8_t), dump_yuv_file);
  //fclose(dump_yuv_file);

  //FILE *dump_yuv_file2 = fopen("/tmp/inputdump.yuv", "wb");
  //fwrite(net_input_buf, MODEL_HEIGHT*MODEL_WIDTH*3/2, sizeof(float), dump_yuv_file2);
  //fclose(dump_yuv_file2);

  s->m->execute(net_input_buf, yuv_buf_len);

  DMonitoringResult ret = {0};
  memcpy(&ret.face_orientation, &s->output[0], sizeof ret.face_orientation);
  memcpy(&ret.face_orientation_meta, &s->output[6], sizeof ret.face_orientation_meta);
  memcpy(&ret.face_position, &s->output[3], sizeof ret.face_position);
  memcpy(&ret.face_position_meta, &s->output[9], sizeof ret.face_position_meta);
  memcpy(&ret.face_prob, &s->output[12], sizeof ret.face_prob);
  memcpy(&ret.left_eye_prob, &s->output[21], sizeof ret.left_eye_prob);
  memcpy(&ret.right_eye_prob, &s->output[30], sizeof ret.right_eye_prob);
  memcpy(&ret.left_blink_prob, &s->output[31], sizeof ret.right_eye_prob);
  memcpy(&ret.right_blink_prob, &s->output[32], sizeof ret.right_eye_prob);
  memcpy(&ret.sg_prob, &s->output[33], sizeof ret.sg_prob);
  ret.face_orientation_meta[0] = softplus(ret.face_orientation_meta[0]);
  ret.face_orientation_meta[1] = softplus(ret.face_orientation_meta[1]);
  ret.face_orientation_meta[2] = softplus(ret.face_orientation_meta[2]);
  ret.face_position_meta[0] = softplus(ret.face_position_meta[0]);
  ret.face_position_meta[1] = softplus(ret.face_position_meta[1]);
  return ret;
}

void dmonitoring_publish(PubMaster &pm, uint32_t frame_id, const DMonitoringResult &res){
  // make msg
  MessageBuilder msg;
  auto framed = msg.initEvent().initDriverState();
  framed.setFrameId(frame_id);

  kj::ArrayPtr<const float> face_orientation(&res.face_orientation[0], ARRAYSIZE(res.face_orientation));
  kj::ArrayPtr<const float> face_orientation_std(&res.face_orientation_meta[0], ARRAYSIZE(res.face_orientation_meta));
  kj::ArrayPtr<const float> face_position(&res.face_position[0], ARRAYSIZE(res.face_position));
  kj::ArrayPtr<const float> face_position_std(&res.face_position_meta[0], ARRAYSIZE(res.face_position_meta));
  framed.setFaceOrientation(face_orientation);
  framed.setFaceOrientationStd(face_orientation_std);
  framed.setFacePosition(face_position);
  framed.setFacePositionStd(face_position_std);
  framed.setFaceProb(res.face_prob);
  framed.setLeftEyeProb(res.left_eye_prob);
  framed.setRightEyeProb(res.right_eye_prob);
  framed.setLeftBlinkProb(res.left_blink_prob);
  framed.setRightBlinkProb(res.right_blink_prob);
  framed.setSgProb(res.sg_prob);

  pm.send("driverState", msg);
}

void dmonitoring_free(DMonitoringModelState* s) {
  delete s->m;
}
