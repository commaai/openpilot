#include <string.h>
#include "monitoring.h"
#include "common/mat.h"
#include "common/timing.h"

// #include <fastcv.h>
#include <libyuv.h>

#define MODEL_WIDTH 160
#define MODEL_HEIGHT 320
#define FULL_W 426

void monitoring_init(MonitoringState* s) {
  s->m = new DefaultRunModel("../../models/monitoring_model_q.dlc", (float*)&s->output, OUTPUT_SIZE, USE_DSP_RUNTIME);
}

MonitoringResult monitoring_eval_frame(MonitoringState* s, void* stream_buf, int width, int height) {

  uint8_t *raw_buf = (uint8_t*) stream_buf;
  uint8_t *raw_y_buf = raw_buf;
  uint8_t *raw_u_buf = raw_y_buf + (width * height);
  uint8_t *raw_v_buf = raw_u_buf + ((width/2) * (height/2));

  int cropped_width = height/2;
  int cropped_height = height;

  int resized_width = MODEL_WIDTH;
  int resized_height = MODEL_HEIGHT;

  uint8_t *cropped_buf = new uint8_t[cropped_width*cropped_height*3/2];
  uint8_t *cropped_y_buf = cropped_buf;
  uint8_t *cropped_u_buf = cropped_y_buf + (cropped_width * cropped_height);
  uint8_t *cropped_v_buf = cropped_u_buf + ((cropped_width/2) * (cropped_height/2));

  if (true) {
    for (int r = 0; r < height/2; r++) {
      memcpy(cropped_y_buf + 2*r*cropped_width, raw_y_buf + 2*r*width + (width - cropped_width), cropped_width);
      memcpy(cropped_y_buf + (2*r+1)*cropped_width, raw_y_buf + (2*r+1)*width + (width - cropped_width), cropped_width);
      memcpy(cropped_u_buf + r*cropped_width/2, raw_u_buf + r*width/2 + ((width/2) - (cropped_width/2)), cropped_width/2);
      memcpy(cropped_v_buf + r*cropped_width/2, raw_v_buf + r*width/2 + ((width/2) - (cropped_width/2)), cropped_width/2);
    }
  } else {
    // not tested
    uint8_t *premirror_cropped_buf = new uint8_t[cropped_width*cropped_height*3/2];
    uint8_t *premirror_cropped_y_buf = premirror_cropped_buf;
    uint8_t *premirror_cropped_u_buf = premirror_cropped_y_buf + (cropped_width * cropped_height);
    uint8_t *premirror_cropped_v_buf = premirror_cropped_u_buf + ((cropped_width/2) * (cropped_height/2));
    for (int r = 0; r < height/2; r++) {
      memcpy(premirror_cropped_y_buf + 2*r*cropped_width, raw_y_buf + 2*r*width, cropped_width);
      memcpy(premirror_cropped_y_buf + (2*r+1)*cropped_width, raw_y_buf + (2*r+1)*width, cropped_width);
      memcpy(premirror_cropped_u_buf + r*cropped_width/2, raw_u_buf + r*width/2, cropped_width/2);
      memcpy(premirror_cropped_v_buf + r*cropped_width/2, raw_v_buf + r*width/2, cropped_width/2);
    }
    libyuv::I420Mirror(premirror_cropped_y_buf, cropped_width,
                       premirror_cropped_u_buf, cropped_width/2,
                       premirror_cropped_v_buf, cropped_width/2,
                       cropped_y_buf, cropped_width,
                       cropped_u_buf, cropped_width/2,
                       cropped_v_buf, cropped_width/2,
                       cropped_width, cropped_height);
  }

  uint8_t *resized_buf = new uint8_t[resized_width*resized_height*3/2];
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

  float *net_input_buf = new float[yuv_buf_len];
  // one shot conversion, O(n) anyway
  // yuvframe2tensor, normalize
  for (int r = 0; r < MODEL_HEIGHT/2; r++) {
    for (int c = 0; c < MODEL_WIDTH/2; c++) {
      // Y_ul
      net_input_buf[(c*MODEL_HEIGHT/2) + r] = (resized_buf[(2*r*resized_width) + (2*c)] - 128.f) * 0.0078125f;
      // Y_ur
      net_input_buf[(c*MODEL_HEIGHT/2) + r + ((MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = (resized_buf[(2*r*resized_width) + (2*c+1)] - 128.f) * 0.0078125f;
      // Y_dl
      net_input_buf[(c*MODEL_HEIGHT/2) + r + (2*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = (resized_buf[(2*r*resized_width+1) + (2*c)] - 128.f) * 0.0078125f;
      // Y_dr
      net_input_buf[(c*MODEL_HEIGHT/2) + r + (3*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = (resized_buf[(2*r*resized_width+1) + (2*c+1)] - 128.f) * 0.0078125f;
      // U
      net_input_buf[(c*MODEL_HEIGHT/2) + r + (4*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = (resized_buf[(resized_width*resized_height) + (r*resized_width/2) + c] - 128.f) * 0.0078125f;
      // V
      net_input_buf[(c*MODEL_HEIGHT/2) + r + (5*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = (resized_buf[(resized_width*resized_height) + ((resized_width/2)*(resized_height/2)) + (r*resized_width/2) + c] - 128.f) * 0.0078125f;
    }
  }

  // FILE *dump_yuv_file = fopen("/sdcard/rawdump.yuv", "wb");
  // fwrite(raw_buf, height*width*3/2, sizeof(uint8_t), dump_yuv_file);
  // fclose(dump_yuv_file);

  // FILE *dump_yuv_file2 = fopen("/sdcard/inputdump.yuv", "wb");
  // fwrite(net_input_buf, MODEL_HEIGHT*MODEL_WIDTH*3/2, sizeof(float), dump_yuv_file2);
  // fclose(dump_yuv_file2);

  delete[] cropped_buf;
  delete[] resized_buf;
  s->m->execute(net_input_buf);
  delete[] net_input_buf;

  MonitoringResult ret = {0};
  memcpy(&ret.face_orientation, &s->output[0], sizeof ret.face_orientation);
  memcpy(&ret.face_position, &s->output[3], sizeof ret.face_position);
  memcpy(&ret.face_prob, &s->output[12], sizeof ret.face_prob);
  memcpy(&ret.left_eye_prob, &s->output[21], sizeof ret.left_eye_prob);
  memcpy(&ret.right_eye_prob, &s->output[30], sizeof ret.right_eye_prob);
  memcpy(&ret.left_blink_prob, &s->output[31], sizeof ret.right_eye_prob);
  memcpy(&ret.right_blink_prob, &s->output[32], sizeof ret.right_eye_prob);
  return ret;
}

void monitoring_publish(PubSocket* sock, uint32_t frame_id, const MonitoringResult res) {
  // make msg
  capnp::MallocMessageBuilder msg;
  cereal::Event::Builder event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(nanos_since_boot());

  auto framed = event.initDriverMonitoring();
  framed.setFrameId(frame_id);

  kj::ArrayPtr<const float> face_orientation(&res.face_orientation[0], ARRAYSIZE(res.face_orientation));
  kj::ArrayPtr<const float> face_position(&res.face_position[0], ARRAYSIZE(res.face_position));
  framed.setFaceOrientation(face_orientation);
  framed.setFacePosition(face_position);
  framed.setFaceProb(res.face_prob);
  framed.setLeftEyeProb(res.left_eye_prob);
  framed.setRightEyeProb(res.right_eye_prob);
  framed.setLeftBlinkProb(res.left_blink_prob);
  framed.setRightBlinkProb(res.right_blink_prob);

  // send message
  auto words = capnp::messageToFlatArray(msg);
  auto bytes = words.asBytes();
  sock->send((char*)bytes.begin(), bytes.size());
}

void monitoring_free(MonitoringState* s) {
  delete s->m;
}
