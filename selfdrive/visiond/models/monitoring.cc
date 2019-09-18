#include <string.h>
#include "monitoring.h"
#include "common/mat.h"
#include "common/timing.h"

#define MODEL_WIDTH 320
#define MODEL_HEIGHT 160

void monitoring_init(MonitoringState* s, cl_device_id device_id, cl_context context) {
  model_input_init(&s->in, MODEL_WIDTH, MODEL_HEIGHT, device_id, context);
  s->m = new SNPEModel("../../models/monitoring_model.dlc", (float*)&s->output, OUTPUT_SIZE);
}

MonitoringResult monitoring_eval_frame(MonitoringState* s, cl_command_queue q,
                           cl_mem yuv_cl, int width, int height) {
  const mat3 front_frame_from_scaled_frame = (mat3){{
    width/426.0f,          0.0, 0.0,
             0.0,height/320.0f, 0.0,
             0.0,          0.0, 1.0,
  }};

  const mat3 scaled_frame_from_cropped_frame = (mat3){{
      1.0, 0.0, 426.0-160.0,
      0.0, 1.0,         0.0,
      0.0, 0.0,         1.0,
  }};

  const mat3 transpose = (mat3){{
      0.0, 1.0, 0.0,
      1.0, 0.0, 0.0,
      0.0, 0.0, 1.0,
  }};

  const mat3 front_frame_from_cropped_frame = matmul3(front_frame_from_scaled_frame, scaled_frame_from_cropped_frame);
  const mat3 front_frame_from_monitoring_frame = matmul3(front_frame_from_cropped_frame, transpose);

  float *net_input_buf = model_input_prepare(&s->in, q, yuv_cl, width, height, front_frame_from_monitoring_frame);
  s->m->execute(net_input_buf);

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

void monitoring_publish(void* sock, uint32_t frame_id, const MonitoringResult res) {
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
        zmq_send(sock, bytes.begin(), bytes.size(), ZMQ_DONTWAIT);
      }

void monitoring_free(MonitoringState* s) {
  model_input_free(&s->in);
  delete s->m;
}
