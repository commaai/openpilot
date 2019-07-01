#include "commonmodel.h"

#include <czmq.h>
#include "cereal/gen/c/log.capnp.h"
#include "common/mat.h"
#include "common/timing.h"

void model_input_init(ModelInput* s, int width, int height,
                      cl_device_id device_id, cl_context context) {
  int err;
  s->device_id = device_id;
  s->context = context;

  transform_init(&s->transform, context, device_id);
  s->transformed_width = width;
  s->transformed_height = height;

  s->transformed_y_cl = clCreateBuffer(s->context, CL_MEM_READ_WRITE,
                                       s->transformed_width*s->transformed_height, NULL, &err);
  assert(err == 0);
  s->transformed_u_cl = clCreateBuffer(s->context, CL_MEM_READ_WRITE,
                                       (s->transformed_width/2)*(s->transformed_height/2), NULL, &err);
  assert(err == 0);
  s->transformed_v_cl = clCreateBuffer(s->context, CL_MEM_READ_WRITE,
                                       (s->transformed_width/2)*(s->transformed_height/2), NULL, &err);
  assert(err == 0);

  s->net_input_size = ((width*height*3)/2)*sizeof(float);
  s->net_input = clCreateBuffer(s->context, CL_MEM_READ_WRITE,
                                s->net_input_size, (void*)NULL, &err);
  assert(err == 0);

  loadyuv_init(&s->loadyuv, context, device_id, s->transformed_width, s->transformed_height);
}

float *model_input_prepare(ModelInput* s, cl_command_queue q,
                           cl_mem yuv_cl, int width, int height,
                           mat3 transform) {
  int err;
  int i = 0;
  transform_queue(&s->transform, q,
                  yuv_cl, width, height,
                  s->transformed_y_cl, s->transformed_u_cl, s->transformed_v_cl,
                  s->transformed_width, s->transformed_height,
                  transform);
  loadyuv_queue(&s->loadyuv, q,
                s->transformed_y_cl, s->transformed_u_cl, s->transformed_v_cl,
                s->net_input);
  float *net_input_buf = (float *)clEnqueueMapBuffer(q, s->net_input, CL_TRUE,
                                            CL_MAP_READ, 0, s->net_input_size,
                                            0, NULL, NULL, &err);
  clFinish(q);
  return net_input_buf;
}

void model_input_free(ModelInput* s) {
  transform_destroy(&s->transform);
  loadyuv_destroy(&s->loadyuv);
}


float sigmoid(float input) {
  return 1 / (1 + expf(-input));
}

float softplus(float input) {
  return log1p(expf(input));
}

void softmax(const float* input, float* output, size_t len) {
  float max_val = -FLT_MAX;
  for(int i = 0; i < len; i++) {
    const float v = input[i];
    if( v > max_val ) {
      max_val = v;
    }
  }

  float denominator = 0;
  for(int i = 0; i < len; i++) {
    float const v = input[i];
    float const v_exp = expf(v - max_val);
    denominator += v_exp;
    output[i] = v_exp;
  }

  const float inv_denominator = 1. / denominator;
  for(int i = 0; i < len; i++) {
    output[i] *= inv_denominator;
  }

}


static cereal_ModelData_PathData_ptr path_to_cereal(struct capn_segment *cs, const PathData data) {
  capn_list32 poly_ptr = capn_new_list32(cs, POLYFIT_DEGREE);
  for (int i=0; i<POLYFIT_DEGREE; i++) {
    capn_set32(poly_ptr, i, capn_from_f32(data.poly[i]));
  }

  cereal_ModelData_PathData_ptr ret = cereal_new_ModelData_PathData(cs);
  struct cereal_ModelData_PathData d = {
    .prob = data.prob,
    .std = data.std,
    .poly = poly_ptr,
  };
  cereal_write_ModelData_PathData(&d, ret);
  return ret;
}

void model_publish(void* sock, uint32_t frame_id,
                   const mat3 transform, const ModelData data) {
  struct capn rc;
  capn_init_malloc(&rc);
  struct capn_segment *cs = capn_root(&rc).seg;

  cereal_ModelData_LeadData_ptr leadp = cereal_new_ModelData_LeadData(cs);
  struct cereal_ModelData_LeadData leadd = (struct cereal_ModelData_LeadData){
    .dist = data.lead.dist,
    .prob = data.lead.prob,
    .std = data.lead.std,
    .relVel = data.lead.rel_v,
    .relVelStd = data.lead.rel_v_std,
  };
  cereal_write_ModelData_LeadData(&leadd, leadp);


  capn_list32 input_transform_ptr = capn_new_list32(cs, 3*3);
  for (int i = 0; i < 3 * 3; i++) {
    capn_set32(input_transform_ptr, i, capn_from_f32(transform.v[i]));
  }

  cereal_ModelData_ModelSettings_ptr settingsp = cereal_new_ModelData_ModelSettings(cs);
  struct cereal_ModelData_ModelSettings settingsd = {
    .inputTransform = input_transform_ptr,
  };
  cereal_write_ModelData_ModelSettings(&settingsd, settingsp);

  cereal_ModelData_ptr modelp = cereal_new_ModelData(cs);
  struct cereal_ModelData modeld = (struct cereal_ModelData){
    .frameId = frame_id,
    .path = path_to_cereal(cs, data.path),
    .leftLane = path_to_cereal(cs, data.left_lane),
    .rightLane = path_to_cereal(cs, data.right_lane),
    .lead = leadp,
    .settings = settingsp,
  };
  cereal_write_ModelData(&modeld, modelp);

  cereal_Event_ptr eventp = cereal_new_Event(cs);
  struct cereal_Event event = {
    .logMonoTime = nanos_since_boot(),
    .valid = true,
    .which = cereal_Event_model,
    .model = modelp,
  };
  cereal_write_Event(&event, eventp);

  capn_setp(capn_root(&rc), 0, eventp.p);
  uint8_t buf[4096];
  ssize_t rs = capn_write_mem(&rc, buf, sizeof(buf), 0);

  zmq_send(sock, buf, rs, ZMQ_DONTWAIT);

  capn_free(&rc);
}
