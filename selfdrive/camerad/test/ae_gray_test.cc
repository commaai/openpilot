// unittest for
// void set_exposure_target(CameraState *c, int x_start, int x_end, int x_skip, int y_start, int y_end, int y_skip)

#include <assert.h>
#include "clutil.h"

#include "visionipc_server.h"
#include "selfdrive/camerad/cameras/camera_common.h"

#define W 240
#define H 160

typedef struct CameraState {
  CameraInfo ci;
  CameraBuf buf;
} CameraState;

// generic camera state
void camera_autoexposure(CameraState *s, float grey_frac) {}

int main() {
  // set up fake camera
  CameraState cs = {};
  CameraState *s = &cs;

  // copied from main.cc
  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  #if defined(QCOM)
    const cl_context_properties props[] = {CL_CONTEXT_PRIORITY_HINT_QCOM, CL_PRIORITY_HINT_HIGH_QCOM, 0};
    cl_context context = CL_CHECK_ERR(clCreateContext(props, 1, &device_id, NULL, NULL, &err));
  #else
    cl_context context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));
  #endif
  VisionIpcServer vipc_server("camerad-test", device_id, context);

  CameraInfo fci = (struct CameraInfo) {
    .frame_width = W,
    .frame_height = H,
    .frame_stride = W*3,
  };

  s->ci = fci;
  assert(s->ci.frame_width != 0);
  s->buf.init(device_id, context, s, &vipc_server, 1, VISION_STREAM_RGB_BACK, VISION_STREAM_YUV_BACK);

  // calculate EV
  printf("hello\n");
  // check output
  return 0;
}