#include <thread>
#include <stdio.h>
#include <poll.h>
#include <assert.h>
#include <unistd.h>
#include <sys/socket.h>

#if defined(QCOM) && !defined(QCOM_REPLAY)
#include "cameras/camera_qcom.h"
#elif QCOM2
#include "cameras/camera_qcom2.h"
#elif WEBCAM
#include "cameras/camera_webcam.h"
#else
#include "cameras/camera_frame_stream.h"
#endif

#include <libyuv.h>

#include "clutil.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/util.h"
#include "visionipc_server.h"

ExitHandler do_exit;

void party(cl_device_id device_id, cl_context context) {
  MultiCameraState cameras = {};
  VisionIpcServer vipc_server("camerad", device_id, context);

  cameras_init(&vipc_server, &cameras, device_id, context);
  cameras_open(&cameras);

  vipc_server.start_listener();

  // priority for cameras
  int err = set_realtime_priority(53);
  LOG("setpriority returns %d", err);


  cameras_run(&cameras);
}

#ifdef QCOM
#include "CL/cl_ext_qcom.h"
#endif

int main(int argc, char *argv[]) {
  set_realtime_priority(53);
#if defined(QCOM)
  set_core_affinity(2);
#elif defined(QCOM2)
  set_core_affinity(6);
#endif

  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);

   // TODO: do this for QCOM2 too
#if defined(QCOM)
  const cl_context_properties props[] = {CL_CONTEXT_PRIORITY_HINT_QCOM, CL_PRIORITY_HINT_HIGH_QCOM, 0};
  cl_context context = CL_CHECK_ERR(clCreateContext(props, 1, &device_id, NULL, NULL, &err));
#else
  cl_context context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));
#endif

  party(device_id, context);

  CL_CHECK(clReleaseContext(context));
}
