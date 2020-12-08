#include <thread>
#include <stdio.h>
#include <signal.h>
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

volatile sig_atomic_t do_exit = 0;

static void set_do_exit(int sig) {
  do_exit = 1;
}

void party(cl_device_id device_id, cl_context context) {
  MultiCameraState cameras = {};
  VisionIpcServer vipc_server("camerad", device_id, context);

  cameras_init(&vipc_server, &cameras, device_id, context);
  cameras_open(&cameras);

  vipc_server.start_listener();

  // priority for cameras
  int err = set_realtime_priority(51);
  LOG("setpriority returns %d", err);


  cameras_run(&cameras);
}

int main(int argc, char *argv[]) {
  set_realtime_priority(51);
#if defined(QCOM)
  set_core_affinity(2);
#elif defined(QCOM2)
  set_core_affinity(6);
#endif

  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  cl_context context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));

  party(device_id, context);

  CL_CHECK(clReleaseContext(context));
}
