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

void party() {
  CLContext ctx = CLContext::getDefault();
  VisionIpcServer vipc_server("camerad", ctx.device_id, ctx.context);

  MultiCameraState cameras = {};
  cameras_init(&vipc_server, &cameras);
  cameras_open(&cameras);

  vipc_server.start_listener();

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

  party();
}
