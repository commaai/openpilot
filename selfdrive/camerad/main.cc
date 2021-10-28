#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <thread>

#include "libyuv.h"

#include "cereal/visionipc/visionipc_server.h"
#include "selfdrive/common/clutil.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"

#ifdef QCOM
#include "selfdrive/camerad/cameras/camera_qcom.h"
#elif QCOM2
#include "selfdrive/camerad/cameras/camera_qcom2.h"
#elif WEBCAM
#include "selfdrive/camerad/cameras/camera_webcam.h"
#else
#include "selfdrive/camerad/cameras/camera_replay.h"
#endif

ExitHandler do_exit;

void party(cl_device_id device_id, cl_context context) {
  MultiCameraState cameras = {};
  VisionIpcServer vipc_server("camerad", device_id, context);

  cameras_init(&vipc_server, &cameras, device_id, context);
  cameras_open(&cameras);

  vipc_server.start_listener();

  cameras_run(&cameras);
}

#ifdef QCOM
#include "CL/cl_ext_qcom.h"
#endif

int main(int argc, char *argv[]) {
  if (!Hardware::PC()) {
    int ret;
    ret = set_realtime_priority(53);
    assert(ret == 0);
    ret = set_core_affinity({Hardware::EON() ? 2 : 6});
    assert(ret == 0 || Params().getBool("IsOffroad")); // failure ok while offroad due to offlining cores
  }

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
