#if defined(QCOM) && !defined(QCOM_REPLAY)
#include "cameras/camera_qcom.h"
#elif QCOM2
#include "cameras/camera_qcom2.h"
#elif WEBCAM
#include "cameras/camera_webcam.h"
#else
#include "cameras/camera_frame_stream.h"
#endif

#include "common/util.h"

ExitHandler do_exit;

int main(int argc, char *argv[]) {
  set_realtime_priority(53);
#if defined(QCOM)
  set_core_affinity(2);
#elif defined(QCOM2)
  set_core_affinity(6);
#endif

  CameraServer s;
  cameras_init(&s);
  cameras_open(&s);
  s.vipc_server->start_listener();
  cameras_run(&s);
}
