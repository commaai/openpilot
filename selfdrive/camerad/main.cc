#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"

#ifdef QCOM
#include "selfdrive/camerad/cameras/camera_qcom.h"
#elif QCOM2
#include "selfdrive/camerad/cameras/camera_qcom2.h"
#elif WEBCAM
#include "selfdrive/camerad/cameras/camera_webcam.h"
#else
#include "selfdrive/camerad/cameras/camera_frame_stream.h"
#endif

ExitHandler do_exit;

void party() {
  MultiCameraState server;
  server.init();

  cameras_init(&server);
  cameras_open(&server);

  server.start();
  cameras_run(&server);
}

int main(int argc, char *argv[]) {
  set_realtime_priority(53);
  if (Hardware::EON()) {
    set_core_affinity(2);
  } else if (Hardware::TICI()) {
    set_core_affinity(6);
  }

  party();
}
