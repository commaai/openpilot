#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <cassert>
#include <sys/resource.h>

#include "visionbuf.h"
#include "visionipc_client.h"
#include "common/swaglog.h"

#include "models/dmonitoring.h"

#ifndef PATH_MAX
#include <linux/limits.h>
#endif


volatile sig_atomic_t do_exit = 0;

static void set_do_exit(int sig) {
  do_exit = 1;
}

int main(int argc, char **argv) {
  setpriority(PRIO_PROCESS, 0, -15);

  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  PubMaster pm({"driverState"});

  // init the models
  DMonitoringModelState dmonitoringmodel;
  dmonitoring_init(&dmonitoringmodel);

  auto vipc_client = VisionIpcClient("camerad", VISION_STREAM_YUV_FRONT, false);
  while (!do_exit) {
    LOGW("connected with buffer size: %d", vipc_client.buffers[0].len);

    double last = 0;
    while (!do_exit) {
      VisionBuf *buf = vipc_client.recv();
      // TODO receive extra data
      VIPCBufExtra extra = {0};

      double t1 = millis_since_boot();
      DMonitoringResult res = dmonitoring_eval_frame(&dmonitoringmodel, buf->addr, buf->width, buf->height);
      double t2 = millis_since_boot();

      // send dm packet
      const float* raw_pred_ptr = send_raw_pred ? (const float *)dmonitoringmodel.output : nullptr;
      dmonitoring_publish(pm, extra.frame_id, res, raw_pred_ptr, (t2-t1)/1000.0);

      LOGD("dmonitoring process: %.2fms, from last %.2fms", t2-t1, t1-last);
      last = t1;
    }
  }

  dmonitoring_free(&dmonitoringmodel);

  return 0;
}
