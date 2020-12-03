#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <cassert>
#include <sys/resource.h>

#include "common/visionbuf.h"
#include "common/visionipc.h"
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
  int err;
  setpriority(PRIO_PROCESS, 0, -15);

  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  PubMaster pm({"driverState"});

  // init the models
  DMonitoringModelState dmonitoringmodel;
  dmonitoring_init(&dmonitoringmodel);

  // loop
  VisionStream stream;
  while (!do_exit) {
    VisionStreamBufs buf_info;
    err = visionstream_init(&stream, VISION_STREAM_YUV_FRONT, true, &buf_info);
    if (err) {
      printf("visionstream connect fail\n");
      usleep(100000);
      continue;
    }
    LOGW("connected with buffer size: %d", buf_info.buf_len);

    double last = 0;
    while (!do_exit) {
      VIPCBuf *buf;
      VIPCBufExtra extra;
      buf = visionstream_get(&stream, &extra);
      if (buf == NULL) {
        printf("visionstream get failed\n");
        break;
      }

      double t1 = millis_since_boot();
      DMonitoringResult res = dmonitoring_eval_frame(&dmonitoringmodel, buf->addr, buf_info.width, buf_info.height);
      double t2 = millis_since_boot();

      // send dm packet
      const float* raw_pred_ptr = send_raw_pred ? (const float *)dmonitoringmodel.output : nullptr;
      dmonitoring_publish(pm, extra.frame_id, res, raw_pred_ptr, (t2-t1)/1000.0);

      LOGD("dmonitoring process: %.2fms, from last %.2fms", t2-t1, t1-last);
      last = t1;
    }
    visionstream_destroy(&stream);
  }

  dmonitoring_free(&dmonitoringmodel);

  return 0;
}
