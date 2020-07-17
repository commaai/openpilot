#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <cassert>

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
  set_realtime_priority(51);

  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  // messaging
  SubMaster sm({"dMonitoringState"});
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
    int chk_counter = 0;
    while (!do_exit) {
      VIPCBuf *buf;
      VIPCBufExtra extra;
      buf = visionstream_get(&stream, &extra);
      if (buf == NULL) {
        printf("visionstream get failed\n");
        visionstream_destroy(&stream);
        break;
      }
      //printf("frame_id: %d %dx%d\n", extra.frame_id, buf_info.width, buf_info.height);
      if (!dmonitoringmodel.is_rhd_checked) {
        if (chk_counter >= RHD_CHECK_INTERVAL) {
          if (sm.update(0) > 0) {
            auto state = sm["dMonitoringState"].getDMonitoringState();
            dmonitoringmodel.is_rhd = state.getIsRHD();
            dmonitoringmodel.is_rhd_checked = state.getRhdChecked();
          }
          chk_counter = 0;
        }
        chk_counter += 1;
      }

      double t1 = millis_since_boot();

      DMonitoringResult res = dmonitoring_eval_frame(&dmonitoringmodel, buf->addr, buf_info.width, buf_info.height);

      double t2 = millis_since_boot();

      // send dm packet
      dmonitoring_publish(pm, extra.frame_id, res);

      LOGD("dmonitoring process: %.2fms, from last %.2fms", t2-t1, t1-last);
      last = t1;
    }

  }

  visionstream_destroy(&stream);

  dmonitoring_free(&dmonitoringmodel);

  return 0;
}
