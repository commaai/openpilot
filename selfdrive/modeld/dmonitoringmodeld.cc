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

#ifdef QCOM2
  set_core_affinity(5);
#endif

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
      dmonitoring_publish(pm, extra.frame_id, res);

      LOGD("dmonitoring process: %.2fms, from last %.2fms", t2-t1, t1-last);
      last = t1;
#ifdef QCOM2
      // this makes it run at about 2.7Hz on tici CPU to deal with modeld lags
      // TODO: DSP needs to be freed (again)
      usleep(250000);
#endif
    }
    visionstream_destroy(&stream);
  }

  dmonitoring_free(&dmonitoringmodel);

  return 0;
}
