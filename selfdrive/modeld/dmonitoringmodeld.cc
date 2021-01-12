#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cassert>
#include <sys/resource.h>

#include "visionbuf.h"
#include "visionipc_client.h"
#include "common/swaglog.h"
#include "common/util.h"

#include "models/dmonitoring.h"

#ifndef PATH_MAX
#include <linux/limits.h>
#endif


ExitHandler do_exit;

int main(int argc, char **argv) {
  setpriority(PRIO_PROCESS, 0, -15);

  PubMaster pm({"driverState"});

  // init the models
  DMonitoringModelState dmonitoringmodel;
  dmonitoring_init(&dmonitoringmodel);

  VisionIpcClient vipc_client = VisionIpcClient("camerad", VISION_STREAM_YUV_FRONT, true);
  while (!do_exit){
    if (!vipc_client.connect(false)){
      util::sleep_for(100);
      continue;
    }
    break;
  }

  while (!do_exit) {
    LOGW("connected with buffer size: %d", vipc_client.buffers[0].len);

    double last = 0;
    while (!do_exit) {
      VisionIpcBufExtra extra = {0};
      VisionBuf *buf = vipc_client.recv(&extra);
      if (buf == nullptr){
        continue;
      }

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
