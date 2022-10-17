#include <sys/resource.h>
#include <limits.h>

#include <cstdio>
#include <cstdlib>

#include "cereal/visionipc/visionipc_client.h"
#include "common/swaglog.h"
#include "common/util.h"
#include "selfdrive/modeld/models/nav.h"

ExitHandler do_exit;

void run_model(NavModelState &model, VisionIpcClient &vipc_client) {
  PubMaster pm({"navModelState"});
  double last = 0;

  while (!do_exit) {
    VisionIpcBufExtra extra = {};
    VisionBuf *buf = vipc_client.recv(&extra);
    if (buf == nullptr) continue;

    double t1 = millis_since_boot();
    NavModelResult model_res = navmodel_eval_frame(&model, buf);
    double t2 = millis_since_boot();

    // send dm packet
    navmodel_publish(pm, extra.frame_id, model_res, (t2 - t1) / 1000.0);

    //printf("dmonitoring process: %.2fms, from last %.2fms\n", t2 - t1, t1 - last);
    last = t1;
  }
}

int main(int argc, char **argv) {
  setpriority(PRIO_PROCESS, 0, -15);

  // init the models
  NavModelState model;
  navmodel_init(&model);

  VisionIpcClient vipc_client = VisionIpcClient("camerad", VISION_STREAM_MAP, true);
  while (!do_exit && !vipc_client.connect(false)) {
    util::sleep_for(100);
  }

  // run the models
  if (vipc_client.connected) {
    LOGW("connected with buffer size: %d", vipc_client.buffers[0].len);
    run_model(model, vipc_client);
  }

  navmodel_free(&model);
  return 0;
}
