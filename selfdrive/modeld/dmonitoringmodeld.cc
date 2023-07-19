#include <sys/resource.h>
#include <limits.h>

#include <cstdio>
#include <cstdlib>

#include "cereal/visionipc/visionipc_client.h"
#include "common/swaglog.h"
#include "common/util.h"
#include "selfdrive/modeld/models/dmonitoring.h"

ExitHandler do_exit;

void run_model(DMonitoringModelState &model, VisionIpcClient &vipc_client) {
  PubMaster pm({"driverStateV2"});
  SubMaster sm({"liveCalibration"});
  float calib[CALIB_LEN] = {0};
  // double last = 0;

  while (!do_exit) {
    VisionIpcBufExtra extra = {};
    VisionBuf *buf = vipc_client.recv(&extra);
    if (buf == nullptr) continue;

    sm.update(0);
    if (sm.updated("liveCalibration")) {
      auto calib_msg = sm["liveCalibration"].getLiveCalibration().getRpyCalib();
      for (int i = 0; i < CALIB_LEN; i++) {
        calib[i] = calib_msg[i];
      }
    }

    double t1 = millis_since_boot();
    DMonitoringModelResult model_res = dmonitoring_eval_frame(&model, buf->addr, buf->width, buf->height, buf->stride, buf->uv_offset, calib);
    double t2 = millis_since_boot();

    // send dm packet
    dmonitoring_publish(pm, extra.frame_id, model_res, (t2 - t1) / 1000.0, model.output);

    // printf("dmonitoring process: %.2fms, from last %.2fms\n", t2 - t1, t1 - last);
    // last = t1;
  }
}

int main(int argc, char **argv) {
  setpriority(PRIO_PROCESS, 0, -15);

  // init the models
  DMonitoringModelState model;
  dmonitoring_init(&model);

  LOGW("connecting to driver stream");
  VisionIpcClient vipc_client = VisionIpcClient("camerad", VISION_STREAM_DRIVER, true);
  while (!do_exit && !vipc_client.connect(false)) {
    util::sleep_for(100);
  }

  // run the models
  if (vipc_client.connected) {
    LOGW("connected with buffer size: %d", vipc_client.buffers[0].len);
    run_model(model, vipc_client);
  }

  dmonitoring_free(&model);
  return 0;
}
