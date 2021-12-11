#include <sys/resource.h>

#include "cereal/visionipc/visionipc_client.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"
#include "selfdrive/modeld/models/dmonitoring.h"

ExitHandler do_exit;

void run_model(DMModel &model, VisionIpcClient &vipc_client) {
  PubMaster pm({"driverState"});

  while (!do_exit) {
    VisionIpcBufExtra extra = {};
    VisionBuf *buf = vipc_client.recv(&extra);
    if (buf == nullptr) continue;

    DMResult res = model.eval_frame((uint8_t *)buf->addr, buf->width, buf->height);
    model.publish(pm, extra.frame_id, res);
  }
}

int main(int argc, char **argv) {
  setpriority(PRIO_PROCESS, 0, -15);

  VisionIpcClient vipc_client = VisionIpcClient("camerad", VISION_STREAM_DRIVER, true);
  while (!do_exit && !vipc_client.connect(false)) {
    util::sleep_for(100);
  }

  // run the models
  if (vipc_client.connected) {
    const VisionBuf &buf = vipc_client.buffers[0];
    LOGW("connected with buffer size: %d", buf.len);

    DMModel model(buf.width, buf.height);
    run_model(model, vipc_client);
  }

  return 0;
}
