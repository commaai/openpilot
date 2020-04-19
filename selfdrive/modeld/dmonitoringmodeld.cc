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
  set_realtime_priority(1);

  // messaging
  Context *msg_context = Context::create();
  PubSocket *dmonitoring_sock = PubSocket::create(msg_context, "driverState");
  SubSocket *dmonstate_sock = SubSocket::create(msg_context, "dMonitoringState", "127.0.0.1", true);
  assert(dmonstate_sock != NULL);

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
        break;
      }
      //printf("frame_id: %d %dx%d\n", extra.frame_id, buf_info.width, buf_info.height);
      if (!dmonitoringmodel.is_rhd_checked) {
        if (chk_counter >= RHD_CHECK_INTERVAL) {
          Message *msg = dmonstate_sock->receive(true);
          if (msg != NULL) {
            auto amsg = kj::heapArray<capnp::word>((msg->getSize() / sizeof(capnp::word)) + 1);
            memcpy(amsg.begin(), msg->getData(), msg->getSize());

            capnp::FlatArrayMessageReader cmsg(amsg);
            cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();

            dmonitoringmodel.is_rhd = event.getDMonitoringState().getIsRHD();
            dmonitoringmodel.is_rhd_checked = event.getDMonitoringState().getRhdChecked();
          }
          chk_counter = 0;
        }
        chk_counter += 1;
      }

      double t1 = millis_since_boot();

      DMonitoringResult res = dmonitoring_eval_frame(&dmonitoringmodel, buf->addr, buf_info.width, buf_info.height);

      double t2 = millis_since_boot();

      // send dm packet
      dmonitoring_publish(dmonitoring_sock, extra.frame_id, res);

      LOGD("dmonitoring process: %.2fms, from last %.2fms", t2-t1, t1-last);
      last = t1;
    }

  }

  visionstream_destroy(&stream);

  delete dmonitoring_sock;
  delete msg_context;
  dmonitoring_free(&dmonitoringmodel);

  return 0;
}
