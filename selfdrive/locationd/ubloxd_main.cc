#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include <ctime>
#include <chrono>
#include <map>
#include <vector>

#include <zmq.h>
#include <capnp/serialize.h>
#include "cereal/gen/cpp/log.capnp.h"

#include "common/params.h"
#include "common/swaglog.h"
#include "common/timing.h"

#include "ublox_msg.h"

volatile sig_atomic_t do_exit = 0; // Flag for process exit on signal

void set_do_exit(int sig) {
  do_exit = 1;
}

using namespace ublox;

int ubloxd_main(poll_ubloxraw_msg_func poll_func, send_gps_event_func send_func) {
  LOGW("starting ubloxd");
  signal(SIGINT, (sighandler_t) set_do_exit);
  signal(SIGTERM, (sighandler_t) set_do_exit);

  UbloxMsgParser parser;
  void *context = zmq_ctx_new();
  void *gpsLocationExternal = zmq_socket(context, ZMQ_PUB);
  zmq_bind(gpsLocationExternal, "tcp://*:8032");
  void *ubloxGnss = zmq_socket(context, ZMQ_PUB);
  zmq_bind(ubloxGnss, "tcp://*:8033");
  // ubloxRaw = 8042
  void *subscriber = zmq_socket(context, ZMQ_SUB);
  zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "", 0);
  zmq_connect(subscriber, "tcp://127.0.0.1:8042");
  while (!do_exit) {
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    int err = poll_func(gpsLocationExternal, ubloxGnss, subscriber, &msg);
    if(err < 0) {
      LOGE_100("zmq_poll error %s in %s", strerror(errno ), __FUNCTION__);
      break;
    } else if(err == 0) {
      continue;
    }
    // format for board, make copy due to alignment issues, will be freed on out of scope
    auto amsg = kj::heapArray<capnp::word>((zmq_msg_size(&msg) / sizeof(capnp::word)) + 1);
    memcpy(amsg.begin(), zmq_msg_data(&msg), zmq_msg_size(&msg));
    capnp::FlatArrayMessageReader cmsg(amsg);
    cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();
    const uint8_t *data = event.getUbloxRaw().begin();
    size_t len = event.getUbloxRaw().size();
    size_t bytes_consumed = 0;
    while(bytes_consumed < len && !do_exit) {
      size_t bytes_consumed_this_time = 0U;
      if(parser.add_data(data + bytes_consumed, (uint32_t)(len - bytes_consumed), bytes_consumed_this_time)) {
        // New message available
        if(parser.msg_class() == CLASS_NAV) {
          if(parser.msg_id() == MSG_NAV_PVT) {
            //LOGD("MSG_NAV_PVT");
            auto words = parser.gen_solution();
            if(words.size() > 0) {
              auto bytes = words.asBytes();
              send_func(parser.msg_class(), parser.msg_id(), gpsLocationExternal, bytes.begin(), bytes.size(), 0);
            }
          } else
            LOGW("Unknown nav msg id: 0x%02X", parser.msg_id());
        } else if(parser.msg_class() == CLASS_RXM) {
          if(parser.msg_id() == MSG_RXM_RAW) {
            //LOGD("MSG_RXM_RAW");
            auto words = parser.gen_raw();
            if(words.size() > 0) {
              auto bytes = words.asBytes();
              send_func(parser.msg_class(), parser.msg_id(), ubloxGnss, bytes.begin(), bytes.size(), 0);
            }
          } else if(parser.msg_id() == MSG_RXM_SFRBX) {
            //LOGD("MSG_RXM_SFRBX");
            auto words = parser.gen_nav_data();
            if(words.size() > 0) {
              auto bytes = words.asBytes();
              send_func(parser.msg_class(), parser.msg_id(), ubloxGnss, bytes.begin(), bytes.size(), 0);
            }
          } else
            LOGW("Unknown rxm msg id: 0x%02X", parser.msg_id());
        } else
          LOGW("Unknown msg class: 0x%02X", parser.msg_class());
        parser.reset();
      }
      bytes_consumed += bytes_consumed_this_time;
    }
    zmq_msg_close(&msg);
  }
  zmq_close(subscriber);
  zmq_close(gpsLocationExternal);
  zmq_close(ubloxGnss);
  zmq_ctx_destroy(context);
  return 0;
}
