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

#include "messaging.hpp"
#include <capnp/serialize.h>
#include "cereal/gen/cpp/log.capnp.h"

#include "common/util.h"
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

  Context * c = Context::create();
  PubSocket * gpsLocationExternal = PubSocket::create(c, "gpsLocationExternal");
  PubSocket * ubloxGnss = PubSocket::create(c, "ubloxGnss");
  SubSocket * ubloxRaw = SubSocket::create(c, "ubloxRaw");

  assert(gpsLocationExternal != NULL);
  assert(ubloxGnss != NULL);
  assert(ubloxRaw != NULL);

  Poller * poller = Poller::create({ubloxRaw});


  while (!do_exit) {
    Message * msg = poll_func(poller);
    if (msg == NULL) continue;

    auto amsg = kj::heapArray<capnp::word>((msg->getSize() / sizeof(capnp::word)) + 1);
    memcpy(amsg.begin(), msg->getData(), msg->getSize());

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
              send_func(gpsLocationExternal, bytes.begin(), bytes.size());
            }
          } else
            LOGW("Unknown nav msg id: 0x%02X", parser.msg_id());
        } else if(parser.msg_class() == CLASS_RXM) {
          if(parser.msg_id() == MSG_RXM_RAW) {
            //LOGD("MSG_RXM_RAW");
            auto words = parser.gen_raw();
            if(words.size() > 0) {
              auto bytes = words.asBytes();
              send_func(ubloxGnss, bytes.begin(), bytes.size());
            }
          } else if(parser.msg_id() == MSG_RXM_SFRBX) {
            //LOGD("MSG_RXM_SFRBX");
            auto words = parser.gen_nav_data();
            if(words.size() > 0) {
              auto bytes = words.asBytes();
              send_func(ubloxGnss, bytes.begin(), bytes.size());
            }
          } else
            LOGW("Unknown rxm msg id: 0x%02X", parser.msg_id());
        } else
          LOGW("Unknown msg class: 0x%02X", parser.msg_class());
        parser.reset();
      }
      bytes_consumed += bytes_consumed_this_time;
    }
    delete msg;
  }

  delete poller;
  delete ubloxRaw;
  delete ubloxGnss;
  delete gpsLocationExternal;
  delete c;

  return 0;
}
