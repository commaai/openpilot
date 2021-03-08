#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <assert.h>
#include <math.h>
#include <ctime>
#include <chrono>

#include "messaging.hpp"
#include "common/util.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/timing.h"

#include "ublox_msg.h"

ExitHandler do_exit;
using namespace ublox;
int ubloxd_main(poll_ubloxraw_msg_func poll_func, send_gps_event_func send_func) {
  LOGW("starting ubloxd");
  kj::Array<capnp::word> buf = kj::heapArray<capnp::word>(1024);
  UbloxMsgParser parser;

  Context * context = Context::create();
  SubSocket * subscriber = SubSocket::create(context, "ubloxRaw");
  assert(subscriber != NULL);
  subscriber->setTimeout(100);

  PubMaster pm({"ubloxGnss", "gpsLocationExternal"});

  while (!do_exit) {
    Message * msg = subscriber->receive();
    if (!msg){
      if (errno == EINTR) {
        do_exit = true;
      }
      continue;
    }
    const size_t size = (msg->getSize() / sizeof(capnp::word)) + 1;
    if (buf.size() < size) {
      buf = kj::heapArray<capnp::word>(size);
    }
    memcpy(buf.begin(), msg->getData(), msg->getSize());

    capnp::FlatArrayMessageReader cmsg(buf.slice(0, size));
    cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();
    auto ubloxRaw = event.getUbloxRaw();

    const uint8_t *data = ubloxRaw.begin();
    size_t len = ubloxRaw.size();
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
              pm.send("gpsLocationExternal", bytes.begin(), bytes.size());
            }
          } else
            LOGW("Unknown nav msg id: 0x%02X", parser.msg_id());
        } else if(parser.msg_class() == CLASS_RXM) {
          if(parser.msg_id() == MSG_RXM_RAW) {
            //LOGD("MSG_RXM_RAW");
            auto words = parser.gen_raw();
            if(words.size() > 0) {
              auto bytes = words.asBytes();
              pm.send("ubloxGnss", bytes.begin(), bytes.size());
            }
          } else if(parser.msg_id() == MSG_RXM_SFRBX) {
            //LOGD("MSG_RXM_SFRBX");
            auto words = parser.gen_nav_data();
            if(words.size() > 0) {
              auto bytes = words.asBytes();
              pm.send("ubloxGnss", bytes.begin(), bytes.size());
            }
          } else
            LOGW("Unknown rxm msg id: 0x%02X", parser.msg_id());
        } else if(parser.msg_class() == CLASS_MON) {
          if(parser.msg_id() == MSG_MON_HW) {
            //LOGD("MSG_MON_HW");
            auto words = parser.gen_mon_hw();
            if(words.size() > 0) {
              auto bytes = words.asBytes();
              pm.send("ubloxGnss", bytes.begin(), bytes.size());
            }
          } else if(parser.msg_id() == MSG_MON_HW2) {
            //LOGD("MSG_MON_HW2");
            auto words = parser.gen_mon_hw2();
            if(words.size() > 0) {
              auto bytes = words.asBytes();
              pm.send("ubloxGnss", bytes.begin(), bytes.size());
            }
          } else {
            LOGW("Unknown mon msg id: 0x%02X", parser.msg_id());
          }
        } else
          LOGW("Unknown msg class: 0x%02X", parser.msg_class());
        parser.reset();
      }
      bytes_consumed += bytes_consumed_this_time;
    }
    delete msg;
  }

  delete subscriber;
  delete context;

  return 0;
}
