#include <cassert>

#include <kaitai/kaitaistream.h>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"
#include "selfdrive/locationd/ublox_msg.h"

ExitHandler do_exit;
using namespace ublox;

int main() {
  LOGW("starting ubloxd");
  AlignedBuffer aligned_buf;
  UbloxMsgParser parser;

  PubMaster pm({"ubloxGnss", "gpsLocationExternal"});

  Context * context = Context::create();
  SubSocket * subscriber = SubSocket::create(context, "ubloxRaw");
  assert(subscriber != NULL);
  subscriber->setTimeout(100);


  while (!do_exit) {
    Message * msg = subscriber->receive();
    if (!msg) {
      if (errno == EINTR) {
        do_exit = true;
      }
      continue;
    }

    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(msg));
    cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();
    auto ubloxRaw = event.getUbloxRaw();

    const uint8_t *data = ubloxRaw.begin();
    size_t len = ubloxRaw.size();
    size_t bytes_consumed = 0;

    while(bytes_consumed < len && !do_exit) {
      size_t bytes_consumed_this_time = 0U;
      if(parser.add_data(data + bytes_consumed, (uint32_t)(len - bytes_consumed), bytes_consumed_this_time)) {

        try {
          auto ublox_msg = parser.gen_msg();
          if (ublox_msg.second.size() > 0) {
            auto bytes = ublox_msg.second.asBytes();
            pm.send(ublox_msg.first.c_str(), bytes.begin(), bytes.size());
          }
        } catch (const std::exception& e) {
          LOGE("Error parsing ublox message %s", e.what());
        }

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
