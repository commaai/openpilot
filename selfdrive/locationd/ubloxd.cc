#include <stdio.h>

#include "messaging.hpp"

#include "ublox_msg.h"

Message * poll_ubloxraw_msg(Poller * poller) {
  auto p = poller->poll(1000);

  if (p.size()) {
    return p[0]->receive();
  } else {
    return NULL;
  }
}

int send_gps_event(PubSocket *s, const void *buf, size_t len) {
  return s->send((char*)buf, len);
}

int main() {
  return ubloxd_main(poll_ubloxraw_msg, send_gps_event);
}