#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <assert.h>
#include <math.h>
#include <ctime>
#include <chrono>

#include "messaging.hpp"

#include "common/params.h"
#include "common/swaglog.h"
#include "common/timing.h"

#include "ublox_msg.h"

const long ZMQ_POLL_TIMEOUT = 1000; // In miliseconds

Message * poll_ubloxraw_msg(Poller * poller) {
  auto p = poller->poll(ZMQ_POLL_TIMEOUT);

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
