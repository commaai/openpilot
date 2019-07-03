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

const long ZMQ_POLL_TIMEOUT = 1000; // In miliseconds

int poll_ubloxraw_msg(void *gpsLocationExternal, void *ubloxGnss, void *subscriber, zmq_msg_t *msg) {
  int err;
  zmq_pollitem_t item = {.socket = subscriber, .events = ZMQ_POLLIN};
  err = zmq_poll (&item, 1, ZMQ_POLL_TIMEOUT);
  if(err <= 0)
    return err;
  return zmq_msg_recv(msg, subscriber, 0);
}

int send_gps_event(uint8_t msg_cls, uint8_t msg_id, void *s, const void *buf, size_t len, int flags) {
  return zmq_send(s, buf, len, flags);
}

int main() {
  return ubloxd_main(poll_ubloxraw_msg, send_gps_event);
}
