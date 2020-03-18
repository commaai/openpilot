#include <stdio.h>
#include <stdint.h>
#include <sys/resource.h>
#include <sys/timerfd.h>
#include <sys/time.h>
#include <utils/Timers.h>
#include <capnp/serialize.h>
#include "messaging.hpp"
#include "common/timing.h"
#include "cereal/gen/cpp/log.capnp.h"

namespace {
  int64_t arm_cntpct() {
    int64_t v;
    asm volatile("mrs %0, cntpct_el0" : "=r"(v));
    return v;
  }
}

int main() {
  setpriority(PRIO_PROCESS, 0, -13);

  int err = 0;
  Context *context = Context::create();

  PubSocket* clock_publisher = PubSocket::create(context, "clocks");
  assert(clock_publisher != NULL);

  int timerfd = timerfd_create(CLOCK_BOOTTIME, 0);
  assert(timerfd >= 0);

  struct itimerspec spec = {0};
  spec.it_interval.tv_sec = 1;
  spec.it_interval.tv_nsec = 0;
  spec.it_value.tv_sec = 1;
  spec.it_value.tv_nsec = 0;

  err = timerfd_settime(timerfd, 0, &spec, 0);
  assert(err == 0);

  uint64_t expirations = 0;
  while ((err = read(timerfd, &expirations, sizeof(expirations)))) {
    if (err < 0) break;

    uint64_t boottime = nanos_since_boot();
    uint64_t monotonic = nanos_monotonic();
    uint64_t monotonic_raw = nanos_monotonic_raw();
    uint64_t wall_time = nanos_since_epoch();

    uint64_t modem_uptime_v = arm_cntpct() / 19200ULL; // 19.2 mhz clock

    capnp::MallocMessageBuilder msg;
    cereal::Event::Builder event = msg.initRoot<cereal::Event>();
    event.setLogMonoTime(boottime);
    auto clocks = event.initClocks();

    clocks.setBootTimeNanos(boottime);
    clocks.setMonotonicNanos(monotonic);
    clocks.setMonotonicRawNanos(monotonic_raw);
    clocks.setWallTimeNanos(wall_time);
    clocks.setModemUptimeMillis(modem_uptime_v);

    auto words = capnp::messageToFlatArray(msg);
    auto bytes = words.asBytes();
    clock_publisher->send((char*)bytes.begin(), bytes.size());
  }

  close(timerfd);
  delete clock_publisher;

  return 0;
}