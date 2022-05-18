#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>

// Apple doesn't have timerfd
#ifdef __APPLE__
#include <thread>
#else
#include <sys/timerfd.h>
#endif

#include <cassert>
#include <chrono>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"

ExitHandler do_exit;

int main() {
  setpriority(PRIO_PROCESS, 0, -13);
  PubMaster pm({"clocks"});

#ifndef __APPLE__
  int timerfd = timerfd_create(CLOCK_BOOTTIME, 0);
  assert(timerfd >= 0);

  struct itimerspec spec = {0};
  spec.it_interval.tv_sec = 1;
  spec.it_interval.tv_nsec = 0;
  spec.it_value.tv_sec = 1;
  spec.it_value.tv_nsec = 0;

  int err = timerfd_settime(timerfd, 0, &spec, 0);
  assert(err == 0);

  uint64_t expirations = 0;
  while (!do_exit && (err = read(timerfd, &expirations, sizeof(expirations)))) {
    if (err < 0) {
      if (errno == EINTR) continue;
      break;
    }
#else
  // Just run at 1Hz on apple
  while (!do_exit) {
    util::sleep_for(1000);
#endif

    uint64_t boottime = nanos_since_boot();
    uint64_t monotonic = nanos_monotonic();
    uint64_t monotonic_raw = nanos_monotonic_raw();
    uint64_t wall_time = nanos_since_epoch();

    MessageBuilder msg;
    auto clocks = msg.initEvent().initClocks();

    clocks.setBootTimeNanos(boottime);
    clocks.setMonotonicNanos(monotonic);
    clocks.setMonotonicRawNanos(monotonic_raw);
    clocks.setWallTimeNanos(wall_time);

    pm.send("clocks", msg);
  }

#ifndef __APPLE__
  close(timerfd);
#endif
  return 0;
}
