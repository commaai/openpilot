#include "common/ratekeeper.h"

#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"

RateKeeper::RateKeeper(const std::string &name, float rate, float print_delay_threshold)
    : name(name),
      print_delay_threshold(std::max(0.f, print_delay_threshold)) {
  interval = 1 / rate;
  last_monitor_time = seconds_since_boot();
  next_frame_time = last_monitor_time + interval;
}

bool RateKeeper::keepTime() {
  bool lagged = monitorTime();
  if (remaining_ > 0) {
    util::sleep_for(remaining_ * 1000);
  }
  return lagged;
}

bool RateKeeper::monitorTime() {
  ++frame_;
  last_monitor_time = seconds_since_boot();
  remaining_ = next_frame_time - last_monitor_time;

  bool lagged = remaining_ < 0;
  if (lagged) {
    if (print_delay_threshold > 0 && remaining_ < -print_delay_threshold) {
      LOGW("%s lagging by %.2f ms", name.c_str(), -remaining_ * 1000);
    }
    next_frame_time = last_monitor_time + interval;
  } else {
    next_frame_time += interval;
  }
  return lagged;
}
