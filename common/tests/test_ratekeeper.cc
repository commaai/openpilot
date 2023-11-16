#include "catch2/catch.hpp"
#include "common/ratekeeper.h"
#include "common/timing.h"
#include "common/util.h"

TEST_CASE("RateKeeper") {
  float freq = GENERATE(10, 50, 100);
  RateKeeper rk("Test RateKeeper", freq);

  int lags = 0;
  int bad_keep_times = 0;
  for (int i = 0; i < freq; ++i) {
    double begin = seconds_since_boot();
    util::sleep_for(util::random_int(0, 1000.0 / freq - 1));
    bool lagged = rk.keepTime();
    lags += lagged;
    bad_keep_times += (seconds_since_boot() - begin - (1 / freq)) > 1e-3;
  }

  // need a tolerance here due to scheduling
  REQUIRE(lags < 5);
  REQUIRE(bad_keep_times < 5);
}
