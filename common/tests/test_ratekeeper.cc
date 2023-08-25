#include "catch2/catch.hpp"
#include "common/ratekeeper.h"
#include "common/timing.h"
#include "common/util.h"

TEST_CASE("RateKeeper") {
  float freq = GENERATE(10, 50, 100);
  RateKeeper rk("Test RateKeeper", freq);
  for (int i = 0; i < freq; ++i) {
    double begin = seconds_since_boot();
    util::sleep_for(util::random_int(0, 1000.0 / freq - 1));
    bool lagged = rk.keepTime();
    REQUIRE(std::abs(seconds_since_boot() - begin - (1 / freq)) < 1e-3);
    REQUIRE(lagged == false);
  }
}
