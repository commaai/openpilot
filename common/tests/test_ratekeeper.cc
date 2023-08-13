#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "common/ratekeeper.h"
#include "common/timing.h"
#include "common/util.h"

int random_int(int min, int max) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(min, max);
  return dist(rng);
}

TEST_CASE("RateKeeper") {
  float freq = GENERATE(10, 50, 100);
  RateKeeper rk("Test RateKeeper", freq);
  for (int i = 0; i < freq; ++i) {
    double begin = seconds_since_boot();
    util::sleep_for(random_int(0, 1000.0 / freq - 1));
    bool lagged = rk.keepTime();
    REQUIRE(std::abs(seconds_since_boot() - begin - (1 / freq)) < 1e-3);
    REQUIRE(lagged == false);
  }
}
