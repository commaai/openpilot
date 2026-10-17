#include "common/tests/native_test.h"
#include "tools/replay/replay.h"
#include "tools/replay/route.h"

void test_route_ranges() {
  struct Range {
    const char *suffix;
    int begin;
    int end;
  };
  const Range ranges[] = {
    {"", 0, -1},
    {"/:15", 0, 15},
    {"/:0", 0, 0},
    {"/0:15", 0, 15},
    {"/5:15", 5, 15},
    {"/5:", 5, -1},
    {"/5", 5, 5},
    {"--5", 5, -1},
  };
  for (const std::string route : {DEMO_ROUTE, "64e16c4237493597/00000033--332a5f9d82"}) {
    const auto full = Route::parseRoute(route);
    REQUIRE(!full.str.empty());
    for (const auto &range : ranges) {
      const auto parsed = Route::parseRoute(route + range.suffix);
      REQUIRE(parsed.str == full.str);
      REQUIRE(parsed.dongle_id == full.dongle_id);
      REQUIRE(parsed.timestamp == full.timestamp);
      REQUIRE(parsed.begin_segment == range.begin);
      REQUIRE(parsed.end_segment == range.end);
    }
  }
}

int main() {
  return run_native_test(test_route_ranges);
}
