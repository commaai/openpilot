#include "catch2/catch.hpp"

#include "tools/loggy/backend/store.h"

#include <string>
#include <vector>

namespace {

loggy::StoreBatch makeBatch() {
  loggy::StoreBatch batch;
  batch.segment = 0;
  batch.coverage = {{0.0, 3.0}};
  batch.series.push_back({
    .path = "/carState/vEgo",
    .range = {0.0, 3.0},
    .points = {{0.0, 1.0}, {1.0, 2.0}, {2.0, 3.0}},
    .segment = 0,
  });
  batch.series.push_back({
    .path = "/gpsLocationExternal/latitude",
    .range = {0.0, 3.0},
    .points = {{0.0, 37.0000}, {1.0, 37.0005}, {2.0, 37.0010}},
    .segment = 0,
  });
  batch.series.push_back({
    .path = "/gpsLocationExternal/longitude",
    .range = {0.0, 3.0},
    .points = {{0.0, -122.0000}, {1.0, -121.9995}, {2.0, -121.9990}},
    .segment = 0,
  });
  batch.series.push_back({
    .path = "/gpsLocationExternal/hasFix",
    .range = {0.0, 3.0},
    .points = {{0.0, 1.0}, {1.0, 1.0}, {2.0, 1.0}},
    .segment = 0,
  });
  batch.series.push_back({
    .path = "/gpsLocationExternal/bearingDeg",
    .range = {0.0, 3.0},
    .points = {{0.0, 90.0}, {1.0, 91.0}, {2.0, 92.0}},
    .segment = 0,
  });
  batch.can_events.push_back({
    .id = loggy::MessageId{.source = 0, .address = 0x123},
    .range = {0.0, 3.0},
    .events = {
      {.mono_time = 0.0, .bus_time = 10, .data = {0x00, 0xF0}},
      {.mono_time = 1.0, .bus_time = 11, .data = {0x01, 0xF0}},
      {.mono_time = 2.0, .bus_time = 12, .data = {0x03, 0x70}},
    },
    .segment = 0,
  });
  batch.can_events.push_back({
    .id = loggy::MessageId{.source = 1, .address = 0x456},
    .range = {0.5, 2.5},
    .events = {
      {.mono_time = 0.5, .bus_time = 20, .data = {0x10, 0x20, 0x30}},
      {.mono_time = 1.5, .bus_time = 21, .data = {0x11, 0x20, 0x30}},
      {.mono_time = 2.5, .bus_time = 22, .data = {0x12, 0x20, 0x31}},
    },
    .segment = 0,
  });
  return batch;
}

}  // namespace

TEST_CASE("Store filters series paths with a bounded result set") {
  loggy::Store store;
  store.stage(makeBatch());
  store.begin_frame();

  const std::vector<std::string> filtered = store.series_paths_matching("gpsLocationExternal", 2);
  REQUIRE(filtered.size() == 2);
  CHECK(filtered[0].find("gpsLocationExternal") != std::string::npos);
  CHECK(filtered[1].find("gpsLocationExternal") != std::string::npos);

  const std::vector<std::string> empty = store.series_paths_matching("missing", 10);
  CHECK(empty.empty());
}
