#include "catch2/catch.hpp"
#include "tools/loggy/backend/extract.h"
#include "tools/loggy/backend/store.h"

namespace {

const loggy::SeriesChunk *findChunk(const loggy::StoreBatch &batch, std::string_view path) {
  for (const auto &chunk : batch.series) {
    if (chunk.path == path) return &chunk;
  }
  return nullptr;
}

}  // namespace

TEST_CASE("SeriesAccumulator emits ordered StoreBatch chunks") {
  loggy::SeriesAccumulator series(3, {"/carState/vEgo", "/carState/aEgo"});
  series.append_fixed_scalar(0, 2.0, 12.0);
  series.append_fixed_scalar(0, 1.0, 10.0);
  series.append_fixed_scalar(1, 1.5, -0.25);
  series.append_scalar("/dynamic/value", 3.0, 7.0);

  loggy::SegmentExtractResult result = series.finish({1.0, 4.0});
  CHECK(result.batch.segment == 3);
  REQUIRE(result.batch.coverage.size() == 1);
  CHECK(result.batch.coverage[0].start_ == 1.0);
  CHECK(result.batch.coverage[0].end == 4.0);

  const loggy::SeriesChunk *speed = findChunk(result.batch, "/carState/vEgo");
  REQUIRE(speed != nullptr);
  REQUIRE(speed->points.size() == 2);
  CHECK(speed->points[0].t == 1.0);
  CHECK(speed->points[0].value == 10.0);
  CHECK(speed->points[1].t == 2.0);
  CHECK(speed->points[1].value == 12.0);

  loggy::Store store;
  store.stage(std::move(result.batch));
  const loggy::DrainResult drain = store.begin_frame();
  CHECK(drain.series_chunks == 3);

  const loggy::SeriesView view = store.series("/carState/vEgo", 0.0, 5.0, 100);
  REQUIRE(view.points.size() == 2);
  CHECK(view.points.front().value == 10.0);
  CHECK(view.coverage.ranges.size() == 1);
}

TEST_CASE("SeriesAccumulator captures enum and deprecated metadata") {
  loggy::SeriesAccumulator series(0, {"/carState/gearShifter"});
  series.capture_enum_info("/carState/gearShifter", {"unknown", "park", "drive"});
  series.mark_deprecated("/carState/cruiseState/speedOffsetDEPRECATED");
  series.note_skipped_deprecated();
  series.append_fixed_scalar(0, 0.5, 2.0);

  loggy::SegmentExtractResult result = series.finish({0.0, 1.0});
  REQUIRE(result.metadata.count("/carState/gearShifter") == 1);
  CHECK(result.metadata["/carState/gearShifter"].enum_names[2] == "drive");
  REQUIRE(result.metadata.count("/carState/cruiseState/speedOffsetDEPRECATED") == 1);
  CHECK(result.metadata["/carState/cruiseState/speedOffsetDEPRECATED"].deprecated);
  CHECK(result.deprecated_series_skipped == 1);
}

TEST_CASE("SeriesAccumulator emits raw CAN event chunks") {
  loggy::SeriesAccumulator series(2, {});
  const uint8_t payload[] = {0x12, 0x34, 0x56};
  series.append_can_frame(loggy::CanServiceKind::Can, 1, 0x123, 44, payload, sizeof(payload), 10.0);

  loggy::SegmentExtractResult result = series.finish({});
  REQUIRE(result.batch.can_events.size() == 1);
  const loggy::CanEventChunk &chunk = result.batch.can_events[0];
  CHECK(chunk.id.source == 1);
  CHECK(chunk.id.address == 0x123);
  CHECK(chunk.segment == 2);
  REQUIRE(chunk.events.size() == 1);
  CHECK(chunk.events[0].mono_time == 10.0);
  CHECK(chunk.events[0].bus_time == 44);
  REQUIRE(chunk.events[0].data.size() == 3);
  CHECK(chunk.events[0].data[2] == 0x56);
}
