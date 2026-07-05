#include "catch2/catch.hpp"
#include "tools/loggy/backend/ingest.h"
#include "tools/loggy/backend/store.h"

#include <vector>

namespace {

loggy::StoreBatch seriesBatch(int segment, std::string path, loggy::TimeRange range, std::vector<loggy::SeriesPoint> points) {
  loggy::StoreBatch batch;
  batch.segment = segment;
  batch.coverage = {range};
  batch.series.push_back({
    .path = std::move(path),
    .range = range,
    .points = std::move(points),
    .segment = segment,
  });
  return batch;
}

loggy::CanEvent eventAt(double t) {
  return {.mono_time = t, .data = {0x11, 0x22}};
}

}  // namespace

TEST_CASE("Store drains staged batches at frame boundary") {
  loggy::Store store;
  store.stage(seriesBatch(0, "/carState/vEgo", {0.0, 10.0}, {{1.0, 12.0}, {2.0, 13.5}}));

  REQUIRE(store.stagedBatchCount() == 1);
  CHECK(store.series("/carState/vEgo", 0.0, 10.0, 100).points.empty());

  const loggy::DrainResult drain = store.beginFrame();
  CHECK(drain.batches == 1);
  CHECK(drain.series_chunks == 1);
  CHECK(drain.series_points == 2);
  CHECK(store.stagedBatchCount() == 0);

  const loggy::SeriesView view = store.series("/carState/vEgo", 0.0, 10.0, 100);
  REQUIRE(view.points.size() == 2);
  CHECK(view.points[0].value == 12.0);
  CHECK(view.coverage.complete);
}

TEST_CASE("Store keeps CAN events sorted across segment merges") {
  loggy::Store store;
  const loggy::MessageId id{.source = 0, .address = 0x123};

  loggy::StoreBatch late;
  late.segment = 2;
  late.coverage = {{20.0, 30.0}};
  late.can_events.push_back({.id = id, .range = {20.0, 30.0}, .events = {eventAt(21.0), eventAt(22.0)}, .segment = 2});
  store.stage(std::move(late));
  store.beginFrame();

  loggy::StoreBatch early;
  early.segment = 1;
  early.coverage = {{10.0, 20.0}};
  early.can_events.push_back({.id = id, .range = {10.0, 20.0}, .events = {eventAt(12.0), eventAt(11.0)}, .segment = 1});
  store.stage(std::move(early));
  store.beginFrame();

  const loggy::CanEventView view = store.canEvents(id, {10.0, 30.0});
  REQUIRE(view.events.size() == 4);
  CHECK(view.events[0].mono_time == 11.0);
  CHECK(view.events[1].mono_time == 12.0);
  CHECK(view.events[2].mono_time == 21.0);
  CHECK(view.events[3].mono_time == 22.0);
  REQUIRE(view.coverage.ranges.size() == 1);
  CHECK(view.coverage.ranges[0].start == 10.0);
  CHECK(view.coverage.ranges[0].end == 30.0);
  CHECK(view.coverage.complete);

  const std::vector<loggy::MessageId> ids = store.canMessageIds();
  REQUIRE(ids.size() == 1);
  CHECK(ids[0] == id);
}

TEST_CASE("Store decimates series views to requested cap") {
  loggy::Store store;
  std::vector<loggy::SeriesPoint> points;
  for (int i = 0; i < 100; ++i) {
    points.push_back({static_cast<double>(i), static_cast<double>(i * 2)});
  }
  store.stage(seriesBatch(0, "/controlsState/lateralControlState/torqueState/error", {0.0, 100.0}, std::move(points)));
  store.beginFrame();

  const loggy::SeriesView view = store.series("/controlsState/lateralControlState/torqueState/error", 0.0, 99.0, 10);
  CHECK(view.total_points == 100);
  REQUIRE(view.points.size() == 10);
  CHECK(view.decimated);
  CHECK(view.points.front().t == 0.0);
  CHECK(view.points.back().t == 99.0);
}

TEST_CASE("SegmentScheduler reprioritizes and publishes staged batches") {
  loggy::Store store;
  loggy::SegmentScheduler scheduler(&store);
  scheduler.setRouteSegments({
    {.segment = 0, .range = {0.0, 10.0}, .log_path = "seg0"},
    {.segment = 1, .range = {10.0, 20.0}, .log_path = "seg1"},
    {.segment = 2, .range = {20.0, 30.0}, .log_path = "seg2"},
    {.segment = 3, .range = {30.0, 40.0}, .log_path = "seg3"},
  });

  scheduler.setTrackerTime(25.0);
  REQUIRE(scheduler.priorityOrder().front().segment == 2);

  scheduler.setVisibleRanges({{0.0, 5.0}});
  REQUIRE(scheduler.priorityOrder().front().segment == 0);

  scheduler.setVisibleRanges({{35.0, 39.0}});
  auto work = scheduler.takeNext();
  REQUIRE(work.has_value());
  CHECK(work->segment == 3);

  CHECK(scheduler.publish(seriesBatch(3, "/carState/aEgo", {30.0, 40.0}, {{35.0, 1.0}})));
  CHECK(store.stagedBatchCount() == 1);
  store.beginFrame();

  const auto segments = scheduler.segments();
  REQUIRE(segments[3].state == loggy::SegmentState::Loaded);
  const loggy::SeriesView view = store.series("/carState/aEgo", 30.0, 40.0, 10);
  REQUIRE(view.points.size() == 1);
  CHECK(view.coverage.complete);
}
