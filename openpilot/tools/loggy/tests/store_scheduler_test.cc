#include "catch2/catch.hpp"
#include "msgq/visionipc/visionipc_server.h"
#include "tools/loggy/backend/ingest.h"
#include "tools/loggy/backend/store.h"
#include "tools/loggy/backend/video.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
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

  REQUIRE(store.staged_batch_count() == 1);
  CHECK(store.series("/carState/vEgo", 0.0, 10.0, 100).points.empty());

  const loggy::DrainResult drain = store.begin_frame();
  CHECK(drain.batches == 1);
  CHECK(drain.series_chunks == 1);
  CHECK(drain.series_points == 2);
  REQUIRE(drain.touched_series_paths.size() == 1);
  CHECK(drain.touched_series_paths[0] == "/carState/vEgo");
  CHECK(store.staged_batch_count() == 0);

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
  store.begin_frame();

  loggy::StoreBatch early;
  early.segment = 1;
  early.coverage = {{10.0, 20.0}};
  early.can_events.push_back({.id = id, .range = {10.0, 20.0}, .events = {eventAt(12.0), eventAt(11.0)}, .segment = 1});
  store.stage(std::move(early));
  store.begin_frame();

  const loggy::CanEventView view = store.can_events(id, {10.0, 30.0});
  REQUIRE(view.events.size() == 4);
  CHECK(view.events[0].mono_time == 11.0);
  CHECK(view.events[1].mono_time == 12.0);
  CHECK(view.events[2].mono_time == 21.0);
  CHECK(view.events[3].mono_time == 22.0);
  REQUIRE(view.coverage.ranges.size() == 1);
  CHECK(view.coverage.ranges[0].start_ == 10.0);
  CHECK(view.coverage.ranges[0].end == 30.0);
  CHECK(view.coverage.complete);

  const std::vector<loggy::MessageId> ids = store.can_message_ids();
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
  store.begin_frame();

  const loggy::SeriesView view = store.series("/controlsState/lateralControlState/torqueState/error", 0.0, 99.0, 10);
  CHECK(view.total_points == 100);
  REQUIRE(view.points.size() == 10);
  CHECK(view.decimated);
  CHECK(view.points.front().t == 0.0);
  CHECK(view.points.back().t == 99.0);

  const loggy::SeriesView full = store.series_full("/controlsState/lateralControlState/torqueState/error", {0.0, 99.0});
  CHECK(full.total_points == 100);
  REQUIRE(full.points.size() == 100);
  CHECK_FALSE(full.decimated);
}

TEST_CASE("Store trims committed data before a retention boundary") {
  loggy::Store store;
  const loggy::MessageId retained_id{.source = 0, .address = 0x123};
  const loggy::MessageId dropped_id{.source = 0, .address = 0x456};

  loggy::StoreBatch batch;
  batch.segment = 0;
  batch.coverage = {{0.0, 4.0}};
  batch.series.push_back({
    .path = "/carState/vEgo",
    .range = {0.0, 3.0},
    .points = {{0.0, 10.0}, {1.0, 11.0}, {2.0, 12.0}, {3.0, 13.0}},
    .segment = 0,
  });
  batch.series.push_back({
    .path = "/carState/oldOnly",
    .range = {0.0, 1.0},
    .points = {{0.5, 1.0}},
    .segment = 0,
  });
  batch.can_events.push_back({
    .id = retained_id,
    .range = {0.0, 3.0},
    .events = {{.mono_time = 0.5, .data = {0x01}}, {.mono_time = 2.5, .data = {0x02}}},
    .segment = 0,
  });
  batch.can_events.push_back({
    .id = dropped_id,
    .range = {0.0, 1.0},
    .events = {{.mono_time = 0.25, .data = {0xff}}},
    .segment = 0,
  });
  store.stage(std::move(batch));
  store.begin_frame();

  const loggy::StoreTrimResult trim = store.trim_before(2.0);
  CHECK(trim.series_points_removed == 3);
  CHECK(trim.series_chunks_removed == 1);
  CHECK(trim.series_paths_removed == 1);
  CHECK(trim.can_events_removed == 2);
  CHECK(trim.can_messages_removed == 1);
  REQUIRE(trim.touched_series_paths.size() == 2);
  CHECK(trim.touched_series_paths[0] == "/carState/oldOnly");
  CHECK(trim.touched_series_paths[1] == "/carState/vEgo");

  const loggy::SeriesView speed = store.series_full("/carState/vEgo", {0.0, 4.0});
  REQUIRE(speed.points.size() == 2);
  CHECK(speed.points[0].t == 2.0);
  CHECK(speed.points[1].t == 3.0);
  REQUIRE(speed.coverage.ranges.size() == 1);
  CHECK(speed.coverage.ranges[0].start_ == 2.0);

  CHECK(store.series_full("/carState/oldOnly", {0.0, 4.0}).points.empty());
  const std::vector<std::string> paths = store.series_paths();
  CHECK(std::find(paths.begin(), paths.end(), "/carState/oldOnly") == paths.end());

  const loggy::CanEventView retained = store.can_events(retained_id, {0.0, 4.0});
  REQUIRE(retained.events.size() == 1);
  CHECK(retained.events[0].mono_time == 2.5);
  REQUIRE(retained.coverage.ranges.size() == 1);
  CHECK(retained.coverage.ranges[0].start_ == 2.0);
  CHECK(store.can_events(dropped_id, {0.0, 4.0}).events.empty());
  const std::vector<loggy::MessageId> ids = store.can_message_ids();
  CHECK(std::find(ids.begin(), ids.end(), dropped_id) == ids.end());
}

TEST_CASE("Store replaces computed series without wiping route data") {
  loggy::Store store;
  store.stage(seriesBatch(0, "/carState/vEgo", {0.0, 2.0}, {{0.0, 10.0}, {1.0, 11.0}}));
  store.stage(seriesBatch(0, "/computed/vEgo_scaled", {0.0, 2.0}, {{0.0, 20.0}, {1.0, 22.0}}));
  store.begin_frame();

  loggy::StoreBatch replace;
  replace.segment = -1;
  replace.replace_series_paths = {"/computed/vEgo_scaled"};
  replace.coverage = {{0.0, 2.0}};
  replace.series.push_back({
    .path = "/computed/vEgo_scaled",
    .range = {0.0, 2.0},
    .points = {{0.0, 30.0}, {1.0, 33.0}},
    .segment = -1,
  });
  store.stage(std::move(replace));
  const loggy::DrainResult drain = store.begin_frame();
  REQUIRE(drain.touched_series_paths.size() == 1);
  CHECK(drain.touched_series_paths[0] == "/computed/vEgo_scaled");

  const loggy::SeriesView computed = store.series_full("/computed/vEgo_scaled", {0.0, 2.0});
  REQUIRE(computed.points.size() == 2);
  CHECK(computed.points[0].value == 30.0);
  CHECK(computed.points[1].value == 33.0);

  const loggy::SeriesView route = store.series_full("/carState/vEgo", {0.0, 2.0});
  REQUIRE(route.points.size() == 2);
  CHECK(route.points[0].value == 10.0);

  loggy::StoreBatch ignored_replace;
  ignored_replace.replace_series_paths = {"/carState/vEgo"};
  store.stage(std::move(ignored_replace));
  store.begin_frame();
  CHECK(store.series_full("/carState/vEgo", {0.0, 2.0}).points.size() == 2);
}

TEST_CASE("SegmentScheduler reprioritizes and publishes staged batches") {
  loggy::Store store;
  loggy::SegmentScheduler scheduler(&store);
  scheduler.set_route_segments({
    {.segment = 0, .range = {0.0, 10.0}, .log_path = "seg0"},
    {.segment = 1, .range = {10.0, 20.0}, .log_path = "seg1"},
    {.segment = 2, .range = {20.0, 30.0}, .log_path = "seg2"},
    {.segment = 3, .range = {30.0, 40.0}, .log_path = "seg3"},
  });

  scheduler.set_tracker_time(25.0);
  {
    // take_next() is the real selection boundary; mark_pending() undoes the pick so later
    // checks in this test still see all four segments as schedulable.
    auto picked = scheduler.take_next();
    REQUIRE(picked.has_value());
    CHECK(picked->segment == 2);
    scheduler.mark_pending(picked->segment);
  }

  scheduler.set_visible_ranges({{0.0, 5.0}});
  {
    auto picked = scheduler.take_next();
    REQUIRE(picked.has_value());
    CHECK(picked->segment == 0);
    scheduler.mark_pending(picked->segment);
  }

  scheduler.set_visible_ranges({{35.0, 39.0}});
  auto work = scheduler.take_next();
  REQUIRE(work.has_value());
  CHECK(work->segment == 3);

  CHECK(scheduler.publish(seriesBatch(3, "/carState/aEgo", {30.0, 40.0}, {{35.0, 1.0}})));
  CHECK(store.staged_batch_count() == 1);
  store.begin_frame();

  const auto segments = scheduler.segments();
  REQUIRE(segments[3].state == loggy::SegmentState::Loaded);
  const loggy::SeriesView view = store.series("/carState/aEgo", 30.0, 40.0, 10);
  REQUIRE(view.points.size() == 1);
  CHECK(view.coverage.complete);
}

TEST_CASE("Video helpers build camera feed indexes from route segments and encode series") {
  loggy::Store store;
  loggy::StoreBatch batch;
  batch.segment = 0;
  batch.coverage = {{0.0, 10.0}};
  batch.series.push_back({
    .path = "/roadEncodeIdx/segmentNum",
    .range = {0.0, 10.0},
    .points = {{0.0, 0.0}, {0.1, 0.0}, {0.2, 2.0}, {0.3, 1.0}},
    .segment = 0,
  });
  batch.series.push_back({
    .path = "/roadEncodeIdx/segmentId",
    .range = {0.0, 10.0},
    .points = {{0.0, 10.0}, {0.1, 11.0}, {0.2, 20.0}, {0.3, 30.0}},
    .segment = 0,
  });
  batch.series.push_back({
    .path = "/roadEncodeIdx/frameId",
    .range = {0.0, 10.0},
    .points = {{0.0, 100.0}, {0.1, 101.0}, {0.2, 200.0}, {0.3, 300.0}},
    .segment = 0,
  });
  batch.series.push_back({
    .path = "/wideRoadEncodeIdx/segmentNum",
    .range = {0.0, 10.0},
    .points = {{0.4, 1.0}},
    .segment = 0,
  });
  batch.series.push_back({
    .path = "/wideRoadEncodeIdx/segmentIdEncode",
    .range = {0.0, 10.0},
    .points = {{0.4, 44.0}},
    .segment = 0,
  });
  store.stage(std::move(batch));
  store.begin_frame();

  const std::vector<loggy::RouteSegment> segments = {
    {.segment = 0, .range = {0.0, 5.0}, .log_path = "qlog0", .road_camera_path = "fcamera0.hevc"},
    {.segment = 1, .range = {5.0, 10.0}, .log_path = "qlog1", .wide_road_camera_path = "ecamera1.hevc"},
  };

  const loggy::CameraFeedIndex road =
    loggy::build_camera_feed_index(segments, store, loggy::CameraViewKind::Road, {0.0, 10.0});
  REQUIRE(road.segment_files.size() == 1);
  CHECK(road.segment_files[0].segment == 0);
  CHECK(road.segment_files[0].path == "fcamera0.hevc");
  REQUIRE(road.entries.size() == 2);
  CHECK(road.entries[0].timestamp == 0.0);
  CHECK(road.entries[0].segment == 0);
  CHECK(road.entries[0].decode_index == 10);
  CHECK(road.entries[0].frame_id == 100);
  CHECK(road.entries[1].decode_index == 11);
  CHECK(road.entries[1].path == "fcamera0.hevc");

  const auto frame = loggy::camera_frame_at_time(road, 0.09);
  REQUIRE(frame.has_value());
  CHECK(frame->decode_index == 11);
  CHECK(loggy::camera_frame_at_time(road, -1.0)->decode_index == 10);
  CHECK(loggy::camera_frame_at_time(road, 99.0)->decode_index == 11);

  const loggy::CameraFeedIndex wide =
    loggy::build_camera_feed_index(segments, store, loggy::CameraViewKind::WideRoad, {0.0, 10.0});
  REQUIRE(wide.segment_files.size() == 1);
  REQUIRE(wide.entries.size() == 1);
  CHECK(wide.entries[0].decode_index == 44);
  CHECK(wide.entries[0].frame_id == 44);
  CHECK(wide.entries[0].path == "ecamera1.hevc");

  CHECK(loggy::camera_view_from_layout_name("driver") == loggy::CameraViewKind::Driver);
  CHECK(loggy::camera_view_from_layout_name("wide_road") == loggy::CameraViewKind::WideRoad);
  CHECK(loggy::camera_view_layout_name(loggy::CameraViewKind::QRoad) == "qroad");
}

TEST_CASE("Live camera helpers expose only VisionIPC-backed camera views") {
  CHECK(loggy::live_camera_view_supported(loggy::CameraViewKind::Road));
  CHECK(loggy::live_camera_view_supported(loggy::CameraViewKind::Driver));
  CHECK(loggy::live_camera_view_supported(loggy::CameraViewKind::WideRoad));
  CHECK_FALSE(loggy::live_camera_view_supported(loggy::CameraViewKind::QRoad));
  CHECK(loggy::live_camera_stream_label(loggy::CameraViewKind::WideRoad) == "wide");

  loggy::LiveCameraFrameSource source;
  source.request_frame(loggy::CameraViewKind::Road);
  CHECK_FALSE(source.status(loggy::CameraViewKind::Road).requested);

  source.set_enabled(true);
  source.request_frame(loggy::CameraViewKind::QRoad);
  const loggy::LiveCameraFrameStatus status = source.status(loggy::CameraViewKind::QRoad);
  CHECK_FALSE(status.supported);
  CHECK(status.requested);
  CHECK_FALSE(status.connected);
  CHECK(status.error.find("unavailable") != std::string::npos);
  CHECK_FALSE(source.take_frame(loggy::CameraViewKind::QRoad).has_value());
  source.set_enabled(false);
  CHECK_FALSE(source.status(loggy::CameraViewKind::QRoad).requested);
}

TEST_CASE("Live camera source receives synthetic VisionIPC frames") {
  const std::string server_name = "loggy_test_camerad_" +
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  VisionIpcServer server(server_name);
  server.create_buffers(VISION_STREAM_ROAD, 4, 4, 4);
  server.start_listener();

  loggy::LiveCameraFrameSource source(server_name);
  source.set_enabled(true);
  source.request_frame(loggy::CameraViewKind::Road);

  std::optional<loggy::DecodedCameraFrame> frame;
  for (uint32_t frame_id = 1; frame_id <= 30 && !frame.has_value(); ++frame_id) {
    VisionBuf *buf = server.get_buffer(VISION_STREAM_ROAD);
    std::fill(buf->y, buf->y + (buf->stride * buf->height), static_cast<uint8_t>(128));
    std::fill(buf->uv, buf->uv + (buf->stride * buf->height / 2), static_cast<uint8_t>(128));
    buf->set_frame_id(frame_id);
    VisionIpcBufExtra extra{
      .frame_id = frame_id,
      .timestamp_sof = static_cast<uint64_t>(frame_id) * 1000000ULL,
      .timestamp_eof = static_cast<uint64_t>(frame_id) * 1000000ULL + 1000ULL,
      .valid = true,
    };
    server.send(buf, &extra);

    for (int i = 0; i < 5 && !frame.has_value(); ++i) {
      frame = source.take_frame(loggy::CameraViewKind::Road);
      if (!frame.has_value()) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  REQUIRE(frame.has_value());
  CHECK(frame->ok);
  CHECK(frame->key.view == loggy::CameraViewKind::Road);
  CHECK(frame->width == 4);
  CHECK(frame->height == 4);
  CHECK(frame->rgba.size() == 4U * 4U * 4U);

  const loggy::LiveCameraFrameStatus status = source.status(loggy::CameraViewKind::Road);
  CHECK(status.supported);
  CHECK(status.connected);
  CHECK(status.has_frame);
  CHECK(status.width == 4);
  CHECK(status.height == 4);
  CHECK(status.received_frames >= 1);
  source.set_enabled(false);
}

TEST_CASE("Camera frame decoder reports asynchronous load failures") {
  loggy::CameraFeedIndex index;
  index.view = loggy::CameraViewKind::Road;
  index.segment_files.push_back({
    .segment = 0,
    .range = {0.0, 1.0},
    .path = "/tmp/loggy_missing_camera_file.hevc",
  });
  index.entries.push_back({
    .timestamp = 0.0,
    .segment = 0,
    .decode_index = 0,
    .frame_id = 0,
    .path = "/tmp/loggy_missing_camera_file.hevc",
  });

  loggy::CameraFrameDecoder decoder;
  decoder.set_camera_index(index);
  decoder.request_frame(0.0);

  std::optional<loggy::DecodedCameraFrame> result;
  for (int i = 0; i < 100 && !result.has_value(); ++i) {
    result = decoder.take_frame();
    if (!result.has_value()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  REQUIRE(result.has_value());
  CHECK_FALSE(result->ok);
  CHECK(result->key.segment == 0);
  CHECK(result->key.decode_index == 0);
  CHECK(result->error.find("load") != std::string::npos);

  const loggy::CameraDecodeStatus status = decoder.status();
  REQUIRE(status.failed_key.has_value());
  CHECK(status.failed_key->segment == 0);
  CHECK_FALSE(status.loading);
}
