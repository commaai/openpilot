#include "catch2/catch.hpp"
#include "tools/loggy/backend/live.h"
#include "tools/loggy/backend/store.h"

#include "openpilot/cereal/messaging/bridge_zmq.h"

#include <capnp/message.h>
#include <capnp/serialize.h>

#include <chrono>
#include <thread>
#include <vector>

namespace {

template <typename BuildFn>
kj::Array<capnp::word> build_event_words(BuildFn build) {
  capnp::MallocMessageBuilder builder;
  cereal::Event::Builder event = builder.initRoot<cereal::Event>();
  event.setValid(true);
  build(event);
  return capnp::messageToFlatArray(builder);
}

template <typename BuildFn>
bool append_built_event(loggy::LiveCerealAccumulator *accumulator, BuildFn build) {
  kj::Array<capnp::word> words = build_event_words(build);
  return accumulator->append_serialized(words.asPtr());
}

}  // namespace

TEST_CASE("live stream helpers normalize addresses and filter services") {
  CHECK(loggy::live_is_local_stream_address(""));
  CHECK(loggy::live_is_local_stream_address("localhost"));
  CHECK(loggy::normalize_live_stream_address("localhost") == "127.0.0.1");
  CHECK(loggy::normalize_live_stream_address("192.168.0.12") == "192.168.0.12");
  CHECK(std::string(loggy::live_source_kind_label(loggy::LiveSourceKind::CerealLocal)) == "Local (MSGQ)");
  CHECK(std::string(loggy::live_source_kind_label(loggy::LiveSourceKind::CerealRemote)) == "Remote (ZMQ)");
  CHECK(std::string(loggy::live_source_kind_label(loggy::LiveSourceKind::DeviceBridge)) == "Device Bridge");
  CHECK(std::string(loggy::live_source_kind_label(loggy::LiveSourceKind::PandaUsb)) == "Panda USB");
  CHECK(std::string(loggy::live_source_kind_label(loggy::LiveSourceKind::SocketCan)) == "SocketCAN");
  CHECK(loggy::live_source_target_label(loggy::LiveSourceConfig{.kind = loggy::LiveSourceKind::DeviceBridge, .address = "192.168.0.10"}) == "192.168.0.10");
  CHECK(loggy::live_source_target_label(loggy::LiveSourceConfig{.kind = loggy::LiveSourceKind::PandaUsb, .address = ""}) == "first Panda");
  CHECK(loggy::live_source_target_label(loggy::LiveSourceConfig{.kind = loggy::LiveSourceKind::PandaUsb, .address = "panda-serial"}) == "panda-serial");
  CHECK(loggy::live_source_target_label(loggy::LiveSourceConfig{.kind = loggy::LiveSourceKind::SocketCan, .address = "vcan0"}) == "vcan0");
  CHECK(loggy::kPandaBusCount == 3);
  CHECK(loggy::live_panda_can_speed_supported(500));
  CHECK(loggy::live_panda_data_speed_supported(2000));
  CHECK_FALSE(loggy::live_panda_can_speed_supported(333));
  CHECK_FALSE(loggy::live_panda_data_speed_supported(333));
  loggy::PandaBusConfig invalid_panda_bus{.can_speed_kbps = 333, .data_speed_kbps = 333, .can_fd = true};
  invalid_panda_bus = loggy::normalize_live_panda_bus_config(invalid_panda_bus);
  CHECK(invalid_panda_bus.can_speed_kbps == 500);
  CHECK(invalid_panda_bus.data_speed_kbps == 2000);
  CHECK(invalid_panda_bus.can_fd);
  CHECK(loggy::live_should_subscribe_service("carState"));
  CHECK_FALSE(loggy::live_should_subscribe_service("roadEncodeIdx"));
  CHECK_FALSE(loggy::live_should_subscribe_service("thumbnail"));
  (void)loggy::live_panda_serials();
  (void)loggy::live_panda_available();
  (void)loggy::live_socketcan_available();

  loggy::LiveExtractBatch can_batch =
    loggy::make_live_can_batch(loggy::MessageId{.source = 0, .address = 0x123}, {0x12, 0x34}, 1.25, 7);
  CHECK(loggy::live_batch_has_data(can_batch));
  REQUIRE(can_batch.store.can_events.size() == 1);
  CHECK(can_batch.store.can_events[0].id == loggy::MessageId{.source = 0, .address = 0x123});
  CHECK(can_batch.store.can_events[0].range.start_ == 1.25);
  REQUIRE(can_batch.store.can_events[0].events.size() == 1);
  CHECK(can_batch.store.can_events[0].events[0].bus_time == 7);
  CHECK(can_batch.store.can_events[0].events[0].data == std::vector<uint8_t>{0x12, 0x34});
}

TEST_CASE("live cereal accumulator emits store, timeline, and logs") {
  loggy::LiveCerealAccumulator accumulator;

  REQUIRE(append_built_event(&accumulator, [](cereal::Event::Builder event) {
    event.setLogMonoTime(10000000000ULL);
    cereal::CarState::Builder car_state = event.initCarState();
    car_state.setVEgo(12.5);
  }));
  REQUIRE(append_built_event(&accumulator, [](cereal::Event::Builder event) {
    event.setLogMonoTime(10100000000ULL);
    auto car_params = event.initCarParams();
    car_params.setCarFingerprint("HONDA CIVIC 2016");
  }));
  REQUIRE(append_built_event(&accumulator, [](cereal::Event::Builder event) {
    event.setLogMonoTime(10500000000ULL);
    cereal::SelfdriveState::Builder state = event.initSelfdriveState();
    state.setEnabled(true);
    state.setAlertStatus(cereal::SelfdriveState::AlertStatus::NORMAL);
    state.setAlertType("testAlert");
    state.setAlertText1("live alert");
  }));
  REQUIRE(append_built_event(&accumulator, [](cereal::Event::Builder event) {
    event.setLogMonoTime(11000000000ULL);
    event.setLogMessage(R"({"levelnum":30,"filename":"live.py","lineno":7,"funcname":"tick","msg":"hello live","ctx":{"mode":"test"}})");
  }));

  loggy::LiveExtractBatch batch = accumulator.take_batch();
  CHECK(batch.has_time_offset);
  CHECK(batch.time_offset == 10.0);
  CHECK(batch.events_seen == 4);
  CHECK(batch.car_fingerprint == "HONDA CIVIC 2016");
  CHECK(batch.events_appended >= 2);
  REQUIRE(batch.range.valid());
  CHECK(batch.range.start_ == 0.0);
  CHECK(batch.range.end >= 1.0);

  loggy::Store store;
  store.stage(std::move(batch.store));
  const loggy::DrainResult drain = store.begin_frame();
  CHECK(drain.series_chunks > 0);
  const loggy::SeriesView speed = store.series_full("/carState/vEgo", {0.0, 1.0});
  REQUIRE(speed.points.size() == 1);
  CHECK(speed.points[0].t == 0.0);
  CHECK(speed.points[0].value == 12.5);

  REQUIRE(batch.timeline_spans.size() == 1);
  CHECK(batch.timeline_spans[0].kind == loggy::TimelineSpanKind::Engaged);
  REQUIRE(batch.logs.size() == 2);
  CHECK(batch.logs[0].origin == loggy::LogOrigin::Alert);
  CHECK(batch.logs[0].message == "live alert");
  CHECK(batch.logs[1].origin == loggy::LogOrigin::Log);
  CHECK(batch.logs[1].level == 30);
  CHECK(batch.logs[1].source == "live.py:7");
  CHECK(batch.logs[1].message == "hello live");
}

TEST_CASE("remote ZMQ live poller receives bridge messages") {
  BridgeZmqContext pub_context;
  BridgeZmqPubSocket pub;
  REQUIRE(pub.connect(&pub_context, "carState") == 0);

  loggy::LiveSourceConfig source;
  source.kind = loggy::LiveSourceKind::CerealRemote;
  source.address = "127.0.0.1";
  source.buffer_seconds = 5.0;

  loggy::LiveCerealPoller poller;
  poller.start(source);
  for (int i = 0; i < 100 && !poller.snapshot().connected; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(poller.snapshot().connected);

  loggy::LiveExtractBatch batch;
  bool got_batch = false;
  for (int i = 0; i < 100 && !got_batch; ++i) {
    kj::Array<capnp::word> words = build_event_words([](cereal::Event::Builder event) {
      event.setLogMonoTime(20000000000ULL);
      cereal::CarState::Builder car_state = event.initCarState();
      car_state.setVEgo(7.25);
    });
    kj::ArrayPtr<const kj::byte> bytes = words.asBytes();
    pub.send(const_cast<char *>(reinterpret_cast<const char *>(bytes.begin())), bytes.size());

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    loggy::LiveExtractBatch candidate;
    if (poller.consume(candidate).has_update && loggy::live_batch_has_data(candidate)) {
      batch = std::move(candidate);
      got_batch = true;
    }
  }
  poller.stop();

  INFO("received_messages=" << poller.snapshot().received_messages);
  INFO("parsed_messages=" << poller.snapshot().parsed_messages);
  INFO("dropped_messages=" << poller.snapshot().dropped_messages);
  INFO("published_batches=" << poller.snapshot().published_batches);
  REQUIRE(got_batch);
  loggy::Store store;
  store.stage(std::move(batch.store));
  store.begin_frame();
  const loggy::SeriesView speed = store.series_full("/carState/vEgo", {0.0, 1.0});
  REQUIRE_FALSE(speed.points.empty());
  CHECK(speed.points.front().value == 7.25);
}
