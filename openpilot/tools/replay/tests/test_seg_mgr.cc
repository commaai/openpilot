#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>
#include <unistd.h>

#include <capnp/serialize.h>

#include "common/tests/native_test.h"
#include "tools/replay/replay.h"

using namespace std::chrono_literals;

namespace {
struct RouteFixture {
  RouteFixture() {
    char path[] = "/tmp/replay-segment-test-XXXXXX";
    const char *created = mkdtemp(path);
    REQUIRE(created);
    dir = created;
    for (int n = 0; n < 3; ++n) {
      auto segment_dir = dir / ("0000010a--a51155e496--" + std::to_string(n));
      std::filesystem::create_directory(segment_dir);
      logs[n] = segment_dir / "rlog";
      write(n, n != 1);
    }
  }
  ~RouteFixture() { std::filesystem::remove_all(dir); }

  void write(int n, bool valid) {
    std::ofstream out(logs[n], std::ios::binary);
    if (valid) {
      capnp::MallocMessageBuilder builder;
      auto event = builder.initRoot<cereal::Event>();
      event.setLogMonoTime((60ULL * n + 1) * 1000000000);
      auto can = event.initCan(1)[0];
      can.setAddress(0x4b0);
      can.setSrc(0);
      const std::array<uint8_t, 8> data{0, 0, 0, 0, 0xb0, 0, 0, 0};
      can.setDat(kj::arrayPtr(data.data(), data.size()));
      auto words = capnp::messageToFlatArray(builder);
      auto bytes = words.asBytes();
      out.write(reinterpret_cast<const char *>(bytes.begin()), bytes.size());
    }
  }

  std::filesystem::path dir;
  std::array<std::filesystem::path, 3> logs;
};

template <typename Predicate>
void waitUntil(Predicate predicate) {
  auto deadline = std::chrono::steady_clock::now() + 8s;
  while (!predicate() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(10ms);
  REQUIRE(predicate());
}

void test_failed_segment_recovers() {
  RouteFixture fixture;
  SegmentManager manager("5beb9b58bd12b691/0000010a--a51155e496", REPLAY_FLAG_NO_VIPC, fixture.dir.string());
  std::atomic<int> failures = 0;
  std::atomic<bool> saw_gap = false;
  manager.setCallback([&]() {
    auto data = manager.getEventData();
    if (data->isSegmentLoaded(2) && !data->isSegmentLoaded(1)) saw_gap = true;
  });
  manager.setBenchmarkCallback([&](int n, const std::string &state) {
    if (n == 1 && state == "load failed" && ++failures == 2) fixture.write(1, true);
  });
  REQUIRE(manager.load());
  manager.setCurrentSegment(0);
  // Failed loads must not block later segments, even with playback paused.
  waitUntil([&]() { return manager.getEventData()->isSegmentLoaded(1); });
  manager.stop();
  REQUIRE(saw_gap);
  REQUIRE(failures == 2);
  auto data = manager.getEventData();
  REQUIRE(data->segments.size() == 3);
  REQUIRE(data->events.size() == 3);
  for (int n = 0; n < 3; ++n) {
    REQUIRE(data->events[n].mono_time == (60ULL * n + 1) * 1000000000);
  }
}

void test_retries_are_bounded() {
  RouteFixture fixture;
  SegmentManager manager("5beb9b58bd12b691/0000010a--a51155e496", REPLAY_FLAG_NO_VIPC, fixture.dir.string());
  std::atomic<int> attempts = 0;
  manager.setBenchmarkCallback([&](int n, const std::string &state) {
    if (n == 1 && state == "loading") ++attempts;
  });
  REQUIRE(manager.load());
  manager.setCurrentSegment(0);
  waitUntil([&]() { return attempts >= 3; });
  manager.setCurrentSegment(1);
  std::this_thread::sleep_for(3200ms);
  manager.stop();
  REQUIRE(attempts == 3);
  REQUIRE(manager.getEventData()->isSegmentLoaded(2));
  REQUIRE(!manager.getEventData()->isSegmentLoaded(1));
}

void test_stop_during_retry_delay() {
  RouteFixture fixture;
  SegmentManager manager("5beb9b58bd12b691/0000010a--a51155e496", REPLAY_FLAG_NO_VIPC, fixture.dir.string());
  REQUIRE(manager.load());
  manager.setCurrentSegment(0);
  waitUntil([&]() { return manager.getEventData()->isSegmentLoaded(2); });
  auto start = std::chrono::steady_clock::now();
  manager.stop();
  REQUIRE(std::chrono::steady_clock::now() - start < 500ms);
}
}  // namespace

int main() {
  return run_native_test([]() {
    test_failed_segment_recovers();
    test_retries_are_bounded();
    test_stop_during_retry_delay();
  });
}
