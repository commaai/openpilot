#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>

#include <capnp/serialize.h>

#include "common/tests/native_test.h"
#include "tools/cabana/streams/replaystream.h"

namespace {

struct Fixture {
  Fixture() {
    char path[] = "/tmp/cabana-log-replay-XXXXXX";
    auto *created = mkdtemp(path);
    REQUIRE(created != nullptr);
    directory = created;
    std::filesystem::create_directory(directory / (route + "--0"));
    std::ofstream log(directory / (route + "--0/qlog"), std::ios::binary);
    auto append = [&](uint64_t offset, auto init) {
      capnp::MallocMessageBuilder message;
      auto event = message.initRoot<cereal::Event>();
      event.setLogMonoTime(origin + offset);
      event.setValid(true);
      init(event);
      auto words = capnp::messageToFlatArray(message);
      auto bytes = words.asBytes();
      log.write(reinterpret_cast<const char *>(bytes.begin()), bytes.size());
    };
    append(0, [](auto e) { e.initInitData(); });
    append(1, [](auto e) { e.initSelfdriveState(); });  // current schema; no legacy migration
    for (int i = 0; i <= 20; ++i) {
      append(i * 50000000ULL, [i](auto e) { e.initCarState().setVEgo(i); });
    }
    append(100000000, [&](auto e) {
      auto idx = e.initNarrowRoadEncodeIdx();
      idx.setType(cereal::EncodeIndex::Type::FULL_H_E_V_C);
      idx.setTimestampSof(origin + 50000000);
      idx.setSegmentNum(0);
    });
    log.close();
    REQUIRE(bool(log));
  }
  ~Fixture() { std::filesystem::remove_all(directory); }
  std::filesystem::path directory;
  const std::string route = "2024-01-01--00-00-00";
  const uint64_t origin = 1000000000000ULL;
};

template <typename Predicate>
bool pumpUntil(Predicate predicate, int timeout_ms = 5000) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    utils::drainMainThreadQueue();
    if (predicate()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

void test_log_replay() {
  Fixture fixture;
  struct ResetCan { ~ResetCan() { can = nullptr; } } reset;
  ReplayStream stream;
  can = &stream;
  REQUIRE(stream.loadRoute(fixture.route, fixture.directory.string(),
                          REPLAY_FLAG_LOAD_ALL_EVENTS | REPLAY_FLAG_NO_VIPC | REPLAY_FLAG_NO_LOOP));
  REQUIRE(stream.logSignalsEnabled());
  stream.start();
  REQUIRE(pumpUntil([&]() { return stream.logRevision() > 0; }));
  REQUIRE(stream.allEvents().empty());
  auto points = cabana::logPoints(stream.logSegments(), {"carState/vEgo"}, stream.beginMonoTime());
  REQUIRE(points.size() == 21);
  REQUIRE(points.front().x == 0 && points.back().x == 1 && points.back().y == 20);
  auto camera = cabana::logPoints(stream.logSegments(), {"narrowRoadEncodeIdx/segmentNum"}, stream.beginMonoTime());
  REQUIRE(camera.size() == 1 && camera[0].x == 0.1);  // exclude replay's synthetic frame event
  const auto start = std::chrono::steady_clock::now();
  REQUIRE(pumpUntil([&]() { return stream.currentSec() >= 0.8; }));
  REQUIRE(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(400));
  stream.pause(true);
  stream.seekTo(0.2);
  REQUIRE(pumpUntil([&]() { return std::abs(stream.currentSec() - 0.2) < 0.01; }));
  const auto revision = stream.logRevision();
  REQUIRE(cabana::logPoints(stream.logSegments(), {"carState/vEgo"}, stream.beginMonoTime()).size() == 21);
  REQUIRE(stream.logRevision() == revision);
}

}  // namespace

int main() {
  return run_native_test([] {
    test_log_replay();
    std::cout << "Log replay integration passed (no CAN, paced playback, seek, frame de-duplication)\n";
  });
}
