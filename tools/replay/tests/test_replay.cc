#define CATCH_CONFIG_MAIN

#include <bzlib.h>
#include <zstd.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

#include "catch2/catch.hpp"
#include "tools/replay/logreader.h"
#include "tools/replay/route.h"

namespace {

std::atomic<int> temp_dir_counter = 0;

struct TempDir {
  TempDir() {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const int id = temp_dir_counter.fetch_add(1);
    path = std::filesystem::temp_directory_path() / ("openpilot-replay-test-" + std::to_string(now) + "-" + std::to_string(id));
    std::filesystem::create_directories(path);
  }

  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }

  std::filesystem::path path;
};

std::string buildEvent(uint64_t mono_time, const std::string &fingerprint) {
  MessageBuilder msg;
  auto event = msg.initEvent(true);
  event.setLogMonoTime(mono_time);
  event.initCarParams().setCarFingerprint(fingerprint);

  auto bytes = msg.toBytes();
  return std::string(reinterpret_cast<const char *>(bytes.begin()), bytes.size());
}

std::string buildLogData() {
  return buildEvent(100, "REPLAY_TEST_A") + buildEvent(200, "REPLAY_TEST_B");
}

std::string compressZst(const std::string &in) {
  std::string out(ZSTD_compressBound(in.size()), '\0');
  size_t compressed_size = ZSTD_compress(out.data(), out.size(), in.data(), in.size(), 1);
  REQUIRE_FALSE(ZSTD_isError(compressed_size));
  out.resize(compressed_size);
  return out;
}

std::string compressBz2(const std::string &in) {
  REQUIRE(in.size() <= std::numeric_limits<unsigned int>::max());
  const auto input_size = static_cast<unsigned int>(in.size());
  unsigned int out_size = input_size + (input_size / 100) + 601;
  std::string out(out_size, '\0');
  int ret = BZ2_bzBuffToBuffCompress(out.data(), &out_size, const_cast<char *>(in.data()), input_size, 9, 0, 30);
  REQUIRE(ret == BZ_OK);
  out.resize(out_size);
  return out;
}

void writeBinaryFile(const std::filesystem::path &path, const std::string &data) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  REQUIRE(out.good());
  out.write(data.data(), data.size());
  REQUIRE(out.good());
}

}  // namespace

TEST_CASE("LogReader loads local compressed logs") {
  const std::string raw = buildLogData();

  SECTION("bz2") {
    TempDir tmp;
    auto path = tmp.path / "rlog.bz2";
    writeBinaryFile(path, compressBz2(raw));

    LogReader log;
    REQUIRE(log.load(path.string()));
    REQUIRE(log.events.size() == 2);
  }

  SECTION("zst") {
    TempDir tmp;
    auto path = tmp.path / "rlog.zst";
    writeBinaryFile(path, compressZst(raw));

    LogReader log;
    REQUIRE(log.load(path.string()));
    REQUIRE(log.events.size() == 2);
  }
}

TEST_CASE("LogReader parses available events from a corrupt log") {
  const std::string event_a = buildEvent(100, "REPLAY_TEST_A");
  const std::string event_b = buildEvent(200, "REPLAY_TEST_B");
  const std::string corrupt_log = event_a + event_b.substr(0, event_b.size() / 2);

  LogReader log;
  REQUIRE(log.load(corrupt_log.data(), corrupt_log.size()));
  REQUIRE(log.events.size() >= 1);
}

TEST_CASE("Route discovers local bz2 log files") {
  TempDir tmp;
  const std::string route_name = "a2a0ccea32023010|2023-07-27--13-01-19";
  auto segment_dir = tmp.path / "a2a0ccea32023010|2023-07-27--13-01-19--0";
  const std::string raw = buildLogData();

  writeBinaryFile(segment_dir / "rlog.bz2", compressBz2(raw));
  writeBinaryFile(segment_dir / "qlog.bz2", compressBz2(raw));

  Route route(route_name, tmp.path.string());
  REQUIRE(route.load());
  REQUIRE(route.segments().size() == 1);

  auto it = route.segments().find(0);
  REQUIRE(it != route.segments().end());
  REQUIRE(it->second.rlog.find("rlog.bz2") != std::string::npos);
  REQUIRE(it->second.qlog.find("qlog.bz2") != std::string::npos);
}
