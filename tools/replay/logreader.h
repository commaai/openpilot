#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "cereal/gen/cpp/log.capnp.h"
#include "tools/replay/util.h"

const CameraType ALL_CAMERAS[] = {RoadCam, DriverCam, WideRoadCam};
const int MAX_CAMERAS = std::size(ALL_CAMERAS);

class Event {
public:
  Event(cereal::Event::Which which, uint64_t mono_time, const kj::ArrayPtr<const capnp::word> &data, int eidx_segnum = -1)
    : which(which), mono_time(mono_time), data(data), eidx_segnum(eidx_segnum) {}

  bool operator<(const Event &other) const {
    return mono_time < other.mono_time || (mono_time == other.mono_time && which < other.which);
  }

  uint64_t mono_time;
  cereal::Event::Which which;
  kj::ArrayPtr<const capnp::word> data;
  int32_t eidx_segnum;
};

class LogReader {
public:
  enum class ProgressStage {
    Downloading,
    Parsing,
  };

  using ProgressCallback = std::function<void(ProgressStage stage, uint64_t current, uint64_t total)>;

  LogReader(const std::vector<bool> &filters = {}) { filters_ = filters; }
  bool load(const std::string &url, std::atomic<bool> *abort = nullptr,
            bool local_cache = false, const ProgressCallback &progress = {});
  bool load(const char *data, size_t size, std::atomic<bool> *abort = nullptr,
            const ProgressCallback &progress = {});
  std::vector<Event> events;

  uint64_t compressed_size() const { return compressed_size_; }
  uint64_t decompressed_size() const { return decompressed_size_; }
  double download_seconds() const { return download_seconds_; }
  double decompress_seconds() const { return decompress_seconds_; }
  double parse_seconds() const { return parse_seconds_; }

private:
  void migrateOldEvents();

  std::string raw_;
  bool requires_migration = true;
  std::vector<bool> filters_;
  MonotonicBuffer buffer_{1024 * 1024};
  uint64_t compressed_size_ = 0;
  uint64_t decompressed_size_ = 0;
  double download_seconds_ = 0.0;
  double decompress_seconds_ = 0.0;
  double parse_seconds_ = 0.0;
};
