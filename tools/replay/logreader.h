#pragma once

#include <string>
#include <vector>

#include "cereal/gen/cpp/log.capnp.h"
#include "system/camerad/cameras/camera_common.h"
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
  LogReader(const std::vector<bool> &filters = {}) { filters_ = filters; }
  bool load(const std::string &url, std::atomic<bool> *abort = nullptr,
            bool local_cache = false, int chunk_size = -1, int retries = 0);
  bool load(const char *data, size_t size, std::atomic<bool> *abort = nullptr);
  std::vector<Event> events;

private:
  std::string raw_;
  std::vector<bool> filters_;
  MonotonicBuffer buffer_{1024 * 1024};
};
