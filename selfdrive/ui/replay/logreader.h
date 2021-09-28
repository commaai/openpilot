#pragma once

#include <unordered_map>
#include <vector>

#include <capnp/serialize.h>
#include "cereal/gen/cpp/log.capnp.h"
#include "selfdrive/camerad/cameras/camera_common.h"

const CameraType ALL_CAMERAS[] = {RoadCam, DriverCam, WideRoadCam};
const int MAX_CAMERAS = std::size(ALL_CAMERAS);

class Event {
public:
  Event(cereal::Event::Which which, uint64_t mono_time) : reader(kj::ArrayPtr<capnp::word>{}) {
    // construct a dummy Event for binary search, e.g std::upper_bound
    this->which = which;
    this->mono_time = mono_time;
  }
  Event(const kj::ArrayPtr<const capnp::word> &amsg) : reader(amsg) {
    words = kj::ArrayPtr<const capnp::word>(amsg.begin(), reader.getEnd());
    event = reader.getRoot<cereal::Event>();
    which = event.which();
    mono_time = event.getLogMonoTime();

    if (which == cereal::Event::ROAD_ENCODE_IDX) {
      uint64_t sof = event.getRoadEncodeIdx().getTimestampSof();
      uint64_t eof = event.getRoadEncodeIdx().getTimestampEof();
      if (sof > 0) {
        mono_time = sof;
      } else if (eof > 0) {
        mono_time = eof;
      }
    }
  }
  inline kj::ArrayPtr<const capnp::byte> bytes() const { return words.asBytes(); }

  struct lessThan {
    inline bool operator()(const Event *l, const Event *r) {
      return l->mono_time < r->mono_time || (l->mono_time == r->mono_time && l->which < r->which);
    }
  };

  uint64_t mono_time;
  cereal::Event::Which which;
  cereal::Event::Reader event;
  capnp::FlatArrayMessageReader reader;
  kj::ArrayPtr<const capnp::word> words;
};

class LogReader {
public:
  LogReader() = default;
  ~LogReader();
  bool load(const std::string &file);

  std::vector<Event*> events;

private:
  std::vector<uint8_t> raw_;
};
