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
  Event(cereal::Event::Which which, uint64_t mono_time) {
    // construct a dummy Event for binary search, e.g std::upper_bound
    this->which = which;
    this->mono_time = mono_time;
  }
  Event(MessageBuilder &msg) {
    // construct a custom event
    buffer_ = capnp::messageToFlatArray(msg);
    init(buffer_);
  }
  Event(const kj::ArrayPtr<const capnp::word> &amsg) { init(amsg); }
  inline kj::ArrayPtr<const capnp::byte> bytes() const { return words.asBytes(); }
  inline bool isCustomEvent() const { return buffer_.size() > 0;}

  struct lessThan {
    inline bool operator()(const Event *l, const Event *r) {
      return l->mono_time < r->mono_time || (l->mono_time == r->mono_time && l->which < r->which);
    }
  };

  uint64_t mono_time;
  cereal::Event::Which which;
  cereal::Event::Reader event;
  std::unique_ptr<capnp::FlatArrayMessageReader> reader;
  kj::ArrayPtr<const capnp::word> words;

private:
  void init(const kj::ArrayPtr<const capnp::word> &amsg) {
    reader = std::make_unique<capnp::FlatArrayMessageReader>(amsg);
    words = kj::ArrayPtr<const capnp::word>(amsg.begin(), reader->getEnd());
    event = reader->getRoot<cereal::Event>();
    which = event.which();
    mono_time = event.getLogMonoTime();
  }

  kj::Array<capnp::word> buffer_;
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
