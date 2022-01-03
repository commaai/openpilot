#pragma once

#if __has_include(<memory_resource>)
#define HAS_MEMORY_RESOURCE 1
#include <memory_resource>
#endif

#include "cereal/gen/cpp/log.capnp.h"
#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/ui/replay/filereader.h"

const CameraType ALL_CAMERAS[] = {RoadCam, DriverCam, WideRoadCam};
const int MAX_CAMERAS = std::size(ALL_CAMERAS);
const int DEFAULT_EVENT_MEMORY_POOL_BLOCK_SIZE = 65000;

class Event {
public:
  Event(cereal::Event::Which which, uint64_t mono_time) : reader(kj::ArrayPtr<capnp::word>{}) {
    // construct a dummy Event for binary search, e.g std::upper_bound
    this->which = which;
    this->mono_time = mono_time;
  }
  Event(const kj::ArrayPtr<const capnp::word> &amsg, bool frame = false);
  inline kj::ArrayPtr<const capnp::byte> bytes() const { return words.asBytes(); }
  inline char *data() const { return (char *)words.asBytes().begin(); }
  inline size_t size() const { return words.asBytes().size(); }
  inline uint64_t monoTime() const { return mono_time; }

  struct lessThan {
    inline bool operator()(const Event *l, const Event *r) {
      return l->mono_time < r->mono_time || (l->mono_time == r->mono_time && l->which < r->which);
    }
  };

#if HAS_MEMORY_RESOURCE
  void *operator new(size_t size, std::pmr::monotonic_buffer_resource *mbr) {
    return mbr->allocate(size);
  }
  void operator delete(void *ptr) {
    // No-op. memory used by EventMemoryPool increases monotonically until the logReader is destroyed. 
  }
#endif

  uint64_t mono_time;
  cereal::Event::Which which;
  cereal::Event::Reader event;
  capnp::FlatArrayMessageReader reader;
  kj::ArrayPtr<const capnp::word> words;
  bool frame;
};

class LogReader {
public:
  LogReader(size_t memory_pool_block_size = DEFAULT_EVENT_MEMORY_POOL_BLOCK_SIZE);
  ~LogReader();
  bool load(const std::string &url, bool local_cache = false, std::atomic<bool> *abort = nullptr, int chunk_size = -1, int retries = 0);
  bool load(const std::byte *data, size_t size, std::atomic<bool> *abort = nullptr);
  inline void setSortByTime(bool b) { sort = b; }
  inline size_t size() const { return events.size(); }
  inline const std::vector<Event*> getEvents() const { return events; }
  inline Event *at(int idx) const {
    if (idx < 0 || idx >= events.size()) return nullptr;
    return events[idx];
  }

  std::vector<Event*> events;

private:
  bool sort = true;
  std::string raw_;
#ifdef HAS_MEMORY_RESOURCE
  std::pmr::monotonic_buffer_resource *mbr_ = nullptr;
  void *pool_buffer_ = nullptr;
#endif
};
