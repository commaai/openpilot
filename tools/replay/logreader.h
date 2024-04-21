#pragma once

#if __has_include(<memory_resource>)
#define HAS_MEMORY_RESOURCE 1
#include <memory_resource>
#endif
#include <memory>
#include <string>
#include <vector>

#include "cereal/gen/cpp/log.capnp.h"
#include "system/camerad/cameras/camera_common.h"

const CameraType ALL_CAMERAS[] = {RoadCam, DriverCam, WideRoadCam};
const int MAX_CAMERAS = std::size(ALL_CAMERAS);
const int DEFAULT_EVENT_MEMORY_POOL_BLOCK_SIZE = 65000;

class Event {
public:
  Event(cereal::Event::Which which, uint64_t mono_time, const kj::ArrayPtr<const capnp::word> &data, int eidx_segnum = -1)
    : which(which), mono_time(mono_time), data(data), eidx_segnum(eidx_segnum) {}

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
  kj::ArrayPtr<const capnp::word> data;
  int32_t eidx_segnum;
};

class LogReader {
public:
  LogReader(size_t memory_pool_block_size = DEFAULT_EVENT_MEMORY_POOL_BLOCK_SIZE);
  ~LogReader();
  bool load(const std::string &url, std::atomic<bool> *abort = nullptr,
            bool local_cache = false, int chunk_size = -1, int retries = 0);
  bool load(const std::byte *data, size_t size, std::atomic<bool> *abort = nullptr);
  std::vector<Event*> events;

private:
  Event *newEvent(cereal::Event::Which which, uint64_t mono_time, const kj::ArrayPtr<const capnp::word> &words, int eidx_segnum = -1);
  bool parse(std::atomic<bool> *abort);
  std::string raw_;
#ifdef HAS_MEMORY_RESOURCE
  std::pmr::monotonic_buffer_resource mbr_{DEFAULT_EVENT_MEMORY_POOL_BLOCK_SIZE * sizeof(Event)};
#endif
};
