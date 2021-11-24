#include "selfdrive/ui/replay/logreader.h"

#include <algorithm>
#include "selfdrive/ui/replay/util.h"

Event::Event(const kj::ArrayPtr<const capnp::word> &amsg, bool frame) : reader(amsg), frame(frame) {
  words = kj::ArrayPtr<const capnp::word>(amsg.begin(), reader.getEnd());
  event = reader.getRoot<cereal::Event>();
  which = event.which();
  mono_time = event.getLogMonoTime();

  // 1) Send video data at t=timestampEof/timestampSof
  // 2) Send encodeIndex packet at t=logMonoTime
  if (frame) {
    auto idx = capnp::AnyStruct::Reader(event).getPointerSection()[0].getAs<cereal::EncodeIndex>();
    // C2 only has eof set, and some older routes have neither
    uint64_t sof = idx.getTimestampSof();
    uint64_t eof = idx.getTimestampEof();
    if (sof > 0) {
      mono_time = sof;
    } else if (eof > 0) {
      mono_time = eof;
    }
  }
}

// class LogReader

LogReader::LogReader(bool local_cache, int chunk_size, int retries, size_t memory_pool_block_size) : FileReader(local_cache, chunk_size, retries) {
#ifdef HAS_MEMORY_RESOURCE
  const size_t buf_size = sizeof(Event) * memory_pool_block_size;
  pool_buffer_ = ::operator new(buf_size);
  mbr_ = new std::pmr::monotonic_buffer_resource(pool_buffer_, buf_size);
#endif
  events.reserve(memory_pool_block_size);
}

LogReader::~LogReader() {
  for (Event *e : events) {
    delete e;
  }
#ifdef HAS_MEMORY_RESOURCE
  delete mbr_;
  ::operator delete(pool_buffer_);
#endif
}

bool LogReader::load(const std::string &file, std::atomic<bool> *abort) {
  raw_ = decompressBZ2(read(file, abort));
  if (raw_.empty()) return false;

  kj::ArrayPtr<const capnp::word> words((const capnp::word *)raw_.data(), raw_.size() / sizeof(capnp::word));
  while (words.size() > 0) {
    try {
#ifdef HAS_MEMORY_RESOURCE
      Event *evt = new (mbr_) Event(words);
#else
      Event *evt = new Event(words);
#endif

      // Add encodeIdx packet again as a frame packet for the video stream
      if (evt->which == cereal::Event::ROAD_ENCODE_IDX ||
          evt->which == cereal::Event::DRIVER_ENCODE_IDX ||
          evt->which == cereal::Event::WIDE_ROAD_ENCODE_IDX) {
#ifdef HAS_MEMORY_RESOURCE
        Event *frame_evt = new (mbr_) Event(words, true);
#else
        Event *frame_evt = new Event(words, true);
#endif
        events.push_back(frame_evt);
      }

      words = kj::arrayPtr(evt->reader.getEnd(), words.end());
      events.push_back(evt);
    } catch (const kj::Exception &e) {
      return false;
    }
  }
  std::sort(events.begin(), events.end(), Event::lessThan());
  return true;
}
