#include "selfdrive/ui/replay/logreader.h"

#include <sstream>
#include "selfdrive/common/util.h"
#include "selfdrive/ui/replay/util.h"

EventMemoryPool::EventMemoryPool(size_t block_size) {
  const size_t buf_size = sizeof(Event) * block_size;
  buffer_ = ::operator new(buf_size);
  mbr_ = new std::pmr::monotonic_buffer_resource(buffer_, buf_size);
}

EventMemoryPool::~EventMemoryPool() {
  ::operator delete(buffer_);
  delete mbr_;
}

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

LogReader::LogReader(size_t memory_pool_block_size) : memory_pool_(memory_pool_block_size) {}

LogReader::~LogReader() {
  for (auto e : events) delete e;
}

bool LogReader::load(const std::string &file) {
  bool is_bz2 = file.rfind(".bz2") == file.length() - 4;
  if (is_bz2) {
    std::ostringstream stream;
    if (!readBZ2File(file, stream)) {
      LOGW("bz2 decompress failed");
      return false;
    }
    raw_ = stream.str();
  } else {
    raw_ = util::read_file(file);
  }

  kj::ArrayPtr<const capnp::word> words((const capnp::word *)raw_.data(), raw_.size() / sizeof(capnp::word));
  while (words.size() > 0) {
    try {
      Event *evt = new (memory_pool_) Event(words);

      // Add encodeIdx packet again as a frame packet for the video stream
      if (evt->which == cereal::Event::ROAD_ENCODE_IDX ||
          evt->which == cereal::Event::DRIVER_ENCODE_IDX ||
          evt->which == cereal::Event::WIDE_ROAD_ENCODE_IDX) {

          Event *frame_evt = new (memory_pool_) Event(words, true);
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
