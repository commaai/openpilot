#include "selfdrive/ui/replay/logreader.h"

#include <cassert>
#include <sstream>
#include "selfdrive/common/util.h"
#include "selfdrive/ui/replay/util.h"

LogReader::~LogReader() {
  for (auto e : events) delete e;
}

bool LogReader::load(const std::string &file, bool is_bz2file) {
  if (is_bz2file) {
    std::stringstream stream;
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
      std::unique_ptr<Event> evt = std::make_unique<Event>(words);

      // Add encodeIdx packet again as a frame packet for the video stream
      if (evt->which == cereal::Event::ROAD_ENCODE_IDX ||
          evt->which == cereal::Event::DRIVER_ENCODE_IDX ||
          evt->which == cereal::Event::WIDE_ROAD_ENCODE_IDX) {

          std::unique_ptr<Event> frame_evt = std::make_unique<Event>(words, true);
          events.push_back(frame_evt.release());
      }

      words = kj::arrayPtr(evt->reader.getEnd(), words.end());
      events.push_back(evt.release());
    } catch (const kj::Exception &e) {
      return false;
    }
  }
  std::sort(events.begin(), events.end(), Event::lessThan());
  return true;
}
