#include "tools/replay/logreader.h"

#include <algorithm>
#include "tools/replay/filereader.h"
#include "tools/replay/util.h"

bool LogReader::load(const std::string &url, std::atomic<bool> *abort, bool local_cache, int chunk_size, int retries) {
  raw_ = FileReader(local_cache, chunk_size, retries).read(url, abort);
  if (raw_.empty()) return false;

  if (url.find(".bz2") != std::string::npos) {
    raw_ = decompressBZ2(raw_, abort);
    if (raw_.empty()) return false;
  }
  return load(raw_.data(), raw_.size(), abort);
}

bool LogReader::load(const char *data, size_t size, std::atomic<bool> *abort) {
  try {
    events.reserve(65000);
    kj::ArrayPtr<const capnp::word> words((const capnp::word *)data, size / sizeof(capnp::word));
    while (words.size() > 0 && !(abort && *abort)) {
      capnp::FlatArrayMessageReader reader(words);
      auto event = reader.getRoot<cereal::Event>();
      auto which = event.which();
      uint64_t mono_time = event.getLogMonoTime();
      auto event_data = kj::arrayPtr(words.begin(), reader.getEnd());

      const Event &evt = events.emplace_back(which, mono_time, event_data);
      // Add encodeIdx packet again as a frame packet for the video stream
      if (evt.which == cereal::Event::ROAD_ENCODE_IDX ||
          evt.which == cereal::Event::DRIVER_ENCODE_IDX ||
          evt.which == cereal::Event::WIDE_ROAD_ENCODE_IDX) {
        auto idx = capnp::AnyStruct::Reader(event).getPointerSection()[0].getAs<cereal::EncodeIndex>();
        if (uint64_t sof = idx.getTimestampSof()) {
          mono_time = sof;
        }
        events.emplace_back(which, mono_time, event_data, idx.getSegmentNum());
      }

      words = kj::arrayPtr(reader.getEnd(), words.end());
    }
  } catch (const kj::Exception &e) {
    rWarning("Failed to parse log : %s.\nRetrieved %zu events from corrupt log", e.getDescription().cStr(), events.size());
  }

  if (!events.empty() && !(abort && *abort)) {
    events.shrink_to_fit();
    std::sort(events.begin(), events.end());
    return true;
  }
  return false;
}
