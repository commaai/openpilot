#include "tools/replay/logreader.h"

#include <algorithm>
#include <utility>
#include "tools/replay/filereader.h"
#include "tools/replay/util.h"

bool LogReader::load(const std::string &url, std::atomic<bool> *abort, bool local_cache, int chunk_size, int retries) {
  std::string data = FileReader(local_cache, chunk_size, retries).read(url, abort);
  if (!data.empty() && url.find(".bz2") != std::string::npos)
    data = decompressBZ2(data, abort);

  bool success = !data.empty() && load(data.data(), data.size(), abort);
  if (filters_.empty())
    raw_ = std::move(data);
  return success;
}

bool LogReader::load(const char *data, size_t size, std::atomic<bool> *abort) {
  try {
    events.reserve(65000);
    kj::ArrayPtr<const capnp::word> words((const capnp::word *)data, size / sizeof(capnp::word));
    while (words.size() > 0 && !(abort && *abort)) {
      capnp::FlatArrayMessageReader reader(words);
      auto event = reader.getRoot<cereal::Event>();
      auto which = event.which();
      auto event_data = kj::arrayPtr(words.begin(), reader.getEnd());
      words = kj::arrayPtr(reader.getEnd(), words.end());

      if (!filters_.empty()) {
        if (which >= filters_.size() || !filters_[which])
          continue;
        auto buf = buffer_.allocate(event_data.size() * sizeof(capnp::word));
        memcpy(buf, event_data.begin(), event_data.size() * sizeof(capnp::word));
        event_data = kj::arrayPtr((const capnp::word *)buf, event_data.size());
      }

      uint64_t mono_time = event.getLogMonoTime();
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
