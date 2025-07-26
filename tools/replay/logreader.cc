#include "tools/replay/logreader.h"

#include <algorithm>
#include <utility>
#include "tools/replay/filereader.h"
#include "tools/replay/util.h"
#include "common/util.h"

bool LogReader::load(const std::string &url, std::atomic<bool> *abort, bool local_cache, int chunk_size, int retries) {
  std::string data = FileReader(local_cache, chunk_size, retries).read(url, abort);
  if (!data.empty()) {
    if (url.find(".bz2") != std::string::npos || util::starts_with(data, "BZh9")) {
      data = decompressBZ2(data, abort);
    } else if (url.find(".zst") != std::string::npos || util::starts_with(data, "\x28\xB5\x2F\xFD")) {
      data = decompressZST(data, abort);
    }
  }

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
      if (which == cereal::Event::Which::SELFDRIVE_STATE) {
        requires_migration = false;
      }

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
        if (idx.getType() == cereal::EncodeIndex::Type::FULL_H_E_V_C) {
          uint64_t sof = idx.getTimestampSof();
          events.emplace_back(which, sof ? sof : mono_time, event_data, idx.getSegmentNum());
        }
      }
    }
  } catch (const kj::Exception &e) {
    rWarning("Failed to parse log : %s.\nRetrieved %zu events from corrupt log", e.getDescription().cStr(), events.size());
  }

  if (requires_migration) {
    migrateOldEvents();
  }

  if (!events.empty() && !(abort && *abort)) {
    events.shrink_to_fit();
    std::sort(events.begin(), events.end());
    return true;
  }
  return false;
}

void LogReader::migrateOldEvents() {
  size_t events_size = events.size();
  for (int i = 0; i < events_size; ++i) {
    // Check if the event is of the old CONTROLS_STATE type
    auto &event = events[i];
    if (event.which == cereal::Event::CONTROLS_STATE) {
      // Read the old event data
      capnp::FlatArrayMessageReader reader(event.data);
      auto old_evt = reader.getRoot<cereal::Event>();
      auto old_state = old_evt.getControlsState();

      // Migrate relevant fields from old CONTROLS_STATE to new SelfdriveState
      MessageBuilder msg;
      auto new_evt = msg.initEvent(old_evt.getValid());
      new_evt.setLogMonoTime(old_evt.getLogMonoTime());
      auto new_state = new_evt.initSelfdriveState();

      new_state.setActive(old_state.getActiveDEPRECATED());
      new_state.setAlertSize(old_state.getAlertSizeDEPRECATED());
      new_state.setAlertSound(old_state.getAlertSound2DEPRECATED());
      new_state.setAlertStatus(old_state.getAlertStatusDEPRECATED());
      new_state.setAlertText1(old_state.getAlertText1DEPRECATED());
      new_state.setAlertText2(old_state.getAlertText2DEPRECATED());
      new_state.setAlertType(old_state.getAlertTypeDEPRECATED());
      new_state.setEnabled(old_state.getEnabledDEPRECATED());
      new_state.setEngageable(old_state.getEngageableDEPRECATED());
      new_state.setExperimentalMode(old_state.getExperimentalModeDEPRECATED());
      new_state.setPersonality(old_state.getPersonalityDEPRECATED());
      new_state.setState(old_state.getStateDEPRECATED());

      // Serialize the new event to the buffer
      auto buf_size = msg.getSerializedSize();
      auto buf = buffer_.allocate(buf_size);
      msg.serializeToBuffer(reinterpret_cast<unsigned char *>(buf), buf_size);

      // Store the migrated event in the events list
      auto event_data = kj::arrayPtr(reinterpret_cast<const capnp::word *>(buf), buf_size);
      events.emplace_back(new_evt.which(), new_evt.getLogMonoTime(), event_data);
    }
  }
}
