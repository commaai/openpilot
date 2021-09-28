#include "selfdrive/ui/replay/logreader.h"

#include <cassert>
#include <bzlib.h>
#include "selfdrive/common/util.h"

static bool decompressBZ2(std::vector<uint8_t> &dest, const char srcData[], size_t srcSize,
                          size_t outputSizeIncrement = 0x100000U) {
  bz_stream strm = {};
  int ret = BZ2_bzDecompressInit(&strm, 0, 0);
  assert(ret == BZ_OK);

  strm.next_in = const_cast<char *>(srcData);
  strm.avail_in = srcSize;
  do {
    strm.next_out = (char *)&dest[strm.total_out_lo32];
    strm.avail_out = dest.size() - strm.total_out_lo32;
    ret = BZ2_bzDecompress(&strm);
    if (ret == BZ_OK && strm.avail_in > 0 && strm.avail_out == 0) {
      dest.resize(dest.size() + outputSizeIncrement);
    }
  } while (ret == BZ_OK);

  BZ2_bzDecompressEnd(&strm);
  dest.resize(strm.total_out_lo32);
  return ret == BZ_STREAM_END;
}

LogReader::~LogReader() {
  for (auto e : events) delete e;
}

bool LogReader::load(const std::string &file) {
  raw_.resize(1024 * 1024 * 64);
  std::string dat = util::read_file(file);
  if (dat.empty() || !decompressBZ2(raw_, dat.data(), dat.size())) {
    LOGW("bz2 decompress failed");
    return false;
  }

  std::unordered_map<uint32_t, uint64_t> frame_monotime[MAX_CAMERAS] = {};
  std::unordered_map<uint32_t, std::pair<const Event *, cereal::EncodeIndex::Reader>> eidx_map[MAX_CAMERAS] = {};
  auto insertEidx = [&](CameraType type, const cereal::EncodeIndex::Reader &reader, const Event* evt) {
    eidx_map[type][reader.getFrameId()] = {evt, reader};
  };

  kj::ArrayPtr<const capnp::word> words((const capnp::word *)raw_.data(), raw_.size() / sizeof(capnp::word));
  while (words.size() > 0) {
    try {
      std::unique_ptr<Event> evt = std::make_unique<Event>(words);
      switch (evt->which) {
        case cereal::Event::ROAD_CAMERA_STATE:
          frame_monotime[RoadCam][evt->event.getRoadCameraState().getFrameId()] = evt->mono_time;
          break;
        case cereal::Event::DRIVER_CAMERA_STATE:
          frame_monotime[DriverCam][evt->event.getDriverCameraState().getFrameId()] = evt->mono_time;
          break;
        case cereal::Event::WIDE_ROAD_CAMERA_STATE:
          frame_monotime[WideRoadCam][evt->event.getWideRoadCameraState().getFrameId()] = evt->mono_time;
          break;
        case cereal::Event::ROAD_ENCODE_IDX:
          insertEidx(RoadCam, evt->event.getRoadEncodeIdx(), evt.get());
          break;
        case cereal::Event::DRIVER_ENCODE_IDX:
          insertEidx(DriverCam, evt->event.getDriverEncodeIdx(), evt.get());
          break;
        case cereal::Event::WIDE_ROAD_ENCODE_IDX:
          insertEidx(WideRoadCam, evt->event.getWideRoadEncodeIdx(), evt.get());
          break;
        default:
          break;
      }
      words = kj::arrayPtr(evt->reader->getEnd(), words.end());
      events.push_back(evt.release());
    } catch (const kj::Exception &e) {
      return false;
    }
  }

  // insert custom eidx events for publish frames.
  for (auto cam : ALL_CAMERAS) {
    for (auto [frame_id, val] : eidx_map[cam]) {
      auto [e, eidx] = val;

      MessageBuilder builder;
      builder.setRoot(e->event);
      auto event = builder.getRoot<cereal::Event>();

      auto it = frame_monotime[cam].find(frame_id);
      if (it != frame_monotime[cam].end()) {
        // set time to frame's monotime
        event.setLogMonoTime(it->second);
      } else if (eidx.getTimestampEof()) {
        // set time to timestampEof
        event.setLogMonoTime(eidx.getTimestampEof());
      } else if (eidx.getTimestampSof()) {
        // set time to timestampSof
        event.setLogMonoTime(eidx.getTimestampSof());
      }
      events.push_back(new Event(builder));
    }
  }
  std::sort(events.begin(), events.end(), Event::lessThan());
  return true;
}
