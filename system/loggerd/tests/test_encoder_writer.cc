#include "catch2/catch.hpp"
#include "cereal/messaging/impl_msgq.h"
#include "cereal/messaging/messaging.h"
#include "common/util.h"

#define private public  // access private members for testing purpose
#include "system/loggerd/encoder_writer.h"

static Message *generate_msg(const EncoderInfo &info, uint32_t segment_num, uint32_t frame_id, uint32_t flags = 0) {
  cereal::EncodeData::Builder (cereal::Event::Builder::*initEncodeData)() = initEncodeData = &cereal::Event::Builder::initRoadEncodeData;
  if (strcmp(info.publish_name, "driverEncodeData") == 0) {
    initEncodeData = &cereal::Event::Builder::initDriverEncodeData;
  } else if (strcmp(info.publish_name, "wideRoadEncodeData") == 0) {
    initEncodeData = &cereal::Event::Builder::initWideRoadEncodeData;
  } else if (strcmp(info.publish_name, "qRoadEncodeData") == 0) {
    initEncodeData = &cereal::Event::Builder::initQRoadEncodeData;
  }

  MessageBuilder msg;
  auto event = msg.initEvent(true);
  cereal::EncodeData::Builder edat = (event.*initEncodeData)();
  auto edata = edat.initIdx();
  uint64_t ts = nanos_since_boot();
  edat.setUnixTimestampNanos(ts);
  edata.setFrameId(frame_id);
  edata.setTimestampSof(ts);
  edata.setTimestampEof(ts);
  edata.setEncodeId(frame_id);
  edata.setSegmentNum(segment_num);
  edata.setSegmentId(frame_id);
  edata.setFlags(flags);

  MSGQMessage *m = new MSGQMessage();
  auto bytes = msg.toBytes();
  m->init((char *)bytes.begin(), bytes.size());
  return m;
}

static void test(LoggerState *logger, std::vector<std::unique_ptr<EncoderWriter>> &encoders, const std::vector<int> &segments) {
  const int frames = 10;
  int prev_seg = -1;
  for (auto seg : segments) {
    for (int i = 0; i < frames; ++i) {
      bool seg_changed = prev_seg >= 0 && seg != prev_seg;
      for (auto &e : encoders) {
        e->write(logger, generate_msg(e->info, seg, seg * frames + i, 0));
        REQUIRE(e->remote_encoder_segment == seg);
        if (seg_changed) {
          REQUIRE(e->marked_ready_to_rotate == true);
          REQUIRE(e->current_encoder_segment != e->remote_encoder_segment);
          REQUIRE(e->q.size() > 0);
          REQUIRE(e->ready_to_rotate > 0);
        } else {
          REQUIRE(e->marked_ready_to_rotate == false);
          REQUIRE(e->current_encoder_segment == e->remote_encoder_segment);
          REQUIRE(e->q.size() == 0);
        }

        if (e->ready_to_rotate == encoders.size()) {
          // q should not empty
          REQUIRE(e->q.size() > 0);
          REQUIRE(logger->next());
          for (auto &en : encoders) {
            en->rotate(logger->segmentPath());
          }
          e->ready_to_rotate = 0;
        }
      }
      prev_seg = seg;
    }
  }

  for (auto &e : encoders) {
    e->flush(logger);
    REQUIRE(e->q.size() == 0);
  }

  REQUIRE(segments.size() == logger->segment() + 1);
}

TEST_CASE("EncoderWriter") {
  LoggerState logger;
  REQUIRE(logger.next());
  std::vector<std::unique_ptr<EncoderWriter>> encoders;
  for (const auto &info : {main_road_encoder_info, main_wide_road_encoder_info, main_driver_encoder_info}) {
    encoders.emplace_back(std::make_unique<EncoderWriter>(logger.segmentPath(), info));
  }

  SECTION("encoderd run after loggerd") {
    test(&logger, encoders, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  }
  SECTION("encoderd run before loggerd") {
    test(&logger, encoders, {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  }
  SECTION("encoderd restarted in the middle)") {
    test(&logger, encoders, {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5});
  }
}
