#include "catch2/catch.hpp"
#include "cereal/messaging/impl_msgq.h"
#include "cereal/messaging/messaging.h"
#include "common/util.h"

#define private public  // access prvate members for testing purpose
#include "system/loggerd/encoder_writer.h"

static Message *generate_msg(const EncoderInfo &info, uint32_t segment_num, uint32_t frame_id, uint32_t flags = 0) {
  cereal::EncodeData::Builder (cereal::Event::Builder::*initEncoder)();
  if (strcmp(info.publish_name, "roadEncodeData") == 0) {
    initEncoder = &cereal::Event::Builder::initRoadEncodeData;
  } else if (strcmp(info.publish_name, "driverEncodeData") == 0) {
    initEncoder = &cereal::Event::Builder::initDriverEncodeData;
  } else if (strcmp(info.publish_name, "wideRoadEncodeData") == 0) {
    initEncoder = &cereal::Event::Builder::initWideRoadEncodeData;
  } else if (strcmp(info.publish_name, "qRoadEncodeData") == 0) {
    initEncoder = &cereal::Event::Builder::initQRoadEncodeData;
  }

  MessageBuilder msg;
  auto event = msg.initEvent(true);
  cereal::EncodeData::Builder edat = (event.*initEncoder)();
  auto edata = edat.initIdx();
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  uint64_t tt = (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
  edat.setUnixTimestampNanos(tt);
  edata.setFrameId(frame_id);
  edata.setTimestampSof(tt);
  edata.setTimestampEof(tt);
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
  int segment = -1;
  char segment_path[4096];
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
          int err = logger_next(logger, LOG_ROOT.c_str(), segment_path, sizeof(segment_path), &segment);
          REQUIRE(err == 0);
          for (auto &en : encoders) {
            en->rotate(segment_path);
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
}

TEST_CASE("EncoderWriter") {
  LoggerState logger{};
  logger_init(&logger, true);

  int segment = -1;
  char segment_path[4096];
  int err = logger_next(&logger, LOG_ROOT.c_str(), segment_path, sizeof(segment_path), &segment);
  REQUIRE(err == 0);

  std::vector<std::unique_ptr<EncoderWriter>> encoders;
  for (const auto &info : {main_road_encoder_info, main_wide_road_encoder_info, main_driver_encoder_info}) {
    encoders.emplace_back(std::make_unique<EncoderWriter>(segment_path, info));
  }

  SECTION("encoderd run after loggerd") {
    test(&logger, encoders, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  }
  SECTION("encoderd run before loggerd") {
    test(&logger, encoders, {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  }
  SECTION("encoderd restarted in the middle)") {
    test(&logger, encoders, {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5});
  }
  SECTION("random encoder segment id)") {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, 50);
    std::vector<int> segments;
    for (int n = 0; n < 40; ++n) {
      segments.push_back(distr(gen));
    }
    test(&logger, encoders, segments);
  }

  logger_close(&logger, nullptr);
}
