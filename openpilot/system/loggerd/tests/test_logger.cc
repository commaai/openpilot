#include "catch2/catch.hpp"
#include "system/loggerd/logger.h"
#include "system/loggerd/video_writer.h"

typedef cereal::Sentinel::SentinelType SentinelType;

void verify_segment(const std::string &route_path, int segment, int max_segment, int required_event_cnt) {
  const std::string segment_path = route_path + "--" + std::to_string(segment);
  SentinelType begin_sentinel = segment == 0 ? SentinelType::START_OF_ROUTE : SentinelType::START_OF_SEGMENT;
  SentinelType end_sentinel = segment == max_segment - 1 ? SentinelType::END_OF_ROUTE : SentinelType::END_OF_SEGMENT;

  REQUIRE(!util::file_exists(segment_path + "/rlog.lock"));
  for (const char *fn : {"/rlog.zst", "/qlog.zst"}) {
    const std::string log_file = segment_path + fn;
    std::string log = util::read_file(log_file);
    REQUIRE(!log.empty());
    std::string decompressed_log = zstd_decompress(log);
    int event_cnt = 0, i = 0;
    kj::ArrayPtr<const capnp::word> words((capnp::word *)decompressed_log.data(), decompressed_log.size() / sizeof(capnp::word));
    while (words.size() > 0) {
      try {
        capnp::FlatArrayMessageReader reader(words);
        auto event = reader.getRoot<cereal::Event>();
        words = kj::arrayPtr(reader.getEnd(), words.end());
        if (i == 0) {
          REQUIRE(event.which() == cereal::Event::INIT_DATA);
        } else if (i == 1) {
          REQUIRE(event.which() == cereal::Event::SENTINEL);
          REQUIRE(event.getSentinel().getType() == begin_sentinel);
          REQUIRE(event.getSentinel().getSignal() == 0);
        } else if (words.size() > 0) {
          REQUIRE(event.which() == cereal::Event::CLOCKS);
          ++event_cnt;
        } else {
          // the last event must be SENTINEL
          REQUIRE(event.which() == cereal::Event::SENTINEL);
          REQUIRE(event.getSentinel().getType() == end_sentinel);
          REQUIRE(event.getSentinel().getSignal() == (end_sentinel == SentinelType::END_OF_ROUTE ? 1 : 0));
        }
        ++i;
      } catch (const kj::Exception &ex) {
        INFO("failed parse " << i << " exception :" << ex.getDescription());
        REQUIRE(0);
        break;
      }
    }
    REQUIRE(event_cnt == required_event_cnt);
  }
}

void write_msg(LoggerState *logger) {
  MessageBuilder msg;
  msg.initEvent().initClocks();
  logger->write(msg.toBytes(), true);
}

TEST_CASE("logger") {
  const int segment_cnt = 100;
  const std::string log_root = "/tmp/test_logger";
  REQUIRE(system(("rm -rf " + log_root).c_str()) == 0);
  std::string route_name;
  {
    LoggerState logger(log_root);
    route_name = logger.routeName();
    for (int i = 0; i < segment_cnt; ++i) {
      REQUIRE(logger.next());
      REQUIRE(util::file_exists(logger.segmentPath() + "/rlog.lock"));
      REQUIRE(logger.segment() == i);
      write_msg(&logger);
    }
    logger.setExitSignal(1);
  }
  for (int i = 0; i < segment_cnt; ++i) {
    verify_segment(log_root + "/" + route_name, i, segment_cnt, 1);
  }
}

TEST_CASE("mark previous segment incomplete restores completion lock") {
  const std::string log_root = "/tmp/test_logger_previous_segment_incomplete";
  REQUIRE(system(("rm -rf " + log_root).c_str()) == 0);
  std::string previous_segment_path;
  {
    LoggerState logger(log_root);
    REQUIRE(logger.next());
    previous_segment_path = logger.segmentPath();
    write_msg(&logger);

    REQUIRE(logger.next());
    REQUIRE(!util::file_exists(previous_segment_path + "/rlog.lock"));
    REQUIRE(util::file_exists(logger.segmentPath() + "/rlog.lock"));

    REQUIRE(logger.mark_previous_segment_incomplete());
    REQUIRE(util::file_exists(previous_segment_path + "/rlog.lock"));
  }

  REQUIRE(util::file_exists(previous_segment_path + "/rlog.lock"));
  REQUIRE(system(("rm -rf " + log_root).c_str()) == 0);
}

TEST_CASE("empty video keeps completion lock") {
  const std::string path = "/tmp/test_empty_video_writer";
  REQUIRE(system(("rm -rf " + path).c_str()) == 0);
  REQUIRE(util::create_directories(path, 0775));

  {
    VideoWriter writer(path.c_str(), "empty.hevc", false, 64, 64, 20, cereal::EncodeIndex::Type::FULL_H_E_V_C);
    REQUIRE(!writer.close());
    REQUIRE(util::file_exists(path + "/empty.hevc.lock"));
  }

  REQUIRE(system(("rm -rf " + path).c_str()) == 0);
}
