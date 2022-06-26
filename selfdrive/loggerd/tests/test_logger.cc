#include <sys/stat.h>

#include <climits>
#include <condition_variable>
#include <sstream>
#include <thread>

#include "catch2/catch.hpp"
#include "cereal/messaging/messaging.h"
#include "common/util.h"
#include "selfdrive/loggerd/logger.h"
#include "selfdrive/ui/replay/util.h"

typedef cereal::Sentinel::SentinelType SentinelType;

void verify_segment(const std::string &route_path, int segment, int max_segment, int required_event_cnt) {
  const std::string segment_path = route_path + "--" + std::to_string(segment);
  SentinelType begin_sentinel = segment == 0 ? SentinelType::START_OF_ROUTE : SentinelType::START_OF_SEGMENT;
  SentinelType end_sentinel = segment == max_segment - 1 ? SentinelType::END_OF_ROUTE : SentinelType::END_OF_SEGMENT;

  REQUIRE(!util::file_exists(segment_path + "/rlog.lock"));
  for (const char *fn : {"/rlog", "/qlog"}) {
    const std::string log_file = segment_path + fn;
    std::string log = util::read_file(log_file);
    REQUIRE(!log.empty());
    int event_cnt = 0, i = 0;
    kj::ArrayPtr<const capnp::word> words((capnp::word *)log.data(), log.size() / sizeof(capnp::word));
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
        INFO("failed parse " << i << " excpetion :" << ex.getDescription());
        REQUIRE(0);
        break;
      }
    }
    REQUIRE(event_cnt == required_event_cnt);
  }
}

void write_msg(Logger *logger) {
  MessageBuilder msg;
  msg.initEvent().initClocks();
  auto bytes = msg.toBytes();
  logger->write(bytes.begin(), bytes.size(), true);
}

TEST_CASE("logger") {
  const std::string log_root = "/tmp/test_logger";
  system(("rm " + log_root + " -rf").c_str());

  ExitHandler do_exit;

  Logger logger(log_root);

  SECTION("single thread logging & rotation(100 segments, one thread)") {
    const int segment_cnt = 100;
    for (int i = 0; i < segment_cnt; ++i) {
      REQUIRE(logger.next());
      REQUIRE(util::file_exists(logger.segmentPath() + "/rlog.lock"));
      REQUIRE(logger.segment() == i);
      write_msg(&logger);
    }
    do_exit = true;
    do_exit.signal = 1;
    logger.close(do_exit.signal);
    for (int i = 0; i < segment_cnt; ++i) {
      verify_segment(log_root + "/" + logger.routeName(), i, segment_cnt, 1);
    }
  }
}
