#include <sys/stat.h>

#include <climits>
#include <condition_variable>
#include <sstream>
#include <thread>

#include "catch2/catch.hpp"
#include "cereal/messaging/messaging.h"
#include "selfdrive/common/util.h"
#include "selfdrive/loggerd/logger.h"
#include "selfdrive/ui/replay/util.h"

typedef cereal::Sentinel::SentinelType SentinelType;

void verify_segment(const std::string &route_path, int segment, int max_segment, int required_event_cnt) {
  const std::string segment_path = route_path + "--" + std::to_string(segment);
  SentinelType begin_sentinel = segment == 0 ? SentinelType::START_OF_ROUTE : SentinelType::START_OF_SEGMENT;
  SentinelType end_sentinel = segment == max_segment - 1 ? SentinelType::END_OF_ROUTE : SentinelType::END_OF_SEGMENT;

  for (const char *fn : {"/rlog.bz2", "/qlog.bz2"}) {
    const std::string log_file = segment_path + fn;
    std::string log = decompressBZ2(util::read_file(log_file));
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

void write_msg(LoggerHandle *logger) {
  MessageBuilder msg;
  msg.initEvent().initClocks();
  auto bytes = msg.toBytes();
  lh_log(logger, bytes.begin(), bytes.size(), true);
}

TEST_CASE("logger") {
  const std::string log_root = "/tmp/test_logger";
  system(("rm " + log_root + " -rf").c_str());

  ExitHandler do_exit;

  LoggerState logger = {};
  logger_init(&logger, "rlog", true);
  char segment_path[PATH_MAX] = {};
  int segment = -1;

  SECTION("single thread logging & rotation(100 segments, one thread)") {
    const int segment_cnt = 100;
    for (int i = 0; i < segment_cnt; ++i) {
      REQUIRE(logger_next(&logger, log_root.c_str(), segment_path, sizeof(segment_path), &segment) == 0);
      REQUIRE(segment == i);
      write_msg(logger.cur_handle);
    }
    do_exit = true;
    do_exit.signal = 1;
    logger_close(&logger, &do_exit);
    for (int i = 0; i < segment_cnt; ++i) {
      verify_segment(log_root + "/" + logger.route_name, i, segment_cnt, 1);
    }
  }
  SECTION("multiple threads logging & rotation(100 segments, 10 threads") {
    const int segment_cnt = 100, thread_cnt = 10;
    std::atomic<int> event_cnt[segment_cnt] = {};
    std::atomic<int> main_segment = -1;

    auto logging_thread = [&]() -> void {
      LoggerHandle *lh = logger_get_handle(&logger);
      REQUIRE(lh != nullptr);
      int segment = main_segment;
      int delayed_cnt = 0;
      while (!do_exit) {
        // write 2 more messages in the current segment and then rotate to the new segment.
        if (main_segment > segment && ++delayed_cnt == 2) {
          lh_close(lh);
          lh = logger_get_handle(&logger);
          segment = main_segment;
          delayed_cnt = 0;
        }
        write_msg(lh);
        event_cnt[segment] += 1;
        usleep(1);
      }
      lh_close(lh);
    };

    // start logging
    std::vector<std::thread> threads;
    for (int i = 0; i < segment_cnt; ++i) {
      REQUIRE(logger_next(&logger, log_root.c_str(), segment_path, sizeof(segment_path), &segment) == 0);
      REQUIRE(segment == i);
      main_segment = segment;
      if (i == 0) {
        for (int j = 0; j < thread_cnt; ++j) {
          threads.push_back(std::thread(logging_thread));
        }
      }
      for (int j = 0; j < 100; ++j) {
        write_msg(logger.cur_handle);
        usleep(1);
      }
      event_cnt[segment] += 100;
    }

    // end logging
    for (auto &t : threads) t.join();
    do_exit = true;
    do_exit.signal = 1;
    logger_close(&logger, &do_exit);
    REQUIRE(logger.cur_handle->refcnt == 0);

    for (int i = 0; i < segment_cnt; ++i) {
      verify_segment(log_root + "/" + logger.route_name, i, segment_cnt, event_cnt[i]);
    }
  }
}
