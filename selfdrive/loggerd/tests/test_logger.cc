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

  REQUIRE(!util::file_exists(segment_path + "/log.lock"));
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

void write_msg(Logger *logger) {
  MessageBuilder msg;
  msg.initEvent().initClocks();
  auto bytes = msg.toBytes();
  logger->write(bytes.begin(), bytes.size(), true);
}

TEST_CASE("logger") {
  const std::string log_root = "/tmp/test_logger";
  system(("rm " + log_root + " -rf").c_str());
  LoggerManager logger_manager(log_root);

  SECTION("single thread logging & rotation(100 segments, one thread)") {
    const int segment_cnt = 100;
    for (int i = 0; i < segment_cnt; ++i) {
      std::shared_ptr logger =  logger_manager.next();
      REQUIRE(util::file_exists(logger->segmentPath() + "/log.lock"));
      REQUIRE(logger->segment() == i);
      write_msg(logger.get());
      if (i == segment_cnt - 1) {
        logger->end_of_route(1);
      }
    }
    for (int i = 0; i < segment_cnt; ++i) {
      verify_segment(logger_manager.routePath(), i, segment_cnt, 1);
    }
  }
  SECTION("multiple threads logging & rotation(100 segments, 10 threads") {
    const int segment_cnt = 100, thread_cnt = 10;
    std::atomic<int> event_cnt[segment_cnt] = {};
    std::atomic<bool> do_exit = false;

    auto logging_thread = [&](std::shared_ptr<Logger> &main_logger) -> void {
      std::shared_ptr logger = main_logger;
      int delayed_cnt = 0;
      while (!do_exit) {
        // write 2 more messages in the current segment and then rotate to the new segment.
        if (main_logger != logger && ++delayed_cnt == 2) {
          logger = main_logger;
          delayed_cnt = 0;
        }
        write_msg(logger.get());
        event_cnt[logger->segment()] += 1;
        usleep(1);
      }
    };

    // start logging
    std::vector<std::thread> threads;
    std::shared_ptr<Logger> main_logger;
    for (int i = 0; i < segment_cnt; ++i) {
      main_logger = logger_manager.next();
      REQUIRE(main_logger->segment() == i);
      if (i == 0) {
        for (int j = 0; i < thread_cnt; ++j) {
          threads.push_back(std::thread(logging_thread, std::ref(main_logger)));
        }
      }
      for (int j = 0; j < 100; ++j) {
        write_msg(main_logger.get());
        usleep(1);
      }
      event_cnt[main_logger->segment()] += 100;
    }

    // end logging
    do_exit = true;
    for (auto &t : threads) t.join();
    main_logger->end_of_route(true);
    REQUIRE(main_logger.use_count() == 1);
    main_logger = nullptr;

    for (int i = 0; i < segment_cnt; ++i) {
      verify_segment(logger_manager.routePath(), i, segment_cnt, event_cnt[i]);
    }
  }
}
