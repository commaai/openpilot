#include "catch2/catch.hpp"
#include "selfdrive/loggerd/logger.h"
#include "selfdrive/ui/replay/util.h"

void verify_segment(const std::string &route_path, int segment, int max_segment, int required_event_cnt) {
  const std::string segment_path = route_path + "--" + std::to_string(segment);
  SentinelType begin_sentinel = segment == 0 ? SentinelType::START_OF_ROUTE : SentinelType::START_OF_SEGMENT;
  SentinelType end_sentinel = segment == max_segment - 1 ? SentinelType::END_OF_ROUTE : SentinelType::END_OF_SEGMENT;

  REQUIRE(!util::file_exists(segment_path + "/.lock"));
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

TEST_CASE("logger") {
  const std::string log_root = "/tmp/test_logger";
  system(("rm " + log_root + " -rf").c_str());
  ExitHandler do_exit;

  SECTION("test Logger") {
    const int segment_cnt = 100;
    std::atomic<int> event_cnt[segment_cnt] = {};
    LoggerManager logger_manager(log_root);
    std::shared_ptr main_logger = logger_manager.next();

    auto logging_thread = [&]() -> void {
      while (!do_exit) {
        std::shared_ptr logger = main_logger;
        MessageBuilder msg;
        msg.initEvent().initClocks();
        logger->write(msg.toBytes(), true);
        event_cnt[logger->segment()] += 1;
      }
    };

    // start logging
    std::vector<std::thread> threads;
    for (int i = 0; i < 5; ++i) {
      threads.emplace_back(logging_thread);
    }
    for (int i = 1; i < segment_cnt; ++i) {
      main_logger = logger_manager.next();
      REQUIRE(main_logger->segment() == i);
      util::sleep_for(50);
    }

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
