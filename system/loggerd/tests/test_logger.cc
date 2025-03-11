#include "catch2/catch.hpp"
#include "system/loggerd/logger.h"
#include "system/loggerd/loggerd.h"

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
  system(("rm " + log_root + " -rf").c_str());
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

TEST_CASE("RemoteEncoder::syncSegment robustness", "[sync]") {
  LoggerdState state;
  RemoteEncoder encoder;
  std::string name = "test_encoder";

  SECTION("Matching segment after offset set") {
    REQUIRE(encoder.syncSegment(&state, name, 5, 5) == true);
    REQUIRE(encoder.encoderd_segment_offset == 0);          // 5 - 5 = 0
    REQUIRE(encoder.current_segment == 5);
    REQUIRE(encoder.seen_first_packet == true);
    REQUIRE(encoder.marked_ready_to_rotate == false);
    REQUIRE(state.ready_to_rotate == 0);

    REQUIRE(encoder.syncSegment(&state, name, 7, 7) == true);  // 7 - 0 = 7 == 7
    REQUIRE(encoder.current_segment == 7);
    REQUIRE(encoder.recording == false);
    REQUIRE(encoder.marked_ready_to_rotate == false);
    REQUIRE(state.ready_to_rotate == 0);
  }

  SECTION("Encoder restarts and sends segment 0") {
    REQUIRE(encoder.syncSegment(&state, name, 1, 1) == true);
    REQUIRE(encoder.encoderd_segment_offset == 0);          // 1 - 1 = 0
    REQUIRE(encoder.current_segment == 1);

    REQUIRE(encoder.syncSegment(&state, name, 0, 2) == false);  // 0 - 0 = 0 < 2
    REQUIRE(encoder.encoderd_segment_offset == -2);         // Adjusted to 0 - 2
    REQUIRE(encoder.marked_ready_to_rotate == false);
    REQUIRE(state.ready_to_rotate == 0);

    REQUIRE(encoder.syncSegment(&state, name, 0, 2) == true);   // 0 - (-2) = 2 == 2
    REQUIRE(encoder.current_segment == 2);
    REQUIRE(encoder.marked_ready_to_rotate == false);
  }

  SECTION("Encoder restarts and sends segment greater than 0") {
    REQUIRE(encoder.syncSegment(&state, name, 0, 0) == true);
    REQUIRE(encoder.encoderd_segment_offset == 0);          // 0 - 0 = 0
    REQUIRE(encoder.current_segment == 0);

    REQUIRE(encoder.syncSegment(&state, name, 2, 3) == false);  // 2 - 0 = 2 < 3
    REQUIRE(encoder.encoderd_segment_offset == -1);         // Adjusted to 2 - 3
    REQUIRE(encoder.marked_ready_to_rotate == false);
    REQUIRE(state.ready_to_rotate == 0);

    REQUIRE(encoder.syncSegment(&state, name, 2, 3) == true);   // 2 - (-1) = 3 == 3
    REQUIRE(encoder.current_segment == 3);
  }

  SECTION("Logger restarts to lower segment") {
    REQUIRE(encoder.syncSegment(&state, name, 5, 5) == true);
    REQUIRE(encoder.encoderd_segment_offset == 0);
    REQUIRE(encoder.current_segment == 5);

    // Logger restarts to segment 0, encoder continues at 6
    REQUIRE(encoder.syncSegment(&state, name, 6, 0) == false);  // 6 - 0 = 6 > 0
    REQUIRE(encoder.marked_ready_to_rotate == true);
    REQUIRE(state.ready_to_rotate == 1);
    REQUIRE(encoder.current_segment == 5);                  // Unchanged until sync

    // Subsequent call with next segment
    REQUIRE(encoder.syncSegment(&state, name, 7, 0) == false);  // 7 - 0 = 7 > 0
    REQUIRE(state.ready_to_rotate == 1);                    // No further increment
  }

  SECTION("Encoder is ahead by more than one segment") {
    REQUIRE(encoder.syncSegment(&state, name, 0, 0) == true);
    REQUIRE(encoder.encoderd_segment_offset == 0);
    REQUIRE(encoder.current_segment == 0);

    REQUIRE(encoder.syncSegment(&state, name, 2, 0) == false);  // 2 - 0 = 2 > 0
    REQUIRE(encoder.marked_ready_to_rotate == true);
    REQUIRE(state.ready_to_rotate == 1);

    REQUIRE(encoder.syncSegment(&state, name, 3, 0) == false);  // 3 - 0 = 3 > 0
    REQUIRE(state.ready_to_rotate == 1);                    // No additional increment
  }

  SECTION("Sync after rotation") {
    REQUIRE(encoder.syncSegment(&state, name, 0, 0) == true);
    REQUIRE(encoder.encoderd_segment_offset == 0);

    REQUIRE(encoder.syncSegment(&state, name, 1, 0) == false);  // 1 - 0 = 1 > 0
    REQUIRE(encoder.marked_ready_to_rotate == true);
    REQUIRE(state.ready_to_rotate == 1);

    // Simulate rotation: logger catches up to encoder
    REQUIRE(encoder.syncSegment(&state, name, 1, 1) == true);   // 1 - 0 = 1 == 1
    REQUIRE(encoder.current_segment == 1);
    REQUIRE(encoder.marked_ready_to_rotate == false);
    REQUIRE(state.ready_to_rotate == 1);                    // Not decremented here
  }

  SECTION("Encoder catches up after being behind") {
    REQUIRE(encoder.syncSegment(&state, name, 0, 0) == true);
    REQUIRE(encoder.encoderd_segment_offset == 0);

    // Logger advances to 1, encoder sends 0
    REQUIRE(encoder.syncSegment(&state, name, 0, 1) == false);  // 0 - 0 = 0 < 1
    REQUIRE(encoder.encoderd_segment_offset == -1);         // Adjusted to 0 - 1

    // Logger advances to 2, encoder sends 1
    REQUIRE(encoder.syncSegment(&state, name, 1, 2) == true);   // 1 - (-1) = 2 == 2
    REQUIRE(encoder.current_segment == 2);
    REQUIRE(encoder.marked_ready_to_rotate == false);
  }
}
