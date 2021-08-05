#define CATCH_CONFIG_MAIN
#include <kj/array.h>

#include <climits>

#include "catch2/catch.hpp"
#include "cereal/messaging/messaging.h"
#include "selfdrive/common/util.h"
#include "selfdrive/loggerd/logger.h"

typedef cereal::Sentinel::SentinelType SentinelType;

const int LOG_COUNT = 1000;
const int ROTATE_CNT = 5;
std::mutex lock;
std::condition_variable cv;
std::condition_variable cv_finish;

namespace {

bool decompressBZ2(std::vector<uint8_t> &dest, const char srcData[], size_t srcSize,
                   size_t outputSizeIncrement = 0x100000U) {
  bz_stream strm = {};
  int ret = BZ2_bzDecompressInit(&strm, 0, 0);
  assert(ret == BZ_OK);
  dest.resize(1024 * 1024);
  strm.next_in = const_cast<char *>(srcData);
  strm.avail_in = srcSize;
  do {
    strm.next_out = (char *)&dest[strm.total_out_lo32];
    strm.avail_out = dest.size() - strm.total_out_lo32;
    ret = BZ2_bzDecompress(&strm);
    if (ret == BZ_OK && strm.avail_in > 0 && strm.avail_out == 0) {
      dest.resize(dest.size() + outputSizeIncrement);
    }
  } while (ret == BZ_OK && strm.avail_in > 0);

  BZ2_bzDecompressEnd(&strm);
  dest.resize(strm.total_out_lo32);
  return ret == BZ_STREAM_END;
}

void verify_logfiles(const std::string &segment_path, uint64_t boottime, uint64_t monotonic, uint64_t required_sum, SentinelType begin_type, SentinelType end_type) {
  for (const char *fn : {"/rlog.bz2", "/qlog.bz2"}) {
    const std::string log_file = segment_path + fn;
    INFO(log_file);
    // if fn is still opened by LoggerHandle after logger_close, log_bz2.size() is zero
    std::string log_bz2 = util::read_file(log_file);
    REQUIRE(log_bz2.size() > 0);

    std::vector<uint8_t> log;
    bool ret = decompressBZ2(log, log_bz2.data(), log_bz2.size());
    REQUIRE(ret);

    uint64_t sum = 0;
    int i = 0;
    kj::ArrayPtr<const capnp::word> words((capnp::word *)log.data(), log.size() / sizeof(capnp::word));
    while (words.size() > 0) {
      try {
        capnp::FlatArrayMessageReader reader(words);
        auto event = reader.getRoot<cereal::Event>();
        if (i == 0) {
          REQUIRE(event.which() == cereal::Event::INIT_DATA);
        } else if (i == 1) {
          REQUIRE(event.which() == cereal::Event::SENTINEL);
          auto sentinel = event.getSentinel();
          REQUIRE(sentinel.getType() == begin_type);
        } else if (event.which() == cereal::Event::CLOCKS) {
          auto clocks = event.getClocks();
          REQUIRE(clocks.getBootTimeNanos() == boottime);
          REQUIRE(clocks.getMonotonicNanos() == monotonic);
          sum += clocks.getModemUptimeMillis();
        }
        words = kj::arrayPtr(reader.getEnd(), words.end());
        if (words.size() == 0) {
          // the last event should be SENTINEL

          // TODO: this check failed sometimes. need to be fixed in LoggerState.
          REQUIRE(event.which() == cereal::Event::SENTINEL);
          auto sentinel = event.getSentinel();
          REQUIRE(sentinel.getType() == end_type);
        }
        ++i;
      } catch (const kj::Exception& ex) {
        INFO("failed parse " << i << " excpetion :" << ex.getDescription());
        REQUIRE(0);
        break;
      }
    }
    REQUIRE(sum == required_sum);
  }
}

void logger_thread(LoggerState *log, int *threads_writing, int segment_cnt, uint64_t boottime, uint64_t monotonic, int number) {
  int curseg = -1;
  for (int cnt = 0; cnt < segment_cnt; ++cnt) {
    LoggerHandle *lh = nullptr;
    {
      std::unique_lock lk(lock);
      cv.wait(lk, [=]() { return log->part > curseg; });
      curseg = log->part;
      *threads_writing += 1;
      lh = logger_get_handle(log);
      cv_finish.notify_one();
    }
    for (int i = 0; i < LOG_COUNT; ++i) {
      MessageBuilder msg;
      auto clocks = msg.initEvent().initClocks();
      clocks.setBootTimeNanos(boottime);
      clocks.setMonotonicNanos(monotonic);
      // this field is used to calculate the sum to ensure that each thread has written all events
      clocks.setModemUptimeMillis(number);
      auto bytes = msg.toBytes();
      lh_log(lh, bytes.begin(), bytes.size(), true);
      usleep(0);
    }
    lh_close(lh);
  }
}

void test_rotate(int thread_cnt, LoggerState *logger, const std::string &log_root, int *threads_writing, uint64_t boottime, uint64_t monotonic) {
  std::vector<std::thread> threads;
  for (uint8_t i = 1; i <= thread_cnt; ++i) {
    threads.push_back(std::thread(logger_thread, logger, threads_writing, ROTATE_CNT, boottime, monotonic, i));
  }

  char segment_path[PATH_MAX];
  int part = -1;
  // rotate ROTATE_CNT times
  for (int i = 0; i < ROTATE_CNT; ++i) {
    int ret = logger_next(logger, log_root.c_str(), segment_path, std::size(segment_path), &part);
    REQUIRE(ret == 0);
    cv.notify_all();
    std::unique_lock lk(lock);
    // rotate to the next segment if all threads are writing in the current segment.
    cv_finish.wait(lk, [=]() { return *threads_writing == thread_cnt; });
    *threads_writing = 0;
  }
  for (auto &t : threads) {
    t.join();
  }
  logger_close(logger);

  std::string full_path = segment_path;
  size_t pos = full_path.rfind("-");
  std::string log_base = full_path.substr(0, pos + 1);
  uint64_t required_sum = 0;
  for (int i = 1; i <= thread_cnt; ++i) {
    required_sum += i * LOG_COUNT;
  }
  for (int i = 0; i < ROTATE_CNT; ++i) {
    SentinelType begin_type = i == 0 ? SentinelType::START_OF_ROUTE : SentinelType::START_OF_SEGMENT;
    SentinelType end_type = i == ROTATE_CNT - 1 ? SentinelType::END_OF_ROUTE : SentinelType::END_OF_SEGMENT;
    verify_logfiles(log_base + std::to_string(i), boottime, monotonic, required_sum, begin_type, end_type);
  }
}
}  // namespace

TEST_CASE("logger") {
  const std::string log_root = "/tmp/log_root";
  
  uint64_t boottime = nanos_since_boot();
  uint64_t monotonic = nanos_monotonic();
  int threads_writing = 0;
  LoggerState logger = {};
  logger_init(&logger, "rlog", true);

  SECTION("one thread logging and rotation") {
    system("rm /tmp/log_root/* -rf");
    test_rotate(1, &logger, log_root, &threads_writing, boottime, monotonic);
  }
  SECTION("two threads logging and rotation") {
    test_rotate(2, &logger, log_root, &threads_writing, boottime, monotonic);
  }
  SECTION("three threads logging and rotation") {
    test_rotate(3, &logger, log_root, &threads_writing, boottime, monotonic);
  }
  SECTION("four threads logging and rotation") {
    test_rotate(4, &logger, log_root, &threads_writing, boottime, monotonic);
  }
}
