#include <chrono>
#include <thread>

#include <QDebug>
#include <QEventLoop>

#include "catch2/catch.hpp"
#include "common/util.h"
#include "tools/replay/replay.h"
#include "tools/replay/util.h"

const std::string TEST_RLOG_URL = "https://commadataci.blob.core.windows.net/openpilotci/0c94aa1e1296d7c6/2021-05-05--19-48-37/0/rlog.bz2";
const std::string TEST_RLOG_CHECKSUM = "5b966d4bb21a100a8c4e59195faeb741b975ccbe268211765efd1763d892bfb3";

const int TEST_REPLAY_SEGMENTS = std::getenv("TEST_REPLAY_SEGMENTS") ? atoi(std::getenv("TEST_REPLAY_SEGMENTS")) : 1;

bool download_to_file(const std::string &url, const std::string &local_file, int chunk_size = 5 * 1024 * 1024, int retries = 3) {
  do {
    if (httpDownload(url, local_file, chunk_size)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  } while (--retries >= 0);
  return false;
}

TEST_CASE("httpMultiPartDownload") {
  char filename[] = "/tmp/XXXXXX";
  close(mkstemp(filename));

  const size_t chunk_size = 5 * 1024 * 1024;
  std::string content;
  SECTION("download to file") {
    REQUIRE(download_to_file(TEST_RLOG_URL, filename, chunk_size));
    content = util::read_file(filename);
  }
  SECTION("download to buffer") {
    for (int i = 0; i < 3 && content.empty(); ++i) {
      content = httpGet(TEST_RLOG_URL, chunk_size);
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    REQUIRE(!content.empty());
  }
  REQUIRE(content.size() == 9112651);
  REQUIRE(sha256(content) == TEST_RLOG_CHECKSUM);
}

TEST_CASE("FileReader") {
  auto enable_local_cache = GENERATE(true, false);
  std::string cache_file = cacheFilePath(TEST_RLOG_URL);
  system(("rm " + cache_file + " -f").c_str());

  FileReader reader(enable_local_cache);
  std::string content = reader.read(TEST_RLOG_URL);
  REQUIRE(sha256(content) == TEST_RLOG_CHECKSUM);
  if (enable_local_cache) {
    REQUIRE(sha256(util::read_file(cache_file)) == TEST_RLOG_CHECKSUM);
  } else {
    REQUIRE(util::file_exists(cache_file) == false);
  }
}

TEST_CASE("LogReader") {
  SECTION("corrupt log") {
    FileReader reader(true);
    std::string corrupt_content = reader.read(TEST_RLOG_URL);
    corrupt_content.resize(corrupt_content.length() / 2);
    corrupt_content = decompressBZ2(corrupt_content);
    LogReader log;
    REQUIRE(log.load((std::byte *)corrupt_content.data(), corrupt_content.size()));
    REQUIRE(log.events.size() > 0);
  }
}

void read_segment(int n, const SegmentFile &segment_file, uint32_t flags) {
  QEventLoop loop;
  Segment segment(n, segment_file, flags);
  QObject::connect(&segment, &Segment::loadFinished, [&]() {
    REQUIRE(segment.isLoaded() == true);
    REQUIRE(segment.log != nullptr);
    REQUIRE(segment.frames[RoadCam] != nullptr);
    if (flags & REPLAY_FLAG_DCAM) {
      REQUIRE(segment.frames[DriverCam] != nullptr);
    }
    if (flags & REPLAY_FLAG_ECAM) {
      REQUIRE(segment.frames[WideRoadCam] != nullptr);
    }

    // test LogReader & FrameReader
    REQUIRE(segment.log->events.size() > 0);
    REQUIRE(std::is_sorted(segment.log->events.begin(), segment.log->events.end(), Event::lessThan()));

    for (auto cam : ALL_CAMERAS) {
      auto &fr = segment.frames[cam];
      if (!fr) continue;

      if (cam == RoadCam || cam == WideRoadCam) {
        REQUIRE(fr->getFrameCount() == 1200);
      }
      auto [nv12_width, nv12_height, nv12_buffer_size] = get_nv12_info(fr->width, fr->height);
      VisionBuf buf;
      buf.allocate(nv12_buffer_size);
      buf.init_yuv(fr->width, fr->height, nv12_width, nv12_width * nv12_height);
      // sequence get 100 frames
      for (int i = 0; i < 100; ++i) {
        REQUIRE(fr->get(i, &buf));
      }
    }

    loop.quit();
  });
  loop.exec();
}

std::string download_demo_route() {
  static std::string data_dir;

  if (data_dir == "") {
    char tmp_path[] = "/tmp/root_XXXXXX";
    data_dir = mkdtemp(tmp_path);

    Route remote_route(DEMO_ROUTE);
    assert(remote_route.load());

    // Create a local route from remote for testing
    const std::string route_name = DEMO_ROUTE.mid(17).toStdString();
    for (int i = 0; i < 2; ++i) {
      std::string log_path = util::string_format("%s/%s--%d/", data_dir.c_str(), route_name.c_str(), i);
      util::create_directories(log_path, 0755);
      REQUIRE(download_to_file(remote_route.at(i).rlog.toStdString(), log_path + "rlog.bz2"));
      REQUIRE(download_to_file(remote_route.at(i).qcamera.toStdString(), log_path + "qcamera.ts"));
    }
  }

  return data_dir;
}


TEST_CASE("Local route") {
  std::string data_dir = download_demo_route();

  auto flags = GENERATE(0, REPLAY_FLAG_QCAMERA);
  Route route(DEMO_ROUTE, QString::fromStdString(data_dir));
  REQUIRE(route.load());
  REQUIRE(route.segments().size() == 2);
  for (int i = 0; i < TEST_REPLAY_SEGMENTS; ++i) {
    read_segment(i, route.at(i), flags);
  }
}

TEST_CASE("Remote route") {
  auto flags = GENERATE(0, REPLAY_FLAG_QCAMERA);
  Route route(DEMO_ROUTE);
  REQUIRE(route.load());
  REQUIRE(route.segments().size() == 13);
  for (int i = 0; i < TEST_REPLAY_SEGMENTS; ++i) {
    read_segment(i, route.at(i), flags);
  }
}

// helper class for unit tests
class TestReplay : public Replay {
 public:
  TestReplay(const QString &route, uint32_t flags = REPLAY_FLAG_NO_FILE_CACHE | REPLAY_FLAG_NO_VIPC) : Replay(route, {}, {}, nullptr, flags) {}
  void test_seek();
  void testSeekTo(int seek_to);
};

void TestReplay::testSeekTo(int seek_to) {
  seekTo(seek_to, false);

  while (true) {
    std::unique_lock lk(stream_lock_);
    stream_cv_.wait(lk, [=]() { return events_updated_ == true; });
    events_updated_ = false;
    if (cur_mono_time_ != route_start_ts_ + seek_to * 1e9) {
      // wake up by the previous merging, skip it.
      continue;
    }

    Event cur_event(cereal::Event::Which::INIT_DATA, cur_mono_time_);
    auto eit = std::upper_bound(events_->begin(), events_->end(), &cur_event, Event::lessThan());
    if (eit == events_->end()) {
      qDebug() << "waiting for events...";
      continue;
    }

    REQUIRE(std::is_sorted(events_->begin(), events_->end(), Event::lessThan()));
    const int seek_to_segment = seek_to / 60;
    const int event_seconds = ((*eit)->mono_time - route_start_ts_) / 1e9;
    current_segment_ = event_seconds / 60;
    INFO("seek to [" << seek_to << "s segment " << seek_to_segment << "], events [" << event_seconds << "s segment" << current_segment_ << "]");
    REQUIRE(event_seconds >= seek_to);
    if (event_seconds > seek_to) {
      auto it = segments_.lower_bound(seek_to_segment);
      REQUIRE(it->first == current_segment_);
    }
    break;
  }
}

void TestReplay::test_seek() {
  // create a dummy stream thread
  stream_thread_ = new QThread(this);
  QEventLoop loop;
  std::thread thread = std::thread([&]() {
    for (int i = 0; i < 10; ++i) {
      testSeekTo(util::random_int(0, 2 * 60));
    }
    loop.quit();
  });
  loop.exec();
  thread.join();
}

TEST_CASE("Replay") {
  TestReplay replay(DEMO_ROUTE);
  REQUIRE(replay.load());
  replay.test_seek();
}
