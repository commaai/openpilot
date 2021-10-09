#include <QCryptographicHash>
#include <QDebug>
#include <QEventLoop>

#include "catch2/catch.hpp"
#include "selfdrive/ui/replay/replay.h"
#include "selfdrive/ui/replay/util.h"

const QString DEMO_ROUTE = "4cf7a6ad03080c90|2021-09-29--13-46-36";
std::string sha_256(const QString &dat) {
  return QString(QCryptographicHash::hash(dat.toUtf8(), QCryptographicHash::Sha256).toHex()).toStdString();
}

TEST_CASE("httpMultiPartDownload") {
  char filename[] = "/tmp/XXXXXX";
  close(mkstemp(filename));

  const char *stream_url = "https://commadataci.blob.core.windows.net/openpilotci/0c94aa1e1296d7c6/2021-05-05--19-48-37/0/rlog.bz2";
  SECTION("5 connections") {
    REQUIRE(httpMultiPartDownload(stream_url, filename, 5));
  }
  SECTION("1 connections") {
    REQUIRE(httpMultiPartDownload(stream_url, filename, 1));
  }
  std::string content = util::read_file(filename);
  REQUIRE(content.size() == 9112651);
  std::string checksum = sha_256(QString::fromStdString(content));
  REQUIRE(checksum == "e44edfbb545abdddfd17020ced2b18b6ec36506152267f32b6a8e3341f8126d6");
}

int random_int(int min, int max) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(min, max);
  return dist(rng);
}

TEST_CASE("Segment") {
  Route demo_route(DEMO_ROUTE);
  REQUIRE(demo_route.load());
  REQUIRE(demo_route.segments().size() == 11);

  QEventLoop loop;
  Segment segment(0, demo_route.at(0), false, false);
  QObject::connect(&segment, &Segment::loadFinished, [&]() {
    REQUIRE(segment.isLoaded() == true);
    REQUIRE(segment.log != nullptr);
    REQUIRE(segment.frames[RoadCam] != nullptr);
    REQUIRE(segment.frames[DriverCam] == nullptr);
    REQUIRE(segment.frames[WideRoadCam] == nullptr);

    // LogReader & FrameReader
    REQUIRE(segment.log->events.size() > 0);
    REQUIRE(std::is_sorted(segment.log->events.begin(), segment.log->events.end(), Event::lessThan()));

    auto &fr = segment.frames[RoadCam];
    REQUIRE(fr->getFrameCount() == 1200);
    std::unique_ptr<uint8_t[]> rgb_buf = std::make_unique<uint8_t[]>(fr->getRGBSize());
    std::unique_ptr<uint8_t[]> yuv_buf = std::make_unique<uint8_t[]>(fr->getYUVSize());
    // sequence get 50 frames
    for (int i = 0; i < 50; ++i) {
      REQUIRE(fr->get(i, rgb_buf.get(), yuv_buf.get()));
    }
    loop.quit();
  });
  loop.exec();
}

// helper class for unit tests
class TestReplay : public Replay {
public:
  TestReplay(const QString &route) : Replay(route, {}, {}) {}
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
    for (int i = 0; i < 50; ++i) {
      testSeekTo(random_int(0, 3 * 60));
    }
    // remove 3 segments
    for (int n : {5, 6, 8}) {
      segments_.erase(n);
    }
    for (int i =0; i < 50; ++i) {
      testSeekTo(random_int(4 * 60, 9 * 60));
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
