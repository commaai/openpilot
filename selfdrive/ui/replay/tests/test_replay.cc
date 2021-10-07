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
  const char *stream_url = "https://commadataci.blob.core.windows.net/openpilotci/0c94aa1e1296d7c6/2021-05-05--19-48-37/0/rlog.bz2";
  char filename[] = "/tmp/XXXXXX";
  int fd = mkstemp(filename);
  close(fd);

  SECTION("http 200") {
    REQUIRE(httpMultiPartDownload(stream_url, filename, 5));
    std::string content = util::read_file(filename);
    REQUIRE(content.size() == 9112651);
    std::string checksum = sha_256(QString::fromStdString(content));
    REQUIRE(checksum == "e44edfbb545abdddfd17020ced2b18b6ec36506152267f32b6a8e3341f8126d6");
  }
  SECTION("http 404") {
    REQUIRE(httpMultiPartDownload(util::string_format("%s_abc", stream_url), filename, 5) == false);
  }
}

int random_int(int min, int max) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(min, max);
  return dist(rng);
}

bool is_events_ordered(const std::vector<Event *> &events) {
  REQUIRE(events.size() > 0);
  uint64_t prev_mono_time = 0;
  cereal::Event::Which prev_which = cereal::Event::INIT_DATA;
  for (auto event : events) {
    if (event->mono_time < prev_mono_time || (event->mono_time == prev_mono_time && event->which < prev_which)) {
      return false;
    }
    prev_mono_time = event->mono_time;
    prev_which = event->which;
  }
  return true;
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
    REQUIRE(is_events_ordered(segment.log->events));
    // sequence get 50 frames {
    REQUIRE(segment.frames[RoadCam]->getFrameCount() == 1200);
    for (int i = 0; i < 50; ++i) {
      REQUIRE(segment.frames[RoadCam]->get(i));
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

  // wait for the seek to finish
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

    REQUIRE(is_events_ordered(*events_));
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
  QEventLoop loop;
  std::thread thread = std::thread([&]() {
    // random seek 50 times in 3 segments
    for (int i = 0; i < 50; ++i) {
      testSeekTo(random_int(0, 3 * 60));
    }
    // random seek 50 times in routes with invalid segments
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
