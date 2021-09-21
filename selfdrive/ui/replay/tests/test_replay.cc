#include "selfdrive/ui/replay/framereader.h"
#include "selfdrive/ui/replay/replay.h"
#include "selfdrive/ui/replay/route.h"

#define CATCH_CONFIG_RUNNER
#include <QCoreApplication>
#include <QDebug>
#include <QTimer>

#include "catch2/catch.hpp"

QString cache_path(const QString &url) {
  QByteArray url_no_query = QUrl(url).toString(QUrl::RemoveQuery).toUtf8();
  return CACHE_DIR + QString(QCryptographicHash::hash(url_no_query, QCryptographicHash::Sha256).toHex());
}

// check if events is ordered
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

Route route(DEMO_ROUTE);

TEST_CASE("Route") {
  REQUIRE(route.load());
  REQUIRE(route.size() == 121);

  QEventLoop loop;
  Segment segment(0, route.at(1));
  REQUIRE(segment.isValid() == true);
  REQUIRE(segment.isLoaded() == false);
  QObject::connect(&segment, &Segment::loadFinished, [&]() {
    REQUIRE(segment.isLoaded() == true);
    REQUIRE(segment.log != nullptr);
    REQUIRE(segment.frames[RoadCam] != nullptr);
    REQUIRE(segment.frames[DriverCam] != nullptr);
    REQUIRE(segment.frames[WideRoadCam] == nullptr);
    loop.quit();
  });
  loop.exec();
}

TEST_CASE("Readers") {
  SECTION("FrameReader") {
    FrameReader fr;
    REQUIRE(fr.load(cache_path(route.at(1).road_cam).toStdString()) == true);
    REQUIRE(fr.valid() == true);
    REQUIRE(fr.getFrameCount() == 1200);
    // random get 50 frames
    // srand(time(NULL));
    // for (int i = 0; i < 50; ++i) {
    //   int idx = rand() % (fr.getFrameCount() - 1);
    //   REQUIRE(fr.get(idx) != nullptr);
    // }
    // sequence get 50 frames {
    for (int i = 0; i < 50; ++i) {
      REQUIRE(fr.get(i) != nullptr);
    }
  }
  SECTION("LogReader") {
    LogReader lr;
    REQUIRE(lr.load(cache_path(route.at(1).rlog).toStdString()) == true);
    REQUIRE(lr.events.size() > 0);
    REQUIRE(is_events_ordered(lr.events));
  }
}

// helper class for unit tests
class TestReplay : public Replay {
 public:
  TestReplay(const QString &route) : Replay(route, {}, {}) {}
  void startTest();
  void testStream();
};

void TestReplay::testStream() {
  // test 10 seconds
  uint64_t evt_start_ts = route_start_ts;
  uint64_t loop_start_ts = nanos_since_boot();
  for (auto e : *events) {
    std::string type;
    KJ_IF_MAYBE(e_, static_cast<capnp::DynamicStruct::Reader>(e->event).which()) {
      type = e_->getProto().getName();
    }

    if (socks.find(type) != socks.end()) {
      long etime = e->mono_time - evt_start_ts;
      if ((etime * 1e-9) > 10) break;

      // keep time
      long rtime = nanos_since_boot() - loop_start_ts;
      long us_behind = ((etime - rtime) * 1e-3);
      if (us_behind > 0) {
        QThread::usleep(us_behind);
      }

      long delay = std::abs(long(nanos_since_boot() - loop_start_ts) - etime);
      REQUIRE(delay <= 1e6);
      auto bytes = e->bytes();
      pm->send(type.c_str(), (capnp::byte *)bytes.begin(), bytes.size());
    }
  }
}

void TestReplay::startTest() {
  QEventLoop loop;
  REQUIRE(load());

  setCurrentSegment(0);

  QTimer *timer = new QTimer(this);
  timer->callOnTimeout([&]() {
    // wait for segmens merged
    int loaded = 0;
    for (int i = 0; i <= FORWARD_SEGS; ++i) {
      REQUIRE(segments[i] != nullptr);
      REQUIRE(segments[i]->isValid());
      loaded += segments[i]->isLoaded();
    }
    if (loaded == FORWARD_SEGS + 1) {
      uint64_t total_events_cnt = 0;
      for (auto &segment : segments) {
        if (segment && segment->isLoaded()) {
          total_events_cnt += segment->log->events.size();
        }
      }

      REQUIRE(route_start_ts > 0);

      // test if all events merged with correct order.
      REQUIRE(events->size() == total_events_cnt);
      REQUIRE(is_events_ordered(*events));

      // test seeking/updating events
      for (int i = 0; i < 100; ++i) {
        srand(time(nullptr));
        int idx = rand() % (events->size() - 2);
        auto current_event = events->at(idx);

        // ensure that no event will be lost, and the previous event will not be sent again
        auto next_event = nextEvent(current_event->mono_time, current_event->which);
        REQUIRE(next_event);
        auto prev_next = --(*next_event);
        REQUIRE((*prev_next)->mono_time == current_event->mono_time);
        REQUIRE((*prev_next)->which == current_event->which);
      }
      auto last_event = events->back();
      auto next_event = nextEvent(last_event->mono_time, last_event->which);
      REQUIRE(!next_event);

      // rand seek in 100 seconds
      for (int i = 0; i < 100; ++i) {
        srand(time(nullptr));
        int seconds = 1 + rand() % 100;
        uint64_t seek_mono_time = route_start_ts + seconds * 1e9;
        auto next_event = nextEvent(seek_mono_time, cereal::Event::Which::INIT_DATA);
        REQUIRE(next_event);
        auto eit = *next_event;
        REQUIRE((*eit)->mono_time > seek_mono_time);
        auto prev_event = --eit;
        REQUIRE(((*prev_event)->mono_time < seek_mono_time || (*prev_event)->mono_time == seek_mono_time && (*prev_event)->which < cereal::Event::Which::INIT_DATA));
      }

      testStream();
      loop.quit();
    }
  });
  timer->start(10);
  loop.exec();
}

TEST_CASE("Replay") {
  TestReplay replay(DEMO_ROUTE);
  replay.startTest();
}

int main(int argc, char **argv) {
  // unit tests for Qt
  QCoreApplication app(argc, argv);
  const int res = Catch::Session().run(argc, argv);
  return (res < 0xff ? res : 0xff);
}
