#include "selfdrive/ui/replay/framereader.h"
#include "selfdrive/ui/replay/replay.h"
#include "selfdrive/ui/replay/route.h"

#define CATCH_CONFIG_RUNNER
#include <QCoreApplication>

#include "catch2/catch.hpp"

static QString cache_path(const QString &url) {
  QByteArray url_no_query = QUrl(url).toString(QUrl::RemoveQuery).toUtf8();
  return CACHE_DIR + QString(QCryptographicHash::hash(url_no_query, QCryptographicHash::Sha256).toHex());
}

TEST_CASE("Route") {
  Route route(DEMO_ROUTE);
  REQUIRE(route.load());
  REQUIRE(route.size() == 121);
  QEventLoop loop;
  SECTION("load segment") {
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
  
}

TEST_CASE("Readers") {
  Route route(DEMO_ROUTE);
  REQUIRE(route.load());
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
    
    // check if events is ordered
    bool sorted = true;
    uint64_t prev_mono_time = 0;
    cereal::Event::Which prev_which = cereal::Event::INIT_DATA;
    for (auto event : lr.events) {
      if (event->mono_time < prev_mono_time || (event->mono_time == prev_mono_time && event->which < prev_which)) {
        sorted = false;
        break;
      }
      prev_mono_time = event->mono_time;
      prev_which = event->which;
    }
    REQUIRE(sorted == true);
  }
}

int main(int argc, char **argv) {
  // unit tests for Qt
  QCoreApplication app(argc, argv);
  const int res = Catch::Session().run(argc, argv);
  return (res < 0xff ? res : 0xff);
}
