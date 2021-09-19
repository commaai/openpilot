#include "selfdrive/ui/replay/framereader.h"
#include "selfdrive/ui/replay/replay.h"
#include "selfdrive/ui/replay/route.h"

#define CATCH_CONFIG_RUNNER
#include <QCoreApplication>

#include "catch2/catch.hpp"

int main(int argc, char **argv) {
  // unit tests for Qt
  QCoreApplication app(argc, argv);
  const int res = Catch::Session().run(argc, argv);
  return (res < 0xff ? res : 0xff);
}

const char *stream_url = "https://commadataci.blob.core.windows.net/openpilotci/0c94aa1e1296d7c6/2021-05-05--19-48-37/0/fcamera.hevc";

TEST_CASE("FrameReader") {
  SECTION("process&get") {
    FrameReader fr;
    bool ret = fr.load(stream_url);
    REQUIRE(ret == true);
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
}

TEST_CASE("route") {
  QString route_name = "0982d79ebb0de295|2021-01-17--17-13-08";
  Route route(route_name);
  REQUIRE(route.load());
  REQUIRE(route.size() == 27);
  for (int i = 0; i < route.size(); ++i) {
    REQUIRE(!route.at(i).rlog.isEmpty());
    REQUIRE(!route.at(i).road_cam.isEmpty());
  }

  SECTION("load segment") {
    QEventLoop loop;
    Segment segment(0, route.at(0));
    REQUIRE(segment.isValid());
    REQUIRE(!segment.isLoaded());
    QObject::connect(&segment, &Segment::loadFinished, [&]() {
      REQUIRE(segment.isLoaded());
      REQUIRE(segment.log != nullptr);
      REQUIRE(segment.frames[RoadCam] != nullptr);
      REQUIRE(segment.frames[DriverCam] == nullptr);
      REQUIRE(segment.frames[WideRoadCam] == nullptr);
      loop.quit();
    });
    loop.exec();
  }

  SECTION("load segment, http error") {
    QEventLoop loop;
    SegmentFile files = route.at(0);
    files.rlog = "https://commadataci.blob.core.windows.net/openpilotci/0c94aa1e1296d7c6/2021-05-05--19-48-37/0/rlog_invalid.bz2";
    Segment segment(0, files);
    REQUIRE(segment.isValid());
    QObject::connect(&segment, &Segment::loadFinished, [&]() {
      REQUIRE(segment.isLoaded() == false);
      REQUIRE(segment.isValid() == false);
      loop.quit();
    });
    loop.exec();
  }
}
