#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include "selfdrive/ui/replay/framereader.h"

const char *stream_url = "https://commadataci.blob.core.windows.net/openpilotci/0c94aa1e1296d7c6/2021-05-05--19-48-37/0/fcamera.hevc";

TEST_CASE("FrameReader") {
  SECTION("process&get") {
    FrameReader fr(stream_url);
    bool ret = fr.process();
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
  SECTION("process with timeout") {
    FrameReader fr(stream_url, 1);
    bool ret = fr.process();
    REQUIRE(ret == false);
    REQUIRE(fr.valid() == false);
    REQUIRE(fr.getFrameCount() < 1200);
  }
}
