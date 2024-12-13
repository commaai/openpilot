#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "tools/replay/replay.h"

const std::string TEST_RLOG_URL = "https://commadataci.blob.core.windows.net/openpilotci/0c94aa1e1296d7c6/2021-05-05--19-48-37/0/rlog.bz2";

TEST_CASE("LogReader") {
  SECTION("corrupt log") {
    FileReader reader(true);
    std::string corrupt_content = reader.read(TEST_RLOG_URL);
    corrupt_content.resize(corrupt_content.length() / 2);
    corrupt_content = decompressBZ2(corrupt_content);
    LogReader log;
    REQUIRE(log.load(corrupt_content.data(), corrupt_content.size()));
    REQUIRE(log.events.size() > 0);
  }
}
