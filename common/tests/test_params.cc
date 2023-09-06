#include "catch2/catch.hpp"
#define private public
#include "common/params.h"
#include "common/util.h"

TEST_CASE("Params/asyncWriter") {
  char tmp_path[] = "/tmp/asyncWriter_XXXXXX";
  const std::string param_path = mkdtemp(tmp_path);
  Params params(param_path);
  auto param_names = {"CarParams", "IsMetric"};
  {
    AsyncWriter async_writer;
    for (const auto &name : param_names) {
      async_writer.queue({param_path, name, "1"});
      // param is empty
      REQUIRE(params.get(name).empty());
    }

    // check if thread is running
    REQUIRE(async_writer.future.valid());
    REQUIRE(async_writer.future.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout);
  }
  // check results
  for (const auto &name : param_names) {
    REQUIRE(params.get(name) == "1");
  }
}
