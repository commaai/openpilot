#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cereal/messaging/messaging.h"
#include "selfdrive/boardd/panda.h"

struct PandaTest : public Panda {
  PandaTest(uint32_t bus_offset) : Panda(bus_offset) {}
  void test_can_packets();
};

void PandaTest::test_can_packets() {
}

TEST_CASE("send/recv can packets") {
  PandaTest test(0);
  test.test_can_packets();
}
