#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cereal/messaging/messaging.h"
#include "selfdrive/boardd/panda.h"

struct PandaTest : public Panda {
  PandaTest(uint32_t bus_offset) : Panda(bus_offset) {}
  void test_can_packets();
};

void PandaTest::test_can_packets() {
  auto can_list_size = GENERATE(1, 10, 20, 50, 100);
  INFO("write " << can_list_size << " packets");

  MessageBuilder msg;
  uint8_t bus = (1 >> 1) & 0x7 + bus_offset;

  auto can_list = msg.initEvent().initSendcan(can_list_size);
  for (uint8_t i = 0; i < can_list.size(); ++i) {
    auto can = can_list[i];
    can.setAddress(i);
    can.setBusTime(i);
    can.setSrc(bus);
    can.setDat(kj::ArrayPtr(&i, sizeof(uint8_t)));
  }

  write_can_packets(can_list.asReader(), [&](uint8_t *data, size_t size) {
    for (int i = 0, counter = 0; i < size; i += 64, counter++) {
      REQUIRE(data[i] == counter);
      uint8_t *can = &data[i + 1];

      uint8_t data_len_code = len_to_dlc(sizeof(uint8_t));
      REQUIRE(can[0] == (data_len_code << 4 | ((bus - bus_offset) << 1)));
    }
  });
}

TEST_CASE("send/recv can packets") {
  PandaTest test(0);
  test.test_can_packets();
}
