#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cereal/messaging/messaging.h"
#include "selfdrive/boardd/panda.h"

struct PandaTest : public Panda {
  PandaTest(uint32_t bus_offset) : Panda(bus_offset) {}
  void test_can_send();
};

void PandaTest::test_can_send() {
  auto can_list_size = GENERATE(1, 2, 3, 10, 20, 40, 60, 80, 100);

  MessageBuilder msg;
  uint8_t bus = (1 >> 1) & 0x7 + bus_offset;
  uint32_t address = 5;
  uint8_t dat[4] = {1, 2, 3, 4};

  size_t total_pakets_size = 0;
  auto can_list = msg.initEvent().initSendcan(can_list_size);
  for (uint8_t i = 0; i < can_list.size(); ++i) {
    auto can = can_list[i];
    can.setAddress(address);
    can.setBusTime(i);
    can.setSrc(bus);
    can.setDat(kj::ArrayPtr(dat, std::size(dat)));
    total_pakets_size += CANPACKET_HEAD_SIZE + sizeof(dat);
  }

  INFO("write " << can_list_size << " packets, total size " << total_pakets_size);

  int chunks = 0;
  write_can_packets(can_list.asReader(), [&](uint8_t *data, size_t size) {
    for (int i = 0, counter = 0; i < size; i += 64, counter++) {
      REQUIRE(data[i] == counter);
      uint8_t *can = &data[i + 1];
      int pos_0 = (len_to_dlc(sizeof(dat)) << 4 | ((bus - bus_offset) << 1));

      REQUIRE(can[0] == pos_0);
      REQUIRE(*(uint32_t*)&can[1] == address << 3);
      ++chunks;
    }
  });
  REQUIRE(chunks == (int)std::ceil(total_pakets_size / 63.0));
}

TEST_CASE("can_send") {
  PandaTest test(0);
  test.test_can_send();
}
