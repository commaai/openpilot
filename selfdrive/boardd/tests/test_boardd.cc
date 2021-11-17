#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cereal/messaging/messaging.h"
#include "selfdrive/boardd/panda.h"

const int CHUNK_SIZE = 64;
const unsigned char dlc_to_len[] = {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 12U, 16U, 20U, 24U, 32U, 48U, 64U};

struct PandaTest : public Panda {
  PandaTest(uint32_t bus_offset) : Panda(bus_offset) {}
  void test_can_send();
};

std::string read_chunks(uint8_t *data, int size) {
  std::string result;
  result.reserve(size);

  int size_left = size;
  for (int i = 0, counter = 0; i < size; i += CHUNK_SIZE, counter++) {
    const int len = std::min(CHUNK_SIZE - 1, size_left - 1);
    result.append((char *)&data[i + 1], len);
    size_left -= (len + 1);
  }
  return result;
}

void PandaTest::test_can_send() {
  auto can_list_size = GENERATE(1, 2, 3, 10, 20, 40, 60, 80, 100);

  MessageBuilder msg;
  uint8_t bus = 1;
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
  std::string unpacked_data;
  write_can_packets(can_list.asReader(), [&](uint8_t *data, size_t size) {
    for (int i = 0, counter = 0; i < size; i += 64, counter++) {
      REQUIRE(data[i] == counter);

      uint8_t *can = &data[i + 1];
      int pos_0 = (len_to_dlc(sizeof(dat)) << 4 | ((bus - bus_offset) << 1));
      REQUIRE(can[0] == pos_0);
      REQUIRE(*(uint32_t *)&can[1] == address << 3);
      ++chunks;
    }
    unpacked_data += read_chunks(data, size);
  });
  REQUIRE(unpacked_data.size() == total_pakets_size);
  REQUIRE(chunks == (int)std::ceil(total_pakets_size / 63.0));

  INFO("test can message integrity");
  for (int pos = 0, pckt_len = 0; pos < unpacked_data.size(); pos += pckt_len) {
    const uint8_t data_len = dlc_to_len[(unpacked_data[pos] >> 4)];
    pckt_len = CANPACKET_HEAD_SIZE + data_len;

    REQUIRE(*(uint32_t *)&unpacked_data[pos + 1] == address << 3);
    REQUIRE(memcmp(dat, &unpacked_data[pos + 5], sizeof(dat)) == 0);
  }
}

TEST_CASE("can_send") {
  PandaTest test(0);
  test.test_can_send();
}
