#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cereal/messaging/messaging.h"
#include "selfdrive/boardd/panda.h"

const int CHUNK_SIZE = 64;
std::string TEST_DATA = [](int size) {
  std::random_device rd;
  std::independent_bits_engine<std::default_random_engine, CHAR_BIT, unsigned char> rbe(rd());
  std::string bytes(size + 1, '\0');
  std::generate(bytes.begin(), bytes.end(), std::ref(rbe));
  return bytes;
}(4);

const unsigned char dlc_to_len[] = {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 12U, 16U, 20U, 24U, 32U, 48U, 64U};

struct PandaTest : public Panda {
  PandaTest(uint32_t bus_offset) : Panda(bus_offset) {}
  void test_can_send(capnp::List<cereal::CanData>::Reader can_data_list, int total_pakets_size);
  void test_can_recv(capnp::List<cereal::CanData>::Reader can_data_list);
};

void PandaTest::test_can_send(capnp::List<cereal::CanData>::Reader can_data_list, int total_pakets_size) {
  std::string unpacked_data;
  write_can_packets(can_data_list, [&](uint8_t *data, size_t size) {
    int size_left = size;
    for (int i = 0, counter = 0; i < size; i += CHUNK_SIZE, counter++) {
      REQUIRE(data[i] == counter);

      const int len = std::min(CHUNK_SIZE - 1, size_left - 1);
      unpacked_data.append((char *)&data[i + 1], len);
      size_left -= (len + 1);
    }
  });
  REQUIRE(unpacked_data.size() == total_pakets_size);

  int cnt = 0;
  INFO("test can message integrity");
  for (int pos = 0, pckt_len = 0; pos < unpacked_data.size(); pos += pckt_len) {
    const uint8_t data_len = dlc_to_len[(unpacked_data[pos] >> 4)];
    pckt_len = CANPACKET_HEAD_SIZE + data_len;

    REQUIRE(*(uint32_t *)&unpacked_data[pos + 1] == cnt << 3);
    REQUIRE(memcmp(TEST_DATA.data(), &unpacked_data[pos + 5], TEST_DATA.size()) == 0);
    ++cnt;
  }
  REQUIRE(cnt == can_data_list.size());
}

void PandaTest::test_can_recv(capnp::List<cereal::CanData>::Reader can_data_list) {
  std::vector<can_frame> frames;
  write_can_packets(can_data_list, [&](uint8_t *data, size_t size) {
    read_can_packets(data, size, frames);
  });

  REQUIRE(frames.size() == can_data_list.size());
  for (int i = 0; i < frames.size(); ++i) {
    REQUIRE(frames[i].address == i);
    REQUIRE(memcmp(TEST_DATA.data(), frames[i].dat.data(), TEST_DATA.size()) == 0);
  }
}

TEST_CASE("test can") {
  auto can_list_size = GENERATE(1, 2, 3, 10, 20, 40, 60, 80, 100, 200);

  MessageBuilder msg;
  size_t total_pakets_size = 0;
  auto can_list = msg.initEvent().initSendcan(can_list_size);
  for (uint8_t i = 0; i < can_list.size(); ++i) {
    auto can = can_list[i];
    can.setAddress(i);
    can.setBusTime(i);
    can.setSrc(1);
    can.setDat(kj::ArrayPtr((uint8_t*)TEST_DATA.data(), TEST_DATA.size()));
    total_pakets_size += CANPACKET_HEAD_SIZE + TEST_DATA.size();
  }

  PandaTest test(0);
  SECTION("can_send") {
    INFO("write " << can_list_size << " packets, total size " << total_pakets_size);
    test.test_can_send(can_list.asReader(), total_pakets_size);
  }

  SECTION("can_receive") {
    INFO("read " << can_list_size << " packets");
    test.test_can_recv(can_list.asReader());
  }
}
