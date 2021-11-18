#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cereal/messaging/messaging.h"
#include "selfdrive/boardd/panda.h"

const int CHUNK_SIZE = 64;
std::string TEST_DATA = [](int size) {
  std::random_device rd;
  std::independent_bits_engine<std::default_random_engine, CHAR_BIT, unsigned char> rbe(rd());
  std::string bytes(size, '\0');
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
  this->pack_can_buffer(can_data_list, [&](uint8_t *chunk, size_t size) {
    int size_left = size;
    for (int i = 0, counter = 0; i < size; i += CHUNK_SIZE, counter++) {
      REQUIRE(chunk[i] == counter);
      const int len = std::min(CHUNK_SIZE - 1, size_left - 1);
      unpacked_data.append((char *)&chunk[i + 1], len);
      size_left -= (len + 1);
    }
  });
  REQUIRE(unpacked_data.size() == total_pakets_size);

  int cnt = 0;
  INFO("test can message integrity");
  for (int pos = 0, pckt_len = 0; pos < unpacked_data.size(); pos += pckt_len) {
    const uint8_t data_len = dlc_to_len[(unpacked_data[pos] >> 4)];
    pckt_len = CANPACKET_HEAD_SIZE + data_len;

    can_header header;
    memcpy(&header, &unpacked_data[pos], CANPACKET_HEAD_SIZE);
    REQUIRE(header.addr == cnt);
    REQUIRE(memcmp(TEST_DATA.data(), &unpacked_data[pos + 5], TEST_DATA.size()) == 0);
    ++cnt;
  }
  REQUIRE(cnt == can_data_list.size());
}

void PandaTest::test_can_recv(capnp::List<cereal::CanData>::Reader can_data_list) {
  std::vector<can_frame> frames;
  this->pack_can_buffer(can_data_list, [&](uint8_t *data, size_t size) {
    this->unpack_can_buffer(data, size, frames);
  });

  REQUIRE(frames.size() == can_data_list.size());
  for (int i = 0; i < frames.size(); ++i) {
    REQUIRE(frames[i].address == i);
    REQUIRE(memcmp(TEST_DATA.data(), frames[i].dat.data(), TEST_DATA.size()) == 0);
  }
}

TEST_CASE("send/recv can packets") {
  auto can_list_size = GENERATE(1, 3, 5, 20, 40, 60, 80, 100, 200);

  MessageBuilder msg;
  size_t total_pakets_size = 0;
  auto can_list = msg.initEvent().initSendcan(can_list_size);
  for (uint8_t i = 0; i < can_list.size(); ++i) {
    auto can = can_list[i];
    can.setAddress(i);
    can.setDat(kj::ArrayPtr((uint8_t *)TEST_DATA.data(), TEST_DATA.size()));
    total_pakets_size += CANPACKET_HEAD_SIZE + TEST_DATA.size();
  }

  INFO("test " << can_list_size << " packets, total size " << total_pakets_size);

  PandaTest test(0);
  SECTION("can_send") {
    test.test_can_send(can_list.asReader(), total_pakets_size);
  }

  SECTION("can_receive") {
    test.test_can_recv(can_list.asReader());
  }
}
