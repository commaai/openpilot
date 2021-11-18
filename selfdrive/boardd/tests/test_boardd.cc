#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <random>

#include "catch2/catch.hpp"
#include "cereal/messaging/messaging.h"
#include "selfdrive/boardd/panda.h"

const int CHUNK_SIZE = 64;
const unsigned char dlc_to_len[] = {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 12U, 16U, 20U, 24U, 32U, 48U, 64U};

int random_int(int min, int max) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(min, max);
  return dist(rng);
}

class PandaTest : public Panda {
 public:
  PandaTest(uint32_t bus_offset);
  void test_can_send(const capnp::List<cereal::CanData>::Reader &can_data_list, int total_pakets_size);
  void test_can_recv(const capnp::List<cereal::CanData>::Reader &can_data_list);
  void benchmark(const capnp::List<cereal::CanData>::Reader &can_data_list);
  std::map<int, std::string> test_data;
};

PandaTest::PandaTest(uint32_t buf_offset_) : Panda(buf_offset_) {
  for (int i = 0; i < std::size(dlc_to_len); ++i) {
    std::random_device rd;
    std::independent_bits_engine<std::default_random_engine, CHAR_BIT, unsigned char> rbe(rd());

    int data_len = dlc_to_len[i];
    std::string bytes(data_len, '\0');
    std::generate(bytes.begin(), bytes.end(), std::ref(rbe));
    test_data[data_len] = bytes;
  }
}

void PandaTest::test_can_send(const capnp::List<cereal::CanData>::Reader &can_data_list, int total_pakets_size) {
  std::vector<uint8_t> unpacked_data;
  this->pack_can_buffer(can_data_list, [&](uint8_t *chunk, size_t size) {
    int size_left = size;
    for (int i = 0, counter = 0; i < size; i += CHUNK_SIZE, counter++) {
      REQUIRE(chunk[i] == counter);

      const int len = std::min(CHUNK_SIZE, size_left);
      unpacked_data.insert(unpacked_data.end(), &chunk[i + 1], &chunk[i + len]);
      size_left -= len;
    }
  });
  REQUIRE(unpacked_data.size() == total_pakets_size);

  int cnt = 0;
  INFO("test can message integrity");
  for (int pos = 0, pckt_len = 0; pos < unpacked_data.size(); pos += pckt_len) {
    can_header header;
    memcpy(&header, &unpacked_data[pos], CANPACKET_HEAD_SIZE);
    const uint8_t data_len = dlc_to_len[header.data_len_code];
    pckt_len = CANPACKET_HEAD_SIZE + data_len;

    REQUIRE(header.addr == cnt);

    REQUIRE(test_data.find(data_len) != test_data.end());
    const std::string &dat = test_data[data_len];
    REQUIRE(memcmp(dat.data(), &unpacked_data[pos + 5], dat.size()) == 0);
    ++cnt;
  }
  REQUIRE(cnt == can_data_list.size());
}

void PandaTest::test_can_recv(const capnp::List<cereal::CanData>::Reader &can_data_list) {
  std::vector<can_frame> frames;
  this->pack_can_buffer(can_data_list, [&](uint8_t *data, size_t size) {
    this->unpack_can_buffer(data, size, frames);
  });

  REQUIRE(frames.size() == can_data_list.size());
  for (int i = 0; i < frames.size(); ++i) {
    REQUIRE(frames[i].address == i);
    REQUIRE(test_data.find(frames[i].dat.size()) != test_data.end());
    const std::string &dat = test_data[frames[i].dat.size()];
    REQUIRE(memcmp(dat.data(), frames[i].dat.data(), dat.size()) == 0);
  }
}

TEST_CASE("send/recv can packets") {
  auto can_list_size = GENERATE(1, 3, 5, 10, 30, 60, 100, 200);
  PandaTest test(0);

  size_t total_pakets_size = 0;
  MessageBuilder msg;
  auto can_list = msg.initEvent().initSendcan(can_list_size);
  for (uint8_t i = 0; i < can_list.size(); ++i) {
    auto can = can_list[i];
    uint32_t id = random_int(0, std::size(dlc_to_len) - 1);
    const std::string &dat = test.test_data[dlc_to_len[id]];
    can.setAddress(i);
    can.setDat(kj::ArrayPtr((uint8_t *)dat.data(), dat.size()));
    total_pakets_size += CANPACKET_HEAD_SIZE + dat.size();
  }

  INFO("test " << can_list_size << " packets, total size " << total_pakets_size);

  SECTION("can_send") {
    test.test_can_send(can_list.asReader(), total_pakets_size);
  }

  SECTION("can_receive") {
    test.test_can_recv(can_list.asReader());
  }
}
