#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "catch2/catch.hpp"
#include "cereal/messaging/messaging.h"
#include "common/util.h"
#include "selfdrive/pandad/panda.h"

struct PandaTest : public Panda {
  PandaTest(uint32_t bus_offset, int can_list_size, cereal::PandaState::PandaType hw_type);
  void test_can_send();
  void test_can_recv(uint32_t chunk_size = 0);
  void test_chunked_can_recv();
  void test_incomplete_can_recv();
  void test_bus_filtering();

  std::map<int, std::string> test_data;
  int can_list_size = 0;
  int total_pakets_size = 0;
  MessageBuilder msg;
  capnp::List<cereal::CanData>::Reader can_data_list;
};

PandaTest::PandaTest(uint32_t bus_offset_, int can_list_size, cereal::PandaState::PandaType hw_type) : can_list_size(can_list_size), Panda(bus_offset_) {
  this->hw_type = hw_type;
  int data_limit = ((hw_type == cereal::PandaState::PandaType::RED_PANDA) ? std::size(dlc_to_len) : 8);
  // prepare test data
  for (int i = 0; i < data_limit; ++i) {
    std::random_device rd;
    std::independent_bits_engine<std::default_random_engine, CHAR_BIT, unsigned char> rbe(rd());

    int data_len = dlc_to_len[i];
    std::string bytes(data_len, '\0');
    std::generate(bytes.begin(), bytes.end(), std::ref(rbe));
    test_data[data_len] = bytes;
  }

  // generate can messages for this panda
  auto can_list = msg.initEvent().initSendcan(can_list_size);
  for (uint8_t i = 0; i < can_list_size; ++i) {
    auto can = can_list[i];
    uint32_t id = util::random_int(0, std::size(dlc_to_len) - 1);
    const std::string &dat = test_data[dlc_to_len[id]];
    can.setAddress(i);
    can.setSrc(util::random_int(0, 2) + bus_offset);
    can.setDat(kj::ArrayPtr((uint8_t *)dat.data(), dat.size()));
    total_pakets_size += sizeof(can_header) + dat.size();
  }

  can_data_list = can_list.asReader();
  INFO("test " << can_list_size << " packets, total size " << total_pakets_size);
}

void PandaTest::test_can_send() {
  std::vector<uint8_t> unpacked_data;
  this->pack_can_buffer(can_data_list, [&](uint8_t *chunk, size_t size) {
    unpacked_data.insert(unpacked_data.end(), chunk, &chunk[size]);
  });
  REQUIRE(unpacked_data.size() == total_pakets_size);

  int cnt = 0;
  INFO("test can message integrity");
  for (int pos = 0, pckt_len = 0; pos < unpacked_data.size(); pos += pckt_len) {
    can_header header;
    memcpy(&header, &unpacked_data[pos], sizeof(can_header));
    const uint8_t data_len = dlc_to_len[header.data_len_code];
    pckt_len = sizeof(can_header) + data_len;

    REQUIRE(header.addr == cnt);
    REQUIRE(test_data.find(data_len) != test_data.end());
    const std::string &dat = test_data[data_len];
    REQUIRE(memcmp(dat.data(), &unpacked_data[pos + sizeof(can_header)], dat.size()) == 0);
    ++cnt;
  }
  REQUIRE(cnt == can_list_size);
}

void PandaTest::test_can_recv(uint32_t rx_chunk_size) {
  std::vector<can_frame> frames;
  this->pack_can_buffer(can_data_list, [&](uint8_t *data, uint32_t size) {
    if (rx_chunk_size == 0) {
      REQUIRE(this->unpack_can_buffer(data, size, frames));
    } else {
      this->receive_buffer_size = 0;
      uint32_t pos = 0;

      while (pos < size) {
        uint32_t chunk_size = std::min(rx_chunk_size, size - pos);
        memcpy(&this->receive_buffer[this->receive_buffer_size], &data[pos], chunk_size);
        this->receive_buffer_size += chunk_size;
        pos += chunk_size;

        REQUIRE(this->unpack_can_buffer(this->receive_buffer, this->receive_buffer_size, frames));
      }
    }
  });

  REQUIRE(frames.size() == can_list_size);
  for (int i = 0; i < frames.size(); ++i) {
    REQUIRE(frames[i].address == i);
    REQUIRE(test_data.find(frames[i].dat.size()) != test_data.end());
    const std::string &dat = test_data[frames[i].dat.size()];
    REQUIRE(memcmp(dat.data(), frames[i].dat.data(), dat.size()) == 0);
  }
}

void PandaTest::test_incomplete_can_recv() {
  MessageBuilder local_msg;
  auto can_list = local_msg.initEvent().initSendcan(1);
  can_list[0].setAddress(0xabc);
  can_list[0].setSrc(bus_offset);
  const std::string dat = "\x01\x02\x03\x04";
  can_list[0].setDat(kj::ArrayPtr((uint8_t *)dat.data(), dat.size()));

  std::vector<uint8_t> packed_data;
  this->pack_can_buffer(can_list.asReader(), [&](uint8_t *chunk, size_t size) {
    packed_data.insert(packed_data.end(), chunk, &chunk[size]);
  });
  REQUIRE(packed_data.size() > sizeof(can_header));

  // Feed all but the last byte. unpack_can_buffer should keep a partial frame
  // in the receive buffer instead of dropping it.
  std::vector<can_frame> frames;
  this->receive_buffer_size = packed_data.size() - 1;
  memcpy(this->receive_buffer, packed_data.data(), this->receive_buffer_size);
  uint32_t partial_size = this->receive_buffer_size;
  REQUIRE(this->unpack_can_buffer(this->receive_buffer, partial_size, frames));
  REQUIRE(frames.empty());
  REQUIRE(partial_size > 0);

  // Provide the missing trailing byte and validate that all frames decode.
  this->receive_buffer[partial_size] = packed_data.back();
  partial_size += 1;
  REQUIRE(this->unpack_can_buffer(this->receive_buffer, partial_size, frames));
  REQUIRE(frames.size() == 1);
  REQUIRE(frames[0].address == 0xabc);
  REQUIRE(frames[0].src == bus_offset);
  REQUIRE(frames[0].dat == dat);
  REQUIRE(partial_size == 0);
}

void PandaTest::test_bus_filtering() {
  MessageBuilder local_msg;
  auto can_list = local_msg.initEvent().initSendcan(3);

  // Bus in range for this panda (bus_offset..bus_offset+3)
  can_list[0].setAddress(0x111);
  can_list[0].setSrc(bus_offset + 1);
  const std::string valid_dat = "\x01\x02\x03\x04";
  can_list[0].setDat(kj::ArrayPtr((uint8_t *)valid_dat.data(), valid_dat.size()));

  // Buses outside this panda's ownership should be ignored by pack_can_buffer.
  can_list[1].setAddress(0x222);
  can_list[1].setSrc((bus_offset + PANDA_BUS_OFFSET) % 8);
  const std::string out_of_range_dat_1 = "\x09\x08\x07\x06";
  can_list[1].setDat(kj::ArrayPtr((uint8_t *)out_of_range_dat_1.data(), out_of_range_dat_1.size()));

  can_list[2].setAddress(0x333);
  can_list[2].setSrc((bus_offset == 0) ? 7 : (bus_offset - 1));
  const std::string out_of_range_dat_2 = "\xaa\xbb\xcc\xdd";
  can_list[2].setDat(kj::ArrayPtr((uint8_t *)out_of_range_dat_2.data(), out_of_range_dat_2.size()));

  std::vector<uint8_t> packed_data;
  this->pack_can_buffer(can_list.asReader(), [&](uint8_t *chunk, size_t size) {
    packed_data.insert(packed_data.end(), chunk, &chunk[size]);
  });

  std::vector<can_frame> frames;
  uint32_t size = packed_data.size();
  REQUIRE(this->unpack_can_buffer(packed_data.data(), size, frames));
  REQUIRE(frames.size() == 1);
  REQUIRE(frames[0].address == 0x111);
  REQUIRE(frames[0].src == bus_offset + 1);
  REQUIRE(frames[0].dat == valid_dat);
}

TEST_CASE("send/recv CAN 2.0 packets") {
  auto bus_offset = GENERATE(0, 4);
  auto can_list_size = GENERATE(1, 3, 5, 10, 30, 60, 100, 200);
  PandaTest test(bus_offset, can_list_size, cereal::PandaState::PandaType::DOS);

  SECTION("can_send") {
    test.test_can_send();
  }
  SECTION("can_receive") {
    test.test_can_recv();
  }
  SECTION("chunked_can_receive") {
    test.test_can_recv(0x40);
  }
  SECTION("incomplete_receive_buffering") {
    test.test_incomplete_can_recv();
  }
  SECTION("bus_filtering") {
    test.test_bus_filtering();
  }
}

TEST_CASE("send/recv CAN FD packets") {
  auto bus_offset = GENERATE(0, 4);
  auto can_list_size = GENERATE(1, 3, 5, 10, 30, 60, 100, 200);
  PandaTest test(bus_offset, can_list_size, cereal::PandaState::PandaType::RED_PANDA);

  SECTION("can_send") {
    test.test_can_send();
  }
  SECTION("can_receive") {
    test.test_can_recv();
  }
  SECTION("chunked_can_receive") {
    test.test_can_recv(0x40);
  }
  SECTION("incomplete_receive_buffering") {
    test.test_incomplete_can_recv();
  }
  SECTION("bus_filtering") {
    test.test_bus_filtering();
  }
}
