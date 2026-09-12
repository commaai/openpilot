#include <climits>
#include <array>

#include "common/tests/native_test.h"
#include "openpilot/cereal/messaging/messaging.h"
#include "selfdrive/pandad/panda.h"
#include "panda/board/spi_protocol.h"

struct PandaTest : public Panda {
  PandaTest(int can_list_size, cereal::PandaState::PandaType hw_type);
  void test_can_send(bool spi_v3 = false);
  void test_large_can_send();
  void test_can_recv(uint32_t chunk_size = 0);
  void test_chunked_can_recv();

  std::map<int, std::string> test_data;
  int can_list_size = 0;
  int total_pakets_size = 0;
  MessageBuilder msg;
  capnp::List<cereal::CanData>::Reader can_data_list;
};

PandaTest::PandaTest(int can_list_size_, cereal::PandaState::PandaType hw_type_) : can_list_size(can_list_size_), Panda() {
  this->hw_type = hw_type_;
  int data_limit = ((hw_type == cereal::PandaState::PandaType::RED_PANDA) ? std::size(dlc_to_len) : 8);
  // prepare test data
  for (int i = 0; i < data_limit; ++i) {
    int data_len = dlc_to_len[i];
    std::string bytes(data_len, '\0');
    for (int j = 0; j < data_len; ++j) bytes[j] = static_cast<char>((i * 31 + j) & 0xff);
    test_data[data_len] = bytes;
  }

  // generate can messages for this panda
  auto can_list = msg.initEvent().initSendcan(can_list_size);
  for (uint8_t i = 0; i < can_list_size; ++i) {
    auto can = can_list[i];
    uint32_t id = i % data_limit;
    const std::string &dat = test_data[dlc_to_len[id]];
    can.setAddress(i);
    can.setSrc(i % 3);
    can.setDat(kj::ArrayPtr((uint8_t *)dat.data(), dat.size()));
    total_pakets_size += sizeof(can_header) + dat.size();
  }

  can_data_list = can_list.asReader();
}

void PandaTest::test_can_send(bool spi_v3) {
  std::vector<uint8_t> unpacked_data;
  this->pack_can_buffer(can_data_list, [&](uint8_t *chunk, size_t size) {
    CHECK(size <= (spi_v3 ? SPI_PROTO_MAX_PAYLOAD : USB_TX_SOFT_LIMIT + sizeof(can_header) + 63));
    size_t pos = 0;
    while (pos < size) {
      CHECK(size - pos >= sizeof(can_header));
      can_header header;
      memcpy(&header, chunk + pos, sizeof(header));
      size_t record_size = sizeof(header) + dlc_to_len[header.data_len_code];
      CHECK(record_size <= size - pos);
      CHECK(calculate_checksum(chunk + pos, record_size) == 0);
      pos += record_size;
    }
    CHECK(pos == size);
    unpacked_data.insert(unpacked_data.end(), chunk, &chunk[size]);
  }, spi_v3);
  CHECK(unpacked_data.size() == total_pakets_size);

  int cnt = 0;
  for (int pos = 0, pckt_len = 0; pos < unpacked_data.size(); pos += pckt_len) {
    can_header header;
    memcpy(&header, &unpacked_data[pos], sizeof(can_header));
    const uint8_t data_len = dlc_to_len[header.data_len_code];
    pckt_len = sizeof(can_header) + data_len;

    CHECK(header.addr == cnt);
    CHECK(test_data.find(data_len) != test_data.end());
    const std::string &dat = test_data[data_len];
    CHECK(memcmp(dat.data(), &unpacked_data[pos + sizeof(can_header)], dat.size()) == 0);
    ++cnt;
  }
  CHECK(cnt == can_list_size);
}

void PandaTest::test_large_can_send() {
  MessageBuilder large_msg;
  auto records = large_msg.initEvent().initSendcan(120);
  for (uint32_t i = 0; i < records.size(); ++i) {
    std::array<uint8_t, 64> payload;
    for (size_t j = 0; j < payload.size(); ++j) payload[j] = (i * 31 + j) & 0xff;
    records[i].setAddress(i);
    records[i].setSrc(i % 3);
    records[i].setDat(kj::ArrayPtr(payload.data(), payload.size()));
  }

  std::vector<uint8_t> legacy_bytes, v3_bytes;
  std::vector<size_t> legacy_sizes, v3_sizes;
  pack_can_buffer(records.asReader(), [&](uint8_t *data, size_t size) {
    CHECK(size == 4 * 70);  // Preserve the legacy 256-byte soft limit.
    legacy_sizes.push_back(size);
    legacy_bytes.insert(legacy_bytes.end(), data, data + size);
  });
  pack_can_buffer(records.asReader(), [&](uint8_t *data, size_t size) {
    CHECK(size <= SPI_PROTO_MAX_PAYLOAD);
    CHECK(size % 70 == 0);  // Every chunk contains complete CAN FD records.
    v3_sizes.push_back(size);
    v3_bytes.insert(v3_bytes.end(), data, data + size);
  }, true);
  CHECK(legacy_sizes.size() == 30);
  CHECK(v3_sizes == std::vector<size_t>({58 * 70, 58 * 70, 4 * 70}));
  CHECK(v3_bytes == legacy_bytes);
  CHECK(v3_bytes.size() == 120 * 70);
}

void PandaTest::test_can_recv(uint32_t rx_chunk_size) {
  std::vector<can_frame> frames;
  this->pack_can_buffer(can_data_list, [&](uint8_t *data, uint32_t size) {
    if (rx_chunk_size == 0) {
      CHECK(this->unpack_can_buffer(data, size, frames));
    } else {
      this->receive_buffer_size = 0;
      uint32_t pos = 0;

      while (pos < size) {
        uint32_t chunk_size = std::min(rx_chunk_size, size - pos);
        memcpy(&this->receive_buffer[this->receive_buffer_size], &data[pos], chunk_size);
        this->receive_buffer_size += chunk_size;
        pos += chunk_size;

        CHECK(this->unpack_can_buffer(this->receive_buffer, this->receive_buffer_size, frames));
      }
    }
  });

  CHECK(frames.size() == can_list_size);
  for (int i = 0; i < frames.size(); ++i) {
    CHECK(frames[i].address == i);
    CHECK(test_data.find(frames[i].dat.size()) != test_data.end());
    const std::string &dat = test_data[frames[i].dat.size()];
    CHECK(memcmp(dat.data(), frames[i].dat.data(), dat.size()) == 0);
  }
}

void test_can_protocol() {
  for (auto hw_type : {cereal::PandaState::PandaType::DOS, cereal::PandaState::PandaType::RED_PANDA}) {
    for (int can_list_size : {1, 3, 5, 10, 30, 60, 100, 200}) {
      PandaTest send_test(can_list_size, hw_type);
      send_test.test_can_send();
      send_test.test_can_send(true);

      PandaTest receive_test(can_list_size, hw_type);
      receive_test.test_can_recv();

      PandaTest chunked_receive_test(can_list_size, hw_type);
      chunked_receive_test.test_can_recv(0x40);
    }
  }
  PandaTest large_test(1, cereal::PandaState::PandaType::RED_PANDA);
  large_test.test_large_can_send();
}

int main() {
  return run_native_test(test_can_protocol);
}
