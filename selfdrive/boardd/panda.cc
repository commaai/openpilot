#include "selfdrive/boardd/panda.h"

#include <cassert>
#include <stdexcept>

#include "cereal/messaging/messaging.h"
#include "panda/board/dlc_to_len.h"
#include "selfdrive/common/gpio.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"

Panda::Panda(std::string serial, uint32_t bus_offset) : bus_offset(bus_offset) {
  if (!open(serial)) {
    throw std::runtime_error("Error connecting to panda");
  }
  hw_type = get_hw_type();
  assert((hw_type != cereal::PandaState::PandaType::WHITE_PANDA) &&
         (hw_type != cereal::PandaState::PandaType::GREY_PANDA));

  has_rtc = (hw_type == cereal::PandaState::PandaType::UNO) ||
            (hw_type == cereal::PandaState::PandaType::DOS);
}

Panda::~Panda() {}

void Panda::set_safety_model(cereal::CarParams::SafetyModel safety_model, int safety_param) {
  write(0xdc, (uint16_t)safety_model, safety_param);
}

void Panda::set_unsafe_mode(uint16_t unsafe_mode) {
  write(0xdf, unsafe_mode, 0);
}

cereal::PandaState::PandaType Panda::get_hw_type() {
  unsigned char hw_query[1] = {0};

  read(0xc1, 0, 0, hw_query, 1);
  return (cereal::PandaState::PandaType)(hw_query[0]);
}

void Panda::set_rtc(struct tm sys_time) {
  // tm struct has year defined as years since 1900
  write(0xa1, (uint16_t)(1900 + sys_time.tm_year), 0);
  write(0xa2, (uint16_t)(1 + sys_time.tm_mon), 0);
  write(0xa3, (uint16_t)sys_time.tm_mday, 0);
  // write(0xa4, (uint16_t)(1 + sys_time.tm_wday), 0);
  write(0xa5, (uint16_t)sys_time.tm_hour, 0);
  write(0xa6, (uint16_t)sys_time.tm_min, 0);
  write(0xa7, (uint16_t)sys_time.tm_sec, 0);
}

struct tm Panda::get_rtc() {
  struct __attribute__((packed)) timestamp_t {
    uint16_t year; // Starts at 0
    uint8_t month;
    uint8_t day;
    uint8_t weekday;
    uint8_t hour;
    uint8_t minute;
    uint8_t second;
  } rtc_time = {0};

  read(0xa0, 0, 0, (unsigned char*)&rtc_time, sizeof(rtc_time));

  struct tm new_time = { 0 };
  new_time.tm_year = rtc_time.year - 1900; // tm struct has year defined as years since 1900
  new_time.tm_mon  = rtc_time.month - 1;
  new_time.tm_mday = rtc_time.day;
  new_time.tm_hour = rtc_time.hour;
  new_time.tm_min  = rtc_time.minute;
  new_time.tm_sec  = rtc_time.second;

  return new_time;
}

void Panda::set_fan_speed(uint16_t fan_speed) {
  write(0xb1, fan_speed, 0);
}

uint16_t Panda::get_fan_speed() {
  uint16_t fan_speed_rpm = 0;
  read(0xb2, 0, 0, (unsigned char*)&fan_speed_rpm, sizeof(fan_speed_rpm));
  return fan_speed_rpm;
}

void Panda::set_ir_pwr(uint16_t ir_pwr) {
  write(0xb0, ir_pwr, 0);
}

health_t Panda::get_state() {
  health_t health {0};
  read(0xd2, 0, 0, (unsigned char*)&health, sizeof(health));
  return health;
}

void Panda::set_loopback(bool loopback) {
  write(0xe5, loopback, 0);
}

std::optional<std::vector<uint8_t>> Panda::get_firmware_version() {
  std::vector<uint8_t> fw_sig_buf(128);
  int read_1 = read(0xd3, 0, 0, &fw_sig_buf[0], 64);
  int read_2 = read(0xd4, 0, 0, &fw_sig_buf[64], 64);
  return ((read_1 == 64) && (read_2 == 64)) ? std::make_optional(fw_sig_buf) : std::nullopt;
}

std::optional<std::string> Panda::get_serial() {
  char serial_buf[17] = {'\0'};
  int err = read(0xd0, 0, 0, (uint8_t*)serial_buf, 16);
  return err >= 0 ? std::make_optional(serial_buf) : std::nullopt;
}

void Panda::set_power_saving(bool power_saving) {
  write(0xe7, power_saving, 0);
}

void Panda::set_usb_power_mode(cereal::PeripheralState::UsbPowerMode power_mode) {
  write(0xe6, (uint16_t)power_mode, 0);
}

void Panda::send_heartbeat(bool engaged) {
  write(0xf3, engaged, 0);
}

void Panda::set_can_speed_kbps(uint16_t bus, uint16_t speed) {
  write(0xde, bus, (speed * 10));
}

void Panda::set_data_speed_kbps(uint16_t bus, uint16_t speed) {
  write(0xf9, bus, (speed * 10));
}

static uint8_t len_to_dlc(uint8_t len) {
  if (len <= 8) {
    return len;
  }
  if (len <= 24) {
    return 8 + ((len - 8) / 4) + ((len % 4) ? 1 : 0);
  } else {
    return 11 + (len / 16) + ((len % 16) ? 1 : 0);
  }
}

static void write_packet(uint8_t *dest, int *write_pos, const uint8_t *src, size_t size) {
  for (int i = 0, &pos = *write_pos; i < size; ++i, ++pos) {
    // Insert counter every 64 bytes (first byte of 64 bytes USB packet)
    if (pos % USBPACKET_MAX_SIZE == 0) {
      dest[pos] = pos / USBPACKET_MAX_SIZE;
      pos++;
    }
    dest[pos] = src[i];
  }
}

void Panda::pack_can_buffer(const capnp::List<cereal::CanData>::Reader &can_data_list,
                            std::function<void(uint8_t *, size_t)> write_func) {
  int32_t pos = 0;
  uint8_t send_buf[2 * USB_TX_SOFT_LIMIT];

  for (auto cmsg : can_data_list) {
    // check if the message is intended for this panda
    uint8_t bus = cmsg.getSrc();
    if (bus < bus_offset || bus >= (bus_offset + PANDA_BUS_CNT)) {
      continue;
    }
    auto can_data = cmsg.getDat();
    uint8_t data_len_code = len_to_dlc(can_data.size());
    assert(can_data.size() <= ((hw_type == cereal::PandaState::PandaType::RED_PANDA) ? 64 : 8));
    assert(can_data.size() == dlc_to_len[data_len_code]);

    can_header header;
    header.addr = cmsg.getAddress();
    header.extended = (cmsg.getAddress() >= 0x800) ? 1 : 0;
    header.data_len_code = data_len_code;
    header.bus = bus - bus_offset;

    write_packet(send_buf, &pos, (uint8_t *)&header, sizeof(can_header));
    write_packet(send_buf, &pos, (uint8_t *)can_data.begin(), can_data.size());
    if (pos >= USB_TX_SOFT_LIMIT) {
      write_func(send_buf, pos);
      pos = 0;
    }
  }

  // send remaining packets
  if (pos > 0) write_func(send_buf, pos);
}

void Panda::can_send(capnp::List<cereal::CanData>::Reader can_data_list) {
  pack_can_buffer(can_data_list, [=](uint8_t* data, size_t size) {
    bulk_write(3, data, size, 5);
  });
}

bool Panda::can_receive(std::vector<can_frame>& out_vec) {
  uint8_t data[RECV_SIZE];
  int recv = bulk_read(0x81, (uint8_t*)data, RECV_SIZE);
  if (!comms_healthy) {
    return false;
  }
  if (recv == RECV_SIZE) {
    LOGW("Panda receive buffer full");
  }

  return (recv <= 0) ? true : unpack_can_buffer(data, recv, out_vec);
}

bool Panda::unpack_can_buffer(uint8_t *data, int size, std::vector<can_frame> &out_vec) {
  recv_buf.clear();
  for (int i = 0; i < size; i += USBPACKET_MAX_SIZE) {
    if (data[i] != i / USBPACKET_MAX_SIZE) {
      LOGE("CAN: MALFORMED USB RECV PACKET");
      comms_healthy = false;
      return false;
    }
    int chunk_len = std::min(USBPACKET_MAX_SIZE, (size - i));
    recv_buf.insert(recv_buf.end(), &data[i + 1], &data[i + chunk_len]);
  }

  int pos = 0;
  while (pos < recv_buf.size()) {
    can_header header;
    memcpy(&header, &recv_buf[pos], CANPACKET_HEAD_SIZE);

    can_frame &canData = out_vec.emplace_back();
    canData.busTime = 0;
    canData.address = header.addr;
    canData.src = header.bus + bus_offset;
    if (header.rejected) { canData.src += CANPACKET_REJECTED; }
    if (header.returned) { canData.src += CANPACKET_RETURNED; }

    const uint8_t data_len = dlc_to_len[header.data_len_code];
    canData.dat.assign((char *)&recv_buf[pos + CANPACKET_HEAD_SIZE], data_len);

    pos += CANPACKET_HEAD_SIZE + data_len;
  }
  return true;
}
