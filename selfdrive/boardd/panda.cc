#include "selfdrive/boardd/panda.h"

#include <unistd.h>

#include <cassert>
#include <stdexcept>

#include "cereal/messaging/messaging.h"
#include "common/swaglog.h"
#include "common/util.h"

Panda::Panda(std::string serial, uint32_t bus_offset) : bus_offset(bus_offset) {
  // TODO: support SPI here one day...
  if (serial.find("spi") != std::string::npos) {
    handle = std::make_unique<PandaSpiHandle>(serial);
  } else {
    handle = std::make_unique<PandaUsbHandle>(serial);
  }

  hw_type = get_hw_type();

  assert((hw_type != cereal::PandaState::PandaType::WHITE_PANDA) &&
         (hw_type != cereal::PandaState::PandaType::GREY_PANDA));

  has_rtc = (hw_type == cereal::PandaState::PandaType::UNO) ||
            (hw_type == cereal::PandaState::PandaType::DOS) ||
            (hw_type == cereal::PandaState::PandaType::TRES);

  return;
}

bool Panda::connected() {
  return handle->connected;
}

bool Panda::comms_healthy() {
  return handle->comms_healthy;
}

std::vector<std::string> Panda::list() {
  return PandaUsbHandle::list();
}

void Panda::set_safety_model(cereal::CarParams::SafetyModel safety_model, uint16_t safety_param) {
  handle->control_write(0xdc, (uint16_t)safety_model, safety_param);
}

void Panda::set_alternative_experience(uint16_t alternative_experience) {
  handle->control_write(0xdf, alternative_experience, 0);
}

cereal::PandaState::PandaType Panda::get_hw_type() {
  unsigned char hw_query[1] = {0};

  handle->control_read(0xc1, 0, 0, hw_query, 1);
  return (cereal::PandaState::PandaType)(hw_query[0]);
}

void Panda::set_rtc(struct tm sys_time) {
  // tm struct has year defined as years since 1900
  handle->control_write(0xa1, (uint16_t)(1900 + sys_time.tm_year), 0);
  handle->control_write(0xa2, (uint16_t)(1 + sys_time.tm_mon), 0);
  handle->control_write(0xa3, (uint16_t)sys_time.tm_mday, 0);
  // handle->control_write(0xa4, (uint16_t)(1 + sys_time.tm_wday), 0);
  handle->control_write(0xa5, (uint16_t)sys_time.tm_hour, 0);
  handle->control_write(0xa6, (uint16_t)sys_time.tm_min, 0);
  handle->control_write(0xa7, (uint16_t)sys_time.tm_sec, 0);
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

  handle->control_read(0xa0, 0, 0, (unsigned char*)&rtc_time, sizeof(rtc_time));

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
  handle->control_write(0xb1, fan_speed, 0);
}

uint16_t Panda::get_fan_speed() {
  uint16_t fan_speed_rpm = 0;
  handle->control_read(0xb2, 0, 0, (unsigned char*)&fan_speed_rpm, sizeof(fan_speed_rpm));
  return fan_speed_rpm;
}

void Panda::set_ir_pwr(uint16_t ir_pwr) {
  handle->control_write(0xb0, ir_pwr, 0);
}

std::optional<health_t> Panda::get_state() {
  health_t health {0};
  int err = handle->control_read(0xd2, 0, 0, (unsigned char*)&health, sizeof(health));
  return err >= 0 ? std::make_optional(health) : std::nullopt;
}

std::optional<can_health_t> Panda::get_can_state(uint16_t can_number) {
  can_health_t can_health {0};
  int err = handle->control_read(0xc2, can_number, 0, (unsigned char*)&can_health, sizeof(can_health));
  return err >= 0 ? std::make_optional(can_health) : std::nullopt;
}

void Panda::set_loopback(bool loopback) {
  handle->control_write(0xe5, loopback, 0);
}

std::optional<std::vector<uint8_t>> Panda::get_firmware_version() {
  std::vector<uint8_t> fw_sig_buf(128);
  int read_1 = handle->control_read(0xd3, 0, 0, &fw_sig_buf[0], 64);
  int read_2 = handle->control_read(0xd4, 0, 0, &fw_sig_buf[64], 64);
  return ((read_1 == 64) && (read_2 == 64)) ? std::make_optional(fw_sig_buf) : std::nullopt;
}

std::optional<std::string> Panda::get_serial() {
  char serial_buf[17] = {'\0'};
  int err = handle->control_read(0xd0, 0, 0, (uint8_t*)serial_buf, 16);
  return err >= 0 ? std::make_optional(serial_buf) : std::nullopt;
}

void Panda::set_power_saving(bool power_saving) {
  handle->control_write(0xe7, power_saving, 0);
}

void Panda::enable_deepsleep() {
  handle->control_write(0xfb, 0, 0);
}

void Panda::send_heartbeat(bool engaged) {
  handle->control_write(0xf3, engaged, 0);
}

void Panda::set_can_speed_kbps(uint16_t bus, uint16_t speed) {
  handle->control_write(0xde, bus, (speed * 10));
}

void Panda::set_data_speed_kbps(uint16_t bus, uint16_t speed) {
  handle->control_write(0xf9, bus, (speed * 10));
}

void Panda::set_canfd_non_iso(uint16_t bus, bool non_iso) {
  handle->control_write(0xfc, bus, non_iso);
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

void Panda::pack_can_buffer(const capnp::List<cereal::CanData>::Reader &can_data_list,
                            std::function<void(uint8_t *, size_t)> write_func) {
  int32_t pos = 0;
  uint8_t send_buf[2 * USB_TX_SOFT_LIMIT];

  uint32_t magic = CAN_TRANSACTION_MAGIC;
  memcpy(&send_buf[0], &magic, sizeof(uint32_t));
  pos += sizeof(uint32_t);

  for (auto cmsg : can_data_list) {
    // check if the message is intended for this panda
    uint8_t bus = cmsg.getSrc();
    if (bus < bus_offset || bus >= (bus_offset + PANDA_BUS_CNT)) {
      continue;
    }
    auto can_data = cmsg.getDat();
    uint8_t data_len_code = len_to_dlc(can_data.size());
    assert(can_data.size() <= 64);
    assert(can_data.size() == dlc_to_len[data_len_code]);

    can_header header;
    header.addr = cmsg.getAddress();
    header.extended = (cmsg.getAddress() >= 0x800) ? 1 : 0;
    header.data_len_code = data_len_code;
    header.bus = bus - bus_offset;

    memcpy(&send_buf[pos], (uint8_t *)&header, sizeof(can_header));
    pos += sizeof(can_header);
    memcpy(&send_buf[pos], (uint8_t *)can_data.begin(), can_data.size());
    pos += can_data.size();

    if (pos >= USB_TX_SOFT_LIMIT) {
      write_func(send_buf, pos);
      pos = sizeof(uint32_t);
    }
  }

  // send remaining packets
  if (pos > sizeof(uint32_t)) write_func(send_buf, pos);
}

void Panda::can_send(capnp::List<cereal::CanData>::Reader can_data_list) {
  pack_can_buffer(can_data_list, [=](uint8_t* data, size_t size) {
    handle->bulk_write(3, data, size, 5);
  });
}

bool Panda::can_receive(std::vector<can_frame>& out_vec) {
  uint8_t data[RECV_SIZE];
  int recv = handle->bulk_read(0x81, (uint8_t*)data, RECV_SIZE);
  if (!comms_healthy()) {
    return false;
  }
  if (recv == RECV_SIZE) {
    LOGW("Panda receive buffer full");
  }

  return (recv <= 0) ? true : unpack_can_buffer(data, recv, out_vec);
}

bool Panda::unpack_can_buffer(uint8_t *data, int size, std::vector<can_frame> &out_vec) {
  if (size < sizeof(uint32_t)) {
    return true;
  }

  uint32_t magic;
  memcpy(&magic, &data[0], sizeof(uint32_t));
  if (magic != CAN_TRANSACTION_MAGIC) {
    LOGE("CAN recv: buffer didn't start with magic");
    handle->comms_healthy = false;
    return false;
  }

  int pos = sizeof(uint32_t);
  while (pos < size) {
    can_header header;
    memcpy(&header, &data[pos], sizeof(can_header));

    can_frame &canData = out_vec.emplace_back();
    canData.busTime = 0;
    canData.address = header.addr;
    canData.src = header.bus + bus_offset;
    if (header.rejected) {
      canData.src += CAN_REJECTED_BUS_OFFSET;
    }
    if (header.returned) {
      canData.src += CAN_RETURNED_BUS_OFFSET;
    }

    const uint8_t data_len = dlc_to_len[header.data_len_code];
    canData.dat.assign((char *)&data[pos + sizeof(can_header)], data_len);

    pos += sizeof(can_header) + data_len;
  }
  return true;
}
