#include "selfdrive/boardd/panda.h"

#include <cassert>
#include <stdexcept>

#include "selfdrive/common/gpio.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"

Panda::Panda(std::string serial, uint32_t bus_offset) : bus_offset(bus_offset), PandaComm(PANDA_VENDOR_ID, PANDA_PRODUCT_ID, serial) {
  hw_type = get_hw_type();
  assert((hw_type != cereal::PandaState::PandaType::WHITE_PANDA) &&
         (hw_type != cereal::PandaState::PandaType::GREY_PANDA));

  has_rtc = (hw_type == cereal::PandaState::PandaType::UNO) ||
            (hw_type == cereal::PandaState::PandaType::DOS);
}

Panda::~Panda() {}

void Panda::set_safety_model(cereal::CarParams::SafetyModel safety_model, int safety_param) {
  usb_write(0xdc, (uint16_t)safety_model, safety_param);
}

void Panda::set_unsafe_mode(uint16_t unsafe_mode) {
  usb_write(0xdf, unsafe_mode, 0);
}

cereal::PandaState::PandaType Panda::get_hw_type() {
  unsigned char hw_query[1] = {0};

  usb_read(0xc1, 0, 0, hw_query, 1);
  return (cereal::PandaState::PandaType)(hw_query[0]);
}

void Panda::set_rtc(struct tm sys_time) {
  // tm struct has year defined as years since 1900
  usb_write(0xa1, (uint16_t)(1900 + sys_time.tm_year), 0);
  usb_write(0xa2, (uint16_t)(1 + sys_time.tm_mon), 0);
  usb_write(0xa3, (uint16_t)sys_time.tm_mday, 0);
  // usb_write(0xa4, (uint16_t)(1 + sys_time.tm_wday), 0);
  usb_write(0xa5, (uint16_t)sys_time.tm_hour, 0);
  usb_write(0xa6, (uint16_t)sys_time.tm_min, 0);
  usb_write(0xa7, (uint16_t)sys_time.tm_sec, 0);
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

  usb_read(0xa0, 0, 0, (unsigned char*)&rtc_time, sizeof(rtc_time));

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
  usb_write(0xb1, fan_speed, 0);
}

uint16_t Panda::get_fan_speed() {
  uint16_t fan_speed_rpm = 0;
  usb_read(0xb2, 0, 0, (unsigned char*)&fan_speed_rpm, sizeof(fan_speed_rpm));
  return fan_speed_rpm;
}

void Panda::set_ir_pwr(uint16_t ir_pwr) {
  usb_write(0xb0, ir_pwr, 0);
}

health_t Panda::get_state() {
  health_t health {0};
  usb_read(0xd2, 0, 0, (unsigned char*)&health, sizeof(health));
  return health;
}

void Panda::set_loopback(bool loopback) {
  usb_write(0xe5, loopback, 0);
}

std::optional<std::vector<uint8_t>> Panda::get_firmware_version() {
  std::vector<uint8_t> fw_sig_buf(128);
  int read_1 = usb_read(0xd3, 0, 0, &fw_sig_buf[0], 64);
  int read_2 = usb_read(0xd4, 0, 0, &fw_sig_buf[64], 64);
  return ((read_1 == 64) && (read_2 == 64)) ? std::make_optional(fw_sig_buf) : std::nullopt;
}

std::optional<std::string> Panda::get_serial() {
  char serial_buf[17] = {'\0'};
  int err = usb_read(0xd0, 0, 0, (uint8_t*)serial_buf, 16);
  return err >= 0 ? std::make_optional(serial_buf) : std::nullopt;
}

void Panda::set_power_saving(bool power_saving) {
  usb_write(0xe7, power_saving, 0);
}

void Panda::set_usb_power_mode(cereal::PeripheralState::UsbPowerMode power_mode) {
  usb_write(0xe6, (uint16_t)power_mode, 0);
}

void Panda::send_heartbeat() {
  usb_write(0xf3, 1, 0);
}

void Panda::can_send(capnp::List<cereal::CanData>::Reader can_data_list) {
  send.resize(4 * can_data_list.size());

  uint32_t msg_cnt = 0;
  for (int i = 0; i < can_data_list.size(); i++) {
    auto cmsg = can_data_list[i];

    // check if the message is intended for this panda
    uint8_t bus = cmsg.getSrc();
    if (bus < bus_offset || bus >= (bus_offset + PANDA_BUS_CNT)) {
      continue;
    }

    if (cmsg.getAddress() >= 0x800) { // extended
      send[msg_cnt*4] = (cmsg.getAddress() << 3) | 5;
    } else { // normal
      send[msg_cnt*4] = (cmsg.getAddress() << 21) | 1;
    }
    auto can_data = cmsg.getDat();
    assert(can_data.size() <= 8);
    send[msg_cnt*4+1] = can_data.size() | ((bus - bus_offset) << 4);
    memcpy(&send[msg_cnt*4+2], can_data.begin(), can_data.size());

    msg_cnt++;
  }

  usb_bulk_write(3, (unsigned char*)send.data(), msg_cnt * 0x10, 5);
}

bool Panda::can_receive(std::vector<can_frame>& out_vec) {
  uint32_t data[RECV_SIZE/4];
  int recv = usb_bulk_read(0x81, (unsigned char*)data, RECV_SIZE);

  // Not sure if this can happen
  if (recv < 0) recv = 0;

  if (recv == RECV_SIZE) {
    LOGW("Receive buffer full");
  }

  if (!comms_healthy) {
    return false;
  }

  // Append to the end of the out_vec, such that we can pass it to multiple pandas
  // We already insert space for all the messages here for speed
  size_t num_msg = recv / 0x10;
  out_vec.reserve(out_vec.size() + num_msg);

  // Populate messages
  for (int i = 0; i < num_msg; i++) {
    can_frame canData;
    if (data[i*4] & 4) {
      // extended
      canData.address = data[i*4] >> 3;
      //printf("got extended: %x\n", data[i*4] >> 3);
    } else {
      // normal
      canData.address = data[i*4] >> 21;
    }
    canData.busTime = data[i*4+1] >> 16;
    int len = data[i*4+1] & 0xF;
    canData.dat.assign((char *)&data[i*4+2], len);
    canData.src = ((data[i*4+1] >> 4) & 0xff) + bus_offset;

    // add to vector
    out_vec.push_back(canData);
  }

  return true;
}
