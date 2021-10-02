#include "selfdrive/boardd/panda.h"

#include <cassert>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

#include "cereal/messaging/messaging.h"
#include "selfdrive/boardd/panda_dfu.h"
#include "selfdrive/boardd/panda_flashing.h"
#include "selfdrive/common/gpio.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"

libusb_device_handle *open_usb(libusb_context *ctx, uint16_t vid, uint16_t pid, std::string serial) {
  libusb_device **dev_list = nullptr;
  ssize_t num_devices = libusb_get_device_list(ctx, &dev_list);
  if (num_devices < 0) {
    LOGE("libusb can't get device list");
    return nullptr;
  }

  libusb_device_handle *handle = nullptr;
  for (size_t i = 0; i < num_devices; ++i) {
    libusb_device_descriptor desc = {};
    int ret = libusb_get_device_descriptor(dev_list[i], &desc);
    if (ret == LIBUSB_SUCCESS && desc.idVendor == vid && desc.idProduct == pid) {
      if (libusb_open(dev_list[i], &handle) == LIBUSB_SUCCESS) {
        unsigned char desc_serial[256] = {'\0'};
        int ret = libusb_get_string_descriptor_ascii(handle, desc.iSerialNumber, desc_serial, std::size(desc_serial) - 1);
        if (ret > 0 && (serial.empty() || serial == (const char *)desc_serial)) {
          break;
        }
        libusb_close(handle);
        handle = nullptr;
      }
    }
  }
  libusb_free_device_list(dev_list, 1);
  return handle;
}

PandaComm::PandaComm(uint16_t vid, uint16_t pid, std::string serial) {
  libusb_context *ctx = nullptr;
  int err = libusb_init(&ctx);
  if (err != 0) {
    LOGE("libusb initialization error");
  }
#if LIBUSB_API_VERSION >= 0x01000106
  libusb_set_option(ctx, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_INFO);
#else
  libusb_set_debug(ctx, 3);
#endif
  dev_handle = open_usb(ctx, vid, pid, serial);
  if (dev_handle) {
    if (libusb_kernel_driver_active(dev_handle, 0) == 1) {
      libusb_detach_kernel_driver(dev_handle, 0);
    }
    bool success = libusb_set_configuration(dev_handle, 1) == 0;
    success = success && libusb_claim_interface(dev_handle, 0) == 0;
    if (success) return;

    libusb_close(dev_handle);
    libusb_exit(ctx);
  }
  throw std::runtime_error("Error connecting to panda");
}

PandaComm::~PandaComm() {
  libusb_release_interface(dev_handle, 0);
  libusb_close(dev_handle);
  libusb_exit(ctx);
}

int PandaComm::usb_transfer(libusb_endpoint_direction dir, uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout) {
  const uint8_t bmRequestType = dir | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;
  int r = LIBUSB_ERROR_NO_DEVICE;
  while (connected) {
    // Release the lock before retrying on failure, so that other threads have a chance to do transfer
    std::lock_guard lk(usb_lock);
    r = libusb_control_transfer(dev_handle, bmRequestType, bRequest, wValue, wIndex, NULL, 0, timeout);
    if (r >= 0) break;

    handle_usb_issue(r, __func__);
  }
  return r;
}

int PandaComm::control_read(uint8_t request_type, uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout) {
  std::lock_guard lk(usb_lock);
  return libusb_control_transfer(dev_handle, (request_type & ~LIBUSB_ENDPOINT_DIR_MASK) | LIBUSB_ENDPOINT_IN, bRequest, wValue, wIndex, data, wLength, timeout);
}

int PandaComm::control_write(uint8_t request_type, uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout) {
  std::lock_guard lk(usb_lock);
  return libusb_control_transfer(dev_handle, (request_type & ~LIBUSB_ENDPOINT_DIR_MASK) | LIBUSB_ENDPOINT_OUT, bRequest, wValue, wIndex, data, wLength, timeout);
}

int PandaComm::usb_write(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout) {
  return usb_transfer(LIBUSB_ENDPOINT_OUT, bRequest, wValue, wIndex, timeout);
}

int PandaComm::usb_read(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout) {
  return usb_transfer(LIBUSB_ENDPOINT_IN, bRequest, wValue, wIndex, timeout);
}

int PandaComm::usb_bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  int transferred = 0;
  std::lock_guard lk(usb_lock);
  while (connected) {
    // Try sending can messages. If the receive buffer on the panda is full it will NAK
    // and libusb will try again. After 5ms, it will time out. We will drop the messages.
    int err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);
    if (err == LIBUSB_ERROR_TIMEOUT) {
      LOGW("Transmit buffer full");
      break;
    } else if (err != 0 || length != transferred) {
      handle_usb_issue(err, __func__);
    }
  };
  return transferred;
}

int PandaComm::usb_bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  int transferred = 0;
  std::lock_guard lk(usb_lock);
  while (connected) {
    int err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);
    if (err == LIBUSB_ERROR_TIMEOUT) {
      break; // timeout is okay to exit, recv still happened
    } else if (err == LIBUSB_ERROR_OVERFLOW) {
      comms_healthy = false;
      LOGE_100("overflow got 0x%x", transferred);
    } else if (err != 0) {
      handle_usb_issue(err, __func__);
    }

  };
  return transferred;
}

void PandaComm::handle_usb_issue(int err, const char func[]) {
  LOGE_100("usb error %d \"%s\" in %s", err, libusb_strerror((enum libusb_error)err), func);
  if (err == LIBUSB_ERROR_NO_DEVICE) {
    LOGE("lost connection");
    connected = false;
  }
  // TODO: check other errors, is simply retrying okay?
}


Panda::Panda() : PandaComm(0xbbaa, 0xddcc){
  hw_type = get_hw_type();
  is_pigeon = (hw_type == cereal::PandaState::PandaType::GREY_PANDA) ||
              (hw_type == cereal::PandaState::PandaType::BLACK_PANDA) ||
              (hw_type == cereal::PandaState::PandaType::UNO) ||
              (hw_type == cereal::PandaState::PandaType::DOS);
  has_rtc = (hw_type == cereal::PandaState::PandaType::UNO) ||
            (hw_type == cereal::PandaState::PandaType::DOS);
}

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

void Panda::set_usb_power_mode(cereal::PandaState::UsbPowerMode power_mode) {
  usb_write(0xe6, (uint16_t)power_mode, 0);
}

void Panda::send_heartbeat() {
  usb_write(0xf3, 1, 0);
}

void Panda::can_send(capnp::List<cereal::CanData>::Reader can_data_list) {
  static std::vector<uint32_t> send;
  const int msg_count = can_data_list.size();

  send.resize(msg_count*0x10);

  for (int i = 0; i < msg_count; i++) {
    auto cmsg = can_data_list[i];
    if (cmsg.getAddress() >= 0x800) { // extended
      send[i*4] = (cmsg.getAddress() << 3) | 5;
    } else { // normal
      send[i*4] = (cmsg.getAddress() << 21) | 1;
    }
    auto can_data = cmsg.getDat();
    assert(can_data.size() <= 8);
    send[i*4+1] = can_data.size() | (cmsg.getSrc() << 4);
    memcpy(&send[i*4+2], can_data.begin(), can_data.size());
  }

  usb_bulk_write(3, (unsigned char*)send.data(), send.size(), 5);
}

int Panda::can_receive(kj::Array<capnp::word>& out_buf) {
  uint32_t data[RECV_SIZE/4];
  int recv = usb_bulk_read(0x81, (unsigned char*)data, RECV_SIZE);

  // Not sure if this can happen
  if (recv < 0) recv = 0;

  if (recv == RECV_SIZE) {
    LOGW("Receive buffer full");
  }

  size_t num_msg = recv / 0x10;
  MessageBuilder msg;
  auto evt = msg.initEvent();
  evt.setValid(comms_healthy);

  // populate message
  auto canData = evt.initCan(num_msg);
  for (int i = 0; i < num_msg; i++) {
    if (data[i*4] & 4) {
      // extended
      canData[i].setAddress(data[i*4] >> 3);
      //printf("got extended: %x\n", data[i*4] >> 3);
    } else {
      // normal
      canData[i].setAddress(data[i*4] >> 21);
    }
    canData[i].setBusTime(data[i*4+1] >> 16);
    int len = data[i*4+1]&0xF;
    canData[i].setDat(kj::arrayPtr((uint8_t*)&data[i*4+2], len));
    canData[i].setSrc((data[i*4+1] >> 4) & 0xff);
  }
  out_buf = capnp::messageToFlatArray(msg);
  return recv;
}
