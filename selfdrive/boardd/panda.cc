#include "selfdrive/boardd/panda.h"

#include <unistd.h>

#include <cassert>
#include <stdexcept>
#include <vector>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/gpio.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"

namespace {

libusb_context *init_usb_ctx() {
  libusb_context *context = nullptr;
  int err = libusb_init(&context);
  if (err != 0) {
    LOGE("libusb initialization error %d", err);
    return nullptr;
  }

#if LIBUSB_API_VERSION >= 0x01000106
  libusb_set_option(context, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_INFO);
#else
  libusb_set_debug(context, 3);
#endif
  return context;
}

struct UsbDeviceList {
  UsbDeviceList(libusb_context* ctx) {
    num_devices = libusb_get_device_list(ctx, &dev_list);
    if (num_devices < 0) {
      LOGE("libusb can't get device list");
    }
  }

  ~UsbDeviceList() {
    if (dev_list) libusb_free_device_list(dev_list, 1);
  }

  auto list(uint16_t vid, uint16_t pid) {
    std::vector<std::pair<libusb_device *, std::string>> result;

    for (size_t i = 0; i < num_devices; ++i) {
      libusb_device_descriptor desc = {};
      libusb_device_handle *handle = nullptr;
      int ret = libusb_get_device_descriptor(dev_list[i], &desc);
      if (ret == 0 && desc.idVendor == vid && desc.idProduct == pid && libusb_open(dev_list[i], &handle) == 0) {
        unsigned char serial[256] = {'\0'};
        libusb_get_string_descriptor_ascii(handle, desc.iSerialNumber, serial, std::size(serial) - 1);
        result.push_back({dev_list[i], (const char *)serial});
        libusb_close(handle);
      }
    }

    return result;
  }

  libusb_device **dev_list = nullptr;
  ssize_t num_devices = 0;
};

libusb_device_handle *open_usb_device(libusb_context *ctx, uint16_t vid, uint16_t pid, const std::string &serial) {
  libusb_device_handle *dev_handle = nullptr;
  UsbDeviceList device_list(ctx);
  for (auto &[dev, device_serial] : device_list.list(vid, pid)) {
    if (serial.empty() || serial == device_serial) {
      libusb_open(dev, &dev_handle);
      break;
    }
  }
  if (dev_handle) {
    if (libusb_kernel_driver_active(dev_handle, 0) == 1) {
      libusb_detach_kernel_driver(dev_handle, 0);
    }
    if (libusb_set_configuration(dev_handle, 1) != 0 || libusb_claim_interface(dev_handle, 0) != 0) {
      libusb_close(dev_handle);
      dev_handle = nullptr;
    }
  }
  return dev_handle;
}

}  // namespace

Panda::Panda(std::string serial) {
  ctx = init_usb_ctx();
  if (!ctx || !(dev_handle = open_usb_device(ctx, PANDA_VENDOR_ID, PANDA_PRODUCT_ID, serial))) {
    cleanup();
    throw std::runtime_error("Error connecting to panda");
  }

  hw_type = get_hw_type();
  assert((hw_type != cereal::PandaState::PandaType::WHITE_PANDA) &&
         (hw_type != cereal::PandaState::PandaType::GREY_PANDA));

  has_rtc = (hw_type == cereal::PandaState::PandaType::UNO) ||
            (hw_type == cereal::PandaState::PandaType::DOS);
}

Panda::~Panda() {
  std::lock_guard lk(usb_lock);
  cleanup();
  connected = false;
}

void Panda::cleanup() {
  if (dev_handle) {
    libusb_release_interface(dev_handle, 0);
    libusb_close(dev_handle);
  }
  if (ctx) {
    libusb_exit(ctx);
  }
}

std::vector<std::string> Panda::list() {
  std::vector<std::string> serials;
  if (libusb_context *context = init_usb_ctx()) {
    UsbDeviceList device_list(context);
    for (auto &[dev, serial] : device_list.list(PANDA_VENDOR_ID, PANDA_PRODUCT_ID)) {
      serials.push_back(serial);
    }
    libusb_exit(context);
  }
  return serials;
}

void Panda::handle_usb_issue(int err, const char func[]) {
  LOGE_100("usb error %d \"%s\" in %s", err, libusb_strerror((enum libusb_error)err), func);
  if (err == LIBUSB_ERROR_NO_DEVICE) {
    LOGE("lost connection");
    connected = false;
  }
  // TODO: check other errors, is simply retrying okay?
}

int Panda::usb_write(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout) {
  int err;
  const uint8_t bmRequestType = LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;

  if (!connected) {
    return LIBUSB_ERROR_NO_DEVICE;
  }

  std::lock_guard lk(usb_lock);
  do {
    err = libusb_control_transfer(dev_handle, bmRequestType, bRequest, wValue, wIndex, NULL, 0, timeout);
    if (err < 0) handle_usb_issue(err, __func__);
  } while (err < 0 && connected);

  return err;
}

int Panda::usb_read(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout) {
  int err;
  const uint8_t bmRequestType = LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;

  if (!connected) {
    return LIBUSB_ERROR_NO_DEVICE;
  }

  std::lock_guard lk(usb_lock);
  do {
    err = libusb_control_transfer(dev_handle, bmRequestType, bRequest, wValue, wIndex, data, wLength, timeout);
    if (err < 0) handle_usb_issue(err, __func__);
  } while (err < 0 && connected);

  return err;
}

int Panda::usb_bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  int err;
  int transferred = 0;

  if (!connected) {
    return 0;
  }

  std::lock_guard lk(usb_lock);
  do {
    // Try sending can messages. If the receive buffer on the panda is full it will NAK
    // and libusb will try again. After 5ms, it will time out. We will drop the messages.
    err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);

    if (err == LIBUSB_ERROR_TIMEOUT) {
      LOGW("Transmit buffer full");
      break;
    } else if (err != 0 || length != transferred) {
      handle_usb_issue(err, __func__);
    }
  } while(err != 0 && connected);

  return transferred;
}

int Panda::usb_bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  int err;
  int transferred = 0;

  if (!connected) {
    return 0;
  }

  std::lock_guard lk(usb_lock);

  do {
    err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);

    if (err == LIBUSB_ERROR_TIMEOUT) {
      break; // timeout is okay to exit, recv still happened
    } else if (err == LIBUSB_ERROR_OVERFLOW) {
      comms_healthy = false;
      LOGE_100("overflow got 0x%x", transferred);
    } else if (err != 0) {
      handle_usb_issue(err, __func__);
    }

  } while(err != 0 && connected);

  return transferred;
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
