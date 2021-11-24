#include "selfdrive/boardd/panda.h"

#include <unistd.h>

#include <cassert>
#include <stdexcept>
#include <vector>

#include "cereal/messaging/messaging.h"
#include "panda/board/dlc_to_len.h"
#include "selfdrive/common/gpio.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"

static int init_usb_ctx(libusb_context **context) {
  assert(context != nullptr);

  int err = libusb_init(context);
  if (err != 0) {
    LOGE("libusb initialization error");
    return err;
  }

#if LIBUSB_API_VERSION >= 0x01000106
  libusb_set_option(*context, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_INFO);
#else
  libusb_set_debug(*context, 3);
#endif

  return err;
}


Panda::Panda(std::string serial, uint32_t bus_offset) : bus_offset(bus_offset) {
  // init libusb
  ssize_t num_devices;
  libusb_device **dev_list = NULL;
  int err = init_usb_ctx(&ctx);
  if (err != 0) { goto fail; }

  // connect by serial
  num_devices = libusb_get_device_list(ctx, &dev_list);
  if (num_devices < 0) { goto fail; }
  for (size_t i = 0; i < num_devices; ++i) {
    libusb_device_descriptor desc;
    libusb_get_device_descriptor(dev_list[i], &desc);
    if (desc.idVendor == 0xbbaa && desc.idProduct == 0xddcc) {
      libusb_open(dev_list[i], &dev_handle);
      if (dev_handle == NULL) { goto fail; }

      unsigned char desc_serial[26] = { 0 };
      int ret = libusb_get_string_descriptor_ascii(dev_handle, desc.iSerialNumber, desc_serial, std::size(desc_serial));
      if (ret < 0) { goto fail; }

      usb_serial = std::string((char *)desc_serial, ret).c_str();
      if (serial.empty() || serial == usb_serial) {
        break;
      }
      libusb_close(dev_handle);
      dev_handle = NULL;
    }
  }
  if (dev_handle == NULL) goto fail;
  libusb_free_device_list(dev_list, 1);
  dev_list = nullptr;

  if (libusb_kernel_driver_active(dev_handle, 0) == 1) {
    libusb_detach_kernel_driver(dev_handle, 0);
  }

  err = libusb_set_configuration(dev_handle, 1);
  if (err != 0) { goto fail; }

  err = libusb_claim_interface(dev_handle, 0);
  if (err != 0) { goto fail; }

  hw_type = get_hw_type();

  assert((hw_type != cereal::PandaState::PandaType::WHITE_PANDA) &&
         (hw_type != cereal::PandaState::PandaType::GREY_PANDA));

  has_rtc = (hw_type == cereal::PandaState::PandaType::UNO) ||
            (hw_type == cereal::PandaState::PandaType::DOS);

  return;

fail:
  if (dev_list != NULL) {
    libusb_free_device_list(dev_list, 1);
  }
  cleanup();
  throw std::runtime_error("Error connecting to panda");
}

Panda::~Panda() {
  std::lock_guard lk(usb_lock);
  cleanup();
  connected = false;
  Panda::all_connected = false;
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
  // init libusb
  ssize_t num_devices;
  libusb_context *context = NULL;
  libusb_device **dev_list = NULL;
  std::vector<std::string> serials;

  int err = init_usb_ctx(&context);
  if (err != 0) { return serials; }

  num_devices = libusb_get_device_list(context, &dev_list);
  if (num_devices < 0) {
    LOGE("libusb can't get device list");
    goto finish;
  }
  for (size_t i = 0; i < num_devices; ++i) {
    libusb_device *device = dev_list[i];
    libusb_device_descriptor desc;
    libusb_get_device_descriptor(device, &desc);
    if (desc.idVendor == 0xbbaa && desc.idProduct == 0xddcc) {
      libusb_device_handle *handle = NULL;
      libusb_open(device, &handle);
      unsigned char desc_serial[26] = { 0 };
      int ret = libusb_get_string_descriptor_ascii(handle, desc.iSerialNumber, desc_serial, std::size(desc_serial));
      libusb_close(handle);

      if (ret < 0) { goto finish; }
      serials.push_back(std::string((char *)desc_serial, ret).c_str());
    }
  }

finish:
  if (dev_list != NULL) {
    libusb_free_device_list(dev_list, 1);
  }
  if (context) {
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

void Panda::set_can_speed_kbps(uint16_t bus, uint16_t speed) {
  usb_write(0xde, bus, (speed * 10));
}

void Panda::set_data_speed_kbps(uint16_t bus, uint16_t speed) {
  usb_write(0xf9, bus, (speed * 10));
}

uint8_t Panda::len_to_dlc(uint8_t len) {
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
  if (send.size() < (can_data_list.size() * CANPACKET_MAX_SIZE)) {
    send.resize(can_data_list.size() * CANPACKET_MAX_SIZE);
  }

  int msg_count = 0;
  while (msg_count < can_data_list.size()) {
    uint32_t pos = 0;
    while (pos < USB_TX_SOFT_LIMIT) {
      if (msg_count == can_data_list.size()) { break; }
      auto cmsg = can_data_list[msg_count];

      // check if the message is intended for this panda
      uint8_t bus = cmsg.getSrc();
      if (bus < bus_offset || bus >= (bus_offset + PANDA_BUS_CNT)) {
        msg_count++;
        continue;
      }
      auto can_data = cmsg.getDat();
      uint8_t data_len_code = len_to_dlc(can_data.size());
      assert(can_data.size() <= (hw_type == cereal::PandaState::PandaType::RED_PANDA) ? 64 : 8);
      assert(can_data.size() == dlc_to_len[data_len_code]);

      can_header header;
      header.addr = cmsg.getAddress();
      header.extended = (cmsg.getAddress() >= 0x800) ? 1 : 0;
      header.data_len_code = data_len_code;
      header.bus = bus - bus_offset;
      memcpy(&send[pos], &header, CANPACKET_HEAD_SIZE);
      memcpy(&send[pos+CANPACKET_HEAD_SIZE], can_data.begin(), can_data.size());

      pos += CANPACKET_HEAD_SIZE + dlc_to_len[data_len_code];
      msg_count++;
    }

    if (pos > 0) { // Helps not to spam with ZLP
      // Counter needs to be inserted every 64 bytes (first byte of 64 bytes USB packet)
      uint8_t counter = 0;
      uint8_t to_write[USB_TX_SOFT_LIMIT+128];
      int ptr = 0;
      for (int i = 0; i < pos; i += 63) {
        to_write[ptr] = counter;
        int copy_size = ((pos - i) < 63) ? (pos - i) : 63;
        memcpy(&to_write[ptr+1], &(send.data()[i]) , copy_size);
        ptr += copy_size + 1;
        counter++;
      }
      write_func(to_write, ptr);
    }
  }
}

void Panda::can_send(capnp::List<cereal::CanData>::Reader can_data_list) {
  pack_can_buffer(can_data_list, [=](uint8_t* data, size_t size) {
    usb_bulk_write(3, data, size, 5);
  });
}

bool Panda::can_receive(std::vector<can_frame>& out_vec) {
  uint8_t data[RECV_SIZE];
  int recv = usb_bulk_read(0x81, (uint8_t*)data, RECV_SIZE);

  // Not sure if this can happen
  if (recv < 0) recv = 0;

  if (recv == RECV_SIZE) {
    LOGW("Receive buffer full");
  }

  if (!comms_healthy) {
    return false;
  }
  return unpack_can_buffer(data, recv, out_vec);
}

bool Panda::unpack_can_buffer(uint8_t *data, int size, std::vector<can_frame> &out_vec) {

  static uint8_t tail[CANPACKET_MAX_SIZE];
  uint8_t tail_size = 0;
  uint8_t counter = 0;
  for (int i = 0; i < size; i += USBPACKET_MAX_SIZE) {
    // Check for counter every 64 bytes (length of USB packet)
    if (counter != data[i]) {
      LOGE("CAN: MALFORMED USB RECV PACKET");
      break;
    }
    counter++;
    uint8_t chunk_len = ((size - i) > USBPACKET_MAX_SIZE) ? 63 : (size - i - 1); // as 1 is always reserved for counter
    uint8_t chunk[USBPACKET_MAX_SIZE + CANPACKET_MAX_SIZE];
    memcpy(chunk, tail, tail_size);
    memcpy(&chunk[tail_size], &data[i+1], chunk_len);
    chunk_len += tail_size;
    tail_size = 0;
    uint8_t pos = 0;
    while (pos < chunk_len) {
      uint8_t data_len = dlc_to_len[(chunk[pos] >> 4)];
      uint8_t pckt_len = CANPACKET_HEAD_SIZE + data_len;
      if (pckt_len <= (chunk_len - pos)) {
        can_header header;
        memcpy(&header, &chunk[pos], CANPACKET_HEAD_SIZE);

        can_frame &canData = out_vec.emplace_back();
        canData.busTime = 0;
        canData.address = header.addr;
        canData.src = header.bus + bus_offset;

        if (header.rejected) { canData.src += CANPACKET_REJECTED; }
        if (header.returned) { canData.src += CANPACKET_RETURNED; }
        canData.dat.assign((char*)&chunk[pos+CANPACKET_HEAD_SIZE], data_len);

        pos += pckt_len;
      } else {
        // Keep partial CAN packet until next USB packet
        tail_size = (chunk_len - pos);
        memcpy(tail, &chunk[pos], tail_size);
        break;
      }
    }
  }
  return true;
}
