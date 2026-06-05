#include "tools/cabana/panda.h"

#include <unistd.h>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <memory>

#include "cereal/messaging/messaging.h"
#include "common/swaglog.h"
#include "common/util.h"

static libusb_context *init_usb_ctx() {
  libusb_context *context = nullptr;
  int err = libusb_init(&context);
  if (err != 0) {
    LOGE("libusb initialization error");
    return nullptr;
  }

#if LIBUSB_API_VERSION >= 0x01000106
  libusb_set_option(context, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_INFO);
#else
  libusb_set_debug(context, 3);
#endif
  return context;
}

Panda::Panda(std::string serial, uint32_t bus_offset) : bus_offset(bus_offset) {
  if (!init_usb_connection(serial)) {
    throw std::runtime_error("Error connecting to panda");
  }

  LOGW("connected to %s over USB", serial.c_str());
  hw_type = get_hw_type();
  can_reset_communications();
}

Panda::~Panda() {
  cleanup_usb();
}

bool Panda::connected() {
  return connected_flag;
}

bool Panda::comms_healthy() {
  return comms_healthy_flag;
}

std::string Panda::hw_serial() {
  return hw_serial_str;
}

std::vector<std::string> Panda::list(bool usb_only) {
  static std::unique_ptr<libusb_context, decltype(&libusb_exit)> context(init_usb_ctx(), libusb_exit);

  ssize_t num_devices;
  libusb_device **dev_list = NULL;
  std::vector<std::string> serials;
  if (!context) { return serials; }

  num_devices = libusb_get_device_list(context.get(), &dev_list);
  if (num_devices < 0) {
    LOGE("libusb can't get device list");
    goto finish;
  }

  for (size_t i = 0; i < num_devices; ++i) {
    libusb_device *device = dev_list[i];
    libusb_device_descriptor desc;
    libusb_get_device_descriptor(device, &desc);
    if (desc.idVendor == 0x3801 && desc.idProduct == 0xddcc) {
      libusb_device_handle *handle = NULL;
      int ret = libusb_open(device, &handle);
      if (ret < 0) { goto finish; }

      unsigned char desc_serial[26] = { 0 };
      ret = libusb_get_string_descriptor_ascii(handle, desc.iSerialNumber, desc_serial, std::size(desc_serial));
      libusb_close(handle);
      if (ret < 0) { goto finish; }

      serials.push_back(std::string((char *)desc_serial, ret));
    }
  }

finish:
  if (dev_list != NULL) {
    libusb_free_device_list(dev_list, 1);
  }
  return serials;
}

void Panda::set_safety_model(cereal::CarParams::SafetyModel safety_model, uint16_t safety_param) {
  control_write(0xdc, (uint16_t)safety_model, safety_param);
}


cereal::PandaState::PandaType Panda::get_hw_type() {
  unsigned char hw_query[1] = {0};

  control_read(0xc1, 0, 0, hw_query, 1);
  return (cereal::PandaState::PandaType)(hw_query[0]);
}




void Panda::send_heartbeat(bool engaged) {
  control_write(0xf3, engaged, 0);
}

void Panda::set_can_speed_kbps(uint16_t bus, uint16_t speed) {
  control_write(0xde, bus, (speed * 10));
}


void Panda::set_data_speed_kbps(uint16_t bus, uint16_t speed) {
  control_write(0xf9, bus, (speed * 10));
}



bool Panda::can_receive(std::vector<can_frame>& out_vec) {
  // Check if enough space left in buffer to store RECV_SIZE data
  assert(receive_buffer_size + RECV_SIZE <= sizeof(receive_buffer));

  int recv = bulk_read(0x81, &receive_buffer[receive_buffer_size], RECV_SIZE);
  if (!comms_healthy()) {
    return false;
  }

  bool ret = true;
  if (recv > 0) {
    receive_buffer_size += recv;
    ret = unpack_can_buffer(receive_buffer, receive_buffer_size, out_vec);
  }
  return ret;
}

void Panda::can_reset_communications() {
  control_write(0xc0, 0, 0);
}

bool Panda::unpack_can_buffer(uint8_t *data, uint32_t &size, std::vector<can_frame> &out_vec) {
  int pos = 0;

  while (pos <= size - sizeof(can_header)) {
    can_header header;
    memcpy(&header, &data[pos], sizeof(can_header));

    const uint8_t data_len = dlc_to_len[header.data_len_code];
    if (pos + sizeof(can_header) + data_len > size) {
      // we don't have all the data for this message yet
      break;
    }

    if (calculate_checksum(&data[pos], sizeof(can_header) + data_len) != 0) {
      LOGE("Panda CAN checksum failed");
      size = 0;
      can_reset_communications();
      return false;
    }

    can_frame &canData = out_vec.emplace_back();
    canData.address = header.addr;
    canData.src = header.bus + bus_offset;
    if (header.rejected) {
      canData.src += CAN_REJECTED_BUS_OFFSET;
    }
    if (header.returned) {
      canData.src += CAN_RETURNED_BUS_OFFSET;
    }

    canData.dat.assign((char *)&data[pos + sizeof(can_header)], data_len);

    pos += sizeof(can_header) + data_len;
  }

  // move the overflowing data to the beginning of the buffer for the next round
  memmove(data, &data[pos], size - pos);
  size -= pos;

  return true;
}

uint8_t Panda::calculate_checksum(uint8_t *data, uint32_t len) {
  uint8_t checksum = 0U;
  for (uint32_t i = 0U; i < len; i++) {
    checksum ^= data[i];
  }
  return checksum;
}

// USB implementation methods
bool Panda::init_usb_connection(const std::string& serial) {
  ssize_t num_devices;
  libusb_device **dev_list = NULL;
  int err = 0;

  ctx = init_usb_ctx();
  if (!ctx) { goto fail; }

  // connect by serial
  num_devices = libusb_get_device_list(ctx, &dev_list);
  if (num_devices < 0) { goto fail; }

  for (size_t i = 0; i < num_devices; ++i) {
    libusb_device_descriptor desc;
    libusb_get_device_descriptor(dev_list[i], &desc);
    if (desc.idVendor == 0x3801 && desc.idProduct == 0xddcc) {
      int ret = libusb_open(dev_list[i], &dev_handle);
      if (dev_handle == NULL || ret < 0) { goto fail; }

      unsigned char desc_serial[26] = { 0 };
      ret = libusb_get_string_descriptor_ascii(dev_handle, desc.iSerialNumber, desc_serial, std::size(desc_serial));
      if (ret < 0) { goto fail; }

      hw_serial_str = std::string((char *)desc_serial, ret);
      if (serial.empty() || serial == hw_serial_str) {
        break;
      }
      libusb_close(dev_handle);
      dev_handle = NULL;
    }
  }
  if (dev_handle == NULL) goto fail;
  libusb_free_device_list(dev_list, 1);

  if (libusb_kernel_driver_active(dev_handle, 0) == 1) {
    libusb_detach_kernel_driver(dev_handle, 0);
  }

  err = libusb_set_configuration(dev_handle, 1);
  if (err != 0) { goto fail; }

  err = libusb_claim_interface(dev_handle, 0);
  if (err != 0) { goto fail; }

  return true;

fail:
  if (dev_list != NULL) {
    libusb_free_device_list(dev_list, 1);
  }
  cleanup_usb();
  return false;
}

void Panda::cleanup_usb() {
  if (dev_handle) {
    libusb_release_interface(dev_handle, 0);
    libusb_close(dev_handle);
    dev_handle = nullptr;
  }

  if (ctx) {
    libusb_exit(ctx);
    ctx = nullptr;
  }
  connected_flag = false;
}

void Panda::handle_usb_issue(int err, const char func[]) {
  LOGE_100("usb error %d \"%s\" in %s", err, libusb_strerror((enum libusb_error)err), func);
  if (err == LIBUSB_ERROR_NO_DEVICE) {
    LOGE("lost connection");
    connected_flag = false;
  }
}

int Panda::control_write(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout) {
  int err;
  const uint8_t bmRequestType = LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;

  if (!connected_flag) {
    return LIBUSB_ERROR_NO_DEVICE;
  }

  do {
    err = libusb_control_transfer(dev_handle, bmRequestType, bRequest, wValue, wIndex, NULL, 0, timeout);
    if (err < 0) handle_usb_issue(err, __func__);
  } while (err < 0 && connected_flag);

  return err;
}

int Panda::control_read(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout) {
  int err;
  const uint8_t bmRequestType = LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;

  if (!connected_flag) {
    return LIBUSB_ERROR_NO_DEVICE;
  }

  do {
    err = libusb_control_transfer(dev_handle, bmRequestType, bRequest, wValue, wIndex, data, wLength, timeout);
    if (err < 0) handle_usb_issue(err, __func__);
  } while (err < 0 && connected_flag);

  return err;
}

int Panda::bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  int err;
  int transferred = 0;

  if (!connected_flag) {
    return 0;
  }

  do {
    err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);
    if (err == LIBUSB_ERROR_TIMEOUT) {
      LOGW("Transmit buffer full");
      break;
    } else if (err != 0 || length != transferred) {
      handle_usb_issue(err, __func__);
    }
  } while (err != 0 && connected_flag);

  return transferred;
}

int Panda::bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  int err;
  int transferred = 0;

  if (!connected_flag) {
    return 0;
  }

  do {
    err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);
    if (err == LIBUSB_ERROR_TIMEOUT) {
      break; // timeout is okay to exit, recv still happened
    } else if (err == LIBUSB_ERROR_OVERFLOW) {
      comms_healthy_flag = false;
      LOGE_100("overflow got 0x%x", transferred);
    } else if (err != 0) {
      handle_usb_issue(err, __func__);
    }
  } while (err != 0 && connected_flag);

  return transferred;
}
