#pragma once

#include <atomic>
#include <mutex>
#include <vector>

#include <libusb-1.0/libusb.h>
#include "panda/board/global_definitions.h"

#define TIMEOUT 0

class USBDevice {
public:
  USBDevice() = default;
  ~USBDevice();
  bool open(const std::string &serial);
  static std::vector<std::string> list();

  inline int write(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout = TIMEOUT) {
    return control_transfer(LIBUSB_ENDPOINT_OUT, bRequest, wValue, wIndex, timeout);
  }
  inline int read(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, uint8_t *data, uint16_t wLength, unsigned int timeout = TIMEOUT) {
    return control_transfer(LIBUSB_ENDPOINT_IN, bRequest, wValue, wIndex, timeout);
  }
  inline int bulk_write(uint8_t endpoint, uint8_t *data, int length, unsigned int timeout = TIMEOUT) {
    return bulk_transfer(LIBUSB_ENDPOINT_OUT | endpoint, data, length, timeout);
  }
  inline int bulk_read(uint8_t endpoint, uint8_t *data, int length, unsigned int timeout = TIMEOUT) {
    return bulk_transfer(LIBUSB_ENDPOINT_IN | endpoint, data, length, timeout);
  }

  std::atomic<bool> comms_healthy = true;
  std::atomic<bool> connected = true;
  std::string usb_serial;

private:
  int control_transfer(libusb_endpoint_direction dir, uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout = TIMEOUT);
  int bulk_transfer(uint8_t endpoint, uint8_t *data, int length, unsigned int timeout = TIMEOUT);

  std::mutex usb_lock;
  libusb_context *ctx = nullptr;
  libusb_device_handle *dev_handle = nullptr;
};
