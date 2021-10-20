#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include <libusb-1.0/libusb.h>

#define TIMEOUT 0
#define PANDA_VENDOR_ID 0xBBAA
#define PANDA_PRODUCT_ID 0xDDCC

class PandaComm {
public:
  PandaComm(uint16_t vid, uint16_t pid, const std::string& serial = {});
  virtual ~PandaComm();
  std::string usb_serial;
  std::atomic<bool> connected = true;
  std::atomic<bool> comms_healthy = true;

  // Static functions
  static std::vector<std::string> list(uint16_t vid, uint16_t pid);

  // HW communication
  int usb_transfer(libusb_endpoint_direction dir, uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout = TIMEOUT);
  int usb_write(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout = TIMEOUT);
  int usb_read(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char* data, uint16_t wLength, unsigned int timeout = TIMEOUT);
  int usb_bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout = TIMEOUT);
  int usb_bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout = TIMEOUT);

protected:
  void handle_usb_issue(int err, const char func[]);

  libusb_context *ctx = nullptr;
  libusb_device_handle* dev_handle = nullptr;
  std::mutex usb_lock;
};
