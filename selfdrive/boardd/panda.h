#pragma once

#include <libusb-1.0/libusb.h>
#include <pthread.h>

#include "cereal/gen/cpp/car.capnp.h"
#include "cereal/gen/cpp/log.capnp.h"

// double the FIFO size
#define RECV_SIZE (0x1000)
#define TIMEOUT 0

struct __attribute__((packed)) timestamp_t {
  uint16_t year;
  uint8_t month;
  uint8_t day;
  uint8_t weekday;
  uint8_t hour;
  uint8_t minute;
  uint8_t second;
};

class Panda {
 private:
  libusb_context *ctx = NULL;
  libusb_device_handle *dev_handle = NULL;
  pthread_mutex_t usb_lock;
  void handle_usb_issue(int err, const char func[]);

 public:
  Panda();
  ~Panda();

  bool connected = true;
  cereal::HealthData::HwType hw_type = cereal::HealthData::HwType::UNKNOWN;
  bool is_pigeon = false;

  // HW communication
  int usb_write(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout=TIMEOUT);
  int usb_read(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout=TIMEOUT);
  int usb_bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout=TIMEOUT);
  int usb_bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout=TIMEOUT);

  // Panda functionality
  int set_safety_model(cereal::CarParams::SafetyModel safety_model, int safety_param=0);
  cereal::HealthData::HwType get_hw_type();
};
