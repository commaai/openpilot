#pragma once

#include <ctime>
#include <cstdint>
#include <pthread.h>

#include <libusb-1.0/libusb.h>

#include "cereal/gen/cpp/car.capnp.h"
#include "cereal/gen/cpp/log.capnp.h"

// double the FIFO size
#define RECV_SIZE (0x1000)
#define TIMEOUT 0

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
  bool has_rtc = false;

  // HW communication
  int usb_write(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout=TIMEOUT);
  int usb_read(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout=TIMEOUT);
  int usb_bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout=TIMEOUT);
  int usb_bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout=TIMEOUT);

  // Panda functionality
  cereal::HealthData::HwType get_hw_type();

  void set_safety_model(cereal::CarParams::SafetyModel safety_model, int safety_param=0);

  void set_rtc(struct tm sys_time);
  struct tm get_rtc();

  void set_fan_speed(uint16_t fan_speed);
  uint16_t get_fan_speed();


};
