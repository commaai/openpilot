#pragma once

#include <mutex>
#include <atomic>
#include <cstdint>
#include <ctime>
#include <functional>
#include <list>
#include <optional>
#include <vector>

#include <libusb-1.0/libusb.h>

#include "cereal/gen/cpp/car.capnp.h"
#include "cereal/gen/cpp/log.capnp.h"

#define TIMEOUT 0
#define PANDA_CAN_CNT 3
#define PANDA_BUS_CNT 4
#define RECV_SIZE (0x4000U)
#define USB_TX_SOFT_LIMIT   (0x100U)
#define USBPACKET_MAX_SIZE  (0x40)
#define CANPACKET_HEAD_SIZE 5U
#define CANPACKET_MAX_SIZE  72U
#define CANPACKET_REJECTED  (0xC0U)
#define CANPACKET_RETURNED  (0x80U)

struct __attribute__((packed)) can_header {
  uint8_t reserved : 1;
  uint8_t bus : 3;
  uint8_t data_len_code : 4;
  uint8_t rejected : 1;
  uint8_t returned : 1;
  uint8_t extended : 1;
  uint32_t addr : 29;
};

struct can_frame {
	long address;
	std::string dat;
	long busTime;
	long src;
};

// comms base class
class PandaCommsHandle {
public:

  PandaCommsHandle(std::string serial) {};
  virtual void cleanup() = 0;

  std::atomic<bool> connected = true;
  std::atomic<bool> comms_healthy = true;
  static std::vector<std::string> list();

  // HW communication
  virtual int control_write(uint8_t request, uint16_t param1, uint16_t param2, unsigned int timeout=TIMEOUT) = 0;
  virtual int control_read(uint8_t request, uint16_t param1, uint16_t param2, unsigned char *data, uint16_t length, unsigned int timeout=TIMEOUT) = 0;
  virtual int bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout=TIMEOUT) = 0;
  virtual int bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout=TIMEOUT) = 0;

protected:
  std::mutex hw_lock;
};

class PandaUsbHandle : public PandaCommsHandle {
public:
  PandaUsbHandle(std::string serial);
  ~PandaUsbHandle();
  int control_write(uint8_t request, uint16_t param1, uint16_t param2, unsigned int timeout=TIMEOUT);
  int control_read(uint8_t request, uint16_t param1, uint16_t param2, unsigned char *data, uint16_t length, unsigned int timeout=TIMEOUT);
  int bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout=TIMEOUT);
  int bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout=TIMEOUT);
  void cleanup();

  static std::vector<std::string> list();

private:
  libusb_context *ctx = NULL;
  libusb_device_handle *dev_handle = NULL;
  std::vector<uint8_t> recv_buf;
  void handle_usb_issue(int err, const char func[]);
};
