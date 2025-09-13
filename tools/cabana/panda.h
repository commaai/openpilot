#pragma once

#include <cstdint>
#include <ctime>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <atomic>
#include <mutex>
#include <libusb-1.0/libusb.h>

#include "cereal/gen/cpp/car.capnp.h"
#include "cereal/gen/cpp/log.capnp.h"
#include "panda/board/health.h"
#include "panda/board/can.h"

#define USB_TX_SOFT_LIMIT   (0x100U)
#define USBPACKET_MAX_SIZE  (0x40)
#define RECV_SIZE (0x4000U)
#define TIMEOUT 0

#define CAN_REJECTED_BUS_OFFSET 0xC0U
#define CAN_RETURNED_BUS_OFFSET 0x80U

#define PANDA_BUS_OFFSET 4

struct __attribute__((packed)) can_header {
  uint8_t reserved : 1;
  uint8_t bus : 3;
  uint8_t data_len_code : 4;
  uint8_t rejected : 1;
  uint8_t returned : 1;
  uint8_t extended : 1;
  uint32_t addr : 29;
  uint8_t checksum : 8;
};

struct can_frame {
  long address;
  std::string dat;
  long src;
};


class Panda {
public:
  Panda(std::string serial="", uint32_t bus_offset=0);
  ~Panda();

  cereal::PandaState::PandaType hw_type = cereal::PandaState::PandaType::UNKNOWN;
  const uint32_t bus_offset;

  bool connected();
  bool comms_healthy();
  std::string hw_serial();

  // Static functions
  static std::vector<std::string> list(bool usb_only=false);

  // Panda functionality
  cereal::PandaState::PandaType get_hw_type();
  void set_safety_model(cereal::CarParams::SafetyModel safety_model, uint16_t safety_param=0U);
  void send_heartbeat(bool engaged);
  void set_can_speed_kbps(uint16_t bus, uint16_t speed);
  void set_data_speed_kbps(uint16_t bus, uint16_t speed);
  bool can_receive(std::vector<can_frame>& out_vec);
  void can_reset_communications();

private:
  // USB connection members
  libusb_context *ctx = nullptr;
  libusb_device_handle *dev_handle = nullptr;
  std::string hw_serial_str;
  std::atomic<bool> connected_flag = true;
  std::atomic<bool> comms_healthy_flag = true;

  // CAN buffer members
  uint8_t receive_buffer[RECV_SIZE + sizeof(can_header) + 64];
  uint32_t receive_buffer_size = 0;

  // Internal methods
  bool init_usb_connection(const std::string& serial);
  void cleanup_usb();
  void handle_usb_issue(int err, const char func[]);
  int control_write(uint8_t request, uint16_t param1, uint16_t param2, unsigned int timeout=TIMEOUT);
  int control_read(uint8_t request, uint16_t param1, uint16_t param2, unsigned char *data, uint16_t length, unsigned int timeout=TIMEOUT);
  int bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout=TIMEOUT);
  int bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout=TIMEOUT);
  bool unpack_can_buffer(uint8_t *data, uint32_t &size, std::vector<can_frame> &out_vec);
  uint8_t calculate_checksum(uint8_t *data, uint32_t len);
};
