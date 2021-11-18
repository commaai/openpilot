#pragma once

#include <atomic>
#include <cstdint>
#include <ctime>
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <vector>

#include <libusb-1.0/libusb.h>

#include "cereal/gen/cpp/car.capnp.h"
#include "cereal/gen/cpp/log.capnp.h"

#define TIMEOUT 0
#define PANDA_BUS_CNT 4
#define RECV_SIZE (0x4000U)
#define USB_TX_SOFT_LIMIT   (0x100U)
#define CANPACKET_HEAD_SIZE 5U
#define CANPACKET_MAX_SIZE  72U
#define CANPACKET_REJECTED  (0xC0U)
#define CANPACKET_RETURNED  (0x80U)

// copied from panda/board/main.c
struct __attribute__((packed)) health_t {
  uint32_t uptime;
  uint32_t voltage;
  uint32_t current;
  uint32_t can_rx_errs;
  uint32_t can_send_errs;
  uint32_t can_fwd_errs;
  uint32_t gmlan_send_errs;
  uint32_t faults;
  uint8_t ignition_line;
  uint8_t ignition_can;
  uint8_t controls_allowed;
  uint8_t gas_interceptor_detected;
  uint8_t car_harness_status;
  uint8_t usb_power_mode;
  uint8_t safety_model;
  int16_t safety_param;
  uint8_t fault_status;
  uint8_t power_save_enabled;
  uint8_t heartbeat_lost;
};

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

class Panda {
 private:
  libusb_context *ctx = NULL;
  libusb_device_handle *dev_handle = NULL;
  std::mutex usb_lock;
  std::vector<uint8_t> send;
  void handle_usb_issue(int err, const char func[]);
  void cleanup();

 public:
  Panda(std::string serial="", uint32_t bus_offset=0);
  ~Panda();

  std::string usb_serial;
  std::atomic<bool> connected = true;
  std::atomic<bool> comms_healthy = true;
  cereal::PandaState::PandaType hw_type = cereal::PandaState::PandaType::UNKNOWN;
  bool has_rtc = false;
  const uint32_t bus_offset;

  // Static functions
  static std::vector<std::string> list();

  // HW communication
  int usb_write(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout=TIMEOUT);
  int usb_read(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout=TIMEOUT);
  int usb_bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout=TIMEOUT);
  int usb_bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout=TIMEOUT);

  // Panda functionality
  cereal::PandaState::PandaType get_hw_type();
  void set_safety_model(cereal::CarParams::SafetyModel safety_model, int safety_param=0);
  void set_unsafe_mode(uint16_t unsafe_mode);
  void set_rtc(struct tm sys_time);
  struct tm get_rtc();
  void set_fan_speed(uint16_t fan_speed);
  uint16_t get_fan_speed();
  void set_ir_pwr(uint16_t ir_pwr);
  health_t get_state();
  void set_loopback(bool loopback);
  std::optional<std::vector<uint8_t>> get_firmware_version();
  std::optional<std::string> get_serial();
  void set_power_saving(bool power_saving);
  void set_usb_power_mode(cereal::PeripheralState::UsbPowerMode power_mode);
  void send_heartbeat();
  void set_can_speed_kbps(uint16_t bus, uint16_t speed);
  void set_data_speed_kbps(uint16_t bus, uint16_t speed);
  uint8_t len_to_dlc(uint8_t len);
  void can_send(capnp::List<cereal::CanData>::Reader can_data_list);
  bool can_receive(std::vector<can_frame>& out_vec);

protected:
  // for unit tests
  Panda(uint32_t bus_offset) : bus_offset(bus_offset) {}
  void pack_can_buffer(const capnp::List<cereal::CanData>::Reader &can_data_list,
                         std::function<void(uint8_t *, size_t)> write_func);
  bool unpack_can_buffer(uint8_t *data, int size, std::vector<can_frame> &out_vec);
};
