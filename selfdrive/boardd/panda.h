#pragma once

#include <ctime>
#include <cstdint>
#include <pthread.h>
#include <mutex>
#include <vector>
#include <optional>
#include <atomic>

#include <libusb-1.0/libusb.h>

#include "cereal/gen/cpp/car.capnp.h"
#include "cereal/gen/cpp/log.capnp.h"

// double the FIFO size
#define RECV_SIZE (0x1000)
#define TIMEOUT 0

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
  uint8_t fault_status;
  uint8_t power_save_enabled;
};

void panda_set_power(bool power);

class PandaComm{
private:
  libusb_device_handle *dev_handle = NULL;
  libusb_context *ctx = NULL;
  std::mutex usb_lock;

  void cleanup();
  void handle_usb_issue(int err, const char func[]);
public:
  PandaComm(uint16_t vid = 0xbbaa, uint16_t pid = 0xddcc);
  ~PandaComm();
  std::atomic<bool> connected = true;

  // HW communication
  int usb_write(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned int timeout=TIMEOUT);
  int usb_read(uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout=TIMEOUT);
  int usb_bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout=TIMEOUT);
  int usb_bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout=TIMEOUT);
  int control_read(uint8_t request_type, uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout=TIMEOUT);
  int control_write(uint8_t request_type, uint8_t bRequest, uint16_t wValue, uint16_t wIndex, unsigned char *data, uint16_t wLength, unsigned int timeout=TIMEOUT);
};

// DynamicPanda class is used while setting up the "real" panda; aka the Panda that is running the firmware
// We need to be able to switch between different states before getting there though
class DynamicPanda{
private:
  PandaComm* c;
  void cleanup();
  void connect();
  void reconnect();
public:
  DynamicPanda();
  ~DynamicPanda();
  std::string get_version();
  std::string get_signature();
  void flash(std::string fw_fn);
  void reset(bool enter_bootstub, bool enter_bootloader);
  void recover();
  bool pandaExists;
  bool bootstub;
};


class Panda {
 private:
  PandaComm* c;
  void cleanup();

 public:
  Panda();
  ~Panda();

  cereal::PandaState::PandaType hw_type = cereal::PandaState::PandaType::UNKNOWN;
  bool is_pigeon = false;
  bool has_rtc = false;

  bool connected();
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
  void set_usb_power_mode(cereal::PandaState::UsbPowerMode power_mode);
  void send_heartbeat();
  void can_send(capnp::List<cereal::CanData>::Reader can_data_list);
  int can_receive(kj::Array<capnp::word>& out_buf);
};
